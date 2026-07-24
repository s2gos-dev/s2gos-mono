"""User asset processing resources.

All asset ingestion happens here, at generation time: plain ``ply_path`` user assets
are copied and placed, and each configured Mitsuba XML scene is parsed and expanded.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from s2gos_utils.io.paths import open_file
from upath import UPath

from ..core.context import SceneResourceContext

# Fallback for an XML shape whose material cannot be resolved
_DEFAULT_MATERIAL = {
    "type": "diffuse",
    "reflectance": {"type": "uniform", "value": 0.5},
}


def process_user_assets(ctx: SceneResourceContext) -> Optional[Path]:
    """Process user assets (single PLYs and Mitsuba XML scenes) for scene placement.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the user-assets sidecar (``None`` when nothing was produced)
    """
    from s2gos_utils.io.paths import copy, mkdir

    processed_objects = []
    inline_materials = {}

    objects_dir = ctx.output_dir / "objects"
    mkdir(objects_dir)

    target_dem_path = ctx.dependency_outputs["target_dem"]
    if target_dem_path is None:
        raise RuntimeError("Target DEM data not available for elevation querying")

    for asset in ctx.user_assets:
        try:
            scene_x, scene_y, final_z = _resolve_placement(
                ctx,
                target_dem_path,
                asset.coordinate,
                asset.coord_type,
                asset.elevation_offset,
            )

            ply_filename = f"{asset.object_id}.ply"
            output_ply_path = objects_dir / ply_filename
            copy(asset.ply_path, output_ply_path)

            object_data = {
                "id": asset.object_id,
                "mesh": f"objects/{ply_filename}",
                "position": [scene_x, scene_y, final_z],
                "scale": asset.scale,
                "rotation": [asset.rotation_x, asset.rotation_y, asset.rotation_z],
                "blender_fix": asset.blender_fix,
            }

            if isinstance(asset.material, dict):
                mat_id = asset.get_inline_material_id()
                inline_materials[mat_id] = asset.material
                object_data["material"] = mat_id
                logging.info(
                    f"Added inline material '{mat_id}' for object '{asset.object_id}'"
                )
            else:
                object_data["material"] = asset.material

            if asset.face_normals is not None:
                object_data["face_normals"] = asset.face_normals

            processed_objects.append(object_data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to process user asset {asset.object_id}: {e}"
            ) from e

    for xml_cfg in ctx.config.xml_scenes:
        try:
            scene_objects, scene_materials = _expand_xml_scene(
                xml_cfg, ctx, target_dem_path, objects_dir
            )
            processed_objects.extend(scene_objects)
            inline_materials.update(scene_materials)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process XML scene {xml_cfg.xml_path}: {e}"
            ) from e

    logging.info(f"Processed {len(processed_objects)} user asset objects")
    if inline_materials:
        logging.info(f"Extracted {len(inline_materials)} material definitions")

    return _finalize_user_assets_sidecar(ctx, processed_objects, inline_materials)


def _finalize_user_assets_sidecar(ctx, processed_objects, inline_materials):
    """Write the user-assets sidecar (objects + materials) and register it."""
    if not (processed_objects or inline_materials):
        return ctx.assets.user_assets_file

    sidecar_data: Dict[str, Any] = {}
    if processed_objects:
        sidecar_data["objects"] = processed_objects
    if inline_materials:
        sidecar_data["materials"] = inline_materials

    sidecar_path = ctx.data_dir / "user_assets.yml"
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    with open_file(sidecar_path, "w") as f:
        yaml.dump(sidecar_data, f, default_flow_style=False, indent=2)
    ctx.assets.user_assets_file = sidecar_path
    logging.info(f"Saved user assets sidecar: {sidecar_path}")
    return ctx.assets.user_assets_file


def _resolve_placement(
    ctx: SceneResourceContext,
    dem_path,
    coordinate,
    coord_type: str,
    elevation_offset: float,
) -> Tuple[float, float, float]:
    """Resolve a config coordinate to scene-frame ``(x, y, z)`` on the terrain."""
    coords = ctx.coordinate_system
    if coord_type == "geographic":
        lon, lat = coordinate
        scene_x, scene_y = coords.latlon_to_scene(lat, lon)
    else:
        scene_x, scene_y = coordinate
        lat, lon = coords.scene_to_latlon(scene_x, scene_y)
    elevation = coords.query_height_from_dem(lat, lon, dem_path)
    return scene_x, scene_y, elevation + elevation_offset


def _resolve_rotations(
    fix_blender_coords: bool,
    rotation_x: float,
    rotation_y: float,
    rotation_z: float,
) -> Tuple[float, float, float]:
    """Map user rotations to world axes, folding in the Blender→Mitsuba correction.

    With ``fix_blender_coords`` a 90° X-rotation is applied, which swaps the Y/Z
    axes; the user's Y/Z rotations are remapped so ``rotation_z`` always means
    "rotate around up" regardless of the fix.
    """
    if fix_blender_coords:
        rx, ry, rz = 90.0 + rotation_x, rotation_z, -rotation_y
    else:
        rx, ry, rz = rotation_x, rotation_y, rotation_z
    return tuple(0.0 if abs(v) < 1e-10 else v for v in (rx, ry, rz))


def _placement_transform(
    x: float, y: float, z: float, rot_x: float, rot_y: float, rot_z: float, scale: float
) -> np.ndarray:
    """Rigid placement ``translate @ Rx @ Ry @ Rz @ scale``.

    Matches the simulator's ``_create_transform_from_object`` convention exactly, so
    placed meshes and baked instance transforms share the same frame. Intrinsic
    ``"XYZ"`` Euler angles are precisely the sequential ``Rx @ Ry @ Rz`` product, and
    the uniform scale folds into the rotation block.
    """
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = (
        Rotation.from_euler("XYZ", [rot_x, rot_y, rot_z], degrees=True).as_matrix()
        * scale
    )
    m[:3, 3] = (x, y, z)
    return m


def _bake_world_transforms(
    placement: np.ndarray, local_transforms: np.ndarray
) -> np.ndarray:
    """Fold the placement into each instance: ``world_i = placement @ local_i``.

    ``np.matmul`` broadcasts the ``(4, 4)`` placement across the ``(N, 4, 4)`` stack of
    instance object-to-world matrices. Returns an ``(N, 4, 4)`` float32 buffer.
    """
    return (placement @ local_transforms).astype(np.float32)


def _validate_scene_files(scene, xml_path: str) -> None:
    """Fail fast when a referenced mesh/data file is missing (aggregate error).

    Gated by the ``XmlSceneConfig.validate_materials`` flag. Validation covers file
    existence only: material *references* may point at the scene library, which is not
    known until scene assembly.
    """
    referenced = [s["file"] for s in scene.shapes]
    referenced += [c["file"] for sg in scene.shapegroups for c in sg["components"]]
    missing = [f for f in referenced if not UPath(f).exists()]
    if missing:
        raise ValueError(
            f"XML scene '{xml_path}' references missing files:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def _expand_xml_scene(
    xml_cfg,
    ctx: SceneResourceContext,
    dem_path,
    objects_dir: UPath,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Expand one configured Mitsuba XML scene into sidecar objects + materials.

    Top-level ``ply`` shapes become placed objects at the scene's base coordinate,
    ``<instance>`` placements are streamed and baked with the placement transform
    into one ``(N, 4, 4)`` buffer per shapegroup (an ``instance_collection`` object).

    Materials defined in the XML are namespaced by the scene prefix so multiple XML
    scenes never collide; ``material_mappings`` targets pass through untouched, and a
    shape whose material cannot be resolved falls back to a uniform diffuse 0.5.
    """
    from s2gos_utils.io.paths import copy

    from ..processors.xml_importer import (
        NO_BSDF,
        _match_filename,
        parse_mitsuba_scene,
        read_instance_transforms,
    )

    xml_path = str(xml_cfg.xml_path)
    prefix = xml_cfg.object_id_prefix or xml_cfg.xml_path.upath.stem
    scene = parse_mitsuba_scene(xml_path)
    if xml_cfg.validate_materials:
        _validate_scene_files(scene, xml_path)

    scene_x, scene_y, final_z = _resolve_placement(
        ctx,
        dem_path,
        list(xml_cfg.base_coordinate),
        xml_cfg.coord_type,
        xml_cfg.elevation_offset,
    )
    rot_x, rot_y, rot_z = _resolve_rotations(
        xml_cfg.fix_blender_coords,
        xml_cfg.rotation_x,
        xml_cfg.rotation_y,
        xml_cfg.rotation_z,
    )

    materials = {f"{prefix}_{mid}": mdef for mid, mdef in scene.materials.items()}

    def _resolve_material(mat_id: str) -> str:
        """Material name to reference for a shape/component.

        XML-defined ids are namespaced; anything unresolved falls back to a synthesized
        uniform diffuse 0.5 (added to the sidecar once).
        """
        if mat_id in scene.materials:
            return f"{prefix}_{mat_id}"
        default_id = f"{prefix}_default"
        materials.setdefault(default_id, _DEFAULT_MATERIAL)
        if mat_id == NO_BSDF:
            logging.info(
                "XML scene '%s': a shape declares no material; using diffuse 0.5.",
                prefix,
            )
        else:
            logging.warning(
                "XML scene '%s': material '%s' is not defined in %s; using diffuse 0.5.",
                prefix,
                mat_id,
                xml_path,
            )
        return default_id

    def _copy_in(src_file: str, dest_name: str) -> str:
        copy(UPath(src_file), objects_dir / dest_name)
        return f"objects/{dest_name}"

    objects: List[Dict[str, Any]] = []

    for shape in scene.shapes:
        stem = UPath(shape["file"]).stem
        object_id = f"{prefix}_{stem}"

        material_ref = None
        for mapping in xml_cfg.material_mappings:
            if _match_filename(stem, mapping.pattern, mapping.mode):
                material_ref = mapping.material
                break
        if material_ref is None:
            material_ref = _resolve_material(shape["material"])

        obj: Dict[str, Any] = {
            "id": object_id,
            "mesh": _copy_in(shape["file"], f"{object_id}.ply"),
            "position": [scene_x, scene_y, final_z],
            "scale": xml_cfg.scale,
            "rotation": [rot_x, rot_y, rot_z],
            "blender_fix": xml_cfg.fix_blender_coords,
            "material": material_ref,
        }
        if "face_normals" in shape:
            obj["face_normals"] = shape["face_normals"]
        objects.append(obj)

    if not scene.shapegroups:
        return objects, materials

    # Shapegroups + streamed instance placements.
    instance_transforms_by_group = (
        read_instance_transforms(xml_path) if scene.instanced else {}
    )
    placement = _placement_transform(
        scene_x, scene_y, final_z, rot_x, rot_y, rot_z, xml_cfg.scale
    )

    for si, sg in enumerate(scene.shapegroups):
        sg_ref = f"{prefix}_{sg['id'] or f'sg{si}'}"
        shapegroup_obj: Dict[str, Any] = {
            "object_id": sg_ref,
            "id": sg_ref,
            "type": "shapegroup",
        }
        for ci, component in enumerate(sg["components"]):
            src_name = UPath(component["file"]).name
            child: Dict[str, Any] = {
                "type": component["type"],
                "filename": _copy_in(component["file"], f"{sg_ref}_c{ci}_{src_name}"),
                "bsdf": {
                    "type": "ref",
                    "id": f"_mat_{_resolve_material(component['material'])}",
                },
            }
            if component["type"] == "ply" and "face_normals" in component:
                child["face_normals"] = component["face_normals"]
            if component["type"] == "ellipsoidsmesh":
                child["extent"] = component.get("extent", 1.0)
            shapegroup_obj[f"component_{ci}"] = child
        objects.append(shapegroup_obj)

        local = instance_transforms_by_group.get(sg["id"])
        if local is None or len(local) == 0:
            logging.warning(
                "XML scene '%s': shapegroup '%s' has no instances", prefix, sg["id"]
            )
            continue
        transforms = _bake_world_transforms(placement, local)
        npy_name = f"{sg_ref}_instances.npy"
        with open_file(objects_dir / npy_name, "wb") as f:
            np.save(f, transforms)
        objects.append(
            {
                "object_id": f"{sg_ref}_instances",
                "type": "instance_collection",
                "shapegroup_ref": sg_ref,
                "data_file": f"objects/{npy_name}",
                "count": int(len(transforms)),
            }
        )
        logging.info(
            "XML scene '%s': shapegroup '%s' -> %d instance(s)",
            prefix,
            sg["id"],
            len(transforms),
        )

    return objects, materials
