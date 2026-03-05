"""User asset processing resources."""

import logging
from pathlib import Path
from typing import Optional

import yaml
from s2gos_utils.io.paths import open_file

from ..core.context import SceneResourceContext


def process_user_assets(ctx: SceneResourceContext) -> Optional[Path]:
    """Process user assets (3D objects) for placement in the scene.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the objects directory containing processed assets
    """

    from s2gos_utils.io.paths import copy, mkdir

    processed_objects = []
    inline_materials = {}

    objects_dir = ctx.output_dir / "objects"
    mkdir(objects_dir)

    target_dem_path = ctx.dependency_outputs["target_dem"]
    if target_dem_path is None:
        raise RuntimeError("Target DEM data not available for elevation querying")

    coords = ctx.coordinate_system

    for i, asset in enumerate(ctx.user_assets):
        try:
            if asset.coord_type == "geographic":
                lon, lat = asset.coordinate
                scene_x, scene_y = coords.latlon_to_scene(lat, lon)
                elevation = coords.query_height_from_dem(lat, lon, target_dem_path)
            else:
                scene_x, scene_y = asset.coordinate
                lat, lon = coords.scene_to_latlon(scene_x, scene_y)
                elevation = coords.query_height_from_dem(lat, lon, target_dem_path)

            final_z = elevation + asset.elevation_offset

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

            if asset.material:
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

    logging.info(f"Processed {len(processed_objects)} user assets")
    if inline_materials:
        logging.info(f"Extracted {len(inline_materials)} inline material definitions")

    if processed_objects or inline_materials:
        sidecar_data = {}
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
