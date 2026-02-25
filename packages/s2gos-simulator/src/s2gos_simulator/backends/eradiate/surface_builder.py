"""Surface, material, and object creation for Eradiate backend."""

import logging
from typing import Any, Dict

import numpy as np
from PIL import Image
from s2gos_utils.io.paths import exists, open_file
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .eradiate_materials import EradiateMaterialAdapter

try:
    import mitsuba as mi

    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SurfaceBuilder:
    """Builder for creating surfaces, materials, and 3D objects."""

    def __init__(self):
        """Initialize surface builder."""
        self.material_adapter = EradiateMaterialAdapter()

    def create_target_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create target surface from SceneDescription.

        Args:
            scene_description: Scene description containing target configuration
            scene_dir: Directory containing scene assets
            hamster_available: Whether HAMSTER spatial albedo data is available

        Returns:
            Dictionary with terrain_material and terrain entries
        """
        target_config = scene_description.target
        target_mesh_path = scene_dir / target_config["mesh"]
        target_texture_path = scene_dir / target_config["selection_texture"]

        _, _, material_dict = self._create_selectbsdf_material(
            "target",
            scene_description,
            scene_dir,
            target_texture_path,
            hamster_available,
        )

        return {
            "terrain_material": material_dict,
            "terrain": {
                "type": "ply",
                "filename": str(target_mesh_path),
                "bsdf": {"type": "ref", "id": "terrain_material"},
                "id": "terrain",
            },
        }

    def create_buffer_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create buffer surface from SceneDescription.

        Args:
            scene_description: Scene description containing buffer configuration
            scene_dir: Directory containing scene assets
            hamster_available: Whether HAMSTER spatial albedo data is available

        Returns:
            Dictionary with buffer_material and buffer_terrain entries
        """
        buffer_config = scene_description.buffer
        buffer_mesh_path = scene_dir / buffer_config["mesh"]
        buffer_texture_path = scene_dir / buffer_config["selection_texture"]
        mask_path = (
            scene_dir / buffer_config["mask_texture"]
            if "mask_texture" in buffer_config
            else None
        )

        _, _, material_dict = self._create_selectbsdf_material(
            "buffer",
            scene_description,
            scene_dir,
            buffer_texture_path,
            hamster_available,
        )

        result = {"buffer_material": material_dict}

        buffer_bsdf_id = "buffer_material"

        if mask_path and exists(mask_path):
            with open_file(mask_path, "rb") as f:
                mask_image = Image.open(f)
                mask_image.load()
            mask_data = np.array(mask_image) / 255.0
            mask_data = np.atleast_3d(mask_data)

            result["buffer_mask"] = {
                "type": "mask",
                "opacity": {
                    "type": "bitmap",
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "data": mask_data,
                },
                "material": {"type": "ref", "id": "buffer_material"},
            }
            buffer_bsdf_id = "buffer_mask"

        result["buffer_terrain"] = {
            "type": "ply",
            "filename": str(buffer_mesh_path),
            "bsdf": {"type": "ref", "id": buffer_bsdf_id},
            "id": "buffer_terrain",
        }

        return result

    def create_background_surface(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        hamster_available: bool = False,
    ) -> Dict[str, Any]:
        """Create background surface from SceneDescription.

        Args:
            scene_description: Scene description containing background configuration
            scene_dir: Directory containing scene assets
            hamster_available: Whether HAMSTER spatial albedo data is available

        Returns:
            Dictionary with background_material and background_surface entries
        """
        background_config = scene_description.background
        elevation = background_config["elevation"]
        background_texture_path = scene_dir / background_config["selection_texture"]
        background_size_km = background_config["size_km"]

        _, _, material_dict = self._create_selectbsdf_material(
            "background",
            scene_description,
            scene_dir,
            background_texture_path,
            hamster_available,
        )

        scale_factor = (background_size_km * 1000) / 2.0
        to_world = mi.ScalarTransform4f.translate(
            [0, 0, elevation]
        ) @ mi.ScalarTransform4f.scale(scale_factor)

        return {
            "background_material": material_dict,
            "background_surface": {
                "type": "rectangle",
                "to_world": to_world,
                "bsdf": {"type": "ref", "id": "background_material"},
                "id": "background_surface",
            },
        }

    def _create_selectbsdf_material(
        self,
        surface_name: str,
        scene_description: SceneDescription,
        scene_dir: UPath,
        texture_path: UPath,
        hamster_available: bool = False,
    ) -> tuple[np.ndarray, list[str], dict]:
        """Create selectbsdf material dictionary for a surface.

        Args:
            surface_name: Name of surface ("target", "buffer", or "background")
            scene_description: Scene description containing material information
            scene_dir: Directory containing scene assets
            texture_path: Path to selection texture file
            hamster_available: Whether HAMSTER data is available

        Returns:
            Tuple of (selection_texture_data, material_ids, material_dict)
            where material_dict is the selectbsdf configuration
        """

        # Helper function to get material IDs
        def get_material_ids_from_scene(scene_desc: SceneDescription) -> list[str]:
            material_indices = scene_desc.material_indices
            material_ids = []
            for texture_index in sorted(material_indices.keys(), key=int):
                material_name = material_indices[texture_index]
                material_ids.append(material_name)
            return material_ids

        with open_file(texture_path, "rb") as f:
            texture_image = Image.open(f)
            texture_image.load()
        selection_texture_data = np.array(texture_image)
        selection_texture_data = np.atleast_3d(selection_texture_data)

        material_indices = scene_description.material_indices
        material_ids = get_material_ids_from_scene(scene_description)

        if hamster_available:
            material_ids = [
                f"{mat_id}_{surface_name}" if int(idx) < 11 else mat_id
                for idx, mat_id in zip(
                    sorted(material_indices.keys(), key=int), material_ids
                )
            ]

        material_id = (
            "terrain_material"
            if surface_name == "target"
            else f"{surface_name}_material"
        )
        bsdf_prefix = "terrain" if surface_name == "target" else surface_name

        material_dict = {
            "type": "selectbsdf",
            "id": material_id,
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": selection_texture_data,
            },
            **{
                f"{bsdf_prefix}_bsdf_{i:02d}": {"type": "ref", "id": f"_mat_{mat_id}"}
                for i, mat_id in enumerate(material_ids)
            },
        }

        return selection_texture_data, material_ids, material_dict

    def translate_materials(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> tuple[dict, dict]:
        """Translate scene materials to Eradiate kernel dictionaries.

        Args:
            scene_description: Scene description with materials to translate
            scene_dir: Base directory for resolving file paths

        Returns:
            Tuple of (kdict, kpmap) containing material definitions
        """
        kdict = {}
        kpmap = {}

        from s2gos_utils.scene.materials import (
            BilambertianMaterial,
            ConductorMaterial,
            DielectricMaterial,
            DiffuseMaterial,
            MeasuredMaterial,
            OceanLegacyMaterial,
            PlasticMaterial,
            PrincipledMaterial,
            RoughConductorMaterial,
            RPVMaterial,
        )

        for mat_name, material in scene_description.materials.items():
            if isinstance(material, DiffuseMaterial):
                mat_def = self.material_adapter.create_diffuse_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_diffuse_kpmap(material)
            elif isinstance(material, BilambertianMaterial):
                mat_def = self.material_adapter.create_bilambertian_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_bilambertian_kpmap(material)
            elif isinstance(material, RPVMaterial):
                mat_def = self.material_adapter.create_rpv_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_rpv_kpmap(material)
            elif isinstance(material, OceanLegacyMaterial):
                mat_def = self.material_adapter.create_ocean_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_ocean_kpmap(material)
            elif isinstance(material, DielectricMaterial):
                mat_def = self.material_adapter.create_dielectric_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_dielectric_kpmap(material)
            elif isinstance(material, ConductorMaterial):
                mat_def = self.material_adapter.create_conductor_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_conductor_kpmap(material)
            elif isinstance(material, RoughConductorMaterial):
                mat_def = self.material_adapter.create_rough_conductor_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_rough_conductor_kpmap(material)
            elif isinstance(material, PlasticMaterial):
                mat_def = self.material_adapter.create_plastic_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_plastic_kpmap(material)
            elif isinstance(material, PrincipledMaterial):
                mat_def = self.material_adapter.create_principled_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_principled_kpmap(material)
            elif isinstance(material, MeasuredMaterial):
                mat_def = self.material_adapter.create_measured_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_measured_kpmap(material)
            else:
                mat_def = self.material_adapter.create_diffuse_kdict(material)
                mat_kdict = {f"_mat_{mat_name}": mat_def}
                mat_kpmap = self.material_adapter.create_diffuse_kpmap(material)

            kdict.update(mat_kdict)
            kpmap.update(mat_kpmap)

        return kdict, kpmap

    def add_hamster_materials(
        self,
        kdict: dict,
        kpmap: dict,
        scene_description: SceneDescription,
        hamster_data_dict: dict | None,
    ) -> None:
        """Add HAMSTER albedo materials to kernel dictionaries.

        Modifies kdict and kpmap in place to add HAMSTER-based materials
        for each surface (target, buffer, background).

        Args:
            kdict: Kernel dictionary to update
            kpmap: Kernel parameter map to update
            scene_description: Scene description with materials
            hamster_data_dict: HAMSTER albedo data (None if not available)
        """
        if hamster_data_dict is None:
            return

        region_material_names = {
            mat_name
            for idx, mat_name in scene_description.material_indices.items()
            if int(idx) >= 11
        }

        for surface_name, hamster_data in hamster_data_dict.items():
            for mat_name in scene_description.materials.keys():
                if mat_name in region_material_names:
                    continue

                hamster_material_id = f"_mat_{mat_name}_{surface_name}"

                hamster_kdict = self.material_adapter.create_hamster_kdict(
                    material_id=hamster_material_id, albedo_data=hamster_data
                )

                hamster_kpmap = self.material_adapter.create_hamster_kpmap(
                    material_id=hamster_material_id, albedo_data=hamster_data
                )

                kdict.update(hamster_kdict)
                kpmap.update(hamster_kpmap)

    def validate_material_ids(self, kdict: dict, scene_description=None) -> None:
        """Validate material IDs to prevent Eradiate parsing issues.

        Args:
            kdict: Kernel dictionary to validate
            scene_description: Optional scene description for additional context
        """
        issues = []

        for key, value in kdict.items():
            if "." in key:
                issues.append(
                    f"Material ID '{key}' contains dots which may cause parsing issues"
                )
            if key.isdigit():
                issues.append(
                    f"Material ID '{key}' is purely numeric which may cause parsing issues"
                )

        for key, value in kdict.items():
            if isinstance(value, dict) and value.get("type") == "shapegroup":
                for comp_key, comp_value in value.items():
                    if isinstance(comp_value, dict) and comp_value.get("type") == "ply":
                        bsdf = comp_value.get("bsdf", {})
                        if bsdf.get("type") == "ref":
                            ref_id = bsdf.get("id")
                            if ref_id and ref_id not in kdict:
                                issues.append(
                                    f"Shape component '{comp_key}' references undefined material '{ref_id}'"
                                )

        if issues:
            logging.warning(f"Material validation found {len(issues)} issues:")
            for issue in issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Material validation passed - no issues found")

    def add_scene_objects(
        self, kdict: dict, scene_description: SceneDescription, scene_dir: UPath
    ) -> None:
        """Add 3D objects from scene description to kernel dictionary.

        Processes all object types: shapegroups, instances, vegetation
        collections, disks, and PLY meshes.

        Args:
            kdict: Kernel dictionary to update
            scene_description: Scene with objects to process
            scene_dir: Base directory for resolving mesh paths
        """
        if not scene_description.objects:
            return

        logging.info(f"Processing {len(scene_description.objects)} objects")

        for obj in scene_description.objects:
            obj_type = obj.get("type", "ply")

            if obj_type == "shapegroup":
                self._add_shapegroup(kdict, obj, scene_dir)
            elif obj_type == "instance":
                self._add_instance(kdict, obj)
            elif obj_type == "vegetation_collection":
                self._expand_vegetation_collection(obj, scene_dir, kdict)
            elif obj_type == "disk":
                self._add_disk(kdict, obj)
            else:
                self._add_ply_mesh(kdict, obj, scene_dir)

    def _add_shapegroup(self, kdict: dict, obj: dict, scene_dir: UPath) -> None:
        """Add a shapegroup object to kernel dictionary.

        Shapegroups are collections of shapes that can be instanced.

        Args:
            kdict: Kernel dictionary to update
            obj: Shapegroup object specification
            scene_dir: Base directory for resolving mesh paths
        """
        obj_dict = {"type": "shapegroup"}
        if "id" in obj:
            obj_dict["id"] = obj["id"]

        for key, value in obj.items():
            if key in ["type", "id", "object_id"]:
                continue

            if isinstance(value, dict) and value.get("type") == "ply":
                shape_dict = self._create_ply_shape_dict(value, scene_dir)
                obj_dict[key] = shape_dict
            elif isinstance(value, dict) and value.get("type") in [
                "sphere",
                "cube",
                "cylinder",
                "rectangle",
                "disk",
            ]:
                obj_dict[key] = value
            elif not isinstance(value, dict):
                obj_dict[key] = value
            else:
                logging.warning(
                    f"Skipping unrecognized entry in shapegroup '{key}': "
                    f"{value.get('type', 'unknown')}"
                )

        obj_id = obj.get("id") or obj.get("object_id", f"shapegroup_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _add_instance(self, kdict: dict, obj: dict) -> None:
        """Add an instance object to kernel dictionary.

        Instances reference shapegroups with transformations.

        Args:
            kdict: Kernel dictionary to update
            obj: Instance object specification
        """
        obj_dict = {"type": "instance", "shapegroup": obj["shapegroup"]}
        if "object_id" in obj:
            obj_dict["id"] = obj["object_id"]

        if "to_world" in obj:
            transform_spec = obj["to_world"]
            if transform_spec.get("type") == "transform":
                to_world = self._create_transform_from_spec(transform_spec)
                obj_dict["to_world"] = to_world
            else:
                obj_dict["to_world"] = obj["to_world"]

        obj_id = obj.get("id") or obj.get("object_id", f"instance_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _add_disk(self, kdict: dict, obj: dict) -> None:
        """Add a disk object to kernel dictionary.

        Creates a circular disk with Lambertian white material.

        Args:
            kdict: Kernel dictionary to update
            obj: Disk object specification
        """
        center = obj["center"]
        radius = obj["radius"]
        disk_id = obj.get("id") or obj.get("object_id", f"disk_{len(kdict)}")

        obj_dict = {
            "type": "disk",
            "to_world": (
                mi.ScalarTransform4f.translate(center)
                @ mi.ScalarTransform4f.scale(radius)
            ),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 1.0},
            },
            "id": disk_id,
        }
        logger.info(f"{obj_dict = }")

        kdict[disk_id] = obj_dict

    def _add_ply_mesh(self, kdict: dict, obj: dict, scene_dir: UPath) -> None:
        """Add a PLY mesh object to kernel dictionary.

        Handles mesh loading with optional materials, transforms, and face normals.

        Args:
            kdict: Kernel dictionary to update
            obj: PLY mesh object specification
            scene_dir: Base directory for resolving mesh paths
        """
        object_mesh_path = scene_dir / obj["mesh"]
        obj_dict = {
            "type": "ply",
            "filename": str(object_mesh_path),
            "id": obj["id"],
        }

        if "face_normals" in obj:
            obj_dict["face_normals"] = obj["face_normals"]

        if "material" in obj:
            material = obj["material"]
            if isinstance(material, str):
                obj_dict["bsdf"] = {"type": "ref", "id": f"_mat_{material}"}
            else:
                obj_dict["bsdf"] = {
                    "type": "diffuse",
                    "reflectance": {"type": "uniform", "value": 0.5},
                }

        if "position" in obj and "scale" in obj:
            to_world = self._create_transform_from_object(obj)
            obj_dict["to_world"] = to_world

        obj_id = obj.get("id") or obj.get("object_id", f"object_{len(kdict)}")
        kdict[obj_id] = obj_dict

    def _expand_vegetation_collection(
        self, vegetation_collection_obj: dict, scene_dir: UPath, kdict: dict
    ):
        """Expand vegetation collection to individual Mitsuba instances efficiently.

        Args:
            vegetation_collection_obj: Vegetation collection object from scene description
            scene_dir: Scene directory for resolving paths
            kdict: Eradiate kernel dictionary to add instances to
        """
        data_file = vegetation_collection_obj["data_file"]
        binary_path = scene_dir / data_file

        try:
            vegetation_data = np.load(binary_path)
            count = len(vegetation_data)
            shapegroup_ref = vegetation_collection_obj["shapegroup_ref"]
            collection_name = vegetation_collection_obj.get(
                "collection_name", "vegetation"
            )

            logging.info(
                f"Expanding vegetation collection '{collection_name}' with {count} instances"
            )

            for i in range(count):
                instance_id = f"vegetation_instance_{collection_name}_{i}"

                x, y, z = (
                    float(vegetation_data[i]["x"]),
                    float(vegetation_data[i]["y"]),
                    float(vegetation_data[i]["z"]),
                )
                rotation = float(vegetation_data[i]["rotation"])
                scale = float(vegetation_data[i]["scale"])
                tilt_x = (
                    float(vegetation_data[i]["tilt_x"])
                    if "tilt_x" in vegetation_data.dtype.names
                    else 0.0
                )
                tilt_y = (
                    float(vegetation_data[i]["tilt_y"])
                    if "tilt_y" in vegetation_data.dtype.names
                    else 0.0
                )

                to_world = mi.ScalarTransform4f.translate([x, y, z])
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], 90)
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], rotation)
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], tilt_x)
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], tilt_y)
                to_world = to_world @ mi.ScalarTransform4f.scale(scale)

                kdict[instance_id] = {
                    "type": "instance",
                    "shapegroup": {"type": "ref", "id": shapegroup_ref},
                    "to_world": to_world,
                }

        except Exception as e:
            logging.error(
                f"Failed to expand vegetation collection '{vegetation_collection_obj.get('collection_name', 'unknown')}': {e}"
            )
            raise

    def _create_ply_shape_dict(self, value: dict, scene_dir: UPath) -> dict:
        """Create PLY shape dictionary for use in shapegroups.

        Args:
            value: PLY shape specification
            scene_dir: Base directory for mesh paths

        Returns:
            Mitsuba shape dictionary
        """
        shape_dict = {"type": "ply"}

        if "filename" in value:
            mesh_path = scene_dir / value["filename"]
            shape_dict["filename"] = str(mesh_path)

        if "face_normals" in value:
            shape_dict["face_normals"] = value["face_normals"]

        if "bsdf" in value:
            shape_dict["bsdf"] = value["bsdf"]

        return shape_dict

    def _create_transform_from_spec(self, transform_spec: dict):
        """Create Mitsuba transform from transform specification.

        Args:
            transform_spec: Dict with 'translate', 'rotate', 'scale' keys

        Returns:
            Mitsuba ScalarTransform4f
        """
        to_world = mi.ScalarTransform4f()

        if "translate" in transform_spec:
            x, y, z = transform_spec["translate"]
            to_world = to_world @ mi.ScalarTransform4f.translate([x, y, z])

        if "rotate" in transform_spec:
            rx, ry, rz = transform_spec["rotate"]
            if rx != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], rx)
            if ry != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], ry)
            if rz != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 0, 1], rz)

        if "scale" in transform_spec:
            scale = transform_spec["scale"]
            to_world = to_world @ mi.ScalarTransform4f.scale(scale)

        return to_world

    def _create_transform_from_object(self, obj: dict):
        """Create Mitsuba transform from object position, rotation, scale.

        Args:
            obj: Object dict with 'position', 'rotation', 'scale' keys

        Returns:
            Mitsuba ScalarTransform4f
        """
        x, y, z = obj["position"]
        scale = obj["scale"]

        to_world = mi.ScalarTransform4f.translate([x, y, z])

        if "rotation" in obj:
            rx, ry, rz = obj["rotation"]
            if rx != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([1, 0, 0], rx)
            if ry != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 1, 0], ry)
            if rz != 0:
                to_world = to_world @ mi.ScalarTransform4f.rotate([0, 0, 1], rz)

        to_world = to_world @ mi.ScalarTransform4f.scale(scale)
        return to_world
