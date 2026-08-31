"""Vegetation placement resource."""

import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from s2gos_utils.io.paths import exists, open_file

from ..core.context import SceneResourceContext
from ..core.exceptions import DataNotFoundError
from ..processors.vegetation import (
    _filter_by_buildings,
    _filter_by_exclusion_zones,
    _filter_by_ways,
    _process_vegetation_with_shared_datasets,
    save_vegetation_collection_binary,
)


def _load_exclusion_zones(ctx: SceneResourceContext) -> List[Dict[str, Any]]:
    """Load all exclusion zones from context."""
    return list(ctx.exclusion_zone_geometries)


def process_target_vegetation(
    ctx: SceneResourceContext,
) -> Any:
    """Process multi-species vegetation placement using landcover data.

    Args:
        ctx: Scene resource context

    Returns:
        List of vegetation placement dictionaries with position, rotation, species data.
        Returns empty list if vegetation is disabled or not configured.

    Raises:
        DataNotFoundError: If required landcover or DEM data is missing
    """
    vegetation_config = ctx.config.vegetation_placement
    if vegetation_config is None or not vegetation_config.enabled:
        logging.info("Vegetation disabled - skipping vegetation placement")
        return None

    if vegetation_config.random_seed is not None:
        random.seed(vegetation_config.random_seed)
        np.random.seed(vegetation_config.random_seed)
        logging.info(
            "Random seed set to %s for reproducible generation",
            vegetation_config.random_seed,
        )

    landcover_path = ctx.dependency_outputs.get("target_landcover")
    dem_path = ctx.dependency_outputs.get("target_dem")

    if landcover_path is None:
        raise DataNotFoundError(
            "Landcover data not available for vegetation placement. "
            "Ensure target_landcover resource is enabled and processed."
        )

    if dem_path is None:
        raise DataNotFoundError(
            "DEM data not available for vegetation placement. "
            "Ensure target_dem resource is enabled and processed."
        )

    if not exists(landcover_path):
        raise DataNotFoundError(f"Landcover file not found: {landcover_path}")
    if not exists(dem_path):
        raise DataNotFoundError(f"DEM file not found: {dem_path}")

    vegetation_instances = _process_vegetation_with_shared_datasets(
        landcover_path, dem_path, vegetation_config
    )

    exclusion_zones = _load_exclusion_zones(ctx)

    if exclusion_zones:
        vegetation_instances = _filter_by_exclusion_zones(
            vegetation_instances, exclusion_zones
        )

    way_exclusion = ctx.config.vegetation_placement.way_exclusion
    vegetation_instances = _filter_by_ways(
        vegetation_instances,
        ctx.ways,
        enabled=way_exclusion.enabled,
        buffer_m=way_exclusion.buffer_m,
    )

    # Buildings always have priority over vegetation.
    vegetation_instances = _filter_by_buildings(
        vegetation_instances, ctx.building_footprints
    )

    if not vegetation_instances:
        return None

    vegetation_objects, vegetation_materials = _build_vegetation_objects(
        vegetation_instances, ctx
    )

    if vegetation_objects:
        sidecar_data = {"objects": vegetation_objects}
        if vegetation_materials:
            sidecar_data["materials"] = vegetation_materials
        sidecar_path = ctx.data_dir / "vegetation_objects.yml"
        ctx.data_dir.mkdir(parents=True, exist_ok=True)
        with open_file(sidecar_path, "w") as f:
            yaml.dump(sidecar_data, f, default_flow_style=False, indent=2)
        ctx.assets.vegetation_objects_file = sidecar_path
        logging.info(f"Saved vegetation objects sidecar: {sidecar_path}")

    return ctx.assets.vegetation_objects_file


def _build_vegetation_objects(
    vegetation_instances: List[Dict[str, Any]],
    ctx: SceneResourceContext,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group vegetation instances by species/asset, save .npy binaries, and build SD objects.

    Returns:
        List of SceneDescription-format object dicts (shapegroups + vegetation_collections).
    """
    from ..processors.xml_importer import create_tree_shapegroup

    species_groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for instance in vegetation_instances:
        species_name = instance.get("species", "unknown")
        asset_xml = instance.get("asset_xml", "tree.xml")
        key = (species_name, asset_xml)
        if key not in species_groups:
            species_groups[key] = []
        species_groups[key].append(instance)

    logging.info(f"Found {len(species_groups)} distinct vegetation species groups")

    objects: List[Dict[str, Any]] = []
    all_materials: Dict[str, Any] = {}  # namespaced materials for the sidecar

    for (species_name, asset_xml), instances in species_groups.items():
        asset_xml_path = asset_xml.upath
        asset_basename = asset_xml_path.stem

        binary_filename = f"{ctx.scene_name}_{species_name}_{asset_basename}.npy"
        binary_path = ctx.data_dir / binary_filename
        ctx.data_dir.mkdir(parents=True, exist_ok=True)

        vegetation_metadata = save_vegetation_collection_binary(instances, binary_path)

        if vegetation_metadata["count"] == 0:
            continue

        vegetation_shapegroup, vegetation_materials = create_tree_shapegroup(
            asset_xml_path, ctx.output_dir
        )

        for mat_id, mat_def in vegetation_materials.items():
            namespaced_mat_id = f"{species_name}_{asset_basename}_{mat_id}"
            all_materials[namespaced_mat_id] = mat_def

        for component_key, component in vegetation_shapegroup.items():
            if isinstance(component, dict) and "bsdf" in component:
                original_mat_id = component["bsdf"]["id"]
                if original_mat_id.startswith("_mat_"):
                    raw_mat_id = original_mat_id[5:]
                    component["bsdf"]["id"] = (
                        f"_mat_{species_name}_{asset_basename}_{raw_mat_id}"
                    )

        species_shapegroup_id = f"vegetation_shapegroup_{species_name}_{asset_basename}"
        species_group_id = f"vegetation_group_{species_name}_{asset_basename}"
        vegetation_shapegroup["id"] = species_group_id

        shapegroup_obj = {
            "object_id": species_shapegroup_id,
            "type": "shapegroup",
            **vegetation_shapegroup,
        }

        # Relative data_file path from output_dir
        data_file_rel = str(binary_path.relative_to(ctx.output_dir))

        vegetation_collection_obj = {
            "object_id": f"vegetation_collection_{species_name}_{asset_basename}",
            "type": "vegetation_collection",
            "shapegroup_ref": species_group_id,
            "data_file": data_file_rel,
            "count": len(instances),
            "collection_name": f"{species_name}_{asset_basename}",
        }

        objects.append(shapegroup_obj)
        objects.append(vegetation_collection_obj)

        logging.info(
            f"Built vegetation objects for {species_name}/{asset_basename}: "
            f"{vegetation_metadata['count']} instances → {binary_filename}"
        )

    return objects, all_materials
