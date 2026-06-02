"""Scene description assembly resource."""

import logging
from pathlib import Path
from typing import Optional

import yaml
from s2gos_utils.io.paths import open_file
from upath import UPath

from ..core.context import SceneResourceContext
from ..core.materials import build_material_index_map, landcover_material_to_index
from ..scene import create_s2gos_scene


def _read_sidecar_yaml(path: Optional[UPath]) -> dict:
    """Read a YAML sidecar file, returning {} if path is None or missing."""
    if path is None or not path.exists():
        return {}
    with open_file(path, "r") as f:
        return yaml.safe_load(f) or {}


def create_scene_description(ctx: SceneResourceContext) -> Optional[Path]:
    """Create the complete scene description from all generated assets.

    This is the final resource that assembles all processed components
    into a complete S2GOS scene description.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated scene description YAML file
    """
    logging.info("=== Creating Scene Description ===")

    target_mesh_path = ctx.dependency_outputs["target_mesh"]
    target_texture_path = ctx.dependency_outputs["target_texture"]

    if target_mesh_path is None or target_texture_path is None:
        raise ValueError(
            "Required target mesh and texture files not found from dependencies"
        )

    mesh_path = str(target_mesh_path.relative_to(ctx.output_dir))
    texture_path = str(target_texture_path.relative_to(ctx.output_dir))

    buffer_mesh_path = None
    buffer_texture_path = None
    buffer_size_km = None

    buffer_mesh_file = ctx.dependency_outputs.get("buffer_mesh")
    buffer_texture_file = ctx.dependency_outputs.get("buffer_texture")

    if (
        ctx.has_buffer
        and buffer_mesh_file is not None
        and buffer_texture_file is not None
    ):
        buffer_mesh_path = str(buffer_mesh_file.relative_to(ctx.output_dir))
        buffer_texture_path = str(buffer_texture_file.relative_to(ctx.output_dir))
        buffer_size_km = ctx.config.buffer.size_km

    background_selection_texture = None
    background_size_km = None

    background_texture_file = ctx.dependency_outputs.get("background_texture")
    if ctx.has_background and background_texture_file is not None:
        background_selection_texture = str(
            background_texture_file.relative_to(ctx.output_dir)
        )
        background_size_km = ctx.config.background.size_km

    buffer_dem_file = None
    if ctx.has_buffer and ctx.assets.buffer_dem_file:
        buffer_dem_file = str(ctx.assets.buffer_dem_file.relative_to(ctx.output_dir))

    hamster_data_paths = {
        k: UPath(v)
        for k, v in _read_sidecar_yaml(ctx.assets.hamster_paths_file).items()
    } or None
    if hamster_data_paths:
        logging.info(
            f"Scene description found HAMSTER data paths: {hamster_data_paths}"
        )
    else:
        logging.info("Scene description: No HAMSTER data paths found in context")

    additional_material_libraries = list(ctx.additional_material_libraries)
    if ctx.config.region_material_defs:
        additional_material_libraries.append(ctx.config.region_material_defs)

    full_index_map = build_material_index_map(ctx)
    landcover_names = set(
        landcover_material_to_index(
            ctx.config.data_sources.material_config_path.upath
        ).keys()
    )
    overlay_only = {k: v for k, v in full_index_map.items() if k not in landcover_names}
    region_material_indices = overlay_only or None

    # Build include_files from available sidecars
    include_files = []
    if ctx.assets.user_assets_file:
        include_files.append(
            str(ctx.assets.user_assets_file.relative_to(ctx.output_dir))
        )
    if ctx.assets.vegetation_objects_file:
        include_files.append(
            str(ctx.assets.vegetation_objects_file.relative_to(ctx.output_dir))
        )

    scene_description = create_s2gos_scene(
        scene_name=ctx.scene_name,
        mesh_path=mesh_path,
        texture_path=texture_path,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.aoi_size_km,
        resolution_m=ctx.dem_resolution_m,
        buffer_mesh_path=buffer_mesh_path,
        buffer_texture_path=buffer_texture_path,
        buffer_size_km=buffer_size_km,
        output_dir=ctx.output_dir,
        buffer_dem_file=buffer_dem_file,
        background_elevation=ctx.config.background.elevation
        if ctx.config.background is not None
        else None,
        background_selection_texture=background_selection_texture,
        background_size_km=background_size_km,
        dem_name=ctx.config.data_sources.dem.name,
        landcover_name=ctx.config.data_sources.landcover.name,
        material_config_path=ctx.config.data_sources.material_config_path.upath,
        landcover_mapping_overrides={},
        atmosphere_config=ctx.config.atmosphere,
        hamster_data_paths=hamster_data_paths,
        additional_material_libraries=additional_material_libraries,
        region_material_indices=region_material_indices,
    )

    # Set include_files on the scene description
    scene_description.include_files = include_files

    # Save scene description to file
    scene_description_file = ctx.output_dir / f"{ctx.scene_name}.yml"
    scene_description.save_yaml(scene_description_file)
    scene_description = scene_description.resolve_includes(ctx.output_dir)

    # Store in assets
    ctx.assets.config_file = scene_description_file
    ctx.assets.scene_description_file = scene_description_file

    # Store scene description in context for pipeline return
    ctx.scene_description = scene_description

    logging.info("=== Scene Generation Complete ===")
    logging.info(f"Scene description saved to: {scene_description_file}")
    if include_files:
        logging.info(f"  Include files: {include_files}")

    # Log summary of generated assets
    logging.info("Generated Assets Summary:")
    logging.info(f"  Target mesh: {target_mesh_path}")
    logging.info(f"  Target texture: {target_texture_path}")

    if buffer_mesh_path:
        logging.info(f"  Buffer mesh: {buffer_mesh_file}")
        logging.info(f"  Buffer texture: {buffer_texture_file}")

    if background_selection_texture:
        logging.info(f"  Background texture: {background_texture_file}")

    if hamster_data_paths:
        logging.info(f"  HAMSTER data: {len(hamster_data_paths)} surface areas")

    return scene_description_file
