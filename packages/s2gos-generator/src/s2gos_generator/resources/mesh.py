"""3D mesh generation resources."""

import logging
from pathlib import Path
from typing import Optional

import trimesh
import xarray as xr
from s2gos_utils.io.paths import expand_mapper

from ..core.context import SceneResourceContext
from ..processors.terrain_mesh import MeshGenerator, TerraformOperation
from ..processors.terrain_mesh.surround import build_buffer_strips_mesh


def generate_target_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate 3D mesh from target area DEM data.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated target mesh PLY file
    """

    dem_file_path = ctx.dependency_outputs["target_dem"]
    if dem_file_path is None:
        raise ValueError("Target DEM file not found from dependencies")

    mesh_generator = MeshGenerator()
    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_terrain.ply"

    dem_dataset = xr.open_zarr(expand_mapper(dem_file_path))
    dem_data = dem_dataset["elevation"]

    refinement_cfg = ctx.config.mesh_refinement

    if refinement_cfg is not None and refinement_cfg.enabled:
        operations: list[TerraformOperation] = []
        if ctx.config.ways is not None and ctx.config.ways.enabled:
            from ..processors.ways import build_way_terraform_operations

            operations.extend(
                build_way_terraform_operations(
                    ctx.ways,
                    dem_data,
                    transition_buffer_m=refinement_cfg.transition_buffer_m,
                    gradient_threshold=ctx.config.ways.mesh_gradient_threshold,
                    thin_way_skip_m=ctx.config.ways.mesh_thin_way_skip_m,
                )
            )

        if operations:
            logging.info(
                "Adaptive mesh refinement: %d operation(s), max_depth=%d",
                len(operations),
                refinement_cfg.max_depth,
            )
        else:
            logging.info(
                "Adaptive mesh refinement: decimation only (no feature operations)"
            )
        mesh = mesh_generator.adaptive_dem_to_mesh(
            dem_data,
            operations,
            refinement_cfg,
            handle_nans=ctx.config.processing.handle_dem_nans,
        )
    else:
        mesh = mesh_generator.dem_to_mesh(
            dem_data,
            handle_nans=ctx.config.processing.handle_dem_nans,
        )

    mesh = mesh_generator.add_uv_coordinates(mesh)
    mesh_generator.save_mesh(mesh, mesh_path)
    ctx.assets.mesh_file = mesh_path

    logging.info("Target mesh: %s", mesh_path)
    return mesh_path


def generate_buffer_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate the buffer mesh as four strips around the target mesh."""
    buffer_dem_file_path = ctx.dependency_outputs["buffer_dem"]
    if buffer_dem_file_path is None:
        raise ValueError("Buffer DEM file not found from dependencies")

    dem_data = xr.open_zarr(expand_mapper(buffer_dem_file_path))["elevation"]
    with ctx.dependency_outputs["target_mesh"].open("rb") as f:
        target_vertices = trimesh.load(f, file_type="ply", process=False).vertices

    mesh = build_buffer_strips_mesh(
        dem_data,
        target_vertices,
        handle_nans=ctx.config.processing.handle_dem_nans,
    )

    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_buffer_terrain.ply"
    MeshGenerator().save_mesh(mesh, mesh_path)
    ctx.assets.buffer_mesh_file = mesh_path

    return mesh_path
