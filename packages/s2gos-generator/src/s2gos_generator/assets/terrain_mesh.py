from __future__ import annotations

import logging

import numpy as np
import trimesh
import xarray as xr
from scipy.ndimage import map_coordinates
from shapely.ops import unary_union

from .adaptive_grid import AdaptiveGrid
from .error_pyramid import DemErrorPyramid
from .terraforming import (
    TerraformOperation,
    apply_operations_batch,
    make_refinement_predicate,
    make_roughness_predicate,
)


def _extract_dem(dem_data: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and extract coordinate arrays from a DEM DataArray.

    Returns:
        (x, y, elev) — 1-D coordinate arrays and 2-D elevation array.
    """
    dem_data.load()
    if "x" in dem_data.dims and "y" in dem_data.dims:
        x, y = dem_data.x.values, dem_data.y.values
    elif "lon" in dem_data.dims and "lat" in dem_data.dims:
        x, y = dem_data.lon.values, dem_data.lat.values
    else:
        raise ValueError("DEM data must have either (x, y) or (lon, lat) coordinates")
    return x, y, dem_data.values


def build_refined_mesh(
    dem_data: xr.DataArray,
    operations: list[TerraformOperation] | None,
    config,
    handle_nans: bool = True,
) -> trimesh.Trimesh:
    """Build an adaptive quadtree mesh with optional terraforming operations.

    The quadtree is refined wherever any operation's ``influence_zone`` intersects
    a cell, then all operations are applied sequentially to the vertex array.

    Args:
        dem_data:   DEM elevation DataArray.
        operations: List of :class:`TerraformOperation` to apply; pass ``None``
                    or an empty list to skip adaptive refinement and flattening.
        config:     :class:`MeshRefinementConfig` (``max_depth``, ``flatten``, …).
        handle_nans: Remove faces whose vertices contain NaN elevations.

    Returns:
        Adaptive :class:`trimesh.Trimesh`.
    """
    x, y, elev = _extract_dem(dem_data)

    dx_dem = (x[-1] - x[0]) / (len(x) - 1)
    dy_dem = (y[-1] - y[0]) / (len(y) - 1)
    x0_dem, y0_dem = float(x[0]), float(y[0])

    def elevation_fn(xy: np.ndarray) -> np.ndarray:
        x_idx = (xy[:, 0] - x0_dem) / dx_dem
        y_idx = (xy[:, 1] - y0_dem) / dy_dem
        return map_coordinates(elev, np.vstack((y_idx, x_idx)), order=1, mode="nearest")

    # 1. Subsample the base coords for the quadtree
    stride = 1 << config.decimation_depth  # 1 when decimation disabled
    x_base = x[::stride]
    y_base = y[::stride]

    # Ensure the AOI's last edge is preserved (array length may not be divisible)
    if x_base[-1] != x[-1]:
        x_base = np.append(x_base, x[-1])
    if y_base[-1] != y[-1]:
        y_base = np.append(y_base, y[-1])

    # 2. Extend max_depth so refinement reaches the correct physical size
    grid = AdaptiveGrid(
        x_base, y_base, max_depth=config.decimation_depth + config.max_depth
    )

    # 3. First Refinement Pass: Terrain roughness decimation
    if config.decimation_depth > 0 and config.decimation_tolerance_m > 0:
        pyramid = DemErrorPyramid(
            elev, x0_dem, y0_dem, dx_dem, dy_dem, config.decimation_depth
        )
        grid.refine(
            make_roughness_predicate(
                pyramid, tolerance_m=config.decimation_tolerance_m
            ),
            max_level=config.decimation_depth,
        )

    # 4. Second Refinement Pass: Terraforming operations (roads, etc.)
    if operations:
        merged_zone = unary_union([op.influence_zone for op in operations])
        predicate = make_refinement_predicate(merged_zone)
        grid.refine(predicate)

    grid.balance()

    vertices, faces = grid.to_mesh(elevation_fn=elevation_fn)

    if config.flatten and operations:
        # Batch: creates shapely points once, uses STRtree.query for all roads
        vertices = apply_operations_batch(vertices, operations, elevation_fn)

    if handle_nans:
        valid = ~np.isnan(vertices[:, 2])
        if not valid.all():
            faces = faces[valid[faces].all(axis=1)]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()

    logging.debug(
        "Adaptive mesh: %d vertices, %d faces (operations=%d)",
        len(mesh.vertices),
        len(mesh.faces),
        len(operations) if operations else 0,
    )
    return mesh
