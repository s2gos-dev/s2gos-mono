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
    apply_road_flatten_batch,
    make_refinement_predicate,
    make_roughness_predicate,
)


def extract_dem(dem_data: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _make_elevation_fn(x: np.ndarray, y: np.ndarray, elev: np.ndarray):
    """Return a bilinear interpolation callable for the given DEM arrays."""
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    x0, y0 = float(x[0]), float(y[0])

    def elevation_fn(xy: np.ndarray) -> np.ndarray:
        x_idx = (xy[:, 0] - x0) / dx
        y_idx = (xy[:, 1] - y0) / dy
        return map_coordinates(elev, np.vstack((y_idx, x_idx)), order=1, mode="nearest")

    return elevation_fn


def build_decimated_grid(
    x: np.ndarray,
    y: np.ndarray,
    elev: np.ndarray,
    decimation_depth: int,
    decimation_tolerance_m: float,
    extra_max_depth: int = 0,
) -> AdaptiveGrid:
    """Build an adaptive quadtree coarsened by 2^decimation_depth, then refined
    back where the local plane-residual exceeds decimation_tolerance_m.

    With decimation_depth=0 this returns a native-resolution grid,
    same cost as a plain uniform grid construction.  extra_max_depth
    reserves additional refinement headroom for downstream feature-based passes.

    Args:
        x:                    1-D x coordinate array (DEM native resolution).
        y:                    1-D y coordinate array (DEM native resolution).
        elev:                 2-D elevation array matching (y × x) shape.
        decimation_depth:     Coarsening factor in powers of 2.
        decimation_tolerance_m: Max plane-residual (metres) before a cell is
                              refined back.  0 disables pyramid-driven refinement.
        extra_max_depth:      Additional depth levels reserved beyond
                              decimation_depth for downstream refinement passes.

    Returns:
        Unbalanced, untriangulated AdaptiveGrid.
    """
    stride = 1 << decimation_depth  # 1 when decimation_depth == 0
    x_base = x[::stride]
    y_base = y[::stride]

    # Preserve the last edge when array length is not stride-aligned
    if (len(x) - 1) % stride != 0:
        x_base = np.append(x_base, x[-1])
    if (len(y) - 1) % stride != 0:
        y_base = np.append(y_base, y[-1])

    grid = AdaptiveGrid(x_base, y_base, max_depth=decimation_depth + extra_max_depth)

    if decimation_depth > 0 and decimation_tolerance_m > 0:
        dx = (x[-1] - x[0]) / (len(x) - 1)
        dy = (y[-1] - y[0]) / (len(y) - 1)
        pyramid = DemErrorPyramid(
            elev, float(x[0]), float(y[0]), dx, dy, decimation_depth
        )
        grid.refine(
            make_roughness_predicate(pyramid, tolerance_m=decimation_tolerance_m),
            max_level=decimation_depth,
        )

    return grid


def refine_grid_for_operations(
    grid: AdaptiveGrid,
    operations: list[TerraformOperation] | None,
) -> None:
    """Refine the grid where any operation's influence_zone intersects (in-place).

    No-op when operations is falsy.
    """
    if not operations:
        return
    merged_zone = unary_union([op.influence_zone for op in operations])
    grid.refine(make_refinement_predicate(merged_zone))


def grid_to_terrain_mesh(
    grid: AdaptiveGrid,
    elevation_fn,
    operations: list[TerraformOperation] | None = None,
    flatten: bool = False,
    handle_nans: bool = True,
) -> trimesh.Trimesh:
    """Balance, triangulate, optionally flatten along operation centrelines, and
    remove NaN-containing faces.

    Args:
        grid:         Populated AdaptiveGrid (balanced in-place here).
        elevation_fn: Callable mapping (N, 2) xy array -> (N,) elevations.
        operations:   TerraformOperations used for flatten; ignored when flatten=False.
        flatten:      Apply batch vertex flattening along operation centrelines.
        handle_nans:  Remove faces whose vertices contain NaN elevations.

    Returns:
        Cleaned trimesh.Trimesh.
    """
    grid.balance()
    vertices, faces = grid.to_mesh(elevation_fn=elevation_fn)

    if flatten and operations:
        vertices = apply_road_flatten_batch(vertices, operations, elevation_fn)

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


def build_refined_mesh(
    dem_data: xr.DataArray,
    operations: list[TerraformOperation] | None,
    config,
    handle_nans: bool = True,
) -> trimesh.Trimesh:
    """Build an adaptive quadtree mesh with optional terraforming operations.

    Orchestrates: DEM extraction -> decimated grid -> operation refinement ->
    triangulation + flatten -> NaN cleanup.

    Args:
        dem_data:   DEM elevation DataArray.
        operations: List of :class:`TerraformOperation` to apply; pass ``None``
                    or an empty list to skip road-influence refinement and flattening.
        config:     :class:`MeshRefinementConfig` (``decimation_depth``,
                    ``decimation_tolerance_m``, ``max_depth``, ``flatten``, …).
        handle_nans: Remove faces whose vertices contain NaN elevations.

    Returns:
        Adaptive :class:`trimesh.Trimesh`.
    """
    x, y, elev = extract_dem(dem_data)
    elevation_fn = _make_elevation_fn(x, y, elev)
    grid = build_decimated_grid(
        x,
        y,
        elev,
        decimation_depth=config.decimation_depth,
        decimation_tolerance_m=config.decimation_tolerance_m,
        extra_max_depth=config.max_depth,
    )
    refine_grid_for_operations(grid, operations)
    return grid_to_terrain_mesh(
        grid,
        elevation_fn,
        operations,
        flatten=config.flatten,
        handle_nans=handle_nans,
    )
