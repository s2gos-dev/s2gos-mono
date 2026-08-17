"""Terrain mesh construction: adaptive quadtree, decimation, and terrain flattening."""

from .adaptive_grid import AdaptiveGrid
from .builder import (
    build_decimated_grid,
    build_refined_mesh,
    extract_dem,
    grid_to_terrain_mesh,
    refine_grid_for_operations,
)
from .error_pyramid import DemErrorPyramid
from .mesh_generator import MeshGenerator
from .terraforming import (
    GradientFilter,
    WayFlattenOperation,
    TerraformOperation,
    apply_way_flatten_batch,
    compute_gradient,
    make_refinement_predicate,
    make_roughness_predicate,
)

__all__ = [
    "AdaptiveGrid",
    "DemErrorPyramid",
    "MeshGenerator",
    "GradientFilter",
    "WayFlattenOperation",
    "TerraformOperation",
    "apply_way_flatten_batch",
    "compute_gradient",
    "make_refinement_predicate",
    "make_roughness_predicate",
    "build_decimated_grid",
    "build_refined_mesh",
    "extract_dem",
    "grid_to_terrain_mesh",
    "refine_grid_for_operations",
]
