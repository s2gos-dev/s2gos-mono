"""Building mesh construction: footprint loading, extrusion onto the DEM, and hip roofs.

Chain: footprints are loaded and extruded (`meshing`), each pitched roof is lifted from a
2D straight skeleton of the footprint (`roof`), and that skeleton is computed by
Felkel & Obdrzalek's algorithm (`skeleton`).
"""

from .meshing import (
    BuildingMeshes,
    BuildingMeshStats,
    build_meshes,
    load_building_footprints,
    make_dem_elevation_sampler,
    quadkeys_for_bbox,
    select_tile_files,
)
from .roof import build_hip_roof, compute_pitched_geometry
from .skeleton import Skeleton

__all__ = [
    "BuildingMeshes",
    "BuildingMeshStats",
    "build_meshes",
    "load_building_footprints",
    "make_dem_elevation_sampler",
    "quadkeys_for_bbox",
    "select_tile_files",
    "build_hip_roof",
    "compute_pitched_geometry",
    "Skeleton",
]
