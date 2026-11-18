"""Raster terrain data: find, merge, and regrid DEM and land-cover tiles for an AOI."""

from .base_processor import BaseTileProcessor
from .datautil import regrid_to_projection
from .dem import DEMProcessor
from .landcover import ESA_LANDCOVER_CLASSES, LandCoverProcessor

__all__ = [
    "BaseTileProcessor",
    "DEMProcessor",
    "LandCoverProcessor",
    "ESA_LANDCOVER_CLASSES",
    "regrid_to_projection",
]
