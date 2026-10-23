"""Water body configuration."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class WaterConfig(BaseModel):
    """Configuration for OSM water bodies, completed with landcover water."""

    WATERWAY_HALF_WIDTH_M: ClassVar[dict[str, float]] = {
        "river": 5.0,
        "canal": 5.0,
        "stream": 3.0,
        "drain": 1.5,
        "ditch": 1.0,
    }

    enabled: bool = Field(True, description="Enable water body processing")
    source: Literal["overpass", "file"] = Field(
        "overpass", description="Data source for water body geometry"
    )
    file_path: Optional[Path] = Field(None, description="Path to water data JSON file")

    default_material: str = Field(
        "water",
        description="Material name for water bodies.",
    )
    default_waterway_half_width_m: float = Field(
        8.0,
        gt=0.0,
        description="Fallback half-width (m) for waterway types not in WATERWAY_HALF_WIDTH_M.",
    )
    exclude_intermittent: bool = Field(
        False,
        description="Skip OSM water features tagged intermittent=yes.",
    )

    mesh_gradient_threshold: float = Field(
        0.0,
        ge=0.0,
        description="Minimum DEM gradient (m/m) along a water body for it to be flattened; 0.0 always flattens.",
    )
    mesh_thin_water_skip_m: float = Field(
        0.0,
        ge=0.0,
        description="Water bodies narrower than this (m) are not flattened; areal width is estimated as 2 * area / perimeter. 0.0 disables.",
    )
    centerline_smoothing_window_m: float = Field(
        150.0,
        ge=0.0,
        description="Smoothing window (m) of the DEM profile along linear waterways before flattening; 0.0 disables.",
    )

    landcover_completion: bool = Field(
        True,
        description="Add landcover water-class areas that OSM misses.",
    )
    landcover_completion_min_area_m2: float = Field(
        400.0,
        ge=0.0,
        description="Minimum area (m^2) of an added landcover water area.",
    )
    landcover_completion_linked_min_area_m2: float = Field(
        1_000_000.0,
        ge=0.0,
        description="Minimum area (m^2) of an added landcover water area that touches OSM water; smaller ones are treated as OSM outline disagreement.",
    )
    landcover_completion_max_slope_deg: float = Field(
        3.0,
        ge=0.0,
        description="Maximum DEM slope (degrees) of an added landcover water pixel.",
    )
    landcover_leak_fallback_material: str = Field(
        "wetland",
        description="Material painted over landcover water pixels outside every water body.",
    )

    erosion_margin_m: float = Field(
        10.0,
        ge=0.0,
        description="Inward erosion (m) of areal water before sampling its elevation, at least 1.5 * dem_resolution_m.",
    )
    outlier_reject_m: float = Field(
        1.5,
        ge=0.0,
        description="DEM samples further than this (m) from the median are ignored when computing water elevation.",
    )
    sloped_water_tolerance_m: float = Field(
        3.0,
        ge=0.0,
        description="Areal water whose smoothed elevation varies more than this (m) follows the terrain instead of one level.",
    )
    sloped_water_smoothing_m: float = Field(
        50.0,
        gt=0.0,
        description="Smoothing scale (m) of the sloped water surface; the level test uses three times this.",
    )
    sloped_water_max_deviation_m: float = Field(
        5.0,
        ge=0.0,
        description="Maximum distance (m) of a sloped water surface from the DEM.",
    )
    shore_flat_margin_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance (m) areal water flattening extends past the shoreline; None derives one texel plus one refined mesh-cell diagonal, 0.0 disables.",
    )

    @model_validator(mode="after")
    def validate_file_source(self):
        if self.source == "file":
            if self.file_path is None:
                raise ValueError("file_path is required when source='file'")
            if not self.file_path.exists():
                raise ValueError(f"water data file not found: {self.file_path}")
        return self
