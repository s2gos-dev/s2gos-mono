"""Water body configuration."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class WaterConfig(BaseModel):
    """Configuration for water bodies: shape comes from OSM, landcover only adds whole bodies OSM misses."""

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
        description="Material name for water bodies, defaulting to the landcover 'permanent_water_bodies' class mapping.",
    )
    default_waterway_half_width_m: float = Field(
        8.0,
        gt=0.0,
        description="Fallback half-width (m) for waterway types not in WATERWAY_HALF_WIDTH_M.",
    )
    exclude_intermittent: bool = Field(
        False,
        description="Skip OSM water features tagged intermittent=yes (e.g. seasonal streams/ponds that don't hold water year-round).",
    )

    mesh_gradient_threshold: float = Field(
        0.0,
        ge=0.0,
        description="Minimum DEM gradient magnitude (m/m) along a water body to trigger mesh flattening; defaults to 0.0 (always flatten) since sloped water is never physically correct.",
    )
    mesh_thin_water_skip_m: float = Field(
        0.0,
        ge=0.0,
        description="Water bodies with total width (m) below this value are skipped (not flattened); 0.0 disables.",
    )
    centerline_smoothing_window_m: float = Field(
        150.0,
        ge=0.0,
        description="Rolling-median smoothing window (m) applied to the DEM along linear water-body centerlines before mesh generation, to avoid baking raw DEM noise into the channel; 0.0 disables.",
    )

    landcover_completion: bool = Field(
        True,
        description=(
            "Add whole water bodies from landcover-only water-class connected "
            "components that OSM entirely misses (never patches OSM boundaries)."
        ),
    )
    landcover_completion_min_area_m2: float = Field(
        400.0,
        ge=0.0,
        description="Minimum area (m^2) for a landcover-only component to be treated as a real water body.",
    )
    landcover_completion_max_slope_deg: float = Field(
        3.0,
        ge=0.0,
        description=(
            "Maximum local DEM slope (degrees) for a landcover-only water pixel to "
            "be considered real water rather than a sloped-bank misclassification."
        ),
    )
    landcover_leak_fallback_material: str = Field(
        "wetland",
        description="Fallback material for landcover-classified water pixels that fall outside every vetted water body.",
    )

    max_water_render_slope_deg: float = Field(
        45.0,
        ge=0.0,
        le=90.0,
        description=(
            "Maximum local DEM slope (degrees) for a vetted water-body pixel to still be "
            "painted as water; steeper pixels are repainted with the nearest land material."
        ),
    )

    default_sea_level_m: float = Field(
        0.0,
        description="Fallback reference elevation (m) when a body has no finite DEM samples.",
    )
    erosion_margin_m: float = Field(
        10.0,
        ge=0.0,
        description="Base inward-erosion margin (m) for areal water polygons before DEM reference-elevation sampling (effective margin is max(this, 1.5 * dem_resolution_m)).",
    )
    outlier_reject_m: float = Field(
        1.5,
        ge=0.0,
        description="Reject DEM samples further than this (m) from the median when computing a body's reference elevation.",
    )
    shore_flat_margin_m: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Distance (m) areal water flattening extends past the shoreline so every "
            "painted water texel lies on flat mesh; None derives it as one texel "
            "diagonal plus one refined mesh-cell diagonal, 0.0 disables."
        ),
    )
    drop_m: float = Field(
        0.0,
        ge=0.0,
        description="Optional additional depth (m) subtracted below the sampled reference elevation.",
    )

    @model_validator(mode="after")
    def validate_file_source(self):
        if self.source == "file":
            if self.file_path is None:
                raise ValueError("file_path is required when source='file'")
            if not self.file_path.exists():
                raise ValueError(f"water data file not found: {self.file_path}")
        return self
