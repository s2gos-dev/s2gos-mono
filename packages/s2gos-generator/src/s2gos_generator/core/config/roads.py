"""Road infrastructure configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DEFAULT_ROAD_WIDTHS: dict[str, float] = {
    "motorway": 12.0,
    "trunk": 10.0,
    "primary": 8.0,
    "secondary": 7.0,
    "tertiary": 6.0,
    "residential": 5.0,
    "unclassified": 5.0,
    "service": 4.0,
    "track": 3.0,
}

DEFAULT_ROAD_WIDTH_FALLBACK = 5.0


class RoadsConfig(BaseModel):
    """Configuration for road infrastructure in scenes."""

    enabled: bool = Field(True, description="Enable road processing")
    source: Literal["overpass", "file"] = Field(
        "overpass", description="Data source for road geometry"
    )
    file_path: Optional[str] = Field(
        None, description="Path to road data JSON file (required when source='file')"
    )
    material_name: str = Field("asphalt", description="Material name to paint on roads")
    highway_types: Optional[list[str]] = Field(
        None,
        description="Highway types to include (None = all paved types from DEFAULT_ROAD_WIDTHS)",
    )
    width_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Override default road widths per highway type (meters)",
    )
    exclude_vegetation: bool = Field(
        True, description="Use road areas as vegetation exclusion zones"
    )
    vegetation_buffer_m: float = Field(
        2.0,
        ge=0.0,
        description="Extra buffer around roads for vegetation exclusion (meters)",
    )

    @model_validator(mode="after")
    def validate_file_source(self):
        if self.source == "file" and self.file_path is None:
            raise ValueError("file_path is required when source='file'")
        return self
