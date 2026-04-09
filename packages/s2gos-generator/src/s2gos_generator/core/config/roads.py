"""Road infrastructure configuration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DEFAULT_LANE_COUNTS: dict[str, int] = {
    "motorway": 3,
    "motorway_link": 3,
    "trunk": 2,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 2,
    "unclassified": 2,
    "living_street": 1,
    "service": 1,
    "pedestrian": 1,
    "track": 1,
    "footway": 1,
    "cycleway": 1,
    "path": 1,
    "bridleway": 1,
    "busway": 2,
}

DEFAULT_LANE_WIDTHS: dict[str, float] = {
    "motorway": 3.5,
    "motorway_link": 3.5,
    "trunk": 3.5,
    "primary": 3.2,
    "primary_link": 3.2,
    "secondary": 3.1,
    "secondary_link": 3.1,
    "tertiary": 3.1,
    "tertiary_link": 3.1,
    "residential": 3.0,
    "unclassified": 3.0,
    "living_street": 3.0,
    "service": 2.5,
    "pedestrian": 2.0,
    "track": 2.5,
    "footway": 1.5,
    "cycleway": 1.5,
    "path": 1.5,
    "bridleway": 2.0,
    "busway": 3.25,
}

DEFAULT_SURFACE_MATERIAL_MAPPING: dict[str, str] = {
    "asphalt": "asphalt",
    "paved": "asphalt",
    "chipseal": "asphalt",
    "concrete": "concrete",
    "concrete:plates": "concrete",
    "cement": "concrete",
    "sett": "concrete",
    "cobblestone": "concrete",
    "unhewn_cobblestone": "concrete",
    "paving_stones": "concrete",
    "gravel": "gravel_road",
    "fine_gravel": "gravel_road",
    "compacted": "gravel_road",
    "pebblestone": "gravel_road",
    "dirt": "baresoil",
    "earth": "baresoil",
    "ground": "baresoil",
    "mud": "baresoil",
    "sand": "baresoil",
    "unpaved": "baresoil",
    "grass": "grassland",
    "grass_paver": "grassland",
}


class RoadsConfig(BaseModel):
    """Configuration for road infrastructure in scenes."""

    enabled: bool = Field(True, description="Enable road processing")
    source: Literal["overpass", "file"] = Field(
        "overpass", description="Data source for road geometry"
    )
    file_path: Optional[str] = Field(None, description="Path to road data JSON file")

    highway_types: Optional[list[str]] = Field(
        None, description="Highway types to include"
    )

    total_width_overrides: dict[str, float] = Field(default_factory=dict)
    lane_width_overrides: dict[str, float] = Field(default_factory=dict)
    lane_count_overrides: dict[str, int] = Field(default_factory=dict)
    surface_material_overrides: dict[str, str] = Field(default_factory=dict)

    default_material: str = Field("asphalt")
    default_lane_width_m: float = Field(3.0, gt=0.0)
    default_shoulder_m: float = Field(0.5, ge=0.0)
    exclude_vegetation: bool = Field(True)
    vegetation_buffer_m: float = Field(2.0, ge=0.0)

    @property
    def surface_material_mapping(self) -> dict[str, str]:
        """Dynamically combine default materials and user overrides.
        This property fixes external dependencies like the scene assembler."""
        return {**DEFAULT_SURFACE_MATERIAL_MAPPING, **self.surface_material_overrides}

    @property
    def lane_width_mapping(self) -> dict[str, float]:
        """Dynamically combine default lane widths and user overrides."""
        return {**DEFAULT_LANE_WIDTHS, **self.lane_width_overrides}

    @property
    def lane_count_mapping(self) -> dict[str, int]:
        """Dynamically combine default lane counts and user overrides."""
        return {**DEFAULT_LANE_COUNTS, **self.lane_count_overrides}

    @model_validator(mode="after")
    def validate_file_source(self):
        if self.source == "file" and self.file_path is None:
            raise ValueError("file_path is required when source='file'")
        return self
