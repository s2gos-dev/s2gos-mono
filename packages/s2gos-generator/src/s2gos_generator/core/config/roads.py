"""Road infrastructure configuration."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field, model_validator


class HighwayDefaults(NamedTuple):
    """Per-highway-type geometry and material defaults."""

    lane_count: int
    lane_width_m: float
    default_material: str


class RoadsConfig(BaseModel):
    """Configuration for road infrastructure in scenes."""

    ROAD_TYPE_TABLE: ClassVar[dict[str, HighwayDefaults]] = {
        "motorway": HighwayDefaults(
            lane_count=3, lane_width_m=3.5, default_material="asphalt"
        ),
        "motorway_link": HighwayDefaults(
            lane_count=3, lane_width_m=3.5, default_material="asphalt"
        ),
        "trunk": HighwayDefaults(
            lane_count=2, lane_width_m=3.5, default_material="asphalt"
        ),
        "primary": HighwayDefaults(
            lane_count=2, lane_width_m=3.2, default_material="asphalt"
        ),
        "primary_link": HighwayDefaults(
            lane_count=1, lane_width_m=3.2, default_material="asphalt"
        ),
        "secondary": HighwayDefaults(
            lane_count=2, lane_width_m=3.1, default_material="asphalt"
        ),
        "secondary_link": HighwayDefaults(
            lane_count=1, lane_width_m=3.1, default_material="asphalt"
        ),
        "tertiary": HighwayDefaults(
            lane_count=2, lane_width_m=3.1, default_material="asphalt"
        ),
        "tertiary_link": HighwayDefaults(
            lane_count=1, lane_width_m=3.1, default_material="asphalt"
        ),
        "residential": HighwayDefaults(
            lane_count=2, lane_width_m=3.0, default_material="asphalt"
        ),
        "unclassified": HighwayDefaults(
            lane_count=2, lane_width_m=3.0, default_material="asphalt"
        ),
        "living_street": HighwayDefaults(
            lane_count=1, lane_width_m=3.0, default_material="asphalt"
        ),
        "service": HighwayDefaults(
            lane_count=1, lane_width_m=2.5, default_material="asphalt"
        ),
        "pedestrian": HighwayDefaults(
            lane_count=1, lane_width_m=2.0, default_material="concrete"
        ),
        "track": HighwayDefaults(
            lane_count=1, lane_width_m=2.5, default_material="gravel_road"
        ),
        "footway": HighwayDefaults(
            lane_count=1, lane_width_m=1.5, default_material="gravel_road"
        ),
        "cycleway": HighwayDefaults(
            lane_count=1, lane_width_m=1.5, default_material="asphalt"
        ),
        "path": HighwayDefaults(
            lane_count=1, lane_width_m=1.5, default_material="gravel_road"
        ),
        "bridleway": HighwayDefaults(
            lane_count=1, lane_width_m=2.0, default_material="gravel_road"
        ),
        "busway": HighwayDefaults(
            lane_count=2, lane_width_m=3.25, default_material="asphalt"
        ),
    }

    DEFAULT_SURFACE_MATERIALS: ClassVar[dict[str, str]] = {
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

    enabled: bool = Field(True, description="Enable road processing")
    source: Literal["overpass", "file"] = Field(
        "overpass", description="Data source for road geometry"
    )
    file_path: Optional[Path] = Field(None, description="Path to road data JSON file")

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
    mesh_gradient_threshold: float = Field(
        0.02,
        ge=0.0,
        description=(
            "Minimum DEM gradient magnitude (m/m) along a road centreline to trigger "
            "cross-slope mesh flattening for that segment (~1.1°). "
            "Set to 0.0 to always flatten regardless of terrain slope."
        ),
    )
    mesh_thin_road_bypass_m: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Roads with total width (m) below this value bypass the gradient check "
            "and are not flattened. Set to 0.0 (default) to disable "
        ),
    )

    @property
    def surface_material_mapping(self) -> dict[str, str]:
        return {**self.DEFAULT_SURFACE_MATERIALS, **self.surface_material_overrides}

    @property
    def lane_width_mapping(self) -> dict[str, float]:
        base = {k: v.lane_width_m for k, v in self.ROAD_TYPE_TABLE.items()}
        return {**base, **self.lane_width_overrides}

    @property
    def lane_count_mapping(self) -> dict[str, int]:
        base = {k: v.lane_count for k, v in self.ROAD_TYPE_TABLE.items()}
        return {**base, **self.lane_count_overrides}

    @model_validator(mode="after")
    def validate_file_source(self):
        if self.source == "file":
            if self.file_path is None:
                raise ValueError("file_path is required when source='file'")
            if not self.file_path.exists():
                raise ValueError(f"road data file not found: {self.file_path}")
        return self
