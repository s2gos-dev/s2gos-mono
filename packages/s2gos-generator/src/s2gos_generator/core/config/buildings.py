from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator


class BuildingsConfig(BaseModel):
    """Configuration for building generation."""

    enabled: bool = True
    file_paths: list[Path] = Field(
        default_factory=list,
        description="One or more GPKG files containing building footprints.",
    )
    layer_name: str = Field("building", description="GPKG layer name to read.")
    height_column: str = Field(
        "height",
        description=(
            "Column with building height. Accepts either a numeric height in "
            "meters or a building-taxonomy string (e.g. 'HHT:10.0', 'H:3', "
            "'HAPP:3', 'HBET:1-3'). Buildings whose value is missing or "
            "unparseable fall back to default_height_m."
        ),
    )
    object_id_prefix: str = Field(
        "building",
        description=(
            "Prefix for the generated scene-object ids. Buildings are merged "
            "into one mesh per material, so ids are '<prefix>' (single "
            "material), '<prefix>_<material>', and '<prefix>_roof_<material>'."
        ),
    )
    default_height_m: float = Field(3.0, gt=0.0)
    story_height_m: float = Field(3.0, gt=0.0)
    material: Union[str, dict[str, float]] = Field(
        "concrete",
        description=(
            "Either a single material name applied to every building, or a "
            "{name: weight} mapping. Weights are normalized; each building is "
            "assigned one material drawn by weighted random sampling, then "
            "buildings are grouped into one mesh per material."
        ),
    )
    material_seed: Optional[int] = Field(
        None,
        description="Seed for reproducible per-building material assignment.",
    )
    pitched_roof_proportion: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of eligible buildings that get a pitched roof. "
            "0 disables pitched roofs entirely (flat-top everywhere)."
        ),
    )
    pitched_roof_min_area_m2: Optional[float] = Field(
        None,
        gt=0.0,
        description="Footprint area below which a building stays flat-topped.",
    )
    pitched_roof_min_height_m: Optional[float] = Field(
        None, gt=0.0, description="Building height below which it stays flat-topped."
    )
    roof_pitch_deg: float = Field(
        30.0, gt=0.0, lt=80.0, description="Pitch angle for hip roofs."
    )
    roof_height_m: float = Field(
        3.0,
        gt=0.0,
        description=(
            "Target roof height (apex above eaves). Capped per-building when the "
            "footprint is too narrow to support it at the configured pitch."
        ),
    )
    roof_material: str = Field(
        "baresoil", description="Material name applied to the roof mesh group."
    )
    roof_seed: Optional[int] = Field(
        None,
        description="Seed for reproducible per-building pitched/flat selection.",
    )
    elevation_offset_m: float = Field(
        0.0, description="Extra Z offset added on top of the DEM-sampled base."
    )
    base_skirt_m: float = Field(
        0.5,
        ge=0.0,
        description=(
            "Extrude this far below the building base so steep terrain doesn't expose "
            "a gap between footprint and DEM."
        ),
    )
    roof_workers: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Number of worker processes to use for per-building mesh construction. "
            "None defaults to max(os.cpu_count() // 2, 1). Set to 1 for sequential "
            "execution (useful for debugging or when subprocess overhead dominates "
            "the workload, e.g. very small scenes)."
        ),
    )

    @model_validator(mode="after")
    def _check_files(self):
        for p in self.file_paths:
            if not Path(p).exists():
                raise ValueError(f"GPKG not found: {p}")
        return self

    @model_validator(mode="after")
    def _check_material(self):
        if isinstance(self.material, dict):
            if not self.material:
                raise ValueError("material dict must not be empty")
            if any(w <= 0 for w in self.material.values()):
                raise ValueError("material weights must be > 0")
            if len(self.material) == 1:
                self.material = next(iter(self.material))
        return self
