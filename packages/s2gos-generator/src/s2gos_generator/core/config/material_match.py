"""Spectral matching configuration.

Optionally re-textures selected landcover classes by matching clustered Sentinel-2
reflectance to a library of candidate materials (Spectral Angle Mapper). See
[SpectralMatchingConfig][s2gos_generator.core.config.material_match.SpectralMatchingConfig].
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from s2gos_utils.io.paths import PathRef
from s2gos_utils.io.resolver import resolver

# Sentinel-2 band centre wavelengths (nm). Only these 10 m bands are supported.
S2_BAND_WAVELENGTHS_NM: dict[str, float] = {
    "B02": 490.0,  # Blue
    "B03": 560.0,  # Green
    "B04": 665.0,  # Red
    "B08": 842.0,  # NIR
}


class SpectralMatchingConfig(BaseModel):
    """Spectral matching of selected landcover classes.

    Presence (non-``None``) on
    [SceneGenConfig][s2gos_generator.core.config.scene.SceneGenConfig] enables the
    feature. For each listed landcover class the generator fetches Sentinel-2
    reflectance, k-means clusters the pixels of that class, and matches each cluster
    to the best-fitting **diffuse** material from ``material_library`` via
    Spectral Angle Mapper — painting those richer materials into the selection
    texture instead of a single flat index.

    Copernicus credentials are resolved by ``credential_id`` through the
    settings/secrets credential provider (``.secrets.yaml``), never embedded here.
    """

    landcover_classes: List[int] = Field(
        ...,
        min_length=1,
        description="ESA WorldCover class codes to diversify (e.g. [30, 60]).",
    )
    material_library: PathRef = Field(
        ...,
        description=(
            "Path to a materials.json-style file. Only its 'diffuse' entries are "
            "used as candidate materials for spectral matching."
        ),
    )
    clusters_per_class: int = Field(
        4,
        ge=1,
        le=64,
        description="Number of k-means clusters per diversified landcover class.",
    )
    acquisition_date: str = Field(
        ...,
        description="Anchor acquisition date 'YYYY-MM-DD' for the Sentinel-2 composite.",
    )
    search_window_days: int = Field(
        310,
        ge=0,
        description="± days around acquisition_date to search for usable scenes.",
    )
    max_cloud_cover: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Maximum eo:cloud_cover (percent) for candidate scenes.",
    )
    bands: List[str] = Field(
        default_factory=lambda: ["B02", "B03", "B04", "B08"],
        description="Sentinel-2 10 m bands used for matching.",
    )
    stac_url: str = Field(
        "https://stac.dataspace.copernicus.eu/v1",
        description="STAC catalog endpoint.",
    )
    credential_id: Optional[str] = Field(
        None,
        description=(
            "Id of an s3 credential (in .secrets.yaml / settings) used to read the "
            "Copernicus 'eodata' bucket. None falls back to ambient AWS_* env vars."
        ),
    )
    random_seed: Optional[int] = Field(
        None,
        ge=0,
        description="Seed for reproducible k-means clustering.",
    )
    max_sam_angle_deg: Optional[float] = Field(
        None,
        gt=0.0,
        le=90.0,
        description=(
            "If set, clusters whose best SAM angle exceeds this threshold keep their "
            "base landcover material instead of a spectral match (quality gate)."
        ),
    )

    @field_validator("material_library")
    @classmethod
    def _validate_library_exists(cls, v):
        path = resolver.resolve(v)
        if not path.exists():
            raise ValueError(f"material_library does not exist: {v}")
        return v

    @field_validator("acquisition_date")
    @classmethod
    def _validate_date(cls, v):
        from datetime import datetime

        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"acquisition_date must be 'YYYY-MM-DD', got {v!r}"
            ) from exc
        return v

    @field_validator("bands")
    @classmethod
    def _validate_bands(cls, v):
        if not v:
            raise ValueError("bands must list at least one Sentinel-2 band")
        unknown = [b for b in v if b not in S2_BAND_WAVELENGTHS_NM]
        if unknown:
            raise ValueError(
                f"Unsupported Sentinel-2 band(s) {unknown}; "
                f"supported: {sorted(S2_BAND_WAVELENGTHS_NM)}"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"Duplicate bands in {v}")
        return v
