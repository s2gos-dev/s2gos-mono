"""Spectral matching: match clustered Sentinel-2 reflectance to materials.

See [SpectralMatchingConfig][s2gos_generator.core.config.material_match.SpectralMatchingConfig]
for the user-facing configuration and ``resources/texture.py`` for how the pieces
are wired into the generation pipeline.
"""

from .library import CandidateSpectrum, load_candidate_library
from .sam import (
    cluster_class_reflectance,
    match_clusters_to_library,
    spectral_angle,
)

__all__ = [
    "CandidateSpectrum",
    "load_candidate_library",
    "cluster_class_reflectance",
    "match_clusters_to_library",
    "spectral_angle",
]
