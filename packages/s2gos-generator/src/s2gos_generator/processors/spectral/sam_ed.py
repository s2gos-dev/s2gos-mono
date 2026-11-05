"""SAM-ED: brightness-aware hybrid of the Spectral Angle Mapper and Euclidean distance."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .library import CandidateSpectrum
from .sam import spectral_angle


def sam_ed_distance(v1: np.ndarray, v2: np.ndarray, weight: float = 0.5) -> float:
    """Weighted sum of normalized spectral angle and Euclidean distance."""
    norms = float(np.linalg.norm(v1)) + float(np.linalg.norm(v2))
    distance = float(np.linalg.norm(v1 - v2)) / norms if norms else 0.0
    return weight * (spectral_angle(v1, v2) / (np.pi / 2)) + (1.0 - weight) * distance


def match_clusters_to_library(
    palette: np.ndarray,
    library: List[CandidateSpectrum],
    max_sam_angle_deg: Optional[float] = None,
    weight: float = 0.5,
) -> List[Optional[CandidateSpectrum]]:
    """Match each cluster to its closest material by SAM-ED, gated on the raw SAM angle."""
    matches: List[Optional[CandidateSpectrum]] = []
    for centroid in palette:
        if not np.isfinite(centroid).all() or not centroid.any():
            matches.append(None)
            continue
        best = min(
            library, key=lambda c: sam_ed_distance(centroid, c.s2_vector, weight)
        )
        angle = np.degrees(spectral_angle(centroid, best.s2_vector))
        rejected = max_sam_angle_deg is not None and angle > max_sam_angle_deg
        matches.append(None if rejected else best)
    return matches
