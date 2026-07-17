"""Clustering and Spectral Angle Mapper matching."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2

from .library import CandidateSpectrum

# Cluster label assigned to pixels outside the class mask / not clustered.
NO_CLUSTER = -1


def spectral_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Spectral angle.

    Brightness-invariant: depends only on spectral *shape*. Returns ``pi/2``
    (maximally dissimilar) when either vector has zero norm.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float(np.pi / 2)
    cos_theta = np.clip(float(np.dot(v1, v2)) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def cluster_class_reflectance(
    refl: np.ndarray,
    class_mask: np.ndarray,
    n_clusters: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster the reflectance of masked pixels with k-means.

    Args:
        refl: ``(band, ny, nx)`` reflectance (band-sorted, same order as the
            candidate library vectors).
        class_mask: ``(ny, nx)`` bool — pixels of the target landcover class.
        n_clusters: Number of clusters (k).
        random_seed: Seed for reproducible clustering.

    Returns:
        ``(label_map, palette)`` where ``label_map`` is ``(ny, nx)`` int with
        :data:`NO_CLUSTER` outside the mask / for non-finite pixels, and
        ``palette`` is ``(n_clusters, band)`` of mean **original** reflectance per
        cluster (rows are NaN for empty clusters).
    """
    nbands, ny, nx = refl.shape
    if class_mask.shape != (ny, nx):
        raise ValueError(
            f"class_mask shape {class_mask.shape} != reflectance grid {(ny, nx)}"
        )

    feats = refl.reshape(nbands, -1).T
    valid = np.all(np.isfinite(feats), axis=1) & class_mask.reshape(-1)

    label_map = np.full(ny * nx, NO_CLUSTER, dtype=int)
    palette = np.full((n_clusters, nbands), np.nan, dtype="float64")

    X = feats[valid]
    if X.shape[0] == 0:
        return label_map.reshape(ny, nx), palette

    # Z-score per band so NIR's larger dynamic range doesn't dominate the distance.
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    k = min(n_clusters, X.shape[0])
    _centroids, labels = kmeans2(Xs, k, minit="++", seed=random_seed, missing="warn")

    for cluster in range(k):
        members = labels == cluster
        if members.any():
            palette[cluster] = X[members].mean(0)
    label_map[valid] = labels
    return label_map.reshape(ny, nx), palette


def match_clusters_to_library(
    palette: np.ndarray,
    library: List[CandidateSpectrum],
    max_sam_angle_deg: Optional[float] = None,
) -> List[Optional[CandidateSpectrum]]:
    """Match each non-empty cluster to its closest library material by SAM.

    Args:
        palette: ``(n_clusters, band)`` mean reflectance per cluster.
        library: Candidate materials with band-aligned vectors.
        max_sam_angle_deg: If set, clusters whose best angle exceeds this keep no
            match (``None`` in the returned list), so the caller can fall back to
            the base landcover material.

    Returns:
        List aligned with ``palette`` rows; entry is the matched
        :class:`CandidateSpectrum`, or ``None`` for an empty cluster, an all-zero
        centroid, or a rejected match, each of which leaves the base landcover material in place.
    """
    matches: List[Optional[CandidateSpectrum]] = []
    for centroid in palette:
        if not np.isfinite(centroid).all() or not centroid.any():
            matches.append(None)
            continue
        best, best_angle = None, float("inf")
        for candidate in library:
            angle = spectral_angle(centroid, candidate.s2_vector)
            if angle < best_angle:
                best, best_angle = candidate, angle
        if max_sam_angle_deg is not None and np.degrees(best_angle) > max_sam_angle_deg:
            matches.append(None)
        else:
            matches.append(best)
    return matches
