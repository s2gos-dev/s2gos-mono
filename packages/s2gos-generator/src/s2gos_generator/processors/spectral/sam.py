"""Clustering and spectral matching (SAM, optionally with a Euclidean brightness term)."""

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


def spectral_distance(v1: np.ndarray, v2: np.ndarray, sam_weight: float = 1.0) -> float:
    """Blend of spectral angle and normalised Euclidean distance, in ``[0, 1]``.

    ``sam_weight * angle / (pi/2) + (1 - sam_weight) * |v1 - v2| / (|v1| + |v2|)``.
    The angle term compares shape only, the Euclidean term also separates
    brightness. ``sam_weight=1.0`` ranks candidates exactly like SAM.
    """
    norms = float(np.linalg.norm(v1)) + float(np.linalg.norm(v2))
    euclidean = float(np.linalg.norm(v1 - v2)) / norms if norms else 0.0
    angle = spectral_angle(v1, v2) / (np.pi / 2)
    return sam_weight * angle + (1.0 - sam_weight) * euclidean


def match_clusters_to_library(
    palette: np.ndarray,
    library: List[CandidateSpectrum],
    max_sam_angle_deg: Optional[float] = None,
    sam_weight: float = 1.0,
) -> List[Optional[CandidateSpectrum]]:
    """Match each non-empty cluster to its closest library material.

    Args:
        palette: ``(n_clusters, band)`` mean reflectance per cluster.
        library: Candidate materials with band-aligned vectors.
        max_sam_angle_deg: If set, only candidates within this spectral angle of a
            cluster are considered, and a cluster with none keeps no match
            (``None`` in the returned list), so the caller can fall back to the
            base landcover material.
        sam_weight: Weight of the spectral angle in :func:`spectral_distance`,
            which ranks the remaining candidates. 1.0 is pure SAM.

    Returns:
        List aligned with ``palette`` rows. Each entry is the matched
        :class:`CandidateSpectrum`, or ``None`` for an empty cluster, an all-zero
        centroid, or no candidate within the angle gate, each of which leaves the
        base landcover material in place.
    """
    matches: List[Optional[CandidateSpectrum]] = []
    for centroid in palette:
        if not np.isfinite(centroid).all() or not centroid.any():
            matches.append(None)
            continue
        allowed = [
            c
            for c in library
            if max_sam_angle_deg is None
            or np.degrees(spectral_angle(centroid, c.s2_vector)) <= max_sam_angle_deg
        ]
        matches.append(
            min(
                allowed,
                key=lambda c: spectral_distance(centroid, c.s2_vector, sam_weight),
                default=None,
            )
        )
    return matches
