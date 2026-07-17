"""Orchestrate spectral diversification of a selection texture.

Pure array logic: given the current selection texture, the landcover class array,
and Sentinel-2 reflectance (all on the same scene grid, same orientation), cluster
each requested class, match clusters to library materials, allocate deduplicated
texture indices, and paint the matched indices into the texture.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .library import CandidateSpectrum
from .sam import NO_CLUSTER, cluster_class_reflectance, match_clusters_to_library
from ...core.config.material_match import SpectralMatchingConfig
from ...core.materials import allocate_matched_indices


def diversify_selection_texture(
    texture_2d: np.ndarray,
    landcover_2d: np.ndarray,
    refl: np.ndarray,
    cfg: SpectralMatchingConfig,
    base_index_map: Dict[str, int],
    library: List[CandidateSpectrum],
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Paint spectrally-matched materials into ``texture_2d``.

    All arrays must share the same ``(ny, nx)`` grid and orientation (south-row-0,
    matching the on-disk selection texture and the landcover ``.values``).

    Args:
        texture_2d: ``(ny, nx)`` uint8 selection indices (modified in place).
        landcover_2d: ``(ny, nx)`` ESA WorldCover class codes.
        refl: ``(band, ny, nx)`` reflectance, band order matching ``library`` vectors.
        cfg: Spectral diversification configuration.
        base_index_map: ``{material_name: index}`` for landcover + overlay materials.
        library: Candidate diffuse materials.

    Returns:
        ``(texture_2d, material_defs, material_indices)`` where ``material_defs``
        maps each *newly introduced* material id to its definition dict, and
        ``material_indices`` maps every matched material id to its texture index.
    """
    ny, nx = texture_2d.shape
    if landcover_2d.shape != (ny, nx):
        raise ValueError(
            f"landcover shape {landcover_2d.shape} != texture grid {(ny, nx)}"
        )
    if refl.shape[1:] != (ny, nx):
        raise ValueError(
            f"reflectance grid {refl.shape[1:]} != texture grid {(ny, nx)}"
        )

    # First pass: cluster + match every class, collecting matched ids so indices
    # can be allocated once (deterministically) before any painting.
    per_class: List[Tuple[int, np.ndarray, List]] = []
    matched_ids: List[str] = []
    for esa_class in cfg.landcover_classes:
        class_mask = landcover_2d == esa_class
        n_pixels = int(class_mask.sum())
        if n_pixels == 0:
            logging.info(
                "Spectral diversification: class %s absent, skipping", esa_class
            )
            continue
        label_map, palette = cluster_class_reflectance(
            refl, class_mask, cfg.clusters_per_class, cfg.random_seed
        )
        n_missing = n_pixels - int((label_map != NO_CLUSTER).sum())
        if n_missing:
            logging.warning(
                "Spectral diversification: class %s — %d/%d pixel(s) have no "
                "reflectance and keep their base material",
                esa_class,
                n_missing,
                n_pixels,
            )
        matches = match_clusters_to_library(palette, library, cfg.max_sam_angle_deg)
        per_class.append((esa_class, label_map, matches))
        matched_ids.extend(m.material_id for m in matches if m is not None)

    index_map = allocate_matched_indices(base_index_map, matched_ids)

    # Second pass: paint each matched cluster; unmatched clusters keep the base index.
    material_defs: Dict[str, Dict[str, Any]] = {}
    material_indices: Dict[str, int] = {}
    for esa_class, label_map, matches in per_class:
        painted = 0
        for cluster_id, candidate in enumerate(matches):
            if candidate is None:
                continue
            idx = index_map[candidate.material_id]
            cluster_pixels = label_map == cluster_id
            texture_2d[cluster_pixels] = idx
            painted += int(cluster_pixels.sum())
            material_indices[candidate.material_id] = idx
            if candidate.material_id not in base_index_map:
                material_defs[candidate.material_id] = candidate.material_def
        logging.info(
            "Spectral diversification: class %s → %d pixel(s) across %d matched cluster(s)",
            esa_class,
            painted,
            sum(1 for m in matches if m is not None),
        )

    return texture_2d, material_defs, material_indices


def matched_materials_to_sidecar(
    material_defs: Dict[str, Any],
    material_indices: Dict[str, int],
    landcover_classes,
) -> dict:
    """Serialize spectral-matching results to the matched-materials sidecar (schema v1)."""
    return {
        "version": 1,
        "materials": material_defs,
        "material_indices": material_indices,
        "source_landcover_classes": list(landcover_classes),
    }


def matched_materials_from_sidecar(data: dict) -> dict:
    """Read the matched-materials sidecar into ``{"materials", "material_indices"}``.

    Returns an empty dict for an unrecognised schema version.
    """
    version = data.get("version", 1)
    if version != 1:
        logging.warning(
            "Unknown matched materials sidecar version %s; skipping", version
        )
        return {}
    return {
        "materials": data.get("materials", {}),
        "material_indices": data.get("material_indices", {}),
    }
