"""Candidate material library for spectral matching.

Loads a ``materials.json``-style library and reduces each **diffuse** material to a
reflectance vector sampled at the Sentinel-2 band centres, so clusters can be
matched against it with the Spectral Angle Mapper (see :mod:`.sam`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from s2gos_utils.io.paths import PathLike, open_dataset
from s2gos_utils.scene.materials import load_materials
from s2gos_utils.scene.materials.definitions import DiffuseMaterial

from ..core.config.material_match import S2_BAND_WAVELENGTHS_NM


@dataclass(frozen=True)
class CandidateSpectrum:
    """A diffuse material reduced to a Sentinel-2 reflectance vector.

    Attributes:
        material_id: Stable id from the library (used for dedup and indexing).
        material_def: The material definition dict (``Material.to_dict`` form,
            with resolved spectral paths) for registration in the SceneDescription.
        s2_vector: Reflectance sampled at the requested band centres, ordered to
            match the ``bands`` passed to :func:`load_candidate_library`.
    """

    material_id: str
    material_def: Dict[str, Any]
    s2_vector: np.ndarray


def _wavelengths_to_nm(wavelengths: np.ndarray) -> np.ndarray:
    """Return wavelengths in nm, converting from µm when values look like µm.

    Reflectance spectra span roughly 400–2500 nm or 0.4–2.5 µm; a max below 100
    unambiguously indicates µm.
    """
    wavelengths = np.asarray(wavelengths, dtype="float64")
    if wavelengths.size and float(np.nanmax(wavelengths)) < 100.0:
        return wavelengths * 1000.0
    return wavelengths


def _interp_to_bands(
    wl: np.ndarray, ref: np.ndarray, band_centers_nm: np.ndarray
) -> np.ndarray | None:
    """Sort a spectrum and interpolate it at the S2 band centres (nm).

    Returns ``None`` when the spectrum does not span the requested bands; matching
    by extrapolation would invent reflectance and bias the spectral angle.
    """
    order = np.argsort(wl)
    wl, ref = wl[order], ref[order]
    if wl.min() > band_centers_nm.min() or wl.max() < band_centers_nm.max():
        return None
    return np.interp(band_centers_nm, wl, ref)


def _file_spectrum_vector(
    refl_param: Dict[str, Any], band_centers_nm: np.ndarray
) -> np.ndarray | None:
    """Sample a file-based reflectance spectrum at the S2 band centres (nm)."""
    da = open_dataset(refl_param["path"])[refl_param["variable"]]
    wl = _wavelengths_to_nm(da[da.dims[0]].values)
    ref = np.asarray(da.values, dtype="float64")
    return _interp_to_bands(wl, ref, band_centers_nm)


def _spectrum_vector(
    refl_param: Dict[str, Any], band_centers_nm: np.ndarray
) -> np.ndarray | None:
    """Reduce a diffuse ``reflectance`` parameter to a band-centre vector."""
    if "path" in refl_param and "variable" in refl_param:
        return _file_spectrum_vector(refl_param, band_centers_nm)
    if refl_param.get("type") == "interpolated":
        wl = _wavelengths_to_nm(np.asarray(refl_param["wavelengths"], dtype="float64"))
        ref = np.asarray(refl_param["values"], dtype="float64")
        return _interp_to_bands(wl, ref, band_centers_nm)
    if refl_param.get("type") == "uniform":
        value = refl_param["value"]
        if isinstance(value, (list, tuple)):  # RGB triplet has no SWIR/NIR info
            return None
        return np.full(band_centers_nm.shape, float(value))
    return None


def load_candidate_library(
    material_library: PathLike,
    bands: List[str],
) -> List[CandidateSpectrum]:
    """Load diffuse candidate materials and their Sentinel-2 reflectance vectors.

    Args:
        material_library: Path to a ``materials.json``-style file.
        bands: Sentinel-2 bands defining the vector ordering (e.g.
            ``["B02", "B03", "B04", "B08"]``).

    Returns:
        One :class:`CandidateSpectrum` per usable diffuse material. Non-diffuse
        materials, RGB-uniform diffuse materials, and spectra that do not span the
        requested bands are skipped (logged).
    """
    band_centers_nm = np.array(
        [S2_BAND_WAVELENGTHS_NM[b] for b in bands], dtype="float64"
    )

    materials = load_materials(material_library)
    library: List[CandidateSpectrum] = []
    skipped = 0
    for material_id, material in materials.items():
        if not isinstance(material, DiffuseMaterial):
            skipped += 1
            continue
        vector = _spectrum_vector(material.reflectance, band_centers_nm)
        if (
            vector is None
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0
        ):
            skipped += 1
            continue
        library.append(
            CandidateSpectrum(
                material_id=material_id,
                material_def=material.to_dict(),
                s2_vector=vector.astype("float64"),
            )
        )

    logging.info(
        "Spectral library: %d diffuse candidate(s) from %s (%d skipped)",
        len(library),
        material_library,
        skipped,
    )
    if not library:
        raise ValueError(
            f"No usable diffuse candidate materials found in {material_library}. "
            "Spectral diversification needs at least one diffuse material whose "
            "reflectance spectrum spans the requested Sentinel-2 bands."
        )
    return library
