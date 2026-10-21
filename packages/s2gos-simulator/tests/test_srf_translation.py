import math

import numpy as np
import pytest
import xarray as xr

from s2gos_simulator.config import (
    HYPSTAR_SRF,
    SpectralRegion,
    SpectralResponse,
    WavelengthGrid,
)

FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))
GAUSSIAN_CUTOFF_SIGMAS = 3


def _srf_from(translator):
    """Return the translate_srf method (bound) from a translator fixture."""
    return translator.translate_srf


class TestTranslateSRFBasic:
    def test_translate_srf_none(self, translator):
        assert _srf_from(translator)(None) is None

    def test_translate_srf_string_passthrough(self, translator):
        result = _srf_from(translator)("sentinel_2a_msi_b02")
        assert result == "sentinel_2a_msi_b02"

    def test_translate_srf_string_returns_string_type(self, translator):
        result = _srf_from(translator)("any_dataset_id")
        assert isinstance(result, str)


class TestTranslateSRFDelta:
    def test_translate_srf_delta_type(self, translator):
        srf = SpectralResponse(type="delta", wavelengths=[550.0, 665.0])
        result = _srf_from(translator)(srf)

        assert result["type"] == "multi_delta"
        assert result["wavelengths"] == [550.0, 665.0]


class TestTranslateSRFUniform:
    def test_translate_srf_uniform_type(self, translator):
        srf = SpectralResponse(type="uniform", wmin=400.0, wmax=800.0)
        result = _srf_from(translator)(srf)
        assert result["type"] == "uniform"
        assert result["wmin"] == 400.0
        assert result["wmax"] == 800.0


class TestTranslateSRFGaussian:
    def test_translate_srf_gaussian_spectral_regions_only(self, translator):
        srf = SpectralResponse(
            type="gaussian",
            spectral_regions=[
                SpectralRegion(name="VNIR", wmin_nm=380.0, wmax_nm=1000.0, fwhm_nm=3.0),
                SpectralRegion(
                    name="SWIR", wmin_nm=1000.0, wmax_nm=1680.0, fwhm_nm=10.0
                ),
            ],
        )
        result = _srf_from(translator)(srf)

        assert result["type"] == "uniform"
        assert result["wmin"] == 380.0
        assert result["wmax"] == 1680.0

    def test_translate_srf_gaussian_fallback_range(self, translator):
        srf = SpectralResponse(type="gaussian", fwhm_nm=10.0)
        result = _srf_from(translator)(srf)

        assert result["type"] == "uniform"
        assert result["wmin"] == 400.0
        assert result["wmax"] == 2500.0

    def test_translate_srf_hypstar_preset_type(self, translator):
        result = _srf_from(translator)(HYPSTAR_SRF)

        assert result["type"] == "uniform"
        assert result["wmin"] == 340.0
        assert result["wmax"] == 1690.0

    def test_translate_srf_gaussian_explicit_output_grid(self, translator):
        fwhm = 8.5
        grid_min, grid_max = 450.0, 750.0
        srf = SpectralResponse(
            type="gaussian",
            spectral_regions=[
                SpectralRegion(
                    name="VNIR", wmin_nm=400.0, wmax_nm=1000.0, fwhm_nm=fwhm
                ),
            ],
            output_grid=WavelengthGrid(
                mode="explicit", wavelengths_nm=[grid_min, 550.0, 650.0, grid_max]
            ),
        )
        result = _srf_from(translator)(srf)

        margin = GAUSSIAN_CUTOFF_SIGMAS * fwhm * FWHM_TO_SIGMA
        assert result["type"] == "uniform"
        assert result["wmin"] == pytest.approx(grid_min - margin)
        assert result["wmax"] == pytest.approx(grid_max + margin)

    def test_translate_srf_gaussian_from_file_covers_file_grid(
        self, translator, tmp_path
    ):
        grid_min, grid_max = 350.2, 1686.9
        vnir, swir = HYPSTAR_SRF.spectral_regions
        path = tmp_path / "reference.nc"
        xr.Dataset(
            coords={"wavelength": np.linspace(grid_min, grid_max, 50)}
        ).to_netcdf(path)
        srf = SpectralResponse(
            type="gaussian",
            spectral_regions=HYPSTAR_SRF.spectral_regions,
            output_grid=WavelengthGrid(mode="from_file", file_path=str(path)),
        )
        result = _srf_from(translator)(srf)

        vnir_margin = GAUSSIAN_CUTOFF_SIGMAS * vnir.fwhm_nm * FWHM_TO_SIGMA
        swir_margin = GAUSSIAN_CUTOFF_SIGMAS * swir.fwhm_nm * FWHM_TO_SIGMA
        assert result["wmin"] == pytest.approx(grid_min - vnir_margin)
        assert result["wmax"] == pytest.approx(grid_max + swir_margin)
