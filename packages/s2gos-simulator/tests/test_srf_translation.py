from s2gos_simulator.config import (
    HYPSTAR_SRF,
    SpectralRegion,
    SpectralResponse,
    WavelengthGrid,
)


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
        assert result["wmin"] == 380.0
        assert result["wmax"] == 1680.0

    def test_translate_srf_gaussian_explicit_output_grid(self, translator):
        srf = SpectralResponse(
            type="gaussian",
            spectral_regions=[
                SpectralRegion(name="VNIR", wmin_nm=400.0, wmax_nm=1000.0, fwhm_nm=8.5),
            ],
            output_grid=WavelengthGrid(
                mode="explicit", wavelengths_nm=[450.0, 550.0, 650.0, 750.0]
            ),
        )
        result = _srf_from(translator)(srf)

        assert result["type"] == "uniform"
        assert result["wmin"] == 450.0
        assert result["wmax"] == 750.0
