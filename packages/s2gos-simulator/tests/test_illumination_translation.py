import pytest
from pydantic import ValidationError

from s2gos_simulator.backends.eradiate.eradiate_translator import EradiateTranslator
from s2gos_simulator.backends.eradiate.geometry_utils import GeometryUtils
from s2gos_simulator.config import ConstantIllumination


class TestIlluminationTranslation:
    def test_translate_directional_illumination(self, translator):
        result = translator.translate_illumination()

        assert result["type"] == "directional"
        assert result["id"] == "illumination"

        assert result["zenith"].magnitude == pytest.approx(30.0)
        assert str(result["zenith"].units) == "degree"
        assert result["azimuth"].magnitude == pytest.approx(135.0)
        assert str(result["azimuth"].units) == "degree"

        assert result["irradiance"] == {
            "type": "solar_irradiance",
            "dataset": "thuillier_2003",
        }

    def test_translate_constant_illumination(self, make_config):
        illum = ConstantIllumination(radiance=2.5, id="custom_const")
        config = make_config(illumination=illum)
        translator = EradiateTranslator(config, GeometryUtils())

        result = translator.translate_illumination()

        assert result == {
            "type": "constant",
            "id": "custom_const",
            "radiance": 2.5,
        }

    def test_translate_unsupported_illumination(self, make_config):
        class InvalidIllumination:
            pass

        with pytest.raises(ValidationError):
            _ = make_config(illumination=InvalidIllumination())
