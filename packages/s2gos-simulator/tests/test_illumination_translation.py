from datetime import datetime

import pytest
from pydantic import ValidationError

from s2gos_simulator.backends.eradiate.eradiate_translator import EradiateTranslator
from s2gos_simulator.backends.eradiate.geometry_utils import GeometryUtils
from s2gos_simulator.config import (
    AstroObjectIllumination,
    ConstantIllumination,
    DirectionalIllumination,
)


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


class TestAstroObjectTranslation:
    """The astro object type must reach Eradiate intact, not be absorbed by
    the directional branch that precedes it in the isinstance chain."""

    def test_translate_astro_object_illumination(self, make_config, astro_illumination):
        config = make_config(illumination=astro_illumination)
        translator = EradiateTranslator(config, GeometryUtils())

        result = translator.translate_illumination()

        assert result["type"] == "astro_object"
        assert result["id"] == "illumination"
        assert result["zenith"].magnitude == pytest.approx(30.0)
        assert result["azimuth"].magnitude == pytest.approx(135.0)
        assert result["angular_diameter"].magnitude == pytest.approx(0.5358)
        assert str(result["angular_diameter"].units) == "degree"
        assert result["irradiance"] == {
            "type": "solar_irradiance",
            "dataset": "thuillier_2003",
        }

    def test_angular_diameter_is_forwarded(self, make_config):
        illum = AstroObjectIllumination(zenith=10.0, azimuth=20.0, angular_diameter=2.0)
        translator = EradiateTranslator(
            make_config(illumination=illum), GeometryUtils()
        )

        result = translator.translate_illumination()

        assert result["angular_diameter"].magnitude == pytest.approx(2.0)

    @pytest.mark.parametrize("angular_diameter", [0.0, -1.0, 180.0, 200.0])
    def test_out_of_range_angular_diameter_rejected(self, angular_diameter):
        with pytest.raises(ValidationError):
            AstroObjectIllumination(angular_diameter=angular_diameter)


class TestIrradianceDatetime:
    """`irradiance_datetime` scales irradiance by the Earth-Sun distance. It
    must be forwarded when set and omitted entirely when not, so illuminations
    without one keep an unscaled spectrum."""

    @pytest.mark.parametrize("cls", [DirectionalIllumination, AstroObjectIllumination])
    def test_datetime_forwarded_when_set(self, make_config, cls):
        obs = datetime(2024, 1, 4, 12, 0, 0)
        illum = cls(zenith=30.0, azimuth=135.0, irradiance_datetime=obs)
        translator = EradiateTranslator(
            make_config(illumination=illum), GeometryUtils()
        )

        assert translator.translate_illumination()["irradiance"] == {
            "type": "solar_irradiance",
            "dataset": "thuillier_2003",
            "datetime": obs,
        }

    @pytest.mark.parametrize("cls", [DirectionalIllumination, AstroObjectIllumination])
    def test_datetime_absent_when_unset(self, make_config, cls):
        illum = cls(zenith=30.0, azimuth=135.0)
        translator = EradiateTranslator(
            make_config(illumination=illum), GeometryUtils()
        )

        assert "datetime" not in translator.translate_illumination()["irradiance"]
