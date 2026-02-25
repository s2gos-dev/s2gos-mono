"""Shared pytest fixtures for s2gos-simulator tests."""

from unittest.mock import MagicMock

import pytest
from upath import UPath

from s2gos_simulator.backends.eradiate.eradiate_translator import EradiateTranslator
from s2gos_simulator.backends.eradiate.geometry_utils import GeometryUtils
from s2gos_simulator.config import (
    ConstantIllumination,
    DirectionalIllumination,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
)
from s2gos_simulator.config.viewing import AngularViewing


def _minimal_satellite_sensor():
    """Create a minimal valid SatelliteSensor for use in fixtures."""
    return SatelliteSensor(
        id="test_sat",
        platform=SatellitePlatform.SENTINEL_2A,
        instrument=SatelliteInstrument.MSI,
        band="2",
        viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        film_resolution=(10, 10),
        target_center_lat=-42.0,
        target_center_lon=-64.5,
        target_size_km=1.0,
    )


@pytest.fixture
def minimal_scene():
    """Mocked SceneDescription with valid location and empty objects list."""
    scene = MagicMock()
    scene.location = {"center_lat": -42.0, "center_lon": -64.5}
    scene.objects = []
    return scene


@pytest.fixture
def scene_dir(tmp_path):
    """Temporary directory as UPath for scene data."""
    return UPath(tmp_path)


@pytest.fixture
def directional_illumination():
    """Minimal DirectionalIllumination fixture."""
    return DirectionalIllumination(zenith=30.0, azimuth=135.0)


@pytest.fixture
def constant_illumination():
    """Minimal ConstantIllumination fixture."""
    return ConstantIllumination(radiance=2.5)


@pytest.fixture
def make_config(directional_illumination):
    """Factory fixture: build a minimal valid SimulationConfig with one satellite sensor."""

    def _make(sensors=None, measurements=None, illumination=None):
        if sensors is None:
            sensors = [_minimal_satellite_sensor()]
        return SimulationConfig(
            name="test",
            illumination=illumination or directional_illumination,
            sensors=sensors,
            measurements=measurements or [],
        )

    return _make


@pytest.fixture
def translator(make_config):
    """EradiateTranslator with a minimal config and real GeometryUtils."""
    config = make_config()
    return EradiateTranslator(config, GeometryUtils())
