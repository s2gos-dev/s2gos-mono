import json
from datetime import datetime

import pytest

from s2gos_simulator.config import (
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
)
from s2gos_simulator.config.viewing import AngularViewing, LookAtViewing


def _illumination():
    return DirectionalIllumination(zenith=30.0, azimuth=135.0)


def _sat_sensor(sensor_id="sat1"):
    return SatelliteSensor(
        id=sensor_id,
        platform=SatellitePlatform.SENTINEL_2A,
        instrument=SatelliteInstrument.MSI,
        band="2",
        viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        film_resolution=(10, 10),
        target_center_lat=-42.0,
        target_center_lon=-64.5,
        target_size_km=1.0,
    )


def _ground_sensor(sensor_id="ground1"):
    return GroundSensor(
        id=sensor_id,
        instrument=GroundInstrumentType.RADIANCEMETER,
        viewing=LookAtViewing(
            origin=[0.0, 0.0, 1.0],
            target=[0.0, 0.0, 0.0],
            terrain_relative_height=False,
        ),
    )


def _location():
    return HemisphericalMeasurementLocation(
        target_x=0.0, target_y=0.0, target_z=0.0, terrain_relative_height=False
    )


def _irradiance_config(irr_id="irr1"):
    return IrradianceConfig(id=irr_id, location=_location())


def _make_config(**kwargs):
    return SimulationConfig(name="test", illumination=_illumination(), **kwargs)


class TestToJsonSerialization:
    def test_model_serialization(self):
        config = _make_config(sensors=[_sat_sensor()])
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        json.loads(json_str)

        schema = config.model_json_schema()
        assert "properties" in schema

    def test_to_json_returns_string(self):
        config = _make_config(sensors=[_sat_sensor()])
        assert isinstance(config.to_json(), str)

    def test_to_json_is_valid_json(self):
        config = _make_config(sensors=[_sat_sensor()])
        json.loads(config.to_json())

    def test_to_json_created_at_is_iso_string(self):
        config = _make_config(sensors=[_sat_sensor()])
        data = json.loads(config.to_json())
        created_at = data["created_at"]
        assert isinstance(created_at, str)
        datetime.fromisoformat(created_at)

    def test_to_json_writes_file(self, tmp_path):
        config = _make_config(sensors=[_sat_sensor()])
        path = tmp_path / "config.json"
        config.to_json(path=path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test"


class TestFromJson:
    def _write(self, config, tmp_path):
        path = tmp_path / "config.json"
        config.to_json(path=path)
        return path

    def test_from_json_roundtrip(self, tmp_path):
        config = _make_config(sensors=[_sat_sensor()])
        restored = SimulationConfig.from_json(self._write(config, tmp_path))
        assert restored.name == "test"
        assert isinstance(restored, SimulationConfig)
        assert len(restored.sensors) == 1

    def test_from_json_preserves_satellite_sensor_fields(self, tmp_path):
        config = _make_config(sensors=[_sat_sensor()])
        restored = SimulationConfig.from_json(self._write(config, tmp_path))
        sensor = restored.sensors[0]
        assert isinstance(sensor, SatelliteSensor)
        assert sensor.id == "sat1"
        assert sensor.platform == SatellitePlatform.SENTINEL_2A
        assert sensor.instrument == SatelliteInstrument.MSI
        assert sensor.band == "2"
        assert sensor.film_resolution == (10, 10)
        assert sensor.target_center_lat == pytest.approx(-42.0)
        assert sensor.target_center_lon == pytest.approx(-64.5)

    def test_from_json_preserves_illumination_fields(self, tmp_path):
        config = _make_config(sensors=[_sat_sensor()])
        restored = SimulationConfig.from_json(self._write(config, tmp_path))
        illumination = restored.illumination
        assert isinstance(illumination, DirectionalIllumination)
        assert illumination.zenith == pytest.approx(30.0)
        assert illumination.azimuth == pytest.approx(135.0)

    def test_from_json_preserves_ground_sensor_fields(self, tmp_path):
        config = _make_config(sensors=[_ground_sensor("ground1")])
        restored = SimulationConfig.from_json(self._write(config, tmp_path))
        sensor = restored.sensors[0]
        assert isinstance(sensor, GroundSensor)
        assert sensor.id == "ground1"
        assert sensor.instrument == GroundInstrumentType.RADIANCEMETER
        assert isinstance(sensor.viewing, LookAtViewing)
        assert sensor.viewing.origin == pytest.approx([0.0, 0.0, 1.0])
        assert sensor.viewing.target == pytest.approx([0.0, 0.0, 0.0])
        assert sensor.viewing.terrain_relative_height is False

    def test_from_json_preserves_measurement_fields(self, tmp_path):
        irr = _irradiance_config("irr1")
        config = _make_config(sensors=[_sat_sensor()], measurements=[irr])
        restored = SimulationConfig.from_json(self._write(config, tmp_path))
        assert len(restored.measurements) == 1
        measurement = restored.measurements[0]
        assert isinstance(measurement, IrradianceConfig)
        assert measurement.id == "irr1"
        assert measurement.location.target_x == pytest.approx(0.0)
        assert measurement.location.target_y == pytest.approx(0.0)
        assert measurement.location.terrain_relative_height is False
