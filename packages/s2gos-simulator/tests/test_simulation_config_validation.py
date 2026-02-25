import pytest

from s2gos_simulator.config import (
    BRFConfig,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    PixelHDRFConfig,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
)
from s2gos_simulator.config.viewing import (
    AngularFromOriginViewing,
    AngularViewing,
    LookAtViewing,
)


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


class TestBasicValidation:
    def test_empty_sensors_and_measurements_raises(self):
        with pytest.raises(ValueError):
            _make_config(sensors=[], measurements=[])

    def test_config_with_sensor_passes(self):
        _make_config(sensors=[_sat_sensor()], measurements=[])

    def test_config_with_measurement_and_no_sensor_passes(self):
        brf = BRFConfig(
            id="brf1",
            viewing=AngularFromOriginViewing(
                origin=[0.0, 0.0, 1.0], zenith=0.0, azimuth=0.0
            ),
        )
        _make_config(sensors=[], measurements=[brf])

    def test_config_with_measurement_and_sensor_passes(self):
        brf = BRFConfig(
            id="brf1",
            viewing=AngularFromOriginViewing(
                origin=[0.0, 0.0, 1.0], zenith=0.0, azimuth=0.0
            ),
        )
        _make_config(sensors=[_sat_sensor()], measurements=[brf])


class TestSensorIDUniqueness:
    def test_duplicate_sensor_ids_raises(self):
        sensor1 = _ground_sensor("dup_id")
        sensor2 = _ground_sensor("dup_id")
        with pytest.raises(ValueError, match="unique"):
            _make_config(sensors=[sensor1, sensor2], measurements=[])

    def test_unique_sensor_ids_passes(self):
        sensor1 = _ground_sensor("sensor_a")
        sensor2 = _ground_sensor("sensor_b")
        _make_config(sensors=[sensor1, sensor2], measurements=[])

    def test_add_sensor_duplicate_raises(self):
        sensor1 = _sat_sensor("sat1")
        config = _make_config(sensors=[sensor1], measurements=[])
        sensor2 = _sat_sensor("sat1")
        with pytest.raises(ValueError):
            config.add_sensor(sensor2)

    def test_add_sensor_unique_passes(self):
        sensor1 = _sat_sensor("sat1")
        config = _make_config(sensors=[sensor1], measurements=[])
        sensor2 = _sat_sensor("sat2")
        config.add_sensor(sensor2)
        assert config.get_sensor("sat2") is not None


class TestMeasurementIDUniqueness:
    def test_duplicate_measurement_ids_raises(self):
        irr1 = _irradiance_config("irr1")
        irr2 = _irradiance_config("irr1")  # same id
        with pytest.raises(ValueError, match="Duplicate measurement ID"):
            _make_config(sensors=[_sat_sensor()], measurements=[irr1, irr2])

    def test_unique_measurement_ids_passes(self):
        irr1 = _irradiance_config("irr1")
        irr2 = _irradiance_config("irr2")
        _make_config(sensors=[_sat_sensor()], measurements=[irr1, irr2])


class TestHDRFValidation:
    def test_hdrf_references_nonexistent_sensor_raises(self):
        irr = _irradiance_config("irr1")
        hdrf = HDRFConfig(
            id="hdrf1",
            radiance_sensor_id="ghost_sensor",
            irradiance_measurement_id="irr1",
        )
        with pytest.raises(ValueError):
            _make_config(
                sensors=[_sat_sensor()],
                measurements=[irr, hdrf],
            )

    def test_hdrf_references_nonexistent_irradiance_raises(self):
        sensor = _ground_sensor("rad1")
        hdrf = HDRFConfig(
            id="hdrf1",
            radiance_sensor_id="rad1",
            irradiance_measurement_id="ghost_irr",
        )
        with pytest.raises(ValueError):
            _make_config(sensors=[sensor], measurements=[hdrf])

    def test_hdrf_references_non_irradiance_measurement_raises(self):
        brf = BRFConfig(
            id="brf1",
            viewing=AngularFromOriginViewing(
                origin=[0.0, 0.0, 1.0], zenith=0.0, azimuth=0.0
            ),
        )
        sensor = _ground_sensor("rad1")
        hdrf = HDRFConfig(
            id="hdrf1",
            radiance_sensor_id="rad1",
            irradiance_measurement_id="brf1",
        )
        with pytest.raises(ValueError):
            _make_config(sensors=[sensor], measurements=[brf, hdrf])

    def test_valid_hdrf_reference_mode_passes(self):
        sensor = _ground_sensor("rad1")
        irr = _irradiance_config("irr1")
        hdrf = HDRFConfig(
            id="hdrf1",
            radiance_sensor_id="rad1",
            irradiance_measurement_id="irr1",
        )
        _make_config(sensors=[sensor], measurements=[irr, hdrf])

    def test_valid_hdrf_with_auto_generation_passes(self):
        hdrf = HDRFConfig(
            id="hdrf_auto",
            instrument="radiancemeter",
            viewing=LookAtViewing(
                origin=[0.0, 0.0, 1.0],
                target=[0.0, 0.0, 0.0],
                terrain_relative_height=False,
            ),
            location=HemisphericalMeasurementLocation(
                target_x=0.0,
                target_y=0.0,
                target_z=0.0,
                terrain_relative_height=False,
            ),
            reference_height_offset_m=0.1,
        )
        _make_config(sensors=[_sat_sensor()], measurements=[hdrf])


class TestPixelHDRFValidation:
    def test_pixel_hdrf_requires_satellite_sensor(self):
        ground = _ground_sensor("my_ground")
        pixel_hdrf = PixelHDRFConfig(
            id="phdrf1",
            satellite_sensor_id="my_ground",
            pixel_indices=[(0, 0)],
        )
        with pytest.raises(ValueError):
            _make_config(sensors=[ground], measurements=[pixel_hdrf])

    def test_pixel_hdrf_valid_satellite_sensor_passes(self):
        sat = _sat_sensor("sat1")
        pixel_hdrf = PixelHDRFConfig(
            id="phdrf1",
            satellite_sensor_id="sat1",
            pixel_indices=[(0, 0)],
        )
        _make_config(sensors=[sat], measurements=[pixel_hdrf])

    def test_pixel_hdrf_references_nonexistent_sensor_raises(self):
        sat = _sat_sensor("sat1")
        pixel_hdrf = PixelHDRFConfig(
            id="phdrf1",
            satellite_sensor_id="nonexistent",
            pixel_indices=[(0, 0)],
        )
        with pytest.raises(ValueError):
            _make_config(sensors=[sat], measurements=[pixel_hdrf])


class TestSatelliteSensorPixelSize:
    def test_pixel_size_1km_100px(self):
        sensor = SatelliteSensor(
            id="s",
            platform=SatellitePlatform.SENTINEL_2A,
            instrument=SatelliteInstrument.MSI,
            band="2",
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            film_resolution=(100, 100),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=1.0,
        )
        px, py = sensor.pixel_size_m
        assert px == pytest.approx(10.0)
        assert py == pytest.approx(10.0)

    def test_pixel_size_returns_tuple(self):
        sensor = _sat_sensor()
        result = sensor.pixel_size_m
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pixel_size_different_width_height(self):
        sensor = SatelliteSensor(
            id="s2",
            platform=SatellitePlatform.SENTINEL_2A,
            instrument=SatelliteInstrument.MSI,
            band="2",
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            film_resolution=(50, 100),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=(1.0, 2.0),
        )
        px, py = sensor.pixel_size_m
        assert px == pytest.approx(20.0)
        assert py == pytest.approx(20.0)
