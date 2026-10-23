import pytest

from s2gos_simulator.config import (
    Band,
    BRFConfig,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    MultispectralInstrument,
    PixelHDRFConfig,
    SatelliteSensor,
    SimulationConfig,
    SpectralResponse,
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
        instrument="sentinel_2a-msi",
        bands=["2"],
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
            instrument="sentinel_2a-msi",
            bands=["2"],
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            film_resolution=(100, 100),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=1.0,
        )
        px, py = sensor.pixel_size_m("2")
        assert px == pytest.approx(10.0)
        assert py == pytest.approx(10.0)

    def test_pixel_size_returns_tuple(self):
        sensor = _sat_sensor()
        result = sensor.pixel_size_m("2")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pixel_size_different_width_height(self):
        sensor = SatelliteSensor(
            id="s2",
            instrument="sentinel_2a-msi",
            bands=["2"],
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            film_resolution=(50, 100),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=(1.0, 2.0),
        )
        px, py = sensor.pixel_size_m("2")
        assert px == pytest.approx(20.0)
        assert py == pytest.approx(20.0)


class TestSatelliteSensorNativeResolution:
    def _msi_sensor(self, band, target_size_km):
        return SatelliteSensor(
            instrument="sentinel_2a-msi",
            bands=[band],
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=target_size_km,
        )

    @pytest.mark.parametrize(
        "band, gsd, film",
        [("4", 10, (120, 60)), ("8a", 20, (60, 30)), ("1", 60, (20, 10))],
    )
    def test_defaults_to_band_gsd(self, band, gsd, film):
        sensor = self._msi_sensor(band, (1.2, 0.6))

        assert sensor.band_film_resolution(band) == film
        assert sensor.pixel_size_m(band) == pytest.approx((gsd, gsd))

    def test_footprint_off_band_grid_raises(self):
        with pytest.raises(ValueError, match="band '1'"):
            self._msi_sensor("1", 1.0)

    def test_instrument_without_gsd_requires_film_resolution(self):
        camera = MultispectralInstrument(
            platform="custom", name="cam", bands={"red": Band(srf="red-srf")}
        )
        with pytest.raises(ValueError, match="film_resolution is required"):
            SatelliteSensor(
                instrument=camera,
                viewing=AngularViewing(zenith=0.0, azimuth=0.0),
                target_center_lat=-42.0,
                target_center_lon=-64.5,
                target_size_km=1.0,
            )


class TestSatelliteSensorInstrument:
    def _sensor(self, **kwargs):
        defaults = dict(
            instrument="sentinel_2a-msi",
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            target_center_lat=-42.0,
            target_center_lon=-64.5,
            target_size_km=1.2,
        )
        return SatelliteSensor(**{**defaults, **kwargs})

    def test_defaults_to_every_band_at_native_gsd(self):
        sensor = self._sensor()

        assert len(sensor.bands) == 13
        assert sensor.band_film_resolution("4") == (120, 120)
        assert sensor.band_film_resolution("1") == (20, 20)
        assert sensor.band_srf("8a") == "sentinel_2a-msi-8a"
        assert sensor.band_measure_id("8a") == "sentinel_2a_msi_nadir_8a"

    def test_unknown_instrument_lists_the_available_ones(self):
        with pytest.raises(ValueError, match="sentinel_2a-msi"):
            self._sensor(instrument="sentinel-2a")

    def test_custom_instrument(self):
        blue = SpectralResponse(type="uniform", wmin=450.0, wmax=520.0)
        red = SpectralResponse(type="uniform", wmin=630.0, wmax=690.0)
        camera = MultispectralInstrument(
            platform="my-sat",
            name="cam",
            bands={"blue": Band(srf=blue, gsd_m=10), "red": Band(srf=red, gsd_m=20)},
        )
        sensor = self._sensor(instrument=camera)

        assert sensor.id == "my_sat_cam_nadir"
        assert sensor.band_film_resolution("red") == (60, 60)
        assert sensor.band_srf("blue") == blue

    def test_hyperspectral_instrument_is_one_measure(self):
        sensor = self._sensor(instrument="chime-hsi", target_size_km=0.3)

        assert sensor.id == "chime_hsi_nadir"
        assert sensor.measured_bands == [None]
        assert sensor.band_measure_id(None) == sensor.id
        assert sensor.band_film_resolution(None) == (10, 10)
        assert sensor.srf.type == "gaussian"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"bands": ["13"]}, "not bands of instrument"),
            ({"bands": []}, "non-empty"),
            (
                {
                    "bands": ["2"],
                    "band_viewing": {"3": AngularViewing(zenith=5.0, azimuth=0.0)},
                },
                "band_viewing has bands",
            ),
            ({"instrument": "chime-hsi", "bands": ["full"]}, "takes no bands"),
            ({"bands": ["2"], "target_size_km": 0.0}, "greater than 0"),
            ({"bands": ["2"], "srf": "sentinel_2a-msi-2"}, "Extra inputs"),
            (
                {
                    "instrument": {
                        "kind": "multispectral",
                        "platform": "x",
                        "name": "y",
                        "bands": {},
                    }
                },
                "at least 1",
            ),
        ],
    )
    def test_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._sensor(**kwargs)

    def test_pixel_measurement_on_multiband_sensor_requires_band(self):
        sensor = self._sensor(id="s2", bands=["2", "11"])
        pixel_hdrf = PixelHDRFConfig(
            id="phdrf1", satellite_sensor_id="s2", pixel_indices=[(0, 0)]
        )
        with pytest.raises(ValueError, match="specify one"):
            _make_config(sensors=[sensor], measurements=[pixel_hdrf])
