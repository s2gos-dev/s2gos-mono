import pytest

from s2gos_simulator.backends.eradiate.backend import EradiateBackend
from s2gos_simulator.config import (
    DirectionalIllumination,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
)
from s2gos_simulator.config.measurements import (
    BHRConfig,
    BRFConfig,
    HDRFConfig,
    PixelBHRConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
)
from s2gos_simulator.config.viewing import AngularViewing


def _sat_sensor(sensor_id="sat1"):
    return SatelliteSensor(
        id=sensor_id,
        platform=SatellitePlatform.SENTINEL_2A,
        instrument=SatelliteInstrument.MSI,
        band="2",
        viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        film_resolution=(100, 100),
        target_center_lat=-42.0,
        target_center_lon=-64.5,
        target_size_km=1.0,
    )


def _make_backend(measurements):
    config = SimulationConfig(
        name="test",
        illumination=DirectionalIllumination(zenith=30.0, azimuth=135.0),
        sensors=[_sat_sensor()],
        measurements=measurements,
    )
    return EradiateBackend(config)


class TestExpandPixelMeasurementConfigs:
    """Tests for EradiateBackend._expand_pixel_measurement_configs.

    All tests call _expand_pixel_measurement_configs with
    scene_description=None / scene_dir=None so terrain lookup is skipped and
    target_z falls back to pixel_config.height_offset_m.  No mocking needed.
    """

    def test_pixel_hdrf_replaced_by_expanded_hdrf(self):
        pixel_config = PixelHDRFConfig(
            id="phdrf",
            satellite_sensor_id="sat1",
            pixel_indices=[(0, 0)],
        )
        backend = _make_backend([pixel_config])

        backend._expand_pixel_measurement_configs(
            scene_description=None, scene_dir=None
        )

        measurements = backend.simulation_config.measurements
        assert not any(isinstance(m, PixelHDRFConfig) for m in measurements)
        hdrf_configs = [m for m in measurements if isinstance(m, HDRFConfig)]
        assert len(hdrf_configs) == 1
        assert hdrf_configs[0].id == "phdrf_r0_c0"

    def test_pixel_brf_replaced_by_expanded_brf(self):
        pixel_config = PixelBRFConfig(
            id="pbrf",
            satellite_sensor_id="sat1",
            pixel_indices=[(0, 1)],
        )
        backend = _make_backend([pixel_config])

        backend._expand_pixel_measurement_configs(
            scene_description=None, scene_dir=None
        )

        measurements = backend.simulation_config.measurements
        assert not any(isinstance(m, PixelBRFConfig) for m in measurements)
        brf_configs = [m for m in measurements if isinstance(m, BRFConfig)]
        assert len(brf_configs) == 1
        assert brf_configs[0].id == "pbrf_r0_c1"

    def test_pixel_bhr_replaced_by_expanded_bhr(self):
        pixel_config = PixelBHRConfig(
            id="pbhr",
            satellite_sensor_id="sat1",
            pixel_indices=[(1, 1)],
            reference_height_offset_m=0.2,
        )
        backend = _make_backend([pixel_config])

        backend._expand_pixel_measurement_configs(
            scene_description=None, scene_dir=None
        )

        measurements = backend.simulation_config.measurements
        assert not any(isinstance(m, PixelBHRConfig) for m in measurements)
        bhr_configs = [m for m in measurements if isinstance(m, BHRConfig)]
        assert len(bhr_configs) == 1
        assert bhr_configs[0].id == "pbhr_r1_c1"
        assert bhr_configs[0].reference_height_offset_m == pytest.approx(0.2)

    def test_multiple_pixels_produce_multiple_measurements(self):
        pixel_config = PixelBRFConfig(
            id="pbrf",
            satellite_sensor_id="sat1",
            pixel_indices=[(0, 0), (1, 2)],
        )
        backend = _make_backend([pixel_config])

        backend._expand_pixel_measurement_configs(
            scene_description=None, scene_dir=None
        )

        measurements = backend.simulation_config.measurements
        assert not any(isinstance(m, PixelBRFConfig) for m in measurements)
        brf_configs = [m for m in measurements if isinstance(m, BRFConfig)]
        assert len(brf_configs) == 2
        ids = {m.id for m in brf_configs}
        assert ids == {"pbrf_r0_c0", "pbrf_r1_c2"}

    def test_pixel_footprint_and_height_offset(self):
        pixel_config = PixelBRFConfig(
            id="pbrf",
            satellite_sensor_id="sat1",
            pixel_indices=[(0, 0)],
            height_offset_m=0.5,
        )
        backend = _make_backend([pixel_config])

        backend._expand_pixel_measurement_configs(
            scene_description=None, scene_dir=None
        )

        measurements = backend.simulation_config.measurements
        brf_configs = [m for m in measurements if isinstance(m, BRFConfig)]
        assert len(brf_configs) == 1
        target = brf_configs[0].viewing.target
        assert target.xmax - target.xmin == pytest.approx(10.0)
        assert target.ymax - target.ymin == pytest.approx(10.0)
        assert target.xmin == pytest.approx(-500.0)
        assert target.ymin == pytest.approx(490.0)
        assert target.xmax == pytest.approx(-490.0)
        assert target.ymax == pytest.approx(500.0)
        assert target.z == pytest.approx(0.5)
