import math

import numpy as np
import pytest

from s2gos_simulator.backends.eradiate.eradiate_translator import EradiateTranslator
from s2gos_simulator.backends.eradiate.geometry_utils import (
    GeometryUtils,
    sanitize_sensor_id,
)
from s2gos_simulator.config import (
    BRFConfig,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
    SpectralResponse,
)
from s2gos_simulator.config.viewing import (
    AngularFromOriginViewing,
    AngularViewing,
    DistantViewing,
    LookAtViewing,
    RectangleTarget,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sentinel_sensor(**kwargs):
    defaults = dict(
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
    defaults.update(kwargs)
    return SatelliteSensor(**defaults)


def _make_translator_for_sensor(sensor, illumination=None):
    illum = illumination or DirectionalIllumination(zenith=30.0, azimuth=135.0)
    config = SimulationConfig(
        name="test", illumination=illum, sensors=[sensor], measurements=[]
    )
    return EradiateTranslator(config, GeometryUtils())


class TestSatelliteSensorTranslation:
    def test_translate_satellite_sensor_structure(self, translator, minimal_scene):
        sensor = _make_sentinel_sensor(
            id="sentinel.2a.test",  # specific ID to test sanitization
            viewing=AngularViewing(zenith=15.0, azimuth=90.0),
            film_resolution=(128, 64),
        )

        result = translator.translate_satellite_sensor(sensor, minimal_scene)

        assert result["type"] == "mpdistant"
        assert result["id"] == "sentinel_2a_test"

        assert result["construct"] == "from_angles"
        assert result["angles"] == [15.0, 90.0]

        assert result["film_resolution"] == (128, 64)

        target = result["target"]
        assert target["type"] == "rectangle"
        assert {"xmin", "xmax", "ymin", "ymax"}.issubset(target.keys())

    def test_sensor_for_reference_only(
        self, directional_illumination, minimal_scene, scene_dir
    ):
        sensor = _make_sentinel_sensor(for_reference_only=True)

        config = SimulationConfig(
            name="test",
            illumination=directional_illumination,
            sensors=[sensor],
            measurements=[],
        )

        t = EradiateTranslator(config, GeometryUtils())
        measures = t.translate_sensors(minimal_scene, scene_dir)

        assert measures == []

    def test_sensor_selection(self, directional_illumination, minimal_scene, scene_dir):
        include_sensor = _make_sentinel_sensor(id="include")
        exclude_sensor = _make_sentinel_sensor(id="exclude")

        config = SimulationConfig(
            name="test",
            illumination=directional_illumination,
            sensors=[include_sensor, exclude_sensor],
            measurements=[],
        )

        t = EradiateTranslator(config, GeometryUtils())
        measures = t.translate_sensors(minimal_scene, scene_dir, sensor_ids="include")

        assert len(measures) == 1
        assert measures[0]["id"] == "include"

    def test_translate_satellite_sensor_scene_missing_location_raises(
        self, translator, minimal_scene
    ):
        sensor = _make_sentinel_sensor()

        minimal_scene.location = {"center_lon": -64.5}
        with pytest.raises(ValueError, match="center_lat"):
            translator.translate_satellite_sensor(sensor, minimal_scene)

        minimal_scene.location = {"center_lat": -42.0}
        with pytest.raises(ValueError, match="center_lon"):
            translator.translate_satellite_sensor(sensor, minimal_scene)

    def test_satellite_sensor_pixel_size_calculation(self):
        s1 = _make_sentinel_sensor(film_resolution=(100, 100), target_size_km=1.0)
        px1, py1 = s1.pixel_size_m
        assert px1 == pytest.approx(10.0)
        assert py1 == pytest.approx(10.0)

        s2 = _make_sentinel_sensor(
            film_resolution=(200, 100), target_size_km=(2.0, 1.0)
        )
        px2, py2 = s2.pixel_size_m
        assert px2 == pytest.approx(10.0)
        assert py2 == pytest.approx(10.0)


class TestDistantViewingToMdistant:
    def _translate(self, translator, minimal_scene, scene_dir, viewing, sensor_id="ds"):
        return translator.translate_distant_viewing_to_mdistant(
            viewing=viewing,
            sensor_id=sensor_id,
            srf=None,
            spp=64,
            scene_description=minimal_scene,
            scene_dir=scene_dir,
        )

    def test_mdistant_nadir_defaults(self, translator, minimal_scene, scene_dir):
        viewing = DistantViewing(
            direction=[0, 0, 1],
            target=None,
            terrain_relative_height=False,
            ray_offset=None,
        )

        result = self._translate(translator, minimal_scene, scene_dir, viewing)

        assert result["construct"] == "from_angles"
        assert result["angles"] == pytest.approx([0.0, 0.0], abs=1e-10)
        assert result["target"] == pytest.approx([0.0, 0.0, 0.0])
        assert "ray_offset" not in result

    def test_mdistant_off_nadir_with_options(
        self, translator, minimal_scene, scene_dir
    ):
        sin45 = math.sin(math.radians(45))
        cos45 = math.cos(math.radians(45))

        viewing = DistantViewing(
            direction=[sin45, 0.0, cos45], terrain_relative_height=False, ray_offset=0.5
        )

        result = self._translate(translator, minimal_scene, scene_dir, viewing)

        assert result["angles"] == pytest.approx([45.0, 0.0], abs=1e-4)
        assert result["ray_offset"] == pytest.approx(0.5)

    def test_mdistant_rectangle_target(self, translator, minimal_scene, scene_dir):
        target = RectangleTarget(xmin=-50, xmax=50, ymin=-30, ymax=30, z=1.0)
        viewing = DistantViewing(target=target, terrain_relative_height=False)

        result = self._translate(translator, minimal_scene, scene_dir, viewing)

        # Assert the whole target dictionary structure
        assert result["target"] == {
            "type": "rectangle",
            "xmin": -50,
            "xmax": 50,
            "ymin": -30,
            "ymax": 30,
            "z": 1.0,
        }

    def test_mdistant_point_target(self, translator, minimal_scene, scene_dir):
        viewing = DistantViewing(
            target=[10.0, 20.0, 0.5], terrain_relative_height=False
        )

        result = self._translate(translator, minimal_scene, scene_dir, viewing)

        assert result["target"] == pytest.approx([10.0, 20.0, 0.5])


class TestGroundSensorTranslation:
    def _translate_ground(self, translator, sensor, minimal_scene, scene_dir):
        return translator.translate_ground_sensor(sensor, minimal_scene, scene_dir, {})

    def test_ground_radiancemeter_lookat(self, translator, minimal_scene, scene_dir):
        sensor = GroundSensor(
            id="rad",
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=LookAtViewing(
                origin=[1, 2, 3], target=[5, 5, 0], terrain_relative_height=False
            ),
        )
        result = self._translate_ground(translator, sensor, minimal_scene, scene_dir)

        assert result["id"] == "rad"
        assert result["type"] == "radiancemeter"
        assert result["origin"] == pytest.approx([1, 2, 3])
        assert result["target"] == pytest.approx([5, 5, 0])

    def test_ground_perspective_camera(self, translator, minimal_scene, scene_dir):
        sensor = GroundSensor(
            id="cam",
            instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            viewing=LookAtViewing(
                origin=[0, 0, 10],
                target=[0, 0, 0],
                up=[0, 1, 0],
                terrain_relative_height=False,
            ),
            fov=70.0,
            resolution=[512, 512],
        )
        result = self._translate_ground(translator, sensor, minimal_scene, scene_dir)

        assert result["id"] == "cam"
        assert result["type"] == "perspective"
        assert result["origin"] == pytest.approx([0, 0, 10])
        assert result["target"] == pytest.approx([0, 0, 0])
        assert result["up"] == pytest.approx([0, 1, 0])
        assert result["fov"] == pytest.approx(70.0)
        assert result["film_resolution"] == [512, 512]

    def test_ground_hypstar(self, translator, minimal_scene, scene_dir):
        sensor = GroundSensor(
            id="hyp",
            instrument=GroundInstrumentType.HYPSTAR,
            viewing=LookAtViewing(
                origin=[0, 0, 1],
                target=[0, 0, 0],
                up=[0, 1, 0],
                terrain_relative_height=False,
            ),
        )
        result = self._translate_ground(translator, sensor, minimal_scene, scene_dir)

        assert result["id"] == "hyp"
        assert result["type"] == "perspective"
        assert result["srf"] == {"type": "uniform", "wmin": 380.0, "wmax": 1680.0}
        assert result["origin"] == pytest.approx([0, 0, 1])
        assert result["target"] == pytest.approx([0, 0, 0])
        assert result["up"] == pytest.approx([0, 1, 0])
        assert result["fov"] == pytest.approx(5.0)
        assert result["film_resolution"] == [32, 32]


class TestIrradianceMeasureCreation:
    def _make_irradiance_config(self, srf=None):
        loc = HemisphericalMeasurementLocation(
            target_x=0.0, target_y=0.0, target_z=1.0, terrain_relative_height=True
        )

        if srf is not None:
            loc = loc.model_copy(update={"srf": srf})
        return IrradianceConfig(id="my_irr", location=loc)

    def test_irradiance_measure_defaults(self, translator):
        irr = self._make_irradiance_config()
        result = translator.create_irradiance_measure(irr, disk_coords=(5.0, -3.0, 1.2))

        assert result["type"] == "hdistant"
        assert result["direction"] == [0, 0, 1]
        assert result["ray_offset"] == pytest.approx(0.1)
        assert result["target"] == pytest.approx([5.0, -3.0, 1.2])
        assert result["srf"] == {"type": "uniform", "wmin": 380.0, "wmax": 1680.0}

    def test_irradiance_measure_custom_srf(self, translator):
        custom_srf = SpectralResponse(type="uniform", wmin=400.0, wmax=900.0)
        irr = self._make_irradiance_config(srf=custom_srf)
        result = translator.create_irradiance_measure(irr, disk_coords=(0.0, 0.0, 0.0))
        assert result["srf"]["wmin"] == pytest.approx(400.0)
        assert result["srf"]["wmax"] == pytest.approx(900.0)


class TestSensorAutoGeneration:
    def _make_hdrf_config(self, srf=None):
        return HDRFConfig(
            id="my_hdrf",
            instrument="radiancemeter",
            viewing=LookAtViewing(
                origin=[0.0, 0.0, 1.0],
                target=[0.0, 0.0, 0.0],
                terrain_relative_height=False,
            ),
            location=HemisphericalMeasurementLocation(
                target_x=0.0, target_y=0.0, target_z=0.0, terrain_relative_height=False
            ),
            reference_height_offset_m=0.1,
            srf=srf,
        )

    def _make_brf_config(self, srf=None):
        return BRFConfig(
            id="my_brf",
            viewing=AngularFromOriginViewing(
                origin=[0.0, 0.0, 1.0], zenith=0.0, azimuth=0.0
            ),
            srf=srf,
        )

    def test_generate_radiance_sensor_for_hdrf(self, translator):
        srf = SpectralResponse(type="uniform", wmin=400.0, wmax=900.0)
        hdrf = self._make_hdrf_config(srf=srf)
        sensor = translator.generate_radiance_sensor_for_hdrf(hdrf)

        assert sensor.id == "hdrf_radiance_my_hdrf"
        assert sensor.instrument == GroundInstrumentType.RADIANCEMETER
        assert sensor.srf == srf

    def test_generate_irradiance_for_hdrf(self, translator):
        srf = SpectralResponse(type="uniform", wmin=400.0, wmax=900.0)
        hdrf = self._make_hdrf_config(srf=srf)
        irr = translator.generate_irradiance_measurement_for_hdrf(hdrf)
        assert irr.id == "hdrf_irradiance_my_hdrf"
        assert irr.location.srf == srf

    def _make_hcrf_config(self, srf=None):
        return HCRFConfig(
            id="my_hcrf",
            viewing=LookAtViewing(
                origin=[0.0, 0.0, 1.0],
                target=[0.0, 0.0, 0.0],
                terrain_relative_height=False,
            ),
            fov=70.0,
            film_resolution=(512, 512),
            platform_type="ground",
            location=HemisphericalMeasurementLocation(
                target_x=0.0, target_y=0.0, target_z=0.0, terrain_relative_height=False
            ),
            reference_height_offset_m=0.1,
            srf=srf,
        )

    def test_generate_camera_sensor_for_hcrf(self, translator):
        hcrf_config = self._make_hcrf_config()
        sensor = translator.generate_camera_sensor_for_hcrf(hcrf_config)
        assert sensor.id == "hcrf_camera_my_hcrf"
        assert sensor.instrument == GroundInstrumentType.PERSPECTIVE_CAMERA

    def test_generate_irradiance_measurement_for_hcrf(self, translator):
        srf = SpectralResponse(type="uniform", wmin=400.0, wmax=900.0)
        hcrf_config = self._make_hcrf_config(srf=srf)
        irr = translator.generate_irradiance_measurement_for_hcrf(hcrf_config)
        assert irr.id == "hcrf_irradiance_my_hcrf"
        assert irr.location.srf == srf

    def test_generate_radiance_sensor_for_brf(self, translator):
        brf = self._make_brf_config()
        sensor = translator.generate_radiance_sensor_for_brf(brf)
        assert sensor.id == "brf_radiance_my_brf"
        assert sensor.instrument == GroundInstrumentType.RADIANCEMETER


class TestGeometryUtils:
    def test_sanitize_sensor_id(self):
        assert sanitize_sensor_id("hypstar_wl500.3nm") == "hypstar_wl500_3nm"
        assert sanitize_sensor_id("a.b.c") == "a_b_c"
        assert sanitize_sensor_id("no_dots_here") == "no_dots_here"
        assert sanitize_sensor_id(None) is None
        assert sanitize_sensor_id("") == ""

    def test_calculate_target_from_up(self):
        geo = GeometryUtils()
        view = AngularFromOriginViewing(
            origin=[0.0, 0.0, 10.0], zenith=0.0, azimuth=0.0
        )
        target, direction = geo.calculate_target_from_angles(view)
        assert target == pytest.approx([0.0, 0.0, 1010.0], abs=1e-6)
        assert direction == pytest.approx([0.0, 0.0, 1.0], abs=1e-10)

    def test_calculate_target_from_down(self):
        geo = GeometryUtils()
        view = AngularFromOriginViewing(
            origin=[0.0, 0.0, 10.0], zenith=180.0, azimuth=0.0
        )
        target, direction = geo.calculate_target_from_angles(view)
        assert target == pytest.approx([0.0, 0.0, -990.0], abs=1e-6)
        assert direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-10)

    def test_calculate_target_from_angles_oblique(self):
        geo = GeometryUtils()
        view = AngularFromOriginViewing(
            origin=[0.0, 0.0, 0.0], zenith=45.0, azimuth=0.0
        )
        target, direction = geo.calculate_target_from_angles(view)
        assert target[0] > 0 and target[2] > 0
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-10)


class TestTerrainRelativeAdjustment:
    def test_ground_sensor_terrain_relative_height_adjusts_z(
        self, translator, minimal_scene, scene_dir, monkeypatch
    ):
        monkeypatch.setattr(
            translator.geometry_utils,
            "query_terrain_elevation",
            lambda *args, **kwargs: 5.0,
        )
        sensor = GroundSensor(
            id="rad",
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=LookAtViewing(
                origin=[0.0, 0.0, 1.0],
                target=[0.0, 0.0, 0.0],
                terrain_relative_height=True,
            ),
        )
        result = translator.translate_ground_sensor(
            sensor, minimal_scene, scene_dir, {}
        )

        assert result["origin"][2] == pytest.approx(6.0)
        assert result["target"][2] == pytest.approx(5.0)

    def test_mdistant_point_target_terrain_adjusted(
        self, translator, minimal_scene, scene_dir, monkeypatch
    ):
        monkeypatch.setattr(
            translator.geometry_utils,
            "query_terrain_elevation",
            lambda *args, **kwargs: 10.0,
        )
        viewing = DistantViewing(target=[0.0, 0.0, 0.5], terrain_relative_height=True)
        result = translator.translate_distant_viewing_to_mdistant(
            viewing=viewing,
            sensor_id="test",
            srf=None,
            spp=64,
            scene_description=minimal_scene,
            scene_dir=scene_dir,
        )

        assert result["target"][2] == pytest.approx(10.5)
