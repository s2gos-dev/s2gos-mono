import numpy as np
import pytest
import xarray as xr

from s2gos_simulator.config import (
    GroundInstrumentType,
    GroundSensor,
    SpectralResponse,
    WavelengthGrid,
)
from s2gos_simulator.config.sensors import PostProcessingOptions
from s2gos_simulator.config.viewing import AngularFromOriginViewing, LookAtViewing
from s2gos_simulator.processors.sensor_processor import SensorProcessor

FILM = 32


def fisheye_result(center_x=0.5):
    """Uniform unit radiance seen through an inscribed image circle.

    Each pixel holds the fraction of its area inside the circle, as Eradiate's film
    does for samples outside the image circle.
    """
    sub = 16
    u = (np.arange(FILM * sub) + 0.5) / (FILM * sub)
    inside = np.hypot(u[None, :] - center_x, u[:, None] - 0.5) <= 0.5
    coverage = inside.reshape(FILM, sub, FILM, sub).mean(axis=(1, 3))
    return xr.Dataset(
        {"radiance": (("w", "y_index", "x_index"), coverage[None])},
        coords={"w": [550.0], "y_index": np.arange(FILM), "x_index": np.arange(FILM)},
        attrs={
            "film_width": FILM,
            "film_height": FILM,
            "center_x": center_x,
            "center_y": 0.5,
            "image_circle_radius": 0.5,
            "rho_max": 1.0,
        },
    )


def average(dataset):
    sensor = GroundSensor(
        id="fish",
        instrument=GroundInstrumentType.FISHEYE_CAMERA,
        viewing=LookAtViewing(origin=[0, 0, 2], target=[0, 0, 10], up=[0, 1, 0]),
        post_processing=PostProcessingOptions(spatial_averaging=True),
    )
    return SensorProcessor(None).process_sensor_result(dataset, sensor)


def test_fisheye_average_is_exact_despite_partly_lit_rim():
    result = average(fisheye_result())

    assert result["radiance"].item() == pytest.approx(1.0, abs=1e-3)


def test_fisheye_average_rejects_a_clipped_image_circle():
    with pytest.raises(ValueError, match="inside the film"):
        average(fisheye_result(center_x=0.3))


def irradiancemeter_result(radiance, w):
    """Raw radiancemeter result on a reference disk, as Eradiate returns it in ckd."""
    w = np.asarray(w, dtype=float)
    shape = (len(w), 1, 1)
    return xr.Dataset(
        {
            "radiance": (("w", "y_index", "x_index"), np.broadcast_to(radiance, shape)),
            "irradiance": ("w", np.full(len(w), 1361.0 / 1000)),
        },
        coords={
            "w": ("w", w, {"units": "nm"}),
            "bin_wmin": ("w", w - 5.0, {"units": "nm"}),
            "bin_wmax": ("w", w + 5.0, {"units": "nm"}),
            "y_index": [0],
            "x_index": [0],
        },
    )


def irradiancemeter(srf):
    return GroundSensor(
        id="irr",
        instrument=GroundInstrumentType.IRRADIANCEMETER,
        viewing=AngularFromOriginViewing(origin=[0.0, 0.0, 2.0]),
        srf=srf,
    )


def test_irradiancemeter_reports_irradiance_and_band_value():
    raw = irradiancemeter_result(1.0 / np.pi, np.arange(405.0, 700.0, 10.0))
    raw["radiance_srf"] = (("y_index", "x_index"), [[1.0 / np.pi]])

    result = SensorProcessor(None).process_sensor_result(raw, irradiancemeter("par"))

    np.testing.assert_allclose(result["irradiance"], 1.0)
    assert result["band_irradiance"].item() == pytest.approx(300.0, rel=1e-4)
    np.testing.assert_allclose(result["toa_irradiance"], 1.361)
    assert "radiance" not in result


def test_irradiancemeter_gaussian_srf_convolves_the_irradiance():
    srf = SpectralResponse(
        type="gaussian",
        fwhm_nm=10.0,
        output_grid=WavelengthGrid(mode="explicit", wavelengths_nm=[500.0, 600.0]),
    )
    raw = irradiancemeter_result(2.0 / np.pi, np.arange(405.0, 700.0, 10.0))

    result = SensorProcessor(None).process_sensor_result(raw, irradiancemeter(srf))

    assert list(result["irradiance"].w.values) == [500.0, 600.0]
    np.testing.assert_allclose(result["irradiance"], 2.0, rtol=1e-6)
    assert "band_irradiance" not in result
