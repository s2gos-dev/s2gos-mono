import numpy as np
import pytest
import xarray as xr

from s2gos_simulator.config import GroundInstrumentType, GroundSensor
from s2gos_simulator.config.sensors import PostProcessingOptions
from s2gos_simulator.config.viewing import LookAtViewing
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
