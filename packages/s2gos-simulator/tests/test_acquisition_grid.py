import numpy as np
import pytest
from s2gos_utils.coordinates import CoordinateSystem

from s2gos_simulator.acquisitions._grid import fit_grid

_LAT, _LON = -23.6, 15.1


def _product(row_heading: float):
    """A 270 x 295 m grid with columns heading 110 deg, whose row 10.3, column 20.7
    is at the target."""
    heading = np.radians([110.0, row_heading])
    east, north = (np.sin(heading) * [270.0, 295.0], np.cos(heading) * [270.0, 295.0])
    rows, cols = np.mgrid[0:30, 0:40]
    x = (cols - 20.7) * east[0] + (rows - 10.3) * east[1]
    y = (cols - 20.7) * north[0] + (rows - 10.3) * north[1]
    lats, lons = CoordinateSystem(_LAT, _LON).scene_to_latlon(x, y)
    return rows, cols, lats, lons


def test_snaps_the_footprint_onto_the_product_pixels():
    grid = fit_grid(*_product(200.0), _LAT, _LON, (3000.0, 3000.0))

    assert grid.rotation == pytest.approx(20.0, abs=1e-6)
    assert grid.pixel_size_m == pytest.approx((270.0, 295.0), abs=1e-3)
    assert grid.shape == (11, 10)
    assert (grid.first_row, grid.first_column) == (6, 16)
    x, y = CoordinateSystem(_LAT, _LON).latlon_to_scene(
        grid.center_lat, grid.center_lon
    )
    column = 270.0 * np.array([np.sin(np.radians(110.0)), np.cos(np.radians(110.0))])
    row = 295.0 * np.array([np.sin(np.radians(200.0)), np.cos(np.radians(200.0))])
    assert (x, y) == pytest.approx((21 - 20.7) * column + (10.5 - 10.3) * row, abs=1e-2)


def test_mirrored_grid_raises():
    with pytest.raises(ValueError, match="not a rotated north-up grid"):
        fit_grid(*_product(20.0), _LAT, _LON, (3000.0, 3000.0))
