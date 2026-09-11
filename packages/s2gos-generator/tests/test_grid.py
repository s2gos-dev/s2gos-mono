import numpy as np
import pytest
from shapely.geometry import box

from s2gos_generator.core.grid import SceneGrid


@pytest.mark.parametrize(
    "aoi_size_m, resolution_m",
    [
        (900.0, 30.0),  # divides exactly
        (905.0, 30.0),  # one cell of overshoot
        (14_000.0, 10.0),
        (14_000.0, 12.0),
    ],
)
def test_covering_keeps_the_resolution_and_bends_the_extent(aoi_size_m, resolution_m):
    """The cell size is exact and the extent gives, never the other way round."""
    grid = SceneGrid.covering(aoi_size_m, resolution_m)

    assert grid.resolution_m == pytest.approx(resolution_m)
    assert grid.size_m >= aoi_size_m
    assert grid.size_m < aoi_size_m + resolution_m  # tightest such extent


def test_nodes_and_cell_centres_are_the_two_registrations():
    """Elevation samples corners, land cover samples the cells between them."""
    grid = SceneGrid.covering(900.0, 30.0)
    nodes = grid.nodes()
    centres = grid.cell_centres()

    assert len(nodes) == grid.n + 1
    assert nodes[0] == pytest.approx(-grid.half_size_m)
    assert nodes[-1] == pytest.approx(grid.half_size_m)
    np.testing.assert_allclose(np.diff(nodes), grid.resolution_m)

    assert len(centres) == grid.n
    np.testing.assert_allclose(centres, (nodes[:-1] + nodes[1:]) / 2.0)


@pytest.mark.parametrize(
    "resolution_m, margin_m",
    [
        (10.0, 0.0),  # 14 km divides by 10 m, so the grid is the AOI
        (12.0, 2.0),  # 1167 cells of 12 m reach 14004 m
    ],
)
def test_uv_window_places_the_aoi_inside_the_grid(resolution_m, margin_m):
    """Where a mesh spanning the AOI lands in a texture that reaches past it."""
    aoi_size_m = 14_000.0
    grid = SceneGrid.covering(aoi_size_m, resolution_m)

    assert grid.size_m - aoi_size_m == pytest.approx(2 * margin_m)

    u0, v0, u1, v1 = grid.uv_window(aoi_size_m)
    expected = margin_m / grid.size_m
    assert (u0, v0) == pytest.approx((expected, expected))
    assert (u1, v1) == pytest.approx((1.0 - expected, 1.0 - expected))


def test_rasterize_returns_scene_row_order():
    """Row 0 is the southernmost. rasterio emits the opposite, so this flips once."""
    grid = SceneGrid(4.0, 4)

    burned = grid.rasterize([(box(-2.0, 0.0, 2.0, 2.0), 1)])

    assert burned.shape == (4, 4)
    assert (burned[2:] == 1).all()
    assert (burned[:2] == 0).all()
