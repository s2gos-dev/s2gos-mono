"""Tests for the shared exclusion keep-mask."""

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from s2gos_generator.processors.exclusion import ResolvedExclusionZone, exclusion_mask


def _zone(geometry, excludes=("vegetation", "buildings")):
    return ResolvedExclusionZone("test", geometry, frozenset(excludes))


_SQUARE = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])


class TestExclusionMask:
    def test_empty_geoms_returns_empty_mask(self):
        mask = exclusion_mask([], [_zone(_SQUARE)])
        assert mask.shape == (0,)

    def test_no_zones_keeps_all(self):
        pts = shapely.points([1, 2], [1, 2])
        assert exclusion_mask(pts, []).all()

    def test_points_inside_and_on_boundary_excluded(self):
        pts = shapely.points([10.0, 50.0, 100.0], [10.0, 25.0, 100.0])
        mask = exclusion_mask(pts, [_zone(_SQUARE)])
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_footprint_inside_removed(self):
        inside = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
        assert not exclusion_mask([inside], [_zone(_SQUARE)])[0]

    def test_footprint_straddling_boundary_removed(self):
        straddling = Polygon([(45, 45), (55, 45), (55, 55), (45, 55)])
        assert not exclusion_mask([straddling], [_zone(_SQUARE)])[0]

    def test_footprint_outside_kept(self):
        outside = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
        assert exclusion_mask([outside], [_zone(_SQUARE)])[0]

    def test_multiple_zones_union(self):
        zones = [_zone(_SQUARE), _zone(Point(200, 200).buffer(5))]
        pts = shapely.points([10.0, 200.0, 300.0], [10.0, 200.0, 300.0])
        np.testing.assert_array_equal(exclusion_mask(pts, zones), [False, False, True])
