"""Tests for feature-driven terrain flattening (processors/terrain_mesh/terraforming.py)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from s2gos_generator.processors.terrain_mesh import (
    GradientFilter,
    WaterFlattenOperation,
    WayFlattenOperation,
    apply_way_flatten_batch,
    compute_gradient,
)

# 10x10 DEM spanning 0..90 m in both axes (10 m spacing).
_X = np.linspace(0.0, 90.0, 10)
_Y = np.linspace(0.0, 90.0, 10)
_FLAT = np.ones((10, 10), dtype=float)
_SLOPE = 0.05
_RAMP = _SLOPE * np.broadcast_to(_X, (10, 10))  # rises 0.05 m/m along x

# A centerline crossing the DEM interior.
_CENTERLINE = LineString([(10.0, 50.0), (80.0, 50.0)])


class TestComputeGradient:
    def test_flat_is_zero_ramp_is_slope(self):
        assert np.allclose(compute_gradient(_FLAT, _X, _Y), 0.0)
        assert np.allclose(compute_gradient(_RAMP, _X, _Y), _SLOPE)


class TestGradientFilter:
    def test_flat_terrain_filtered_out(self):
        gf = GradientFilter(_FLAT, _X, _Y)
        assert gf.exceeds_threshold(_CENTERLINE, 0.02) is False
        assert gf.build_operations([_CENTERLINE], [3.0], 10.0, threshold=0.02) == []

    def test_sloped_terrain_kept(self):
        gf = GradientFilter(_RAMP, _X, _Y)
        assert gf.exceeds_threshold(_CENTERLINE, 0.02) is True
        ops = gf.build_operations([_CENTERLINE], [3.0], 10.0, threshold=0.02)
        assert len(ops) == 1
        assert isinstance(ops[0], WayFlattenOperation)

    def test_zero_threshold_keeps_everything(self):
        gf = GradientFilter(_FLAT, _X, _Y)  # flat would otherwise be filtered
        ops = gf.build_operations([_CENTERLINE], [3.0], 10.0, threshold=0.0)
        assert len(ops) == 1

    def test_thin_way_skip_excludes_narrow_segments(self):
        gf = GradientFilter(_RAMP, _X, _Y)
        # half_width 1.0 -> width 2.0, below the 5 m skip threshold -> dropped.
        narrow = gf.build_operations(
            [_CENTERLINE], [1.0], 10.0, threshold=0.0, thin_way_skip_m=5.0
        )
        wide = gf.build_operations(
            [_CENTERLINE], [5.0], 10.0, threshold=0.0, thin_way_skip_m=5.0
        )
        assert narrow == []
        assert len(wide) == 1


def _flat_to_zero(xy):
    """Elevation sampler: reference elevation is 0 everywhere."""
    return np.zeros(len(xy))


class TestWayFlatten:
    def test_alpha_blend_full_transition_and_outside(self):
        # Vertical centerline at x=0; distance from it is |x|. half_width 5, buffer 10.
        op = WayFlattenOperation(
            LineString([(0.0, -1000.0), (0.0, 1000.0)]), half_width=5.0, buffer_m=10.0
        )
        # x=0 and x=3 are within half_width (alpha 1 -> snapped to ref 0);
        # x=10 is mid-transition (alpha 0.5 -> blends to 50); x=20 is outside (unchanged).
        verts = np.array(
            [
                [0.0, 0.0, 100.0],
                [3.0, 0.0, 100.0],
                [10.0, 0.0, 100.0],
                [20.0, 0.0, 100.0],
            ]
        )
        out = op.apply(verts.copy(), _flat_to_zero)

        assert out[0, 2] == pytest.approx(0.0)
        assert out[1, 2] == pytest.approx(0.0)
        assert out[2, 2] == pytest.approx(50.0)  # alpha = 1 - (10-5)/10 = 0.5
        assert out[3, 2] == pytest.approx(100.0)  # beyond influence zone

    def test_endpoint_adjacent_vertex_within_half_width_is_flattened(self):
        # Regression test: the influence-zone buffer used to pre-select
        # candidate vertices must be a superset of the actual per-vertex
        # distance-to-centerline test everywhere, including near the
        # centerline's own endpoints. A "flat" cap buffer excludes points
        # past the endpoint's perpendicular plane even when they're well
        # within half_width by straight-line (radial) distance -- this
        # vertex sits 3m past the centerline's end (y=100) but only
        # sqrt(3^2+3^2)~=4.24m from the endpoint, inside half_width=5.
        op = WayFlattenOperation(
            LineString([(0.0, 0.0), (0.0, 100.0)]), half_width=5.0, buffer_m=0.0
        )
        verts = np.array([[3.0, 103.0, 100.0]])
        out = op.apply(verts.copy(), _flat_to_zero)
        assert out[0, 2] == pytest.approx(0.0)

    def test_batch_matches_single_apply(self):
        op = WayFlattenOperation(
            LineString([(0.0, -1000.0), (0.0, 1000.0)]), half_width=5.0, buffer_m=10.0
        )
        verts = np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0], [20.0, 0.0, 100.0]])

        single = op.apply(verts.copy(), _flat_to_zero)
        batched = apply_way_flatten_batch(verts.copy(), [op], _flat_to_zero)
        assert np.allclose(single, batched)


def _elevation_at(value):
    """Elevation sampler: reference elevation is a constant everywhere."""

    def _fn(xy):
        return np.full(len(xy), value)

    return _fn


class TestWaterFlatten:
    _SQUARE = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    def test_alpha_blend_full_transition_and_outside(self):
        # Interior point, boundary point, mid-transition, and beyond buffer_m.
        op = WaterFlattenOperation(
            self._SQUARE, buffer_m=10.0, anchor_xy=(5.0, 5.0), reference_z=2.0
        )
        verts = np.array(
            [
                [5.0, 5.0, 100.0],  # inside -> alpha 1
                [12.0, 5.0, 100.0],  # 2 m outside boundary -> alpha 0.8
                [15.0, 5.0, 100.0],  # 5 m outside -> alpha 0.5
                [25.0, 5.0, 100.0],  # beyond buffer_m -> unchanged
            ]
        )
        out = op.apply(verts.copy(), _elevation_at(2.0))

        assert out[0, 2] == pytest.approx(2.0)
        assert out[1, 2] == pytest.approx(0.8 * 2.0 + 0.2 * 100.0)
        assert out[2, 2] == pytest.approx(0.5 * 2.0 + 0.5 * 100.0)
        assert out[3, 2] == pytest.approx(100.0)

    def test_batch_matches_single_apply(self):
        op = WaterFlattenOperation(
            self._SQUARE, buffer_m=10.0, anchor_xy=(5.0, 5.0), reference_z=2.0
        )
        verts = np.array([[5.0, 5.0, 100.0], [15.0, 5.0, 100.0], [25.0, 5.0, 100.0]])

        single = op.apply(verts.copy(), _elevation_at(2.0))
        batched = apply_way_flatten_batch(verts.copy(), [op], _elevation_at(2.0))
        assert np.allclose(single, batched)

    def test_mixed_road_and_water_operations_batch_together(self):
        # Confirms apply_way_flatten_batch is polymorphic across operation
        # types -- both apply independently to their own influence zones.
        road_op = WayFlattenOperation(
            LineString([(-1000.0, 0.0), (1000.0, 0.0)]), half_width=2.0, buffer_m=0.0
        )
        water_op = WaterFlattenOperation(
            self._SQUARE, buffer_m=0.0, anchor_xy=(5.0, 5.0), reference_z=2.0
        )
        verts = np.array(
            [
                [0.0, 0.0, 100.0],  # on the road centerline
                [5.0, 5.0, 100.0],  # inside the water square
                [500.0, 500.0, 100.0],  # inside neither
            ]
        )
        out = apply_way_flatten_batch(
            verts.copy(), [road_op, water_op], _elevation_at(2.0)
        )
        assert out[0, 2] == pytest.approx(2.0)
        assert out[1, 2] == pytest.approx(2.0)
        assert out[2, 2] == pytest.approx(100.0)
