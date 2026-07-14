import math
import random
import types

from s2gos_generator.processors.vegetation import (
    _apply_spacing_filter_optimized,
    _batch_elevation_lookup,
    _calculate_max_instances_per_pixel,
    _filter_by_exclusion_zones,
    _filter_by_roads,
    _generate_pixel_vegetation_positions,
)


def _make_species(name, scale_min, scale_max):
    sp = types.SimpleNamespace(
        name=name,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    sp.get_asset_paths_and_weights = lambda: (["tree.xml"], [1.0])
    return sp


def _make_veg_config(min_spacing):
    return types.SimpleNamespace(min_spacing=min_spacing)


def _pos(x, y):
    return {"x": x, "y": y}


def _instance(x, y):
    return {"position": [x, y, 0.0], "species": "oak", "asset_xml": "tree.xml"}


class TestCalculateMaxInstancesPerPixel:
    def test_zero_min_spacing_returns_50(self):
        assert _calculate_max_instances_per_pixel(30.0, 30.0, 0) == 100

    def test_specific_value_30x30_at_5m(self):
        result = _calculate_max_instances_per_pixel(30.0, 30.0, 5.0)
        assert result == 27

    def test_larger_pixel_fits_more(self):
        small = _calculate_max_instances_per_pixel(30.0, 30.0, 5.0)
        large = _calculate_max_instances_per_pixel(60.0, 60.0, 5.0)
        assert large > small

    def test_larger_spacing_fits_fewer(self):
        fine = _calculate_max_instances_per_pixel(30.0, 30.0, 5.0)
        coarse = _calculate_max_instances_per_pixel(30.0, 30.0, 10.0)
        assert coarse < fine


class TestApplySpacingFilter:
    def test_empty_returns_empty(self):
        result = _apply_spacing_filter_optimized([], 5.0)
        assert result == []

    def test_single_always_kept(self):
        positions = [_pos(0.0, 0.0)]
        result = _apply_spacing_filter_optimized(positions, 5.0)
        assert len(result) == 1

    def test_zero_spacing_returns_all(self):
        positions = [_pos(0.0, 0.0), _pos(0.1, 0.0), _pos(0.0, 0.1)]
        result = _apply_spacing_filter_optimized(positions, 0.0)
        assert len(result) == len(positions)

    def test_first_position_always_kept(self):
        positions = [_pos(0.0, 0.0), _pos(0.0, 0.0)]
        result = _apply_spacing_filter_optimized(positions, 5.0)
        assert result[0] is positions[0]

    def test_invariant_all_output_pairs_satisfy_spacing(self):
        random.seed(42)
        min_spacing = 5.0
        positions = [
            _pos(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(50)
        ]
        result = _apply_spacing_filter_optimized(positions, min_spacing)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dx = result[i]["x"] - result[j]["x"]
                dy = result[i]["y"] - result[j]["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                assert dist >= min_spacing, (
                    f"Pair ({i},{j}) violates min_spacing: {dist:.4f} < {min_spacing}"
                )

    def test_positions_at_exact_spacing_both_kept(self):
        min_spacing = 5.0
        positions = [_pos(0.0, 0.0), _pos(min_spacing, 0.0)]
        result = _apply_spacing_filter_optimized(positions, min_spacing)
        assert len(result) == 2


class TestGeneratePixelVegetationPositions:
    def test_positions_within_pixel_bounds(self):
        species = _make_species("oak", 10.0, 35.0)
        veg_config = _make_veg_config(0.0)
        center_x, center_y = 100.0, 200.0
        resolution = 30.0
        result = _generate_pixel_vegetation_positions(
            center_x, center_y, resolution, resolution, 10, species, veg_config
        )
        half = resolution / 2.0
        for pos in result:
            assert center_x - half <= pos["x"] <= center_x + half
            assert center_y - half <= pos["y"] <= center_y + half

    def test_within_pixel_spacing_respected(self):
        species = _make_species("oak", 10.0, 35.0)
        min_spacing = 5.0
        veg_config = _make_veg_config(min_spacing)
        result = _generate_pixel_vegetation_positions(
            0.0, 0.0, 60.0, 60.0, 20, species, veg_config
        )
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dx = result[i]["x"] - result[j]["x"]
                dy = result[i]["y"] - result[j]["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                assert dist >= min_spacing, (
                    f"Pair ({i},{j}) violates min_spacing: {dist:.4f} < {min_spacing}"
                )

    def test_at_most_n_instances(self):
        species = _make_species("oak", 10.0, 35.0)
        veg_config = _make_veg_config(0.0)
        n = 7
        result = _generate_pixel_vegetation_positions(
            0.0, 0.0, 30.0, 30.0, n, species, veg_config
        )
        assert len(result) <= n

    def test_zero_n_instances_returns_empty(self):
        species = _make_species("oak", 10.0, 35.0)
        veg_config = _make_veg_config(0.0)
        result = _generate_pixel_vegetation_positions(
            0.0, 0.0, 30.0, 30.0, 0, species, veg_config
        )
        assert result == []


class TestFilterByExclusionZones:
    def test_empty_instances_returns_empty(self):
        from shapely.geometry import Point

        zones = [{"geometry": Point(0, 0).buffer(10)}]
        result = _filter_by_exclusion_zones([], zones)
        assert result == []

    def test_no_zones_returns_all(self):
        instances = [_instance(10.0, 10.0), _instance(20.0, 20.0)]
        result = _filter_by_exclusion_zones(instances, [])
        assert result is instances

    def test_all_interior_points_excluded(self):
        from shapely.geometry import Point

        big_circle = Point(0, 0).buffer(1000)
        zones = [{"geometry": big_circle}]
        instances = [
            _instance(float(x), float(y)) for x in range(-5, 6) for y in range(-5, 6)
        ]
        result = _filter_by_exclusion_zones(instances, zones)
        assert result == []

    def test_all_exterior_points_kept(self):
        from shapely.geometry import Point

        tiny_circle = Point(0, 0).buffer(1)
        zones = [{"geometry": tiny_circle}]
        instances = [
            _instance(float(x), float(y))
            for x in range(100, 110)
            for y in range(100, 110)
        ]
        result = _filter_by_exclusion_zones(instances, zones)
        assert len(result) == len(instances)

    def test_mixed_inside_outside(self):
        from shapely.geometry import Point

        circle = Point(0, 0).buffer(5)
        zones = [{"geometry": circle}]
        inside = _instance(0.0, 0.0)
        outside = _instance(100.0, 100.0)
        result = _filter_by_exclusion_zones([inside, outside], zones)
        assert len(result) == 1
        assert result[0] is outside


def _roads():
    """A single 4 m wide asphalt road running along the y-axis (x in [-2, 2])."""
    from shapely.geometry import LineString

    from s2gos_generator.processors.roads import Road

    centerline = LineString([(0.0, -1000.0), (0.0, 1000.0)])
    return [Road(centerline=centerline, width=4.0, material="asphalt")]


class TestFilterByRoads:
    def test_excludes_positions_on_roads_keeps_others(self):
        on_road = _instance(0.0, 10.0)
        edge = _instance(2.0, 10.0)  # exactly on the edge -> boundary-safe exclusion
        off_road = _instance(50.0, 10.0)

        result = _filter_by_roads(
            [on_road, edge, off_road], _roads(), enabled=True, buffer_m=0.0
        )
        assert result == [off_road]

    def test_buffer_widens_exclusion(self):
        near = _instance(5.0, 0.0)  # 3 m outside the [-2,2] strip
        # No buffer keeps it; a 5 m buffer pulls it into the exclusion zone.
        kept = _filter_by_roads([near], _roads(), enabled=True, buffer_m=0.0)
        excluded = _filter_by_roads([near], _roads(), enabled=True, buffer_m=5.0)
        assert kept == [near]
        assert excluded == []

    def test_noop_when_disabled(self):
        instances = [_instance(0.0, 0.0)]  # squarely on the road
        assert (
            _filter_by_roads(instances, _roads(), enabled=False, buffer_m=0.0)
            is instances
        )

    def test_noop_when_no_roads(self):
        instances = [_instance(0.0, 0.0)]
        assert _filter_by_roads(instances, [], enabled=True, buffer_m=0.0) is instances


class TestBatchElevationLookup:
    def _make_dem(self):
        import numpy as np
        import xarray as xr

        z = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
        return xr.DataArray(
            z,
            dims=("y", "x"),
            coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]},
        )

    def test_drops_out_of_bounds(self):
        import math

        dem = self._make_dem()
        positions = [
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 2.0},
            {"x": 5.0, "y": 5.0},
        ]
        result = _batch_elevation_lookup(positions, dem)
        assert len(result) == 2
        for pos in result:
            assert math.isfinite(pos["elevation"])

    def test_empty_input(self):
        dem = self._make_dem()
        assert _batch_elevation_lookup([], dem) == []
