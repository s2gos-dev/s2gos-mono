"""Tests for water body support: OSM parsing, landcover completion, the water
sidecar round-trip, terraform-operation building, and conditional pipeline wiring."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Polygon, box, mapping

from s2gos_generator.core.config.water import WaterConfig
from s2gos_generator.processors.terrain_mesh import (
    WaterFlattenOperation,
    WayFlattenOperation,
)
from s2gos_generator.processors.water import (
    WaterBody,
    _group_areal_bodies_by_adjacency,
    build_water_terraform_operations,
    complete_with_landcover,
    fetch_osm_data,
    parse_water_bodies,
    smooth_dem_along_linear_bodies,
    water_bodies_from_sidecar,
    water_bodies_to_sidecar,
)
from s2gos_generator.resources.water import process_target_water


class _CoordStub:
    # 0.001 deg -> 100 m, centred on (45, 15); good enough for a +-5 km scene.
    def latlon_to_scene(self, lat, lon):
        return ((lon - 15.0) * 100000.0, (lat - 45.0) * 100000.0)


BOUNDS = box(-5000, -5000, 5000, 5000)


def _way(nodes, **tags):
    return {
        "type": "way",
        "tags": tags,
        "geometry": [{"lat": la, "lon": lo} for la, lo in nodes],
    }


def _relation(members, **tags):
    return {"type": "relation", "tags": tags, "members": members}


def _outer_member(nodes):
    return {
        "type": "way",
        "role": "outer",
        "geometry": [{"lat": la, "lon": lo} for la, lo in nodes],
    }


def _parse(*elements, cfg=None):
    return parse_water_bodies(
        {"elements": list(elements)}, cfg or WaterConfig(), _CoordStub(), BOUNDS
    )


class TestParseWaterBodies:
    def test_closed_natural_water_way_becomes_areal_body(self):
        square = [
            (45.0, 15.0),
            (45.0, 15.002),
            (45.002, 15.002),
            (45.002, 15.0),
            (45.0, 15.0),
        ]
        bodies = _parse(_way(square, natural="water"))
        assert len(bodies) == 1
        assert bodies[0].material == "water"
        assert bodies[0].anchor_xy is None
        assert bodies[0].geometry.area > 0

    def test_riverbank_closed_way_becomes_areal_body(self):
        square = [
            (45.0, 15.0),
            (45.0, 15.001),
            (45.001, 15.001),
            (45.001, 15.0),
            (45.0, 15.0),
        ]
        bodies = _parse(_way(square, waterway="riverbank"))
        assert len(bodies) == 1

    def test_linear_river_way_becomes_buffered_body_with_anchor(self):
        bodies = _parse(
            _way(
                [(45.0, 15.0), (45.002, 15.0)],
                waterway="river",
                name="fiume morto nuovo",
            )
        )
        assert len(bodies) == 1
        body = bodies[0]
        assert body.anchor_xy is not None
        # centerline/half_width must be retained -- required for
        # build_water_terraform_operations to flatten this like a road
        # (flat across the channel, gradient preserved along its length).
        assert body.centerline is not None
        assert body.half_width == pytest.approx(25.0)  # river default
        # river half-width defaults to 25.0 m -> buffered polygon area > 0
        assert body.geometry.area > 0
        minx, miny, maxx, maxy = body.geometry.bounds
        assert (maxx - minx) == pytest.approx(50.0, abs=1.0)  # 2 * half_width

    def test_canal_uses_its_own_half_width(self):
        bodies = _parse(
            _way(
                [(45.0, 15.0), (45.0, 15.002)],
                waterway="canal",
                name="canale demaniale",
            )
        )
        assert len(bodies) == 1
        minx, miny, maxx, maxy = bodies[0].geometry.bounds
        assert (maxy - miny) == pytest.approx(
            10.0, abs=1.0
        )  # 2 * canal half-width (5.0)

    def test_osm_width_tag_overrides_type_table(self):
        bodies = _parse(
            _way([(45.0, 15.0), (45.0, 15.002)], waterway="stream", width="10")
        )
        minx, miny, maxx, maxy = bodies[0].geometry.bounds
        assert (maxy - miny) == pytest.approx(10.0, abs=1.0)  # width tag wins -> half=5

    def test_relation_outer_members_assembled_into_one_body(self):
        outer = [
            (45.0, 15.0),
            (45.0, 15.003),
            (45.003, 15.003),
            (45.003, 15.0),
            (45.0, 15.0),
        ]
        bodies = _parse(_relation([_outer_member(outer)], natural="water"))
        assert len(bodies) == 1
        assert bodies[0].anchor_xy is None

    def test_relation_outer_boundary_split_across_arcs_is_stitched(self):
        # Real large water features (rivers, reservoirs, seas) commonly split
        # their outer boundary across many way segments, each an *open* arc
        # that only forms a ring once stitched end-to-end with its
        # neighbors. Force-closing each arc independently (a straight chord
        # from its own last point to its own first point) would fabricate a
        # bogus, oversized wedge instead of the true square.
        a, b, c, d = (45.0, 15.0), (45.0, 15.003), (45.003, 15.003), (45.003, 15.0)
        arc1 = _outer_member([a, b, c])  # open: a -> b -> c
        arc2 = _outer_member([c, d, a])  # open: c -> d -> a (closes the ring)
        bodies = _parse(_relation([arc1, arc2], natural="water"))
        assert len(bodies) == 1
        # 300m x 300m square (0.001 deg == 100m per _CoordStub) -- not the
        # much larger chord-closed wedge the old per-arc closing produced.
        assert bodies[0].geometry.area == pytest.approx(90000.0, rel=0.01)

    def test_relation_member_way_not_also_emitted_standalone(self):
        outer = [
            (45.0, 15.0),
            (45.0, 15.003),
            (45.003, 15.003),
            (45.003, 15.0),
            (45.0, 15.0),
        ]
        member = _outer_member(outer)
        member["ref"] = 42  # Overpass relation members reference ways by "ref"
        way = _way(outer, natural="water")
        way["id"] = 42
        relation = _relation([member], natural="water")
        bodies = _parse(relation, way)
        assert len(bodies) == 1  # not 2 -- the standalone way was suppressed

    def test_non_water_way_ignored(self):
        assert _parse(_way([(45.0, 15.0), (45.002, 15.0)], highway="residential")) == []

    def test_body_clipped_to_scene_bounds(self):
        bodies = _parse(_way([(45.0, 15.0), (45.1, 15.0)], waterway="river"))
        assert len(bodies) == 1
        _, ymin, _, ymax = bodies[0].geometry.bounds
        assert ymax <= 5000.0 + 1e-6

    def test_intermittent_way_kept_by_default(self):
        bodies = _parse(
            _way([(45.0, 15.0), (45.002, 15.0)], waterway="stream", intermittent="yes")
        )
        assert len(bodies) == 1

    def test_intermittent_way_excluded_when_configured(self):
        cfg = WaterConfig(exclude_intermittent=True)
        bodies = _parse(
            _way([(45.0, 15.0), (45.002, 15.0)], waterway="stream", intermittent="yes"),
            cfg=cfg,
        )
        assert bodies == []
        # a non-intermittent stream in the same call is still kept
        bodies = _parse(
            _way([(45.0, 15.0), (45.002, 15.0)], waterway="stream"), cfg=cfg
        )
        assert len(bodies) == 1

    def test_intermittent_relation_excluded_when_configured(self):
        square = [
            (45.0, 15.0),
            (45.0, 15.003),
            (45.003, 15.003),
            (45.003, 15.0),
            (45.0, 15.0),
        ]
        cfg = WaterConfig(exclude_intermittent=True)
        bodies = _parse(
            _relation([_outer_member(square)], natural="water", intermittent="yes"),
            cfg=cfg,
        )
        assert bodies == []


class TestWaterConfig:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(source="file", file_path=None), "file_path is required"),
            (
                dict(source="file", file_path=Path("/no/such/water_file.json")),
                "not found",
            ),
        ],
        ids=["missing-path", "nonexistent-path"],
    )
    def test_file_source_validation_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WaterConfig(**kwargs)


class TestFetchOsmData:
    def test_malformed_json_file_returns_none(self, tmp_path):
        bad = tmp_path / "water.json"
        bad.write_text("not json{")
        cfg = WaterConfig(source="file", file_path=bad)
        assert fetch_osm_data(cfg, 0.0, 0.0, 1.0, 1.0) is None


class TestWaterSidecar:
    def test_roundtrip_preserves_bodies(self):
        bodies = [
            WaterBody(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), "water"),
            WaterBody(
                Polygon([(20, 0), (30, 0), (30, 5), (20, 5)]),
                "water",
                anchor_xy=(25.0, 2.5),
                centerline=LineString([(20, 2.5), (30, 2.5)]),
                half_width=2.5,
            ),
        ]
        restored = water_bodies_from_sidecar(water_bodies_to_sidecar(bodies))

        def _key(b):
            return (
                b.material,
                b.anchor_xy,
                b.half_width,
                list(b.geometry.exterior.coords),
                list(b.centerline.coords) if b.centerline is not None else None,
            )

        assert [_key(b) for b in restored] == [_key(b) for b in bodies]

    def test_unknown_version_returns_empty(self):
        assert water_bodies_from_sidecar({"version": 2, "water_layers": []}) == []


class TestCompleteWithLandcover:
    def _grid(self, n=40, extent=400.0):
        coords = np.linspace(-extent, extent, n)
        return coords

    def test_adds_missed_flat_body_above_min_area(self):
        n = 40
        coords = self._grid(n=n)
        lc = np.full((n, n), 10, dtype=np.uint8)
        lc[10:20, 10:20] = 80  # a flat pond OSM never saw
        landcover_data = xr.DataArray(
            lc, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )
        elev = np.zeros((n, n))
        dem_data = xr.DataArray(
            elev, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )

        cfg = WaterConfig(landcover_completion_min_area_m2=1.0)
        new_bodies = complete_with_landcover([], landcover_data, dem_data, cfg)
        assert len(new_bodies) == 1
        assert new_bodies[0].material == "water"

    def test_sloped_pixels_excluded(self):
        n = 40
        coords = self._grid(n=n)
        lc = np.full((n, n), 10, dtype=np.uint8)
        lc[10:14, 10:30] = 80  # a strip, but steeply sloped -> bank, not water
        landcover_data = xr.DataArray(
            lc, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )
        elev = np.zeros((n, n))
        elev[10:14, 10:30] = np.tile(np.linspace(0, 50, 20), (4, 1))
        dem_data = xr.DataArray(
            elev, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )

        cfg = WaterConfig(landcover_completion_min_area_m2=1.0)
        assert complete_with_landcover([], landcover_data, dem_data, cfg) == []

    def test_tiny_component_below_min_area_excluded(self):
        n = 40
        coords = self._grid(n=n)
        lc = np.full((n, n), 10, dtype=np.uint8)
        lc[10, 10] = 80  # a single pixel
        landcover_data = xr.DataArray(
            lc, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )
        elev = np.zeros((n, n))
        dem_data = xr.DataArray(
            elev, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )

        cfg = WaterConfig(landcover_completion_min_area_m2=10_000.0)
        assert complete_with_landcover([], landcover_data, dem_data, cfg) == []

    def test_no_water_class_returns_empty(self):
        n = 20
        coords = self._grid(n=n)
        lc = np.full((n, n), 10, dtype=np.uint8)
        landcover_data = xr.DataArray(
            lc, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )
        elev = np.zeros((n, n))
        dem_data = xr.DataArray(
            elev, coords={"y": coords, "x": coords}, dims=["y", "x"]
        )
        assert (
            complete_with_landcover([], landcover_data, dem_data, WaterConfig()) == []
        )


class TestGroupArealBodiesByAdjacency:
    def test_touching_polygons_grouped_together(self):
        left = Polygon([(-10, -10), (0, -10), (0, 10), (-10, 10)])
        right = Polygon([(0, -10), (10, -10), (10, 10), (0, 10)])
        bodies = [WaterBody(left, "water"), WaterBody(right, "water")]
        clusters = _group_areal_bodies_by_adjacency(bodies)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_separate_polygons_not_grouped(self):
        a = Polygon([(-10, -10), (0, -10), (0, 10), (-10, 10)])
        b = Polygon([(1000, -10), (1010, -10), (1010, 10), (1000, 10)])
        bodies = [WaterBody(a, "water"), WaterBody(b, "water")]
        clusters = _group_areal_bodies_by_adjacency(bodies)
        assert len(clusters) == 2

    def test_empty_input_returns_empty(self):
        assert _group_areal_bodies_by_adjacency([]) == []


class TestSmoothDemAlongLinearBodies:
    def _noisy_sloped_dem(
        self, seed=0, noise_std=0.8, slope=-0.005, n=101, extent=1000.0
    ):
        x = np.linspace(0.0, extent, n)
        y = np.linspace(0.0, extent, n)
        xx, yy = np.meshgrid(x, y)
        elev = 10.0 + slope * yy
        rng = np.random.default_rng(seed)
        elev += rng.normal(0.0, noise_std, elev.shape)
        return xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

    def test_smoothing_reduces_noise_but_keeps_trend(self):
        from s2gos_generator.processors.terrain_mesh import extract_dem
        from s2gos_generator.processors.water import _sample_dem_points

        dem = self._noisy_sloped_dem()
        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )

        smoothed = smooth_dem_along_linear_bodies(dem, [body], smooth_window_m=150.0)

        dx, dy, delev = extract_dem(dem)
        sx, sy, selev = extract_dem(smoothed)
        pts = np.array([[500.0, d] for d in np.linspace(0, 1000, 50)])
        raw = _sample_dem_points(dx, dy, delev, pts)
        sm = _sample_dem_points(sx, sy, selev, pts)

        # Noise (sample-to-sample jitter) is substantially reduced...
        assert np.std(np.diff(sm)) < np.std(np.diff(raw)) * 0.5
        # ...but the real downhill trend survives.
        assert sm[0] > sm[-1]

    def test_margin_is_floored_to_dem_pixel_pitch(self):
        # Regression test: the smoothed band (half_width + margin_m) must be
        # wide enough that the bilinear elevation lookup used later never
        # straddles the smoothed/untouched boundary -- otherwise a query
        # point near the (too-narrow) band edge blends a smoothed cell with
        # an adjacent raw, noisy one. With a coarse 50 m DEM grid and the
        # default margin_m=10.0, the *unfloored* band (half_width=5 + 10 =
        # 15 m) would leave a cell 50 m from the centerline untouched; the
        # floored effective margin (>= 2x the 50 m pixel pitch = 100 m) must
        # extend the band far enough to smooth it anyway.
        x = np.arange(0.0, 1001.0, 50.0)
        y = np.arange(0.0, 1001.0, 50.0)
        xx, yy = np.meshgrid(x, y)
        elev = np.full_like(xx, 10.0)
        # A deliberately huge, isolated spike 50 m from the centerline --
        # outside the old, unfloored band (15 m) but inside the pitch-floored
        # one (>= 100 m).
        spike_row = np.argmin(np.abs(y - 500.0))
        spike_col = np.argmin(np.abs(x - 550.0))
        elev[spike_row, spike_col] = 999.0
        dem = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )

        smoothed = smooth_dem_along_linear_bodies(
            dem, [body], smooth_window_m=150.0, margin_m=10.0
        )

        assert float(smoothed.values[spike_row, spike_col]) != pytest.approx(999.0)

    def test_zero_window_disables_smoothing(self):
        dem = self._noisy_sloped_dem()
        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )
        result = smooth_dem_along_linear_bodies(dem, [body], smooth_window_m=0.0)
        assert result is dem

    def test_no_linear_bodies_returns_input_unchanged(self):
        dem = self._noisy_sloped_dem()
        result = smooth_dem_along_linear_bodies(dem, [], smooth_window_m=150.0)
        assert result is dem

    def test_far_from_centerline_untouched(self):
        dem = self._noisy_sloped_dem()
        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )
        smoothed = smooth_dem_along_linear_bodies(
            dem, [body], smooth_window_m=150.0, margin_m=10.0
        )
        # Far from the centerline (x=50, well outside half_width+margin=15) values are untouched.
        assert np.array_equal(
            smoothed.sel(x=50.0, method="nearest").values,
            dem.sel(x=50.0, method="nearest").values,
        )

    def test_endpoint_outlier_cluster_is_rejected(self):
        # Real-data regression: a building/bridge/DSM artifact concentrated
        # right at an OSM way's endpoint can span *several* consecutive raw
        # samples, not just one -- a plain rolling median centered there is
        # still pulled toward the anomaly since it dominates its own window.
        # The near-flat channel elevation elsewhere must survive.
        x = np.linspace(0.0, 1000.0, 101)
        y = np.linspace(0.0, 1000.0, 101)
        xx, yy = np.meshgrid(x, y)
        elev = np.full_like(xx, 1.0)

        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        # Bake a localized anomaly (e.g. a bridge deck) into the raw DEM near
        # the centerline's start, spanning several sample spacings (~60m).
        anomaly = (np.abs(xx - 500.0) <= 5.0) & (yy <= 60.0)
        elev_with_anomaly = elev.copy()
        elev_with_anomaly[anomaly] = 8.0
        dem_anomaly = xr.DataArray(
            elev_with_anomaly, coords={"y": y, "x": x}, dims=["y", "x"]
        )

        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )
        smoothed = smooth_dem_along_linear_bodies(
            dem_anomaly,
            [body],
            smooth_window_m=150.0,
            sample_spacing_m=20.0,
            outlier_reject_m=1.5,
        )

        from s2gos_generator.processors.terrain_mesh import extract_dem
        from s2gos_generator.processors.water import _sample_dem_points

        sx, sy, selev = extract_dem(smoothed)
        pt_at_start = np.array([[500.0, 5.0]])  # right inside the anomaly zone
        z = _sample_dem_points(sx, sy, selev, pt_at_start)[0]
        # Without outlier rejection this reads close to the anomaly's 8.0;
        # with it, it should be pulled back close to the true 1.0 channel level.
        assert z < 3.0

    def test_outlier_reject_disabled_still_smooths(self):
        x = np.linspace(0.0, 1000.0, 101)
        y = np.linspace(0.0, 1000.0, 101)
        xx, yy = np.meshgrid(x, y)
        elev = np.full_like(xx, 1.0)
        anomaly = (np.abs(xx - 500.0) <= 5.0) & (yy <= 60.0)
        elev[anomaly] = 8.0
        dem = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )
        # outlier_reject_m=0 disables the rejection pass -- result must still
        # be a valid array (no crash), not asserting a specific value.
        smoothed = smooth_dem_along_linear_bodies(
            dem, [body], smooth_window_m=150.0, outlier_reject_m=0.0
        )
        assert smoothed is not None

    def test_steep_mountain_stream_grade_is_preserved(self):
        # Regression test: a fixed outlier-rejection threshold must not flag real, sustained steep grade as noise.
        x = np.linspace(0.0, 1000.0, 101)
        y = np.linspace(0.0, 1000.0, 101)
        xx, yy = np.meshgrid(x, y)
        # A steep, steadily accelerating descent (~40 m drop over 1000 m), like a real tributary nearing confluence.
        elev = 50.0 - 0.02 * yy - 0.00002 * yy**2
        rng = np.random.default_rng(0)
        elev += rng.normal(0.0, 0.3, elev.shape)
        dem = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )

        smoothed = smooth_dem_along_linear_bodies(
            dem,
            [body],
            smooth_window_m=150.0,
            sample_spacing_m=20.0,
            outlier_reject_m=1.5,
        )

        from s2gos_generator.processors.terrain_mesh import extract_dem
        from s2gos_generator.processors.water import _sample_dem_points

        sx, sy, selev = extract_dem(smoothed)
        pts = np.array([[500.0, d] for d in np.linspace(0, 1000, 50)])
        sm = _sample_dem_points(sx, sy, selev, pts)

        # The downhill trend must largely survive, not collapse into a flat plateau.
        assert sm[0] - sm[-1] > 25.0
        # And smoothly -- no single step anywhere close to the total drop.
        assert np.abs(np.diff(sm)).max() < 5.0

    def test_final_profile_has_no_median_staircase(self):
        # Regression test: a rolling *median* final pass is locally constant
        # over long runs of the window and then jumps discretely to the next
        # plateau -- a staircase, not a low-pass filter -- which bakes visible
        # reflection banding into an otherwise-smooth channel. The final pass
        # must behave like a continuous low-pass filter instead: no long run
        # of (near-)identical consecutive values.
        from s2gos_generator.processors.terrain_mesh import extract_dem
        from s2gos_generator.processors.water import _sample_dem_points

        dem = self._noisy_sloped_dem(noise_std=0.8)
        cl = LineString([(500.0, 0.0), (500.0, 1000.0)])
        body = WaterBody(
            geometry=cl.buffer(5.0), material="water", centerline=cl, half_width=5.0
        )

        smoothed = smooth_dem_along_linear_bodies(
            dem, [body], smooth_window_m=150.0, sample_spacing_m=20.0
        )

        sx, sy, selev = extract_dem(smoothed)
        pts = np.array([[500.0, d] for d in np.linspace(0, 1000, 100)])
        sm = _sample_dem_points(sx, sy, selev, pts)

        # A median-plateau staircase produces long runs of (near-)exactly
        # equal consecutive samples (flat shelves), interrupted by a handful
        # of larger jumps. A continuous rolling mean does not: the longest
        # run of near-zero consecutive differences should be short relative
        # to the sample count.
        diffs = np.abs(np.diff(sm))
        flat = diffs < 1e-9
        longest_flat_run = 0
        current = 0
        for is_flat in flat:
            current = current + 1 if is_flat else 0
            longest_flat_run = max(longest_flat_run, current)
        assert longest_flat_run < 10


class TestBuildWaterTerraformOperations:
    def test_reference_elevation_matches_flat_dem(self):
        x = np.linspace(-500.0, 500.0, 50)
        y = np.linspace(-500.0, 500.0, 50)
        elev = np.full((50, 50), 3.0)
        dem_data = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        body = WaterBody(
            Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)]), "water"
        )
        ops = build_water_terraform_operations(
            [body], dem_data, transition_buffer_m=10.0, dem_resolution_m=20.0
        )
        assert len(ops) == 1
        assert ops[0].reference_z == pytest.approx(3.0, abs=0.1)
        assert isinstance(ops[0], WaterFlattenOperation)

    def test_linear_body_becomes_road_flatten_operation(self):
        # A river/canal/stream must flatten like a road -- flat across the
        # channel, but sampled per-vertex along the original centerline, so
        # any real downstream gradient is preserved rather than snapped to a
        # single body-wide elevation.
        x = np.linspace(-500.0, 500.0, 50)
        y = np.linspace(-500.0, 500.0, 50)
        dem_data = xr.DataArray(
            np.zeros((50, 50)), coords={"y": y, "x": x}, dims=["y", "x"]
        )

        centerline = LineString([(0.0, -400.0), (0.0, 400.0)])
        body = WaterBody(
            geometry=centerline.buffer(5.0, cap_style="flat"),
            material="water",
            anchor_xy=(0.0, 0.0),
            centerline=centerline,
            half_width=5.0,
        )
        ops = build_water_terraform_operations(
            [body], dem_data, transition_buffer_m=10.0, dem_resolution_m=20.0
        )
        assert len(ops) == 1
        assert isinstance(ops[0], WayFlattenOperation)

    def test_adjacent_linear_segments_meet_at_shared_endpoint_elevation(self):
        # Two consecutive OSM way segments of the same river, sharing an
        # endpoint node. Each must flatten to the SAME elevation at that
        # shared point -- this is exactly the "step"/"lane" artifact fixed
        # here: independently-sampled single reference elevations per
        # segment produced visible discontinuities at segment boundaries.
        x = np.linspace(-500.0, 500.0, 100)
        y = np.linspace(-500.0, 500.0, 100)
        # A real downstream gradient: elevation rises steadily along y.
        yy = np.tile(y.reshape(-1, 1), (1, 100))
        elev = yy * 0.01  # 1 cm per metre -- a gentle, real longitudinal slope
        dem_data = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        upstream_cl = LineString([(0.0, 0.0), (0.0, 200.0)])
        downstream_cl = LineString([(0.0, -200.0), (0.0, 0.0)])
        bodies = [
            WaterBody(
                geometry=upstream_cl.buffer(5.0, cap_style="flat"),
                material="water",
                anchor_xy=(0.0, 100.0),
                centerline=upstream_cl,
                half_width=5.0,
            ),
            WaterBody(
                geometry=downstream_cl.buffer(5.0, cap_style="flat"),
                material="water",
                anchor_xy=(0.0, -100.0),
                centerline=downstream_cl,
                half_width=5.0,
            ),
        ]
        ops = build_water_terraform_operations(
            bodies, dem_data, transition_buffer_m=0.0, dem_resolution_m=10.0
        )
        assert len(ops) == 2

        def _elevation_fn(xy):
            from s2gos_generator.processors.terrain_mesh.builder import (
                _make_elevation_fn,
            )

            dem_x, dem_y, dem_elev = x, y, elev
            return _make_elevation_fn(dem_x, dem_y, dem_elev)(xy)

        # Sample right at the shared endpoint (0, 0) through each operation.
        verts_upstream = np.array([[0.0, 0.0, 999.0]])
        verts_downstream = np.array([[0.0, 0.0, 999.0]])
        out_up = ops[0].apply(verts_upstream.copy(), _elevation_fn)
        out_down = ops[1].apply(verts_downstream.copy(), _elevation_fn)
        assert out_up[0, 2] == pytest.approx(out_down[0, 2], abs=1e-9)

    def test_elongated_areal_polygon_stays_water_flatten_operation(self):
        # A long/narrow AREAL polygon (e.g. a whole river mapped as one
        # riverbank polygon, no stored centerline/half_width) must NOT be
        # given a derived straight-chord centerline -- that was tried and
        # reverted: for a curved/bent real body it samples the DEM at
        # effectively noisy positions near bends, producing a wavy/streaky
        # artifact worse than a single flat level. Elongated or not, an
        # areal body without a real OSM centerline stays single-scalar.
        x = np.linspace(-500.0, 500.0, 100)
        y = np.linspace(-500.0, 500.0, 100)
        dem_data = xr.DataArray(
            np.full((100, 100), 5.0), coords={"y": y, "x": x}, dims=["y", "x"]
        )

        strip = Polygon([(-5, -400), (5, -400), (5, 400), (-5, 400)])
        body = WaterBody(geometry=strip, material="water")
        ops = build_water_terraform_operations(
            [body], dem_data, transition_buffer_m=10.0, dem_resolution_m=10.0
        )
        assert len(ops) == 1
        assert isinstance(ops[0], WaterFlattenOperation)

    def test_touching_areal_bodies_share_one_reference_elevation(self):
        # Two adjacent (touching) compact areal bodies -- e.g. consecutive
        # riverbank-way polygons -- must merge into ONE WaterFlattenOperation
        # sharing a single reference elevation, not two independently
        # sampled (and therefore likely different) flat levels meeting at a
        # visible transition line.
        x = np.linspace(-500.0, 500.0, 100)
        y = np.linspace(-500.0, 500.0, 100)
        elev = np.zeros((100, 100))
        elev[:, x < 0] = 2.0  # left half at 2.0m
        elev[:, x >= 0] = 8.0  # right half at 8.0m -- a real local difference
        dem_data = xr.DataArray(elev, coords={"y": y, "x": x}, dims=["y", "x"])

        left = Polygon([(-100, -50), (0, -50), (0, 50), (-100, 50)])
        right = Polygon(
            [(0, -50), (100, -50), (100, 50), (0, 50)]
        )  # shares the x=0 edge
        bodies = [
            WaterBody(geometry=left, material="water"),
            WaterBody(geometry=right, material="water"),
        ]
        ops = build_water_terraform_operations(
            bodies,
            dem_data,
            transition_buffer_m=10.0,
            dem_resolution_m=10.0,
            erosion_margin_m=5.0,
        )
        assert len(ops) == 1  # merged into one operation, not two
        assert isinstance(ops[0], WaterFlattenOperation)

    def test_no_bodies_returns_empty(self):
        x = np.linspace(-500.0, 500.0, 10)
        y = np.linspace(-500.0, 500.0, 10)
        dem_data = xr.DataArray(
            np.zeros((10, 10)), coords={"y": y, "x": x}, dims=["y", "x"]
        )
        assert (
            build_water_terraform_operations([], dem_data, transition_buffer_m=10.0)
            == []
        )

    def test_thin_water_skip_excludes_narrow_bodies(self):
        x = np.linspace(-500.0, 500.0, 50)
        y = np.linspace(-500.0, 500.0, 50)
        dem_data = xr.DataArray(
            np.zeros((50, 50)), coords={"y": y, "x": x}, dims=["y", "x"]
        )

        narrow = WaterBody(
            Polygon([(-1, -100), (1, -100), (1, 100), (-1, 100)]), "water"
        )
        ops = build_water_terraform_operations(
            [narrow], dem_data, transition_buffer_m=10.0, thin_water_skip_m=5.0
        )
        assert ops == []

    def test_thin_water_skip_excludes_narrow_linear_bodies(self):
        x = np.linspace(-500.0, 500.0, 50)
        y = np.linspace(-500.0, 500.0, 50)
        dem_data = xr.DataArray(
            np.zeros((50, 50)), coords={"y": y, "x": x}, dims=["y", "x"]
        )

        centerline = LineString([(0.0, -100.0), (0.0, 100.0)])
        narrow = WaterBody(
            geometry=centerline.buffer(1.0, cap_style="flat"),
            material="water",
            centerline=centerline,
            half_width=1.0,  # width 2.0, below the 5.0 skip threshold
        )
        ops = build_water_terraform_operations(
            [narrow], dem_data, transition_buffer_m=10.0, thin_water_skip_m=5.0
        )
        assert ops == []


def _write_water_sidecar(path, *, version=1):
    path.write_text(
        json.dumps(
            {
                "version": version,
                "water_layers": [
                    {
                        "material_name": "water",
                        "bodies": [
                            {
                                "geometry": mapping(
                                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
                                ),
                                "anchor_xy": None,
                            }
                        ],
                    }
                ],
            }
        )
    )


class TestWaterCtxSidecar:
    def test_ctx_water_bodies_reads_back_sidecar(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        assert ctx.water_bodies == []  # no sidecar -> empty

        sidecar = tmp_path / "water_bodies.json"
        _write_water_sidecar(sidecar)
        ctx.assets.water_file = sidecar
        ctx._water_bodies = None  # reset lazy cache

        bodies = ctx.water_bodies
        assert len(bodies) == 1
        assert isinstance(bodies[0], WaterBody)
        assert bodies[0].material == "water"

    def test_ctx_water_bodies_rejects_unknown_version(
        self, make_minimal_config, tmp_path
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        sidecar = tmp_path / "water_bodies.json"
        _write_water_sidecar(sidecar, version=99)
        ctx.assets.water_file = sidecar
        assert ctx.water_bodies == []

    def test_process_target_water_writes_grouped_sidecar(
        self, make_minimal_config, monkeypatch
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(
            make_minimal_config(
                water=WaterConfig(enabled=True, landcover_completion=False)
            )
        )
        Path(str(ctx.data_dir)).mkdir(parents=True, exist_ok=True)

        canned = {
            "elements": [
                {
                    "type": "way",
                    "tags": {"natural": "water"},
                    "geometry": [
                        {"lat": 45.0, "lon": 15.0},
                        {"lat": 45.0, "lon": 15.002},
                        {"lat": 45.002, "lon": 15.002},
                        {"lat": 45.002, "lon": 15.0},
                        {"lat": 45.0, "lon": 15.0},
                    ],
                },
            ]
        }
        monkeypatch.setattr(
            "s2gos_generator.resources.water.fetch_osm_data",
            lambda *a, **k: canned,
        )

        sidecar_path = process_target_water(ctx)
        assert sidecar_path is not None
        assert ctx.assets.water_file == sidecar_path

        data = json.loads(Path(str(sidecar_path)).read_text())
        assert data["version"] == 1
        materials = [layer["material_name"] for layer in data["water_layers"]]
        assert materials == ["water"]


def _write_way_sidecar_for_overlap(path, *, version=1):
    """A vertical way strip (x in [-3.5, 3.5], y in [0, 100]) crossing the
    water square written by ``_write_water_sidecar`` (x, y in [0, 10])."""
    path.write_text(
        json.dumps(
            {
                "version": version,
                "way_layers": [
                    {
                        "material_name": "asphalt",
                        "ways": [
                            {
                                "centerline": mapping(
                                    LineString([(0.0, 0.0), (0.0, 100.0)])
                                ),
                                "width": 7.0,
                            }
                        ],
                    }
                ],
            }
        )
    )


class TestWaterWayOverlap:
    """A way crossing a water body (a bridge, out of scope to model) should
    keep its way material in the texture; the water body's elevation is
    untouched since it's computed from ``water_bodies`` directly."""

    def test_way_polygon_subtracted_from_water_texture_footprint(
        self, make_minimal_config, tmp_path
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())

        water_sidecar = tmp_path / "water_bodies.json"
        _write_water_sidecar(water_sidecar)  # water square (0,0)-(10,10)
        ctx.assets.water_file = water_sidecar

        way_sidecar = tmp_path / "ways.json"
        _write_way_sidecar_for_overlap(way_sidecar)  # way strip x in [-3.5, 3.5]
        ctx.assets.ways_file = way_sidecar

        way_poly = ctx.way_polygons_by_material["asphalt"]
        water_poly = ctx.water_polygons_by_material["water"]

        # The way footprint is no longer part of the water texture footprint...
        assert water_poly.intersection(way_poly).area == pytest.approx(0.0, abs=1e-9)
        # ...but the water body still covers everything outside the way strip
        # (regression guard against over-subtracting the whole body).
        assert water_poly.area == pytest.approx(100.0 - 10.0 * 3.5, abs=1e-9)

        # Mesh elevation is unaffected: the raw water body geometry (what
        # build_water_terraform_operations consumes) still spans the full square.
        assert ctx.water_bodies[0].geometry.area == pytest.approx(100.0, abs=1e-9)

    def test_water_polygons_unchanged_when_no_ways(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        sidecar = tmp_path / "water_bodies.json"
        _write_water_sidecar(sidecar)
        ctx.assets.water_file = sidecar

        assert ctx.way_polygons_by_material == {}
        water_poly = ctx.water_polygons_by_material["water"]
        assert water_poly.area == pytest.approx(100.0, abs=1e-9)


class TestWaterWiring:
    """`target_water` is registered only when water is enabled."""

    def test_target_water_registered_only_when_enabled(self, make_minimal_config):
        from s2gos_generator.core.pipeline import SceneGenerationPipeline

        without = SceneGenerationPipeline(make_minimal_config())
        deps_without = without.get_resource_dependencies()
        assert "target_water" not in deps_without
        assert "target_water" not in deps_without["target_texture"]

        with_water = SceneGenerationPipeline(
            make_minimal_config(water=WaterConfig(enabled=True))
        )
        deps_with = with_water.get_resource_dependencies()
        assert set(deps_with["target_water"]) == {"target_dem", "target_landcover"}
        # Resolved as an optional dependency of the mesh and texture steps.
        assert "target_water" in deps_with["target_mesh"]
        assert "target_water" in deps_with["target_texture"]
