"""Tests for water bodies: OSM parsing, landcover completion, sidecar, terraforming and wiring."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from shapely import affinity
from shapely.geometry import LineString, box

from s2gos_generator.core.config.water import WaterConfig
from s2gos_generator.core.fingerprints import (
    _WATER_MESH_FIELDS,
    _WATER_SOURCE_FIELDS,
    _WATER_TEXTURE_FIELDS,
    ResourceFingerprints,
)
from s2gos_generator.processors.terrain_mesh import (
    WaterFlattenOperation,
    WayFlattenOperation,
    extract_dem,
)
from s2gos_generator.processors.water import (
    WaterBody,
    _sample_dem_points,
    build_water_terraform_operations,
    complete_with_landcover,
    parse_water_bodies,
    smooth_dem_along_linear_bodies,
    water_bodies_from_sidecar,
    water_bodies_to_sidecar,
)
from s2gos_generator.resources.water import process_target_water


class _CoordStub:
    # 0.001 deg -> 100 m, centred on (45, 15).
    def latlon_to_scene(self, lat, lon):
        return ((lon - 15.0) * 100000.0, (lat - 45.0) * 100000.0)


BOUNDS = box(-5000, -5000, 5000, 5000)
SQUARE = [(45.0, 15.0), (45.0, 15.003), (45.003, 15.003), (45.003, 15.0), (45.0, 15.0)]


def _nodes(nodes):
    return [{"lat": la, "lon": lo} for la, lo in nodes]


def _way(nodes, **tags):
    return {"type": "way", "tags": tags, "geometry": _nodes(nodes)}


def _member(nodes, role="outer"):
    return {"type": "way", "role": role, "geometry": _nodes(nodes)}


def _relation(members, **tags):
    return {"type": "relation", "tags": tags, "members": members}


def _parse(*elements, cfg=None):
    return parse_water_bodies(
        {"elements": list(elements)}, cfg or WaterConfig(), _CoordStub(), BOUNDS
    )


def _grid(values, extent=400.0):
    n = values.shape[0]
    coords = np.linspace(-extent, extent, n)
    return xr.DataArray(values, coords={"y": coords, "x": coords}, dims=["y", "x"])


class TestParseWaterBodies:
    def test_closed_natural_water_way_becomes_areal_body(self):
        (body,) = _parse(_way(SQUARE, natural="water"))
        assert body.centerline is None
        assert body.geometry.area == pytest.approx(90000.0, rel=0.01)

    def test_linear_river_keeps_centerline_and_type_half_width(self):
        (body,) = _parse(_way([(45.0, 15.0), (45.002, 15.0)], waterway="river"))
        assert body.centerline is not None
        assert body.half_width == WaterConfig.WATERWAY_HALF_WIDTH_M["river"]

    def test_osm_width_tag_overrides_type_table(self):
        (body,) = _parse(
            _way([(45.0, 15.0), (45.0, 15.002)], waterway="stream", width="10")
        )
        assert body.half_width == 5.0

    def test_relation_boundary_split_across_arcs_is_stitched(self):
        a, b, c, d = SQUARE[:4]
        (body,) = _parse(
            _relation([_member([a, b, c]), _member([c, d, a])], natural="water")
        )
        assert body.geometry.area == pytest.approx(90000.0, rel=0.01)

    def test_relation_inner_members_become_holes(self):
        island = [
            (45.001, 15.001),
            (45.001, 15.002),
            (45.002, 15.002),
            (45.002, 15.001),
        ]
        (body,) = _parse(
            _relation(
                [_member(SQUARE), _member(island + island[:1], role="inner")],
                natural="water",
            )
        )
        assert len(body.geometry.interiors) == 1
        assert body.geometry.area == pytest.approx(80000.0, rel=0.01)

    def test_relation_member_way_not_also_emitted_standalone(self):
        member = {**_member(SQUARE), "ref": 42}
        way = {**_way(SQUARE, natural="water"), "id": 42}
        assert len(_parse(_relation([member], natural="water"), way)) == 1

    def test_intermittent_way_excluded_when_configured(self):
        line = [(45.0, 15.0), (45.002, 15.0)]
        cfg = WaterConfig(exclude_intermittent=True)
        assert _parse(_way(line, waterway="stream", intermittent="yes"), cfg=cfg) == []
        assert len(_parse(_way(line, waterway="stream"), cfg=cfg)) == 1


def test_sidecar_roundtrip_preserves_bodies():
    bodies = [
        WaterBody(box(0, 0, 10, 10), "water"),
        WaterBody(box(20, 0, 30, 5), "water", LineString([(20, 2.5), (30, 2.5)]), 2.5),
    ]
    restored = water_bodies_from_sidecar(water_bodies_to_sidecar(bodies))
    assert [(b.geometry, b.centerline, b.half_width) for b in restored] == [
        (b.geometry, b.centerline, b.half_width) for b in bodies
    ]


class TestCompleteWithLandcover:
    CFG = WaterConfig(landcover_completion_min_area_m2=1.0)

    def _complete(self, lc, elev=None, osm=(), cfg=CFG):
        elev = np.zeros(lc.shape) if elev is None else elev
        return complete_with_landcover(list(osm), _grid(lc), _grid(elev), cfg)

    def test_adds_missed_flat_body(self):
        lc = np.full((40, 40), 10, dtype=np.uint8)
        lc[10:20, 10:20] = 80
        assert len(self._complete(lc)) == 1

    def test_sloped_pixels_excluded(self):
        lc = np.full((40, 40), 10, dtype=np.uint8)
        lc[10:14, 10:30] = 80
        elev = np.zeros((40, 40))
        elev[10:14, 10:30] = np.tile(np.linspace(0, 50, 20), (4, 1))
        assert self._complete(lc, elev) == []

    def test_small_area_touching_osm_excluded(self):
        lc = np.full((40, 40), 10, dtype=np.uint8)
        lc[10:20, 10:30] = 80
        osm = WaterBody(box(-200, -40, 150, -20), "water")
        assert self._complete(lc, osm=[osm]) == []

    def test_area_beside_osm_across_land_kept(self):
        lc = np.full((40, 40), 10, dtype=np.uint8)
        lc[10:20, 10:20] = 80
        osm = WaterBody(box(-212, -190, -208, -20), "water")
        assert len(self._complete(lc, osm=[osm])) == 1

    def test_large_area_touching_osm_kept(self):
        lc = np.full((40, 40), 10, dtype=np.uint8)
        lc[:, :15] = 80
        osm = WaterBody(box(-120, -10, 300, 10), "water")
        cfg = WaterConfig(
            landcover_completion_min_area_m2=1.0,
            landcover_completion_linked_min_area_m2=50_000.0,
        )
        bodies = self._complete(lc, osm=[osm], cfg=cfg)
        assert sum(b.geometry.area for b in bodies) > 50_000.0


class TestSmoothDemAlongLinearBodies:
    CL = LineString([(500.0, 0.0), (500.0, 1000.0)])

    def _smooth_profile(self, elev):
        x = np.linspace(0.0, 1000.0, elev.shape[0])
        dem = xr.DataArray(elev, coords={"y": x, "x": x}, dims=["y", "x"])
        body = WaterBody(self.CL.buffer(5.0), "water", self.CL, 5.0)
        smoothed = smooth_dem_along_linear_bodies(dem, [body], smooth_window_m=150.0)
        pts = np.array([[500.0, d] for d in np.linspace(0, 1000, 50)])
        return (
            _sample_dem_points(*extract_dem(dem), pts),
            _sample_dem_points(*extract_dem(smoothed), pts),
        )

    def _yy(self):
        return np.meshgrid(*[np.linspace(0.0, 1000.0, 101)] * 2)[1]

    def test_reduces_noise_but_keeps_trend(self):
        elev = 10.0 - 0.005 * self._yy()
        elev += np.random.default_rng(0).normal(0.0, 0.8, elev.shape)
        raw, smoothed = self._smooth_profile(elev)
        assert np.std(np.diff(smoothed)) < 0.5 * np.std(np.diff(raw))
        assert smoothed[0] > smoothed[-1]

    def test_outlier_cluster_at_endpoint_rejected(self):
        yy = self._yy()
        elev = np.ones_like(yy)
        elev[(np.abs(yy.T - 500.0) <= 5.0) & (yy <= 60.0)] = 8.0
        _, smoothed = self._smooth_profile(elev)
        assert smoothed[0] < 3.0

    def test_steep_grade_preserved(self):
        yy = self._yy()
        elev = 50.0 - 0.02 * yy - 0.00002 * yy**2
        elev += np.random.default_rng(0).normal(0.0, 0.3, elev.shape)
        _, smoothed = self._smooth_profile(elev)
        assert smoothed[0] - smoothed[-1] > 25.0
        assert np.abs(np.diff(smoothed)).max() < 5.0


class TestBuildWaterTerraformOperations:
    def _flat_dem(self, z=3.0, n=50, extent=500.0):
        return _grid(np.full((n, n), z), extent)

    def test_level_body_flattens_to_dem_elevation(self):
        (op,) = build_water_terraform_operations(
            [WaterBody(box(-100, -100, 100, 100), "water")],
            self._flat_dem(),
            transition_buffer_m=10.0,
            dem_resolution_m=20.0,
        )
        assert isinstance(op, WaterFlattenOperation)
        assert op.surface is None
        assert op.reference_z == pytest.approx(3.0, abs=0.1)

    def test_shore_flat_margin_flattens_past_shoreline(self):
        (op,) = build_water_terraform_operations(
            [WaterBody(box(-100, -100, 100, 100), "water")],
            self._flat_dem(),
            transition_buffer_m=10.0,
            shore_flat_margin_m=5.0,
        )
        vertices = np.array(
            [[104.0, 0.0, 10.0], [110.0, 0.0, 10.0], [120.0, 0.0, 10.0]]
        )
        op.apply(vertices, lambda xy: np.zeros(len(xy)))
        assert vertices[0, 2] == pytest.approx(3.0)
        assert 3.0 < vertices[1, 2] < 10.0
        assert vertices[2, 2] == 10.0

    def test_linear_body_becomes_way_flatten_operation(self):
        cl = LineString([(0.0, -400.0), (0.0, 400.0)])
        (op,) = build_water_terraform_operations(
            [WaterBody(cl.buffer(5.0, cap_style="flat"), "water", cl, 5.0)],
            self._flat_dem(0.0),
            transition_buffer_m=10.0,
        )
        assert isinstance(op, WayFlattenOperation)

    def test_touching_level_bodies_share_one_operation(self):
        x = np.linspace(-500.0, 500.0, 100)
        elev = np.where(x < 0, 2.0, 4.0)[None, :].repeat(100, axis=0)
        bodies = [
            WaterBody(box(-100, -50, 0, 50), "water"),
            WaterBody(box(0, -50, 100, 50), "water"),
        ]
        (op,) = build_water_terraform_operations(
            bodies, _grid(elev, 500.0), transition_buffer_m=10.0, dem_resolution_m=10.0
        )
        assert op.surface is None

    def test_sloped_areal_river_follows_terrain(self):
        x = np.linspace(-200.0, 200.0, 41)
        y = np.linspace(-3000.0, 3000.0, 601)

        def terrain(v):
            return 0.05 * v + np.where(v > 1000.0, 20.0, 0.0)

        dem = xr.DataArray(
            np.tile(terrain(y)[:, None], (1, 41)),
            coords={"y": y, "x": x},
            dims=["y", "x"],
        )
        bodies = [
            WaterBody(box(-20, y0, 20, y0 + 1000), "water")
            for y0 in range(-2800, 2800, 1000)
        ]
        (op,) = build_water_terraform_operations(
            bodies, dem, transition_buffer_m=10.0, dem_resolution_m=10.0
        )
        along = np.arange(-2700.0, 2700.0, 10.0)
        surface_z = op.target_z(np.column_stack((np.zeros_like(along), along)))
        assert surface_z[-1] - surface_z[0] > 250.0
        assert np.abs(surface_z - terrain(along)).max() <= 5.0 + 1e-6

    def test_thin_water_skip_uses_true_width_of_diagonal_body(self):
        strip = affinity.rotate(box(-1, -100, 1, 100), 45)
        assert (
            build_water_terraform_operations(
                [WaterBody(strip, "water")],
                self._flat_dem(0.0),
                transition_buffer_m=10.0,
                thin_water_skip_m=5.0,
            )
            == []
        )


def test_water_cache_key_ignores_mesh_settings(make_minimal_config):
    def water_key(**kwargs):
        cfg = make_minimal_config(water=WaterConfig(**kwargs))
        return ResourceFingerprints._target_water(cfg)

    assert water_key() == water_key(sloped_water_smoothing_m=80.0)
    assert water_key() != water_key(exclude_intermittent=True)
    assert set(WaterConfig.model_fields) == (
        _WATER_SOURCE_FIELDS | _WATER_MESH_FIELDS | _WATER_TEXTURE_FIELDS
    )


def test_process_target_water_writes_sidecar(make_minimal_config, monkeypatch):
    from s2gos_generator.core.context import SceneResourceContext

    ctx = SceneResourceContext(
        make_minimal_config(water=WaterConfig(landcover_completion=False))
    )
    Path(str(ctx.data_dir)).mkdir(parents=True, exist_ok=True)
    canned = {"elements": [_way(SQUARE, natural="water")]}
    monkeypatch.setattr(
        "s2gos_generator.resources.water.fetch_osm_data", lambda *a, **k: canned
    )

    sidecar_path = process_target_water(ctx)
    assert ctx.assets.water_file == sidecar_path
    assert json.loads(Path(str(sidecar_path)).read_text())["version"] == 1
    assert len(ctx.water_bodies) == 1


def test_target_water_registered_only_when_enabled(make_minimal_config):
    from s2gos_generator.core.pipeline import SceneGenerationPipeline

    deps = SceneGenerationPipeline(make_minimal_config()).get_resource_dependencies()
    assert "target_water" not in deps

    deps = SceneGenerationPipeline(
        make_minimal_config(water=WaterConfig())
    ).get_resource_dependencies()
    assert set(deps["target_water"]) == {"target_dem", "target_landcover"}
    assert "target_water" in deps["target_mesh"]
    assert "target_water" in deps["target_texture"]
