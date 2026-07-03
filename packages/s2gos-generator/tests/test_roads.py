"""Tests for road support: OSM parsing, width/material resolution, the road
sidecar round-trip, and conditional pipeline wiring."""

import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box, mapping

from s2gos_generator.core.config.roads import HighwayOverride, RoadsConfig
from s2gos_generator.processors.roads import (
    Road,
    _get_road_material,
    _get_road_width,
    _parse_osm_width,
    fetch_osm_data,
    parse_roads,
    roads_from_sidecar,
    roads_to_sidecar,
)
from s2gos_generator.resources.roads import process_target_roads

TABLE = RoadsConfig.ROAD_TYPE_TABLE
SURFACES = RoadsConfig.DEFAULT_SURFACE_MATERIALS


class TestParseOsmWidth:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("5.5", 5.5),
            ("5 m", 5.0),
            ("5.5m", 5.5),
            ("wide", None),
            ("", None),
        ],
    )
    def test_parse_osm_width(self, raw, expected):
        assert _parse_osm_width(raw) == expected


class TestRoadWidth:
    """`_get_road_width` resolution order: override total -> OSM width -> lane math."""

    @staticmethod
    def _width(tags, hw_type, overrides=None):
        return _get_road_width(tags, hw_type, overrides or {}, TABLE, 3.0, 0.5)

    def test_override_total_width_wins_over_everything(self):
        overrides = {"residential": HighwayOverride(total_width_m=12.0)}
        # OSM width and lanes are present but must be ignored.
        assert (
            self._width({"width": "99", "lanes": "4"}, "residential", overrides) == 12.0
        )

    def test_osm_width_tag_used_when_no_override(self):
        assert self._width({"width": "7.5"}, "residential") == 7.5

    def test_lane_math_uses_table_lane_width_and_shoulder(self):
        # residential: lane_width 3.0; explicit 2 lanes -> 2*3.0 + 2*0.5 = 7.0.
        assert self._width({"lanes": "2"}, "residential") == pytest.approx(7.0)

    def test_non_integer_lanes_tag_is_ignored(self):
        # "abc" is not an int -> fall back to service's table lane_count (1):
        # 1 * 2.5 + 2*0.5 = 3.5 (and crucially does not raise).
        assert self._width({"lanes": "abc"}, "service") == pytest.approx(3.5)

    def test_oneway_falls_back_to_one_lane_for_unknown_type(self):
        # Unknown highway type, no lanes tag, no table defaults.
        oneway = self._width({"oneway": "yes"}, "unknown_type")
        twoway = self._width({}, "unknown_type")
        assert oneway == pytest.approx(1 * 3.0 + 1.0)  # 1 lane fallback
        assert twoway == pytest.approx(2 * 3.0 + 1.0)  # 2 lane fallback


class TestRoadMaterial:
    """`_get_road_material` resolution order: OSM surface -> override -> table -> default."""

    @staticmethod
    def _material(tags, hw_type, overrides=None, default="asphalt"):
        return _get_road_material(
            tags, hw_type, overrides or {}, TABLE, SURFACES, default
        )

    @pytest.mark.parametrize(
        "surface,expected",
        [("gravel", "gravel_road"), ("dirt", "baresoil")],
    )
    def test_known_surface_tag_is_mapped(self, surface, expected):
        assert self._material({"surface": surface}, "residential") == expected

    def test_unknown_surface_falls_back_to_default_material(self):
        assert (
            self._material({"surface": "moonrock"}, "residential", default="asphalt")
            == "asphalt"
        )

    def test_override_default_material_used_when_no_surface(self):
        overrides = {"residential": HighwayOverride(default_material="concrete")}
        assert self._material({}, "residential", overrides) == "concrete"

    def test_type_table_default_used_when_no_surface_or_override(self):
        assert self._material({}, "track") == "gravel_road"


class TestParseRoads:
    """`parse_roads`: OSM ways -> Road segments in scene coords, clipped to bounds."""

    class _CoordStub:
        # 0.001 deg -> 100 m, centred on (45, 15); good enough for a ±5 km scene.
        def latlon_to_scene(self, lat, lon):
            return ((lon - 15.0) * 100000.0, (lat - 45.0) * 100000.0)

    BOUNDS = box(-5000, -5000, 5000, 5000)

    @staticmethod
    def _way(nodes, **tags):
        return {
            "type": "way",
            "tags": tags,
            "geometry": [{"lat": la, "lon": lo} for la, lo in nodes],
        }

    def _parse(self, *elements, cfg=None):
        return parse_roads(
            {"elements": list(elements)},
            cfg or RoadsConfig(),
            self._CoordStub(),
            self.BOUNDS,
        )

    def test_basic_way_becomes_one_road(self):
        roads = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], highway="residential")
        )
        assert len(roads) == 1
        assert roads[0].material == "asphalt"  # residential table default
        assert roads[0].width > 0
        coords = list(roads[0].centerline.coords)
        assert coords[0] == pytest.approx((0.0, 0.0))
        assert coords[-1] == pytest.approx((0.0, 200.0))

    @pytest.mark.parametrize(
        "element",
        [
            {"type": "node", "tags": {"highway": "residential"}},  # not a way
            {
                "type": "way",
                "tags": {},
                "geometry": [{"lat": 45.0, "lon": 15.0}],
            },  # no highway
            {
                "type": "way",
                "tags": {"highway": "residential"},
                "geometry": [{"lat": 45.0, "lon": 15.0}],
            },  # < 2 nodes
        ],
        ids=["not-a-way", "no-highway-tag", "too-few-nodes"],
    )
    def test_skips_invalid_elements(self, element):
        assert self._parse(element) == []

    def test_highway_types_filter_excludes_unlisted(self):
        cfg = RoadsConfig(highway_types=["motorway"])
        roads = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], highway="residential"), cfg=cfg
        )
        assert roads == []

    def test_centerline_clipped_to_scene_bounds(self):
        roads = self._parse(self._way([(45.0, 15.0), (45.1, 15.0)], highway="primary"))
        assert len(roads) == 1
        ymin, ymax = roads[0].centerline.bounds[1], roads[0].centerline.bounds[3]
        assert ymin == pytest.approx(0.0)
        assert ymax == pytest.approx(5000.0)

    def test_reentrant_way_decomposes_into_multiple_roads(self):
        # A way that leaves the AOI over the top and re-enters yields a
        # MultiLineString on clip; it must become one Road per component.
        nodes = [(45.0, 14.96), (45.06, 14.96), (45.06, 15.04), (45.0, 15.04)]
        roads = self._parse(self._way(nodes, highway="secondary"))
        assert len(roads) == 2
        # Both components inherit the same width and material.
        assert {r.material for r in roads} == {"asphalt"}
        assert len({r.width for r in roads}) == 1


class TestRoadsConfig:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(source="file", file_path=None), "file_path is required"),
            (
                dict(source="file", file_path=Path("/no/such/road_file.json")),
                "not found",
            ),
        ],
        ids=["missing-path", "nonexistent-path"],
    )
    def test_file_source_validation_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RoadsConfig(**kwargs)


class TestFetchOsmData:
    def test_malformed_json_file_returns_none(self, tmp_path):
        bad = tmp_path / "roads.json"
        bad.write_text("not json{")
        cfg = RoadsConfig(source="file", file_path=bad)
        assert fetch_osm_data(cfg, 0.0, 0.0, 1.0, 1.0) is None


def _write_road_sidecar(path, *, version=1):
    path.write_text(
        json.dumps(
            {
                "version": version,
                "road_layers": [
                    {
                        "material_name": "asphalt",
                        "roads": [
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


class TestRoadSidecar:
    """Round-trip: producing the sidecar (process_target_roads) and reading it back."""

    def test_ctx_roads_reads_back_sidecar(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        assert ctx.roads == []  # no sidecar -> empty

        sidecar = tmp_path / "roads.json"
        _write_road_sidecar(sidecar)
        ctx.assets.roads_file = sidecar
        ctx._roads = None  # reset lazy cache

        roads = ctx.roads
        assert len(roads) == 1
        assert isinstance(roads[0], Road)
        assert roads[0].material == "asphalt"
        assert roads[0].width == 7.0
        assert list(roads[0].centerline.coords) == [(0.0, 0.0), (0.0, 100.0)]

    def test_ctx_roads_rejects_unknown_version(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        sidecar = tmp_path / "roads.json"
        _write_road_sidecar(sidecar, version=99)
        ctx.assets.roads_file = sidecar
        assert ctx.roads == []

    def test_process_target_roads_writes_grouped_sidecar(
        self, make_minimal_config, monkeypatch
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config(roads=RoadsConfig(enabled=True)))
        Path(str(ctx.data_dir)).mkdir(parents=True, exist_ok=True)

        canned = {
            "elements": [
                {
                    "type": "way",
                    "tags": {"highway": "residential", "surface": "asphalt"},
                    "geometry": [
                        {"lat": 45.0, "lon": 15.0},
                        {"lat": 45.002, "lon": 15.0},
                    ],
                },
                {
                    "type": "way",
                    "tags": {"highway": "track", "surface": "gravel"},
                    "geometry": [
                        {"lat": 45.0, "lon": 15.0},
                        {"lat": 45.0, "lon": 15.002},
                    ],
                },
            ]
        }
        monkeypatch.setattr(
            "s2gos_generator.resources.roads.fetch_osm_data",
            lambda *a, **k: canned,
        )

        sidecar_path = process_target_roads(ctx)
        assert sidecar_path is not None
        assert ctx.assets.roads_file == sidecar_path

        data = json.loads(Path(str(sidecar_path)).read_text())
        assert data["version"] == 1
        materials = [layer["material_name"] for layer in data["road_layers"]]
        assert materials == ["asphalt", "gravel_road"]


class TestRoadsWiring:
    """`target_roads` is registered only when roads are enabled."""

    def test_target_roads_registered_only_when_enabled(self, make_minimal_config):
        from s2gos_generator.core.pipeline import SceneGenerationPipeline

        without = SceneGenerationPipeline(make_minimal_config())
        deps_without = without.get_resource_dependencies()
        assert "target_roads" not in deps_without
        assert "target_roads" not in deps_without["target_texture"]

        with_roads = SceneGenerationPipeline(
            make_minimal_config(roads=RoadsConfig(enabled=True))
        )
        deps_with = with_roads.get_resource_dependencies()
        assert deps_with["target_roads"] == []
        # Resolved as an optional dependency of the mesh and texture steps.
        assert "target_roads" in deps_with["target_mesh"]
        assert "target_roads" in deps_with["target_texture"]


class TestRoadsSidecar:
    """`roads_to_sidecar` / `roads_from_sidecar` round-trip and versioning."""

    def test_roundtrip_preserves_segments(self):
        roads = [
            Road(LineString([(0, 0), (10, 0)]), 4.0, "asphalt"),
            Road(LineString([(0, 5), (5, 5)]), 3.0, "gravel_road"),
        ]
        restored = roads_from_sidecar(roads_to_sidecar(roads))
        assert [(r.material, r.width, list(r.centerline.coords)) for r in restored] == [
            (r.material, r.width, list(r.centerline.coords)) for r in roads
        ]

    def test_unknown_version_returns_empty(self):
        assert roads_from_sidecar({"version": 2, "road_layers": []}) == []
