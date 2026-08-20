"""Tests for way support: OSM parsing, width/material resolution, the ways
sidecar round-trip, and conditional pipeline wiring."""

import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box, mapping

from s2gos_generator.core.config.ways import RailwayOverride, RoadOverride, WaysConfig
from s2gos_generator.processors.ways import (
    Way,
    _get_railway_material,
    _get_railway_width,
    _get_road_material,
    _get_road_width,
    _parse_osm_width,
    fetch_osm_data,
    parse_ways,
    ways_from_sidecar,
    ways_to_sidecar,
)
from s2gos_generator.resources.ways import process_target_ways

TABLE = WaysConfig.ROAD_TYPE_TABLE
RAIL_TABLE = WaysConfig.RAILWAY_TYPE_TABLE
SURFACES = WaysConfig.DEFAULT_SURFACE_MATERIALS


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
        overrides = {"residential": RoadOverride(total_width_m=12.0)}
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

    def test_override_lane_count_used_over_table_default(self):
        # residential table lane_width_m=3.0; override to 4 lanes -> 4*3.0 + 2*0.5 = 13.0.
        overrides = {"residential": RoadOverride(lane_count=4)}
        assert self._width({}, "residential", overrides) == pytest.approx(13.0)

    def test_override_lane_width_used_over_table_default(self):
        # residential table lane_count=2; override lane_width to 4.0 -> 2*4.0 + 2*0.5 = 9.0.
        overrides = {"residential": RoadOverride(lane_width_m=4.0)}
        assert self._width({}, "residential", overrides) == pytest.approx(9.0)

    def test_osm_lanes_tag_wins_over_override_lane_count(self):
        # An explicit OSM lanes tag takes priority over override.lane_count.
        overrides = {"residential": RoadOverride(lane_count=4)}
        assert self._width({"lanes": "2"}, "residential", overrides) == pytest.approx(7.0)


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
        overrides = {"residential": RoadOverride(default_material="concrete")}
        assert self._material({}, "residential", overrides) == "concrete"

    def test_type_table_default_used_when_no_surface_or_override(self):
        assert self._material({}, "track") == "gravel_road"


class TestRailwayWidth:
    """`_get_railway_width` resolution order: override total -> OSM width -> track math."""

    @staticmethod
    def _width(tags, rail_type, overrides=None):
        return _get_railway_width(tags, rail_type, overrides or {}, RAIL_TABLE, 3.0)

    def test_override_total_width_wins_over_everything(self):
        overrides = {"rail": RailwayOverride(total_width_m=15.0)}
        # OSM width and tracks are present but must be ignored.
        assert self._width({"width": "99", "tracks": "4"}, "rail", overrides) == 15.0

    def test_osm_width_tag_used_when_no_override(self):
        assert self._width({"width": "6.5"}, "rail") == 6.5

    def test_osm_tracks_tag_used_over_table_default(self):
        # rail table: track_count=1, track_width_m=4.5; explicit 2 tracks -> 2*4.5 = 9.0.
        assert self._width({"tracks": "2"}, "rail") == pytest.approx(9.0)

    def test_non_integer_tracks_tag_is_ignored(self):
        # "abc" is not an int -> fall back to the table track_count (1):
        # 1 * 4.5 = 4.5 (and crucially does not raise).
        assert self._width({"tracks": "abc"}, "rail") == pytest.approx(4.5)

    def test_override_track_count_used_when_no_tracks_tag(self):
        # 3 tracks (override) * table track_width_m (4.5) = 13.5.
        overrides = {"rail": RailwayOverride(track_count=3)}
        assert self._width({}, "rail", overrides) == pytest.approx(13.5)

    def test_override_track_width_used_when_no_override_count_or_tag(self):
        # table track_count (1) * override track_width_m (6.0) = 6.0.
        overrides = {"rail": RailwayOverride(track_width_m=6.0)}
        assert self._width({}, "rail", overrides) == pytest.approx(6.0)

    def test_unknown_type_falls_back_to_default_track_width(self):
        # Unknown rail type: no table defaults, tracks fallback=1, default_track_width_m=3.0.
        assert self._width({}, "unknown_type") == pytest.approx(3.0)


class TestRailwayMaterial:
    """`_get_railway_material` resolution order: OSM surface -> override -> table -> default.

    Mirrors `TestRoadMaterial` -- railway material resolution now consults an OSM
    ``surface`` tag exactly like roads do, reusing the same `DEFAULT_SURFACE_MATERIALS`
    table.
    """

    @staticmethod
    def _material(tags, rail_type, overrides=None, default="gravel_road"):
        return _get_railway_material(
            tags, rail_type, overrides or {}, RAIL_TABLE, SURFACES, default
        )

    def test_known_surface_tag_is_mapped(self):
        assert self._material({"surface": "concrete"}, "rail") == "concrete"

    def test_unknown_surface_falls_back_to_default_material(self):
        assert (
            self._material({"surface": "moonrock"}, "rail", default="gravel_road")
            == "gravel_road"
        )

    def test_surface_tag_wins_over_override(self):
        # Matches _get_road_material's priority: an OSM surface tag wins even when a
        # config override is also set for that type.
        overrides = {"rail": RailwayOverride(default_material="concrete")}
        assert self._material({"surface": "gravel"}, "rail", overrides) == "gravel_road"

    def test_override_default_material_used_over_table(self):
        overrides = {"rail": RailwayOverride(default_material="concrete")}
        assert self._material({}, "rail", overrides) == "concrete"

    def test_type_table_default_used_when_no_override(self):
        assert self._material({}, "tram") == "asphalt"  # tram table default

    def test_unknown_type_falls_back_to_default_material(self):
        assert self._material({}, "unknown_type", default="gravel_road") == "gravel_road"


class TestParseWays:
    """`parse_ways`: OSM ways -> Way segments in scene coords, clipped to bounds."""

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
        return parse_ways(
            {"elements": list(elements)},
            cfg or WaysConfig(),
            self._CoordStub(),
            self.BOUNDS,
        )

    def test_basic_way_becomes_one_way(self):
        ways = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], highway="residential")
        )
        assert len(ways) == 1
        assert ways[0].material == "asphalt"  # residential table default
        assert ways[0].width > 0
        assert ways[0].kind == "road"
        coords = list(ways[0].centerline.coords)
        assert coords[0] == pytest.approx((0.0, 0.0))
        assert coords[-1] == pytest.approx((0.0, 200.0))

    def test_railway_way_becomes_one_way(self):
        # railway=rail with no explicit tracks tag: 1 track * 4.5m default width.
        ways = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], railway="rail")
        )
        assert len(ways) == 1
        assert ways[0].material == "gravel_road"  # rail table default
        assert ways[0].width == pytest.approx(4.5)
        assert ways[0].kind == "railway"
        coords = list(ways[0].centerline.coords)
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

    def test_road_types_filter_excludes_unlisted(self):
        cfg = WaysConfig(road_types=["motorway"])
        ways = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], highway="residential"), cfg=cfg
        )
        assert ways == []

    def test_railway_types_filter_excludes_unlisted(self):
        cfg = WaysConfig(railway_types=["tram"])
        ways = self._parse(
            self._way([(45.0, 15.0), (45.002, 15.0)], railway="rail"), cfg=cfg
        )
        assert ways == []

    def test_centerline_clipped_to_scene_bounds(self):
        ways = self._parse(self._way([(45.0, 15.0), (45.1, 15.0)], highway="primary"))
        assert len(ways) == 1
        ymin, ymax = ways[0].centerline.bounds[1], ways[0].centerline.bounds[3]
        assert ymin == pytest.approx(0.0)
        assert ymax == pytest.approx(5000.0)

    def test_reentrant_way_decomposes_into_multiple_ways(self):
        # A way that leaves the AOI over the top and re-enters yields a
        # MultiLineString on clip; it must become one Way per component.
        nodes = [(45.0, 14.96), (45.06, 14.96), (45.06, 15.04), (45.0, 15.04)]
        ways = self._parse(self._way(nodes, highway="secondary"))
        assert len(ways) == 2
        # Both components inherit the same width and material.
        assert {w.material for w in ways} == {"asphalt"}
        assert len({w.width for w in ways}) == 1


class TestWaysConfig:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(source="file", file_path=None), "file_path is required"),
            (
                dict(source="file", file_path=Path("/no/such/way_file.json")),
                "not found",
            ),
        ],
        ids=["missing-path", "nonexistent-path"],
    )
    def test_file_source_validation_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WaysConfig(**kwargs)


class TestFetchOsmData:
    def test_malformed_json_file_returns_none(self, tmp_path):
        bad = tmp_path / "ways.json"
        bad.write_text("not json{")
        cfg = WaysConfig(source="file", file_path=bad)
        assert fetch_osm_data(cfg, 0.0, 0.0, 1.0, 1.0) is None


def _write_ways_sidecar(path, *, version=1):
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


class TestWaysSidecarIntegration:
    """Round-trip: producing the sidecar (process_target_ways) and reading it back."""

    def test_ctx_ways_reads_back_sidecar(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        assert ctx.ways == []  # no sidecar -> empty

        sidecar = tmp_path / "ways.json"
        _write_ways_sidecar(sidecar)
        ctx.assets.ways_file = sidecar
        ctx._ways = None  # reset lazy cache

        ways = ctx.ways
        assert len(ways) == 1
        assert isinstance(ways[0], Way)
        assert ways[0].material == "asphalt"
        assert ways[0].width == 7.0
        assert list(ways[0].centerline.coords) == [(0.0, 0.0), (0.0, 100.0)]

    def test_ctx_ways_rejects_unknown_version(self, make_minimal_config, tmp_path):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config())
        sidecar = tmp_path / "ways.json"
        _write_ways_sidecar(sidecar, version=99)
        ctx.assets.ways_file = sidecar
        assert ctx.ways == []

    def test_process_target_ways_writes_grouped_sidecar(
        self, make_minimal_config, monkeypatch
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(make_minimal_config(ways=WaysConfig(enabled=True)))
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
            "s2gos_generator.resources.ways.fetch_osm_data",
            lambda *a, **k: canned,
        )

        sidecar_path = process_target_ways(ctx)
        assert sidecar_path is not None
        assert ctx.assets.ways_file == sidecar_path

        data = json.loads(Path(str(sidecar_path)).read_text())
        assert data["version"] == 2
        materials = [layer["material_name"] for layer in data["way_layers"]]
        assert materials == ["asphalt", "gravel_road"]


class TestWaysWiring:
    """`target_ways` is registered only when ways are enabled."""

    def test_target_ways_registered_only_when_enabled(self, make_minimal_config):
        from s2gos_generator.core.pipeline import SceneGenerationPipeline

        without = SceneGenerationPipeline(make_minimal_config())
        deps_without = without.get_resource_dependencies()
        assert "target_ways" not in deps_without
        assert "target_ways" not in deps_without["target_texture"]

        with_ways = SceneGenerationPipeline(
            make_minimal_config(ways=WaysConfig(enabled=True))
        )
        deps_with = with_ways.get_resource_dependencies()
        assert deps_with["target_ways"] == []
        # Resolved as an optional dependency of the mesh and texture steps.
        assert "target_ways" in deps_with["target_mesh"]
        assert "target_ways" in deps_with["target_texture"]


class TestWaysSidecar:
    """`ways_to_sidecar` / `ways_from_sidecar` round-trip and versioning."""

    def test_roundtrip_preserves_segments(self):
        ways = [
            Way(LineString([(0, 0), (10, 0)]), 4.0, "asphalt", kind="road"),
            Way(LineString([(0, 5), (5, 5)]), 3.0, "gravel_road", kind="railway"),
        ]
        sidecar = ways_to_sidecar(ways)
        assert sidecar["version"] == 2
        restored = ways_from_sidecar(sidecar)
        assert [
            (w.material, w.width, w.kind, list(w.centerline.coords)) for w in restored
        ] == [(w.material, w.width, w.kind, list(w.centerline.coords)) for w in ways]

    def test_unknown_version_returns_empty(self):
        assert ways_from_sidecar({"version": 3, "way_layers": []}) == []

    def test_version_1_sidecar_without_kind_defaults_to_unknown(self):
        # Version 1 predates the "kind" field; legacy files must still load.
        legacy = {
            "version": 1,
            "way_layers": [
                {
                    "material_name": "asphalt",
                    "ways": [
                        {
                            "centerline": mapping(LineString([(0.0, 0.0), (0.0, 100.0)])),
                            "width": 7.0,
                        }
                    ],
                }
            ],
        }
        restored = ways_from_sidecar(legacy)
        assert len(restored) == 1
        assert restored[0].kind == "unknown"
