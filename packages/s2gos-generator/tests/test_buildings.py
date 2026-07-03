"""Unit tests for the building mesh primitives in ``assets/buildings.py``.

These cover the pure, deterministic pieces (height parsing, material
distribution, name sanitization), the flat-roof fallback that the pipeline
relies on when hip-roof construction fails or produces unusable geometry, and
the ``build_meshes`` entry point that groups footprints into one mesh
per material.
"""

from __future__ import annotations

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
import pytest
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from s2gos_generator.core.config import BuildingsConfig
from s2gos_generator.processors.buildings.meshing import (
    _BuildingTask,
    _parse_height,
    _process_one_building,
    _resolve_material_distribution,
    _safe_name,
    build_meshes,
    quadkeys_for_bbox,
    select_tile_files,
)


class TestParseHeight:
    def test_numeric_value_used_directly(self):
        assert _parse_height(12.5, story_height=3.0, default=3.0) == (12.5, True)

    def test_numeric_string_used_directly(self):
        # Plain numeric strings are not taxonomy codes -> fall back to default.
        assert _parse_height("not-a-height", story_height=3.0, default=7.0) == (
            7.0,
            False,
        )

    def test_non_positive_numeric_falls_back(self):
        assert _parse_height(0.0, story_height=3.0, default=4.0) == (4.0, False)
        assert _parse_height(-5.0, story_height=3.0, default=4.0) == (4.0, False)

    def test_bool_is_not_treated_as_numeric(self):
        # bool is a subclass of int; it must not be read as a height.
        assert _parse_height(True, story_height=3.0, default=4.0) == (4.0, False)

    @pytest.mark.parametrize("value", [None, "", "nan", float("nan")])
    def test_missing_values_fall_back(self, value):
        assert _parse_height(value, story_height=3.0, default=8.0) == (8.0, False)

    def test_taxonomy_explicit_height(self):
        # HHT: is an explicit height in meters, independent of story_height.
        assert _parse_height("HHT:21.0", story_height=3.0, default=4.0) == (21.0, True)

    def test_taxonomy_story_count(self):
        # H: is a story count multiplied by story_height.
        assert _parse_height("H:4", story_height=3.0, default=4.0) == (12.0, True)

    def test_taxonomy_story_range_averaged(self):
        # HBET: is a story range; midpoint * story_height.
        assert _parse_height("HBET:2-4", story_height=3.0, default=4.0) == (9.0, True)

    def test_taxonomy_prefers_explicit_height_over_story_count(self):
        assert _parse_height("HHT:30+H:4", story_height=3.0, default=4.0) == (
            30.0,
            True,
        )

    def test_unparseable_taxonomy_falls_back(self):
        assert _parse_height("HHT:abc", story_height=3.0, default=5.0) == (5.0, False)


class TestResolveMaterialDistribution:
    def test_single_string(self):
        names, weights = _resolve_material_distribution("brick")
        assert names == ["brick"]
        assert weights is None

    def test_dict_weights_normalized(self):
        names, weights = _resolve_material_distribution({"brick": 3.0, "glass": 1.0})
        assert names == ["brick", "glass"]
        np.testing.assert_allclose(weights, [0.75, 0.25])
        assert weights.sum() == pytest.approx(1.0)

    def test_dict_preserves_key_order(self):
        names, _ = _resolve_material_distribution({"c": 1.0, "a": 1.0, "b": 1.0})
        assert names == ["c", "a", "b"]


class TestSafeName:
    def test_spaces_and_illegal_chars_replaced(self):
        assert _safe_name("my building/name 1") == "my_building_name_1"

    def test_legal_chars_preserved(self):
        assert _safe_name("Block-2.A_v1") == "Block-2.A_v1"

    def test_run_of_illegal_chars_collapses(self):
        assert _safe_name("a@@@b") == "a_b"


def _square(size: float = 8.0) -> Polygon:
    return Polygon([(0, 0), (size, 0), (size, size), (0, size)])


class TestProcessOneBuildingFallback:
    def _task(self, **overrides) -> _BuildingTask:
        defaults = dict(
            idx=0,
            geom=_square(),
            height_m=10.0,
            base_z=100.0,
            skirt_m=0.5,
            material_name="brick",
            pitched=False,
            pitch_deg=35.0,
            target_roof_height=3.0,
        )
        defaults.update(overrides)
        return _BuildingTask(**defaults)

    def test_flat_building_builds_wall_only(self):
        result = _process_one_building(self._task(pitched=False))
        assert isinstance(result.wall_mesh, trimesh.Trimesh)
        assert result.roof_mesh is None
        assert result.pitched_succeeded is False

    def test_falls_back_to_flat_when_hip_roof_fails(self, monkeypatch):
        # Force build_hip_roof to fail; the pipeline must still emit a
        # full-height flat building rather than dropping it.
        monkeypatch.setattr(
            "s2gos_generator.processors.buildings.meshing.build_hip_roof",
            lambda *a, **k: None,
        )
        result = _process_one_building(self._task(pitched=True))
        assert isinstance(result.wall_mesh, trimesh.Trimesh)
        assert result.roof_mesh is None
        assert result.pitched_attempted is True
        assert result.pitched_succeeded is False
        # Flat fallback uses the full height, so the wall reaches base_z + height.
        assert result.wall_mesh.bounds[1][2] == pytest.approx(110.0, abs=1e-6)

    def test_pitched_success_produces_roof(self):
        result = _process_one_building(self._task(pitched=True))
        # On a simple square the skeleton succeeds, so we get a roof mesh.
        assert isinstance(result.roof_mesh, trimesh.Trimesh)
        assert result.pitched_succeeded is True

    def test_empty_geometry_yields_no_mesh(self):
        result = _process_one_building(self._task(geom=Polygon()))
        assert result.wall_mesh is None

    def test_multipolygon_handled(self):
        geom = MultiPolygon([_square(4.0), Polygon([(10, 10), (14, 10), (14, 14)])])
        result = _process_one_building(self._task(geom=geom, pitched=False))
        assert isinstance(result.wall_mesh, trimesh.Trimesh)


def _footprints_gdf(n: int = 3, size: float = 8.0, gap: float = 20.0):
    """A few square footprints in scene-local meters, laid out in a row."""
    geoms = [
        Polygon(
            [
                (i * gap, 0),
                (i * gap + size, 0),
                (i * gap + size, size),
                (i * gap, size),
            ]
        )
        for i in range(n)
    ]
    return gpd.GeoDataFrame({"height": [10.0] * n}, geometry=geoms)


def _flat_elev_fn(z: float = 100.0):
    return lambda xy: np.full(len(xy), z)


class TestBuildMeshes:
    def test_single_material_groups_into_one_mesh(self):
        gdf = _footprints_gdf(n=3)
        cfg = BuildingsConfig(material="concrete")
        result = build_meshes(gdf, _flat_elev_fn(), cfg)

        assert result.single_material is True
        assert list(result.material_meshes) == ["concrete"]
        assert isinstance(result.material_meshes["concrete"], trimesh.Trimesh)
        assert result.roof_mesh is None
        assert result.stats.total == 3
        assert result.stats.per_material_counts == {"concrete": 3}

    def test_weighted_materials_are_deterministic_with_seed(self):
        gdf = _footprints_gdf(n=20)
        cfg = BuildingsConfig(
            material={"brick": 1.0, "glass": 1.0},
            material_seed=42,
        )
        a = build_meshes(gdf, _flat_elev_fn(), cfg).stats.per_material_counts
        b = build_meshes(gdf, _flat_elev_fn(), cfg).stats.per_material_counts

        assert a == b  # fixed seed -> reproducible assignment
        assert sum(a.values()) == 20
        assert set(a).issubset({"brick", "glass"})

    def test_pitched_roof_produces_roof_mesh(self):
        gdf = _footprints_gdf(n=2)
        cfg = BuildingsConfig(
            material="concrete",
            pitched_roof_proportion=1.0,
            roof_seed=0,
        )
        result = build_meshes(gdf, _flat_elev_fn(), cfg)

        assert isinstance(result.roof_mesh, trimesh.Trimesh)
        assert result.stats.pitched == 2

    def test_empty_input_yields_no_meshes(self):
        gdf = gpd.GeoDataFrame({"height": []}, geometry=[])
        cfg = BuildingsConfig(material="concrete")
        result = build_meshes(gdf, _flat_elev_fn(), cfg)

        assert result.stats.total == 0
        assert result.material_meshes == {}
        assert result.roof_mesh is None


def test_quadkeys_for_bbox_round_trips_to_containing_tile():
    """A point bbox selects exactly the one tile whose bounds contain it."""
    lon, lat, zoom = 10.40, 43.70, 6
    (qk,) = quadkeys_for_bbox((lon, lat, lon, lat), zoom)
    b = mercantile.bounds(mercantile.quadkey_to_tile(qk))
    assert b.west <= lon <= b.east and b.south <= lat <= b.north


def test_select_tile_files_returns_overlapping_present_tile(tmp_path):
    """Only the tile overlapping the AOI and present on disk is returned."""
    bbox = (10.40, 43.70, 10.41, 43.71)
    (inside,) = quadkeys_for_bbox(bbox, zoom=6)
    (outside,) = quadkeys_for_bbox((-120.0, 35.0, -120.0, 35.0), zoom=6)

    pd.DataFrame(
        {
            "quadkey": [inside, outside],
            "filename": [f"{inside}.gpkg", f"{outside}.gpkg"],
        }
    ).to_csv(tmp_path / "index.csv", index=False)
    (tmp_path / f"{inside}.gpkg").touch()
    (tmp_path / f"{outside}.gpkg").touch()

    assert select_tile_files(tmp_path, bbox, "index.csv") == [
        tmp_path / f"{inside}.gpkg"
    ]
