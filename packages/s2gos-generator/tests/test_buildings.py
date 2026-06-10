"""Unit tests for the building pipeline helpers in ``resources/buildings.py``.

These cover the pure, deterministic pieces (height parsing, material
distribution, name sanitization) plus the flat-roof fallback that the pipeline
relies on when hip-roof construction fails or produces unusable geometry.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from s2gos_generator.resources.buildings import (
    _BuildingTask,
    _parse_height,
    _process_one_building,
    _resolve_material_distribution,
    _safe_name,
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
            "s2gos_generator.resources.buildings.build_hip_roof",
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
