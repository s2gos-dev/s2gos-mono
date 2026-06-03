"""Unit tests for the pipeline caching system (cache.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from s2gos_utils.io.paths import open_file
from upath import UPath

from s2gos_generator.core.cache import (
    MANIFEST_VERSION,
    CachedDAGExecutor,
    CacheManifest,
    _validate_user_asset_files,
    _validate_vegetation_files,
    capture_asset_paths,
    restore_context,
)
from s2gos_generator.core.fingerprints import (
    ResourceFingerprints,
    _stable_hash,
    compute_all_hashes,
)
from s2gos_generator.core.resource_registry import ResourceRegistry

# ──────────────────────────────────────────────────────────────────────────────
# Helpers & Fixtures
# ──────────────────────────────────────────────────────────────────────────────


class _StubAssets:
    """Minimal stub for SceneAssets."""

    def __init__(self):
        for attr in [
            "dem_file",
            "buffer_dem_file",
            "landcover_file",
            "buffer_landcover_file",
            "background_landcover_file",
            "mesh_file",
            "buffer_mesh_file",
            "selection_texture_file",
            "preview_texture_file",
            "buffer_selection_texture_file",
            "buffer_preview_texture_file",
            "background_selection_texture_file",
            "background_preview_texture_file",
            "vegetation_objects_file",
            "user_assets_file",
            "hamster_paths_file",
            "config_file",
            "scene_description_file",
        ]:
            setattr(self, attr, None)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _make_ctx(tmp_path: Path, config: Any = None) -> Any:
    return SimpleNamespace(
        output_dir=UPath(tmp_path),
        assets=_StubAssets(),
        dependency_outputs={},
        config=config or SimpleNamespace(),
    )


@pytest.fixture
def minimal_registry() -> ResourceRegistry:
    r = ResourceRegistry()
    _noop = lambda ctx: None  # noqa
    r.register("target_dem", [], _noop)
    r.register("target_landcover", [], _noop)
    r.register("target_mesh", ["target_dem"], _noop)
    r.register("target_texture", ["target_landcover"], _noop)
    r.register("scene_description", ["target_mesh", "target_texture"], _noop)
    return r


@pytest.fixture
def minimal_config() -> Any:
    return SimpleNamespace(
        location=SimpleNamespace(
            center_lat=45.0,
            center_lon=15.0,
            aoi_size_km=10.0,
            model_dump=lambda: {
                "center_lat": 45.0,
                "center_lon": 15.0,
                "aoi_size_km": 10.0,
            },
        ),
        data_sources=SimpleNamespace(
            dem=SimpleNamespace(model_dump=lambda: {"type": "indexed_geotiff"}),
            landcover=SimpleNamespace(model_dump=lambda: {"type": "indexed_geotiff"}),
        ),
        processing=SimpleNamespace(
            dem_fillna_value=0.0,
            flatten_dem=False,
            handle_dem_nans=True,
            generate_texture_preview=True,
        ),
        dem_resolution_m=30.0,
        landcover_resolution_m=30.0,
        texture_resolution_m=10,
        buffer=None,
        background=None,
        snow=None,
        material_regions=[],
        user_assets=[],
        xml_scenes=[],
        vegetation_exclusion_zones=[],
        vegetation_placement=None,
        hamster=None,
        roads=None,
    )


class TestHashingAndFingerprints:
    def test_stable_hash(self):
        d1, d2 = {"x": 1, "y": 2}, {"y": 2, "x": 1}
        assert _stable_hash(d1) == _stable_hash(d2)  # Order independent
        assert _stable_hash({"a": 1}) != _stable_hash({"a": 2})  # Value dependent
        assert len(_stable_hash({"k": "v"})) == 16  # Fixed length

    def test_resource_fingerprints(self, minimal_config):
        cfg = minimal_config
        assert ResourceFingerprints.get("nonexistent", cfg) == {}

        fp_dem = ResourceFingerprints.get("target_dem", cfg)
        assert all(
            k in fp_dem for k in ("center_lat", "dem_resolution_m", "dem_fillna_value")
        )

        fp_texture = ResourceFingerprints.get("target_texture", cfg)
        assert fp_texture.get("snow") is None
        assert "texture_resolution_m" in fp_texture
        cfg.snow = SimpleNamespace(
            model_dump=lambda: {"season_month": "june", "material_index": 6}
        )
        assert (
            ResourceFingerprints.get("target_texture", cfg)["snow"]["season_month"]
            == "june"
        )
        assert (
            ResourceFingerprints.get("target_texture", cfg)["texture_resolution_m"]
            is not None
        )
        assert "snow" not in ResourceFingerprints.get("background_texture", cfg)

        cfg.buffer = SimpleNamespace(size_km=60.0, resolution_m=100.0)
        cfg.material_regions = [
            SimpleNamespace(
                applies_to=["target"],
                model_dump=lambda: {"region_id": "r1", "applies_to": ["target"]},
            ),
            SimpleNamespace(
                applies_to=["buffer"],
                model_dump=lambda: {"region_id": "r2", "applies_to": ["buffer"]},
            ),
        ]

        assert ResourceFingerprints.get("buffer_dem", cfg)["buffer_size_km"] == 60.0

        fp_buf_tex = ResourceFingerprints.get("buffer_texture", cfg)
        assert len(fp_buf_tex["material_regions"]) == 1
        assert fp_buf_tex["material_regions"][0]["region_id"] == "r2"

    def test_compute_all_hashes(self, minimal_config, minimal_registry):
        h1 = compute_all_hashes(minimal_config, minimal_registry)
        assert len(h1["target_dem"]) == 16
        assert h1 == compute_all_hashes(minimal_config, minimal_registry)

        minimal_config.location.center_lat = 50.0
        h2 = compute_all_hashes(minimal_config, minimal_registry)
        assert h1["target_dem"] != h2["target_dem"]
        assert h1["scene_description"] != h2["scene_description"]

        minimal_config.processing.generate_texture_preview = False
        h3 = compute_all_hashes(minimal_config, minimal_registry)
        assert h2["target_dem"] == h3["target_dem"]
        assert h2["target_texture"] != h3["target_texture"]


class TestManifestAndContextHandling:
    def test_cache_manifest(self, tmp_path):
        cm = CacheManifest(UPath(tmp_path))
        assert cm.load() == {}

        valid_data = {
            "manifest_version": MANIFEST_VERSION,
            "generator_version": "1.0",
            "resources": {"target_dem": {"effective_hash": "abc"}},
        }
        cm.save(valid_data)
        assert cm.load()["resources"]["target_dem"]["effective_hash"] == "abc"

        (cm.cache_dir / "manifest.json").write_text("NOT JSON {{{")
        assert cm.load() == {}

        cm.save({"manifest_version": 999, "generator_version": "1.0", "resources": {}})
        assert cm.load() == {}

        with patch("s2gos_generator.core.cache.get_version", return_value="2.0"):
            cm.save(valid_data)
            assert cm.load() == {}

        cm.clear()
        assert not cm.cache_dir.exists()

    @pytest.mark.parametrize(
        "resource, field, filename, yaml_content",
        [
            (
                "target_vegetation",
                "vegetation_objects_file",
                "vegetation_objects.yml",
                {"objects": [{"id": "veg"}]},
            ),
            (
                "hamster_data",
                "hamster_paths_file",
                "hamster_paths.yml",
                {"target": "/hamster.zarr"},
            ),
            (
                "user_assets",
                "user_assets_file",
                "user_assets.yml",
                {"objects": [{"id": "obj1"}]},
            ),
        ],
    )
    def test_capture_and_restore_context(
        self, tmp_path, resource, field, filename, yaml_content
    ):
        ctx = _make_ctx(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_path = data_dir / filename

        with open_file(file_path, "w") as f:
            yaml.dump(yaml_content, f)

        # 1. Test Restore
        entry = {"asset_fields": {field: f"data/{filename}"}}
        result = restore_context(resource, entry, ctx)
        assert getattr(ctx.assets, field) == UPath(file_path)
        assert result == UPath(file_path)

        # 2. Test Capture (Simulate a pre-existing state missing this file)
        ctx.assets = _StubAssets()
        assets_before = ctx.assets.to_dict()
        setattr(ctx.assets, field, UPath(file_path))

        captured = capture_asset_paths(resource, ctx, assets_before)
        assert field in captured
        assert f"data/{filename}" in captured[field]


class TestDeepValidators:
    def test_vegetation_and_user_assets_validation(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        veg_yaml = data_dir / "veg.yml"
        assets_yaml = data_dir / "assets.yml"

        # Vegetation deep validation
        with open_file(veg_yaml, "w") as f:
            yaml.dump(
                {
                    "objects": [
                        {
                            "type": "vegetation_collection",
                            "data_file": "data/pine.npy",
                            "count": 10,
                        }
                    ]
                },
                f,
            )

        assert not _validate_vegetation_files(
            UPath(tmp_path), {"vegetation_objects_file": UPath(veg_yaml)}
        )
        (data_dir / "pine.npy").touch()
        assert _validate_vegetation_files(
            UPath(tmp_path), {"vegetation_objects_file": UPath(veg_yaml)}
        )

        # User Assets deep validation
        with open_file(assets_yaml, "w") as f:
            yaml.dump({"objects": [{"id": "tower", "mesh": "data/tower.ply"}]}, f)

        assert not _validate_user_asset_files(
            UPath(tmp_path), {"user_assets_file": UPath(assets_yaml)}
        )
        (data_dir / "tower.ply").touch()
        assert _validate_user_asset_files(
            UPath(tmp_path), {"user_assets_file": UPath(assets_yaml)}
        )


class TestCachedDAGExecutor:
    def test_cache_lifecycle(self, tmp_path, minimal_config):
        """Verifies cache miss → manifest written → hit → force rerun lifecycle."""
        run_log = []
        r = ResourceRegistry()
        r.register(
            "target_dem",
            [],
            lambda ctx: run_log.append("dem"),
        )
        r.register(
            "scene_description", ["target_dem"], lambda ctx: run_log.append("sd")
        )  # NEVER_CACHE

        executor = CachedDAGExecutor(r)
        ctx = _make_ctx(tmp_path, config=minimal_config)

        # 1. First run (Cache Misses)
        executor.execute(ctx, use_cache=True)
        assert run_log == ["dem", "sd"]
        assert (UPath(tmp_path) / ".cache" / "manifest.json").exists()
        run_log.clear()

        # 2. Second run (Cache Hit for dem, Miss for sd)
        executor.execute(ctx, use_cache=True)
        assert run_log == ["sd"]  # 'dem' was skipped!
        run_log.clear()

        # 3. Third run with use_cache=False (Force Rerun)
        executor.execute(ctx, use_cache=False)
        assert run_log == ["dem", "sd"]

    def test_error_is_wrapped_as_runtime_error(self, tmp_path, minimal_config):
        """Resource exceptions are re-raised as RuntimeError with the original cause."""
        r = ResourceRegistry()
        r.register(
            "target_dem", [], lambda ctx: (_ for _ in ()).throw(ValueError("boom"))
        )
        executor = CachedDAGExecutor(r)
        ctx = _make_ctx(tmp_path, config=minimal_config)

        with pytest.raises(RuntimeError) as exc_info:
            executor.execute(ctx, use_cache=True)
        assert isinstance(exc_info.value.__cause__, ValueError)
