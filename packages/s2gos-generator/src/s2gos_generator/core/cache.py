from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import yaml
from upath import UPath

from .context import SceneResourceContext
from .fingerprints import compute_all_hashes
from .resource_registry import DAGExecutor

try:
    from ..._version import get_version as _get_version
except Exception:
    _get_version = None

# Bump when fingerprint definitions change — forces a full rebuild.
MANIFEST_VERSION = 1


def get_version() -> Optional[str]:
    """Return the current generator package version, or ``None`` if unavailable."""
    if _get_version is None:
        return None
    try:
        return _get_version()
    except Exception:
        return None


class CacheManifest:
    """Manages the on-disk JSON manifest at ``<output_dir>/.cache/manifest.json``."""

    def __init__(self, output_dir: UPath) -> None:
        self.cache_dir = output_dir / ".cache"
        self.manifest_path = self.cache_dir / "manifest.json"

    def load(self) -> dict:
        """Load and validate the manifest.  Returns ``{}`` on any failure."""
        if not self.manifest_path.exists():
            return {}

        try:
            with self.manifest_path.open("r") as f:
                data = json.load(f)
        except Exception as exc:
            logging.warning("Failed to read cache manifest: %s", exc)
            return {}

        if data.get("manifest_version") != MANIFEST_VERSION:
            logging.info(
                "Cache manifest version mismatch (%s vs %s) — rebuilding",
                data.get("manifest_version"),
                MANIFEST_VERSION,
            )
            return {}

        current = get_version()
        if current is not None and data.get("generator_version") != current:
            logging.info(
                "Generator version changed (%s → %s) — rebuilding",
                data.get("generator_version"),
                current,
            )
            return {}

        return data

    def save(self, data: dict) -> None:
        """Atomically write *data* to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_dir / "_manifest_tmp.json"
        try:
            with tmp.open("w") as f:
                json.dump(data, f, indent=2, default=str)
            tmp.rename(self.manifest_path)
        except Exception as exc:
            logging.warning("Failed to save cache manifest: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def clear(self) -> None:
        """Remove the entire ``.cache`` directory."""
        if self.cache_dir.exists():
            shutil.rmtree(str(self.cache_dir))


@dataclass
class ResourceCacheSpec:
    asset_fields: List[str]
    result_field: Optional[str] = None
    validator: Optional[Callable[[UPath, Dict[str, UPath]], bool]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.result_field is None and self.asset_fields:
            self.result_field = self.asset_fields[0]


def _validate_vegetation_files(
    output_dir: UPath, asset_paths: Dict[str, UPath]
) -> bool:
    """Check that every .npy referenced in vegetation_objects.yml exists."""
    veg_file = asset_paths.get("vegetation_objects_file")
    if veg_file is None or not veg_file.exists():
        return False
    try:
        from s2gos_utils.io.paths import open_file

        with open_file(veg_file, "r") as f:
            data = yaml.safe_load(f)
        for obj in data.get("objects", []):
            if obj.get("type") == "vegetation_collection":
                data_file = obj.get("data_file")
                if data_file and not (output_dir / data_file).exists():
                    logging.info(
                        "Cache miss: target_vegetation (secondary file missing: %s)",
                        data_file,
                    )
                    return False
    except Exception as exc:
        logging.warning("Deep validation failed for vegetation: %s", exc)
        return False
    return True


def _validate_user_asset_files(
    output_dir: UPath, asset_paths: Dict[str, UPath]
) -> bool:
    """Check that every .ply mesh referenced in user_assets.yml exists."""
    assets_file = asset_paths.get("user_assets_file")
    if assets_file is None or not assets_file.exists():
        return False
    try:
        from s2gos_utils.io.paths import open_file

        with open_file(assets_file, "r") as f:
            data = yaml.safe_load(f)
        for obj in data.get("objects", []):
            mesh = obj.get("mesh")
            if mesh and not (output_dir / mesh).exists():
                logging.info(
                    "Cache miss: user_assets (secondary file missing: %s)", mesh
                )
                return False
    except Exception as exc:
        logging.warning("Deep validation failed for user_assets: %s", exc)
        return False
    return True


_RESOURCE_CACHE_SPECS: Dict[str, ResourceCacheSpec] = {
    "target_dem": ResourceCacheSpec(["dem_file"]),
    "buffer_dem": ResourceCacheSpec(["buffer_dem_file"]),
    "target_landcover": ResourceCacheSpec(["landcover_file"]),
    "buffer_landcover": ResourceCacheSpec(["buffer_landcover_file"]),
    "background_landcover": ResourceCacheSpec(["background_landcover_file"]),
    "target_mesh": ResourceCacheSpec(["mesh_file"]),
    "buffer_mesh": ResourceCacheSpec(["buffer_mesh_file"]),
    "target_texture": ResourceCacheSpec(
        [
            "selection_texture_file",
            "preview_texture_file",
        ],
    ),
    "buffer_texture": ResourceCacheSpec(
        ["buffer_selection_texture_file", "buffer_preview_texture_file"],
    ),
    "background_texture": ResourceCacheSpec(
        ["background_selection_texture_file", "background_preview_texture_file"],
    ),
    "user_assets": ResourceCacheSpec(
        ["user_assets_file"], validator=_validate_user_asset_files
    ),
    "target_vegetation": ResourceCacheSpec(
        ["vegetation_objects_file"], validator=_validate_vegetation_files
    ),
    "hamster_data": ResourceCacheSpec(["hamster_paths_file"]),
}


def capture_asset_paths(
    resource_id: str,
    ctx: SceneResourceContext,
    assets_before: Dict[str, Any],
) -> Dict[str, str]:
    """Return a ``{field: rel_path}`` dict for newly-set asset files."""
    asset_fields = _RESOURCE_CACHE_SPECS.get(
        resource_id, ResourceCacheSpec([])
    ).asset_fields

    asset_paths: Dict[str, str] = {}
    for asset_field in asset_fields:
        value = getattr(ctx.assets, asset_field, None)
        if value is not None and assets_before.get(asset_field) is None:
            try:
                rel = str(UPath(value).relative_to(ctx.output_dir))
                asset_paths[asset_field] = rel
            except (ValueError, TypeError):
                asset_paths[asset_field] = str(value)
    return asset_paths


def restore_context(
    resource_id: str,
    entry: dict,
    ctx: SceneResourceContext,
) -> Any:
    """Restore asset paths from *entry* and return the cached result."""
    for asset_field, rel_path in entry.get("asset_fields", {}).items():
        setattr(ctx.assets, asset_field, ctx.output_dir / rel_path)
    return _get_cached_result(resource_id, ctx)


def _get_cached_result(resource_id: str, ctx: SceneResourceContext) -> Any:
    """Return the value to store in ``results[resource_id]``."""
    result_field = _RESOURCE_CACHE_SPECS.get(
        resource_id, ResourceCacheSpec([])
    ).result_field
    if result_field is not None:
        return getattr(ctx.assets, result_field, None)
    return None


class CachedDAGExecutor(DAGExecutor):
    """DAGExecutor that skips up-to-date resources using hashing.

    Resources are grouped into two categories:

    * **NEVER_CACHE** — always regenerate (e.g. ``scene_description``).
    * **CACHEABLE** — all other resources; skipped when hash matches and all
      output files still exist on disk.
    """

    NEVER_CACHE = frozenset({"scene_description"})

    def execute(  # type: ignore[override]
        self,
        context: SceneResourceContext,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Execute the DAG with optional caching.

        Note: Cache validity is based on hashing configuration fields (paths,
        parameters), not the contents of external files at those paths. In-place
        replacement of referenced files will not be detected as a cache miss.
        """
        manifest_helper = CacheManifest(context.output_dir)
        manifest = manifest_helper.load() if use_cache else {}
        effective_hashes = compute_all_hashes(context.config, self.registry)
        resources_entry: dict = manifest.setdefault("resources", {})
        results: Dict[str, Any] = {}

        for resource_id in self.registry.get_execution_order():
            resource = self.registry.get_resource(resource_id)
            context.dependency_outputs = results

            if resource_id in self.NEVER_CACHE:
                logging.info("Cache skip: %s (always regenerate)", resource_id)
                try:
                    result = resource(context)
                except Exception as exc:
                    raise RuntimeError(
                        f"Resource '{resource_id}' failed: {exc}"
                    ) from exc
                results[resource_id] = result

            elif use_cache and self._is_cache_valid(
                resource_id, effective_hashes, manifest, context
            ):
                logging.info("Cache hit: %s (skipping)", resource_id)
                results[resource_id] = restore_context(
                    resource_id, resources_entry[resource_id], context
                )

            else:
                logging.info("Executing: %s", resource_id)
                assets_snapshot = dict(context.assets.to_dict())
                try:
                    result = resource(context)
                except Exception as exc:
                    raise RuntimeError(
                        f"Resource '{resource_id}' failed: {exc}"
                    ) from exc
                results[resource_id] = result
                self._update_manifest_entry(
                    resource_id,
                    effective_hashes,
                    context,
                    assets_snapshot,
                    resources_entry,
                )

        if use_cache:
            manifest["manifest_version"] = MANIFEST_VERSION
            manifest["generator_version"] = get_version() or "unknown"
            manifest_helper.save(manifest)

        return results

    def _is_cache_valid(
        self,
        resource_id: str,
        hashes: Dict[str, str],
        manifest: dict,
        context: SceneResourceContext,
    ) -> bool:
        """Return True when the hash matches *and* all output files exist."""
        resources = manifest.get("resources", {})
        if resource_id not in resources:
            return False
        entry = resources[resource_id]
        if entry.get("effective_hash") != hashes.get(resource_id):
            return False
        for rel_path in entry.get("asset_fields", {}).values():
            if not (context.output_dir / rel_path).exists():
                return False

        # Deep validation: check secondary files (e.g. .npy, .ply)
        spec = _RESOURCE_CACHE_SPECS.get(resource_id)
        if spec is not None and spec.validator is not None:
            asset_paths = {
                field: context.output_dir / rel_path
                for field, rel_path in entry.get("asset_fields", {}).items()
            }
            if not spec.validator(context.output_dir, asset_paths):
                return False

        return True

    def _update_manifest_entry(
        self,
        resource_id: str,
        hashes: Dict[str, str],
        context: SceneResourceContext,
        assets_snapshot: Dict[str, Any],
        resources_entry: dict,
    ) -> None:
        asset_fields = capture_asset_paths(resource_id, context, assets_snapshot)
        resources_entry[resource_id] = {
            "effective_hash": hashes.get(resource_id, ""),
            "asset_fields": asset_fields,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
