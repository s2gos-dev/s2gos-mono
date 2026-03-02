from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from upath import UPath

from .context import SceneResourceContext
from .resource_registry import DAGExecutor, ResourceRegistry

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


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _stable_hash(data: dict) -> str:
    """Return a 16-character deterministic hex digest of a dict."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Per-resource config fingerprints
# ──────────────────────────────────────────────────────────────────────────────


class ResourceFingerprints:
    """Per-resource config fingerprints for Merkle-tree cache invalidation.

    Each static method returns *only* the config fields that the resource
    directly reads.  Upstream fields are automatically captured through the
    dependency hash chain.
    """

    @staticmethod
    def get(resource_id: str, config) -> dict:
        """Return the own fingerprint dict for *resource_id*."""
        method = getattr(ResourceFingerprints, f"_{resource_id}", None)
        if method is None:
            return {}
        return method(config)

    # ── AOI resources ─────────────────────────────────────────────────────────

    @staticmethod
    def _aoi(config) -> dict:
        return config.location.model_dump()

    @staticmethod
    def _buffer_aoi(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "buffer_size_km": config.buffer.size_km if config.buffer else None,
        }

    @staticmethod
    def _background_aoi(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "background_size_km": config.background.size_km
            if config.background
            else None,
        }

    # ── DEM resources ─────────────────────────────────────────────────────────

    @staticmethod
    def _target_dem(config) -> dict:
        return {
            "target_resolution_m": config.target_resolution_m,
            "dem": config.data_sources.dem.model_dump(),
            "dem_fillna_value": config.processing.dem_fillna_value,
            "flatten_dem": config.processing.flatten_dem,
        }

    @staticmethod
    def _buffer_dem(config) -> dict:
        return {
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "dem": config.data_sources.dem.model_dump(),
            "dem_fillna_value": config.processing.dem_fillna_value,
            "flatten_dem": config.processing.flatten_dem,
        }

    # ── Landcover resources ───────────────────────────────────────────────────

    @staticmethod
    def _target_landcover(config) -> dict:
        return {
            "target_resolution_m": config.target_resolution_m,
            "landcover": config.data_sources.landcover.model_dump(),
        }

    @staticmethod
    def _buffer_landcover(config) -> dict:
        return {
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "landcover": config.data_sources.landcover.model_dump(),
        }

    @staticmethod
    def _background_landcover(config) -> dict:
        return {
            "background_resolution_m": (
                config.background.resolution_m if config.background else None
            ),
            "landcover": config.data_sources.landcover.model_dump(),
        }

    # ── Mesh resources ────────────────────────────────────────────────────────

    @staticmethod
    def _target_mesh(config) -> dict:
        return {"handle_dem_nans": config.processing.handle_dem_nans}

    @staticmethod
    def _buffer_mesh(config) -> dict:
        return {"handle_dem_nans": config.processing.handle_dem_nans}

    # ── Texture resources ─────────────────────────────────────────────────────

    @staticmethod
    def _target_texture(config) -> dict:
        target_regions = [
            r.model_dump() for r in config.material_regions if "target" in r.applies_to
        ]
        return {
            "snow": config.snow.model_dump() if config.snow else None,
            "material_regions": target_regions,
            "generate_texture_preview": config.processing.generate_texture_preview,
        }

    @staticmethod
    def _buffer_texture(config) -> dict:
        buffer_regions = [
            r.model_dump() for r in config.material_regions if "buffer" in r.applies_to
        ]
        return {
            "snow": config.snow.model_dump() if config.snow else None,
            "material_regions": buffer_regions,
            "generate_texture_preview": config.processing.generate_texture_preview,
        }

    @staticmethod
    def _background_texture(config) -> dict:
        bg_regions = [
            r.model_dump()
            for r in config.material_regions
            if "background" in r.applies_to
        ]
        return {
            "material_regions": bg_regions,
            "generate_texture_preview": config.processing.generate_texture_preview,
        }

    # ── Optional resources ────────────────────────────────────────────────────

    @staticmethod
    def _user_assets(config) -> dict:
        return {
            "user_assets": [a.model_dump() for a in config.user_assets],
            "xml_scenes": [x.model_dump() for x in config.xml_scenes],
        }

    @staticmethod
    def _vegetation_exclusion_zones(config) -> dict:
        return {
            "vegetation_exclusion_zones": [
                z.model_dump() for z in config.vegetation_exclusion_zones
            ],
        }

    @staticmethod
    def _target_vegetation(config) -> dict:
        return {
            "vegetation_placement": (
                config.vegetation_placement.model_dump()
                if config.vegetation_placement
                else None
            ),
            "random_seed": config.random_seed,
        }

    @staticmethod
    def _hamster_data(config) -> dict:
        return {
            "hamster": config.hamster.model_dump() if config.hamster else None,
            "target_resolution_m": config.target_resolution_m,
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "background_resolution_m": (
                config.background.resolution_m if config.background else None
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Merkle hash computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_all_hashes(config, registry: ResourceRegistry) -> Dict[str, str]:
    """Compute effective Merkle hashes for every resource in *registry*.

    ``effective_hash(R) = sha256(own_fingerprint(R) || sorted dep hashes)``
    """
    memo: Dict[str, str] = {}

    def _compute(resource_id: str) -> str:
        if resource_id in memo:
            return memo[resource_id]

        own_fp = ResourceFingerprints.get(resource_id, config)
        resource = registry.get_resource(resource_id)
        dep_hashes = [_compute(dep) for dep in sorted(resource.dependencies)]

        combined = hashlib.sha256()
        combined.update(_stable_hash(own_fp).encode())
        for dh in dep_hashes:
            combined.update(dh.encode())

        result = combined.hexdigest()[:16]
        memo[resource_id] = result
        return result

    for resource_id in registry.resources:
        _compute(resource_id)

    return memo


# ──────────────────────────────────────────────────────────────────────────────
# Cache manifest
# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# Asset / context field mappings
# ──────────────────────────────────────────────────────────────────────────────

# ctx.assets fields that each resource sets
_ASSET_FIELDS: Dict[str, List[str]] = {
    "target_dem": ["dem_file"],
    "buffer_dem": ["buffer_dem_file"],
    "target_landcover": ["landcover_file"],
    "buffer_landcover": ["buffer_landcover_file"],
    "background_landcover": ["background_landcover_file"],
    "target_mesh": ["mesh_file"],
    "buffer_mesh": ["buffer_mesh_file"],
    "target_texture": ["selection_texture_file", "preview_texture_file"],
    "buffer_texture": ["buffer_selection_texture_file", "buffer_preview_texture_file"],
    "background_texture": [
        "background_selection_texture_file",
        "background_preview_texture_file",
    ],
}

# Which ctx.assets field holds the primary return value (None = custom logic)
_RESULT_FIELD: Dict[str, Optional[str]] = {
    "target_dem": "dem_file",
    "buffer_dem": "buffer_dem_file",
    "target_landcover": "landcover_file",
    "buffer_landcover": "buffer_landcover_file",
    "background_landcover": "background_landcover_file",
    "target_mesh": "mesh_file",
    "buffer_mesh": "buffer_mesh_file",
    "target_texture": "selection_texture_file",
    "buffer_texture": "buffer_selection_texture_file",
    "background_texture": "background_selection_texture_file",
    "user_assets": None,
    "target_vegetation": None,
    "hamster_data": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Context restorer
# ──────────────────────────────────────────────────────────────────────────────


class ContextRestorer:
    """Saves and restores per-resource context state to/from disk."""

    def __init__(self, cache_dir: UPath) -> None:
        self.cache_dir = cache_dir

    # ── save ──────────────────────────────────────────────────────────────────

    def save_assets(
        self,
        resource_id: str,
        ctx: SceneResourceContext,
        assets_before: Dict[str, Any],
    ) -> Dict[str, str]:
        """Return a ``{field: rel_path}`` dict for newly-set asset files."""
        fields = _ASSET_FIELDS.get(resource_id, [])
        asset_paths: Dict[str, str] = {}
        for field in fields:
            value = getattr(ctx.assets, field, None)
            if value is not None and assets_before.get(field) is None:
                try:
                    rel = str(UPath(value).relative_to(ctx.output_dir))
                    asset_paths[field] = rel
                except (ValueError, TypeError):
                    asset_paths[field] = str(value)
        return asset_paths

    def save_complex_state(
        self, resource_id: str, ctx: SceneResourceContext
    ) -> Optional[str]:
        """Persist complex context state; return relative path or ``None``."""
        if resource_id == "user_assets":
            return self._save_user_assets(ctx)
        if resource_id == "target_vegetation":
            return self._save_vegetation(ctx)
        if resource_id == "hamster_data":
            return self._save_hamster(ctx)
        if resource_id in ("target_texture", "buffer_texture", "background_texture"):
            return self._save_region_material_indices(ctx)
        return None

    def _save_user_assets(self, ctx: SceneResourceContext) -> Optional[str]:
        state: Dict[str, Any] = {
            "processed_objects": [],
            "inline_materials": {},
            "exclusion_zones": [],
        }

        for obj in getattr(ctx, "processed_objects", []):
            try:
                state["processed_objects"].append(_serialize_obj(obj))
            except Exception as exc:
                logging.warning("Could not serialize processed_object: %s", exc)

        inline = getattr(ctx, "inline_materials", {})
        if inline:
            state["inline_materials"] = {
                k: _serialize_obj(v) for k, v in inline.items()
            }

        for zone in getattr(ctx, "vegetation_exclusion_zones", []):
            source = zone.get("source", "") if isinstance(zone, dict) else ""
            if source.startswith("asset_"):
                entry = dict(zone)
                geom = entry.pop("geometry", None)
                if geom is not None:
                    try:
                        entry["geometry_wkt"] = geom.wkt
                    except Exception:
                        entry["geometry_wkt"] = str(geom)
                state["exclusion_zones"].append(entry)

        return self._write_state("user_assets_state.json", state)

    def _save_vegetation(self, ctx: SceneResourceContext) -> Optional[str]:
        instances = getattr(ctx, "vegetation_instances", [])
        return self._write_state("vegetation_state.json", instances)

    def _save_hamster(self, ctx: SceneResourceContext) -> Optional[str]:
        paths = getattr(ctx, "hamster_data_paths", {}) or {}
        return self._write_state(
            "hamster_state.json", {k: str(v) for k, v in paths.items()}
        )

    def _save_region_material_indices(self, ctx: SceneResourceContext) -> Optional[str]:
        indices = getattr(ctx, "region_material_indices", None)
        if indices is None:
            return None
        return self._write_state("region_material_indices.json", indices)

    def _write_state(self, filename: str, data: Any) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / filename
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        return f".cache/{filename}"

    # ── restore ───────────────────────────────────────────────────────────────

    def restore(
        self,
        resource_id: str,
        entry: dict,
        ctx: SceneResourceContext,
    ) -> Any:
        """Restore context state from *entry* and return the cached result."""
        # Restore ctx.assets fields
        for field, rel_path in entry.get("asset_fields", {}).items():
            setattr(ctx.assets, field, ctx.output_dir / rel_path)

        # Restore complex state
        state_rel = entry.get("context_state_file")
        if state_rel:
            state_path = ctx.output_dir / state_rel
            if resource_id == "user_assets":
                self._restore_user_assets(ctx, state_path)
            elif resource_id == "target_vegetation":
                self._restore_vegetation(ctx, state_path)
            elif resource_id == "hamster_data":
                self._restore_hamster(ctx, state_path)
            elif resource_id in (
                "target_texture",
                "buffer_texture",
                "background_texture",
            ):
                self._restore_region_material_indices(ctx, state_path)

        return self._get_result(resource_id, entry, ctx)

    def _restore_user_assets(self, ctx: SceneResourceContext, path: UPath) -> None:
        if not path.exists():
            return
        with path.open("r") as f:
            state = json.load(f)
        ctx.processed_objects = state.get("processed_objects", [])
        ctx.inline_materials = state.get("inline_materials", {})
        for zone in state.get("exclusion_zones", []):
            entry = dict(zone)
            wkt_str = entry.pop("geometry_wkt", None)
            if wkt_str:
                try:
                    from shapely import wkt as shapely_wkt

                    entry["geometry"] = shapely_wkt.loads(wkt_str)
                except Exception:
                    pass
            ctx.vegetation_exclusion_zones.append(entry)

    def _restore_vegetation(self, ctx: SceneResourceContext, path: UPath) -> None:
        if not path.exists():
            return
        with path.open("r") as f:
            ctx.vegetation_instances = json.load(f)

    def _restore_hamster(self, ctx: SceneResourceContext, path: UPath) -> None:
        if not path.exists():
            return
        with path.open("r") as f:
            data = json.load(f)
        ctx.hamster_data_paths = {k: UPath(v) for k, v in data.items()}

    def _restore_region_material_indices(
        self, ctx: SceneResourceContext, path: UPath
    ) -> None:
        if not path.exists():
            return
        with path.open("r") as f:
            ctx.region_material_indices = json.load(f)

    def _get_result(
        self, resource_id: str, entry: dict, ctx: SceneResourceContext
    ) -> Any:
        """Return the value to store in ``results[resource_id]``."""
        result_field = _RESULT_FIELD.get(resource_id)
        if result_field is not None:
            return getattr(ctx.assets, result_field, None)

        if resource_id == "user_assets":
            asset_fields = entry.get("asset_fields", {})
            if asset_fields:
                first_rel = next(iter(asset_fields.values()))
                return ctx.output_dir / first_rel
            return None

        if resource_id == "target_vegetation":
            return getattr(ctx, "vegetation_instances", [])

        if resource_id == "hamster_data":
            return getattr(ctx, "hamster_data_paths", {})

        return None


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _serialize_obj(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-compatible structure."""
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_obj(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ──────────────────────────────────────────────────────────────────────────────
# Cached DAG executor
# ──────────────────────────────────────────────────────────────────────────────


class CachedDAGExecutor(DAGExecutor):
    """DAGExecutor that skips up-to-date resources using Merkle hashing.

    Resources are grouped into three categories:

    * **ALWAYS_EXECUTE** — cheap in-memory resources; always run but their
      hashes still feed downstream Merkle checks.
    * **NEVER_CACHE** — always regenerate (e.g. ``scene_description``).
    * **CACHEABLE** — all other resources; skipped when hash matches and all
      output files still exist on disk.
    """

    ALWAYS_EXECUTE = frozenset(
        {"aoi", "buffer_aoi", "background_aoi", "vegetation_exclusion_zones"}
    )
    NEVER_CACHE = frozenset({"scene_description"})

    def execute(  # type: ignore[override]
        self,
        context: SceneResourceContext,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Execute the DAG with optional caching."""
        manifest_helper = CacheManifest(context.output_dir)
        manifest = manifest_helper.load() if use_cache else {}
        effective_hashes = compute_all_hashes(context.config, self.registry)
        restorer = ContextRestorer(context.output_dir / ".cache")
        resources_entry: dict = manifest.setdefault("resources", {})
        results: Dict[str, Any] = {}

        for resource_id in self.registry.get_execution_order():
            resource = self.registry.get_resource(resource_id)
            context.dependency_outputs = results

            if resource_id in self.ALWAYS_EXECUTE:
                try:
                    result = resource(context)
                except Exception as exc:
                    raise RuntimeError(
                        f"Resource '{resource_id}' failed: {exc}"
                    ) from exc
                results[resource_id] = result

            elif resource_id in self.NEVER_CACHE:
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
                results[resource_id] = restorer.restore(
                    resource_id, resources_entry[resource_id], context
                )

            else:
                if resource_id == "target_vegetation":
                    self._reseed_rng(context)
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
                    restorer,
                    resources_entry,
                )

        if use_cache:
            manifest["manifest_version"] = MANIFEST_VERSION
            manifest["generator_version"] = get_version() or "unknown"
            manifest_helper.save(manifest)

        return results

    # ── private helpers ───────────────────────────────────────────────────────

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
        return True

    def _update_manifest_entry(
        self,
        resource_id: str,
        hashes: Dict[str, str],
        context: SceneResourceContext,
        assets_snapshot: Dict[str, Any],
        restorer: ContextRestorer,
        resources_entry: dict,
    ) -> None:
        asset_fields = restorer.save_assets(resource_id, context, assets_snapshot)
        state_file = restorer.save_complex_state(resource_id, context)
        resources_entry[resource_id] = {
            "effective_hash": hashes.get(resource_id, ""),
            "asset_fields": asset_fields,
            "context_state_file": state_file,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _reseed_rng(context: SceneResourceContext) -> None:
        """Re-seed global RNGs with a vegetation-specific deterministic seed."""
        seed = context.config.random_seed
        if seed is not None:
            veg_seed = hash((seed, "target_vegetation")) % (2**32)
            random.seed(veg_seed)
            np.random.seed(veg_seed)
            logging.info(
                "Re-seeded RNG for deterministic vegetation (seed=%d)", veg_seed
            )
