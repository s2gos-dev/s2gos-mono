from __future__ import annotations

import hashlib
import json
from typing import Dict

from .resource_registry import ResourceRegistry


def _stable_hash(data: dict) -> str:
    """Return a 16-character deterministic hex digest of a dict."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ResourceFingerprints:
    """Per-resource config fingerprints for cache.

    Each static method returns *only* the config fields that the resource
    directly reads.  Upstream fields are automatically captured through the
    dependency chain.
    """

    @staticmethod
    def get(resource_id: str, config) -> dict:
        """Return the own fingerprint dict for *resource_id*."""
        method = getattr(ResourceFingerprints, f"_{resource_id}", None)
        if method is None:
            return {}
        return method(config)

    @staticmethod
    def _target_dem(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "aoi_size_km": config.location.aoi_size_km,
            "target_resolution_m": config.target_resolution_m,
            "dem": config.data_sources.dem.model_dump(),
            "dem_fillna_value": config.processing.dem_fillna_value,
            "flatten_dem": config.processing.flatten_dem,
        }

    @staticmethod
    def _buffer_dem(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "buffer_size_km": config.buffer.size_km if config.buffer else None,
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "dem": config.data_sources.dem.model_dump(),
            "dem_fillna_value": config.processing.dem_fillna_value,
            "flatten_dem": config.processing.flatten_dem,
        }

    @staticmethod
    def _target_landcover(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "aoi_size_km": config.location.aoi_size_km,
            "target_resolution_m": config.target_resolution_m,
            "landcover": config.data_sources.landcover.model_dump(),
        }

    @staticmethod
    def _buffer_landcover(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "buffer_size_km": config.buffer.size_km if config.buffer else None,
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "landcover": config.data_sources.landcover.model_dump(),
        }

    @staticmethod
    def _background_landcover(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "background_size_km": (
                config.background.size_km if config.background else None
            ),
            "background_resolution_m": (
                config.background.resolution_m if config.background else None
            ),
            "landcover": config.data_sources.landcover.model_dump(),
        }

    @staticmethod
    def _target_mesh(config) -> dict:
        return {"handle_dem_nans": config.processing.handle_dem_nans}

    @staticmethod
    def _buffer_mesh(config) -> dict:
        return {"handle_dem_nans": config.processing.handle_dem_nans}

    @staticmethod
    def _target_roads(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "aoi_size_km": config.location.aoi_size_km,
            "roads": config.roads.model_dump() if config.roads else None,
        }

    @staticmethod
    def _target_texture(config) -> dict:
        target_regions = [
            r.model_dump() for r in config.material_regions if "target" in r.applies_to
        ]
        return {
            "snow": config.snow.model_dump() if config.snow else None,
            "material_regions": target_regions,
            "roads": getattr(config, "roads", None) and config.roads.model_dump(),
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

    @staticmethod
    def _user_assets(config) -> dict:
        return {
            "user_assets": [a.model_dump() for a in config.user_assets],
            "xml_scenes": [x.model_dump() for x in config.xml_scenes],
        }

    @staticmethod
    def _target_vegetation(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "vegetation_exclusion_zones": [
                z.model_dump() for z in config.vegetation_exclusion_zones
            ],
            "asset_exclusion_zones": [
                {
                    "coordinate": a.coordinate,
                    "coord_type": a.coord_type,
                    "exclusion_zone": a.exclusion_zone,
                }
                for a in config.user_assets
                if a.exclusion_zone is not None
            ],
            "xml_scene_exclusion_zones": [
                {
                    "base_coordinate": list(x.base_coordinate),
                    "coord_type": x.coord_type,
                    "exclusion_zone": x.exclusion_zone,
                }
                for x in config.xml_scenes
                if x.exclusion_zone is not None
            ],
            "vegetation_placement": (
                config.vegetation_placement.model_dump()
                if config.vegetation_placement
                else None
            ),
        }

    @staticmethod
    def _hamster_data(config) -> dict:
        return {
            "center_lat": config.location.center_lat,
            "center_lon": config.location.center_lon,
            "aoi_size_km": config.location.aoi_size_km,
            "buffer_size_km": config.buffer.size_km if config.buffer else None,
            "background_size_km": (
                config.background.size_km if config.background else None
            ),
            "hamster": config.hamster.model_dump() if config.hamster else None,
            "target_resolution_m": config.target_resolution_m,
            "buffer_resolution_m": config.buffer.resolution_m
            if config.buffer
            else None,
            "background_resolution_m": (
                config.background.resolution_m if config.background else None
            ),
        }


def compute_all_hashes(config, registry: ResourceRegistry) -> Dict[str, str]:
    """Compute effective Merkle hashes for every resource in *registry*."""
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
