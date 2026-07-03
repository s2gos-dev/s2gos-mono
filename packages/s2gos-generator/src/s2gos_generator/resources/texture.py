"""Texture generation resources."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import expand_mapper

from ..core.context import SceneResourceContext
from ..core.materials import build_material_index_map
from ..processors.terrain_material import TerrainMaterialGenerator
from ..processors.texture import (
    apply_region_materials,
    apply_roads,
    apply_roads_to_preview,
)


def _apply_spectral_matching(
    ctx: SceneResourceContext,
    texture_2d: np.ndarray,
    landcover_path: Path,
    base_index_map: dict[str, int],
) -> tuple[np.ndarray, bool]:
    """Re-texture landcover classes from Sentinel-2 + a material library.

    Clusters the reflectance of each configured landcover class and paints
    SAM-matched materials into ``texture_2d``. Writes a sidecar describing the new
    materials and their indices (read later by the scene step). Returns
    ``(texture_2d, changed)``.
    """
    from ..processors.spectral.diversify import (
        diversify_selection_texture,
        matched_materials_to_sidecar,
    )
    from ..processors.spectral.library import load_candidate_library

    cfg = ctx.config.spectral_matching
    s2_path = ctx.dependency_outputs.get("target_sentinel2")
    if s2_path is None:
        logging.warning("Spectral matching enabled but Sentinel-2 data missing")
        return texture_2d, False

    refl = (
        xr.open_zarr(expand_mapper(s2_path))["reflectance"]
        .sel(band=list(cfg.bands))
        .values
    )
    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        landcover_2d = ds[list(ds.data_vars)[0]].values

    library = load_candidate_library(cfg.material_library.upath, list(cfg.bands))
    texture_2d, material_defs, material_indices = diversify_selection_texture(
        texture_2d, landcover_2d, refl, cfg, base_index_map, library
    )

    if not material_indices:
        logging.info("Spectral matching: no clusters matched, texture unchanged")
        return texture_2d, False

    sidecar = matched_materials_to_sidecar(
        material_defs, material_indices, cfg.landcover_classes
    )
    sidecar_path = ctx.data_dir / "matched_materials.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(sidecar, f)
    ctx.assets.matched_materials_file = sidecar_path
    logging.info(
        "Spectral matching: %d material(s), %d new, sidecar %s",
        len(material_indices),
        len(material_defs),
        sidecar_path,
    )
    return texture_2d, True


def _generate_texture(
    ctx: SceneResourceContext,
    landcover_path: Path,
    base_name: str,
    dem_file_path: Optional[Path],
    season_month: Optional[int],
    snow_material_index: Optional[int],
    snow_thermoprops: Optional[Path],
    area_name: str,  # "target" | "buffer" | "background" — used for region filtering and log
    random_seed: Optional[int] = None,
) -> tuple[Path, Optional[Path]]:
    material_gen = TerrainMaterialGenerator()
    selection_texture_path, preview_texture_path = (
        material_gen.generate_textures_from_file(
            landcover_file_path=landcover_path,
            output_dir=ctx.textures_dir,
            base_name=base_name,
            create_preview=ctx.config.processing.generate_texture_preview,
            dem_file_path=dem_file_path,
            season_month=season_month,
            snow_material_index=snow_material_index,
            coordinate_system=ctx.coordinate_system,
            snow_thermoprops=snow_thermoprops,
            random_seed=random_seed,
        )
    )
    material_index_map = build_material_index_map(ctx)

    with Image.open(selection_texture_path) as img:
        raw = np.array(img)
    texture_2d = raw[:, :, 0] if raw.ndim == 3 else raw.copy()
    dirty = False

    if ctx.config.material_regions:
        applicable_regions = [
            r for r in ctx.config.material_regions if area_name in r.applies_to
        ]
        if applicable_regions:
            texture_2d, changed = apply_region_materials(
                texture_2d,
                landcover_path,
                applicable_regions,
                ctx.coordinate_system,
                material_index_map,
                f"{area_name} texture",
            )
            dirty |= changed

    if ctx.config.spectral_matching is not None and area_name == "target":
        texture_2d, changed = _apply_spectral_matching(
            ctx, texture_2d, landcover_path, material_index_map
        )
        dirty |= changed

    road_mask: Optional[np.ndarray] = None
    if ctx.dependency_outputs.get("target_roads") is not None and area_name == "target":
        texture_2d, road_mask = apply_roads(
            texture_2d,
            landcover_path,
            ctx.road_polygons_by_material,
            material_index_map,
            ctx.config.texture_resolution_m,
            area_name,
        )
        if road_mask is not None:
            dirty = True

    if dirty:
        Image.fromarray(texture_2d, mode="L").save(selection_texture_path)

    if road_mask is not None and preview_texture_path is not None:
        apply_roads_to_preview(preview_texture_path, road_mask)

    return selection_texture_path, preview_texture_path


def _snow_params(ctx: SceneResourceContext, dem_dep_key: Optional[str]):
    """Resolve the snow-related arguments for ``_generate_texture`` from ctx.config.snow.

    Returns (dem_file_path, season_month, snow_material_index, snow_thermoprops,
    random_seed).
    """
    snow = ctx.config.snow
    dem_file_path = ctx.dependency_outputs.get(dem_dep_key) if dem_dep_key else None
    if dem_file_path is None:
        logging.warning("Seasonal snow requested but DEM not available")
    thermoprops = snow.thermoprops.thermoprops_file if snow.thermoprops else None
    return (
        dem_file_path,
        snow.season_month,
        snow.material_index,
        thermoprops,
        snow.random_seed,
    )


@dataclass(frozen=True)
class _AreaSpec:
    """Per-area differences between the target/buffer/background texture steps."""

    name: str  # also the area_name passed to _generate_texture
    landcover_key: str
    dem_key: Optional[str]
    require_landcover: bool  # target raises if missing; others warn and skip
    applies_snow: bool
    resolution: Callable[[SceneResourceContext], float]
    name_infix: str  # "" | "buffer" | "background"
    selection_field: str
    preview_field: str


_TARGET = _AreaSpec(
    "target",
    "target_landcover",
    "target_dem",
    True,
    True,
    lambda ctx: ctx.landcover_resolution_m,
    "",
    "selection_texture_file",
    "preview_texture_file",
)
_BUFFER = _AreaSpec(
    "buffer",
    "buffer_landcover",
    "buffer_dem",
    False,
    True,
    lambda ctx: ctx.config.buffer.resolution_m,
    "buffer",
    "buffer_selection_texture_file",
    "buffer_preview_texture_file",
)
_BACKGROUND = _AreaSpec(
    "background",
    "background_landcover",
    None,
    False,
    False,
    lambda ctx: ctx.config.background.resolution_m,
    "background",
    "background_selection_texture_file",
    "background_preview_texture_file",
)


def _generate_area_texture(
    ctx: SceneResourceContext, spec: _AreaSpec
) -> Optional[Path]:
    """Generate the selection/preview textures for one scene area."""
    landcover_path = ctx.dependency_outputs.get(spec.landcover_key)
    if landcover_path is None:
        if spec.require_landcover:
            raise ValueError(f"{spec.name} landcover file not found from dependencies")
        logging.warning("%s landcover file not found from dependencies", spec.name)
        return None

    dem_file_path = season_month = snow_material_index = snow_thermoprops = None
    random_seed = None
    if spec.applies_snow and ctx.config.snow is not None:
        (
            dem_file_path,
            season_month,
            snow_material_index,
            snow_thermoprops,
            random_seed,
        ) = _snow_params(ctx, spec.dem_key)

    resolution_m = spec.resolution(ctx)
    prefix = f"{spec.name_infix}_" if spec.name_infix else ""
    base_name = f"{ctx.scene_name}_{prefix}{resolution_m}m"

    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        landcover_path,
        base_name,
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        spec.name,
        random_seed,
    )

    setattr(ctx.assets, spec.selection_field, selection_texture_path)
    if preview_texture_path:
        setattr(ctx.assets, spec.preview_field, preview_texture_path)

    logging.info("%s texture: %s", spec.name, selection_texture_path)
    return selection_texture_path


def generate_target_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from target-area landcover data."""
    return _generate_area_texture(ctx, _TARGET)


def generate_buffer_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from buffer-area landcover data (if buffer is enabled)."""
    return _generate_area_texture(ctx, _BUFFER)


def generate_background_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from background-area landcover data (if enabled)."""
    return _generate_area_texture(ctx, _BACKGROUND)
