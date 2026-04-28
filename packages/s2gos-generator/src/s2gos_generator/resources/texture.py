"""Texture generation resources."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from PIL import Image

from ..assets.terrain_material import TerrainMaterialGenerator
from ..core.context import SceneResourceContext
from ..core.materials import build_material_index_map


def _lc_bounds(lc_data: xr.DataArray) -> tuple[float, float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax, native_res) from landcover pixel-centre coordinates."""
    x = lc_data.coords["x"].values
    y = lc_data.coords["y"].values
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    native_res = abs(xmax - xmin) / (len(x) - 1) if len(x) > 1 else 0.0
    return xmin, xmax, ymin, ymax, native_res


def _apply_region_materials_to_array(
    texture_2d: np.ndarray,
    landcover_path: Path,
    applicable_regions: list,
    ctx: SceneResourceContext,
    material_index_map: dict[str, int],
    area_name: str = "texture",
) -> tuple[np.ndarray, bool]:
    """Apply material region overlays to an in-memory texture array.

    Args:
        texture_2d: 2-D array
        landcover_path: Path to landcover zarr file
        applicable_regions: List of MaterialRegion configs to apply
        ctx: Scene resource context
        material_index_map: Mapping of material name to texture index
        area_name: Name for logging (e.g., "target texture", "buffer texture")

    Returns:
        (texture_2d, modified) where modified is True if any pixels were changed.
    """
    with xr.open_zarr(landcover_path) as ds:
        landcover_data = ds[list(ds.data_vars)[0]]
        width_px = len(landcover_data.coords["x"].values)
        height_px = len(landcover_data.coords["y"].values)
        xmin, xmax, ymin, ymax, _ = _lc_bounds(landcover_data)
        scene_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        landcover_2d = None
        if any(r.landcover_filter is not None for r in applicable_regions):
            landcover_2d = np.flipud(landcover_data.values)

    from ..core.region_geometry import geometry_from_dict

    coord_system = ctx.coordinate_system

    modified = False
    for region_config in applicable_regions:
        material_idx = material_index_map[region_config.material_name]

        try:
            geometry = geometry_from_dict(region_config.geometry)
            mask = geometry.to_mask(width_px, height_px, scene_bounds, coord_system)
            mask_flipped = np.flipud(mask)
            binary_mask = mask_flipped > 0

            if region_config.landcover_filter is not None and landcover_2d is not None:
                binary_mask = binary_mask & np.isin(
                    landcover_2d, region_config.landcover_filter
                )

            pixels_modified = np.sum(binary_mask)
            if pixels_modified > 0:
                texture_2d[binary_mask] = material_idx
                modified = True
                logging.info(
                    f"Applied region '{region_config.region_id}' to {area_name}: "
                    f"{pixels_modified} pixels → material '{region_config.material_name}' "
                    f"(index {material_idx})"
                )
        except Exception as e:
            logging.error(
                f"Failed to apply region '{region_config.region_id}' to {area_name}: {e}"
            )

    if modified:
        logging.info(f"Updated {area_name} with material regions")

    return texture_2d, modified


def _apply_roads_to_array(
    texture_2d: np.ndarray,
    landcover_path: Path,
    ctx: SceneResourceContext,
    road_material_indices: dict[str, int],
    area_name: str = "target",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Rasterize road polygons onto an in-memory texture array.

    Args:
        texture_2d: 2-D uint8 array to modify (may be resized if texture_resolution_m
            is finer than the landcover resolution)
        landcover_path: Path to landcover zarr (for resolution/bounds)
        ctx: Scene resource context
        road_material_indices: Mapping of material_name to texture index
        area_name: Logging label

    Returns:
        (texture_2d, union_mask) — texture_2d may have new dimensions after
        upsampling; ``union_mask`` is a boolean array of every painted road pixel
        (same shape as the returned ``texture_2d``), or ``None`` if no roads were
        applied. The caller can reuse it to overlay roads on the preview texture.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    road_geoms = ctx.road_polygons_by_material
    if not road_geoms:
        return texture_2d, None

    with xr.open_zarr(landcover_path) as ds:
        lc_data = ds[list(ds.data_vars)[0]]
        native_width_px = len(lc_data.coords["x"].values)
        native_height_px = len(lc_data.coords["y"].values)
        xmin, xmax, ymin, ymax, native_res = _lc_bounds(lc_data)

    texture_res = ctx.config.texture_resolution_m
    if texture_res is not None and texture_res < native_res:
        scale = native_res / texture_res
        target_width = round(native_width_px * scale)
        target_height = round(native_height_px * scale)
        raster_res = texture_res
    else:
        target_width = native_width_px
        target_height = native_height_px
        raster_res = native_res

    half_px = raster_res / 2
    transform = from_bounds(
        xmin - half_px,
        ymin - half_px,
        xmax + half_px,
        ymax + half_px,
        target_width,
        target_height,
    )

    if (target_height, target_width) != texture_2d.shape:
        texture_2d = np.array(
            Image.fromarray(texture_2d, mode="L").resize(
                (target_width, target_height), Image.NEAREST
            )
        )

    union_mask = np.zeros((target_height, target_width), dtype=bool)
    for material_name, merged_poly in road_geoms.items():
        mat_idx = road_material_indices.get(material_name)
        if mat_idx is None:
            continue

        road_mask = rasterize(
            [(merged_poly, 1)],
            out_shape=(target_height, target_width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        road_mask = np.flipud(road_mask) > 0

        pixels_modified = int(road_mask.sum())
        if pixels_modified > 0:
            texture_2d[road_mask] = mat_idx
            union_mask |= road_mask
            logging.info(
                "Applied roads [%s] to %s texture: %d pixels → material index %d (%dx%d px @ %.1fm/px)",
                material_name,
                area_name,
                pixels_modified,
                mat_idx,
                target_width,
                target_height,
                raster_res,
            )

    return texture_2d, (union_mask if union_mask.any() else None)


def _apply_roads_to_preview(
    preview_path: Path,
    road_mask: np.ndarray,
    debug_color: tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Resize the RGB preview to match ``road_mask`` and paint roads on it.

    The mask comes in selection-texture orientation (row 0 = south, no flipud
    on save), but the RGB preview is saved with ``flip_vertical=True`` for
    Blender's V-at-bottom convention (row 0 = north). Flip the mask once more
    before applying so roads land on the correct latitude in the preview.
    """
    target_h, target_w = road_mask.shape
    with Image.open(preview_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (target_w, target_h):
            rgb = rgb.resize((target_w, target_h), Image.NEAREST)
        arr = np.array(rgb)
    arr[np.flipud(road_mask)] = debug_color
    Image.fromarray(arr, mode="RGB").save(preview_path)


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
            texture_2d, changed = _apply_region_materials_to_array(
                texture_2d,
                landcover_path,
                applicable_regions,
                ctx,
                material_index_map,
                f"{area_name} texture",
            )
            dirty |= changed

    road_mask: Optional[np.ndarray] = None
    if ctx.dependency_outputs.get("target_roads") is not None and area_name == "target":
        texture_2d, road_mask = _apply_roads_to_array(
            texture_2d, landcover_path, ctx, material_index_map, area_name
        )
        if road_mask is not None:
            dirty = True

    if dirty:
        Image.fromarray(texture_2d, mode="L").save(selection_texture_path)

    if road_mask is not None and preview_texture_path is not None:
        _apply_roads_to_preview(preview_texture_path, road_mask)

    return selection_texture_path, preview_texture_path


def generate_target_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from target area land cover data.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated target selection texture file
    """

    landcover_file_path = ctx.dependency_outputs["target_landcover"]
    if landcover_file_path is None:
        raise ValueError("Target landcover file not found from dependencies")

    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None
    random_seed = None

    if ctx.config.snow is not None:
        dem_file_path = ctx.dependency_outputs["target_dem"]
        season_month = ctx.config.snow.season_month
        snow_material_index = ctx.config.snow.material_index
        random_seed = ctx.config.snow.random_seed
        if ctx.config.snow.thermoprops:
            snow_thermoprops = ctx.config.snow.thermoprops.thermoprops_file

        if dem_file_path is None:
            logging.warning("Seasonal snow requested but DEM not available")

    resolution_str = f"{ctx.landcover_resolution_m}m"
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        landcover_file_path,
        f"{ctx.scene_name}_{resolution_str}",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "target",
        random_seed,
    )

    ctx.assets.selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.preview_texture_file = preview_texture_path

    logging.info(f"Target texture: {selection_texture_path}")
    return selection_texture_path


def generate_buffer_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from buffer area land cover data (if buffer is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer selection texture file, or None if buffer disabled
    """
    buffer_landcover_file_path = ctx.dependency_outputs["buffer_landcover"]
    if buffer_landcover_file_path is None:
        logging.warning("Buffer landcover file not found from dependencies")
        return None

    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None
    random_seed = None

    if ctx.config.snow is not None:
        dem_file_path = ctx.dependency_outputs.get("buffer_dem")
        season_month = ctx.config.snow.season_month
        snow_material_index = ctx.config.snow.material_index
        random_seed = ctx.config.snow.random_seed
        if ctx.config.snow.thermoprops:
            snow_thermoprops = ctx.config.snow.thermoprops.thermoprops_file

        if dem_file_path is None:
            logging.warning("Seasonal snow requested for buffer but DEM not available")

    buffer_resolution_m = ctx.config.buffer.resolution_m
    resolution_str = f"{buffer_resolution_m}m"
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        buffer_landcover_file_path,
        f"{ctx.scene_name}_buffer_{resolution_str}",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "buffer",
        random_seed,
    )

    ctx.assets.buffer_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.buffer_preview_texture_file = preview_texture_path

    return selection_texture_path


def generate_background_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from background area land cover data (if background is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated background selection texture file, or None if background disabled
    """
    background_landcover_file_path = ctx.dependency_outputs["background_landcover"]
    if background_landcover_file_path is None:
        logging.warning("Background landcover file not found from dependencies")
        return None

    # Background is always flat - no snow application
    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None

    background_resolution_m = ctx.config.background.resolution_m
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        background_landcover_file_path,
        f"{ctx.scene_name}_background_{background_resolution_m}m",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "background",
    )

    ctx.assets.background_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.background_preview_texture_file = preview_texture_path

    return selection_texture_path
