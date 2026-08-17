"""Texture-array algorithms: painting material regions and roads onto an
in-memory selection texture, plus preview overlay."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import expand_mapper


def _lc_bounds(lc_data: xr.DataArray) -> tuple[float, float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax, native_res) from landcover pixel-centre coordinates."""
    x = lc_data.coords["x"].values
    y = lc_data.coords["y"].values
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    native_res = abs(xmax - xmin) / (len(x) - 1) if len(x) > 1 else 0.0
    return xmin, xmax, ymin, ymax, native_res


def apply_region_materials(
    texture_2d: np.ndarray,
    landcover_path: Path,
    applicable_regions: list,
    coord_system,
    material_index_map: dict[str, int],
    area_name: str = "texture",
) -> tuple[np.ndarray, bool]:
    """Apply material region overlays to an in-memory texture array.

    Args:
        texture_2d: 2-D array
        landcover_path: Path to landcover zarr file
        applicable_regions: List of MaterialRegion configs to apply
        coord_system: Scene coordinate system (from ``ctx.coordinate_system``)
        material_index_map: Mapping of material name to texture index
        area_name: Name for logging (e.g., "target texture", "buffer texture")

    Returns:
        (texture_2d, modified) where modified is True if any pixels were changed.
    """
    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        landcover_data = ds[list(ds.data_vars)[0]]
        width_px = len(landcover_data.coords["x"].values)
        height_px = len(landcover_data.coords["y"].values)
        xmin, xmax, ymin, ymax, _ = _lc_bounds(landcover_data)
        scene_bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

        landcover_2d = None
        if any(r.landcover_filter is not None for r in applicable_regions):
            landcover_2d = np.flipud(landcover_data.values)

    from ...core.region_geometry import geometry_from_dict

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
                    f"{pixels_modified} pixels -> material '{region_config.material_name}' "
                    f"(index {material_idx})"
                )
        except Exception as e:
            logging.error(
                f"Failed to apply region '{region_config.region_id}' to {area_name}: {e}"
            )

    if modified:
        logging.info(f"Updated {area_name} with material regions")

    return texture_2d, modified


def apply_ways(
    texture_2d: np.ndarray,
    landcover_path: Path,
    way_polygons_by_material: dict,
    road_material_indices: dict[str, int],
    texture_resolution_m: Optional[float] = None,
    area_name: str = "target",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Rasterize road polygons onto an in-memory texture array.

    Args:
        texture_2d: 2-D uint8 array to modify (may be resized if texture_resolution_m
            is finer than the landcover resolution)
        landcover_path: Path to landcover zarr (for resolution/bounds)
        way_polygons_by_material: Merged road polygon per material name
            (from ``ctx.way_polygons_by_material``)
        road_material_indices: Mapping of material_name to texture index
        texture_resolution_m: Target texture resolution (from
            ``ctx.config.texture_resolution_m``); upsamples when finer than native
        area_name: Logging label

    Returns:
        (texture_2d, union_mask) — texture_2d may have new dimensions after
        upsampling; ``union_mask`` is a boolean array of every painted road pixel
        (same shape as the returned ``texture_2d``), or ``None`` if no roads were
        applied. The caller can reuse it to overlay roads on the preview texture.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    road_geoms = way_polygons_by_material
    if not road_geoms:
        return texture_2d, None

    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        lc_data = ds[list(ds.data_vars)[0]]
        native_width_px = len(lc_data.coords["x"].values)
        native_height_px = len(lc_data.coords["y"].values)
        xmin, xmax, ymin, ymax, native_res = _lc_bounds(lc_data)

    texture_res = texture_resolution_m
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
                "Applied roads [%s] to %s texture: %d pixels -> material index %d (%dx%d px @ %.1fm/px)",
                material_name,
                area_name,
                pixels_modified,
                mat_idx,
                target_width,
                target_height,
                raster_res,
            )

    return texture_2d, (union_mask if union_mask.any() else None)


def apply_ways_to_preview(
    preview_path: Path,
    road_mask: np.ndarray,
    debug_color: tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Resize the RGB preview to match ``road_mask`` and paint roads on it."""
    target_h, target_w = road_mask.shape
    with Image.open(preview_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (target_w, target_h):
            rgb = rgb.resize((target_w, target_h), Image.NEAREST)
        arr = np.array(rgb)
    arr[np.flipud(road_mask)] = debug_color
    Image.fromarray(arr, mode="RGB").save(preview_path)
