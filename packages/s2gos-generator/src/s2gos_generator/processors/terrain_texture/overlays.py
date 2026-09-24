"""Texture-array algorithms: painting material regions and ways onto an
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


def _rasterize_setup(
    texture_2d: np.ndarray,
    landcover_path: Path,
    texture_resolution_m: Optional[float],
) -> tuple[np.ndarray, int, int, "Affine", float]:
    """Resolve target raster dimensions/transform for painting onto texture_2d.

    Resizes texture_2d (NEAREST) to the upsampled target dimensions when
    texture_resolution_m is finer than the landcover's native resolution, and
    returns the transform whose bounds margin is always half a *native*
    landcover pixel, not half a (possibly upsampled) raster pixel

    Returns (texture_2d, target_width, target_height, transform, raster_res).
    """
    from rasterio.transform import from_bounds

    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        lc_data = ds[list(ds.data_vars)[0]]
        native_width_px = len(lc_data.coords["x"].values)
        native_height_px = len(lc_data.coords["y"].values)
        xmin, xmax, ymin, ymax, native_res = _lc_bounds(lc_data)

    if texture_resolution_m is not None and texture_resolution_m < native_res:
        scale = native_res / texture_resolution_m
        target_width = round(native_width_px * scale)
        target_height = round(native_height_px * scale)
        raster_res = texture_resolution_m
    else:
        target_width = native_width_px
        target_height = native_height_px
        raster_res = native_res

    if (target_height, target_width) != texture_2d.shape:
        texture_2d = np.array(
            Image.fromarray(texture_2d, mode="L").resize(
                (target_width, target_height), Image.NEAREST
            )
        )

    half_px = native_res / 2
    transform = from_bounds(
        xmin - half_px,
        ymin - half_px,
        xmax + half_px,
        ymax + half_px,
        target_width,
        target_height,
    )

    return texture_2d, target_width, target_height, transform, raster_res


def apply_ways(
    texture_2d: np.ndarray,
    landcover_path: Path,
    way_polygons_by_material: dict,
    way_material_indices: dict[str, int],
    texture_resolution_m: Optional[float] = None,
    area_name: str = "target",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Rasterize way polygons onto an in-memory texture array.

    Args:
        texture_2d: 2-D uint8 array to modify (may be resized if texture_resolution_m
            is finer than the landcover resolution)
        landcover_path: Path to landcover zarr (for resolution/bounds)
        way_polygons_by_material: Merged way polygon per material name
            (from ``ctx.way_polygons_by_material``)
        way_material_indices: Mapping of material_name to texture index
        texture_resolution_m: Target texture resolution (from
            ``ctx.config.texture_resolution_m``); upsamples when finer than native
        area_name: Logging label

    Returns:
        (texture_2d, union_mask) — texture_2d may have new dimensions after
        upsampling; ``union_mask`` is a boolean array of every painted way pixel
        (same shape as the returned ``texture_2d``), or ``None`` if no ways were
        applied. The caller can reuse it to overlay ways on the preview texture.
    """
    from rasterio.features import rasterize

    way_geoms = way_polygons_by_material
    if not way_geoms:
        return texture_2d, None

    texture_2d, target_width, target_height, transform, raster_res = _rasterize_setup(
        texture_2d, landcover_path, texture_resolution_m
    )

    union_mask = np.zeros((target_height, target_width), dtype=bool)
    for material_name, merged_poly in way_geoms.items():
        mat_idx = way_material_indices.get(material_name)
        if mat_idx is None:
            logging.warning(
                "Way material %r has no texture index — those segments keep the "
                "underlying landcover material, even though the terrain under them "
                "is still flattened and still excludes vegetation. Check that %r is "
                "reachable from the way config and defined in the material library.",
                material_name,
                material_name,
            )
            continue

        way_mask = rasterize(
            [(merged_poly, 1)],
            out_shape=(target_height, target_width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        way_mask = np.flipud(way_mask) > 0

        pixels_modified = int(way_mask.sum())
        if pixels_modified > 0:
            texture_2d[way_mask] = mat_idx
            union_mask |= way_mask
            logging.info(
                "Applied ways [%s] to %s texture: %d pixels -> material index %d (%dx%d px @ %.1fm/px)",
                material_name,
                area_name,
                pixels_modified,
                mat_idx,
                target_width,
                target_height,
                raster_res,
            )

    return texture_2d, (union_mask if union_mask.any() else None)


def strip_unvetted_water_pixels(
    texture_2d: np.ndarray,
    landcover_path: Path,
    water_polygons_by_material: dict,
    water_material_index: int,
    fallback_material_index: int,
    texture_resolution_m: Optional[float] = None,
    area_name: str = "target",
) -> tuple[np.ndarray, bool]:
    """Remap water-material pixels that fall outside every vetted water body.

    Returns (texture_2d, modified).
    """
    all_polys = [
        p
        for p in water_polygons_by_material.values()
        if p is not None and not p.is_empty
    ]
    if not all_polys:
        return texture_2d, False

    from rasterio.features import rasterize
    from shapely.ops import unary_union

    water_union = unary_union(all_polys)

    texture_2d, target_width, target_height, transform, _raster_res = _rasterize_setup(
        texture_2d, landcover_path, texture_resolution_m
    )

    water_raster = rasterize(
        [(water_union, 1)],
        out_shape=(target_height, target_width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    vetted_mask = np.flipud(water_raster) > 0

    leak_mask = (texture_2d == water_material_index) & ~vetted_mask
    n_leaked = int(leak_mask.sum())
    if n_leaked > 0:
        texture_2d[leak_mask] = fallback_material_index
        logging.info(
            "Water: remapped %d landcover-only water pixel(s) outside vetted "
            "water bodies to material index %d (%s texture)",
            n_leaked,
            fallback_material_index,
            area_name,
        )
    return texture_2d, n_leaked > 0


def strip_steep_water_pixels(
    texture_2d: np.ndarray,
    dem_path: Path,
    landcover_path: Path,
    water_mask: np.ndarray,
    water_material_index: int,
    max_slope: float,
    texture_resolution_m: Optional[float] = None,
    area_name: str = "target",
) -> tuple[np.ndarray, bool]:
    """Repaint water-material pixels whose underlying DEM slope exceeds max_slope with the nearest non-water material."""
    if water_mask is None or not water_mask.any():
        return texture_2d, False

    from scipy.ndimage import distance_transform_edt, map_coordinates

    from ..terrain_mesh import compute_gradient, extract_dem

    with xr.open_zarr(expand_mapper(dem_path)) as ds:
        dem_data = ds[list(ds.data_vars)[0]]
        dem_x, dem_y, dem_elev = extract_dem(dem_data)
    slope = compute_gradient(dem_elev, dem_x, dem_y)

    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        lc_data = ds[list(ds.data_vars)[0]]
        xmin, xmax, ymin, ymax, native_res = _lc_bounds(lc_data)

    height, width = texture_2d.shape
    raster_res = texture_resolution_m if texture_resolution_m else native_res
    half_px = native_res / 2
    # Pixel-centre world coordinates of texture_2d, which has row 0 = ymin (matches apply_ways' own flipud convention).
    xs = np.linspace(
        xmin - half_px + raster_res / 2, xmax + half_px - raster_res / 2, width
    )
    ys = np.linspace(
        ymin - half_px + raster_res / 2, ymax + half_px - raster_res / 2, height
    )
    xx, yy = np.meshgrid(xs, ys)

    dx = (dem_x[-1] - dem_x[0]) / (len(dem_x) - 1)
    dy = (dem_y[-1] - dem_y[0]) / (len(dem_y) - 1)
    x_idx = (xx - dem_x[0]) / dx
    y_idx = (yy - dem_y[0]) / dy
    slope_on_texture = map_coordinates(
        slope, np.vstack((y_idx.ravel(), x_idx.ravel())), order=1, mode="nearest"
    ).reshape(xx.shape)

    remove_mask = water_mask & (slope_on_texture > max_slope)
    n_removed = int(remove_mask.sum())
    if n_removed == 0:
        return texture_2d, False

    # Nearest-fill from actual land only, never from another water pixel.
    is_water = texture_2d == water_material_index
    _, nearest_idx = distance_transform_edt(is_water, return_indices=True)
    texture_2d = texture_2d.copy()
    texture_2d[remove_mask] = texture_2d[tuple(idx[remove_mask] for idx in nearest_idx)]

    logging.info(
        "Water: repainted %d steep water pixel(s) (slope > %.3f m/m) with "
        "nearest land material (%s texture)",
        n_removed,
        max_slope,
        area_name,
    )
    return texture_2d, True


def apply_ways_to_preview(
    preview_path: Path,
    way_mask: np.ndarray,
    debug_color: tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Resize the RGB preview to match ``way_mask`` and paint ways on it."""
    target_h, target_w = way_mask.shape
    with Image.open(preview_path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (target_w, target_h):
            rgb = rgb.resize((target_w, target_h), Image.NEAREST)
        arr = np.array(rgb)
    arr[np.flipud(way_mask)] = debug_color
    Image.fromarray(arr, mode="RGB").save(preview_path)
