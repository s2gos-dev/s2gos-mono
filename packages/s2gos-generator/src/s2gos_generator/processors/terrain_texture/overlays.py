"""Texture-array algorithms: painting material regions and ways onto an
in-memory selection texture, plus preview overlay."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import expand_mapper

from ...core.grid import SceneGrid


def apply_region_materials(
    texture_2d: np.ndarray,
    grid: SceneGrid,
    landcover_path: Path,
    applicable_regions: list,
    coord_system,
    material_index_map: dict[str, int],
    area_name: str = "texture",
) -> tuple[np.ndarray, bool]:
    """Apply material region overlays to an in-memory texture array.

    Args:
        texture_2d: 2-D array, in scene row order (row 0 is the southernmost)
        grid: Grid the texture lives on
        landcover_path: Path to landcover zarr file, read only for ``landcover_filter``
        applicable_regions: List of MaterialRegion configs to apply
        coord_system: Scene coordinate system (from ``ctx.coordinate_system``)
        material_index_map: Mapping of material name to texture index
        area_name: Name for logging (e.g., "target texture", "buffer texture")

    Returns:
        (texture_2d, modified) where modified is True if any pixels were changed.
    """
    landcover_2d = None
    if any(r.landcover_filter is not None for r in applicable_regions):
        with xr.open_zarr(expand_mapper(landcover_path)) as ds:
            landcover_2d = ds[list(ds.data_vars)[0]].values

    from ...core.region_geometry import geometry_from_dict

    modified = False
    for region_config in applicable_regions:
        material_idx = material_index_map[region_config.material_name]

        try:
            geometry = geometry_from_dict(region_config.geometry)
            binary_mask = geometry.to_mask(grid, coord_system) > 0

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
    grid: SceneGrid,
    way_polygons_by_material: dict,
    way_material_indices: dict[str, int],
    texture_resolution_m: Optional[float] = None,
    area_name: str = "target",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Rasterize way polygons onto an in-memory texture array.

    Args:
        texture_2d: 2-D uint8 array to modify (may be resized if texture_resolution_m
            is finer than the grid's resolution)
        grid: Grid the texture lives on
        way_polygons_by_material: Merged way polygon per material name
            (from ``ctx.way_polygons_by_material``)
        way_material_indices: Mapping of material_name to texture index
        texture_resolution_m: Target texture resolution (from
            ``ctx.config.texture_resolution_m``). Upsamples when finer than native
        area_name: Logging label

    Returns:
        (texture_2d, union_mask). ``texture_2d`` may have new dimensions after
        upsampling. ``union_mask`` is a boolean array of every painted way pixel
        (same shape as the returned ``texture_2d``), or ``None`` if no ways were
        applied. The caller can reuse it to overlay ways on the preview texture.

    Upsampling changes the pixel count but not the ground covered, so the area's UV
    window still holds.
    """
    way_geoms = way_polygons_by_material
    if not way_geoms:
        return texture_2d, None

    raster_grid = grid
    if texture_resolution_m is not None and texture_resolution_m < grid.resolution_m:
        # A resolution that does not divide the extent lands on a slightly different
        # one. The extent is what the UV window needs, and that is preserved exactly.
        raster_grid = SceneGrid(grid.size_m, round(grid.size_m / texture_resolution_m))

    if (raster_grid.n, raster_grid.n) != texture_2d.shape:
        texture_2d = np.array(
            Image.fromarray(texture_2d, mode="L").resize(
                (raster_grid.n, raster_grid.n), Image.NEAREST
            )
        )

    union_mask = np.zeros((raster_grid.n, raster_grid.n), dtype=bool)
    for material_name, merged_poly in way_geoms.items():
        mat_idx = way_material_indices.get(material_name)
        if mat_idx is None:
            logging.warning(
                "Way material %r has no texture index, so those segments keep the "
                "underlying landcover material, even though the terrain under them "
                "is still flattened and still excludes vegetation. Check that %r is "
                "reachable from the way config and defined in the material library.",
                material_name,
                material_name,
            )
            continue

        way_mask = raster_grid.rasterize([(merged_poly, 1)], all_touched=True) > 0

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
                raster_grid.n,
                raster_grid.n,
                raster_grid.resolution_m,
            )

    return texture_2d, (union_mask if union_mask.any() else None)


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
    # The preview PNG is north-up for humans, unlike everything else in this module.
    arr[np.flipud(way_mask)] = debug_color
    Image.fromarray(arr, mode="RGB").save(preview_path)
