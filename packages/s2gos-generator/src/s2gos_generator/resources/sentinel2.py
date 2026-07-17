"""Sentinel-2 reflectance resource for spectral matching."""

import logging
from typing import Optional

import xarray as xr
from odc.geo.xr import assign_crs
from s2gos_utils.io.paths import expand_mapper
from upath import UPath

from ..core.context import SceneResourceContext


def process_target_sentinel2(ctx: SceneResourceContext) -> Optional[UPath]:
    """Fetch Sentinel-2 reflectance onto the target landcover grid (cached zarr).

    Returns the path to the reflectance zarr, or ``None`` if spectral
    matching is not configured.
    """
    cfg = ctx.config.spectral_matching
    if cfg is None:
        return None

    landcover_path = ctx.dependency_outputs["target_landcover"]
    if landcover_path is None:
        raise ValueError("Target landcover required for Sentinel-2 fetch")

    from ..processors.spectral.sentinel2 import fetch_s2_reflectance

    with xr.open_zarr(expand_mapper(landcover_path)) as ds:
        lc = ds[list(ds.data_vars)[0]]
        grid_y = lc.coords["y"].values
        grid_x = lc.coords["x"].values

    scene_crs_wkt = ctx.coordinate_system.scene_crs.to_wkt()
    reflectance = fetch_s2_reflectance(
        acquisition_date=cfg.acquisition_date,
        search_window_days=cfg.search_window_days,
        bands=cfg.bands,
        max_cloud_cover=cfg.max_cloud_cover,
        stac_url=cfg.stac_url,
        scene_crs_wkt=scene_crs_wkt,
        grid_y=grid_y,
        grid_x=grid_x,
        aoi_polygon=ctx.target_aoi_polygon,
        min_coverage=cfg.min_coverage,
        credential_id=cfg.credential_id,
    )

    cache_path = ctx.data_dir / "sentinel2_reflectance.zarr"
    assign_crs(reflectance.to_dataset(name="reflectance"), crs=scene_crs_wkt).to_zarr(
        cache_path, mode="w"
    )

    ctx.assets.sentinel2_file = cache_path
    logging.info("Sentinel-2 reflectance: %s", cache_path)
    return cache_path
