"""Fetch Sentinel-2 L2A reflectance onto the scene grid.

Ported from ``experimenting/sentinel2_stac.ipynb``. Searches a Copernicus STAC
catalog around an anchor date, composites the fewest dates that tile the AOI, and
regrids the result onto the generator's oblique-Mercator scene grid so it aligns
by coordinate with the landcover/selection-texture raster.

Network access and Copernicus credentials (via environment) are required.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np
import rioxarray  # noqa: F401 — registers the xarray ``.rio`` accessor used below
import xarray as xr

# Asset suffix for the 10 m bands in the Copernicus sentinel-2-l2a collection.
_ASSET_SUFFIX = "_10m"
# Sentinel-2 L2A digital-number -> bottom-of-atmosphere reflectance (baseline N0400+).
_DN_SCALE = 1e-4
_DN_OFFSET = -0.1


def _configure_s3_credential(credential_id: Optional[str]) -> Dict[str, str]:
    """Apply a configured s3 credential for GDAL ``/vsis3/`` reads.

    Resolves ``credential_id`` through the settings/secrets credential provider
    (``.secrets.yaml``) and exports the **access key / secret** into the process
    environment — rasterio forbids those two as ``rasterio.Env`` options and reads
    them from ``os.environ`` (via boto3) instead. The non-credential GDAL options
    (endpoint, virtual-hosting, https) are returned for inclusion in ``gdal_env``.

    Returns ``{}`` when no id is given, so GDAL falls back to ambient ``AWS_*`` env.
    """
    if not credential_id:
        return {}

    from urllib.parse import urlparse

    from s2gos_utils.setting.credentials import get_credential

    cred = get_credential(credential_id)
    if getattr(cred, "type", None) != "s3":
        raise ValueError(
            f"Credential '{credential_id}' must be type 's3' for Sentinel-2 reads, "
            f"got '{getattr(cred, 'type', None)}'."
        )

    os.environ["AWS_ACCESS_KEY_ID"] = cred.key
    os.environ["AWS_SECRET_ACCESS_KEY"] = cred.secret

    gdal_opts: Dict[str, str] = {"AWS_VIRTUAL_HOSTING": "FALSE"}
    if cred.endpoint_url:
        parsed = urlparse(cred.endpoint_url)
        host = parsed.netloc or parsed.path
        gdal_opts["AWS_S3_ENDPOINT"] = host
        gdal_opts["AWS_HTTPS"] = "YES" if parsed.scheme != "http" else "NO"
    return gdal_opts


def _utm_epsg(center_lat: float, center_lon: float) -> int:
    """EPSG code of the UTM zone containing the AOI centre."""
    zone = int((center_lon + 180) // 6) + 1
    return (32600 if center_lat >= 0 else 32700) + zone


def _date_window(acquisition_date: str, window_days: int) -> str:
    """Build a STAC ``start/end`` datetime string around the anchor date."""
    from datetime import datetime, timedelta

    anchor = datetime.strptime(acquisition_date, "%Y-%m-%d")
    start = (anchor - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (anchor + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return f"{start}/{end}"


def _patch_stackstac_datetime() -> None:
    """Drop the removed ``infer_datetime_format`` kwarg stackstac passes to pandas."""
    import pandas as pd
    import stackstac.prepare

    original = pd.to_datetime

    def patched(*args, **kwargs):
        kwargs.pop("infer_datetime_format", None)
        return original(*args, **kwargs)

    stackstac.prepare.pd.to_datetime = patched


def _load_date(da: xr.DataArray, date, retries: int = 4, base_delay: float = 2.0):
    """Compute one date with exponential backoff to ride out HTTP 429s."""
    last_exc = None
    for attempt in range(retries):
        try:
            return da.sel(date=date).compute()
        except Exception as exc:  # noqa: BLE001 — transient network failures
            last_exc = exc
            time.sleep(base_delay * 2**attempt)
    raise last_exc


def _pixel_coverage(layer: xr.DataArray) -> float:
    """Fraction of the AOI grid with valid (non-NaN) data in the first band."""
    return float((~np.isnan(layer.isel(band=0))).mean())


def _accumulate_until_covered(
    order,
    load_fn,
    min_coverage: float = 0.99,
    max_dates: int = 8,
):
    """Mosaic dates (in ``order``) until pixel coverage of the AOI is complete.

    Coverage is judged by **actual valid pixels**, not catalog tile ids: each loaded
    date is mosaicked onto the accumulator (closest dates kept first, so they win on
    overlap and later dates only fill NaN gaps). This rides through failed/partial
    reads — which ``errors_as_nodata`` turns into silent NaN — by pulling further
    dates to fill the gap, instead of trusting a tile-id set that hides the hole.

    Args:
        order: Candidate dates, already sorted closest-first to the anchor.
        load_fn: ``load_fn(date) -> DataArray`` for one mosaicked date (band, y, x).
        min_coverage: Stop once this fraction of the grid is filled.
        max_dates: Hard cap on dates loaded, to bound network cost.

    Returns:
        ``(composite, coverage)``. Raises if no date loads at all.
    """
    accum = None
    n_used, errors = 0, []
    for d in order:
        if accum is not None and _pixel_coverage(accum) >= min_coverage:
            break
        if n_used >= max_dates:
            break
        try:
            layer = load_fn(d)
        except Exception as exc:  # noqa: BLE001 — transient network failures
            errors.append(exc)
            logging.warning(
                "Spectral S2: skip %s (%s)", str(d)[:10], type(exc).__name__
            )
            continue
        n_used += 1
        # Closest dates are added first, so on overlap the existing accumulator
        # (closer in time) wins and the new layer only fills its remaining NaN gaps.
        accum = layer if accum is None else accum.combine_first(layer)

    if accum is None:
        cause = repr(errors[-1]) if errors else "no candidate dates found"
        raise RuntimeError(
            f"Sentinel-2 fetch failed: every candidate date failed to load "
            f"(network/endpoint issue, not the composite logic). Last error: {cause}"
        )

    coverage = _pixel_coverage(accum)
    if coverage < min_coverage:
        logging.warning(
            "Spectral S2: composite coverage %.1f%% below target %.1f%% after %d "
            "date(s); AOI may be partially missing (persistent cloud gaps or repeated "
            "read failures across all candidate dates).",
            coverage * 100,
            min_coverage * 100,
            n_used,
        )
    return accum, coverage


def _closest_in_time_composite(
    da: xr.DataArray, target, min_coverage: float = 0.99, max_dates: int = 8
):
    """Composite the dates closest to ``target`` until the AOI is (nearly) filled."""
    import stackstac

    daily = (
        da.assign_coords(date=da.time.dt.floor("1D"))
        .groupby("date")
        .map(lambda g: stackstac.mosaic(g, dim="time"))
    )
    order = sorted(daily.date.values, key=lambda d: abs(d - target))
    return _accumulate_until_covered(
        order,
        load_fn=lambda d: _load_date(daily, d),
        min_coverage=min_coverage,
        max_dates=max_dates,
    )


def _regrid_to_scene(
    composite: xr.DataArray,
    aoi_epsg: int,
    scene_crs_wkt: str,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
) -> xr.DataArray:
    """Warp the UTM composite onto the scene grid defined by ``grid_y``/``grid_x``."""
    from rasterio.enums import Resampling

    template = xr.DataArray(
        np.zeros((len(grid_y), len(grid_x)), "float32"),
        dims=("y", "x"),
        coords={"y": grid_y, "x": grid_x},
    ).rio.write_crs(scene_crs_wkt)

    regridded = (
        composite.rio.write_crs(f"EPSG:{aoi_epsg}")
        .rio.write_nodata(np.nan)
        .rio.reproject_match(template, resampling=Resampling.bilinear)
        .sortby("y")
    )
    return regridded


def fetch_s2_reflectance(
    center_lat: float,
    center_lon: float,
    target_res_m: float,
    acquisition_date: str,
    search_window_days: int,
    bands: Sequence[str],
    max_cloud_cover: float,
    stac_url: str,
    scene_crs_wkt: str,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    aoi_polygon,
    credential_id: Optional[str] = None,
) -> xr.DataArray:
    """Fetch a Sentinel-2 reflectance composite on the scene grid.

    A pure fetch: caching is owned by the generator's fingerprint cache (see
    :mod:`..core.cache`), so the caller is responsible for persisting the result.

    Args:
        center_lat, center_lon: Scene centre (WGS84 degrees).
        target_res_m: Stacking resolution (10 m for the supported bands).
        acquisition_date: Anchor date ``YYYY-MM-DD``.
        search_window_days: ± days around the anchor to search.
        bands: Sentinel-2 bands, defining the returned band order.
        max_cloud_cover: Maximum ``eo:cloud_cover`` percent.
        stac_url: STAC catalog endpoint.
        scene_crs_wkt: Scene oblique-Mercator CRS as WKT.
        grid_y, grid_x: Scene grid coordinates (from the landcover raster) to
            regrid onto, guaranteeing coordinate alignment.
        aoi_polygon: AOI polygon in WGS84 (shapely) for the STAC query.
        credential_id: Id of an s3 credential (``.secrets.yaml``) for the
            Copernicus 'eodata' bucket; None falls back to ambient AWS_* env.

    Returns:
        ``(band, y, x)`` reflectance DataArray; band coord equals ``bands``, y
        ascending (south-row-0), reflectance clipped to ``>= 0``.
    """
    bands = list(bands)

    import dask
    import pystac_client
    import stackstac
    from rasterio.errors import RasterioIOError
    from shapely.geometry import mapping

    _patch_stackstac_datetime()
    dask.config.set(scheduler="single-threaded")

    aoi_epsg = _utm_epsg(center_lat, center_lon)
    catalog = pystac_client.Client.open(stac_url)
    catalog.add_conforms_to("ITEM_SEARCH")

    assets = [f"{b}{_ASSET_SUFFIX}" for b in bands]
    params = {
        "max_items": 100,
        "collections": "sentinel-2-l2a",
        "datetime": _date_window(acquisition_date, search_window_days),
        "intersects": mapping(aoi_polygon),
        "filter": {
            "op": "<",
            "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
        },
    }
    items = list(catalog.search(**params).items_as_dicts())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found near {acquisition_date} "
            f"(±{search_window_days} d, cloud < {max_cloud_cover}%) for the AOI."
        )

    gdal_env = {
        "GDAL_NUM_THREADS": -1,
        "GDAL_HTTP_UNSAFESSL": "YES",
        "GDAL_HTTP_TCP_KEEPALIVE": "YES",
        "GDAL_HTTP_MAX_RETRY": 5,
        "GDAL_HTTP_RETRY_DELAY": 1,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
    }
    gdal_env.update(_configure_s3_credential(credential_id))

    stack = stackstac.stack(
        items=items,
        assets=assets,
        resolution=(target_res_m, target_res_m),
        bounds_latlon=aoi_polygon.bounds,
        epsg=aoi_epsg,
        errors_as_nodata=(RasterioIOError(".*"),),
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(gdal_env),
    )
    stack = stack.sel(band=assets)  # preserve requested band order

    target = np.datetime64(acquisition_date)
    composite, coverage = _closest_in_time_composite(stack, target)
    logging.info("Spectral S2: composite coverage %.1f%%", coverage * 100)
    if coverage == 0.0:
        hint = (
            f"credential '{credential_id}' (check key/secret/endpoint in .secrets.yaml)"
            if credential_id
            else "missing credentials (set credential_id, or export AWS_* env vars)"
        )
        raise RuntimeError(
            "Sentinel-2 composite is empty (0% coverage). Every band read returned "
            f"nodata — most likely an S3 authentication failure: {hint}."
        )

    regridded = _regrid_to_scene(composite, aoi_epsg, scene_crs_wkt, grid_y, grid_x)

    refl = np.clip(regridded.values * _DN_SCALE + _DN_OFFSET, 0.0, None).astype(
        "float32"
    )
    return xr.DataArray(
        refl,
        dims=("band", "y", "x"),
        coords={"band": bands, "y": regridded.y.values, "x": regridded.x.values},
    )
