"""Fetch Sentinel-2 L2A reflectance onto the scene grid."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

# Asset suffix for the 10 m bands in the Copernicus sentinel-2-l2a collection.
_ASSET_SUFFIX = "_10m"
# Sentinel-2 L2A digital-number -> bottom-of-atmosphere reflectance (baseline N0400+).
_DN_SCALE = 1e-4
_DN_OFFSET = -0.1
# Digital-number fill value marking pixels without data.
_DN_NODATA = 0
# Pixels to erode off each date's valid region at internal data/nodata
# boundaries (swath edges) before compositing.
_EDGE_TRIM_PX = 3


def _configure_s3_credential(
    credential_id: Optional[str],
) -> Tuple[Optional[Dict[str, str]], Dict[str, str]]:
    """Resolve an s3 credential into ``odc.stac.configure_rio`` arguments.

    Resolves ``credential_id`` through the settings/secrets credential provider
    (``.secrets.yaml``).

    Returns ``(None, {})`` when no id is given, so reads fall back to the
    ambient boto3 credential chain (``AWS_*`` env, profiles).
    """
    if not credential_id:
        return None, {}

    from urllib.parse import urlparse

    from s2gos_utils.setting.credentials import get_credential

    cred = get_credential(credential_id)
    if getattr(cred, "type", None) != "s3":
        raise ValueError(
            f"Credential '{credential_id}' must be type 's3' for Sentinel-2 reads, "
            f"got '{getattr(cred, 'type', None)}'."
        )

    aws = {"aws_access_key_id": cred.key, "aws_secret_access_key": cred.secret}
    gdal_opts: Dict[str, str] = {}
    if cred.endpoint_url:
        parsed = urlparse(cred.endpoint_url)
        # GDAL's AWS_S3_ENDPOINT wants the bare host; the scheme goes to AWS_HTTPS.
        aws["endpoint_url"] = parsed.netloc or parsed.path
        gdal_opts["AWS_HTTPS"] = "YES" if parsed.scheme != "http" else "NO"
    return aws, gdal_opts


def _date_window(acquisition_date: str, window_days: int) -> str:
    """Build a STAC ``start/end`` datetime string around the anchor date."""
    from datetime import datetime, timedelta

    anchor = datetime.strptime(acquisition_date, "%Y-%m-%d")
    start = (anchor - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (anchor + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return f"{start}/{end}"


def _scene_geobox(grid_y: np.ndarray, grid_x: np.ndarray, scene_crs_wkt: str):
    """GeoBox of the scene grid (north-up) built from pixel-centre coords.

    ``grid_y``/``grid_x`` are the ascending, regularly spaced pixel-centre
    coordinates of the landcover raster in the scene CRS.
    """
    from affine import Affine
    from odc.geo.geobox import GeoBox

    res_y = float(grid_y[1] - grid_y[0])
    res_x = float(grid_x[1] - grid_x[0])
    for name, coords, res in (("grid_y", grid_y, res_y), ("grid_x", grid_x, res_x)):
        if res <= 0 or not np.allclose(
            np.diff(coords), res, rtol=0, atol=1e-6 * abs(res)
        ):
            raise ValueError(
                f"{name} must be ascending and regularly spaced to define a GeoBox"
            )

    transform = Affine(
        res_x,
        0.0,
        float(grid_x[0]) - res_x / 2,
        0.0,
        -res_y,
        float(grid_y[-1]) + res_y / 2,
    )
    return GeoBox((len(grid_y), len(grid_x)), transform, scene_crs_wkt)


def _to_band_array(ds: xr.Dataset, bands: Sequence[str]) -> xr.DataArray:
    """Stack per-band variables into a (band, time, y, x) float32 DataArray.

    Selects the ``{band}_10m`` variables in the requested order and converts
    the DN nodata fill (0) to NaN, which is what the composite logic expects.
    """
    da = ds[[f"{b}{_ASSET_SUFFIX}" for b in bands]].to_array(dim="band")
    da = da.assign_coords(band=list(bands)).astype("float32")
    return da.where(da != _DN_NODATA)


def _align_to_grid(
    da: xr.DataArray, grid_y: np.ndarray, grid_x: np.ndarray
) -> xr.DataArray:
    """Flip the north-up composite to ascending y and pin the exact grid coords.

    odc derives coordinates from the geobox affine, which can drift from the
    landcover's linspace-built coords in the last float ulp; after a sanity
    check the exact grid arrays are assigned so downstream alignment (zarr,
    texture matching) is bit-identical.
    """
    da = da.sortby("y")
    tol_y = 1e-6 * abs(float(grid_y[1] - grid_y[0]))
    tol_x = 1e-6 * abs(float(grid_x[1] - grid_x[0]))
    if not (
        np.allclose(da.y.values, grid_y, rtol=0, atol=tol_y)
        and np.allclose(da.x.values, grid_x, rtol=0, atol=tol_x)
    ):
        raise ValueError(
            "Loaded Sentinel-2 grid does not match the scene grid — "
            "geobox construction and landcover coords disagree."
        )
    return da.assign_coords(y=grid_y, x=grid_x)


def _load_date(da: xr.DataArray, date, retries: int = 4, base_delay: float = 2.0):
    """Compute one date with exponential backoff to ride out HTTP 429s.

    The dask scheduler is scoped to this compute call (sequential reads keep
    the request rate friendly to the CDSE endpoint) instead of being set
    process-globally.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return da.sel(time=date).compute(scheduler="single-threaded")
        except Exception as exc:  # noqa: BLE001 — transient network failures
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(base_delay * 2**attempt)
    raise last_exc


def _trim_swath_edges(
    layer: xr.DataArray, buffer_px: int = _EDGE_TRIM_PX
) -> xr.DataArray:
    """NaN-out valid pixels within ``buffer_px`` of a nodata region (swath-edge rim)."""
    from scipy.ndimage import binary_erosion

    valid = ~np.isnan(layer.values).any(axis=0)
    if valid.all():  # date fully covers the scene — nothing to trim
        return layer
    eroded = binary_erosion(valid, iterations=buffer_px, border_value=1)
    return layer.where(xr.DataArray(eroded, dims=("y", "x")))


def _pixel_coverage(layer: xr.DataArray) -> float:
    """Fraction of the scene grid with valid (non-NaN) data in the first band."""
    return float((~np.isnan(layer.isel(band=0))).mean())


def _date_str(d) -> str:
    """Render a date-like value (np.datetime64, str, ...) as 'YYYY-MM-DD'."""
    return str(d)[:10]


def _accumulate_until_covered(
    order,
    load_fn,
    min_coverage: float = 0.99,
):
    """Mosaic dates (in ``order``) until pixel coverage of the AOI is complete.

    Coverage is judged by **actual valid pixels**, not catalog tile ids: each loaded
    date is mosaicked onto the accumulator (closest dates kept first, so they win on
    overlap and later dates only fill NaN gaps). This rides through failed/partial
    reads, by pulling further dates to fill the gap, instead of trusting a tile-id set that hides
    the hole.

    Stops at ``min_coverage`` or once candidates run out, whichever comes first;
    pixels no candidate covers stay NaN, which downstream matching reads as
    "keep the base landcover material".

    Args:
        order: Candidate dates, already sorted closest-first to the anchor.
        load_fn: ``load_fn(date) -> DataArray`` for one mosaicked date (band, y, x).
        min_coverage: Stop once this fraction of the grid is filled.

    Returns:
        ``(composite, coverage)``. Raises if no date loads at all.
    """
    accum, coverage = None, 0.0
    dates_used, last_error = [], None
    for d in order:
        if accum is not None and coverage >= min_coverage:
            break
        try:
            layer = load_fn(d)
        except Exception as exc:  # noqa: BLE001 — transient network failures
            last_error = exc
            logging.warning(
                "Spectral S2: skip %s (%s: %s)", _date_str(d), type(exc).__name__, exc
            )
            continue
        accum = layer if accum is None else accum.combine_first(layer)
        coverage = _pixel_coverage(accum)
        dates_used.append(d)
        logging.info(
            "Spectral S2: + %s -> coverage %.1f%%", _date_str(d), coverage * 100
        )

    if accum is None:
        cause = repr(last_error) if last_error else "no candidate dates found"
        raise RuntimeError(
            f"Sentinel-2 fetch failed: every candidate date failed to load "
            f"(network/endpoint issue, not the composite logic). Last error: {cause}"
        )

    logging.info(
        "Spectral S2: composite from %d date(s) [%s], coverage %.1f%%",
        len(dates_used),
        ", ".join(_date_str(d) for d in dates_used),
        coverage * 100,
    )
    if coverage < min_coverage:
        logging.warning(
            "Spectral S2: composite coverage %.1f%% below target %.1f%% after %d "
            "date(s); AOI may be partially missing (persistent cloud gaps or repeated "
            "read failures across all candidate dates).",
            coverage * 100,
            min_coverage * 100,
            len(dates_used),
        )
    return accum, coverage


def fetch_s2_reflectance(
    acquisition_date: str,
    search_window_days: int,
    bands: Sequence[str],
    max_cloud_cover: float,
    stac_url: str,
    scene_crs_wkt: str,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    aoi_polygon,
    min_coverage: float,
    credential_id: Optional[str] = None,
) -> xr.DataArray:
    """Fetch a Sentinel-2 reflectance composite on the scene grid.

    Every matching item is warped by odc-stac from its native CRS straight
    onto the scene grid (resolution and extent come from ``grid_y``/``grid_x``),
    same-day items are mosaicked via ``groupby="solar_day"``, and days closest
    to the anchor date are composited until the grid is covered.

    Args:
        acquisition_date: Anchor date ``YYYY-MM-DD``.
        search_window_days: ± days around the anchor to search.
        bands: Sentinel-2 bands, defining the returned band order.
        max_cloud_cover: Maximum ``eo:cloud_cover`` percent — reported per whole
            granule (~110 km), not for the actual AOI, so a passing
            granule may still be cloudy over the AOI.
        stac_url: STAC catalog endpoint.
        scene_crs_wkt: Scene oblique-Mercator CRS as WKT.
        grid_y, grid_x: Scene grid coordinates (from the landcover raster) to
            load onto, guaranteeing coordinate alignment.
        aoi_polygon: AOI polygon in WGS84 (shapely) for the STAC query.
        min_coverage: Fraction of the scene grid that must be filled; dates
            are composited closest-first until this coverage is reached.
        credential_id: Id of an s3 credential (``.secrets.yaml``) for the
            Copernicus 'eodata' bucket; None falls back to ambient AWS_* env.

    Returns:
        ``(band, y, x)`` reflectance DataArray; band coord equals ``bands``, y
        ascending (south-row-0), reflectance clipped to ``>= 0``. Pixels no
        candidate date covers are NaN, which downstream matching reads as
        "keep the base landcover material".
    """
    bands = list(bands)

    import odc.stac
    import pystac_client
    from shapely.geometry import mapping

    catalog = pystac_client.Client.open(stac_url)
    catalog.add_conforms_to("ITEM_SEARCH")

    search = catalog.search(
        collections="sentinel-2-l2a",
        datetime=_date_window(acquisition_date, search_window_days),
        intersects=mapping(aoi_polygon),
        filter={
            "op": "<",
            "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
        },
    )
    items = search.item_collection()
    if len(items) == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found near {acquisition_date} "
            f"(±{search_window_days} d, cloud < {max_cloud_cover}%) for the AOI."
        )

    aws, cred_gdal_opts = _configure_s3_credential(credential_id)
    gdal_opts = {
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "GDAL_HTTP_TCP_KEEPALIVE": "YES",
        "GDAL_HTTP_MAX_RETRY": 5,
        "GDAL_HTTP_RETRY_DELAY": 1,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
    }
    gdal_opts.update(cred_gdal_opts)
    odc.stac.configure_rio(cloud_defaults=True, aws=aws, **gdal_opts)

    ds = odc.stac.load(
        items,
        bands=[f"{b}{_ASSET_SUFFIX}" for b in bands],
        geobox=_scene_geobox(grid_y, grid_x, scene_crs_wkt),
        groupby="solar_day",
        resampling="bilinear",
        dtype="uint16",
        nodata=_DN_NODATA,
        chunks={},
        fail_on_error=False,
    )
    stack = _to_band_array(ds, bands)

    target = np.datetime64(acquisition_date)
    order = sorted(stack.time.values, key=lambda t: abs(t - target))
    composite, coverage = _accumulate_until_covered(
        order,
        load_fn=lambda d: _trim_swath_edges(_load_date(stack, d)),
        min_coverage=min_coverage,
    )
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

    aligned = _align_to_grid(composite, grid_y, grid_x)

    refl = np.clip(aligned.values * _DN_SCALE + _DN_OFFSET, 0.0, None).astype("float32")
    return xr.DataArray(
        refl,
        dims=("band", "y", "x"),
        coords={"band": bands, "y": grid_y, "x": grid_x},
    )
