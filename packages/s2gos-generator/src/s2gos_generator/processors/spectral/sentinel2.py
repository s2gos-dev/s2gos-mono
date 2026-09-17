"""Fetch a Sentinel-2 L2A reflectance composite onto the scene grid.

Searches a STAC catalog around an acquisition date and mosaics the clearest dates
onto the landcover grid. Rasters store integer DN, not reflectance:
``reflectance = DN * scale + offset``, with per-band ``scale``/``offset`` read from
the product metadata (:func:`_band_radiometry`). A pixel survives only if that
reflectance is positive and its SCL scene class is not excluded.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from urllib3 import Retry

# Asset suffix for the 10 m bands in the Copernicus sentinel-2-l2a collection.
_ASSET_SUFFIX = "_10m"
# Scene-classification asset in the Copernicus sentinel-2-l2a collection (20 m).
_SCL_ASSET = "SCL_20m"

_DN_NODATA = 0
_SCL_NODATA = 0

# STAC raster-extension keys carrying the DN -> reflectance transform.
_RADIOMETRY_KEYS = frozenset({"raster:scale", "raster:offset"})

# Skip fetching a date's reflectance when its valid pixels would fill less than
# this fraction of the composite's remaining gap (a cheap 20 m SCL read decides
# before any 10 m band read). Never applied to the first date.
_SKIP_MIN_GAP_FILL = 0.25

_STAC_PAGE_LIMIT = 100
_STAC_TIMEOUT_S = (10, 60)


class S2FetchError(RuntimeError):
    """No usable Sentinel-2 imagery for the AOI in the search window.

    A *data availability* failure, so callers may degrade to "no spectral
    matching". Misconfiguration raises plain :class:`RuntimeError` instead.
    """


def _date_window(acquisition_date: str, window_days: int) -> str:
    """Build a STAC ``start/end`` string spanning ``+-window_days`` around the anchor."""
    from datetime import datetime, timedelta

    anchor = datetime.strptime(acquisition_date, "%Y-%m-%d")
    start = (anchor - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (anchor + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return f"{start}/{end}"


def _configure_s3_credential(
    credential_id: Optional[str],
) -> Tuple[Optional[Dict[str, str]], Dict[str, str]]:
    """Resolve an s3 credential into ``odc.stac.configure_rio`` arguments.

    Resolves ``credential_id`` through the settings/secrets credential provider
    (``.secrets.yaml``). Returns ``(None, {})`` when no id is given, so reads
    fall back to the ambient boto3 credential chain (``AWS_*`` env, profiles).
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


def _band_radiometry(
    collection, bands: Sequence[str], stac_url: str
) -> Dict[str, Tuple[float, float]]:
    """Per-band ``(scale, offset)`` for the DN -> reflectance conversion.

    Reads the raster-extension keys ``raster:scale``/``raster:offset``, which CDSE
    declares on the collection's ``item_assets`` once for every scene. The values
    are ESA's ``QUANTIFICATION_VALUE`` and ``BOA_ADD_OFFSET``, we read them from
    STAC ourselves because odc-stac 0.5.2 does not surface scale/offset.

    Raises:
        RuntimeError: A band is missing either key — a bad ``stac_url`` rather
            than a transient gap, hence deliberately not :class:`S2FetchError`.
    """
    item_assets = getattr(collection, "item_assets", None) or {}
    radiometry: Dict[str, Tuple[float, float]] = {}
    for band in bands:
        asset_key = f"{band}{_ASSET_SUFFIX}"
        definition = item_assets.get(asset_key)
        props = definition.to_dict() if definition is not None else {}
        if not _RADIOMETRY_KEYS <= props.keys():
            raise RuntimeError(
                f"{stac_url} declares no raster:scale/raster:offset for "
                f"{asset_key} on the sentinel-2-l2a collection item_assets."
            )
        radiometry[band] = (
            float(props["raster:scale"]),
            float(props["raster:offset"]),
        )
    return radiometry


def _to_band_array(
    ds: xr.Dataset, bands: Sequence[str], radiometry: Dict[str, Tuple[float, float]]
) -> xr.DataArray:
    """Stack per-band DN variables into a ``(band, time, y, x)`` reflectance array.

    Applies each band's ``scale``/``offset`` and NaNs pixels at or below zero
    reflectance.
    """
    da = ds[[f"{b}{_ASSET_SUFFIX}" for b in bands]].to_array(dim="band")
    da = da.assign_coords(band=list(bands)).astype("float32")
    scale = xr.DataArray(
        [radiometry[b][0] for b in bands], coords={"band": list(bands)}, dims="band"
    )
    offset = xr.DataArray(
        [radiometry[b][1] for b in bands], coords={"band": list(bands)}, dims="band"
    )
    reflectance = (da * scale + offset).astype("float32")
    return reflectance.where(reflectance > 0)


def _scene_geobox(grid_y: np.ndarray, grid_x: np.ndarray, scene_crs_wkt: str):
    """GeoBox of the scene grid (north-up) built from pixel-centre coords.

    ``grid_y``/``grid_x`` are the ascending, regularly spaced pixel-centre
    coordinates of the landcover raster in the scene CRS. odc-stac warps every
    item onto this geobox, which is what makes the fetched reflectance align
    pixel-for-pixel with the landcover.
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


def _align_to_grid(
    da: xr.DataArray, grid_y: np.ndarray, grid_x: np.ndarray
) -> xr.DataArray:
    """Flip the north-up composite to ascending y and pin the exact grid coords."""
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
    """Compute one date, retrying with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return da.sel(time=date).compute(scheduler="single-threaded")
        except Exception as exc:  # noqa: BLE001 — transient network failures
            last_exc = exc
            if attempt < retries - 1:
                delay = base_delay * 2**attempt
                logging.info(
                    "Spectral S2: read of %s failed (%s), retrying in %.0f s "
                    "(attempt %d/%d)",
                    _date_str(date),
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
    raise last_exc


class _LoggingRetry(Retry):
    """``Retry`` that logs each attempt instead of sleeping silently."""

    def increment(self, method=None, url=None, response=None, error=None, **kwargs):
        # Raises MaxRetryError once exhausted, so reaching the log means we retry.
        nxt = super().increment(
            method=method, url=url, response=response, error=error, **kwargs
        )
        logging.info(
            "Spectral S2: STAC %s -> %s, retrying in %.0f s (%d attempt(s) left)",
            url,
            getattr(response, "status", None) or repr(error),
            nxt.get_backoff_time(),
            nxt.total,
        )
        return nxt


def _stac_retry() -> Retry:
    """Retry policy for CDSE STAC requests."""
    return _LoggingRetry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,  # None means every method
        respect_retry_after_header=False,
    )


def _open_catalog(stac_url: str):
    """Open the STAC catalog with retries that survive CDSE rate limiting."""
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO

    catalog = pystac_client.Client.open(
        stac_url,
        stac_io=StacApiIO(max_retries=_stac_retry()),
        timeout=_STAC_TIMEOUT_S,
    )
    catalog.add_conforms_to("ITEM_SEARCH")
    return catalog


def _scl_valid_mask(scl: xr.DataArray, exclude: Sequence[int]) -> xr.DataArray:
    """``(y, x)`` bool mask of pixels usable per the scene classification.

    False where SCL is nodata (0) or one of the ``exclude`` classes.
    """
    return ~scl.isin(sorted({_SCL_NODATA, *exclude}))


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
    valid_mask_fn=None,
    min_gap_fill: float = _SKIP_MIN_GAP_FILL,
):
    """Mosaic dates in ``order`` until the AOI is covered.

    Coverage counts *actual valid pixels*, not catalog tile ids. Closest
    dates composite first and win on overlap; later ones only fill NaN gaps, and
    whatever no date covers stays NaN — downstream matching reads that as "keep
    the base landcover material".

    With ``valid_mask_fn`` each date's cheap 20 m SCL mask is read before its four
    10 m bands, so rejections cost less.

    Args:
        order: Candidate dates, already sorted closest-first to the anchor.
        load_fn: ``load_fn(date) -> DataArray`` for one mosaicked date (band, y, x).
        min_coverage: Stop once this fraction of the grid is filled.
        valid_mask_fn: Optional ``valid_mask_fn(date) -> (y, x) bool DataArray``
            of pixels usable for compositing (e.g. cloud-free per SCL).
        min_gap_fill: Fraction of the remaining gap a non-first date must promise
            to fill to be worth fetching.

    Returns:
        ``(composite, coverage)``.

    Raises:
        S2FetchError: No date contributed — every candidate failed to load, was
            screened out by the scene classification, or both.
    """
    accum, coverage = None, 0.0
    dates_used, last_error = [], None
    screened, thin = 0, 0
    for d in order:
        if accum is not None and coverage >= min_coverage:
            break
        try:
            if valid_mask_fn is not None:
                valid = valid_mask_fn(d)
                if not bool(valid.any()):
                    screened += 1
                    logging.info(
                        "Spectral S2: skip %s — no pixel passes the scene "
                        "classification",
                        _date_str(d),
                    )
                    continue
                # A gap-fill gate needs a gap; the first date is always taken.
                if accum is not None:
                    gaps = accum.isel(band=0, drop=True).isnull()
                    gap_px = int(gaps.sum())
                    new_px = int((valid & gaps).sum())
                    if new_px < min_gap_fill * max(gap_px, 1):
                        thin += 1
                        logging.info(
                            "Spectral S2: skip %s, valid pixels fill %.2f%% of "
                            "the remaining gap",
                            _date_str(d),
                            100 * new_px / max(gap_px, 1),
                        )
                        continue
                layer = load_fn(d).where(valid)
            else:
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
            "Spectral S2: + %s -> coverage %.2f%% (%d px uncovered)",
            _date_str(d),
            coverage * 100,
            int(accum.isel(band=0).isnull().sum()),
        )

    if accum is None:
        if screened and last_error is not None:
            raise S2FetchError(
                f"Sentinel-2 fetch failed: {screened} candidate date(s) were "
                f"screened out by the scene classification and the rest failed to "
                f"load. Last error: {last_error!r}"
            )
        if screened:
            raise S2FetchError(
                f"Sentinel-2 fetch failed: all {screened} candidate date(s) were "
                "screened out by the scene classification,  the AOI looks "
                "persistently cloudy. Widen search_window_days, raise "
                "max_cloud_cover, or relax scl_exclude."
            )
        cause = repr(last_error) if last_error else "no candidate dates found"
        raise S2FetchError(
            f"Sentinel-2 fetch failed: every candidate date failed to load "
            f"(network/endpoint issue, not the composite logic). Last error: {cause}"
        )

    logging.info(
        "Spectral S2: composite from %d date(s) [%s], %d screened out, %d too "
        "thin, coverage %.2f%% (%d px uncovered)",
        len(dates_used),
        ", ".join(_date_str(d) for d in dates_used),
        screened,
        thin,
        coverage * 100,
        int(accum.isel(band=0).isnull().sum()),
    )
    if coverage < min_coverage:
        logging.warning(
            "Spectral S2: composite coverage %.2f%% below target %.2f%% after %d "
            "date(s); AOI may be partially missing (persistent cloud gaps or "
            "repeated read failures across all candidate dates).",
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
    scl_exclude: Optional[Sequence[int]] = None,
    credential_id: Optional[str] = None,
) -> xr.DataArray:
    """Fetch a Sentinel-2 reflectance composite on the scene grid.

    odc-stac warps every matching item from its native CRS straight onto the scene
    grid, same-day items are mosaicked via ``groupby="solar_day"``, and dates
    closest to the anchor are composited until ``min_coverage`` is reached.

    Args:
        acquisition_date: Anchor date ``YYYY-MM-DD``.
        search_window_days: ± days around the anchor to search.
        bands: Sentinel-2 bands, defining the returned band order.
        max_cloud_cover: Maximum ``eo:cloud_cover`` percent — a coarse granule
            (~110 km) pre-filter; ``scl_exclude`` does the per-pixel screening.
        stac_url: STAC catalog endpoint. Must publish per-band radiometry.
        scene_crs_wkt: Scene oblique-Mercator CRS as WKT.
        grid_y, grid_x: Scene grid coordinates from the landcover raster; these
            fix the output resolution and extent, guaranteeing alignment.
        aoi_polygon: AOI polygon in WGS84 (shapely) for the STAC query.
        min_coverage: Fraction of the scene grid that must be filled.
        scl_exclude: SCL classes masked out per pixel (nodata 0 is always
            masked). None disables SCL loading and masking entirely.
        credential_id: Id of an s3 credential (``.secrets.yaml``) for the
            Copernicus 'eodata' bucket; None falls back to ambient AWS_* env.

    Returns:
        ``(band, y, x)`` reflectance, band coord equal to ``bands``, y ascending
        (south-row-0). Pixels no date covered are NaN, which downstream matching
        reads as "keep the base landcover material".

    Raises:
        S2FetchError: No usable imagery — empty search, or no date survived
            loading and screening.
        RuntimeError: The catalog publishes no per-band radiometry (see
            :func:`_band_radiometry`) — a misconfiguration, not a transient gap.
    """
    bands = list(bands)

    import odc.stac
    from shapely.geometry import mapping

    catalog = _open_catalog(stac_url)

    search = catalog.search(
        collections="sentinel-2-l2a",
        datetime=_date_window(acquisition_date, search_window_days),
        intersects=mapping(aoi_polygon),
        filter={
            "op": "<",
            "args": [{"property": "eo:cloud_cover"}, max_cloud_cover],
        },
        limit=_STAC_PAGE_LIMIT,
    )
    items = search.item_collection()
    if len(items) == 0:
        raise S2FetchError(
            f"No Sentinel-2 scenes found near {acquisition_date} "
            f"(±{search_window_days} d, cloud < {max_cloud_cover}%) for the AOI."
        )

    radiometry = _band_radiometry(
        catalog.get_collection("sentinel-2-l2a"), bands, stac_url
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

    # Load raw DN: _to_band_array applies the radiometry, so we keep control of
    # the conversion and the reflectance>0 masking. SCL is categorical and must be
    # resampled nearest, never interpolated.
    ds = odc.stac.load(
        items,
        bands=[f"{b}{_ASSET_SUFFIX}" for b in bands]
        + ([_SCL_ASSET] if scl_exclude is not None else []),
        geobox=_scene_geobox(grid_y, grid_x, scene_crs_wkt),
        groupby="solar_day",
        resampling={"*": "bilinear", _SCL_ASSET: "nearest"},
        dtype="uint16",
        nodata=_DN_NODATA,
        chunks={},
        fail_on_error=False,
    )
    stack = _to_band_array(ds, bands, radiometry)
    valid_mask_fn = (
        None
        if scl_exclude is None
        else lambda d: _scl_valid_mask(_load_date(ds[_SCL_ASSET], d), scl_exclude)
    )

    target = np.datetime64(acquisition_date)
    order = sorted(stack.time.values, key=lambda t: abs(t - target))
    composite, coverage = _accumulate_until_covered(
        order,
        load_fn=lambda d: _load_date(stack, d),
        min_coverage=min_coverage,
        valid_mask_fn=valid_mask_fn,
    )
    if coverage == 0.0:
        hint = (
            f"credential '{credential_id}' (check key/secret/endpoint in .secrets.yaml)"
            if credential_id
            else "missing credentials (set credential_id, or export AWS_* env vars)"
        )
        raise S2FetchError(
            "Sentinel-2 composite is empty (0% coverage). Every band read returned "
            f"nodata — most likely an S3 authentication failure: {hint}."
        )

    return (
        _align_to_grid(composite, grid_y, grid_x)
        .astype("float32")
        .transpose("band", "y", "x")
        .drop_vars("time", errors="ignore")
    )
