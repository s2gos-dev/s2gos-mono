from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

from ._fetch import FootprintOutsideProduct, closest_fitting
from ._grid import Grid, fit_grid
from ..config.illumination import DirectionalIllumination, solar_position
from ..config.sensors import SatelliteSensor
from ..config.viewing import AngularViewing

logger = logging.getLogger(__name__)

_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


class _Product(NamedTuple):
    """Geometry files of an instrument's L1 product."""

    type: str
    """Product type in the product name."""

    coordinates: tuple[str, str, str]
    """(file, latitude, longitude) of the tie-point grid."""

    tie_step: tuple[int, int]
    """Image (rows, columns) per tie point."""

    image_grid: str | None
    """Manifest grid whose offsets place the tie points in the image, None when tie
    point (0, 0) is image pixel (0, 0) (OLCI)."""

    orthorectified: bool
    """Whether the image grid lies on the terrain (SLSTR) rather than the ellipsoid
    (OLCI L1B)."""

    time: tuple[str, str]
    """(file, variable) of the per-row time."""

    views: dict[str, tuple[str, str, str, str, str]]
    """Per view, (file, view zenith, view azimuth, sun zenith, sun azimuth)."""


_PRODUCTS = {
    "olci": _Product(
        type="OL_1_EFR",
        coordinates=("tie_geo_coordinates", "latitude", "longitude"),
        tie_step=(1, 64),
        image_grid=None,
        orthorectified=False,
        time=("time_coordinates", "time_stamp"),
        views={"nadir": ("tie_geometries", "OZA", "OAA", "SZA", "SAA")},
    ),
    "slstr": _Product(
        type="SL_1_RBT",
        coordinates=("geodetic_tx", "latitude_tx", "longitude_tx"),
        tie_step=(2, 32),
        image_grid="0.5 km stripe A",
        orthorectified=True,
        time=("time_in", "time_stamp_i"),
        views={
            view: (
                f"geometry_t{v}",
                f"sat_zenith_t{v}",
                f"sat_azimuth_t{v}",
                f"solar_zenith_t{v}",
                f"solar_azimuth_t{v}",
            )
            for view, v in (("nadir", "n"), ("oblique", "o"))
        },
    ),
}

_PLATFORMS = {"S3A": "sentinel_3a", "S3B": "sentinel_3b"}


class ProductGeometry(NamedTuple):
    """Acquisition geometry of a Sentinel-3 product at a target point."""

    time: datetime
    """UTC time of the product row nearest to the target."""

    instrument: str
    """Instrument key in `INSTRUMENTS`, e.g. 'sentinel_3b-olci'."""

    viewing: AngularViewing
    """View zenith/azimuth at the target, Eradiate convention."""

    sun_zenith: float
    """Sun zenith angle at the target in degrees."""

    sun_azimuth: float
    """Sun azimuth angle at the target in degrees, Eradiate convention."""


def _instrument_name(name: str) -> str:
    for instrument_name, product in _PRODUCTS.items():
        if product.type in name:
            return instrument_name
    raise ValueError(f"Not an OLCI EFR or SLSTR RBT product: {name!r}")


def _view_product(name: str, view: str) -> _Product:
    product = _PRODUCTS[name]
    if view not in product.views:
        raise ValueError(
            f"{name} has no {view!r} view, expected one of {list(product.views)}"
        )
    return product


def _direction(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """(east, north, up) unit vectors from zenith and azimuth clockwise from north."""
    zenith, azimuth = np.radians(zenith), np.radians(azimuth)
    return np.stack(
        [
            np.sin(zenith) * np.sin(azimuth),
            np.sin(zenith) * np.cos(azimuth),
            np.cos(zenith),
        ],
        axis=-1,
    )


def _eradiate_angles(vector: np.ndarray) -> tuple[float, float]:
    east, north, up = vector
    zenith = np.degrees(np.arctan2(np.hypot(east, north), up))
    return float(zenith), float(np.degrees(np.arctan2(north, east)) % 360.0)


def _open(product_dir: Path, file: str) -> xr.Dataset:
    with xr.open_dataset(product_dir / f"{file}.nc") as ds:
        return ds.load()


def _nearest(
    lats: np.ndarray, lons: np.ndarray, lat: float, lon: float, valid: np.ndarray
) -> tuple[int, int]:
    distance = np.where(
        valid, (lats - lat) ** 2 + ((lons - lon) * np.cos(np.radians(lat))) ** 2, np.inf
    )
    return np.unravel_index(np.argmin(distance), distance.shape)


def _image_layout(
    product_dir: Path, product: _Product, view: str, tie_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """(row, column) of tie point (0, 0) in the image, then the image's (rows, columns).

    SLSTR takes them from the manifest's along-track (start) and across-track (track)
    offsets of each grid and from the image grid's size.
    """
    rows_per_tie, cols_per_tie = product.tie_step
    if product.image_grid is None:
        return 0, 0, tie_shape[0] * rows_per_tie, (tie_shape[1] - 1) * cols_per_tie + 1
    manifest = (product_dir / "xfdumanifest.xml").read_text()

    def layout(grid: str) -> tuple[int, int, int, int]:
        match = re.search(
            rf'<slstr:{view}ImageSize grid="{grid}">\s*'
            r"<sentinel3:startOffset>(\d+)</sentinel3:startOffset>\s*"
            r"<sentinel3:trackOffset>(\d+)</sentinel3:trackOffset>\s*"
            r"<sentinel3:rows>(\d+)</sentinel3:rows>\s*"
            r"<sentinel3:columns>(\d+)</sentinel3:columns>",
            manifest,
        )
        if match is None:
            raise ValueError(
                f"{product_dir.name} manifest has no {view} {grid!r} image grid"
            )
        return int(match[1]), int(match[2]), int(match[3]), int(match[4])

    tie_start, tie_track, _, _ = layout("Tie Points")
    start, track, rows, columns = layout(product.image_grid)
    row_offset = rows_per_tie * tie_start - start
    return row_offset, track - cols_per_tie * tie_track, rows, columns


def read_product_footprint(
    product_dir: str | Path,
    lat: float,
    lon: float,
    size_km: float | tuple[float, float],
    view: str = "nadir",
) -> Grid:
    """Snap a footprint onto a view's image grid, giving the matching real pixels.

    The grid is fitted on the tie points around the target mapped to image indices.
    The tie points lie on the product grid, whereas OLCI's per-pixel
    ``geo_coordinates`` are displaced by relief toward the satellite.

    Args:
        product_dir: Path to the ``.SEN3`` product directory.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        size_km: Approximate footprint size (km). Float for square, tuple for
            (width, height) along the product's rows and columns.
        view: "nadir", or "oblique" for the SLSTR rear view.

    Returns:
        The snapped footprint, whose pixels crop the real image (SLSTR stripe A,
        ``_an`` or ``_ao``).

    Raises:
        FootprintOutsideProduct: If the footprint leaves the image, so part of it has
            no real pixels to match.
        ValueError: If the product or the view is unknown.
    """
    product_dir = Path(product_dir)
    product = _view_product(_instrument_name(product_dir.name), view)
    if isinstance(size_km, (int, float)):
        size_km = (size_km, size_km)
    file, lat_var, lon_var = product.coordinates
    coordinates = _open(product_dir, file)
    lats, lons = coordinates[lat_var].values, coordinates[lon_var].values
    valid = np.isfinite(lats) & np.isfinite(lons)
    row, col = _nearest(lats, lons, lat, lon, valid)
    window = (slice(max(row - 20, 0), row + 21), slice(max(col - 1, 0), col + 2))
    rows, cols = np.meshgrid(
        np.arange(lats.shape[0])[window[0]],
        np.arange(lats.shape[1])[window[1]],
        indexing="ij",
    )
    near = valid[window]
    row_offset, col_offset, image_rows, image_columns = _image_layout(
        product_dir, product, view, lats.shape
    )
    grid = fit_grid(
        rows[near] * product.tie_step[0] + row_offset,
        cols[near] * product.tie_step[1] + col_offset,
        lats[window][near],
        lons[window][near],
        lat,
        lon,
        tuple(size * 1000 for size in size_km),
    )
    last_row = grid.first_row + grid.shape[1]
    last_column = grid.first_column + grid.shape[0]
    if (
        grid.first_row < 0
        or grid.first_column < 0
        or last_row > image_rows
        or last_column > image_columns
    ):
        raise FootprintOutsideProduct(
            f"Footprint at ({lat}, {lon}) covers rows {grid.first_row}:{last_row} and "
            f"columns {grid.first_column}:{last_column}, which leave the "
            f"{image_rows} x {image_columns} {view} image of {product_dir.name}"
        )
    return grid


def read_product_geometry(
    product_dir: str | Path, lat: float, lon: float, view: str = "nadir"
) -> ProductGeometry:
    """Read the acquisition geometry at a point from a Sentinel-3 L1 product.

    Works on an OLCI EFR or SLSTR RBT product directory (``.SEN3``), either as
    downloaded by `fetch_product_geometry` or a complete unpacked product. Sun and
    view angles are interpolated linearly on the product's tie-point grid, as unit
    vectors so the azimuth wrap needs no special case.

    Args:
        product_dir: Path to the ``.SEN3`` product directory.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        view: "nadir", or "oblique" for the SLSTR rear view.

    Returns:
        The product geometry at (lat, lon), angles in Eradiate convention.

    Raises:
        ValueError: If the product, its platform or the view is unknown, or the point
            lies outside the view's swath or within a few tie points of the
            antimeridian, which is not supported.
    """
    product_dir = Path(product_dir)
    product = _view_product(_instrument_name(product_dir.name), view)
    platform = _PLATFORMS.get(product_dir.name[:3])
    if platform is None:
        raise ValueError(f"Unrecognized Sentinel-3 platform in {product_dir.name!r}")

    file, lat_var, lon_var = product.coordinates
    coordinates = _open(product_dir, file)
    lats, lons = coordinates[lat_var].values, coordinates[lon_var].values
    file, view_zenith, view_azimuth, sun_zenith, sun_azimuth = product.views[view]
    geometry = _open(product_dir, file)
    angles = [
        geometry[v].values for v in (view_zenith, view_azimuth, sun_zenith, sun_azimuth)
    ]

    valid = np.isfinite(lats) & np.isfinite(lons)
    for grid in angles:
        valid &= np.isfinite(grid)
    row, col = _nearest(lats, lons, lat, lon, valid)
    window = (slice(max(row - 6, 0), row + 7), slice(max(col - 2, 0), col + 3))
    near = valid[window]
    if np.ptp(lons[window][near]) > 180.0:
        raise ValueError(
            f"Target ({lat}, {lon}) lies too close to the antimeridian in "
            f"{product_dir.name}, which is not supported"
        )
    points = np.c_[lons[window][near], lats[window][near]]

    def at_target(zenith: np.ndarray, azimuth: np.ndarray) -> tuple[float, float]:
        vectors = _direction(zenith[window][near], azimuth[window][near])
        vector = griddata(points, vectors, (lon, lat), method="linear")
        if not np.isfinite(vector).all():
            raise ValueError(
                f"Target ({lat}, {lon}) lies outside the {view} swath of "
                f"{product_dir.name}"
            )
        return _eradiate_angles(vector)

    zenith, azimuth = at_target(*angles[:2])
    sun_zenith, sun_azimuth = at_target(*angles[2:])

    file, time_var = product.time
    time = _open(product_dir, file)[time_var].values[row]
    time = datetime.fromisoformat(np.datetime_as_string(time, unit="us"))

    return ProductGeometry(
        time=time.replace(tzinfo=timezone.utc),
        instrument=f"{platform}-{_instrument_name(product_dir.name)}",
        viewing=AngularViewing(zenith=zenith, azimuth=azimuth),
        sun_zenith=sun_zenith,
        sun_azimuth=sun_azimuth,
    )


def create_acquisition(
    product_dir: str | Path,
    target_center_lat: float,
    target_center_lon: float,
    bands: list[str],
    target_size_km: float | tuple[float, float],
    view: str = "nadir",
    samples_per_pixel: int = 64,
    irradiance_dataset: str = "coddington_2022-1_nm",
    **kwargs,
) -> tuple[DirectionalIllumination, SatelliteSensor]:
    """Create the illumination and sensor of a real Sentinel-3 acquisition.

    Sun and view angles both come from the product, at the footprint's center. The sun
    is read from the product because its time stamps are not exactly the instant the
    geometry refers to, so the time only scales the irradiance by the Earth-Sun distance.
    Every band shares the view's angles.

    The sensor images the product's own pixels, with the footprint moved by up to half a
    pixel and rounded to whole pixels (see `read_product_footprint`). Reverse the
    output's rows to compare it with those pixels. Like the product, OLCI pixels lie on
    the ellipsoid, so relief shifts what they see toward the satellite, and SLSTR pixels
    lie on the terrain. SLSTR sensors are named after their view unless given an `id`.

    Args:
        product_dir: Path to the ``.SEN3`` product directory, e.g. from
            `fetch_product_geometry`.
        target_center_lat: Target center latitude (WGS84 decimal degrees)
        target_center_lon: Target center longitude (WGS84 decimal degrees)
        bands: Band identifiers of the OLCI or SLSTR instrument (e.g. "8").
        target_size_km: Approximate target area size (km). Float for square, tuple for
            (width, height) along the product's rows and columns.
        view: "nadir", or "oblique" for the SLSTR rear view.
        samples_per_pixel: Monte Carlo samples per pixel. Default 64
        irradiance_dataset: Name of the solar irradiance dataset to use.
            Default `coddington_2022-1_nm` (TSIS-1 HSRS).
        **kwargs: Additional sensor parameters, other than the grid fields set here
            (`target_center_lat/lon`, `target_size_km`, `film_resolution`,
            `grid_rotation`, `orthorectified`).

    Returns:
        The illumination and one SatelliteSensor imaging every band.

    Raises:
        FootprintOutsideProduct: If the footprint leaves the real image, see
            `read_product_footprint`.
        ValueError: As raised by `read_product_geometry` or `SatelliteSensor`.
    """
    product_dir = Path(product_dir)
    product = _PRODUCTS[_instrument_name(product_dir.name)]
    grid = read_product_footprint(
        product_dir, target_center_lat, target_center_lon, target_size_km, view
    )
    geometry = read_product_geometry(
        product_dir, grid.center_lat, grid.center_lon, view
    )
    logger.info(
        "Sentinel-3 acquisition: %s %s view sensed at %s, product rows %d:%d "
        "and columns %d:%d, pixels of %.0f x %.0f m at grid rotation %.1f deg",
        geometry.instrument,
        view,
        geometry.time,
        grid.first_row,
        grid.first_row + grid.shape[1],
        grid.first_column,
        grid.first_column + grid.shape[0],
        *grid.pixel_size_m,
        grid.rotation,
    )
    if len(product.views) > 1:
        kwargs.setdefault("id", f"{geometry.instrument.replace('-', '_')}_{view}")

    illumination = DirectionalIllumination(
        zenith=geometry.sun_zenith,
        azimuth=geometry.sun_azimuth,
        irradiance_dataset=irradiance_dataset,
        irradiance_datetime=geometry.time,
    )
    sensor = SatelliteSensor(
        instrument=geometry.instrument,
        bands=bands,
        viewing=geometry.viewing,
        target_center_lat=grid.center_lat,
        target_center_lon=grid.center_lon,
        target_size_km=tuple(
            n * step / 1000 for n, step in zip(grid.shape, grid.pixel_size_m)
        ),
        film_resolution=grid.shape,
        grid_rotation=grid.rotation,
        orthorectified=product.orthorectified,
        samples_per_pixel=samples_per_pixel,
        **kwargs,
    )
    return illumination, sensor


def fetch_product_geometry(
    instrument_name: str,
    date: str,
    lat: float,
    lon: float,
    size_km: float | tuple[float, float],
    cache_dir: str | Path,
    site_name: str | None = None,
    search_window_days: int = 2,
    satellite: str | None = None,
    credential_id: str | None = None,
    view: str = "nadir",
) -> Path:
    """Download or reuse the geometry files of the closest product holding a footprint.

    Only daytime products are considered, and non-time-critical (NT) ones are preferred
    over near-real-time (NR) ones. A product whose image doesn't hold the footprint (see
    `read_product_footprint`) is skipped. A warning names the pass used when it is on
    another day than `date` or a closer product was skipped, because the sun and view
    geometry then differ. Only the geometry files and the manifest are downloaded, into
    a ``.SEN3`` directory laid out as the real product.

    Args:
        instrument_name: "olci" (OL_1_EFR) or "slstr" (SL_1_RBT).
        date: Acquisition date "YYYY-MM-DD".
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        size_km: Approximate footprint size (km). Float for square, tuple for
            (width, height) along the product's rows and columns.
        cache_dir: Directory to keep the downloaded files in, or read the cached
            ones from.
        site_name: Prefix for the cache directory name, which always includes the
            lat/lon.
        search_window_days: +-days around `date` to search for the closest product.
        satellite: "S3A" or "S3B" to restrict the search to one satellite, with its
            own cache directory.
        credential_id: Id of an s3 credential (.secrets.yaml) for the Copernicus
            'eodata' bucket. Defaults to the ambient AWS credential chain.
        view: The view that must hold the footprint, "nadir" or "oblique" for the SLSTR
            rear view, whose image is much narrower.

    Returns:
        Local path to the product's ``.SEN3`` directory.

    Raises:
        ValueError: `instrument_name` or `view` is unknown or `credential_id` is not an
            s3 credential.
        RuntimeError: No daytime product near `date` at (lat, lon) holds the footprint.
    """
    if instrument_name not in _PRODUCTS:
        raise ValueError(f"No Sentinel-3 product for instrument {instrument_name!r}")
    _view_product(instrument_name, view)
    anchor = datetime.strptime(date, "%Y-%m-%d").date()
    prefix = f"{site_name}_" if site_name else ""
    if satellite:
        prefix += f"{satellite.upper()}_"
    cache = Path(cache_dir) / (
        f"S3_{instrument_name}_{prefix}{lat:.4f}_{lon:.4f}_{anchor:%Y%m%d}"
    )
    return closest_fitting(
        cache,
        anchor,
        lambda path: read_product_footprint(path, lat, lon, size_km, view),
        lambda: _search_products(
            instrument_name,
            anchor,
            lat,
            lon,
            cache,
            search_window_days,
            satellite,
            credential_id,
        ),
    )


def _search_products(
    instrument_name: str,
    anchor: date_type,
    lat: float,
    lon: float,
    cache: Path,
    search_window_days: int,
    satellite: str | None,
    credential_id: str | None,
) -> dict:
    """Downloads of the daytime products over a point, by product name."""
    from s2gos_utils.setting.credentials import get_credential

    if credential_id:
        cred_type = getattr(get_credential(credential_id), "type", None)
        if cred_type != "s3":
            raise ValueError(
                f"Credential '{credential_id}' must be type 's3' for Sentinel-3 reads, "
                f"got '{cred_type}'."
            )

    product = _PRODUCTS[instrument_name]
    window = timedelta(days=search_window_days)
    query = (
        f"Collection/Name eq 'SENTINEL-3' and contains(Name,'{product.type}') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
        f"and ContentDate/Start gt {anchor - window:%Y-%m-%d}T00:00:00.000Z "
        f"and ContentDate/Start lt {anchor + window + timedelta(days=1):%Y-%m-%d}"
        "T00:00:00.000Z"
    )
    if satellite:
        query += f" and startswith(Name,'{satellite.upper()}')"
    params = urllib.parse.urlencode(
        {"$filter": query, "$top": "100", "$select": "Name,S3Path,ContentDate"}
    )
    items = []
    url = f"{_CATALOGUE_URL}?{params}"
    while url:
        with urllib.request.urlopen(url, timeout=60) as resp:
            page = json.load(resp)
        items += page["value"]
        url = page.get("@odata.nextLink")

    def daytime(item) -> bool:
        start = datetime.fromisoformat(item["ContentDate"]["Start"].rstrip("Z"))
        try:
            solar_position(start, lat, lon)
        except ValueError:
            return False
        return True

    items = [item for item in items if daytime(item)]
    if not items:
        raise RuntimeError(
            f"No daytime Sentinel-3 {product.type} product found near {anchor} "
            f"(+-{search_window_days} d) at ({lat}, {lon})."
        )
    return {
        item["Name"]: partial(_download_product, item, product, cache, credential_id)
        for item in items
    }


def _download_product(
    item: dict, product: _Product, cache: Path, credential_id: str | None
) -> Path:
    from s2gos_utils.io.paths import PathRef

    logger.info("Downloading the geometry of %s", item["Name"])
    files = {product.coordinates[0], product.time[0]}
    files.update(file for file, *_ in product.views.values())
    files = {f"{file}.nc" for file in files} | {"xfdumanifest.xml"}
    partial_dir = cache / f"{item['Name']}.part"
    partial_dir.mkdir(parents=True, exist_ok=True)
    remote = f"s3://{item['S3Path'].lstrip('/')}"
    for file in sorted(files):
        data = PathRef(href=f"{remote}/{file}", cid=credential_id).upath.read_bytes()
        (partial_dir / file).write_bytes(data)
    return partial_dir.rename(cache / item["Name"])
