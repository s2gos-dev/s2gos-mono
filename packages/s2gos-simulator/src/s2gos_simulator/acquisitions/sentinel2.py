from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import Transformer

from ..config.illumination import DirectionalIllumination
from ..config.sensors import SatelliteInstrument, SatellitePlatform, SatelliteSensor
from ..config.viewing import AngularViewing

logger = logging.getLogger(__name__)

# CDSE STAC catalog serving Sentinel-2 tile metadata assets.
_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_STAC_COLLECTION = "sentinel-2-l1c"  # L1C, not L2A, to match TOA sensor setups.

_MSI_BAND_ID_ORDER = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "8a",
    "9",
    "10",
    "11",
    "12",
]


def _sensing_time(root: ET.Element) -> datetime:
    sensing_time = root.find(".//SENSING_TIME").text
    return datetime.strptime(sensing_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _angle_grid(node: ET.Element) -> np.ndarray:
    return np.array([v.text.split() for v in node.iter("VALUES")], dtype=float)


def _view_vector_grid(grids: list[ET.Element]) -> np.ndarray:
    """Per-node (east, north, up) view vectors summed over detectors; NaN where none covers."""
    vectors = []
    for detector in grids:
        zenith = np.deg2rad(_angle_grid(detector.find("Zenith")))
        azimuth = np.deg2rad(_angle_grid(detector.find("Azimuth")))
        vectors.append(
            np.stack(
                [
                    np.sin(zenith) * np.sin(azimuth),
                    np.sin(zenith) * np.cos(azimuth),
                    np.cos(zenith),
                ],
                axis=-1,
            )
        )
    vectors = np.stack(vectors)
    covered = np.isfinite(vectors).all(axis=-1)
    grid = np.where(covered[..., None], vectors, 0.0).sum(axis=0)
    grid[~covered.any(axis=0)] = np.nan
    return grid


def _view_geometry(
    root: ET.Element, band: str, lat: float, lon: float
) -> AngularViewing:
    """Bilinearly interpolates one band's viewing angles at (lat, lon), as unit
    vectors so the azimuth wrap needs no special case."""
    if band not in _MSI_BAND_ID_ORDER:
        raise ValueError(f"Unknown MSI band {band!r}")
    band_id = _MSI_BAND_ID_ORDER.index(band)

    grids = root.findall(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']")
    if not grids:
        raise ValueError(
            f"No Viewing_Incidence_Angles_Grids for band {band!r} (bandId {band_id})"
        )
    grid = _view_vector_grid(grids)

    crs = root.find(".//Tile_Geocoding/HORIZONTAL_CS_CODE").text
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    origin = root.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
    step = grids[0].find("Zenith")
    row = (float(origin.find("ULY").text) - y) / float(step.find("ROW_STEP").text)
    col = (x - float(origin.find("ULX").text)) / float(step.find("COL_STEP").text)

    rows, cols = grid.shape[:2]
    if not (0 <= row <= rows - 1 and 0 <= col <= cols - 1):
        raise ValueError(f"({lat}, {lon}) lies outside the tile")
    i, j = min(int(row), rows - 2), min(int(col), cols - 2)
    fi, fj = row - i, col - j
    weights = np.array([[(1 - fi) * (1 - fj), (1 - fi) * fj], [fi * (1 - fj), fi * fj]])
    corners = grid[i : i + 2, j : j + 2]
    valid = np.isfinite(corners[..., 0])
    if weights[valid].sum() == 0:
        raise ValueError(
            f"({lat}, {lon}) lies outside the imaged swath for band {band!r}"
        )

    east, north, up = (weights[valid, None] * corners[valid]).sum(axis=0)
    zenith = np.degrees(np.arctan2(np.hypot(east, north), up))
    azimuth = np.degrees(np.arctan2(north, east)) % 360.0

    return AngularViewing(zenith=float(zenith), azimuth=float(azimuth))


def _platform(root: ET.Element) -> SatellitePlatform:
    tile_id = root.find(".//TILE_ID").text
    if tile_id.startswith("S2A"):
        return SatellitePlatform.SENTINEL_2A
    if tile_id.startswith("S2B"):
        return SatellitePlatform.SENTINEL_2B
    raise ValueError(f"Unrecognized Sentinel-2 platform in TILE_ID {tile_id!r}")


def read_tile_metadata(
    mtd_tl_path: str | Path, lat: float, lon: float, bands: list[str]
) -> tuple[datetime, SatellitePlatform, dict[str, AngularViewing]]:
    """Reads the acquisition geometry at a point from a Sentinel-2 MTD_TL.xml.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file, e.g. from `fetch_tile_metadata`.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        bands: Band identifiers as used by SentinelMSIBand (e.g. "2", "8a").

    Returns:
        The tile's SENSING_TIME (UTC-aware), the platform from the TILE_ID prefix,
        and per band the viewing zenith/azimuth at (lat, lon) in Eradiate
        convention, interpolated on the per-detector angle grids.

    Raises:
        ValueError: If the platform is not 2A/2B, a band is unknown or has no
            angle grids, or the point lies outside the tile or the imaged swath.
    """
    root = ET.parse(str(mtd_tl_path)).getroot()
    return (
        _sensing_time(root),
        _platform(root),
        {band: _view_geometry(root, band, lat, lon) for band in bands},
    )


def create_acquisition(
    mtd_tl_path: str | Path,
    target_center_lat: float,
    target_center_lon: float,
    bands: list[str],
    target_size_km: float | tuple[float, float] = 1.0,
    samples_per_pixel: int = 64,
    irradiance_dataset: str = "coddington_2022-1_nm",
    **kwargs,
) -> tuple[DirectionalIllumination, list[SatelliteSensor]]:
    """Create the illumination and MSI sensors of a real Sentinel-2 acquisition.

    Sun and view geometry both come from the tile's MTD_TL.xml: the sun is
    computed at the tile's SENSING_TIME, and each band views the target with its
    own angles interpolated at the target center. Films default to each band's
    native GSD (10, 20 or 60 m); pass `film_resolution` to override. Use
    `read_tile_metadata` directly for custom sensors or illumination.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file, e.g. from `fetch_tile_metadata`.
        target_center_lat: Target center latitude (WGS84 decimal degrees)
        target_center_lon: Target center longitude (WGS84 decimal degrees)
        bands: Band identifiers as used by SentinelMSIBand (e.g. "2", "8a").
        target_size_km: Target area size (km). Float for square, tuple for
            (width, height). Must be a whole multiple of every band's GSD unless
            `film_resolution` is given.
        samples_per_pixel: Monte Carlo samples per pixel. Default 64
        irradiance_dataset: Name of the solar irradiance dataset to use.
            Default `coddington_2022-1_nm` (TSIS-1 HSRS).
        **kwargs: Additional sensor parameters

    Returns:
        The illumination and one SatelliteSensor per band.

    Raises:
        ValueError: As raised by `read_tile_metadata`, or if the footprint is
            not a whole multiple of a band's GSD and no `film_resolution` is given.
    """
    time, platform, viewing = read_tile_metadata(
        mtd_tl_path, target_center_lat, target_center_lon, bands
    )
    logger.info("Sentinel-2 acquisition: %s sensed at %s", platform.value, time)

    illumination = DirectionalIllumination.from_date_and_location(
        time, target_center_lat, target_center_lon, irradiance_dataset
    )
    sensors = [
        SatelliteSensor(
            platform=platform,
            instrument=SatelliteInstrument.MSI,
            band=band,
            viewing=viewing[band],
            target_center_lat=target_center_lat,
            target_center_lon=target_center_lon,
            target_size_km=target_size_km,
            samples_per_pixel=samples_per_pixel,
            **kwargs,
        )
        for band in bands
    ]
    return illumination, sensors


def fetch_tile_metadata(
    date: str,
    lat: float,
    lon: float,
    cache_dir: str | Path,
    site_name: str | None = None,
    search_window_days: int = 6,
    credential_id: str = "aws_s2",
) -> Path:
    """Downloads (or reuses a cached) MTD_TL.xml for the S2 tile closest to a date.

    Args:
        date: Acquisition date "YYYY-MM-DD".
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        cache_dir: Directory to store the downloaded file in, or read the cached
            one from.
        site_name: Used in the cache filename; defaults to the lat/lon.
        search_window_days: +-days around `date` to search for the closest tile.
        credential_id: Id of an s3 credential (.secrets.yaml) for the Copernicus
            'eodata' bucket.

    Returns:
        Local path to the tile's MTD_TL.xml.

    Raises:
        RuntimeError: No Sentinel-2 L1C tile found near `date` at (lat, lon).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = site_name or f"{lat:.4f}_{lon:.4f}"
    cached_path = cache_dir / f"MTD_TL_{stem}_{date.replace('-', '')}.xml"
    if cached_path.exists():
        return cached_path

    import boto3
    import pystac_client
    from s2gos_utils.setting.credentials import get_credential

    catalog = pystac_client.Client.open(_STAC_URL)
    catalog.add_conforms_to("ITEM_SEARCH")

    anchor = datetime.strptime(date, "%Y-%m-%d")
    window_start = (anchor - timedelta(days=search_window_days)).strftime("%Y-%m-%d")
    window_end = (anchor + timedelta(days=search_window_days)).strftime("%Y-%m-%d")
    search = catalog.search(
        collections=[_STAC_COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{window_start}/{window_end}",
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L1C tile found near {date} (+-{search_window_days} d) "
            f"at ({lat}, {lon})."
        )
    item = min(
        items,
        key=lambda it: abs(it.datetime - anchor.replace(tzinfo=it.datetime.tzinfo)),
    )

    href = item.assets["granule_metadata"].href  # s3://eodata/...
    bucket, key = href.removeprefix("s3://").split("/", 1)

    from urllib.parse import urlparse

    cred = get_credential(credential_id)
    endpoint_url = None
    if cred.endpoint_url:
        parsed = urlparse(cred.endpoint_url)
        netloc = parsed.netloc or parsed.path
        scheme = parsed.scheme or "https"
        endpoint_url = f"{scheme}://{netloc}"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=cred.key,
        aws_secret_access_key=cred.secret,
        endpoint_url=endpoint_url,
    )
    s3.download_file(bucket, key, str(cached_path))

    return cached_path
