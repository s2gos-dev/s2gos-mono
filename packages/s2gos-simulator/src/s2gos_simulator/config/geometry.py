from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import Transformer

from .sensors import SatellitePlatform
from .viewing import AngularViewing

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


def overpass_time_from_tile_metadata(mtd_tl_path: str | Path) -> datetime:
    """Reads the acquisition time from a Sentinel-2 MTD_TL.xml tile metadata file.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file.

    Returns:
        The tile's SENSING_TIME as a timezone-aware UTC datetime.
    """
    root = ET.parse(str(mtd_tl_path)).getroot()
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


def view_geometry_from_tile_metadata(
    mtd_tl_path: str | Path, band: str, lat: float, lon: float
) -> AngularViewing:
    """Reads one band's viewing zenith/azimuth at a point from MTD_TL.xml.

    Bilinearly interpolates the Viewing_Incidence_Angles_Grids at (lat, lon),
    as unit vectors so the azimuth wrap needs no special case.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file.
        band: Band identifier as used by SentinelMSIBand (e.g. "2", "8a").
        lat, lon: Target coordinates.

    Returns:
        An AngularViewing with zenith/azimuth in Eradiate convention.

    Raises:
        ValueError: If the band is unknown or has no angle grids in the file,
            or the point lies outside the tile or the imaged swath.
    """
    if band not in _MSI_BAND_ID_ORDER:
        raise ValueError(f"Unknown MSI band {band!r}")
    band_id = _MSI_BAND_ID_ORDER.index(band)

    root = ET.parse(str(mtd_tl_path)).getroot()
    grids = root.findall(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']")
    if not grids:
        raise ValueError(
            f"No Viewing_Incidence_Angles_Grids for band {band!r} "
            f"(bandId {band_id}) in {mtd_tl_path}"
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
        raise ValueError(f"({lat}, {lon}) lies outside the tile in {mtd_tl_path}")
    i, j = min(int(row), rows - 2), min(int(col), cols - 2)
    fi, fj = row - i, col - j
    weights = np.array([[(1 - fi) * (1 - fj), (1 - fi) * fj], [fi * (1 - fj), fi * fj]])
    corners = grid[i : i + 2, j : j + 2]
    valid = np.isfinite(corners[..., 0])
    if weights[valid].sum() == 0:
        raise ValueError(
            f"({lat}, {lon}) lies outside the imaged swath for band {band!r} "
            f"in {mtd_tl_path}"
        )

    east, north, up = (weights[valid, None] * corners[valid]).sum(axis=0)
    zenith = np.degrees(np.arctan2(np.hypot(east, north), up))
    azimuth = np.degrees(np.arctan2(north, east)) % 360.0

    return AngularViewing(zenith=float(zenith), azimuth=float(azimuth))


def platform_from_tile_metadata(mtd_tl_path: str | Path) -> SatellitePlatform:
    """Reads which Sentinel-2 satellite (2A/2B) captured a tile from its MTD_TL.xml.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file.

    Returns:
        SatellitePlatform.SENTINEL_2A or SENTINEL_2B, from the TILE_ID prefix.
    """
    root = ET.parse(str(mtd_tl_path)).getroot()
    tile_id = root.find(".//TILE_ID").text
    if tile_id.startswith("S2A"):
        return SatellitePlatform.SENTINEL_2A
    if tile_id.startswith("S2B"):
        return SatellitePlatform.SENTINEL_2B
    raise ValueError(f"Unrecognized Sentinel-2 platform in TILE_ID {tile_id!r}")


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
        lat, lon: Target coordinates.
        cache_dir: Directory the downloaded (or previously cached) file.
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
