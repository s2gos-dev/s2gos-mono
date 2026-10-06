from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

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


def view_geometry_from_tile_metadata(
    mtd_tl_path: str | Path, band: str
) -> AngularViewing:
    """Reads the tile-mean viewing zenith/azimuth angle for one band from MTD_TL.xml.

    Uses the Mean_Viewing_Incidence_Angle (one value per band, averaged over
    all detectors) rather than the per-detector angle grids. Applies the same
    skyfield-to-Eradiate azimuth conversion used by
    DirectionalIllumination.from_date_and_location, so sun and view azimuths
    share one convention.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file.
        band: Band identifier as used by SentinelMSIBand (e.g. "2", "8a").

    Returns:
        An AngularViewing with zenith/azimuth in Eradiate convention.

    Raises:
        ValueError: If the band is unknown or has no entry in the file.
    """
    if band not in _MSI_BAND_ID_ORDER:
        raise ValueError(f"Unknown MSI band {band!r}")
    band_id = _MSI_BAND_ID_ORDER.index(band)

    root = ET.parse(str(mtd_tl_path)).getroot()
    node = root.find(f".//Mean_Viewing_Incidence_Angle[@bandId='{band_id}']")
    if node is None:
        raise ValueError(
            f"No Mean_Viewing_Incidence_Angle for band {band!r} "
            f"(bandId {band_id}) in {mtd_tl_path}"
        )

    zenith = float(node.find("ZENITH_ANGLE").text)
    azimuth_raw = float(node.find("AZIMUTH_ANGLE").text)

    # Same convention swap as DirectionalIllumination.from_date_and_location:
    # S2 metadata azimuth: 0=North, clockwise. Eradiate: 0=East, counter-clockwise.
    eradiate_azimuth = (90.0 - azimuth_raw) % 360.0

    return AngularViewing(zenith=zenith, azimuth=eradiate_azimuth)


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
