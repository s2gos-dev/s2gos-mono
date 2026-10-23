from __future__ import annotations

import logging
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import Transformer

from ..config.illumination import DirectionalIllumination
from ..config.sensors import (
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SentinelMSIBand,
)
from ..config.viewing import AngularViewing

logger = logging.getLogger(__name__)

# CDSE STAC catalog serving Sentinel-2 tile metadata assets.
_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_STAC_COLLECTION = "sentinel-2-l1c"  # L1C, not L2A, to match TOA sensor setups.


def _find(node: ET.Element, path: str) -> ET.Element:
    found = node.find(path)
    if found is None:
        raise ValueError(f"MTD_TL.xml has no {path!r} element")
    return found


def _sensing_time(root: ET.Element) -> datetime:
    sensing_time = _find(root, ".//SENSING_TIME").text
    return datetime.strptime(sensing_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _angle_grid(node: ET.Element) -> np.ndarray:
    return np.array([v.text.split() for v in node.iter("VALUES")], dtype=float)


def _view_vector_grid(grids: list[ET.Element]) -> np.ndarray:
    """Per-node (east, north, up) view vectors averaged over the detectors that
    cover the node; NaN where none does."""
    vectors = []
    for detector in grids:
        zenith = np.deg2rad(_angle_grid(_find(detector, "Zenith")))
        azimuth = np.deg2rad(_angle_grid(_find(detector, "Azimuth")))
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
    count = covered.sum(axis=0)
    grid = (
        np.where(covered[..., None], vectors, 0.0).sum(axis=0)
        / np.maximum(count, 1)[..., None]
    )
    grid[count == 0] = np.nan
    return grid


def _view_geometry(root: ET.Element, band: str, x: float, y: float) -> AngularViewing:
    """Bilinearly interpolates one band's viewing angles at tile-CRS (x, y), as
    unit vectors so the azimuth wrap needs no special case."""
    band_id = list(SentinelMSIBand).index(SentinelMSIBand(band))

    grids = root.findall(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']")
    if not grids:
        raise ValueError(
            f"No Viewing_Incidence_Angles_Grids for band {band!r} (bandId {band_id})"
        )
    grid = _view_vector_grid(grids)

    origin = _find(root, ".//Tile_Geocoding/Geoposition[@resolution='10']")
    step = _find(grids[0], "Zenith")
    row = (float(_find(origin, "ULY").text) - y) / float(_find(step, "ROW_STEP").text)
    col = (x - float(_find(origin, "ULX").text)) / float(_find(step, "COL_STEP").text)

    rows, cols = grid.shape[:2]
    if not (0 <= row <= rows - 1 and 0 <= col <= cols - 1):
        raise ValueError(f"Target ({x:.0f}, {y:.0f}) lies outside the tile")
    i, j = min(int(row), rows - 2), min(int(col), cols - 2)
    fi, fj = row - i, col - j
    weights = np.array([[(1 - fi) * (1 - fj), (1 - fi) * fj], [fi * (1 - fj), fi * fj]])
    corners = grid[i : i + 2, j : j + 2]
    valid = np.isfinite(corners[..., 0])
    if weights[valid].sum() == 0:
        raise ValueError(
            f"Target ({x:.0f}, {y:.0f}) lies outside the imaged swath for band {band!r}"
        )

    east, north, up = (weights[valid, None] * corners[valid]).sum(axis=0)
    zenith = np.degrees(np.arctan2(np.hypot(east, north), up))
    azimuth = np.degrees(np.arctan2(north, east)) % 360.0

    return AngularViewing(zenith=float(zenith), azimuth=float(azimuth))


def _platform(root: ET.Element) -> SatellitePlatform:
    tile_id = _find(root, ".//TILE_ID").text
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
    crs = _find(root, ".//Tile_Geocoding/HORIZONTAL_CS_CODE").text
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return (
        _sensing_time(root),
        _platform(root),
        {band: _view_geometry(root, band, x, y) for band in bands},
    )


def create_acquisition(
    mtd_tl_path: str | Path,
    target_center_lat: float,
    target_center_lon: float,
    bands: list[str],
    target_size_km: float | tuple[float, float],
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
    credential_id: str | None = None,
) -> Path:
    """Downloads (or reuses a cached) MTD_TL.xml for the S2 tile closest to a date.

    Logs a warning when the tile was sensed on another day than `date`, which
    means a different orbit and so a different view geometry.

    Args:
        date: Acquisition date "YYYY-MM-DD".
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        cache_dir: Directory to store the downloaded file in, or read the cached
            one from.
        site_name: Prefix for the cache filename, which always includes the lat/lon.
        search_window_days: +-days around `date` to search for the closest tile.
        credential_id: Id of an s3 credential (.secrets.yaml) for the Copernicus
            'eodata' bucket. Defaults to the ambient AWS credential chain.

    Returns:
        Local path to the tile's MTD_TL.xml.

    Raises:
        ValueError: `credential_id` is not an s3 credential.
        RuntimeError: No Sentinel-2 L1C tile found near `date` at (lat, lon).
    """
    anchor = datetime.strptime(date, "%Y-%m-%d").date()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{site_name}_" if site_name else ""
    cached_path = cache_dir / f"MTD_TL_{prefix}{lat:.4f}_{lon:.4f}_{anchor:%Y%m%d}.xml"
    if not cached_path.exists():
        _download_tile_metadata(
            anchor, lat, lon, cached_path, search_window_days, credential_id
        )

    sensed = _sensing_time(ET.parse(str(cached_path)).getroot()).date()
    if sensed != anchor:
        logger.warning(
            "No Sentinel-2 pass on %s at (%s, %s); using the %s pass from %s",
            anchor,
            lat,
            lon,
            sensed,
            cached_path,
        )
    return cached_path


def _download_tile_metadata(
    anchor: date_type,
    lat: float,
    lon: float,
    path: Path,
    search_window_days: int,
    credential_id: str | None,
) -> None:
    import pystac_client
    from s2gos_utils.io.paths import PathRef
    from s2gos_utils.setting.credentials import get_credential

    if credential_id:
        cred_type = getattr(get_credential(credential_id), "type", None)
        if cred_type != "s3":
            raise ValueError(
                f"Credential '{credential_id}' must be type 's3' for Sentinel-2 reads, "
                f"got '{cred_type}'."
            )

    catalog = pystac_client.Client.open(_STAC_URL)
    catalog.add_conforms_to("ITEM_SEARCH")

    window = timedelta(days=search_window_days)
    search = catalog.search(
        collections=[_STAC_COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{anchor - window}/{anchor + window}",
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L1C tile found near {anchor} (+-{search_window_days} d) "
            f"at ({lat}, {lon})."
        )
    item = min(items, key=lambda it: abs((it.datetime.date() - anchor).days))
    logger.info("Downloading MTD_TL.xml of %s", item.id)

    href = item.assets["granule_metadata"].href
    path.write_bytes(PathRef(href=href, cid=credential_id).upath.read_bytes())
