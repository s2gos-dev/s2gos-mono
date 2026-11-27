from __future__ import annotations

import logging
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import Transformer

from ._fetch import FootprintOutsideProduct, closest_fitting
from ..config.illumination import DirectionalIllumination
from ..config.instruments import MSI_BANDS
from ..config.sensors import ProjectedGrid, SatelliteSensor
from ..config.viewing import AngularViewing

logger = logging.getLogger(__name__)

# CDSE STAC catalog serving Sentinel-2 tile metadata assets.
_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
_STAC_COLLECTION = "sentinel-2-l1c"  # L1C, not L2A, to match TOA sensor setups.


class TileGeometry(NamedTuple):
    """Acquisition geometry of a Sentinel-2 tile at a target point."""

    time: datetime
    """The tile's SENSING_TIME (UTC-aware)."""

    instrument: str
    """Instrument key in `INSTRUMENTS`, e.g. 'sentinel_2a-msi'."""

    viewing: dict[str, AngularViewing]
    """Per band, the view zenith/azimuth at the target, Eradiate convention."""

    grid: ProjectedGrid
    """The tile's CRS and upper-left corner."""


class TileFootprint(NamedTuple):
    """A footprint on a Sentinel-2 tile's pixels."""

    center_lat: float
    center_lon: float

    bounds: dict[str, float]
    """`xmin`, `xmax`, `ymin` and `ymax` in the tile's CRS, as
    `SatelliteSensor.grid_bounds`. They crop the real bands to match."""


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
    cover the node, or NaN where none does."""
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


def _band_angles(root: ET.Element, band: str) -> tuple[np.ndarray, float, float]:
    """One band's view vector grid, with its row and column spacing in meters."""
    if band not in MSI_BANDS:
        raise ValueError(
            f"Unknown MSI band {band!r}, expected one of {list(MSI_BANDS)}"
        )
    band_id = MSI_BANDS[band].band_id

    grids = root.findall(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']")
    if not grids:
        raise ValueError(
            f"No Viewing_Incidence_Angles_Grids for band {band!r} (bandId {band_id})"
        )
    step = _find(grids[0], "Zenith")
    return (
        _view_vector_grid(grids),
        float(_find(step, "ROW_STEP").text),
        float(_find(step, "COL_STEP").text),
    )


def _view_geometry(root: ET.Element, band: str, x: float, y: float) -> AngularViewing:
    """Bilinearly interpolate one band's viewing angles at tile-CRS (x, y), as
    unit vectors so the azimuth wrap needs no special case."""
    grid, row_step, col_step = _band_angles(root, band)
    x0, y0 = _tile_grid(root).origin
    row, col = (y0 - y) / row_step, (x - x0) / col_step

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


def _instrument(root: ET.Element) -> str:
    """Key of the tile's MSI in `INSTRUMENTS`, from the TILE_ID prefix."""
    tile_id = _find(root, ".//TILE_ID").text
    if tile_id.startswith("S2A"):
        return "sentinel_2a-msi"
    if tile_id.startswith("S2B"):
        return "sentinel_2b-msi"
    raise ValueError(f"Unrecognized Sentinel-2 platform in TILE_ID {tile_id!r}")


def _tile_grid(root: ET.Element) -> ProjectedGrid:
    origin = _find(root, ".//Tile_Geocoding/Geoposition[@resolution='10']")
    return ProjectedGrid(
        crs=_find(root, ".//Tile_Geocoding/HORIZONTAL_CS_CODE").text,
        origin=(float(_find(origin, "ULX").text), float(_find(origin, "ULY").text)),
    )


def read_tile_metadata(
    mtd_tl_path: str | Path, lat: float, lon: float, bands: list[str]
) -> TileGeometry:
    """Read the acquisition geometry at a point from a Sentinel-2 MTD_TL.xml.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file, e.g. from `fetch_tile_metadata`.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        bands: MSI band identifiers (e.g. "2", "8a").

    Returns:
        The tile's sensing time, its instrument from the TILE_ID prefix and its grid,
        and per band the viewing angles at (lat, lon) interpolated on the per-detector
        angle grids.

    Raises:
        ValueError: If the platform is not 2A/2B, a band is unknown or has no
            angle grids, or the point lies outside the tile or the imaged swath.
    """
    root = ET.parse(str(mtd_tl_path)).getroot()
    grid = _tile_grid(root)
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(
        lon, lat
    )
    return TileGeometry(
        time=_sensing_time(root),
        instrument=_instrument(root),
        viewing={band: _view_geometry(root, band, x, y) for band in bands},
        grid=grid,
    )


def read_tile_footprint(
    mtd_tl_path: str | Path,
    lat: float,
    lon: float,
    bands: list[str],
    size_km: float | tuple[float, float],
) -> TileFootprint:
    """Place a footprint on a tile's pixels, checking that the tile holds all of it.

    The footprint moves onto the coarsest band's lattice. It must then lie inside the
    tile, and every corner must have the four surrounding nodes of each band's 5 km
    angle grid covered, because real tiles are often only partly imaged. The angle grids
    only approximate the edge of the imaged area, so an accepted footprint can reach
    about 200 m into missing data, and a footprint up to about 7 km inside the imaged
    area can be rejected.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file, e.g. from `fetch_tile_metadata`.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        bands: MSI band identifiers (e.g. "2", "8a").
        size_km: Footprint size (km). Float for square, tuple for (width, height).

    Returns:
        The footprint's center and its bounds in the tile's CRS.

    Raises:
        FootprintOutsideProduct: If the tile doesn't hold the footprint.
        ValueError: If a band is unknown or has no angle grids.
    """
    root = ET.parse(str(mtd_tl_path)).getroot()
    grid = _tile_grid(root)
    angles = {band: _band_angles(root, band) for band in bands}
    width, height = (size_km, size_km) if isinstance(size_km, (int, float)) else size_km
    step = max(MSI_BANDS[band].gsd_m for band in bands)
    bounds = grid.snapped_bounds(lat, lon, (width * 1000, height * 1000), (step, step))

    x0, y0 = grid.origin
    size = _find(root, ".//Tile_Geocoding/Size[@resolution='10']")
    tile_width = int(_find(size, "NCOLS").text) * 10.0
    tile_height = int(_find(size, "NROWS").text) * 10.0
    if (
        bounds["xmin"] < x0
        or bounds["xmax"] > x0 + tile_width
        or bounds["ymax"] > y0
        or bounds["ymin"] < y0 - tile_height
    ):
        raise FootprintOutsideProduct(
            f"Footprint at ({lat}, {lon}) leaves the tile: {bounds}"
        )

    for band, (vectors, row_step, col_step) in angles.items():
        covered = np.isfinite(vectors[..., 0])
        rows, cols = covered.shape
        for x in (bounds["xmin"], bounds["xmax"]):
            for y in (bounds["ymin"], bounds["ymax"]):
                row, col = (y0 - y) / row_step, (x - x0) / col_step
                i, j = min(int(row), rows - 2), min(int(col), cols - 2)
                if not (
                    row <= rows - 1
                    and col <= cols - 1
                    and covered[i : i + 2, j : j + 2].all()
                ):
                    raise FootprintOutsideProduct(
                        f"Footprint at ({lat}, {lon}) leaves the imaged swath of "
                        f"band {band!r}: corner ({x:.0f}, {y:.0f}) has no data"
                    )

    center_lon, center_lat = Transformer.from_crs(
        grid.crs, "EPSG:4326", always_xy=True
    ).transform(
        (bounds["xmin"] + bounds["xmax"]) / 2, (bounds["ymin"] + bounds["ymax"]) / 2
    )
    return TileFootprint(center_lat, center_lon, bounds)


def create_acquisition(
    mtd_tl_path: str | Path,
    target_center_lat: float,
    target_center_lon: float,
    bands: list[str],
    target_size_km: float | tuple[float, float],
    samples_per_pixel: int = 64,
    irradiance_dataset: str = "coddington_2022-1_nm",
    **kwargs,
) -> tuple[DirectionalIllumination, SatelliteSensor]:
    """Create the illumination and MSI sensor of a real Sentinel-2 acquisition.

    Sun and view geometry both come from the tile's MTD_TL.xml, at the footprint's center
    on the tile's lattice: the sun is computed at the tile's SENSING_TIME, and each band
    has its own view angles. Use `read_tile_metadata` directly for custom sensors or
    illumination.

    The sensor images the L1C tile's own pixels: its footprint sits on the tile's lattice
    and follows the UTM grid, it is orthorectified, and each band is at its native GSD.
    `output_format="eopf"` also writes the bands as one EOPF L1C product.

    Args:
        mtd_tl_path: Path to an MTD_TL.xml file, e.g. from `fetch_tile_metadata`.
        target_center_lat: Target center latitude (WGS84 decimal degrees)
        target_center_lon: Target center longitude (WGS84 decimal degrees)
        bands: MSI band identifiers (e.g. "2", "8a").
        target_size_km: Target area size (km). Float for square, tuple for
            (width, height). Must be a whole multiple of every band's GSD.
        samples_per_pixel: Monte Carlo samples per pixel. Default 64
        irradiance_dataset: Name of the solar irradiance dataset to use.
            Default `coddington_2022-1_nm` (TSIS-1 HSRS).
        **kwargs: Additional sensor parameters, e.g. `output_format="eopf"`, other
            than the fields set here (`grid` included).

    Returns:
        The illumination and one SatelliteSensor imaging every band, each with
        its own viewing angles in `band_viewing`.

    Raises:
        FootprintOutsideProduct: If the tile doesn't hold the footprint, see
            `read_tile_footprint`.
        ValueError: As raised by `read_tile_metadata`, or if the footprint is
            not a whole multiple of a band's GSD.
    """
    footprint = read_tile_footprint(
        mtd_tl_path, target_center_lat, target_center_lon, bands, target_size_km
    )
    geometry = read_tile_metadata(
        mtd_tl_path, footprint.center_lat, footprint.center_lon, bands
    )
    logger.info(
        "Sentinel-2 acquisition: %s sensed at %s, tile x %.0f:%.0f and y %.0f:%.0f",
        geometry.instrument,
        geometry.time,
        footprint.bounds["xmin"],
        footprint.bounds["xmax"],
        footprint.bounds["ymin"],
        footprint.bounds["ymax"],
    )
    kwargs.setdefault("orthorectified", True)

    illumination = DirectionalIllumination.from_date_and_location(
        geometry.time, footprint.center_lat, footprint.center_lon, irradiance_dataset
    )
    sensor = SatelliteSensor(
        instrument=geometry.instrument,
        bands=bands,
        viewing=geometry.viewing[bands[0]],
        band_viewing=geometry.viewing,
        grid=geometry.grid,
        target_center_lat=target_center_lat,
        target_center_lon=target_center_lon,
        target_size_km=target_size_km,
        samples_per_pixel=samples_per_pixel,
        **kwargs,
    )
    return illumination, sensor


def fetch_tile_metadata(
    date: str,
    lat: float,
    lon: float,
    bands: list[str],
    size_km: float | tuple[float, float],
    cache_dir: str | Path,
    site_name: str | None = None,
    search_window_days: int = 6,
    credential_id: str | None = None,
) -> Path:
    """Download or reuse the MTD_TL.xml of the closest tile holding a footprint.

    A tile that doesn't hold the footprint (see `read_tile_footprint`) is skipped. A
    warning names the pass used when it is on another day than `date` or a closer tile
    was skipped, because the view geometry then differs.

    Args:
        date: Acquisition date "YYYY-MM-DD".
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        bands: MSI band identifiers (e.g. "2", "8a").
        size_km: Footprint size (km). Float for square, tuple for (width, height).
        cache_dir: Directory to keep the downloaded files in, or read the cached
            ones from.
        site_name: Prefix for the cache directory name, which always includes the
            lat/lon.
        search_window_days: +-days around `date` to search for the closest tile.
        credential_id: Id of an s3 credential (.secrets.yaml) for the Copernicus
            'eodata' bucket. Defaults to the ambient AWS credential chain.

    Returns:
        Local path to the tile's MTD_TL.xml.

    Raises:
        ValueError: A band is unknown or `credential_id` is not an s3 credential.
        RuntimeError: No Sentinel-2A/2B L1C tile near `date` at (lat, lon) holds the
            footprint.
    """
    anchor = datetime.strptime(date, "%Y-%m-%d").date()
    prefix = f"{site_name}_" if site_name else ""
    cache = Path(cache_dir) / f"S2_{prefix}{lat:.4f}_{lon:.4f}_{anchor:%Y%m%d}"
    return closest_fitting(
        cache,
        anchor,
        lambda path: read_tile_footprint(path, lat, lon, bands, size_km),
        lambda: _search_tiles(
            anchor, lat, lon, cache, search_window_days, credential_id
        ),
    )


def _search_tiles(
    anchor: date_type,
    lat: float,
    lon: float,
    cache: Path,
    search_window_days: int,
    credential_id: str | None,
) -> dict:
    """Downloads of the Sentinel-2A/2B L1C tiles over a point, by file name."""
    import pystac_client
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
    platforms = {"sentinel-2a", "sentinel-2b"}
    items = [it for it in search.items() if it.properties.get("platform") in platforms]
    if not items:
        raise RuntimeError(
            f"No Sentinel-2A/2B L1C tile found near {anchor} (+-{search_window_days} d) "
            f"at ({lat}, {lon})."
        )
    return {
        f"{item.id}.xml": partial(
            _download_tile, item, cache / f"{item.id}.xml", credential_id
        )
        for item in items
    }


def _download_tile(item, path: Path, credential_id: str | None) -> Path:
    from s2gos_utils.io.paths import PathRef

    logger.info("Downloading MTD_TL.xml of %s", item.id)
    href = item.assets["granule_metadata"].href
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.part")
    partial_path.write_bytes(PathRef(href=href, cid=credential_id).upath.read_bytes())
    partial_path.rename(path)
    return path
