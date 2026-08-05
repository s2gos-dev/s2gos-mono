from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from .viewing import AngularViewing

# Sentinel-2 MTD_TL.xml identifies bands by a numeric "bandId" (0-12).
# This is the fixed order ESA uses; index == bandId. It matches the
# band strings accepted by SentinelMSIBand (see config/sensors.py).
_MSI_BAND_ID_ORDER = [
    "1", "2", "3", "4", "5", "6", "7", "8", "8a", "9", "10", "11", "12",
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