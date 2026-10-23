"""Combine the per-band results of a Sentinel-2 MSI sensor into one EOPF-like L1C store.

The store is an ``xr.DataTree`` laid out like ESA's EOPF Sentinel-2 L1C product::

    /                                      attrs: stac_discovery, other_metadata
    measurements/reflectance/r10m|r20m|r60m  b02, b8a, ... on (y, x), float32
    conditions/geometry                    mean_sun_angles (angle),
                                           mean_viewing_incidence_angles (band, angle)
    quality/l1c_quicklook/r10m             tci (band, y, x) uint8, only when B02/B03/B04
                                           are all simulated

Each band stays on the grid it was rendered at, so nothing is resampled. All bands share
the footprint's upper-left corner, so coarse pixels nest exactly inside the fine ones as in
a real tile. Angles follow the Sentinel-2 convention (azimuth from north, clockwise).
Open with ``xr.open_datatree(path, engine="zarr")``.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.io.paths import expand_mapper
from s2gos_utils.scene.description import SceneDescription
from upath import UPath

from .._version import get_version
from ..config.illumination import (
    AbstractDirectionalIllumination,
    Illumination,
    solar_position,
)
from ..config.sensors import SatelliteSensor

logger = logging.getLogger(__name__)

AU_KM = 149_597_870.7

MSI_BANDS = {
    "b01": ("coastal", 0.020),
    "b02": ("blue", 0.065),
    "b03": ("green", 0.035),
    "b04": ("red", 0.030),
    "b05": ("rededge", 0.015),
    "b06": ("rededge", 0.015),
    "b07": ("rededge", 0.020),
    "b08": ("nir", 0.105),
    "b8a": ("nir08", 0.020),
    "b09": ("nir09", 0.020),
    "b10": ("cirrus", 0.030),
    "b11": ("swir16", 0.090),
    "b12": ("swir22", 0.180),
}
"""Common name and nominal FWHM of each MSI band, as in the EOPF L1C STAC ``bands``.

As in EOPF, ``full_width_half_max`` is in µm while ``center_wavelength`` is in nm.
"""

SUN_DISTANCE_FACTOR_KEY = (
    "reflectance_correction_factor_from_the_Sun-Earth_distance_variation_computed_"
    "using_the_acquisition_date"
)
"""EOPF ``other_metadata`` key of the Earth-Sun distance factor U = (1 AU / d)²."""

TCI_MAX_REFLECTANCE = 0.357
"""TOA reflectance mapped to 255 in the TCI quicklook, linear from 0.

ESA does not publish the L1C TCI stretch. This value reproduces the ``tci`` of EOPF L1C
samples to within one count.
"""


def s2_azimuth(eradiate_azimuth: float) -> float:
    """Convert an Eradiate azimuth (0 = east, CCW) to Sentinel-2's (0 = north, CW)."""
    return (90.0 - eradiate_azimuth) % 360.0


def band_name(band: str) -> str:
    """EOPF variable name of a band: ``"b04"``, ``"b8a"``."""
    return f"b{band.zfill(2)}"


def _resolution_group(sensor: SatelliteSensor, band: str) -> str:
    px, py = sensor.pixel_size_m(band)
    return f"r{px:g}m" if math.isclose(px, py) else f"r{px:g}x{py:g}m"


def _reflectance(
    result: xr.Dataset,
    sensor: SatelliteSensor,
    band: str,
    bounds: Dict[str, float],
    crs_wkt: str,
) -> xr.DataArray:
    """Turn an Eradiate mpdistant result into a north-up band on its own grid.

    Like EOPF, the band carries its grid as STAC ``proj:*`` attributes.
    """
    if "brf_srf" not in result:
        raise ValueError(
            f"Band {band!r} of sensor '{sensor.id}' has no 'brf_srf' variable, "
            "a band SRF is required"
        )
    nx, ny = sensor.band_film_resolution(band)
    px, py = sensor.pixel_size_m(band)

    xmin, ymax = float(bounds["xmin"]), float(bounds["ymax"])
    wavelength = f" {float(result['w_srf']):.1f} nm" if "w_srf" in result else ""

    # Eradiate film rows start at the south edge, L1C rows at the north edge.
    data = result["brf_srf"].squeeze(drop=True).transpose("y_index", "x_index").values
    return xr.DataArray(
        data[::-1].astype(np.float32),
        dims=("y", "x"),
        coords={
            "x": xmin + (np.arange(nx) + 0.5) * px,
            "y": ymax - (np.arange(ny) + 0.5) * py,
        },
        attrs={
            "long_name": "TOA reflectance from MSI acquisition at spectral band "
            f"{band_name(band)}{wavelength}",
            "units": "dimensionless",
            "proj:wkt2": crs_wkt,
            "proj:shape": [ny, nx],
            "proj:transform": [px, 0.0, xmin, 0.0, -py, ymax, 0.0, 0.0, 1.0],
            "proj:bbox": [xmin, ymax - ny * py, xmin + nx * px, ymax],
        },
    )


def _geometry(sensor: SatelliteSensor, illumination: Illumination) -> xr.Dataset:
    viewing = [sensor.band_viewing_for(band) for band in sensor.bands]
    geometry = xr.Dataset(
        {
            "mean_viewing_incidence_angles": (
                ("band", "angle"),
                [[v.zenith, s2_azimuth(v.azimuth)] for v in viewing],
                {"units": "deg"},
            )
        },
        coords={
            "band": [band_name(band) for band in sensor.bands],
            "angle": ["zenith", "azimuth"],
        },
    )
    if isinstance(illumination, AbstractDirectionalIllumination):
        geometry["mean_sun_angles"] = (
            "angle",
            [illumination.zenith, s2_azimuth(illumination.azimuth)],
            {"units": "deg"},
        )
    return geometry


def _quicklook(groups: Dict[str, xr.Dataset]) -> Optional[Tuple[str, xr.Dataset]]:
    """TCI true-colour quicklook of the group holding B04/B03/B02, if any."""
    rgb = ["b04", "b03", "b02"]
    for name, group in groups.items():
        if all(band in group for band in rgb):
            reflectance = xr.concat([group[band] for band in rgb], dim="band")
            counts = np.round(reflectance * 255 / TCI_MAX_REFLECTANCE).clip(0, 255)
            tci = counts.fillna(0).astype(np.uint8).assign_coords(band=[1, 2, 3])
            proj = {
                k: v for k, v in group["b04"].attrs.items() if k.startswith("proj:")
            }
            tci.attrs = {"long_name": "TCI: True Color Image", **proj}
            return name, xr.Dataset({"tci": tci})
    return None


def _sun_distance_factor(
    illumination: Illumination, lat: float, lon: float
) -> Optional[float]:
    """U = (1 AU / d)², the factor the irradiance was scaled by, or None if it was not."""
    time = getattr(illumination, "irradiance_datetime", None)
    if time is None:
        return None
    return (AU_KM / solar_position(time, lat, lon).distance_km) ** 2


def _band_entry(
    band: str,
    result: xr.Dataset,
    illumination: Illumination,
    sun_distance_factor: Optional[float],
) -> Dict[str, Any]:
    """STAC ``bands`` entry, with the simulated SRF centre and solar illumination."""
    name = band_name(band)
    common_name, fwhm = MSI_BANDS[name]
    entry = {"name": name, "common_name": common_name, "full_width_half_max": fwhm}
    if "w_srf" in result:
        entry["center_wavelength"] = float(result["w_srf"])
    if "irradiance_srf" in result and isinstance(
        illumination, AbstractDirectionalIllumination
    ):
        # Eradiate's irradiance is E(d)·cos(SZA) in W m⁻² nm⁻¹, ESUN is E(1 AU) per µm.
        irradiance = float(result["irradiance_srf"].squeeze())
        entry["solar_illumination"] = (
            irradiance
            / math.cos(math.radians(illumination.zenith))
            / (sun_distance_factor or 1.0)
            * 1000.0
        )
    return entry


def _stac_discovery(
    sensor: SatelliteSensor,
    band_results: Dict[str, xr.Dataset],
    coords: CoordinateSystem,
    bounds: Dict[str, float],
    crs_wkt: str,
    illumination: Illumination,
    sun_distance_factor: Optional[float],
    product_id: str,
) -> Dict[str, Any]:
    xmin, ymin = float(bounds["xmin"]), float(bounds["ymin"])
    xmax, ymax = float(bounds["xmax"]), float(bounds["ymax"])
    lat, lon = coords.scene_to_latlon(
        np.array([xmin, xmax, xmax, xmin]), np.array([ymin, ymin, ymax, ymax])
    )
    ring = [[float(x), float(y)] for x, y in zip(lon, lat)]
    time = getattr(illumination, "irradiance_datetime", None)
    time = time.isoformat() if time else None

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
            "https://stac-extensions.github.io/product/v0.1.0/schema.json",
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
        ],
        "id": product_id,
        "bbox": [min(lon), min(lat), max(lon), max(lat)],
        "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]},
        "properties": {
            "datetime": time,
            "start_datetime": time,
            "end_datetime": time,
            "created": datetime.now(timezone.utc).isoformat(),
            "platform": sensor.spec.platform,
            "constellation": "sentinel-2",
            "mission": "copernicus",
            "instrument": "msi",
            "instruments": ["msi"],
            "product:type": "S02MSIL1C",
            "processing:level": "L1C",
            "processing:software": {"s2gos-simulator": get_version()},
            "proj:wkt2": crs_wkt,
            "proj:bbox": [xmin, ymin, xmax, ymax],
            "bands": [
                _band_entry(band, band_results[band], illumination, sun_distance_factor)
                for band in sensor.bands
            ],
        },
    }


def write_msi_l1c(
    sensor: SatelliteSensor,
    band_results: Dict[str, xr.Dataset],
    scene_description: SceneDescription,
    illumination: Illumination,
    metadata: Dict[str, Any],
    path: UPath,
) -> xr.DataTree:
    """Write the bands of one MSI sensor as a single EOPF-like L1C zarr store.

    Args:
        sensor: The Sentinel-2 MSI sensor that produced ``band_results``
        band_results: Post-processed Eradiate dataset per band of ``sensor``
        scene_description: Scene description (for the scene coordinate system)
        illumination: Simulation illumination (sun angles and acquisition time)
        metadata: Simulation metadata, stored as ``other_metadata``
        path: Output zarr path

    Returns:
        The written tree (see module docstring). x/y are pixel centres in scene
        metres, whose CRS is each band's ``proj:wkt2``
    """
    loc = scene_description.location
    coords = CoordinateSystem(loc["center_lat"], loc["center_lon"])
    size = sensor.target_size_km
    w_km, h_km = (size, size) if isinstance(size, (int, float)) else size
    bounds = coords.create_rectangle(
        sensor.target_center_lat, sensor.target_center_lon, w_km, h_km
    )

    crs_wkt = coords.scene_crs.to_wkt()
    groups: Dict[str, xr.Dataset] = {}
    for band in sensor.bands:
        group = groups.setdefault(_resolution_group(sensor, band), xr.Dataset())
        group[band_name(band)] = _reflectance(
            band_results[band], sensor, band, bounds, crs_wkt
        )

    nodes = {f"measurements/reflectance/{name}": ds for name, ds in groups.items()}
    nodes["conditions/geometry"] = _geometry(sensor, illumination)
    quicklook = _quicklook(groups)
    if quicklook:
        name, ds = quicklook
        nodes[f"quality/l1c_quicklook/{name}"] = ds
    tree = xr.DataTree.from_dict(nodes)
    sun_distance_factor = _sun_distance_factor(
        illumination, sensor.target_center_lat, sensor.target_center_lon
    )
    other_metadata = {
        **metadata,
        "sensor_id": sensor.id,
        "samples_per_pixel": sensor.samples_per_pixel,
        "target_center_lat": sensor.target_center_lat,
        "target_center_lon": sensor.target_center_lon,
        "target_size_km": sensor.target_size_km,
    }
    if sun_distance_factor is not None:
        other_metadata[SUN_DISTANCE_FACTOR_KEY] = sun_distance_factor
    tree.attrs = {
        "stac_discovery": _stac_discovery(
            sensor,
            band_results,
            coords,
            bounds,
            crs_wkt,
            illumination,
            sun_distance_factor,
            path.stem,
        ),
        "other_metadata": other_metadata,
    }
    tree.to_zarr(expand_mapper(path), mode="w")
    logger.info(f"Saved Sentinel-2 product ({', '.join(groups)}) → {path.name}")
    return tree
