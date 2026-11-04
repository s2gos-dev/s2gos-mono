"""Combine the per-band results of a Sentinel-2 MSI sensor into one EOPF-like L1C store.

The store is an ``xr.DataTree`` laid out like ESA's EOPF Sentinel-2 L1C product::

    /                                      attrs: stac_discovery, other_metadata
    measurements/reflectance/r10m|r20m|r60m  b02, b8a, ... on (y, x), uint16 counts
    conditions/geometry                    mean_sun_angles (angle),
                                           mean_viewing_incidence_angles (band, angle),
                                           sun_angles (angle, y, x),
                                           viewing_incidence_angles
                                           (band, detector, angle, y, x)
    conditions/mask/detector_footprint/rNNm  b02, ... uint8 detector number
    conditions/mask/l1c_classification/r60m  b00 uint8 cloud/snow bitmask
    quality/mask/r10m|r20m|r60m            b02, ... uint8 quality bitmask
    quality/l1c_quicklook/r10m             tci (band, y, x) uint8, only when B02/B03/B04
                                           are all simulated

Each band stays on the grid it was rendered at, so nothing is resampled. All bands share
the footprint's upper-left corner, so coarse pixels nest exactly inside the fine ones as in
a real tile. The store is in the CRS of the sensor's `projected_grid` (the tile's UTM
zone), with the pixels, the 60 m classification mask and the 5 km angle grids on that
grid's lattices. Reflectance is stored as L1C digital counts (see
`REFLECTANCE_ENCODING`) and decodes to float on open. The angle grids hold the geometry
that was simulated, one sun and one viewing direction per band. Angles follow the
Sentinel-2 convention (azimuth from north, clockwise). The masks describe a simulated
scene: one detector (`DETECTOR`), no clouds or snow, no defective pixels; only pixels
clipped to the largest storable count are flagged ``saturated_pixel``.
Open with ``xr.open_datatree(path, engine="zarr")``.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from s2gos_utils.io.paths import expand_mapper
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
"""Common name and nominal FWHM (µm) of each MSI band, as in the EOPF L1C STAC ``bands``."""

SUN_DISTANCE_FACTOR_KEY = (
    "reflectance_correction_factor_from_the_sun_earth_distance_variation_computed_"
    "using_the_acquisition_date"
)
"""EOPF ``other_metadata`` key of the Earth-Sun distance factor U = (1 AU / d)²."""

TCI_MAX_REFLECTANCE = 0.357
"""TOA reflectance mapped to 255 in the TCI quicklook, linear from 0.

ESA does not publish the L1C TCI stretch. This value reproduces the ``tci`` of EOPF L1C
samples to within one count.
"""

REFLECTANCE_ENCODING = {
    "dtype": "uint16",
    "scale_factor": 1e-4,
    "add_offset": -0.1,
    "_FillValue": 0,
}
"""On-disk encoding of L1C reflectance (processing baseline 04.00 onwards)."""

MAX_REFLECTANCE = (
    REFLECTANCE_ENCODING["add_offset"]
    + np.iinfo(REFLECTANCE_ENCODING["dtype"]).max * REFLECTANCE_ENCODING["scale_factor"]
)
"""Largest reflectance `REFLECTANCE_ENCODING` can store; larger values are clipped to it."""

SATURATED_PIXEL = 128
"""``quality/mask`` bit of a pixel clipped to `MAX_REFLECTANCE`."""

ANGLE_GRID_STEP_M = 5000.0
"""Node spacing of the L1C sun and viewing angle grids."""

DETECTOR = "d01"
"""Detector of the whole footprint, in the angle grids and the detector footprint mask."""

DETECTOR_FOOTPRINT_ATTRS = {
    "long_name": "detector footprint mask provided in the final reference frame "
    "(ground geometry)",
    "flag_values": list(range(13)),
    "flag_meanings": ["no_detector"] + [f"d{i:02d}" for i in range(1, 13)],
}
"""EOPF attributes of ``conditions/mask/detector_footprint`` bands."""

QUALITY_MASK_ATTRS = {
    "long_name": "quality mask provided in the final reference frame (ground geometry)",
    "flag_masks": [1, 2, 4, 8, 16, 32, 64, 128],
    "flag_meanings": [
        "anc_lost",
        "anc_deg",
        "msi_lost",
        "msi_deg",
        "defective_pixel",
        "no_data_pixel",
        "partially_corrected_pixel",
        "saturated_pixel",
    ],
}
"""EOPF attributes of ``quality/mask`` bands."""

CLASSIFICATION_ATTRS = {
    "long_name": "cloud classification bitmask provided in the final reference frame "
    "(ground geometry)",
    "flag_masks": [1, 2, 4, 128],
    "flag_meanings": ["opaque", "cirrus", "snow_ice", "novalue"],
}
"""EOPF attributes of the ``conditions/mask/l1c_classification/r60m`` ``b00`` mask."""

CLASSIFICATION_PIXEL_M = 60.0
"""Pixel size of the cloud classification mask."""


def s2_azimuth(eradiate_azimuth: float) -> float:
    """Convert an Eradiate azimuth (0 = east, CCW) to Sentinel-2's (0 = north, CW)."""
    return (90.0 - eradiate_azimuth) % 360.0


def band_name(band: str) -> str:
    """EOPF variable name of a band: ``"b04"``, ``"b8a"``."""
    return f"b{band.zfill(2)}"


def _resolution(px: float, py: float) -> str:
    return f"{px:g}m" if math.isclose(px, py) else f"{px:g}x{py:g}m"


def _resolution_group(sensor: SatelliteSensor, band: str) -> str:
    return f"r{_resolution(*sensor.pixel_size_m(band))}"


def _pixel_coords(
    bounds: Dict[str, float], nx: int, ny: int, px: float, py: float
) -> Dict[str, xr.DataArray]:
    """Pixel-centre x/y of an ``nx`` × ``ny`` grid from the footprint's upper-left corner."""
    xmin, ymax = float(bounds["xmin"]), float(bounds["ymax"])
    return {
        axis: xr.DataArray(
            values,
            dims=axis,
            attrs={
                "long_name": f"{axis} coordinates of image at {_resolution(px, py)} "
                "in meters from up-left pixel",
                "units": "m",
            },
        )
        for axis, values in (
            ("x", xmin + (np.arange(nx) + 0.5) * px),
            ("y", ymax - (np.arange(ny) + 0.5) * py),
        )
    }


def _reflectance(
    result: xr.Dataset,
    sensor: SatelliteSensor,
    band: str,
    bounds: Dict[str, float],
    crs_wkt: str,
) -> xr.DataArray:
    """Turn an Eradiate mpdistant result into a north-up band on its own grid.

    Like EOPF, the band carries its grid as STAC ``proj:*`` attributes. Values are
    clipped to the range `REFLECTANCE_ENCODING` can store.
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
    min_reflectance = (
        REFLECTANCE_ENCODING["add_offset"] + REFLECTANCE_ENCODING["scale_factor"]
    )
    valid_max = np.iinfo(REFLECTANCE_ENCODING["dtype"]).max
    band_data = xr.DataArray(
        np.clip(data[::-1], min_reflectance, MAX_REFLECTANCE),
        dims=("y", "x"),
        coords=_pixel_coords(bounds, nx, ny, px, py),
        attrs={
            "long_name": "TOA reflectance from MSI acquisition at spectral band "
            f"{band_name(band)}{wavelength}",
            "units": "digital_counts",
            "valid_min": 1,
            "valid_max": int(valid_max),
            "proj:wkt2": crs_wkt,
            "proj:shape": [ny, nx],
            "proj:transform": [px, 0.0, xmin, 0.0, -py, ymax, 0.0, 0.0, 1.0],
            "proj:bbox": [xmin, ymax - ny * py, xmin + nx * px, ymax],
        },
    )
    band_data.encoding = dict(REFLECTANCE_ENCODING)
    return band_data


def _lattice_bounds(
    bounds: Dict[str, float], origin: Tuple[float, float], step: float
) -> Dict[str, float]:
    """Smallest rectangle covering ``bounds`` whose edges lie on the ``step`` lattice
    through ``origin``."""
    x0, y0 = origin
    return {
        "xmin": x0 + math.floor((bounds["xmin"] - x0) / step) * step,
        "xmax": x0 + math.ceil((bounds["xmax"] - x0) / step) * step,
        "ymin": y0 - math.ceil((y0 - bounds["ymin"]) / step) * step,
        "ymax": y0 - math.floor((y0 - bounds["ymax"]) / step) * step,
    }


def _angle_grid_nodes(
    bounds: Dict[str, float], origin: Tuple[float, float]
) -> Dict[str, np.ndarray]:
    """Nodes of the angle grid lattice through ``origin`` that cover the footprint."""
    step = ANGLE_GRID_STEP_M
    b = _lattice_bounds(bounds, origin, step)
    return {
        "x": np.arange(b["xmin"], b["xmax"] + step / 2, step),
        "y": np.arange(b["ymax"], b["ymin"] - step / 2, -step),
    }


def _geometry(
    sensor: SatelliteSensor,
    illumination: Illumination,
    bounds: Dict[str, float],
    origin: Tuple[float, float],
) -> xr.Dataset:
    viewing = np.array(
        [
            [v.zenith, s2_azimuth(v.azimuth)]
            for v in map(sensor.band_viewing_for, sensor.bands)
        ]
    )
    nodes = _angle_grid_nodes(bounds, origin)
    shape = (len(nodes["y"]), len(nodes["x"]))

    def attrs(long_name: str) -> Dict[str, str]:
        return {"long_name": long_name, "units": "degrees"}

    geometry = xr.Dataset(
        {
            "mean_viewing_incidence_angles": (
                ("band", "angle"),
                viewing,
                attrs("mean viewing incidence angles (all detectors)"),
            ),
            "viewing_incidence_angles": (
                ("band", "detector", "angle", "y", "x"),
                np.broadcast_to(
                    viewing[:, None, :, None, None], (*viewing.shape[:1], 1, 2, *shape)
                ),
                attrs("viewing incidence angles"),
            ),
        },
        coords={
            "band": ("band", [band_name(band) for band in sensor.bands]),
            "detector": (
                "detector",
                [DETECTOR],
                {"long_name": "detector module number"},
            ),
            "angle": (
                "angle",
                ["zenith", "azimuth"],
                {"long_name": "zenith and azimuth angles"},
            ),
            **{
                axis: (
                    axis,
                    values,
                    {
                        "long_name": f"{axis} coordinates of angles grid, in steps of "
                        f"{ANGLE_GRID_STEP_M / 1000:g}km from up-left pixel",
                        "units": "m",
                    },
                )
                for axis, values in nodes.items()
            },
        },
    )
    geometry["band"].attrs["long_name"] = "spectral band number"
    if isinstance(illumination, AbstractDirectionalIllumination):
        sun = np.array([illumination.zenith, s2_azimuth(illumination.azimuth)])
        geometry["mean_sun_angles"] = (
            "angle",
            sun,
            attrs("mean solar angles (all detectors, all bands)"),
        )
        geometry["sun_angles"] = (
            ("angle", "y", "x"),
            np.broadcast_to(sun[:, None, None], (2, *shape)),
            attrs("solar angles"),
        )
    return geometry


def _masks(
    groups: Dict[str, xr.Dataset],
    bounds: Dict[str, float],
    origin: Tuple[float, float],
) -> Dict[str, xr.Dataset]:
    """L1C detector footprint, quality and cloud classification masks (see module docstring)."""

    def mask(like: xr.DataArray, values, attrs: Dict[str, Any]) -> xr.DataArray:
        return xr.DataArray(
            np.broadcast_to(values, like.shape).astype(np.uint8),
            dims=like.dims,
            coords=like.coords,
            attrs=attrs,
        )

    detector = int(DETECTOR[1:])
    nodes = {}
    for name, group in groups.items():
        nodes[f"conditions/mask/detector_footprint/{name}"] = xr.Dataset(
            {b: mask(group[b], detector, DETECTOR_FOOTPRINT_ATTRS) for b in group}
        )
        nodes[f"quality/mask/{name}"] = xr.Dataset(
            {
                b: mask(
                    group[b],
                    np.where(group[b] >= MAX_REFLECTANCE, SATURATED_PIXEL, 0),
                    QUALITY_MASK_ATTRS,
                )
                for b in group
            }
        )

    step = CLASSIFICATION_PIXEL_M
    lattice = _lattice_bounds(bounds, origin, step)
    coords = _pixel_coords(
        lattice,
        round((lattice["xmax"] - lattice["xmin"]) / step),
        round((lattice["ymax"] - lattice["ymin"]) / step),
        step,
        step,
    )
    classification = xr.DataArray(
        np.zeros((coords["y"].size, coords["x"].size), dtype=np.uint8),
        dims=("y", "x"),
        coords=coords,
        attrs=CLASSIFICATION_ATTRS,
    )
    nodes[f"conditions/mask/l1c_classification/r{step:g}m"] = xr.Dataset(
        {"b00": classification}
    )
    return nodes


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
    sensor: SatelliteSensor,
    band: str,
    result: xr.Dataset,
    illumination: Illumination,
    sun_distance_factor: Optional[float],
) -> Dict[str, Any]:
    """STAC ``bands`` entry, with the simulated SRF centre and solar illumination."""
    name = band_name(band)
    common_name, fwhm = MSI_BANDS[name]
    entry = {
        "name": name,
        "eo:common_name": common_name,
        "eo:full_width_half_max": fwhm,
        "raster:spatial_resolution": f"{min(sensor.pixel_size_m(band)):g}",
    }
    if "w_srf" in result:
        entry["eo:center_wavelength"] = float(result["w_srf"]) / 1000.0
    if "irradiance_srf" in result and isinstance(
        illumination, AbstractDirectionalIllumination
    ):
        # Eradiate's irradiance is E(d)·cos(SZA) in W m⁻² nm⁻¹, ESUN is E(1 AU) per µm.
        irradiance = float(result["irradiance_srf"].squeeze())
        entry["eo:solar_illumination"] = (
            irradiance
            / math.cos(math.radians(illumination.zenith))
            / (sun_distance_factor or 1.0)
            * 1000.0
        )
    return entry


def _band_description(
    sensor: SatelliteSensor, band_results: Dict[str, xr.Dataset]
) -> Dict[str, Dict[str, Any]]:
    """EOPF ``other_metadata`` ``band_description``, keyed like ``"01"``, ``"8A"``."""
    description = {}
    for band in sensor.bands:
        entry = {
            "bandwidth": f"{MSI_BANDS[band_name(band)][1] * 1000:.1f}",
            "units": "nm",
        }
        if "w_srf" in band_results[band]:
            entry["central_wavelength"] = float(band_results[band]["w_srf"])
        description[band.zfill(2).upper()] = entry
    return description


def _stac_discovery(
    sensor: SatelliteSensor,
    band_results: Dict[str, xr.Dataset],
    bounds: Dict[str, float],
    crs: CRS,
    illumination: Illumination,
    sun_distance_factor: Optional[float],
    product_id: str,
) -> Dict[str, Any]:
    xmin, ymin = float(bounds["xmin"]), float(bounds["ymin"])
    xmax, ymax = float(bounds["xmax"]), float(bounds["ymax"])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]
    )
    ring = [[float(x), float(y)] for x, y in zip(lon, lat)]
    time = getattr(illumination, "irradiance_datetime", None)
    time = time.isoformat() if time else None
    epsg = crs.to_epsg()
    pixel = max(max(sensor.pixel_size_m(band)) for band in sensor.bands)

    stac = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
            "https://stac-extensions.github.io/product/v0.1.0/schema.json",
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
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
            "proj:wkt2": crs.to_wkt(),
            "proj:bbox": [xmin, ymin, xmax, ymax],
            "proj:transform": [pixel, 0.0, xmin, 0.0, -pixel, ymax, 0.0, 0.0, 1.0],
            "gsd": min(min(sensor.pixel_size_m(band)) for band in sensor.bands),
            "eo:cloud_cover": 0.0,
            "bands": [
                _band_entry(
                    sensor, band, band_results[band], illumination, sun_distance_factor
                )
                for band in sensor.bands
            ],
        },
    }
    if epsg is not None:
        stac["properties"]["proj:code"] = f"EPSG:{epsg}"
    return stac


def product_name(
    sensor: SatelliteSensor, illumination: Illumination, simulation_name: str
) -> str:
    """Store name, e.g. ``S2A_MSIL1C_20240622T091715_gobabeb_s2.zarr``.

    It starts like an ESA L1C product name, which EOPF readers such as xarray-eopf
    use to recognise the product type. The sensing time is left out when the
    illumination has none, and taken as UTC when naive.
    """
    parts = [f"S2{sensor.spec.platform[-1].upper()}", "MSIL1C"]
    time = getattr(illumination, "irradiance_datetime", None)
    if time is not None:
        if time.tzinfo is not None:
            time = time.astimezone(timezone.utc)
        parts.append(f"{time:%Y%m%dT%H%M%S}")
    return "_".join([*parts, simulation_name, sensor.id]) + ".zarr"


def write_msi_l1c(
    sensor: SatelliteSensor,
    band_results: Dict[str, xr.Dataset],
    illumination: Illumination,
    metadata: Dict[str, Any],
    output_dir: UPath,
    simulation_name: str,
) -> xr.DataTree:
    """Write the bands of one MSI sensor as a single EOPF-like L1C zarr store.

    Args:
        sensor: The Sentinel-2 MSI sensor that produced ``band_results``
        band_results: Post-processed Eradiate dataset per band of ``sensor``
        illumination: Simulation illumination (sun angles and acquisition time)
        metadata: Simulation metadata, stored as ``other_metadata``
        output_dir: Directory of the store, named by `product_name`
        simulation_name: Name of the simulation, part of the store name

    Returns:
        The stored tree, opened lazily (see module docstring). x/y are pixel centres
        in the CRS of ``sensor.projected_grid``, which each band's ``proj:wkt2`` holds
    """
    path = output_dir / product_name(sensor, illumination, simulation_name)
    bounds, grid = sensor.grid_bounds(), sensor.projected_grid
    crs, origin = CRS(grid.crs), grid.origin

    crs_wkt = crs.to_wkt()
    groups: Dict[str, xr.Dataset] = {}
    for band in sensor.bands:
        group = groups.setdefault(_resolution_group(sensor, band), xr.Dataset())
        group[band_name(band)] = _reflectance(
            band_results[band], sensor, band, bounds, crs_wkt
        )

    nodes = {f"measurements/reflectance/{name}": ds for name, ds in groups.items()}
    nodes["conditions/geometry"] = _geometry(sensor, illumination, bounds, origin)
    nodes.update(_masks(groups, bounds, origin))
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
        "horizontal_crs_code": f"EPSG:{crs.to_epsg()}",
        "horizontal_crs_name": crs.name,
        "band_description": _band_description(sensor, band_results),
    }
    if isinstance(illumination, AbstractDirectionalIllumination):
        other_metadata.update(
            {
                "mean_sun_zenith_angle_in_deg_for_all_bands_all_detectors": (
                    illumination.zenith
                ),
                "mean_sun_azimuth_angle_in_deg_for_all_bands_all_detectors": (
                    s2_azimuth(illumination.azimuth)
                ),
            }
        )
    if sun_distance_factor is not None:
        other_metadata[SUN_DISTANCE_FACTOR_KEY] = sun_distance_factor
    tree.attrs = {
        "stac_discovery": _stac_discovery(
            sensor,
            band_results,
            bounds,
            crs,
            illumination,
            sun_distance_factor,
            path.stem,
        ),
        "other_metadata": other_metadata,
    }
    tree.to_zarr(expand_mapper(path), mode="w")
    logger.info(f"Saved Sentinel-2 product ({', '.join(groups)}) → {path.name}")
    return xr.open_datatree(expand_mapper(path), engine="zarr")
