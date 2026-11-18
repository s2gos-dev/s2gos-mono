"""Satellite instruments a `SatelliteSensor` can carry.

An instrument is data: its platform, its name, and either its bands (multispectral) or
its spectral response (hyperspectral). Built-in instruments live in `INSTRUMENTS` and are
referenced by key. A custom instrument is passed inline, or added to `INSTRUMENTS` to
make it available by key.

A satellite is spelled `sentinel-3b` as a `platform`, `sentinel_3b-olci` as an
`INSTRUMENTS` key (like Eradiate's SRF ids), `sentinel_3b_olci_nadir` in a sensor id, and
`S3B` in the acquisition readers (like its product names).
"""

from typing import Annotated, Dict, Literal, NamedTuple, Optional, Union

from pydantic import BaseModel, Field, PositiveFloat

from .spectral import SpectralRegion, SpectralResponse, SRFType, WavelengthGrid


class Band(BaseModel):
    """One band of a multispectral instrument."""

    srf: SRFType = Field(
        ..., description="Spectral response: an Eradiate SRF id or a SpectralResponse"
    )
    gsd_m: Optional[PositiveFloat] = Field(
        None, description="Native ground sampling distance (m), the default pixel size"
    )


class MultispectralInstrument(BaseModel):
    """An instrument with discrete bands, each simulated as its own measure."""

    kind: Literal["multispectral"] = "multispectral"
    platform: str = Field(
        ..., description="Platform carrying the instrument, e.g. 'sentinel-2a'"
    )
    name: str = Field(
        ..., description="Instrument name, e.g. 'msi'. Selects the product writer"
    )
    bands: Dict[str, Band] = Field(..., min_length=1, description="Bands by identifier")


class HyperspectralInstrument(BaseModel):
    """An instrument simulated as one measure over its spectral response."""

    kind: Literal["hyperspectral"] = "hyperspectral"
    platform: str = Field(
        ..., description="Platform carrying the instrument, e.g. 'chime'"
    )
    name: str = Field(..., description="Instrument name, e.g. 'hsi'")
    srf: SRFType = Field(
        ...,
        description=(
            "Spectral response, e.g. a Gaussian channel set applied after the "
            "simulation"
        ),
    )
    gsd_m: Optional[PositiveFloat] = Field(
        None, description="Native ground sampling distance (m), the default pixel size"
    )


Instrument = Annotated[
    Union[MultispectralInstrument, HyperspectralInstrument],
    Field(discriminator="kind"),
]

# CHIME (Copernicus Hyperspectral Imaging Mission), early ESA specifications:
# 400-2500 nm, FWHM < 11 nm, 8.4 nm sampling interval, 30 m pixels.
CHIME_SPECTRAL_REGIONS = [
    SpectralRegion(name="VNIR", wmin_nm=400.0, wmax_nm=1000.0, fwhm_nm=8.5),
    SpectralRegion(name="SWIR1", wmin_nm=1000.0, wmax_nm=1800.0, fwhm_nm=10.0),
    SpectralRegion(name="SWIR2", wmin_nm=1800.0, wmax_nm=2500.0, fwhm_nm=11.0),
]

CHIME_SRF = SpectralResponse(
    type="gaussian",
    spectral_regions=CHIME_SPECTRAL_REGIONS,
    output_grid=WavelengthGrid(mode="regular", wmin_nm=400, wmax_nm=2500, step_nm=8.4),
)


class MsiBand(NamedTuple):
    """Facts about one Sentinel-2 MSI band."""

    band_id: int
    """Index of the band in the bandId attributes of MTD_TL.xml."""

    gsd_m: int
    """Native ground sampling distance (m)."""

    common_name: str
    """Common name, as in the EOPF L1C STAC ``bands``."""

    fwhm_um: float
    """Nominal FWHM (µm), as in the EOPF L1C STAC ``bands``."""


MSI_BANDS = {
    "1": MsiBand(0, 60, "coastal", 0.020),
    "2": MsiBand(1, 10, "blue", 0.065),
    "3": MsiBand(2, 10, "green", 0.035),
    "4": MsiBand(3, 10, "red", 0.030),
    "5": MsiBand(4, 20, "rededge", 0.015),
    "6": MsiBand(5, 20, "rededge", 0.015),
    "7": MsiBand(6, 20, "rededge", 0.020),
    "8": MsiBand(7, 10, "nir", 0.105),
    "8a": MsiBand(8, 20, "nir08", 0.020),
    "9": MsiBand(9, 60, "nir09", 0.020),
    "10": MsiBand(10, 60, "cirrus", 0.030),
    "11": MsiBand(11, 20, "swir16", 0.090),
    "12": MsiBand(12, 20, "swir22", 0.180),
}
"""Sentinel-2 MSI bands by identifier."""

_MSI_GSD_M = {band: spec.gsd_m for band, spec in MSI_BANDS.items()}

OLCI_GSD_M = {str(band): 300 for band in range(1, 22)}
"""Sentinel-3 OLCI full-resolution ground sampling distance (m) per band."""

SLSTR_GSD_M = {str(band): 500 for band in range(1, 7)}
"""Sentinel-3 SLSTR ground sampling distance (m) of the solar-reflective bands S1-S6.
The thermal bands are left out because the simulation has no thermal emission."""


def _eradiate_instrument(
    platform: str, name: str, gsd_m: Dict[str, float]
) -> MultispectralInstrument:
    """A multispectral instrument whose band SRFs are Eradiate's datasets."""
    prefix = f"{platform.replace('-', '_')}-{name}"
    bands = {
        band: Band(srf=f"{prefix}-{band}", gsd_m=gsd) for band, gsd in gsd_m.items()
    }
    return MultispectralInstrument(platform=platform, name=name, bands=bands)


INSTRUMENTS: Dict[str, Instrument] = {
    "sentinel_2a-msi": _eradiate_instrument("sentinel-2a", "msi", _MSI_GSD_M),
    "sentinel_2b-msi": _eradiate_instrument("sentinel-2b", "msi", _MSI_GSD_M),
    "sentinel_3a-olci": _eradiate_instrument("sentinel-3a", "olci", OLCI_GSD_M),
    "sentinel_3b-olci": _eradiate_instrument("sentinel-3b", "olci", OLCI_GSD_M),
    "sentinel_3a-slstr": _eradiate_instrument("sentinel-3a", "slstr", SLSTR_GSD_M),
    "sentinel_3b-slstr": _eradiate_instrument("sentinel-3b", "slstr", SLSTR_GSD_M),
    "chime-hsi": HyperspectralInstrument(
        platform="chime", name="hsi", srf=CHIME_SRF, gsd_m=30.0
    ),
}
"""Built-in instruments by key. Add an entry to make an instrument available by key."""
