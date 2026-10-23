"""Satellite instruments a `SatelliteSensor` can carry.

An instrument is data: its platform, its name, and either its bands (multispectral) or
its spectral response (hyperspectral). Built-in instruments live in `INSTRUMENTS` and are
referenced by key. A custom instrument is passed inline, or added to `INSTRUMENTS` to
make it available by key.
"""

from typing import Annotated, Dict, Literal, Optional, Union

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

MSI_GSD_M = {
    "1": 60,
    "2": 10,
    "3": 10,
    "4": 10,
    "5": 20,
    "6": 20,
    "7": 20,
    "8": 10,
    "8a": 20,
    "9": 60,
    "10": 60,
    "11": 20,
    "12": 20,
}
"""Sentinel-2 MSI native ground sampling distance (m) per band."""

OLCI_GSD_M = {str(band): 300 for band in range(1, 22)}
"""Sentinel-3 OLCI full-resolution ground sampling distance (m) per band."""


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
    "sentinel_2a-msi": _eradiate_instrument("sentinel-2a", "msi", MSI_GSD_M),
    "sentinel_2b-msi": _eradiate_instrument("sentinel-2b", "msi", MSI_GSD_M),
    "sentinel_3a-olci": _eradiate_instrument("sentinel-3a", "olci", OLCI_GSD_M),
    "sentinel_3b-olci": _eradiate_instrument("sentinel-3b", "olci", OLCI_GSD_M),
    "chime-hsi": HyperspectralInstrument(
        platform="chime", name="hsi", srf=CHIME_SRF, gsd_m=30.0
    ),
}
"""Built-in instruments by key. Add an entry to make an instrument available by key."""
