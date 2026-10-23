from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from .instruments import (
    CHIME_SRF,
    INSTRUMENTS,
    HyperspectralInstrument,
    Instrument,
)
from .spectral import SpectralRegion, SpectralResponse, SRFType, WavelengthGrid
from .viewing import (
    AngularFromOriginViewing,
    AngularViewing,
    DistantViewing,
    HemisphericalViewing,
    LookAtViewing,
    ViewingType,
)


class PlatformType(str, Enum):
    """Observation platform types."""

    SATELLITE = "satellite"
    GROUND = "ground"


class GroundInstrumentType(str, Enum):
    """Enum for ground instrument types for robustness."""

    HYPSTAR = "hypstar"
    PERSPECTIVE_CAMERA = "perspective_camera"
    PYRANOMETER = "pyranometer"
    FLUX_METER = "flux_meter"
    DHP_CAMERA = "dhp_camera"
    RADIANCEMETER = "radiancemeter"


class PostProcessingOptions(BaseModel):
    """Post-processing pipeline options for ground sensors.

    Controls spatial averaging, SRF convolution, circular FOV mask, and RGB output.
    """

    apply_srf: bool = Field(default=True, description="Apply Gaussian SRF convolution")
    spatial_averaging: bool = Field(
        default=False, description="Average over spatial dimensions (x_index, y_index)"
    )
    spatial_statistic: Literal["mean", "median"] = Field(
        default="mean", description="Statistic for spatial averaging"
    )
    apply_circular_mask: bool = Field(
        default=False,
        description=(
            "Apply circular FOV mask before spatial averaging. "
            "Pixels outside the circular FOV are set to NaN and excluded. "
            "Enable for HYPSTAR-style circular aperture sensors."
        ),
    )
    generate_rgb_image: bool = Field(
        default=False,
        description=(
            "Generate RGB visualization before spatial averaging. "
            "Requires at least 3 wavelengths covering RGB range (440-660nm). "
            "Image saved as {sensor_id}_rgb.png in output directory."
        ),
    )
    rgb_wavelengths: Tuple[float, float, float] = Field(
        default=(660.0, 550.0, 440.0),
        description="Target wavelengths (nm) for RGB channels (red, green, blue)",
    )
    rgb_brightness_factor: float = Field(
        default=1.8,
        gt=0.0,
        description="Brightness multiplier for RGB visualization",
    )


class BaseSensor(BaseModel):
    """Base sensor configuration."""

    id: Optional[str] = Field(
        None, description="Unique sensor identifier (auto-generated if not provided)"
    )
    platform_type: PlatformType = Field(..., description="Platform type identifier")
    viewing: ViewingType = Field(..., description="Viewing geometry configuration")
    produces: List[Literal["radiance", "irradiance", "hdrf", "brf", "bhr"]] = Field(
        ["radiance"],
        description="List of radiative quantities to be produced by this sensor configuration",
    )
    samples_per_pixel: int = Field(
        64, ge=1, description="Number of Monte Carlo samples per pixel"
    )
    noise_model: Optional[Dict[str, Any]] = Field(
        None, description="Noise model configuration"
    )


class SatelliteSensor(BaseSensor):
    """Satellite sensor: an instrument looking at a target footprint.

    A multispectral instrument is simulated as one measure per band, a hyperspectral one
    as a single measure over its spectral response.
    """

    model_config = ConfigDict(extra="forbid")

    platform_type: Literal[PlatformType.SATELLITE] = Field(
        PlatformType.SATELLITE, description="Platform type (always 'satellite')"
    )
    viewing: AngularViewing = Field(
        ..., description="Viewing geometry (zenith/azimuth angles)"
    )
    instrument: Union[str, Instrument] = Field(
        ...,
        description=(
            "Key of a built-in instrument in INSTRUMENTS (e.g. 'sentinel_2a-msi', "
            "'chime-hsi'), or a custom instrument"
        ),
    )
    bands: Optional[List[str]] = Field(
        None,
        description=(
            "Bands to simulate, each as its own measure. Defaults to every band of a "
            "multispectral instrument. Leave unset for a hyperspectral instrument"
        ),
    )
    band_viewing: Dict[str, AngularViewing] = Field(
        default_factory=dict,
        description=(
            "Per-band viewing geometry overriding 'viewing' (e.g. the per-band "
            "angles of a real Sentinel-2 acquisition)"
        ),
    )
    film_resolution: Optional[Tuple[PositiveInt, PositiveInt]] = Field(
        None,
        description=(
            "Pixel grid (width, height) applied to every band. Defaults to each "
            "band's native GSD, in which case target_size_km must be a whole "
            "multiple of it. Required when the instrument has no GSD"
        ),
    )
    target_center_lat: float = Field(
        ..., description="Target center latitude in WGS84 decimal degrees"
    )
    target_center_lon: float = Field(
        ..., description="Target center longitude in WGS84 decimal degrees"
    )
    target_size_km: Union[PositiveFloat, Tuple[PositiveFloat, PositiveFloat]] = Field(
        ...,
        description="Target area size: float for square (km), tuple for rectangular (width_km, height_km)",
    )
    for_reference_only: bool = Field(
        default=False,
        description=(
            "If True, this sensor is used for geometry specification only "
            "(e.g., for PixelHDRF/PixelBRF coordinate mapping) and will NOT be simulated. "
            "Use this when you already have satellite data and only need the sensor's "
            "geometry (lat/lon, resolution, viewing angles) for pixel index mapping. "
            "The sensor remains accessible for coordinate queries but is excluded from "
            "Eradiate measure translation."
        ),
    )

    @field_validator("instrument")
    @classmethod
    def check_instrument_key(cls, v):
        if isinstance(v, str) and v not in INSTRUMENTS:
            raise ValueError(
                f"Unknown instrument {v!r}. Available: {sorted(INSTRUMENTS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_and_set_defaults(self):
        """Check the bands against the instrument and fill in the defaults."""
        if self.is_hyperspectral:
            self._check_hyperspectral()
        else:
            self._resolve_bands()
        self._set_defaults()
        for band in self.measured_bands:
            self.band_film_resolution(band)  # raises if the grid doesn't fit
        return self

    def _check_hyperspectral(self):
        if self.bands is not None or self.band_viewing:
            raise ValueError(
                f"Hyperspectral instrument '{self.spec.name}' takes no bands or "
                "band_viewing"
            )

    def _resolve_bands(self):
        known = list(self.spec.bands)
        if self.bands is None:
            self.bands = known
        if not self.bands or len(set(self.bands)) != len(self.bands):
            raise ValueError(f"'bands' must be non-empty and unique, got {self.bands}")
        unknown = [band for band in self.bands if band not in known]
        if unknown:
            raise ValueError(
                f"Bands {unknown} are not bands of instrument '{self.spec.name}'. "
                f"Valid bands: {known}"
            )
        outside = sorted(set(self.band_viewing) - set(self.bands))
        if outside:
            raise ValueError(f"band_viewing has bands not in 'bands': {outside}")

    def _set_defaults(self):
        for viewing in [self.viewing, *self.band_viewing.values()]:
            if viewing.target is None:
                viewing.target = [0.0, 0.0, 0.0]
        if self.id is None:
            zenith = self.viewing.zenith
            view = f"oblique{int(zenith)}" if zenith > 0 else "nadir"
            self.id = f"{self.spec.platform}_{self.spec.name}_{view}".replace("-", "_")

    @property
    def spec(self) -> Instrument:
        """The instrument: the `INSTRUMENTS` entry for a key, else the inline one."""
        if isinstance(self.instrument, str):
            return INSTRUMENTS[self.instrument]
        return self.instrument

    @property
    def is_hyperspectral(self) -> bool:
        """Whether the sensor is one measure over its instrument's spectral response."""
        return isinstance(self.spec, HyperspectralInstrument)

    @property
    def srf(self) -> Optional[SRFType]:
        """Spectral response applied after the simulation (hyperspectral only)."""
        return self.spec.srf if self.is_hyperspectral else None

    @property
    def measured_bands(self) -> List[Optional[str]]:
        """Band of each measure: `bands`, or a single None for a hyperspectral sensor."""
        return [None] if self.is_hyperspectral else self.bands

    def _footprint_m(self) -> Tuple[float, float]:
        if isinstance(self.target_size_km, (int, float)):
            return self.target_size_km * 1000, self.target_size_km * 1000
        width_km, height_km = self.target_size_km
        return width_km * 1000, height_km * 1000

    def resolve_band(self, band: Optional[str] = None) -> Optional[str]:
        """The given band, the only band when none is given, or None if hyperspectral.

        Raises:
            ValueError: if a band is given to a hyperspectral sensor, the band is not
                one of `bands`, or none is given for a multi-band sensor.
        """
        if self.is_hyperspectral and band is not None:
            raise ValueError(f"Hyperspectral sensor '{self.id}' has no bands")
        if self.is_hyperspectral:
            return None
        if band is None and len(self.bands) > 1:
            raise ValueError(f"Sensor '{self.id}' has bands {self.bands}, specify one")
        if band is None:
            return self.bands[0]
        if band not in self.bands:
            raise ValueError(f"Sensor '{self.id}' has no band {band!r} ({self.bands})")
        return band

    def band_film_resolution(self, band: Optional[str] = None) -> Tuple[int, int]:
        """Pixel grid (width, height) of a band: `film_resolution`, else its native GSD.

        Raises:
            ValueError: if no GSD is known, or the footprint is not a whole multiple
                of it.
        """
        if self.film_resolution is not None:
            return self.film_resolution
        gsd = self.spec.gsd_m if self.is_hyperspectral else self.spec.bands[band].gsd_m
        if gsd is None:
            raise ValueError(
                f"film_resolution is required for sensor '{self.id}': instrument "
                f"'{self.spec.name}' has no native GSD for band {band!r}"
            )
        footprint_m = self._footprint_m()
        film = tuple(round(size / gsd) for size in footprint_m)
        if not all(math.isclose(n * gsd, size) for n, size in zip(film, footprint_m)):
            raise ValueError(
                f"Footprint {self.target_size_km} km is not a whole multiple of "
                f"band {band!r}'s {gsd:g} m GSD"
            )
        return film

    def band_srf(self, band: Optional[str] = None) -> SRFType:
        """SRF of a measure: the hyperspectral instrument's, else the band's."""
        return self.spec.srf if self.is_hyperspectral else self.spec.bands[band].srf

    def band_viewing_for(self, band: Optional[str] = None) -> AngularViewing:
        """Viewing geometry of a band: its `band_viewing` override, else `viewing`."""
        return self.band_viewing.get(band, self.viewing)

    def band_measure_id(self, band: Optional[str] = None) -> str:
        """Id of a measure: the sensor id for a single measure, else `<id>_<band>`."""
        return self.id if len(self.measured_bands) == 1 else f"{self.id}_{band}"

    def pixel_size_m(self, band: Optional[str] = None) -> Tuple[float, float]:
        """Ground pixel size (x, y) of a band in meters."""
        width_m, height_m = self._footprint_m()
        width, height = self.band_film_resolution(band)
        return width_m / width, height_m / height


class GroundSensor(BaseSensor):
    """Ground sensor configuration."""

    platform_type: Literal[PlatformType.GROUND] = Field(
        PlatformType.GROUND, description="Platform type (always 'ground')"
    )
    instrument: GroundInstrumentType = Field(..., description="Ground instrument type")
    viewing: Union[
        LookAtViewing, AngularFromOriginViewing, HemisphericalViewing, DistantViewing
    ] = Field(..., description="Viewing geometry")
    fov: Optional[float] = Field(
        None,
        description="Field of view in degrees (for camera-like instruments: HYPSTAR, perspective_camera, dhp_camera)",
    )
    resolution: Optional[List[int]] = Field(
        None,
        description="Film resolution [width, height] (for camera-like instruments: HYPSTAR, perspective_camera, dhp_camera)",
    )
    post_processing: Optional[PostProcessingOptions] = Field(
        None,
        description="Post-processing pipeline options (spatial averaging, SRF, circular mask, etc.)",
    )
    srf: Optional[SRFType] = Field(None, description="Spectral response function")

    @field_validator("srf", mode="before")
    @classmethod
    def resolve_srf_preset(cls, v):
        presets = {"hypstar": HYPSTAR_SRF, "chime": CHIME_SRF}
        return presets.get(v, v) if isinstance(v, str) else v

    @model_validator(mode="before")
    @classmethod
    def set_instrument_defaults(cls, data: Any) -> Any:
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(data, dict):
            return data

        inst = data.get("instrument")
        is_hypstar = inst == GroundInstrumentType.HYPSTAR or inst == "HYPSTAR"

        if is_hypstar:
            if data.get("fov") is None:
                logger.warning("HYPSTAR sensor missing 'fov'. Defaulting to 5.0°")
                data["fov"] = 5.0

            if data.get("resolution") is None:
                logger.warning(
                    "HYPSTAR sensor missing 'resolution'. Defaulting to [32, 32]"
                )
                data["resolution"] = [32, 32]

            if data.get("post_processing") is None:
                data["post_processing"] = PostProcessingOptions(
                    apply_circular_mask=True
                )

            if data.get("srf") is None:
                data["srf"] = SpectralResponse(
                    type="gaussian",
                    spectral_regions=HYPSTAR_SPECTRAL_REGIONS,
                )

        return data

    @model_validator(mode="after")
    def validate_compatibility_and_id(self):
        camera_instruments = [
            GroundInstrumentType.HYPSTAR,
            GroundInstrumentType.PERSPECTIVE_CAMERA,
            GroundInstrumentType.DHP_CAMERA,
        ]

        if self.instrument in camera_instruments:
            if not isinstance(self.viewing, (LookAtViewing, AngularFromOriginViewing)):
                raise ValueError(
                    f"'{self.instrument.value}' requires a pointing view "
                    f"('directional' or 'angular_from_origin'), not '{self.viewing.type}'."
                )

        if self.id is None:
            if hasattr(self.viewing, "origin"):
                self.id = (
                    f"ground_{self.instrument.value}_{int(self.viewing.origin[2])}m"
                )
            elif isinstance(self.viewing, DistantViewing):
                self.id = f"ground_{self.instrument.value}_distant"
            else:
                self.id = f"ground_{self.instrument.value}"

        return self


# HYPSTAR (HYPERNETS Land Network) spectral characteristics
# - VNIR detector (Si): 380–1000 nm, FWHM ≈ 3 nm
# - SWIR detector (InGaAs): 1000–1680 nm, FWHM ≈ 10 nm
HYPSTAR_SPECTRAL_REGIONS = [
    SpectralRegion(name="VNIR", wmin_nm=380.0, wmax_nm=1000.0, fwhm_nm=3.0),
    SpectralRegion(name="SWIR", wmin_nm=1000.0, wmax_nm=1680.0, fwhm_nm=10.0),
]

HYPSTAR_SRF = SpectralResponse(
    type="gaussian",
    spectral_regions=HYPSTAR_SPECTRAL_REGIONS,
)


def create_hypstar_sensor(
    viewing: Union[LookAtViewing, AngularFromOriginViewing],
    fov: float = 5.0,
    resolution: tuple[int, int] = (32, 32),
    reference_file: Optional[str] = None,
    wavelength_variable: str = "wavelength",
    sensor_id: Optional[str] = None,
    **kwargs,
) -> GroundSensor:
    """Create a HYPSTAR ground sensor.

    HYPSTAR (HYPERNETS Land Network) specifications:
    - VNIR detector (Si): 380–1000 nm, FWHM ≈ 3 nm
    - SWIR detector (InGaAs): 1000–1680 nm, FWHM ≈ 10 nm

    When reference_file is provided, SRF post-processing outputs wavelengths
    from that file (for direct comparison against real HYPSTAR observations).
    Without it, simulation wavelengths are used.

    Args:
        viewing: Pointing configuration (LookAtViewing or AngularFromOriginViewing)
        fov: Field of view in degrees. Default 5.0 (HYPSTAR default)
        resolution: Film resolution [width, height]. Default (32, 32)
        reference_file: Path to HYPERNETS L2A NetCDF file. If provided, SRF
            post-processing will output at the file's wavelength grid.
        wavelength_variable: Variable name for wavelengths in reference file
        sensor_id: Optional sensor ID. Auto-generated if not provided
        **kwargs: Additional GroundSensor parameters

    Returns:
        GroundSensor configured for HYPSTAR simulation
    """
    output_grid = None
    if reference_file is not None:
        output_grid = WavelengthGrid(
            mode="from_file",
            file_path=reference_file,
            wavelength_variable=wavelength_variable,
        )

    srf = SpectralResponse(
        type="gaussian",
        spectral_regions=HYPSTAR_SPECTRAL_REGIONS,
        output_grid=output_grid,
    )

    kwargs.setdefault(
        "post_processing",
        PostProcessingOptions(
            apply_circular_mask=True, spatial_averaging=True, generate_rgb_image=True
        ),
    )

    return GroundSensor(
        id=sensor_id,
        instrument=GroundInstrumentType.HYPSTAR,
        viewing=viewing,
        fov=fov,
        resolution=list(resolution),
        srf=srf,
        **kwargs,
    )
