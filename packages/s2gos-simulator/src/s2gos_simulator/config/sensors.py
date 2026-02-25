from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

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
    UAV = "uav"
    GROUND = "ground"


class SatellitePlatform(str, Enum):
    """Supported satellite platforms."""

    SENTINEL_2A = "sentinel-2a"
    SENTINEL_2B = "sentinel-2b"
    SENTINEL_3A = "sentinel-3a"
    SENTINEL_3B = "sentinel-3b"
    LANDSAT_8 = "landsat-8"
    LANDSAT_9 = "landsat-9"
    MODIS_TERRA = "modis-terra"
    MODIS_AQUA = "modis-aqua"
    CHIME = "chime"
    ENDMAP = "endmap"
    CUSTOM = "custom"  # For user-defined platforms with custom SRF


class SentinelMSIBand(str, Enum):
    """Sentinel-2 MSI band identifiers."""

    B01 = "1"
    B02 = "2"
    B03 = "3"
    B04 = "4"
    B05 = "5"
    B06 = "6"
    B07 = "7"
    B08 = "8"
    B8A = "8a"
    B09 = "9"
    B10 = "10"
    B11 = "11"
    B12 = "12"


class SentinelOLCIBand(str, Enum):
    """Sentinel-3 OLCI band identifiers."""

    OA01 = "1"
    OA02 = "2"
    OA03 = "3"
    OA04 = "4"
    OA05 = "5"
    OA06 = "6"
    OA07 = "7"
    OA08 = "8"
    OA09 = "9"
    OA10 = "10"
    OA11 = "11"
    OA12 = "12"
    OA13 = "13"
    OA14 = "14"
    OA15 = "15"
    OA16 = "16"
    OA17 = "17"
    OA18 = "18"
    OA19 = "19"
    OA20 = "20"
    OA21 = "21"


class LandsatOLIBand(str, Enum):
    """Landsat OLI band identifiers."""

    B1 = "1"  # Coastal aerosol
    B2 = "2"  # Blue
    B3 = "3"  # Green
    B4 = "4"  # Red
    B5 = "5"  # NIR
    B6 = "6"  # SWIR 1
    B7 = "7"  # SWIR 2
    B8 = "8"  # Panchromatic
    B9 = "9"  # Cirrus


class SatelliteInstrument(str, Enum):
    """Supported satellite instruments."""

    MSI = "msi"  # Sentinel-2 MultiSpectral Instrument
    OLCI = "olci"  # Sentinel-3 Ocean and Land Colour Instrument
    OLI = "oli"  # Landsat Operational Land Imager
    MODIS = "modis"  # MODIS sensor
    VNIR = "vnir"  # Generic VNIR instrument
    HSI = "hsi"  # Hyperspectral instrument
    CUSTOM = "custom"  # For custom instruments


# Platform to instrument mapping
PLATFORM_INSTRUMENTS = {
    SatellitePlatform.SENTINEL_2A: [SatelliteInstrument.MSI],
    SatellitePlatform.SENTINEL_2B: [SatelliteInstrument.MSI],
    SatellitePlatform.SENTINEL_3A: [SatelliteInstrument.OLCI],
    SatellitePlatform.SENTINEL_3B: [SatelliteInstrument.OLCI],
    SatellitePlatform.LANDSAT_8: [SatelliteInstrument.OLI],  # TODO: Implement
    SatellitePlatform.LANDSAT_9: [SatelliteInstrument.OLI],  # TODO: Implement
    SatellitePlatform.MODIS_TERRA: [SatelliteInstrument.MODIS],  # TODO: Implement
    SatellitePlatform.MODIS_AQUA: [SatelliteInstrument.MODIS],  # TODO: Implement
    SatellitePlatform.CHIME: [SatelliteInstrument.HSI],
    SatellitePlatform.ENDMAP: [SatelliteInstrument.HSI],  # TODO: Implement
    SatellitePlatform.CUSTOM: [SatelliteInstrument.CUSTOM],
}

# Instrument to band mapping
INSTRUMENT_BANDS = {
    SatelliteInstrument.MSI: SentinelMSIBand,
    SatelliteInstrument.OLCI: SentinelOLCIBand,
    SatelliteInstrument.OLI: LandsatOLIBand,  # TODO: Implement
    SatelliteInstrument.MODIS: None,  # TODO: Implement
    SatelliteInstrument.VNIR: None,
    SatelliteInstrument.HSI: None,
    SatelliteInstrument.CUSTOM: None,  # Accept any string for custom
}


class UAVInstrumentType(str, Enum):
    """Enum for UAV-mounted instrument types."""

    PERSPECTIVE_CAMERA = "perspective_camera"
    RADIANCEMETER = "radiancemeter"


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
    srf: Optional[SRFType] = Field(None, description="Spectral response function")

    @field_validator("srf", mode="before")
    @classmethod
    def resolve_srf_preset(cls, v):
        if not isinstance(v, str):
            return v

        presets = {
            "hypstar": lambda: HYPSTAR_SRF,
            "chime": lambda: CHIME_SRF,
        }

        if v in presets:
            return presets[v]()

        return v

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
    """Satellite sensor configuration."""

    platform_type: Literal[PlatformType.SATELLITE] = Field(
        PlatformType.SATELLITE, description="Platform type (always 'satellite')"
    )
    viewing: AngularViewing = Field(
        ..., description="Viewing geometry (zenith/azimuth angles)"
    )
    platform: SatellitePlatform = Field(
        ..., description="Satellite platform identifier"
    )
    instrument: SatelliteInstrument = Field(
        ..., description="Satellite instrument identifier"
    )
    band: str = Field(
        ..., description="Band identifier (validated against instrument-specific enum)"
    )
    film_resolution: Tuple[int, int] = Field(
        ..., description="Pixel grid dimensions (width, height) for 2D imaging"
    )
    target_center_lat: float = Field(
        ..., description="Target center latitude in WGS84 decimal degrees"
    )
    target_center_lon: float = Field(
        ..., description="Target center longitude in WGS84 decimal degrees"
    )
    target_size_km: Union[float, Tuple[float, float]] = Field(
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

    @model_validator(mode="after")
    def validate_and_set_defaults(self):
        """Validate platform/instrument/band combination and set defaults."""

        if self.platform != SatellitePlatform.CUSTOM:
            valid_instruments = PLATFORM_INSTRUMENTS.get(self.platform, [])
            if self.instrument not in valid_instruments:
                raise ValueError(
                    f"Instrument '{self.instrument.value}' is not valid for platform '{self.platform.value}'. "
                    f"Valid instruments: {[inst.value for inst in valid_instruments]}"
                )

        if (
            self.platform != SatellitePlatform.CUSTOM
            and self.instrument != SatelliteInstrument.CUSTOM
        ):
            band_enum = INSTRUMENT_BANDS.get(self.instrument)
            if band_enum is not None:
                try:
                    valid_band = band_enum(self.band)
                    self.band = valid_band.value
                except ValueError:
                    valid_bands = [band.value for band in band_enum]
                    raise ValueError(
                        f"Band '{self.band}' is not valid for instrument '{self.instrument.value}'. "
                        f"Valid bands: {valid_bands}"
                    )

        if self.platform == SatellitePlatform.CUSTOM and self.srf is None:
            raise ValueError(
                "Custom platform requires an explicit SRF. "
                "Please provide an SRF using the 'srf' parameter."
            )

        if self.viewing.target is None:
            self.viewing.target = [0.0, 0.0, 0.0]

        # TODO: Could be less eradiate specific?
        if self.srf is None and self.platform != SatellitePlatform.CUSTOM:
            platform_norm = self.platform.value.replace("-", "_")
            self.srf = f"{platform_norm}-{self.instrument.value}-{self.band}"

        if self.id is None:
            if self.platform == SatellitePlatform.CUSTOM:
                base_id = f"custom_{self.instrument.value}_{self.band}"
            else:
                platform_clean = self.platform.value.replace("-", "_")
                base_id = f"{platform_clean}_{self.instrument.value}_{self.band}"

            if self.viewing.zenith > 0:
                self.id = f"{base_id}_oblique{int(self.viewing.zenith)}"
            else:
                self.id = f"{base_id}_nadir"

        # Validate film resolution limits
        width, height = self.film_resolution
        if width <= 0 or height <= 0:
            raise ValueError("Film resolution dimensions must be positive integers")
        if width > 2048 or height > 2048:
            raise ValueError(
                f"Film resolution {self.film_resolution} exceeds maximum of 2048x2048 pixels. "
                "Consider reducing resolution for memory and performance reasons."
            )

        return self

    @property
    def pixel_size_m(self) -> Tuple[float, float]:
        """
        Calculate ground pixel size in meters.

        Returns:
            Tuple of (pixel_size_x_m, pixel_size_y_m) in meters per pixel
        """
        if isinstance(self.target_size_km, (int, float)):
            # Square area
            width_km = height_km = self.target_size_km
        else:
            # Rectangular area
            width_km, height_km = self.target_size_km

        pixel_size_x = (width_km * 1000) / self.film_resolution[0]
        pixel_size_y = (height_km * 1000) / self.film_resolution[1]

        return pixel_size_x, pixel_size_y


class UAVSensor(BaseSensor):
    """UAV sensor configuration."""

    platform_type: Literal[PlatformType.UAV] = Field(
        PlatformType.UAV, description="Platform type (always 'uav')"
    )
    instrument: UAVInstrumentType = Field(..., description="UAV instrument type")
    viewing: Union[LookAtViewing, AngularFromOriginViewing] = Field(
        ..., description="Viewing geometry"
    )
    fov: Optional[float] = Field(
        None, description="Field of view in degrees (required for perspective_camera)"
    )
    resolution: Optional[List[int]] = Field(
        None,
        description="Film resolution [width, height] (required for perspective_camera)",
    )
    post_processing: Optional[PostProcessingOptions] = Field(
        None,
        description="Post-processing pipeline options (spatial averaging, SRF, circular mask, etc.)",
    )

    @model_validator(mode="after")
    def validate_instrument_config(self):
        import logging

        logger = logging.getLogger(__name__)

        """Ensures fields match the instrument type."""
        if self.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            if self.fov is None or self.resolution is None:
                logger.warning(
                    f"Warning: UAV Sensor ({self.instrument.value}) is missing "
                )

        elif self.instrument == UAVInstrumentType.RADIANCEMETER:
            if self.fov is not None or self.resolution is not None:
                logger.warning(
                    f"Warning: 'fov' and 'resolution' are ignored for '{self.instrument.value}'. "
                    "Setting them to None."
                )
                self.fov = None
                self.resolution = None

        return self

    @model_validator(mode="after")
    def generate_default_id(self):
        """Auto-generates an ID based on instrument and altitude if missing."""
        if self.id is None:
            origin = getattr(self.viewing, "origin", [0, 0, 0])
            z_height = int(origin[2])

            self.id = f"uav_{self.instrument.value}_{z_height}m"

        return self


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


# CHIME (Copernicus Hyperspectral Imaging Mission) spectral characteristics
# Based on early ESA specifications:
# - Spectral range: 400-2500 nm
# - Spectral resolution: < 11 nm (worst case at FOV edge)
# - Spectral Sampling Interval (SSI): 8.4 nm
# - Spatial resolution: 30 m
# - Swath: ~130 km
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


def create_chime_sensor(
    target_center_lat: float,
    target_center_lon: float,
    target_size_km: Union[float, Tuple[float, float]] = 1.0,
    zenith: float = 0.0,
    azimuth: float = 0.0,
    samples_per_pixel: int = 64,
    ssi_nm: float = 8.4,
    fwhm_nm: Optional[float] = None,
    spectral_regions: Optional[List[SpectralRegion]] = None,
    wmin_nm: float = 400.0,
    wmax_nm: float = 2500.0,
    sensor_id: Optional[str] = None,
    **kwargs,
) -> SatelliteSensor:
    """Create a CHIME hyperspectral satellite sensor with sensible defaults.

    CHIME (Copernicus Hyperspectral Imaging Mission) specifications:
    - Spectral range: 400-2500 nm
    - Spectral resolution: < 11 nm (FWHM)
    - Spectral Sampling Interval (SSI): 8.4 nm
    - Spatial resolution: 30 m
    - Swath: ~130 km

    The sensor uses Gaussian SRF post-processing with wavelength-dependent FWHM
    to simulate the instrument's spectral response characteristics.

    Args:
        target_center_lat: Target center latitude (WGS84 decimal degrees)
        target_center_lon: Target center longitude (WGS84 decimal degrees)
        target_size_km: Target area size (km). Float for square, tuple for (width, height)
        zenith: Viewing zenith angle (0=nadir). Default 0.0
        azimuth: Viewing azimuth angle. Default 0.0
        samples_per_pixel: Monte Carlo samples per pixel. Default 64
        ssi_nm: Spectral Sampling Interval (nm). Default 8.4 per CHIME spec
        fwhm_nm: Single FWHM for all wavelengths (overrides spectral_regions)
        spectral_regions: Wavelength-dependent FWHM regions. If None, uses CHIME defaults
        wmin_nm: Minimum wavelength (nm). Default 400.0
        wmax_nm: Maximum wavelength (nm). Default 2500.0
        sensor_id: Optional sensor ID. Auto-generated if not provided
        **kwargs: Additional sensor parameters

    Returns:
        SatelliteSensor configured for CHIME hyperspectral simulation
    """
    # Use CHIME default spectral regions if not provided
    if spectral_regions is None and fwhm_nm is None:
        spectral_regions = CHIME_SPECTRAL_REGIONS

    # Build Gaussian SRF configuration
    srf = SpectralResponse(
        type="gaussian",
        fwhm_nm=fwhm_nm,
        spectral_regions=spectral_regions if fwhm_nm is None else None,
        output_grid=WavelengthGrid(
            mode="regular",
            wmin_nm=wmin_nm,
            wmax_nm=wmax_nm,
            step_nm=ssi_nm,
        ),
    )

    # Generate sensor ID if not provided
    if sensor_id is None:
        if zenith > 0:
            sensor_id = f"chime_hsi_oblique{int(zenith)}"
        else:
            sensor_id = "chime_hsi_nadir"
    film_size = (target_size_km * 1000) // 30
    adjusted_target_size = (film_size * 30) / 1000
    return SatelliteSensor(
        id=sensor_id,
        platform=SatellitePlatform.CHIME,
        instrument=SatelliteInstrument.HSI,
        band="full",  # Hyperspectral uses full spectrum, not discrete bands
        viewing=AngularViewing(zenith=zenith, azimuth=azimuth),
        film_resolution=(film_size, film_size),
        target_center_lat=target_center_lat,
        target_center_lon=target_center_lon,
        target_size_km=adjusted_target_size,
        samples_per_pixel=samples_per_pixel,
        srf=srf,
        **kwargs,
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
        PostProcessingOptions(apply_circular_mask=True, spatial_averaging=True),
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
