from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .spectral import SRFType
from .viewing import (
    AngularFromOriginViewing,
    DistantViewing,
    LookAtViewing,
)


class HemisphericalMeasurementLocation(BaseModel):
    """Location specification for hemispheric measurements.

    Used when measuring at a point with hemispherical integration
    (e.g., BOA irradiance, polar HDRF). Specifies target location
    using either geographic (lat/lon) or scene (x/y/z) coordinates.

    This class provides the location where a hemispheric measurement
    is performed, typically for measurements that integrate over all
    directions (hemisphere) at a specific point in space.
    """

    # Target location (EITHER lat/lon OR x/y/z, not both)
    target_lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Target latitude in decimal degrees (WGS84)",
    )
    target_lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Target longitude in decimal degrees (WGS84)",
    )

    target_x: Optional[float] = Field(
        None,
        description="Target X coordinate in scene coordinate system (meters)",
    )
    target_y: Optional[float] = Field(
        None,
        description="Target Y coordinate in scene coordinate system (meters)",
    )
    target_z: Optional[float] = Field(
        None,
        description="Target Z coordinate in scene coordinate system (meters)",
    )

    # Measurement parameters
    samples_per_pixel: int = Field(
        64,
        ge=1,
        description="Number of samples per pixel for Monte Carlo integration",
    )
    srf: Optional[SRFType] = Field(
        None,
        description="Spectral response function",
    )

    # Height offset (for terrain-relative measurements)
    height_offset_m: float = Field(
        0.0,
        ge=0.0,
        description="Height offset above terrain surface (meters)",
    )
    terrain_relative_height: bool = Field(
        True,
        description="If True, z-coordinates are interpreted as offsets from terrain elevation",
    )

    @model_validator(mode="after")
    def validate_coordinates(self) -> "HemisphericalMeasurementLocation":
        """Validate that either geographic or Cartesian coordinates are provided.

        Ensures:
        - Either lat/lon OR x/y/z are fully specified (not both)
        - No partial coordinate specifications
        """
        geo_present = self.target_lat is not None and self.target_lon is not None
        cartesian_present = (
            self.target_x is not None
            and self.target_y is not None
            and self.target_z is not None
        )

        any_geo = self.target_lat is not None or self.target_lon is not None
        if any_geo and not geo_present:
            raise ValueError(
                "If using geographic coordinates, both 'target_lat' and 'target_lon' are required"
            )

        any_cartesian = (
            self.target_x is not None
            or self.target_y is not None
            or self.target_z is not None
        )
        if any_cartesian and not cartesian_present:
            raise ValueError(
                "If using Cartesian coordinates, 'target_x', 'target_y', and 'target_z' are all required"
            )

        if geo_present and cartesian_present:
            raise ValueError(
                "Cannot specify both geographic (lat/lon) and Cartesian (x/y/z) coordinates"
            )

        if not geo_present and not cartesian_present:
            raise ValueError(
                "Must specify either geographic (lat/lon) or Cartesian (x/y/z) coordinates"
            )

        # Geographic mode: must have terrain_relative_height=True
        # (geographic coords don't provide absolute elevation, only lat/lon)
        if geo_present and not self.terrain_relative_height:
            raise ValueError(
                "When using geographic coordinates (lat/lon), terrain_relative_height must be True. "
                "Geographic coordinates require terrain elevation query to determine absolute height."
            )

        # Cartesian mode with absolute coordinates: ensure target_z is meaningful
        if cartesian_present and not self.terrain_relative_height:
            # target_z will be used as absolute elevation
            # (No additional validation needed - existing checks ensure target_z is present)
            pass

        return self


class IrradianceConfig(BaseModel):
    """Configuration for BOA irradiance measurement using white reference disk.

    Measures downward hemispheric irradiance at bottom-of-atmosphere using the
    white reference disk technique:

    - Places a small white Lambertian disk (ρ=1.0) at the measurement location
    - By sampling the incident radiance we indirectly get the total incident flux (E = π × L)
    """

    type: Literal["irradiance"] = Field(
        "irradiance", description="Measurement type (always 'irradiance')"
    )
    id: str = Field(description="Unique identifier for this irradiance measurement")
    location: HemisphericalMeasurementLocation = Field(
        ..., description="Geographic or scene location for the measurement"
    )
    samples_per_pixel: int = Field(512, description="Monte Carlo samples per pixel")


class BRFConfig(BaseModel):
    """Configuration for BRF measurement **without** atmosphere.

    BRF = (π × L) / (E_toa × cos(SZA))

    Two modes:

    1. Reference mode: Specify radiance_sensor_id (sensor already exists)
    2. Auto-generation mode: Specify viewing, backend creates sensors
    """

    type: Literal["brf"] = Field("brf", description="Measurement type (always 'brf')")
    id: str = Field(description="Unique identifier")

    # Mode 1: Reference existing sensor
    radiance_sensor_id: Optional[str] = Field(
        None, description="ID of sensor providing radiance measurement"
    )

    # Mode 2: Auto-generation mode
    viewing: Optional[Union[AngularFromOriginViewing, DistantViewing]] = Field(
        None, description="Viewing geometry for radiancemeter (auto-generation)"
    )

    # Optional overrides
    srf: Optional[SRFType] = Field(None, description="Spectral response function")
    samples_per_pixel: int = Field(
        default=1024, ge=1, description="Monte Carlo samples per pixel"
    )
    terrain_relative_height: bool = Field(
        default=True, description="Interpret z-coordinates as terrain-relative offsets"
    )

    @model_validator(mode="after")
    def validate_brf_mode(self):
        """Validate that either sensor reference OR auto-generation specs are provided."""
        has_ref = self.radiance_sensor_id is not None
        has_autogen = self.viewing is not None

        if has_ref and has_autogen:
            raise ValueError(
                "Specify either radiance_sensor_id OR auto-generation mode (location + viewing), not both"
            )

        if not has_ref and not has_autogen:
            raise ValueError(
                "Must specify either radiance_sensor_id "
                "OR auto-generation mode (location + viewing)"
            )

        return self


class HDRFConfig(BaseModel):
    """Configuration for Hemispherical-Directional Reflectance Factor (HDRF).

    HDRF = π × L_r / E_i

    1. Reference mode: Specify radiance_sensor_id + irradiance_measurement_id
    2. Auto-generation mode: Specify instrument + location/viewing, backend creates sensor + measurement
    """

    type: Literal["hdrf"] = Field(
        "hdrf", description="Measurement type (always 'hdrf')"
    )
    id: str = Field(description="Unique identifier")

    # Mode 1: Reference existing sensor + measurement
    radiance_sensor_id: Optional[str] = Field(
        None, description="ID of sensor providing radiance measurement"
    )
    irradiance_measurement_id: Optional[str] = Field(
        None, description="ID of IrradianceConfig providing BOA irradiance"
    )

    # Mode 2: Auto-generation mode
    instrument: Optional[Literal["radiancemeter", "hemispherical"]] = Field(
        None, description="Instrument type for auto-generation"
    )
    location: Optional[HemisphericalMeasurementLocation] = Field(
        None, description="Location for hemispherical HDRF (auto-generation)"
    )
    viewing: Optional[
        Union[LookAtViewing, AngularFromOriginViewing, DistantViewing]
    ] = Field(
        None, description="Viewing geometry for radiancemeter HDRF (auto-generation)"
    )
    reference_height_offset_m: Optional[float] = Field(
        None,
        description="Height offset for reference irradiance disk (auto-generation)",
    )
    srf: Optional[SRFType] = Field(None, description="SRF for auto-generated sensors")
    samples_per_pixel: int = Field(
        64, description="Samples per pixel for auto-generated sensors"
    )
    terrain_relative_height: bool = Field(
        True, description="Terrain-relative height interpretation"
    )

    @model_validator(mode="after")
    def validate_hdrf_mode(self):
        """Validate that either sensor references OR auto-generation specs are provided."""
        has_refs = (
            self.radiance_sensor_id is not None
            and self.irradiance_measurement_id is not None
        )
        has_autogen = self.instrument is not None

        if has_refs and has_autogen:
            raise ValueError(
                "Specify either sensor/measurement references OR auto-generation mode, not both"
            )

        if not has_refs and not has_autogen:
            raise ValueError(
                "Must specify either references (radiance_sensor_id + irradiance_measurement_id) "
                "OR auto-generation mode (instrument + location/viewing)"
            )

        # Validate auto-generation mode
        if has_autogen:
            if self.instrument == "hemispherical" and not self.location:
                raise ValueError("instrument='hemispherical' requires 'location'")
            if self.instrument == "radiancemeter":
                if not self.viewing:
                    raise ValueError("instrument='radiancemeter' requires 'viewing'")
                # For HDRF, location is also needed for irradiance measurement
                if not self.location:
                    raise ValueError(
                        "instrument='radiancemeter' for HDRF requires 'location' "
                        "for irradiance measurement generation"
                    )
            if not self.reference_height_offset_m:
                raise ValueError(
                    "Auto-generation mode requires 'reference_height_offset_m'"
                )

        return self


class BasePixelMeasurementConfig(BaseModel):
    """Base configuration for pixel-based measurements at satellite pixel centers.

    Maps satellite pixels to scene coordinates for reflectance measurements.
    Subclasses define the specific measurement type (BRF, HDRF).
    """

    id: str = Field(description="Unique identifier for this pixel measurement")
    satellite_sensor_id: str = Field(
        description="ID of SatelliteSensor to use for geometry (target bounds, resolution)"
    )
    pixel_indices: List[Tuple[int, int]] = Field(
        description="List of (row, col) pixel indices to measure. Row 0 is top (north), col 0 is left (west)."
    )
    height_offset_m: float = Field(
        default=0.5,
        ge=0.0,
        description="Height above terrain surface for measurements (meters)",
    )
    samples_per_pixel: int = Field(
        default=512,
        ge=1,
        description="Monte Carlo samples per pixel",
    )
    srf: Optional[SRFType] = Field(
        default=None,
        description="Spectral response function. If None, inherits from satellite sensor.",
    )

    @field_validator("pixel_indices")
    @classmethod
    def validate_pixel_indices(cls, v):
        """Validate pixel indices are non-negative tuples."""
        if not v:
            raise ValueError("pixel_indices cannot be empty")
        for i, (row, col) in enumerate(v):
            if row < 0 or col < 0:
                raise ValueError(
                    f"Pixel index {i} has negative value: ({row}, {col}). "
                    "Row and column must be >= 0."
                )
        return v


class PixelHDRFConfig(BasePixelMeasurementConfig):
    """BOA HDRF measurement at satellite pixel.

    HDRF = π × L_surface / E_boa

    Uses satellite sensor geometry to map pixels to scene coordinates.
    Requires 2N simulations for N pixels (irradiance + radiance per pixel).
    """

    type: Literal["pixel_hdrf"] = Field(
        "pixel_hdrf", description="Measurement type (always 'pixel_hdrf')"
    )


class PixelBRFConfig(BasePixelMeasurementConfig):
    """BRF measurement at satellite pixel (no atmosphere).

    BRF = (π × L) / (E_toa × cos(SZA))

    Uses satellite sensor geometry to map pixels to scene coordinates.
    Simpler than PixelHDRF - uses TOA irradiance from results, no white disk needed.
    Requires N simulations for N pixels (radiance only).
    """

    type: Literal["pixel_brf"] = Field(
        "pixel_brf", description="Measurement type (always 'pixel_brf')"
    )


class PixelBHRConfig(BasePixelMeasurementConfig):
    """BHR measurement at satellite pixel.

    BHR = radiosity_surface / radiosity_white_reference

    Uses satellite sensor geometry to map pixels to scene coordinates.
    Requires 2N simulations for N pixels (surface + reference per pixel).
    """

    type: Literal["pixel_bhr"] = Field(
        "pixel_bhr", description="Measurement type (always 'pixel_bhr')"
    )
    reference_height_offset_m: float = Field(
        default=0.1,
        ge=0.0,
        description="Height offset for white reference patch above surface (meters)",
    )


class HCRFConfig(BaseModel):
    """Configuration for Hemispherical-Conical Reflectance Factor (HCRF).

    Two modes:

    1. Reference mode: Specify radiance_sensor_id + irradiance_measurement_id
    2. Auto-generation mode: Specify camera geometry, backend creates a sensor + irradiance measurement.
    """

    type: Literal["hcrf"] = Field(
        "hcrf", description="Measurement type (always 'hcrf')"
    )
    id: str = Field(description="Unique identifier")

    # Mode 1: Reference existing sensor + measurement
    radiance_sensor_id: Optional[str] = Field(
        None,
        description="ID of camera sensor providing conical radiance (must have FOV)",
    )
    irradiance_measurement_id: Optional[str] = Field(
        None, description="ID of IrradianceConfig providing BOA irradiance"
    )

    # Mode 2: Auto-generation mode
    platform_type: Optional[Literal["ground", "uav"]] = Field(
        None, description="Platform type for auto-generated camera"
    )
    viewing: Optional[Union[LookAtViewing, AngularFromOriginViewing]] = Field(
        None, description="Camera viewing geometry (auto-generation)"
    )
    fov: Optional[float] = Field(
        None, gt=0.0, le=180.0, description="Camera field of view (auto-generation)"
    )
    film_resolution: Optional[Tuple[int, int]] = Field(
        None, description="Camera resolution (auto-generation)"
    )
    location: Optional[HemisphericalMeasurementLocation] = Field(
        None,
        description="Location for the reference irradiance measurement (auto-generation)",
    )
    srf: Optional[SRFType] = Field(
        None, description="Spectral response function for auto-generated sensors"
    )
    reference_height_offset_m: Optional[float] = Field(
        None,
        description="Height offset for reference irradiance disk (auto-generation)",
    )
    samples_per_pixel: int = Field(64, description="Samples per pixel")
    terrain_relative_height: bool = Field(True, description="Terrain-relative height")

    @model_validator(mode="after")
    def validate_hcrf_mode(self):
        """Validate that either sensor references OR auto-generation specs are provided."""
        has_refs = (
            self.radiance_sensor_id is not None
            and self.irradiance_measurement_id is not None
        )
        has_autogen = self.viewing is not None and self.fov is not None

        if has_refs and has_autogen:
            raise ValueError(
                "Specify either references OR auto-generation mode, not both"
            )

        if not has_refs and not has_autogen:
            raise ValueError(
                "Must specify either references (radiance_sensor_id + irradiance_measurement_id) "
                "OR auto-generation mode (viewing + fov)"
            )

        # Validate auto-generation mode
        if has_autogen:
            if not self.platform_type:
                raise ValueError("Auto-generation mode requires 'platform_type'")
            if not self.location:
                raise ValueError("Auto-generation mode requires 'location'")
            if not self.film_resolution:
                raise ValueError("Auto-generation mode requires 'film_resolution'")
            if self.reference_height_offset_m is None:
                raise ValueError(
                    "Auto-generation mode requires 'reference_height_offset_m'"
                )

        return self


class BHRConfig(HemisphericalMeasurementLocation):
    """Configuration for Bi-Hemispherical Reflectance (BHR) measurement.

    BHR = radiosity_surface / radiosity_white_reference

    Uses distant_flux measure type in Eradiate. Requires two simulations
    per measurement: one for the surface radiosity and one for a white
    reference patch (ρ=1.0 Lambertian disk) at the same location.
    """

    type: Literal["bhr"] = Field("bhr", description="Measurement type (always 'bhr')")
    id: Optional[str] = Field(
        None,
        description="Unique identifier",
    )
    reference_height_offset_m: float = Field(
        default=0.1,
        ge=0.0,
        description="Height offset for white reference patch above surface (meters)",
    )
    viewing: Optional[DistantViewing] = Field(
        None,
        description="Optional viewing geometry for pixel-level surface radiosity. "
        "When set (e.g. DistantViewing with RectangleTarget), the "
        "distant_flux surface measure uses this target instead of the "
        "point coordinates.",
    )


MeasurementConfig = Annotated[
    Union[
        IrradianceConfig,
        BRFConfig,
        HDRFConfig,
        HCRFConfig,
        PixelHDRFConfig,
        PixelBRFConfig,
        PixelBHRConfig,
        BHRConfig,
    ],
    Field(discriminator="type"),
]
