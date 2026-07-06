"""S2GOS Simulator - Radiative transfer simulation components."""

# Configuration system
# Import backends with optional dependencies
from .backends import ERADIATE_AVAILABLE, EradiateSimulator
from .backends.eradiate.backend import EradiateBackend
from .bhr_processor import BHRProcessor
from .config import (
    AngularFromOriginViewing,
    AngularViewing,
    BHRConfig,
    BRFConfig,
    ConstantIllumination,
    DirectionalIllumination,
    DistantViewing,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    PixelBHRConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
    PlatformType,
    RectangleTarget,
    SatelliteSensor,
    SimulationConfig,
    SpectralRegion,
    SpectralResponse,
    WavelengthGrid,
    create_chime_sensor,
    create_hypstar_sensor,
)
from .hdrf_processor import HDRFProcessor

__version__ = "0.2.0"

__all__ = [
    # Configuration System
    "SimulationConfig",
    "SatelliteSensor",
    "GroundSensor",
    "DirectionalIllumination",
    "ConstantIllumination",
    "AngularViewing",
    "AngularFromOriginViewing",
    "LookAtViewing",
    "HemisphericalViewing",
    "HemisphericalMeasurementLocation",
    "SpectralResponse",
    "SpectralRegion",
    "WavelengthGrid",
    "PlatformType",
    "GroundInstrumentType",
    "create_chime_sensor",
    "create_hypstar_sensor",
    # Measurement Configs
    "IrradianceConfig",
    "HDRFConfig",
    "HCRFConfig",
    "BRFConfig",
    "BHRConfig",
    "PixelBHRConfig",
    "PixelBRFConfig",
    "PixelHDRFConfig",
    # Viewing and Target Types
    "DistantViewing",
    "RectangleTarget",
    # Backends and Processors
    "EradiateSimulator",
    "EradiateBackend",
    "ERADIATE_AVAILABLE",
    "BHRProcessor",
    "HDRFProcessor",
]
