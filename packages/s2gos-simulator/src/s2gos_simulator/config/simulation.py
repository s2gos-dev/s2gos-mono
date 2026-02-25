from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from s2gos_utils import validate_config_version
from s2gos_utils.io.paths import open_file, read_json
from s2gos_utils.typing import PathLike

from .illumination import ConstantIllumination, DirectionalIllumination
from .measurements import (
    BRFConfig,
    HCRFConfig,
    HDRFConfig,
    IrradianceConfig,
    MeasurementConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
)
from .sensors import GroundSensor, SatelliteSensor, UAVSensor
from .spectral import SpectralResponse
from .._version import get_version


class ProcessingLevel(str, Enum):
    """Data processing levels."""

    L1B = "l1b"
    L1C = "l1c"  # Orthorectified


class ProcessingConfig(BaseModel):
    """Processing configuration."""

    orthorectified: bool = Field(
        False, description="Whether to produce orthorectified output"
    )


class SimulationConfig(BaseModel):
    """Comprehensive simulation configuration containing everything needed to set up a simulation.

    This single configuration class includes:

    - Sensor definitions for all platforms
    - Illumination settings
    - Measurement types
    """

    # Metadata
    config_version: str = Field(
        default_factory=get_version, description="Configuration schema version"
    )
    name: str = Field(..., description="Simulation name")
    description: Optional[str] = Field(None, description="Simulation description")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp (auto-populated)"
    )

    # Core configuration
    illumination: Union[DirectionalIllumination, ConstantIllumination] = Field(
        default_factory=None, description="Illumination configuration"
    )
    sensors: List[Union[SatelliteSensor, UAVSensor, GroundSensor]] = Field(
        default_factory=list, description="List of sensors to simulate"
    )
    measurements: List[MeasurementConfig] = Field(
        default_factory=list,
        description="Unified list of radiative quantity measurements (HDRF, BRF, irradiance, etc.)",
    )

    # Processing configuration
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Processing configuration"
    )

    # Advanced options
    enable_noise: bool = Field(False, description="Enable noise modeling")

    backend_hints: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Backend-specific configuration hints (e.g., {'eradiate': {'mode': 'ckd_double'}}).

        For Eradiate backend:
        - 'mode': Spectral mode ('mono', 'ckd', 'mono_polarized', 'ckd_polarized', etc.)
        """,
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }

    @field_serializer("created_at", when_used="json")
    def serialize_datetime_iso(self, created_at: datetime):
        return created_at.isoformat()

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v):
        """Validate sensor list."""
        if v:
            sensor_ids = [sensor.id for sensor in v]
            if len(sensor_ids) != len(set(sensor_ids)):
                raise ValueError("Sensor IDs must be unique")
        return v

    @model_validator(mode="after")
    def validate_simulation_config(self):
        """Validate simulation configuration."""
        if not self.sensors and not self.measurements:
            raise ValueError("At least one sensor or measurement must be specified")

        self._validate_measurement_references()
        self._validate_sensor_references()

        return self

    def _validate_measurement_references(self):
        """Validate cross-references between measurements.

        Ensures that:
        - Measurement IDs are unique
        - reference_irradiance_id references an existing measurement
        - The referenced measurement is an IrradianceConfig
        """
        measurement_ids = set()
        irradiance_ids = set()

        for measurement in self.measurements:
            m_id = getattr(measurement, "id", None)
            if m_id:
                if m_id in measurement_ids:
                    raise ValueError(f"Duplicate measurement ID: '{m_id}'")
                measurement_ids.add(m_id)

                if isinstance(measurement, IrradianceConfig):
                    irradiance_ids.add(m_id)

        errors = []
        for measurement in self.measurements:
            ref_id = getattr(measurement, "irradiance_measurement_id", None)
            if ref_id is not None:
                m_id = getattr(measurement, "id", "unknown")

                if ref_id not in measurement_ids:
                    errors.append(
                        f"Measurement '{m_id}' references unknown measurement "
                        f"'{ref_id}' as irradiance_measurement_id"
                    )
                elif ref_id not in irradiance_ids:
                    errors.append(
                        f"Measurement '{m_id}' references '{ref_id}' as irradiance, "
                        f"but it is not an IrradianceConfig"
                    )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def _validate_sensor_references(self):
        """Validate that measurements reference existing sensors and measurements."""
        sensor_ids = {s.id for s in self.sensors}
        measurement_ids = {
            getattr(m, "id", None) for m in self.measurements if getattr(m, "id", None)
        }

        errors = []
        for measurement in self.measurements:
            # Check HDRF references
            if isinstance(measurement, HDRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"HDRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in measurement_ids:
                        errors.append(
                            f"HDRF '{measurement.id}' references unknown measurement "
                            f"'{measurement.irradiance_measurement_id}'"
                        )
                    # Validate that referenced measurement is IrradianceConfig
                    ref_measurement = next(
                        (
                            m
                            for m in self.measurements
                            if getattr(m, "id", None)
                            == measurement.irradiance_measurement_id
                        ),
                        None,
                    )
                    if ref_measurement and not isinstance(
                        ref_measurement, IrradianceConfig
                    ):
                        errors.append(
                            f"HDRF '{measurement.id}' references '{measurement.irradiance_measurement_id}' "
                            f"as irradiance, but it is not an IrradianceConfig"
                        )

            # Check HCRF references
            if isinstance(measurement, HCRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"HCRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )
                    # Validate that referenced sensor has FOV (is a camera)
                    sensor = self.get_sensor(measurement.radiance_sensor_id)
                    if sensor and not hasattr(sensor, "fov"):
                        errors.append(
                            f"HCRF '{measurement.id}' references sensor "
                            f"'{measurement.radiance_sensor_id}' which is not a camera (no FOV)"
                        )
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in measurement_ids:
                        errors.append(
                            f"HCRF '{measurement.id}' references unknown measurement "
                            f"'{measurement.irradiance_measurement_id}'"
                        )
                    # Validate it's IrradianceConfig
                    ref_measurement = next(
                        (
                            m
                            for m in self.measurements
                            if getattr(m, "id", None)
                            == measurement.irradiance_measurement_id
                        ),
                        None,
                    )
                    if ref_measurement and not isinstance(
                        ref_measurement, IrradianceConfig
                    ):
                        errors.append(
                            f"HCRF '{measurement.id}' references '{measurement.irradiance_measurement_id}' "
                            f"as irradiance, but it is not an IrradianceConfig"
                        )

            # Check BRF sensor references (atmosphere-less BRF)
            if isinstance(measurement, BRFConfig):
                if measurement.radiance_sensor_id:
                    if measurement.radiance_sensor_id not in sensor_ids:
                        errors.append(
                            f"BRF '{measurement.id}' references unknown sensor "
                            f"'{measurement.radiance_sensor_id}'"
                        )

            if isinstance(measurement, PixelBRFConfig):
                if measurement.satellite_sensor_id not in sensor_ids:
                    errors.append(
                        f"PixelBRF '{measurement.id}' references unknown sensor "
                        f"'{measurement.satellite_sensor_id}'"
                    )
                else:
                    sensor = self.get_sensor(measurement.satellite_sensor_id)
                    if sensor and not isinstance(sensor, SatelliteSensor):
                        errors.append(
                            f"PixelBRF '{measurement.id}' references sensor "
                            f"'{measurement.satellite_sensor_id}' which is not a SatelliteSensor "
                            f"(type: {type(sensor).__name__})"
                        )

            # Check PixelHDRF sensor references
            if isinstance(measurement, PixelHDRFConfig):
                if measurement.satellite_sensor_id not in sensor_ids:
                    errors.append(
                        f"PixelHDRF '{measurement.id}' references unknown sensor "
                        f"'{measurement.satellite_sensor_id}'"
                    )
                else:
                    # Validate that referenced sensor is a SatelliteSensor
                    sensor = self.get_sensor(measurement.satellite_sensor_id)
                    if sensor and not isinstance(sensor, SatelliteSensor):
                        errors.append(
                            f"PixelHDRF '{measurement.id}' references sensor "
                            f"'{measurement.satellite_sensor_id}' which is not a SatelliteSensor "
                            f"(type: {type(sensor).__name__})"
                        )

        if errors:
            raise ValueError(
                "Sensor reference validation failed:\n  - " + "\n  - ".join(errors)
            )

    @model_validator(mode="after")
    def validate_hdrf_hcrf_requires_irradiance(self):
        """Validate that HDRF/HCRF measurements have corresponding irradiance measurements.

        In reference mode, HDRF/HCRF must reference an existing IrradianceConfig.
        In auto-generation mode, the backend will create the irradiance measurement automatically.
        """
        # Get all irradiance measurement IDs
        irradiance_ids = {
            m.id for m in self.measurements if isinstance(m, IrradianceConfig)
        }

        errors = []
        for measurement in self.measurements:
            if isinstance(measurement, (HDRFConfig, HCRFConfig)):
                # In reference mode, must reference existing irradiance measurement
                if measurement.irradiance_measurement_id:
                    if measurement.irradiance_measurement_id not in irradiance_ids:
                        errors.append(
                            f"{measurement.__class__.__name__} '{measurement.id}' references "
                            f"irradiance_measurement_id='{measurement.irradiance_measurement_id}' "
                            f"which doesn't exist. Available: {sorted(irradiance_ids) if irradiance_ids else 'none'}"
                        )
                # In auto-gen mode, backend will create irradiance measurement
                # (no validation needed here, as it happens during backend execution)

        if errors:
            raise ValueError(
                "HDRF/HCRF validation failed:\n  - " + "\n  - ".join(errors)
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self, path: Optional[PathLike] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            with open_file(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, path: PathLike) -> "SimulationConfig":
        """Load from JSON file with version compatibility checking."""
        data = read_json(path)

        # Validate configuration version
        validated_data = validate_config_version(
            "simulation_config", data, get_version(), "simulation configuration"
        )

        return cls(**validated_data)

    def add_sensor(
        self, sensor: Union[SatelliteSensor, UAVSensor, GroundSensor]
    ) -> None:
        """Add a sensor to the configuration."""
        if sensor.id in [s.id for s in self.sensors]:
            raise ValueError(f"Sensor ID '{sensor.id}' already exists")
        self.sensors.append(sensor)

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor by ID."""
        self.sensors = [s for s in self.sensors if s.id != sensor_id]

    def get_sensor(
        self, sensor_id: str
    ) -> Optional[Union[SatelliteSensor, UAVSensor, GroundSensor]]:
        """Get a sensor by ID."""
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    @property
    def output_quantities(self) -> List[str]:
        """Get output quantities based on sensors."""
        # In the new system, sensors specify what they produce in the 'produces' field
        quantities = []
        for sensor in self.sensors:
            if hasattr(sensor, "produces"):
                quantities.extend(sensor.produces)
            else:
                # Default to radiance for all sensors
                quantities.append("radiance")
        return list(set(quantities))

    @property
    def wavelength_range(self) -> tuple:
        """Get wavelength range based on sensors."""
        min_wl, max_wl = float("inf"), 0.0

        for sensor in self.sensors:
            if isinstance(sensor.srf, SpectralResponse):
                if sensor.srf.type == "delta" and sensor.srf.wavelengths:
                    min_wl = min(min_wl, min(sensor.srf.wavelengths))
                    max_wl = max(max_wl, max(sensor.srf.wavelengths))
                elif (
                    sensor.srf.type == "uniform" and sensor.srf.wmin and sensor.srf.wmax
                ):
                    min_wl = min(min_wl, sensor.srf.wmin)
                    max_wl = max(max_wl, sensor.srf.wmax)

        if min_wl == float("inf"):
            return (400.0, 2500.0)

        return (min_wl, max_wl)

    def validate_configuration(self) -> List[str]:
        """Validate the complete configuration and return any errors."""
        errors = []

        if not self.sensors:
            errors.append("No sensors defined")

        sensor_ids = [sensor.id for sensor in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            errors.append("Sensor IDs must be unique")

        return errors
