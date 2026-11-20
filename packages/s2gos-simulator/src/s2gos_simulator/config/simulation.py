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
from s2gos_utils.io.paths import PathLike, open_file, read_json

from .illumination import (
    AstroObjectIllumination,
    ConstantIllumination,
    DirectionalIllumination,
)
from .measurements import (
    BasePixelMeasurementConfig,
    BRFConfig,
    HCRFConfig,
    HDRFConfig,
    MeasurementConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
)
from .sensors import GroundInstrumentType, GroundSensor, SatelliteSensor
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
    illumination: Union[
        DirectionalIllumination, AstroObjectIllumination, ConstantIllumination
    ] = Field(..., discriminator="type", description="Illumination configuration")
    sensors: List[Union[SatelliteSensor, GroundSensor]] = Field(
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
        - 'moment': If True, use the moment integrator and add 'radiance_var' (per-pixel
          Monte Carlo variance of the estimate, already divided by spp) to the results
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
        """Validate that measurement IDs are unique."""
        measurement_ids = set()
        for measurement in self.measurements:
            m_id = getattr(measurement, "id", None)
            if m_id:
                if m_id in measurement_ids:
                    raise ValueError(f"Duplicate measurement ID: '{m_id}'")
                measurement_ids.add(m_id)

    def _validate_sensor_references(self):
        """Validate that measurements reference existing, suitable sensors."""
        sensor_ids = {s.id for s in self.sensors}

        errors = []
        for measurement in self.measurements:
            if isinstance(measurement, (HDRFConfig, HCRFConfig)):
                kind = "HDRF" if isinstance(measurement, HDRFConfig) else "HCRF"
                if (
                    measurement.radiance_sensor_id
                    and measurement.radiance_sensor_id not in sensor_ids
                ):
                    errors.append(
                        f"{kind} '{measurement.id}' references unknown sensor "
                        f"'{measurement.radiance_sensor_id}'"
                    )

            irradiance_id = getattr(measurement, "irradiance_sensor_id", None)
            if irradiance_id:
                error = self._irradiance_reference_error(irradiance_id)
                if error:
                    errors.append(
                        f"{measurement.type.upper()} '{measurement.id}': {error}"
                    )

            # Check HCRF references
            if isinstance(measurement, HCRFConfig) and measurement.radiance_sensor_id:
                # Validate that referenced sensor has FOV (is a camera)
                sensor = self.get_sensor(measurement.radiance_sensor_id)
                if sensor and not hasattr(sensor, "fov"):
                    errors.append(
                        f"HCRF '{measurement.id}' references sensor "
                        f"'{measurement.radiance_sensor_id}' which is not a camera (no FOV)"
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

            radiance_id = getattr(measurement, "radiance_sensor_id", None)
            sensor = self.get_sensor(radiance_id) if radiance_id else None
            if isinstance(sensor, SatelliteSensor) and not sensor.is_hyperspectral:
                errors.append(
                    f"'{measurement.id}' references multispectral satellite sensor "
                    f"'{radiance_id}', whose results are per band. Use a pixel "
                    "measurement instead"
                )

            if isinstance(measurement, BasePixelMeasurementConfig):
                sensor = self.get_sensor(measurement.satellite_sensor_id)
                if isinstance(sensor, SatelliteSensor):
                    try:
                        sensor.resolve_band(measurement.band)
                    except ValueError as e:
                        errors.append(f"'{measurement.id}': {e}")

        if errors:
            raise ValueError(
                "Sensor reference validation failed:\n  - " + "\n  - ".join(errors)
            )

    def _irradiance_reference_error(self, sensor_id: str) -> Optional[str]:
        """Why a sensor cannot serve as BOA irradiance reference, or None if it can.

        HDRF, HCRF and BHR divide by the irradiance on a horizontal plane, so the
        reference must be an upward-facing irradiancemeter.
        """
        sensor = self.get_sensor(sensor_id)
        if sensor is None:
            return f"irradiance_sensor_id '{sensor_id}' is not a known sensor"
        if (
            not isinstance(sensor, GroundSensor)
            or sensor.instrument != GroundInstrumentType.IRRADIANCEMETER
        ):
            return f"irradiance sensor '{sensor_id}' is not an irradiancemeter"
        if not sensor.faces_up:
            return f"irradiance irradiancemeter '{sensor_id}' does not face straight up"
        return None

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

    def add_sensor(self, sensor: Union[SatelliteSensor, GroundSensor]) -> None:
        """Add a sensor to the configuration."""
        if sensor.id in [s.id for s in self.sensors]:
            raise ValueError(f"Sensor ID '{sensor.id}' already exists")
        self.sensors.append(sensor)

    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor by ID."""
        self.sensors = [s for s in self.sensors if s.id != sensor_id]

    def get_sensor(
        self, sensor_id: str
    ) -> Optional[Union[SatelliteSensor, GroundSensor]]:
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

    def validate_configuration(self) -> List[str]:
        """Validate the complete configuration and return any errors."""
        errors = []

        if not self.sensors:
            errors.append("No sensors defined")

        sensor_ids = [sensor.id for sensor in self.sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            errors.append("Sensor IDs must be unique")

        return errors
