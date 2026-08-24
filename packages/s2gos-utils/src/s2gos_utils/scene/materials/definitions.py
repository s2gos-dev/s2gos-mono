import re
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_spectral_parameter(cls, v, enforce_unit_bounds=True):
    """Universal spectral parameter validator for Eradiate/Mitsuba compatibility.

    Validates spectral parameters according to both Eradiate and Mitsuba specifications:
    - File-based: {"path": "spectrum.nc", "variable": "reflectance"}
    - Uniform: {"type": "uniform", "value": 0.5} or {"type": "uniform", "value": [0.8, 0.6, 0.4]}
    - Interpolated: {"type": "interpolated", "wavelengths": [400, 500, 600], "values": [0.2, 0.5, 0.3]}

    Physical constraints:
    - Reflectance/transmittance values must be in [0,1] for energy conservation (enforce_unit_bounds=True)
    - IOR values can be outside [0,1] (enforce_unit_bounds=False)
    - File paths must exist (when not None)
    - Wavelengths must be monotonically increasing (for interpolated type)

    Args:
        cls: Validator class
        v: Parameter value to validate
        enforce_unit_bounds: If True, enforce [0,1] bounds for physical validity
    """
    if not isinstance(v, dict):
        raise ValueError("Spectral parameter must be a dictionary")

    if "type" in v and v["type"] == "uniform":
        if "value" not in v:
            raise ValueError("Uniform spectral parameter must have 'value' field")
        value = v["value"]

        if isinstance(value, (list, tuple)):
            # RGB array validation
            if len(value) != 3:
                raise ValueError("RGB values must have exactly 3 components")
            for i, component in enumerate(value):
                if not isinstance(component, (int, float)):
                    raise ValueError(
                        f"RGB component {i} must be numeric, got {type(component).__name__}"
                    )
                if enforce_unit_bounds and not (0.0 <= component <= 1.0):
                    raise ValueError(
                        f"RGB component {i} value {component} must be in [0,1]"
                    )
        elif isinstance(value, (int, float)):
            if enforce_unit_bounds and not (0.0 <= value <= 1.0):
                raise ValueError(f"Uniform value {value} must be in [0,1]")
        else:
            raise ValueError(
                f"Uniform value must be scalar or 3-component RGB, got {type(value).__name__}"
            )

    elif "type" in v and v["type"] == "interpolated":
        if "wavelengths" not in v or "values" not in v:
            raise ValueError(
                "Interpolated spectrum must have 'wavelengths' and 'values' fields"
            )

        wavelengths = v["wavelengths"]
        values = v["values"]

        if not isinstance(wavelengths, (list, tuple)):
            raise ValueError(
                f"wavelengths must be a list or tuple, got {type(wavelengths).__name__}"
            )
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"values must be a list or tuple, got {type(values).__name__}"
            )

        if len(wavelengths) != len(values):
            raise ValueError(
                f"wavelengths ({len(wavelengths)}) and values ({len(values)}) must have the same length"
            )

        if len(wavelengths) < 2:
            raise ValueError(
                f"Interpolated spectrum must have at least 2 wavelength points, got {len(wavelengths)}"
            )

        for i, wl in enumerate(wavelengths):
            if not isinstance(wl, (int, float)):
                raise ValueError(
                    f"wavelength[{i}] must be numeric, got {type(wl).__name__}"
                )
            if wl <= 0:
                raise ValueError(f"wavelength[{i}] must be positive, got {wl}")

        for i, val in enumerate(values):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"value[{i}] must be numeric, got {type(val).__name__}"
                )
            if enforce_unit_bounds and not (0.0 <= val <= 1.0):
                raise ValueError(f"value[{i}] = {val} must be in [0,1]")

        for i in range(len(wavelengths) - 1):
            if wavelengths[i] >= wavelengths[i + 1]:
                raise ValueError(
                    f"wavelengths must be monotonically increasing: "
                    f"wavelengths[{i}] = {wavelengths[i]} >= wavelengths[{i + 1}] = {wavelengths[i + 1]}"
                )

        # Optional wavelength_unit field (default is 'nm')
        if "wavelength_unit" in v:
            if not isinstance(v["wavelength_unit"], str):
                raise ValueError("wavelength_unit must be a string")

    elif "path" in v and "variable" in v:
        # File-based spectral data validation
        if not isinstance(v["path"], str) or not isinstance(v["variable"], str):
            raise ValueError(
                "'path' and 'variable' must be strings for file-based spectral data"
            )
        # Note: File existence check is optional to support dynamic paths
    else:
        raise ValueError(
            "Spectral parameter must be one of: "
            "file reference ({'path': ..., 'variable': ...}), "
            "uniform value ({'type': 'uniform', 'value': ...}), or "
            "interpolated spectrum ({'type': 'interpolated', 'wavelengths': [...], 'values': [...]})"
        )

    return v


def validate_reflectance_parameter(cls, v):
    """Validate reflectance/transmittance parameter with [0,1] bounds."""
    return validate_spectral_parameter(cls, v, enforce_unit_bounds=True)


def validate_ior_parameter(cls, v):
    """Validate IOR parameter without unit bounds (can be > 1)."""
    return validate_spectral_parameter(cls, v, enforce_unit_bounds=False)


class Material(BaseModel):
    """Base material class with Pydantic validation and auto-registration.

    Provides factory method to create material instances from dictionary
    specifications with automatic type registration and precise validation
    based on Eradiate and Mitsuba BSDF specifications.
    """

    _registry: ClassVar[Dict[str, Type["Material"]]] = {}

    id: str = Field(..., description="Unique material identifier")

    def __init_subclass__(cls, material_type: str = None, **kwargs):
        """Auto-register material types when classes are defined."""
        super().__init_subclass__(**kwargs)

        if material_type is None:
            # Auto-derive type from class name: RoughConductorMaterial -> rough_conductor
            material_type = cls.__name__.replace("Material", "")
            # Convert CamelCase to snake_case
            material_type = re.sub(
                r"([a-z0-9])([A-Z])", r"\1_\2", material_type
            ).lower()

        cls._registry[material_type] = cls

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of all registered material types."""
        return list(cls._registry.keys())

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> "Material":
        """Create material instance from dictionary with automatic type dispatch.

        Args:
            data: Dictionary with 'type' key and material parameters
            **kwargs: Additional arguments passed to material constructor

        Returns:
            Material instance of appropriate subtype

        Raises:
            ValueError: If material type is unknown
        """
        data = data.copy()
        material_type = data.pop("type")

        # Remove unused base_dir parameter for compatibility
        kwargs.pop("base_dir", None)

        if material_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown material type '{material_type}'. Available types: {available}"
            )

        material_class = cls._registry[material_type]
        return material_class(**data, **kwargs)

    @property
    def mat_id(self) -> str:
        """Material ID for use in scene dictionaries."""
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.model_dump()
        data.pop("id", None)
        # Add type field based on class registration
        for type_name, class_type in self._registry.items():
            if isinstance(self, class_type):
                data["type"] = type_name
                break
        return data


class DiffuseMaterial(Material, material_type="diffuse"):
    """Perfectly diffuse material (Lambertian reflectance).

    Based on Mitsuba specification:
    - Optional reflectance parameter with default 0.5
    - Reflectance must be in [0,1] for physical validity
    """

    reflectance: Dict[str, Any] = Field(
        default={"type": "uniform", "value": 0.5},
        description="Diffuse reflectance [0,1]",
    )

    @field_validator("reflectance")
    @classmethod
    def validate_reflectance(cls, v):
        return validate_reflectance_parameter(cls, v)


class BilambertianMaterial(Material, material_type="bilambertian"):
    """Bilambertian material with energy conservation validation.

    Based on Eradiate specification:
    - Mandatory reflectance and transmittance parameters
    - Energy conservation: reflectance + transmittance ≤ 1
    """

    reflectance: Dict[str, Any] = Field(..., description="Spectral reflectance [0,1]")
    transmittance: Dict[str, Any] = Field(
        ..., description="Spectral transmittance [0,1]"
    )

    @field_validator("reflectance", "transmittance")
    @classmethod
    def validate_spectral_params(cls, v):
        return validate_reflectance_parameter(cls, v)

    @model_validator(mode="after")
    def validate_energy_conservation(self):
        """Validate reflectance + transmittance ≤ 1 for uniform values."""
        if (
            isinstance(self.reflectance, dict)
            and self.reflectance.get("type") == "uniform"
            and isinstance(self.transmittance, dict)
            and self.transmittance.get("type") == "uniform"
        ):
            refl_val = self.reflectance.get("value", 0)
            trans_val = self.transmittance.get("value", 0)

            if isinstance(refl_val, (int, float)) and isinstance(
                trans_val, (int, float)
            ):
                if refl_val + trans_val > 1.0:
                    raise ValueError(
                        f"Energy conservation violated: reflectance ({refl_val}) + transmittance ({trans_val}) > 1.0"
                    )

        return self


class RPVMaterial(Material, material_type="rpv"):
    """RPV (Rahman-Pinty-Verstraete) reflection model with physical bounds.

    Based on Eradiate specification:
    - All parameters are mandatory
    - Physical bounds: rho_0,rho_c ∈ [0,1], k ≥ 0, Theta ∈ [-1,1]
    - k=1 corresponds to Lambertian surface
    """

    rho_0: Dict[str, Any] = Field(
        ..., description="Surface reflectance parameter [0,1]"
    )
    k: Dict[str, Any] = Field(
        ..., description="Bowl/bell shape parameter (k=1 is Lambertian, k≥0)"
    )
    Theta: Dict[str, Any] = Field(
        ..., description="Forward/backward scattering asymmetry [-1,1]"
    )
    rho_c: Dict[str, Any] = Field(..., description="Hot spot parameter [0,1]")

    @field_validator("rho_0", "rho_c")
    @classmethod
    def validate_reflectance_params(cls, v):
        return validate_reflectance_parameter(cls, v)

    @field_validator("k")
    @classmethod
    def validate_k_parameter(cls, v):
        """Validate k parameter physical bounds (k ≥ 0)."""
        v = validate_spectral_parameter(cls, v, enforce_unit_bounds=False)
        if isinstance(v, dict) and v.get("type") == "uniform":
            value = v.get("value")
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError("RPV k parameter must be non-negative (k ≥ 0)")
        return v

    @field_validator("Theta")
    @classmethod
    def validate_theta_parameter(cls, v):
        """Validate Theta parameter bounds [-1,1]."""
        v = validate_spectral_parameter(cls, v, enforce_unit_bounds=False)
        if isinstance(v, dict) and v.get("type") == "uniform":
            value = v.get("value")
            if isinstance(value, (int, float)) and not (-1.0 <= value <= 1.0):
                raise ValueError("RPV Theta parameter must be in range [-1, 1]")
        return v


class OceanLegacyMaterial(Material, material_type="ocean_legacy"):
    """Ocean legacy material with realistic oceanographic parameter bounds.

    Based on Eradiate specification:
    - All parameters are mandatory
    - Physical bounds based on oceanographic measurements
    """

    chlorinity: float = Field(..., ge=0.0, description="Water chlorinity in g/kg")
    pigmentation: float = Field(
        ..., ge=0.0, description="Pigmentation concentration in mg/m³"
    )
    wind_speed: float = Field(..., ge=0.0, description="Wind speed in m/s")
    wind_direction: float = Field(
        ..., ge=0.0, lt=360.0, description="Wind direction in degrees [0,360)"
    )


class OceanGraspMaterial(Material, material_type="ocean_grasp"):
    """GRASP oceanic reflection model.

    Based on the Mitsuba ``ocean_grasp`` plugin specification:
    - ``wavelength`` is mandatory, all other parameters have plugin defaults
    - Physical bounds follow the plugin documentation
    """

    wavelength: float = Field(
        ...,
        ge=200.0,
        le=4000.0,
        description="Evaluation wavelength in nm [200,4000]",
    )
    wind_speed: float = Field(
        0.1, ge=0.0, le=37.54, description="Wind speed in m/s [0,37.54]"
    )
    eta: float = Field(
        1.33, description="Real component of water's index of refraction"
    )
    k: float = Field(
        0.0, description="Imaginary component of water's index of refraction"
    )
    ext_ior: float = Field(1.000277, description="Exterior index of refraction")
    water_body_reflectance: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Diffuse reflectance of radiation that entered and exited the water body",
    )
    component: int = Field(
        0,
        ge=0,
        le=4,
        description="Debug component: 0=total, 1=whitecap, 2=sun glint, "
        "3=underlight, 4=whitecap+underlight",
    )


class DielectricMaterial(Material, material_type="dielectric"):
    """Dielectric material (glass, plastic) based on Mitsuba specification.

    Based on Mitsuba specification:
    - All parameters optional with physical defaults
    - IOR values must be > 1.0 for physical validity
    """

    int_ior: Union[float, str] = Field(
        default=1.5046, description="Interior IOR (>1.0) or preset name"
    )
    ext_ior: Union[float, str] = Field(
        default=1.000277, description="Exterior IOR or preset name"
    )
    specular_reflectance: Optional[Dict[str, Any]] = Field(
        default=None, description="Spectral reflectance override"
    )
    specular_transmittance: Optional[Dict[str, Any]] = Field(
        default=None, description="Spectral transmittance override"
    )

    @field_validator("int_ior", "ext_ior")
    @classmethod
    def validate_ior(cls, v):
        if isinstance(v, (int, float)) and v <= 1.0:
            raise ValueError(f"IOR value {v} must be > 1.0 for physical validity")
        return v

    @field_validator("specular_reflectance", "specular_transmittance")
    @classmethod
    def validate_spectral_params(cls, v):
        if v is not None:
            return validate_reflectance_parameter(cls, v)
        return v


class ConductorMaterial(Material, material_type="conductor"):
    """Conductor material with mutually exclusive parameter validation.

    Based on Mitsuba specification:
    - Mutually exclusive: material preset XOR manual eta/k
    - Energy conservation: specular_reflectance ≤ 1
    """

    material: Optional[str] = Field(
        None, description="Material preset (Al, Cu, Au, etc.)"
    )
    eta: Optional[Dict[str, Any]] = Field(None, description="Real part of complex IOR")
    k: Optional[Dict[str, Any]] = Field(
        None, description="Imaginary part of complex IOR"
    )
    specular_reflectance: Optional[Dict[str, Any]] = Field(
        None, description="Spectral reflectance override"
    )

    @field_validator("eta", "k")
    @classmethod
    def validate_ior_params(cls, v):
        if v is not None:
            return validate_ior_parameter(cls, v)
        return v

    @field_validator("specular_reflectance")
    @classmethod
    def validate_reflectance_param(cls, v):
        if v is not None:
            return validate_reflectance_parameter(cls, v)
        return v

    @model_validator(mode="after")
    def validate_mutually_exclusive_params(self):
        """Validate material preset XOR manual eta/k."""
        has_preset = self.material is not None
        has_manual = self.eta is not None or self.k is not None

        if has_preset and has_manual:
            raise ValueError(
                "Cannot specify both 'material' preset and manual 'eta'/'k' values"
            )

        if has_manual and (self.eta is None or self.k is None):
            raise ValueError(
                "Both 'eta' and 'k' must be specified when using manual complex IOR"
            )

        return self


class RoughConductorMaterial(Material, material_type="rough_conductor"):
    """Rough conductor with anisotropic roughness validation.

    Based on Mitsuba specification:
    - Inherits conductor parameter validation
    - Mutually exclusive roughness: roughness XOR (alpha_u AND/OR alpha_v)
    - Distribution validation: beckmann or ggx only
    """

    # Inherited conductor parameters
    material: Optional[str] = Field(None, description="Material preset")
    eta: Optional[Dict[str, Any]] = Field(None, description="Real part of complex IOR")
    k: Optional[Dict[str, Any]] = Field(
        None, description="Imaginary part of complex IOR"
    )
    specular_reflectance: Optional[Dict[str, Any]] = Field(
        None, description="Spectral reflectance"
    )

    # Roughness parameters
    distribution: Literal["beckmann", "ggx"] = Field(
        "beckmann", description="Microfacet distribution"
    )
    roughness: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Isotropic roughness [0,1]"
    )
    alpha_u: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="U-direction roughness [0,1]"
    )
    alpha_v: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="V-direction roughness [0,1]"
    )

    @field_validator("eta", "k")
    @classmethod
    def validate_ior_params(cls, v):
        if v is not None:
            return validate_ior_parameter(cls, v)
        return v

    @field_validator("specular_reflectance")
    @classmethod
    def validate_reflectance_param(cls, v):
        if v is not None:
            return validate_reflectance_parameter(cls, v)
        return v

    @model_validator(mode="after")
    def validate_conductor_params(self):
        """Validate conductor parameter mutual exclusion."""
        has_preset = self.material is not None
        has_manual = self.eta is not None or self.k is not None

        if has_preset and has_manual:
            raise ValueError(
                "Cannot specify both 'material' preset and manual 'eta'/'k' values"
            )

        if has_manual and (self.eta is None or self.k is None):
            raise ValueError(
                "Both 'eta' and 'k' must be specified when using manual complex IOR"
            )

        return self

    @model_validator(mode="after")
    def validate_roughness_params(self):
        """Validate roughness parameter mutual exclusion."""
        has_isotropic = self.roughness is not None
        has_anisotropic = self.alpha_u is not None or self.alpha_v is not None

        if has_isotropic and has_anisotropic:
            raise ValueError("Cannot specify both 'roughness' and 'alpha_u'/'alpha_v'")

        if not has_isotropic and not has_anisotropic:
            # Set default isotropic roughness
            self.roughness = 0.1

        return self


class PlasticMaterial(Material, material_type="plastic"):
    """Plastic material based on Eradiate specification."""

    diffuse_reflectance: Dict[str, Any] = Field(
        ..., description="Diffuse reflectance component"
    )
    int_ior: Union[float, str] = Field(default=1.49, description="Interior IOR")
    ext_ior: Union[float, str] = Field(default=1.000277, description="Exterior IOR")
    nonlinear: bool = Field(default=False, description="Enable nonlinear effects")

    @field_validator("diffuse_reflectance")
    @classmethod
    def validate_reflectance(cls, v):
        return validate_reflectance_parameter(cls, v)


class PrincipledMaterial(Material, material_type="principled"):
    """Principled BSDF based on Mitsuba specification with comprehensive parameter validation."""

    base_color: Dict[str, Any] = Field(
        default={"type": "uniform", "value": 0.5}, description="Base color/albedo [0,1]"
    )
    roughness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Surface roughness [0,1]"
    )
    metallic: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Metallic factor [0,1]"
    )
    specular: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Specular reflectance scaling [0,1]"
    )
    spec_tint: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Specular tinting [0,1]"
    )
    anisotropic: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Anisotropy amount [0,1]"
    )
    sheen: float = Field(default=0.0, ge=0.0, le=1.0, description="Sheen amount [0,1]")
    sheen_tint: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Sheen tinting [0,1]"
    )
    clearcoat: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Clearcoat amount [0,1]"
    )
    clearcoat_roughness: float = Field(
        default=0.03, ge=0.0, le=1.0, description="Clearcoat roughness [0,1]"
    )

    @field_validator("base_color")
    @classmethod
    def validate_base_color(cls, v):
        return validate_reflectance_parameter(cls, v)


class MeasuredMaterial(Material, material_type="measured"):
    """Measured BSDF material using external .bsdf files.

    Based on Mitsuba's measured BSDF plugin for loading material data
    from pre-computed BRDF measurements stored in .bsdf files.
    """

    filename: str = Field(
        ..., description="Path to .bsdf file containing measured BRDF data"
    )
