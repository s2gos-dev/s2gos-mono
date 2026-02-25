"""Eradiate-specific material functionality for S2GOS simulator backend."""

import logging
from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr
from eradiate import KernelContext
from eradiate.kernel import SceneParameter
from upath import UPath

try:
    import mitsuba as mi
    from eradiate import KernelContext
    from eradiate.kernel import (
        KernelSceneParameterFlags,
        SceneParameter,
        SearchSceneParameter,
    )
    from eradiate.scenes.spectra import InterpolatedSpectrum

    ERADIATE_AVAILABLE = True
except ImportError:
    KernelContext = None
    InterpolatedSpectrum = None
    SceneParameter = None
    SearchSceneParameter = None
    KernelSceneParameterFlags = None
    mi = None
    ERADIATE_AVAILABLE = False


def _create_spectral_callable(
    spectral_dict: Dict[str, Any],
) -> Callable[["KernelContext"], float]:
    """Create callable function from spectral parameter dictionary.

    Supports spectral file references, uniform values, and interpolated spectra:
    - File reference: {"path": "spectrum.nc", "variable": "reflectance"}
    - Uniform value: {"type": "uniform", "value": 0.5}
    - Interpolated: {"type": "interpolated", "wavelengths": [400, 500, 600], "values": [0.2, 0.5, 0.3]}

    Args:
        spectral_dict: Dictionary with spectral data specification

    Returns:
        Callable that evaluates spectral data for given KernelContext
    """
    if not ERADIATE_AVAILABLE:
        # Fallback when Eradiate is not available
        def dummy_func(_ctx) -> float:
            return 0.5

        return dummy_func

    if spectral_dict.get("type") == "uniform":
        uniform_value = spectral_dict["value"]

        if isinstance(uniform_value, (int, float)):

            def uniform_scalar_func(_ctx) -> float:
                return float(uniform_value)

            logging.info(f"Using uniform scalar value: {uniform_value}")
            return uniform_scalar_func

        elif isinstance(uniform_value, (list, tuple)) and len(uniform_value) == 3:
            r, g, b = uniform_value

            def uniform_rgb_func(ctx: "KernelContext") -> float:
                wavelength = (
                    ctx.si.wavelength.m_as("nanometer")
                    if hasattr(ctx.si, "wavelength")
                    else 550.0
                )

                if wavelength <= 440:
                    return float(b)
                elif wavelength <= 540:  # Green-ish
                    return float(g)
                else:
                    return float(r)

            logging.info(f"Using uniform RGB value: {uniform_value}")
            return uniform_rgb_func

        else:
            raise ValueError(
                f"Invalid uniform value format: {uniform_value}. Expected float, RGB array, or single value."
            )

    if spectral_dict.get("type") == "interpolated":
        from eradiate.units import unit_registry as ureg

        wavelengths = spectral_dict["wavelengths"]
        values = spectral_dict["values"]
        wavelength_unit = spectral_dict.get("wavelength_unit", "nm")

        wavelengths_with_units = np.array(wavelengths) * ureg(wavelength_unit)
        values_array = np.array(values) * ureg.dimensionless

        try:
            spectrum = InterpolatedSpectrum(
                wavelengths=wavelengths_with_units, values=values_array
            )

            def interpolated_func(ctx: "KernelContext") -> float:
                return spectrum.eval(ctx.si).m_as("dimensionless")

            logging.info(
                f"Created interpolated spectrum: {len(wavelengths)} wavelength points "
                f"from {wavelengths[0]} to {wavelengths[-1]} {wavelength_unit}"
            )
            return interpolated_func

        except Exception as e:
            raise ValueError(
                f"Failed to create interpolated spectrum from wavelengths={wavelengths}, values={values}: {e}"
            ) from e

    if "path" not in spectral_dict or "variable" not in spectral_dict:
        raise ValueError(
            f"Invalid spectral dictionary format: {spectral_dict}. "
            "Must be one of: file reference ({'path': ..., 'variable': ...}), "
            "uniform value ({'type': 'uniform', 'value': ...}), or "
            "interpolated spectrum ({'type': 'interpolated', 'wavelengths': [...], 'values': [...]})."
        )

    file_path = spectral_dict["path"]
    variable = spectral_dict["variable"]

    full_path = UPath(file_path)
    if not full_path.is_absolute():
        raise ValueError(
            f"Spectral file path must be absolute: {file_path}. "
            "Paths should be resolved to absolute at material load time."
        )

    # Load spectral data using Eradiate
    try:
        # Check if file exists before trying to load
        from s2gos_utils.io.paths import exists

        if not exists(full_path):
            raise FileNotFoundError(f"Spectral data file not found: {full_path}")

        # Load the spectral data directly with xarray
        dataset = xr.load_dataset(full_path)
        if variable not in dataset.data_vars:
            raise KeyError(
                f"Variable '{variable}' not found in dataset. Available: {list(dataset.data_vars.keys())}"
            )

        da = dataset[variable]
        spectrum = InterpolatedSpectrum.from_dataarray(dataarray=da)

        def spectral_func(ctx: "KernelContext") -> float:
            return spectrum.eval(ctx.si).m_as("dimensionless")

        logging.info(
            f"Successfully loaded spectral data: {full_path} (variable: {variable})"
        )
        return spectral_func

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Spectral data file not found: {full_path}") from e
    except KeyError as e:
        raise ValueError(
            f"Variable '{variable}' not found in spectral dataset {full_path}. {e}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to load spectral data from {full_path} (variable: {variable}): {e}"
        ) from e


def _create_scalar_callable(
    value: Union[int, float],
) -> Callable[["KernelContext"], float]:
    """Create callable function from scalar value for Eradiate context.

    Args:
        value: Scalar value (int or float)

    Returns:
        Callable that returns the scalar value for any KernelContext
    """

    def scalar_func(_ctx: "KernelContext") -> float:
        return float(value)

    return scalar_func


def _declare_mono_scene_parameter(
    func: Callable[["KernelContext"], float], node_id: str, param: str
) -> "SceneParameter":
    """Create SceneParameter for monochromatic mode.

    Args:
        func: Function that computes parameter value from kernel context
        node_id: Eradiate node identifier
        param: Parameter path relative to the node

    Returns:
        SceneParameter configured for spectral evaluation
    """
    if not ERADIATE_AVAILABLE:
        raise ImportError("Eradiate is not available")

    return SceneParameter(
        func,
        search=SearchSceneParameter(
            node_type=mi.BSDF, node_id=node_id, parameter_relpath=param
        ),
        flags=KernelSceneParameterFlags.SPECTRAL,
    )


class EradiateMaterialAdapter:
    """Adapter for converting S2GOS materials to Eradiate kernel dictionaries."""

    @staticmethod
    def load_spectral_data(file_path: UPath, variable: str = "reflectance") -> Any:
        """Load spectral data from file using Eradiate's InterpolatedSpectrum."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        import xarray as xr

        data = xr.open_dataset(file_path)
        return InterpolatedSpectrum.from_dataarray(dataarray=data[variable])

    @staticmethod
    def create_diffuse_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for diffuse material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {
            "type": "diffuse",
            "reflectance": 0.0,
        }

        return result

    @staticmethod
    def create_diffuse_kpmap(material) -> dict:
        """Generate Eradiate parameter map for diffuse material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = f"_mat_{material.mat_id}"
        param = "reflectance.value"
        # Convert dict to callable function
        reflectance_func = _create_spectral_callable(material.reflectance)
        result[f"{material.mat_id}.{param}"] = _declare_mono_scene_parameter(
            reflectance_func, node_id=node_id, param=param
        )

        return result

    @staticmethod
    def create_bilambertian_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for bilambertian material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {"type": "bilambertian"}

        return result

    @staticmethod
    def create_bilambertian_kpmap(material) -> dict:
        """Generate Eradiate parameter map for bilambertian material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = f"_mat_{material.mat_id}"
        for param, spectral_dict in [
            ("reflectance.value", material.reflectance),
            ("transmittance.value", material.transmittance),
        ]:
            # Convert dict to callable function
            spectral_func = _create_spectral_callable(spectral_dict)
            result[f"{material.mat_id}.{param}"] = _declare_mono_scene_parameter(
                spectral_func, node_id=node_id, param=param
            )

        return result

    @staticmethod
    def create_rpv_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for RPV material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {
            "type": "rpv",
            "rho_c": 0.5,
        }

        return result

    @staticmethod
    def create_rpv_kpmap(material) -> dict:
        """Generate Eradiate parameter map for RPV material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}

        node_id = f"_mat_{material.mat_id}"
        for param, spectral_dict in [
            ("rho_0.value", material.rho_0),
            ("k.value", material.k),
            ("g.value", material.Theta),
            ("rho_c.value", material.rho_c),
        ]:
            # Convert dict to callable function
            spectral_func = _create_spectral_callable(spectral_dict)
            result[f"{material.mat_id}.{param}"] = _declare_mono_scene_parameter(
                spectral_func, node_id=node_id, param=param
            )

        return result

    @staticmethod
    def create_ocean_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for ocean material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {
            "type": "ocean_legacy",
            "wavelength": 550.0,  # Default wavelength for ocean legacy material
            "chlorinity": getattr(material, "chlorinity", 0.0),
            "pigmentation": getattr(material, "pigmentation", 5.0),
            "wind_speed": getattr(material, "wind_speed", 2.0),
            "wind_direction": getattr(material, "wind_direction", 90.0),
        }
        return kdict

    @staticmethod
    def create_ocean_kpmap(_material) -> dict:
        """Generate Eradiate parameter map for ocean material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        return {}

    @staticmethod
    def create_dielectric_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for dielectric material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {
            "type": "dielectric",
            "int_ior": material.int_ior,
            "ext_ior": material.ext_ior,
        }

        if material.specular_reflectance is not None:
            kdict["specular_reflectance"] = 0.0
        if material.specular_transmittance is not None:
            kdict["specular_transmittance"] = 0.0

        return kdict

    @staticmethod
    def create_dielectric_kpmap(material) -> dict:
        """Generate Eradiate parameter map for dielectric material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}
        node_id = f"_mat_{material.mat_id}"

        if material.specular_reflectance is not None:
            reflectance_func = _create_spectral_callable(material.specular_reflectance)
            result[f"{material.mat_id}.specular_reflectance.value"] = (
                _declare_mono_scene_parameter(
                    reflectance_func,
                    node_id=node_id,
                    param="specular_reflectance.value",
                )
            )

        if material.specular_transmittance is not None:
            transmittance_func = _create_spectral_callable(
                material.specular_transmittance
            )
            result[f"{material.mat_id}.specular_transmittance.value"] = (
                _declare_mono_scene_parameter(
                    transmittance_func,
                    node_id=node_id,
                    param="specular_transmittance.value",
                )
            )

        return result

    @staticmethod
    def create_conductor_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for conductor material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {"type": "conductor"}

        # Handle material preset directly (S2GOS format)
        if hasattr(material, "material") and material.material is not None:
            kdict["material"] = material.material
        else:
            # Handle IOR specification (legacy format)
            ior_spec = getattr(material, "ior", None)
            if isinstance(ior_spec, str):
                kdict["material"] = ior_spec
            elif isinstance(ior_spec, dict):
                if "preset" in ior_spec:
                    kdict["material"] = ior_spec["preset"]
                elif "eta" in ior_spec and "k" in ior_spec:
                    # Add scalar values directly; spectral values are handled by kpmap
                    if isinstance(ior_spec["eta"], (int, float)):
                        kdict["eta"] = ior_spec["eta"]
                    if isinstance(ior_spec["k"], (int, float)):
                        kdict["k"] = ior_spec["k"]

        # Handle explicit eta/k values (S2GOS format)
        if hasattr(material, "eta") and material.eta is not None:
            if isinstance(material.eta, dict):
                # Spectral handled by kpmap
                pass
            else:
                kdict["eta"] = material.eta
        if hasattr(material, "k") and material.k is not None:
            if isinstance(material.k, dict):
                # Spectral handled by kpmap
                pass
            else:
                kdict["k"] = material.k

        # Handle optional specular reflectance (placeholder for kpmap)
        if getattr(material, "specular_reflectance", None) is not None:
            kdict["specular_reflectance"] = 0.0

        return kdict

    @staticmethod
    def create_conductor_kpmap(material) -> dict:
        """Generate Eradiate parameter map for conductor material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}
        node_id = f"_mat_{material.mat_id}"

        # Handle spectral eta/k values (S2GOS format)
        for param_name in ["eta", "k"]:
            if hasattr(material, param_name):
                param_value = getattr(material, param_name)
                if isinstance(param_value, dict):
                    spectral_func = _create_spectral_callable(param_value)
                    param_path = f"{param_name}.value"
                    result[f"{material.mat_id}.{param_path}"] = (
                        _declare_mono_scene_parameter(
                            spectral_func, node_id=node_id, param=param_path
                        )
                    )

        # Handle spectral IOR from a dictionary (legacy format)
        ior_spec = getattr(material, "ior", None)
        if isinstance(ior_spec, dict):
            for param_name in ["eta", "k"]:
                if param_name in ior_spec and isinstance(ior_spec[param_name], dict):
                    # Avoid duplicate processing
                    if f"{material.mat_id}.{param_name}.value" not in result:
                        spectral_func = _create_spectral_callable(ior_spec[param_name])
                        param_path = f"{param_name}.value"
                        result[f"{material.mat_id}.{param_path}"] = (
                            _declare_mono_scene_parameter(
                                spectral_func, node_id=node_id, param=param_path
                            )
                        )

        # Handle spectral reflectance override
        spec_refl = getattr(material, "specular_reflectance", None)
        if isinstance(spec_refl, dict):
            reflectance_func = _create_spectral_callable(spec_refl)
            param_path = "specular_reflectance.value"
            result[f"{material.mat_id}.{param_path}"] = _declare_mono_scene_parameter(
                reflectance_func, node_id=node_id, param=param_path
            )

        return result

    @staticmethod
    def create_rough_conductor_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for rough conductor material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {"type": "roughconductor"}

        if hasattr(material, "distribution"):
            kdict["distribution"] = material.distribution

        has_anisotropic = (
            hasattr(material, "alpha_u") and material.alpha_u is not None
        ) or (hasattr(material, "alpha_v") and material.alpha_v is not None)

        if has_anisotropic:
            if (
                hasattr(material, "alpha_u")
                and material.alpha_u is not None
                and not isinstance(material.alpha_u, dict)
            ):
                kdict["alpha_u"] = material.alpha_u
            if (
                hasattr(material, "alpha_v")
                and material.alpha_v is not None
                and not isinstance(material.alpha_v, dict)
            ):
                kdict["alpha_v"] = material.alpha_v
        else:
            # Use isotropic roughness (alpha or roughness field)
            alpha = getattr(material, "alpha", getattr(material, "roughness", None))
            if alpha is not None and not isinstance(alpha, dict):
                kdict["alpha"] = alpha

        # Sampling strategy
        if hasattr(material, "sample_visible"):
            kdict["sample_visible"] = material.sample_visible

        # Handle material preset directly (S2GOS format)
        if hasattr(material, "material") and material.material is not None:
            kdict["material"] = material.material
        else:
            # Handle IOR specification (legacy format)
            ior_spec = getattr(material, "ior", None)
            if isinstance(ior_spec, str):
                kdict["material"] = ior_spec
            elif isinstance(ior_spec, dict):
                if "preset" in ior_spec:
                    kdict["material"] = ior_spec["preset"]
                elif "eta" in ior_spec and "k" in ior_spec:
                    if isinstance(ior_spec["eta"], (int, float)):
                        kdict["eta"] = ior_spec["eta"]
                    if isinstance(ior_spec["k"], (int, float)):
                        kdict["k"] = ior_spec["k"]

        # Handle explicit eta/k values (S2GOS format)
        if hasattr(material, "eta") and material.eta is not None:
            if isinstance(material.eta, dict):
                # Spectral handled by kpmap
                pass
            else:
                kdict["eta"] = material.eta
        if hasattr(material, "k") and material.k is not None:
            if isinstance(material.k, dict):
                # Spectral handled by kpmap
                pass
            else:
                kdict["k"] = material.k

        # Optional specular reflectance (placeholder for kpmap)
        if getattr(material, "specular_reflectance", None) is not None:
            kdict["specular_reflectance"] = 0.0

        return kdict

    @staticmethod
    def create_rough_conductor_kpmap(material) -> dict:
        """Generate Eradiate parameter map for rough conductor material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}
        node_id = f"_mat_{material.mat_id}"

        # Spectral roughness (alpha, alpha_u, alpha_v)
        # Use a mapping to avoid repetitive code
        param_map = {
            "alpha": "alpha.value",
            "roughness": "alpha.value",
            "alpha_u": "alpha_u.value",
            "alpha_v": "alpha_v.value",
        }
        for attr, param_path in param_map.items():
            if f"{node_id}.{param_path}" in result:
                continue  # Avoid overwriting with alias

            if (
                attr == "alpha"
                and not hasattr(material, "alpha")
                and hasattr(material, "roughness")
            ):
                spectral_dict = getattr(material, "roughness", None)
            else:
                spectral_dict = getattr(material, attr, None)
            if isinstance(spectral_dict, dict):
                spectral_func = _create_spectral_callable(spectral_dict)
                result[f"{material.mat_id}.{param_path}"] = (
                    _declare_mono_scene_parameter(
                        spectral_func, node_id=node_id, param=param_path
                    )
                )

        # Handle spectral eta/k values (S2GOS format, same as smooth conductor)
        for param_name in ["eta", "k"]:
            if hasattr(material, param_name):
                param_value = getattr(material, param_name)
                if isinstance(param_value, dict):
                    spectral_func = _create_spectral_callable(param_value)
                    param_path = f"{param_name}.value"
                    result[f"{material.mat_id}.{param_path}"] = (
                        _declare_mono_scene_parameter(
                            spectral_func, node_id=node_id, param=param_path
                        )
                    )

        ior_spec = getattr(material, "ior", None)
        if isinstance(ior_spec, dict):
            for param_name in ["eta", "k"]:
                if param_name in ior_spec and isinstance(ior_spec[param_name], dict):
                    # Avoid duplicate processing
                    if f"{material.mat_id}.{param_name}.value" not in result:
                        spectral_func = _create_spectral_callable(ior_spec[param_name])
                        param_path = f"{param_name}.value"
                        result[f"{material.mat_id}.{param_path}"] = (
                            _declare_mono_scene_parameter(
                                spectral_func, node_id=node_id, param=param_path
                            )
                        )

        spec_refl = getattr(material, "specular_reflectance", None)
        if isinstance(spec_refl, dict):
            reflectance_func = _create_spectral_callable(spec_refl)
            param_path = "specular_reflectance.value"
            result[f"{material.mat_id}.{param_path}"] = _declare_mono_scene_parameter(
                reflectance_func, node_id=node_id, param=param_path
            )

        return result

    @staticmethod
    def create_plastic_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for plastic material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {
            "type": "plastic",
            "int_ior": material.int_ior,
            "ext_ior": material.ext_ior,
            "diffuse_reflectance": 0.0,
        }

        # Add optional parameters
        if (
            hasattr(material, "roughness")
            and material.roughness is not None
            and material.roughness > 0.0
        ):
            kdict["alpha"] = material.roughness

        if hasattr(material, "nonlinear") and material.nonlinear:
            kdict["nonlinear"] = True

        return kdict

    @staticmethod
    def create_plastic_kpmap(material) -> dict:
        """Generate Eradiate parameter map for plastic material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}
        node_id = f"_mat_{material.mat_id}"

        diffuse_func = _create_spectral_callable(material.diffuse_reflectance)
        result[f"{node_id}.diffuse_reflectance.value"] = _declare_mono_scene_parameter(
            diffuse_func, node_id=node_id, param="diffuse_reflectance.value"
        )

        return result

    @staticmethod
    def create_hamster_kdict(material_id: str, albedo_data: xr.DataArray) -> dict:
        """Generate Eradiate kernel dictionary for HAMSTER albedo material.

        Args:
            material_id: Material identifier (e.g., "_mat_baresoil_hamster")
            albedo_data: HAMSTER albedo DataArray with (lat, lon, wavelength) dimensions

        Returns:
            Dictionary with texture and material definitions for HAMSTER albedo
        """
        texture_data = np.ones_like(albedo_data.values[:, :, :3])
        texture_id = f"texture_{material_id}"

        # Validate texture data - ensure it has valid dimensions
        if texture_data.size == 0:
            raise ValueError(
                f"HAMSTER texture data is empty (0x0) for {material_id}. "
                "This likely means the HAMSTER data has no coverage for the scene area."
            )

        result = {
            texture_id: {
                "type": "bitmap",
                "id": texture_id,
                "filter_type": "bilinear",
                "wrap_mode": "clamp",
                "data": texture_data,
                "raw": True,
            },
            material_id: {
                "type": "diffuse",
                "reflectance": {"type": "ref", "id": texture_id},
            },
        }

        return result

    @staticmethod
    def create_hamster_kpmap(material_id: str, albedo_data: xr.DataArray) -> dict:
        """Generate Eradiate parameter map for HAMSTER albedo material.

        Args:
            material_id: Material identifier
            albedo_data: HAMSTER albedo DataArray with spectral interpolation support

        Returns:
            Dictionary with scene parameters for spectral interpolation
        """
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        def hamster_spectral_func(ctx: "KernelContext") -> np.ndarray:
            """Interpolate HAMSTER albedo data for given wavelength."""
            interpolated = albedo_data.sel(wavelength=ctx.si.w, method="nearest")
            return interpolated.values[..., np.newaxis]

        param_key = f"{material_id}.reflectance.data"

        result = {
            param_key: SceneParameter(
                hamster_spectral_func,
                search=SearchSceneParameter(
                    node_type=mi.BSDF,
                    node_id=material_id,
                    parameter_relpath="reflectance.data",
                ),
                flags=KernelSceneParameterFlags.SPECTRAL,
            )
        }

        return result

    @staticmethod
    def create_principled_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for principled material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {"type": "principled"}
        return result

    @staticmethod
    def create_principled_kpmap(material) -> dict:
        """Generate Eradiate parameter map for principled material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        result = {}
        node_id = f"_mat_{material.mat_id}"

        # Handle base_color parameter
        base_color_func = _create_spectral_callable(material.base_color)
        result[f"{node_id}.base_color.value"] = _declare_mono_scene_parameter(
            base_color_func, node_id=node_id, param="base_color.value"
        )

        # Handle scalar parameters
        scalar_params = [
            ("metallic", "metallic"),
            ("roughness", "roughness"),
            ("specular", "specular"),
            ("specular_tint", "specular_tint"),
            ("anisotropic", "anisotropic"),
            ("sheen", "sheen"),
            ("sheen_tint", "sheen_tint"),
            ("clearcoat", "clearcoat"),
            ("clearcoat_roughness", "clearcoat_roughness"),
        ]

        for attr_name, param_name in scalar_params:
            if hasattr(material, attr_name):
                attr_value = getattr(material, attr_name)
                if attr_value is not None:
                    scalar_func = _create_scalar_callable(attr_value)
                    result[f"{node_id}.{param_name}"] = _declare_mono_scene_parameter(
                        scalar_func, node_id=node_id, param=param_name
                    )

        return result

    @staticmethod
    def create_measured_kdict(material) -> dict:
        """Generate Eradiate kernel dictionary for measured BSDF material."""
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        kdict = {"type": "measured", "filename": material.filename}

        return kdict

    @staticmethod
    def create_measured_kpmap(material) -> dict:
        """Generate Eradiate parameter map for measured BSDF material.

        Measured BSDF materials don't require spectral parameter updates
        as all data is contained in the .bsdf file.
        """
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate is not available")

        # No spectral parameters needed for measured BSDF
        return {}
