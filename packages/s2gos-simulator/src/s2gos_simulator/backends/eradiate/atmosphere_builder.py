"""Atmosphere configuration builder for Eradiate backend."""

import logging

import numpy as np
from s2gos_utils.scene import SceneDescription

logger = logging.getLogger(__name__)

try:
    from eradiate.radprops import AbsorptionDatabase
    from eradiate.scenes.atmosphere import (
        ExponentialParticleDistribution,
        GaussianParticleDistribution,
        HeterogeneousAtmosphere,
        HomogeneousAtmosphere,
        MolecularAtmosphere,
        ParticleLayer,
        UniformParticleDistribution,
    )
    from eradiate.units import unit_registry as ureg

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class AtmosphereBuilder:
    """Builder for creating Eradiate atmosphere configurations from scene descriptions."""

    def __init__(self):
        """Initialize atmosphere builder."""
        pass

    def create_geometry_from_atmosphere(self, scene_description: SceneDescription):
        """Create geometry with bounds matching the atmosphere configuration.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Geometry dictionary with TOA altitude
        """
        atmosphere = scene_description.atmosphere
        toa = atmosphere["toa"]

        geometry = {
            "type": "plane_parallel",
            "toa_altitude": toa,
        }

        return geometry

    def create_atmosphere_from_config(self, scene_description: SceneDescription):
        """Create atmosphere based on scene description format.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Eradiate atmosphere object (MolecularAtmosphere, HomogeneousAtmosphere, or HeterogeneousAtmosphere)

        Raises:
            ValueError: If atmosphere type is unknown or not specified
        """
        atmosphere = scene_description.atmosphere
        atmosphere_type = atmosphere["type"] if "type" in atmosphere else None

        if not atmosphere_type:
            raise ValueError("Atmosphere configuration must specify 'type' field")

        if atmosphere_type == "molecular":
            return self._create_molecular_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "homogeneous":
            return self._create_homogeneous_atmosphere_from_scene(atmosphere)
        elif atmosphere_type == "heterogeneous":
            return self._create_heterogeneous_atmosphere_from_scene(atmosphere)
        else:
            raise ValueError(f"Unknown atmosphere type: {atmosphere_type}")

    def _create_molecular_atmosphere_from_dict(self, mol_dict):
        """Create molecular atmosphere from dictionary.

        Supports either joseki identifiers or CAMS NetCDF files.

        Args:
            mol_dict: Dictionary with molecular atmosphere configuration

        Returns:
            MolecularAtmosphere object
        """
        if "thermoprops_file" in mol_dict:
            import xarray as xr
            from upath import UPath

            thermoprops_file = UPath(mol_dict["thermoprops_file"])
            thermoprops = xr.open_dataset(thermoprops_file).squeeze(drop=True)
        else:
            thermoprops_id = mol_dict.get(
                "thermoprops_identifier", "afgl_1986-us_standard"
            )
            altitude_min = mol_dict["altitude_min"]
            altitude_max = mol_dict["altitude_max"]
            altitude_step = mol_dict["altitude_step"]
            num_steps = int((altitude_max - altitude_min) / altitude_step) + 1

            thermoprops = {
                "identifier": thermoprops_id,
                "z": np.linspace(altitude_min, altitude_max, num_steps) * ureg.m,
            }

        absorption_data = (
            mol_dict.get("absorption_database") or AbsorptionDatabase.default()
        )

        atmosphere = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data=absorption_data,
            has_absorption=mol_dict.get("has_absorption", True),
            has_scattering=mol_dict.get("has_scattering", True),
        )

        return atmosphere

    def _create_particle_layer_from_dict(self, layer_dict):
        """Create particle layer from dictionary.

        Args:
            layer_dict: Dictionary with particle layer configuration

        Returns:
            ParticleLayer object
        """
        # Core fields always present from config
        dist_type = layer_dict["distribution_type"]  # Always serialized

        if dist_type == "exponential":
            # Distribution params are optional, use defaults if not present
            if "rate" in layer_dict.keys():
                if "scale" in layer_dict.keys():
                    logger.warning(
                        "scale and rate should be mutually exclusive in exponential distribution, using rate"
                    )
                distribution = ExponentialParticleDistribution(
                    scale=layer_dict.get("rate", 5.0)
                )
            else:
                distribution = ExponentialParticleDistribution(
                    rate=layer_dict.get("scale", 0.2)
                )
        elif dist_type == "gaussian":
            # Gaussian params may not be present, use defaults
            distribution = GaussianParticleDistribution(
                mean=layer_dict.get("center_altitude", 0.5),
                std=layer_dict.get("width", 1 / 6),
            )
        else:
            distribution = UniformParticleDistribution(
                {"bounds": layer_dict.get("bounds", [0, 1])}
            )

        layer = ParticleLayer(
            dataset=layer_dict["aerosol_dataset"],
            tau_ref=layer_dict["optical_thickness"],
            w_ref=layer_dict["reference_wavelength"],
            bottom=layer_dict["altitude_bottom"],
            top=layer_dict["altitude_top"],
            distribution=distribution,
            has_absorption=layer_dict["has_absorption"],
        )

        return layer

    def _create_molecular_atmosphere_from_scene(self, atmosphere_dict):
        """Create molecular atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            MolecularAtmosphere object
        """
        if "molecular_atmosphere" in atmosphere_dict:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            return self._create_molecular_atmosphere_from_dict(mol_dict)
        else:
            return self._create_molecular_atmosphere_from_dict({})

    def _create_homogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create homogeneous atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HomogeneousAtmosphere object
        """
        atmosphere = HomogeneousAtmosphere(
            boa=atmosphere_dict["boa"],
            toa=atmosphere_dict["toa"],
            particle_layers=[
                ParticleLayer(
                    dataset=atmosphere_dict["aerosol_ds"],
                    optical_thickness=atmosphere_dict["aerosol_ot"],
                    altitude_bottom=atmosphere_dict["boa"],
                    altitude_top=atmosphere_dict["toa"],
                    reference_wavelength=atmosphere_dict["reference_wavelength"],
                )
            ],
        )

        return atmosphere

    def _create_heterogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create heterogeneous atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HeterogeneousAtmosphere object
        """
        has_molecular = (
            atmosphere_dict.get("has_molecular_atmosphere", False)
            or "molecular_atmosphere" in atmosphere_dict
        )
        has_particles = (
            atmosphere_dict.get("has_particle_layers", False)
            or "particle_layers" in atmosphere_dict
        )

        molecular_atmosphere = None
        particle_layers = []

        if has_molecular:
            mol_dict = atmosphere_dict["molecular_atmosphere"]
            molecular_atmosphere = self._create_molecular_atmosphere_from_dict(mol_dict)

        if has_particles:
            for layer_dict in atmosphere_dict["particle_layers"]:
                layer = self._create_particle_layer_from_dict(layer_dict)
                if layer:
                    particle_layers.append(layer)

        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=molecular_atmosphere, particle_layers=particle_layers
        )

        return atmosphere

    def create_simple_mono_atmosphere(self):
        """Create simple molecular atmosphere for mono mode debugging.

        Uses US Standard atmosphere with GECKO absorption database.
        Suitable for fast RGB sanity checks in mono mode.

        Returns:
            MolecularAtmosphere object
        """
        # Simple US Standard atmosphere
        altitude_min = 0.0
        altitude_max = 120000.0
        altitude_step = 1000.0
        num_steps = int((altitude_max - altitude_min) / altitude_step) + 1

        thermoprops = {
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(altitude_min, altitude_max, num_steps) * ureg.m,
        }

        atmosphere = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data="gecko",
            has_absorption=True,
            has_scattering=True,
        )

        return atmosphere
