"""Atmosphere configuration builder for Eradiate backend."""

import logging
import math

import numpy as np
from s2gos_utils.io.paths import open_dataset, to_upath
from s2gos_utils.scene import SceneDescription

logger = logging.getLogger(__name__)

try:
    import joseki
    import pint
    import pinttrs
    from eradiate.attrs import define
    from eradiate.radprops import get_default_absdb
    from eradiate.scenes.atmosphere import (
        ExponentialParticleDistribution,
        GaussianParticleDistribution,
        HeterogeneousAtmosphere,
        HomogeneousAtmosphere,
        MolecularAtmosphere,
        ParticleLayer,
        UniformParticleDistribution,
    )
    from eradiate.units import unit_context_config as ucc
    from eradiate.units import unit_registry as ureg
    from joseki.profiles.core import extrapolate

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


if ERADIATE_AVAILABLE:

    @define(eq=False, slots=False, init=False)
    class _ParticleLayer(ParticleLayer):
        """Particle layer whose bottom may lie below sea level."""

        bottom: pint.Quantity = pinttrs.field(
            default=ureg.Quantity(0.0, ureg.km), units=ucc.deferred("length")
        )


def _extend_below(thermoprops, altitude):
    """Extend a profile down to altitude, linearly extrapolated by joseki.

    Levels are prepended in steps of the lowest level spacing, so regularly
    spaced profiles (as Eradiate requires) stay regular.
    """
    jureg = joseki.unit_registry
    z = jureg.Quantity(thermoprops.z.values, thermoprops.z.attrs["units"]).m_as("m")
    if altitude >= z[0]:
        return thermoprops
    dz = z[1] - z[0]
    z_extra = z[0] - dz * np.arange(math.ceil((z[0] - altitude) / dz), 0, -1)
    logger.info(
        f"Extending thermophysical profile below {z[0]:.0f} m to reach {altitude:.0f} m"
    )
    return extrapolate(thermoprops, z_extra=z_extra * jureg.m, direction="down")


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
            Geometry dictionary with TOA and ground altitudes

        Raises:
            ValueError: If the bottom of atmosphere is below Eradiate's
                plane-parallel atmosphere cuboid, which reaches down to -1 % of TOA
        """
        atmosphere = scene_description.atmosphere
        if not atmosphere:
            return {"type": "plane_parallel"}

        if atmosphere["boa"] <= -0.01 * atmosphere["toa"]:
            raise ValueError(
                f"Bottom of atmosphere ({atmosphere['boa']} m) must be above -1 % "
                f"of toa ({-0.01 * atmosphere['toa']} m); increase toa."
            )

        return {
            "type": "plane_parallel",
            "toa_altitude": atmosphere["toa"],
            "ground_altitude": atmosphere["boa"],
        }

    def create_atmosphere_from_config(self, scene_description: SceneDescription):
        """Create atmosphere based on scene description format.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Eradiate atmosphere object (MolecularAtmosphere, HomogeneousAtmosphere, or
            HeterogeneousAtmosphere), or None if the scene has no atmosphere

        Raises:
            ValueError: If atmosphere type is unknown or not specified
        """
        atmosphere = scene_description.atmosphere
        if not atmosphere:
            logger.info("No atmosphere configured, running without atmosphere")
            return None

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

    def _create_molecular_atmosphere_from_dict(self, mol_dict, boa, toa):
        """Create molecular atmosphere from dictionary.

        Supports either joseki identifiers or CAMS NetCDF files. Identifier
        profiles are evaluated from boa to toa every altitude_step; file profiles
        keep their own levels and are extended down to boa.

        Args:
            mol_dict: Dictionary with molecular atmosphere configuration
            boa: Bottom of atmosphere altitude in meters
            toa: Top of atmosphere altitude in meters

        Returns:
            MolecularAtmosphere object
        """
        if "thermoprops_file" in mol_dict:
            thermoprops_file = to_upath(mol_dict["thermoprops_file"])
            thermoprops = _extend_below(
                open_dataset(thermoprops_file).squeeze(drop=True), boa
            )
        else:
            thermoprops_id = mol_dict.get(
                "thermoprops_identifier", "afgl_1986-us_standard"
            )
            num_steps = int((toa - boa) / mol_dict["altitude_step"]) + 1

            thermoprops = joseki.interp(
                _extend_below(joseki.make(identifier=thermoprops_id), boa),
                z_new=np.linspace(boa, toa, num_steps) * ureg.m,
            )

        absorption_data = mol_dict.get("absorption_database") or get_default_absdb()

        atmosphere = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data=absorption_data,
            has_absorption=mol_dict.get("has_absorption", True),
            has_scattering=mol_dict.get("has_scattering", True),
        )

        return atmosphere

    def _create_particle_layer_from_dict(self, layer_dict):
        """Create particle layer from dictionary.

        Distribution parameters are given in meters (exponential rate in 1/m
        or scale in m, Gaussian center altitude and width in m) and converted
        to Eradiate's coordinate normalized over the layer thickness.

        Args:
            layer_dict: Dictionary with particle layer configuration

        Returns:
            ParticleLayer object
        """
        bottom = layer_dict["altitude_bottom"]
        top = layer_dict["altitude_top"]
        thickness = top - bottom
        dist_type = layer_dict["distribution_type"]

        if dist_type == "exponential":
            if "rate" in layer_dict:
                distribution = ExponentialParticleDistribution(
                    rate=layer_dict["rate"] * thickness
                )
            elif "scale" in layer_dict:
                distribution = ExponentialParticleDistribution(
                    rate=thickness / layer_dict["scale"]
                )
            else:
                distribution = ExponentialParticleDistribution()
        elif dist_type == "gaussian":
            distribution = GaussianParticleDistribution(
                mean=(layer_dict["center_altitude"] - bottom) / thickness,
                std=layer_dict["width"] / thickness,
            )
        else:
            distribution = UniformParticleDistribution()

        return _ParticleLayer(
            dataset=layer_dict["aerosol_dataset"],
            tau_ref=layer_dict["optical_thickness"],
            w_ref=layer_dict["reference_wavelength"],
            bottom=bottom,
            top=top,
            distribution=distribution,
            has_absorption=layer_dict["has_absorption"],
        )

    def _create_molecular_atmosphere_from_scene(self, atmosphere_dict):
        """Create molecular atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            MolecularAtmosphere object
        """
        return self._create_molecular_atmosphere_from_dict(
            atmosphere_dict["molecular_atmosphere"],
            atmosphere_dict["boa"],
            atmosphere_dict["toa"],
        )

    def _create_homogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create homogeneous atmosphere from scene description.

        A uniform medium spanning the geometry. A null scattering coefficient
        selects Eradiate's standard air scattering coefficient.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HomogeneousAtmosphere object
        """
        kwargs = {
            "sigma_a": atmosphere_dict["sigma_a"],
            "phase": {"type": atmosphere_dict["phase"]},
        }
        if atmosphere_dict["sigma_s"] is not None:
            kwargs["sigma_s"] = atmosphere_dict["sigma_s"]
        return HomogeneousAtmosphere(**kwargs)

    def _create_heterogeneous_atmosphere_from_scene(self, atmosphere_dict):
        """Create heterogeneous atmosphere from scene description.

        Args:
            atmosphere_dict: Atmosphere configuration dictionary

        Returns:
            HeterogeneousAtmosphere object
        """
        molecular_atmosphere = None
        if "molecular_atmosphere" in atmosphere_dict:
            molecular_atmosphere = self._create_molecular_atmosphere_from_dict(
                atmosphere_dict["molecular_atmosphere"],
                atmosphere_dict["boa"],
                atmosphere_dict["toa"],
            )

        particle_layers = [
            self._create_particle_layer_from_dict(layer_dict)
            for layer_dict in atmosphere_dict.get("particle_layers", [])
        ]

        atmosphere = HeterogeneousAtmosphere(
            molecular_atmosphere=molecular_atmosphere, particle_layers=particle_layers
        )

        return atmosphere
