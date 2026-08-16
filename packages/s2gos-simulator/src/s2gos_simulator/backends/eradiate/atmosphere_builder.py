"""Atmosphere configuration builder for Eradiate backend."""

import logging

import numpy as np
from s2gos_utils.io.paths import open_dataset, to_upath
from s2gos_utils.scene import SceneDescription

logger = logging.getLogger(__name__)

try:
    from eradiate.radprops import absdb_factory, get_default_absdb
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

def _resolve_ground_altitude(
    scene_description: SceneDescription,
    scene_dir,
    toa_altitude: float,
) -> float:
    """Adjust the atmosphere ground altitude.

    Adjusts the plane-parallel medium volume floor so below-sea-level terrain
    stays inside it. Above-sea-level scenes clamp to 0.0 and are not affected.

    We have a 10% pad in case buffer has a lower altitude than the DEM:
        z_min = scene DEM minimum elevation
        z0    = z_min - 0.10 * |z_min|      (10% depth pad)
        z0    = min(0.0, z0)                no changes if the min z is above sea level
    at -0.01 * toa_altitude (see PlaneParallelGeometry.atmosphere_shape).

    Args:
        scene_description: Scene description.
        scene_dir: Scene directory containing data/dem_*.zarr.
        toa_altitude: Top-of-atmosphere altitude in meters.

    Returns:
        Ground altitude in meters (<= 0.0).
    """
    from s2gos_simulator.terrain_query import TerrainQuery

    z_min = TerrainQuery(scene_description, scene_dir).min_elevation()
    if z_min is None:
        logger.warning(
            "Could not determine scene minimum elevation; "
            "defaulting ground_altitude to 0.0 m."
        )
        return 0.0

    z0 = z_min - 0.10 * abs(z_min)
    z0 = min(0.0, z0)

    floor_ = -0.01 * toa_altitude
    if z0 <= floor_:
        raise ValueError(
            f"Resolved ground_altitude ({z0:.1f} m) is at or below the "
            f"plane-parallel atmosphere cuboid floor ({floor_:.1f} m = "
            f"-0.01 * toa_altitude, see PlaneParallelGeometry.atmosphere_shape). "
            f"Below-sea terrain would leave the medium shape. "
            f"Raise toa_altitude (currently {toa_altitude:.0f} m) so that "
            f"-0.01 * toa < {z0:.1f} m."
        )

    logger.info(
        f"Resolved ground_altitude = {z0:.1f} m "
        f"(scene z_min = {z_min:.1f} m, toa = {toa_altitude:.0f} m)."
    )
    return z0

def _build_thermoprops(identifier, altitude_max, altitude_step, ground_altitude):
    """Build a thermophysical profile, extended below MSL when needed.

    Above sea level (ground_altitude None or >= 0) this returns the legacy
    ``{"identifier", "z"}`` dict, identical to the previous behaviour.

    Below sea level (ground_altitude < 0) the joseki ``{identifier, z}`` dict
    path cannot be used: it forwards to ``joseki.make`` which rejects negative
    altitudes. Instead we build the profile as an ``xr.Dataset`` — make the
    native profile, linearly extrapolate p/t/n/x_* down to ground_altitude,
    then resample onto a grid pinned at ground_altitude and reaching >= toa.
    ``MolecularAtmosphere`` accepts the resulting dataset directly.

    The interp grid is constructed so its bottom is exactly ground_altitude
    (required: PlaneParallelGeometry raises if zgrid bottom != ground_altitude)
    and its top is >= altitude_max (required: check_geometry_atmosphere raises
    if the profile does not cover the zgrid top).

    Args:
        identifier: joseki atmosphere identifier (e.g. 'afgl_1986-us_standard').
        altitude_max: profile/geometry top in meters.
        altitude_step: vertical step in meters.
        ground_altitude: resolved ground altitude in meters (<= 0), or None.

    Returns:
        dict for the legacy path, or an xr.Dataset for the extended path.
    """
    if ground_altitude is None or ground_altitude >= 0.0:
        num_steps = int((altitude_max - 0.0) / altitude_step) + 1
        return {
            "identifier": identifier,
            "z": np.linspace(0.0, altitude_max, num_steps) * ureg.m,
        }

    # Below-MSL: build and extend as an xr.Dataset.
    import joseki
    from joseki.units import ureg as jureg
    from joseki.profiles.core import extrapolate, interp

    ds = joseki.make(identifier)

    # Extra levels strictly below 0, down to ground_altitude.
    z_extra = np.arange(ground_altitude, 0.0, altitude_step) * jureg.m
    ds = extrapolate(ds, z_extra=z_extra, direction="down")

    # Native profile top (m), after downward extrapolation (joseki z is in km).
    z_units = ds["z"].attrs.get("units", "m")
    native_top_m = float(ds["z"].values.max()) * (1000.0 if z_units == "km" else 1.0)
    grid_top = min(altitude_max, native_top_m)

    # Regularly-spaced grid (eradiate's ZGrid requires equal steps), pinned
    # exactly at both ends: bottom = ground_altitude, top = grid_top. Using
    # linspace guarantees regular spacing and lands on native data (no NaN).
    # Number of intervals ~ target step, at least 1.
    n_intervals = max(1, int(round((grid_top - ground_altitude) / altitude_step)))
    z_new = np.linspace(ground_altitude, grid_top, n_intervals + 1)
    ds = interp(ds, z_new=z_new * jureg.m)

    logger.info(
        f"Extended thermoprops below MSL: floor={ground_altitude:.1f} m, "
        f"top={z_new[-1]:.1f} m ({len(z_new)} levels, "
        f"step~{(z_new[1]-z_new[0]):.1f} m)."
    )
    return ds

def _clamp_pressure_to_absdb(thermoprops, absorption_data):
    """Cap profile pressure at the absorption database's tabulated ceiling.

    Below MSL the deepest layers can have a pressure above the absorption
    database's maximum tabulated pressure. Eradiate's default out-of-bounds
    rule then returns sigma_a = 0 (absorption) for those layers, this translates into
    zero absorption in the highest pressure. This caps only the pressure ``p`` given to 
    the absorption lookup; ``n`` and ``t`` are untouched. Rayleigh scattering and the 
    lookup's temperature axis stay exact.

    Only xr.Dataset thermoprops are considered (the below-MSL path). The legacy
    dict path (above MSL) never exceeds the ceiling and is returned unchanged.
    Within a Dataset, only layers actually above the ceiling are modified, so
    every other scene stays identical.

    Args:
        thermoprops: dict (above-MSL, passed through) or xr.Dataset (below-MSL).
        absorption_data: the resolved absorption database object.

    Returns:
        The thermoprops, with ``p`` capped if and only if it exceeded the
        ceiling; otherwise the original object unchanged.
    """
    # Legacy dict path (above MSL): nothing to clamp.
    if not hasattr(thermoprops, "data_vars") or "p" not in thermoprops:
        return thermoprops

    # Read the ceiling from the db metadata; bail out safely if not as expected.
    try:
        stop = absorption_data.metadata["pressure_grid"]["parameters"]["stop"]
        if stop.get("units") not in ("pascal", "Pa", None):
            logger.warning(
                f"Absorption DB pressure ceiling has unexpected units "
                f"{stop.get('units')!r}; skipping pressure clamp."
            )
            return thermoprops
        ceiling = float(stop["value"])
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning(
            f"Could not read absorption DB pressure ceiling ({exc!r}); "
            f"skipping pressure clamp."
        )
        return thermoprops

    p = thermoprops["p"]
    p_max = float(p.values.max())
    if p_max <= ceiling:
        return thermoprops  # no layer over the ceiling -> bit-identical

    n_over = int((p.values > ceiling).sum())
    clamped = thermoprops.copy()
    clamped["p"] = p.clip(max=ceiling)
    clamped["p"].attrs = dict(p.attrs)  # preserve units metadata
    logger.info(
        f"Capped absorption-lookup pressure at DB ceiling {ceiling:.0f} Pa: "
        f"{n_over} layer(s) exceeded it (deepest was {p_max:.0f} Pa, "
        f"{100 * (p_max / ceiling - 1):.2f}% over). n and t unchanged."
    )
    return clamped


class AtmosphereBuilder:
    """Builder for creating Eradiate atmosphere configurations from scene descriptions."""

    def __init__(self):
        """Initialize atmosphere builder."""
        pass

    def create_geometry_from_atmosphere(self, scene_description: SceneDescription, scene_dir):
        """Create geometry with bounds matching the atmosphere configuration.

        Args:
            scene_description: Scene description containing atmosphere config
            scene_dir: Scene directory (for DEM-based ground altitude resolution)

        Returns:
            Geometry dictionary with TOA and ground altitudes
        """
        atmosphere = scene_description.atmosphere
        toa = atmosphere["toa"]
        ground_altitude = _resolve_ground_altitude(scene_description, scene_dir, toa)

        geometry = {
            "type": "plane_parallel",
            "toa_altitude": toa,
            "ground_altitude": ground_altitude,
        }
        
        return geometry

    def create_atmosphere_from_config(self, scene_description: SceneDescription, ground_altitude=None):
        """Create atmosphere based on scene description format.

        Args:
            scene_description: Scene description containing atmosphere config

        Returns:
            Eradiate atmosphere object (MolecularAtmosphere, HomogeneousAtmosphere, or HeterogeneousAtmosphere)

        Raises:
            ValueError: If atmosphere type is unknown or not specified
        """
        self._ground_altitude = ground_altitude

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
            thermoprops_file = to_upath(mol_dict["thermoprops_file"])
            thermoprops = open_dataset(thermoprops_file).squeeze(drop=True)
        else:
            thermoprops_id = mol_dict.get(
                "thermoprops_identifier", "afgl_1986-us_standard"
            )
            altitude_max = mol_dict["altitude_max"]
            altitude_step = mol_dict["altitude_step"]
            ground_altitude = getattr(self, "_ground_altitude", None)

            thermoprops = _build_thermoprops(
                identifier=thermoprops_id,
                altitude_max=altitude_max,
                altitude_step=altitude_step,
                ground_altitude=ground_altitude,
            )
        
        
        absorption_data = mol_dict.get("absorption_database") or get_default_absdb()
        if isinstance(absorption_data, str):
            absorption_data = absdb_factory.create(absorption_data)
        thermoprops = _clamp_pressure_to_absdb(thermoprops, absorption_data)

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

    def create_simple_mono_atmosphere(self, ground_altitude=None):
        """Create simple molecular atmosphere for mono mode debugging.

        Uses US Standard atmosphere with GECKO absorption database.
        Suitable for fast RGB sanity checks in mono mode.

        Returns:
            MolecularAtmosphere object
        """
        # US Standard atmosphere, extended below MSL when ground_altitude < 0.
        thermoprops = _build_thermoprops(
            identifier="afgl_1986-us_standard",
            altitude_max=120000.0,
            altitude_step=1000.0,
            ground_altitude=ground_altitude,
        )
        absorption_data = absdb_factory.create("gecko")
        thermoprops = _clamp_pressure_to_absdb(thermoprops, absorption_data)
        
        atmosphere = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data="gecko",
            has_absorption=True,
            has_scattering=True,
        )

        return atmosphere
