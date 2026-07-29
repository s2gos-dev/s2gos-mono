"""Flux density (plane irradiance) measurement using the reference disk technique.

Measures the spectral flux density incident on a collector plane by:
1. Placing a small reference disk with Lambertian reflectance (ρ=1.0) at the
   target location, oriented along the collector normal
2. Viewing it with an hdistant measure covering the hemisphere about that normal
3. Converting the measured radiance to flux density: E = π × L_mean

With the default upward-facing normal this is bottom-of-atmosphere downwelling
irradiance; pointed downward it measures outgoing (reflected) flux.
"""

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

logger = logging.getLogger(__name__)

REFERENCE_DISK_RADIUS_M = 0.01  # 1cm radius for consistent BOA measurement


def insert_reference_disk(
    scene_description: SceneDescription,
    x: float,
    y: float,
    z: float,
    disk_id: str,
    radius: float = REFERENCE_DISK_RADIUS_M,
    normal: Optional[List[float]] = None,
) -> tuple:
    """Return a copy of scene_description with a white Lambertian disk inserted.

    The disk has ρ=1.0 (white Lambertian material implied by type="disk").
    Used for flux density (HDRF workflow) and BHR reference simulations.

    Args:
        normal: Optional collector normal. When omitted the disk keeps Mitsuba's
            default +z orientation, which is what BHR and upward-facing flux
            measurements expect.

    Returns:
        (modified scene, (x, y, z)) — coordinates passed through for convenience.
    """
    disk = {"object_id": disk_id, "type": "disk", "center": [x, y, z], "radius": radius}
    if normal is not None:
        disk["normal"] = list(normal)
    new_objects = (scene_description.objects or []).copy()
    new_objects.insert(0, disk)
    return replace(scene_description, objects=new_objects), (x, y, z)


class FluxProcessor:
    """Processor for flux density measurements using the reference disk technique."""

    def __init__(self, backend):
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def requires_flux(self) -> bool:
        """Check if any flux measurements are configured."""
        from .config import FluxConfig

        return any(
            isinstance(m, FluxConfig) for m in self.simulation_config.measurements
        )

    def _to_scene_coords(
        self, lat: float, lon: float, scene: SceneDescription
    ) -> Tuple[float, float]:
        """Convert lat/lon to scene XY coordinates."""
        from s2gos_utils.coordinates import CoordinateSystem

        coord_sys = CoordinateSystem(
            scene.location["center_lat"], scene.location["center_lon"]
        )
        return coord_sys.latlon_to_scene(lat, lon)

    def _to_lat_lon_coords(
        self, x: float, y: float, scene: SceneDescription
    ) -> Tuple[float, float]:
        """Convert scene XY coordinates to lat/lon."""
        from s2gos_utils.coordinates import CoordinateSystem

        coord_sys = CoordinateSystem(
            scene.location["center_lat"], scene.location["center_lon"]
        )
        return coord_sys.scene_to_latlon(x, y)

    def _get_disk_elevation(
        self,
        scene: SceneDescription,
        scene_dir: UPath,
        lat: float,
        lon: float,
        height_offset_m: float,
    ) -> float:
        """Get disk elevation: terrain + height_offset_m."""
        from s2gos_simulator.terrain_query import TerrainQuery

        terrain_elev = TerrainQuery(
            scene, scene_dir
        ).query_elevation_at_geographic_coords(lat, lon, raise_on_error=False)
        disk_elev = terrain_elev + height_offset_m

        logger.info(
            f"Disk at ({lat:.4f}, {lon:.4f}): "
            f"terrain={terrain_elev:.1f}m + offset={height_offset_m:.1f}m = {disk_elev:.1f}m"
        )
        return disk_elev

    def create_reference_disk_scene(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        location,
        height_offset_m: float,
        disk_id: str = "flux_reference_disk",
        normal: Optional[List[float]] = None,
    ) -> Tuple[SceneDescription, Tuple[float, float, float]]:
        """Create scene with white Lambertian disk (ρ=1.0) at target location.

        Args:
            normal: Optional collector normal; see :func:`insert_reference_disk`.

        Returns modified scene and disk coordinates (x, y, z).
        """
        from s2gos_simulator.terrain_query import TerrainQuery

        if height_offset_m < 0:
            raise ValueError(f"height_offset_m must be >= 0, got {height_offset_m}")

        target_lat, target_lon = location.target_lat, location.target_lon

        tq = TerrainQuery(scene_description, scene_dir)
        if not tq.validate_coordinate_bounds(
            target_lat, target_lon, max_distance_km=50.0
        ):
            logger.warning(
                f"Coordinates ({target_lat}, {target_lon}) far from scene center"
            )

        # Get disk position
        if location.target_x is None:
            x, y = self._to_scene_coords(target_lat, target_lon, scene_description)
            z = self._get_disk_elevation(
                scene_description, scene_dir, target_lat, target_lon, height_offset_m
            )
        else:
            x, y = location.target_x, location.target_y
            target_lat, target_lon = self._to_lat_lon_coords(x, y, scene_description)

            if location.terrain_relative_height:
                z = self._get_disk_elevation(
                    scene_description,
                    scene_dir,
                    target_lat,
                    target_lon,
                    location.height_offset_m,
                )
            else:
                z = location.target_z

        logger.info(
            f"Created reference disk at ({x:.1f}, {y:.1f}, {z:.1f}), normal={normal}"
        )
        return insert_reference_disk(
            scene_description, x, y, z, disk_id=disk_id, normal=normal
        )

    def radiance_to_flux_density(
        self,
        radiance: xr.DataArray,
    ) -> xr.DataArray:
        """Convert disk radiance to flux density: E = π × L_mean.

        Averages over hemisphere sampling dimensions (from hdistant measure),
        preserves wavelength dimension.
        """
        hemisphere_dims = [d for d in radiance.dims if d != "w"]
        logger.debug(f"Radiance dims: {radiance.dims}, shape: {radiance.shape}")
        L_mean = radiance.mean(dim=hemisphere_dims) if hemisphere_dims else radiance
        E = np.pi * L_mean  # E = π × L for Lambertian ρ=1.0

        logger.info(f"Flux density: mean={float(E.mean()):.3e} W/m²/nm")

        E.attrs.update(
            {
                "quantity": "flux_density",
                "standard_name": "spectral_flux_density",
                "units": "W m^-2 nm^-1",
                "conversion": (
                    "E = π × mean(L) over the hemisphere about the sensor normal "
                    "(L is from a perfect white disk)"
                ),
            }
        )

        return E

    def execute_flux_measurements(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
    ) -> tuple[Dict[str, xr.Dataset], Dict[str, tuple[float, float, float]]]:
        """Execute flux density measurements.

        Returns:
            Tuple of (results, disk_coords) — disk_coords maps measurement ID
            to (x, y, z) scene coordinates of the reference disk.
        """
        import eradiate
        from s2gos_utils.io.paths import mkdir

        from .backends.eradiate.geometry_utils import sanitize_sensor_id

        logger.info("=" * 60)
        logger.info("Flux Density Measurements")
        logger.info("=" * 60)
        mkdir(output_dir)

        results = {}
        disk_coords: Dict[str, tuple[float, float, float]] = {}
        # Filter for FluxConfig instances from unified measurements list
        from .config import FluxConfig

        flux_configs = [
            m for m in self.simulation_config.measurements if isinstance(m, FluxConfig)
        ]

        disk_scenes = {}
        for config in flux_configs:
            logger.info(f"\n[{config.id}] Creating reference disk...")
            disk_scene, coords = self.create_reference_disk_scene(
                scene_description,
                scene_dir,
                config.location,
                config.location.height_offset_m,
                disk_id=f"disk_{config.id}",
                normal=config.normal,
            )
            disk_coords[config.id] = coords
            disk_scenes[config.id] = disk_scene

        for config in flux_configs:
            disk_scene = disk_scenes[config.id]

            # Only this config's disk is present in this scene, so only its
            # measure may be created — otherwise the others would target points
            # in empty air.
            experiment = self.backend.create_experiment(
                disk_scene,
                scene_dir,
                disk_coords_map={config.id: disk_coords[config.id]},
            )

            measure_map = {
                getattr(m, "id", f"measure_{i}"): i
                for i, m in enumerate(experiment.measures)
            }

            # Measure IDs are sanitized during translation (dots → underscores).
            measure_id = sanitize_sensor_id(config.id)
            if measure_id not in measure_map:
                raise RuntimeError(
                    f"Measure '{measure_id}' not found. Available: {list(measure_map.keys())}"
                )

            measure_idx = measure_map[measure_id]
            eradiate.run(experiment, measures=measure_idx)

            result = experiment.results[measure_id]
            if "radiance" not in result:
                raise RuntimeError(
                    f"No 'radiance' in results. Available: {list(result.data_vars)}"
                )

            E = self.radiance_to_flux_density(result["radiance"])

            dataset_vars = {"flux_density": E}
            if "irradiance" in result:
                E_toa = result["irradiance"]
                toa_dims = [d for d in E_toa.dims if d != "w"]
                if toa_dims:
                    E_toa = E_toa.mean(dim=toa_dims)
                E_toa.attrs.update(
                    {"quantity": "toa_irradiance", "units": "W m^-2 nm^-1"}
                )
                dataset_vars["toa_irradiance"] = E_toa

            result_ds = xr.Dataset(dataset_vars)

            # hdistant's viewing_angles ignore both `direction` and `orientation`,
            # so they describe the default +z frame and are wrong for any tilted
            # collector. Drop them rather than ship misleading coordinates.
            result_ds = result_ds.drop_vars(
                [c for c in ("vza", "vaa") if c in result_ds.coords]
            )

            x, y, z = disk_coords[config.id]
            result_ds.attrs.update(
                {
                    "normal_zenith_deg": config.normal_zenith,
                    "normal_azimuth_deg": config.normal_azimuth,
                    "normal_vector": list(config.normal),
                    "azimuth_convention": "east_referenced (0=East, 90=North)",
                    "disk_x": x,
                    "disk_y": y,
                    "disk_z": z,
                    "disk_radius_m": REFERENCE_DISK_RADIUS_M,
                }
            )

            output_file = output_dir / f"{config.id}.zarr"
            from s2gos_utils.io.paths import expand_mapper

            result_ds.to_zarr(expand_mapper(output_file), mode="w")
            logger.info(f"  ✓ Saved {output_file.name}")

            results[config.id] = result_ds

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Complete: {len(results)} measurements")
        logger.info(f"{'=' * 60}\n")
        return results, disk_coords
