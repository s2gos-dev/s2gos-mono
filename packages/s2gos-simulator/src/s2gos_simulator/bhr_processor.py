"""BHR (Bi-Hemispherical Reflectance) Processor.

BHR is computed as the ratio of surface radiosity to white reference radiosity:
    BHR = radiosity_surface / radiosity_white_reference

This uses Eradiate's distant_flux measure type which outputs radiosity
(power per unit area), not radiance. The workflow requires two simulations:
1. Surface simulation: measures radiosity reflected from the actual surface
2. Reference simulation: measures radiosity from a white Lambertian disk (ρ=1.0)
"""

import logging
from typing import Dict, List, Tuple

import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .config import BHRConfig
from .irradiance_processor import insert_reference_disk

logger = logging.getLogger(__name__)


class BHRProcessor:
    """Processor for BHR measurements using distant_flux measure type.

    BHR = radiosity_surface / radiosity_white_reference

    This processor handles:
    1. Detection of BHR measurements in config
    2. Creation of white reference scenes with Lambertian disk
    3. Execution of paired surface/reference simulations
    4. BHR computation from radiosity ratios
    """

    def __init__(self, backend):
        """Initialize BHR processor.

        Args:
            backend: EradiateBackend instance
        """
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def get_bhr_configs(self) -> List[BHRConfig]:
        """Get all BHR measurement configurations.

        Returns:
            List of BHRConfig instances
        """
        return [
            m for m in self.simulation_config.measurements if isinstance(m, BHRConfig)
        ]

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

        logger.debug(
            f"Disk at ({lat:.4f}, {lon:.4f}): "
            f"terrain={terrain_elev:.1f}m + offset={height_offset_m:.1f}m = {disk_elev:.1f}m"
        )
        return disk_elev

    def _get_target_coords(
        self,
        bhr_config: BHRConfig,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Tuple[float, float, float]:
        """Get target coordinates for BHR measurement.

        Args:
            bhr_config: BHR configuration
            scene_description: Scene description
            scene_dir: Scene directory path

        Returns:
            Tuple of (x, y, z) coordinates in scene coordinate system
        """
        if bhr_config.target_x is not None:
            # Cartesian coordinates provided
            x, y = bhr_config.target_x, bhr_config.target_y

            if bhr_config.terrain_relative_height:
                target_lat, target_lon = self._to_lat_lon_coords(
                    x, y, scene_description
                )
                z = self._get_disk_elevation(
                    scene_description,
                    scene_dir,
                    target_lat,
                    target_lon,
                    bhr_config.height_offset_m,
                )
            else:
                z = bhr_config.target_z
        else:
            # Geographic coordinates provided
            x, y = self._to_scene_coords(
                bhr_config.target_lat, bhr_config.target_lon, scene_description
            )
            z = self._get_disk_elevation(
                scene_description,
                scene_dir,
                bhr_config.target_lat,
                bhr_config.target_lon,
                bhr_config.height_offset_m,
            )

        return (x, y, z)

    def create_white_reference_scene(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        bhr_config: BHRConfig,
    ) -> Tuple[SceneDescription, Tuple[float, float, float]]:
        """Create scene with white Lambertian disk (ρ=1.0) at target location.

        The disk is placed at the target location with an additional height offset
        specified by reference_height_offset_m to avoid z-fighting with the surface.

        Args:
            scene_description: Original scene description
            scene_dir: Scene directory path
            bhr_config: BHR configuration

        Returns:
            Tuple of (modified scene description, disk coordinates (x, y, z))
        """
        # Get base target coordinates
        x, y, z = self._get_target_coords(bhr_config, scene_description, scene_dir)

        # Add reference height offset for the white disk
        disk_z = z + bhr_config.reference_height_offset_m

        disk_id = f"bhr_white_reference_{bhr_config.id}"

        logger.info(
            f"Created BHR white reference disk '{disk_id}' at "
            f"({x:.2f}, {y:.2f}, {disk_z:.2f}) m"
        )
        return insert_reference_disk(scene_description, x, y, disk_z, disk_id=disk_id)

    def compute_bhr(
        self,
        surface_radiosity: xr.DataArray,
        reference_radiosity: xr.DataArray,
        bhr_config: BHRConfig,
    ) -> xr.Dataset:
        """Compute BHR from surface and reference radiosity values.

        BHR = radiosity_surface / radiosity_white_reference

        Args:
            surface_radiosity: Radiosity from surface simulation
            reference_radiosity: Radiosity from white reference simulation
            bhr_config: BHR configuration

        Returns:
            Dataset containing BHR values
        """
        from .backends.eradiate.reflectance_computation import compute_bhr

        bhr_ds = compute_bhr(
            surface_radiosity=surface_radiosity,
            reference_radiosity=reference_radiosity,
            measurement_id=bhr_config.id,
            extra_attrs={
                "reference_height_offset_m": bhr_config.reference_height_offset_m,
            },
        )

        return bhr_ds

    def execute_bhr_measurements(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        radiosity_dir: UPath | None = None,
    ) -> Dict[str, xr.Dataset]:
        """Execute all BHR measurements.

        For each BHR config:
        1. Run surface simulation with distant_flux measure
        2. Create white reference scene and run reference simulation
        3. Compute BHR from radiosity ratio

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory for results

        Returns:
            Dictionary of BHR results keyed by measurement ID
        """
        import eradiate
        from s2gos_utils.io.paths import mkdir

        logger.info("=" * 60)
        logger.info("BHR (Bi-Hemispherical Reflectance) Measurements")
        logger.info("=" * 60)

        mkdir(output_dir)
        if radiosity_dir is not None:
            mkdir(radiosity_dir)
        results = {}
        bhr_configs = self.get_bhr_configs()

        if not bhr_configs:
            logger.debug("No BHR measurements configured")
            return results

        for bhr_config in bhr_configs:
            logger.info(f"\n[{bhr_config.id}] BHR Measurement")

            try:
                # Get target coordinates for this measurement
                target_coords = self._get_target_coords(
                    bhr_config, scene_description, scene_dir
                )
                logger.info(
                    f"  Target: ({target_coords[0]:.2f}, {target_coords[1]:.2f}, "
                    f"{target_coords[2]:.2f}) m"
                )

                # Step 1: Surface simulation
                logger.info("  Step 1: Running surface radiosity simulation...")
                surface_measure = (
                    self.backend.eradiate_translator.create_distant_flux_measure(
                        bhr_config, target_coords, is_reference=False
                    )
                )
                surface_experiment = self.backend.create_experiment(
                    scene_description, scene_dir, measures=[surface_measure]
                )

                surface_measure_id = f"bhr_surface_{bhr_config.id}".replace(".", "_")
                measure_idx = self._get_measure_index(
                    surface_experiment, surface_measure_id
                )
                eradiate.run(surface_experiment, measures=measure_idx)

                surface_result = surface_experiment.results[surface_measure_id]
                surface_radiosity = self._extract_radiosity(surface_result)
                logger.info(
                    f"  Surface radiosity: mean={float(surface_radiosity.mean()):.4e}"
                )

                if radiosity_dir is not None:
                    surface_file = (
                        radiosity_dir
                        / f"{self.simulation_config.name}_{bhr_config.id}_surface.zarr"
                    )
                    surface_result.to_zarr(surface_file, mode="w")
                    logger.info(f"  Saved surface radiosity → {surface_file.name}")

                # Step 2: White reference simulation
                logger.info("  Step 2: Running white reference radiosity simulation...")
                ref_scene, disk_coords = self.create_white_reference_scene(
                    scene_description, scene_dir, bhr_config
                )

                ref_measure = (
                    self.backend.eradiate_translator.create_distant_flux_measure(
                        bhr_config, disk_coords, is_reference=True
                    )
                )
                ref_experiment = self.backend.create_experiment(
                    ref_scene, scene_dir, measures=[ref_measure]
                )

                ref_measure_id = f"bhr_reference_{bhr_config.id}".replace(".", "_")
                measure_idx = self._get_measure_index(ref_experiment, ref_measure_id)
                eradiate.run(ref_experiment, measures=measure_idx)

                ref_result = ref_experiment.results[ref_measure_id]
                reference_radiosity = self._extract_radiosity(ref_result)
                logger.info(
                    f"  Reference radiosity: mean={float(reference_radiosity.mean()):.4e}"
                )

                if radiosity_dir is not None:
                    reference_file = (
                        radiosity_dir
                        / f"{self.simulation_config.name}_{bhr_config.id}_reference.zarr"
                    )
                    ref_result.to_zarr(reference_file, mode="w")
                    logger.info(f"  Saved reference radiosity → {reference_file.name}")

                # Step 3: Compute BHR
                logger.info("  Step 3: Computing BHR...")
                bhr_ds = self.compute_bhr(
                    surface_radiosity, reference_radiosity, bhr_config
                )

                # Add coordinates metadata
                bhr_ds.attrs["target_x"] = target_coords[0]
                bhr_ds.attrs["target_y"] = target_coords[1]
                bhr_ds.attrs["target_z"] = target_coords[2]

                # Save result
                output_file = (
                    output_dir
                    / f"{self.simulation_config.name}_{bhr_config.id}_bhr.zarr"
                )
                bhr_ds.attrs["id"] = bhr_config.id
                bhr_ds.to_zarr(output_file, mode="w")

                results[bhr_config.id] = bhr_ds

                bhr_mean = float(bhr_ds["bhr"].mean())
                logger.info(f"  BHR computed: mean={bhr_mean:.4f}")
                logger.info(f"  Saved to {output_file.name}")

            except Exception as e:
                logger.error(f"Failed to compute BHR for '{bhr_config.id}': {e}")
                raise

        logger.info(f"\n{'=' * 60}")
        logger.info(f"BHR complete: {len(results)} measurements")
        logger.info(f"{'=' * 60}\n")

        return results

    def _get_measure_index(self, experiment, measure_id: str) -> int:
        """Get the index of a measure by ID.

        Args:
            experiment: Eradiate experiment
            measure_id: Measure ID to find

        Returns:
            Index of the measure

        Raises:
            RuntimeError: If measure not found
        """
        for i, measure in enumerate(experiment.measures):
            if getattr(measure, "id", None) == measure_id:
                return i
        available = [
            getattr(m, "id", f"measure_{i}") for i, m in enumerate(experiment.measures)
        ]
        raise RuntimeError(f"Measure '{measure_id}' not found. Available: {available}")

    def _extract_radiosity(self, result: xr.Dataset) -> xr.DataArray:
        """Extract radiosity from distant_flux measure result.

        The distant_flux measure outputs 'radiosity' or 'radiant_flux' depending
        on Eradiate version.

        Args:
            result: Eradiate result dataset

        Returns:
            Radiosity DataArray

        Raises:
            RuntimeError: If no radiosity variable found
        """
        # Try common variable names for distant_flux output
        for var_name in ["radiosity", "radiant_flux", "flux"]:
            if var_name in result:
                radiosity = result[var_name]
                # Average over any non-wavelength dimensions
                non_w_dims = [d for d in radiosity.dims if d != "w"]
                if non_w_dims:
                    radiosity = radiosity.mean(dim=non_w_dims)
                return radiosity

        raise RuntimeError(
            f"No radiosity variable found in result. "
            f"Available variables: {list(result.data_vars)}"
        )
