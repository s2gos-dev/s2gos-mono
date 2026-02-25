"""Eradiate backend for S2GOS simulator.

This is the main orchestration layer that coordinates the specialized modules
for geometry, atmosphere, surfaces, sensors, and result processing.
"""

import logging
from typing import Dict, List, Optional

import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .atmosphere_builder import AtmosphereBuilder
from .constants import VALID_ERADIATE_MODES
from .eradiate_translator import EradiateTranslator
from .geometry_utils import GeometryUtils
from .result_processor import ResultProcessor
from .surface_builder import SurfaceBuilder
from ..base import SimulationBackend
from ...bhr_processor import BHRProcessor
from ...brf_processor import BRFProcessor
from ...config import (
    BHRConfig,
    BRFConfig,
    DistantViewing,
    GroundInstrumentType,
    HCRFConfig,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    PixelBHRConfig,
    PixelBRFConfig,
    PixelHDRFConfig,
    PlatformType,
    RectangleTarget,
    SimulationConfig,
)
from ...hdrf_processor import HDRFProcessor
from ...irradiance_processor import IrradianceProcessor
from ...processors.sensor_processor import SensorProcessor

logger = logging.getLogger(__name__)

try:
    import eradiate
    from eradiate.experiments import AtmosphereExperiment

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class EradiateBackend(SimulationBackend):
    """Enhanced Eradiate backend with modular architecture.

    This backend orchestrates specialized modules for different aspects of
    the simulation: geometry, atmosphere, surfaces, sensors, and results.
    """

    def __init__(self, simulation_config: SimulationConfig):
        """Initialize the Eradiate backend with modular components.

        Args:
            simulation_config: Simulation configuration
        """
        super().__init__(simulation_config)

        eradiate_hints = simulation_config.backend_hints.get("eradiate", {})
        self._eradiate_mode = eradiate_hints.get("mode", "mono")

        if ERADIATE_AVAILABLE:
            try:
                eradiate.set_mode(self._eradiate_mode)
            except Exception as e:
                raise ValueError(
                    f"Failed to set Eradiate mode '{self._eradiate_mode}': {e}. "
                    f"Check if this mode is supported in your Eradiate installation."
                )

        self.geometry_utils = GeometryUtils()
        self.atmosphere_builder = AtmosphereBuilder()
        self.surface_builder = SurfaceBuilder()
        self.eradiate_translator = EradiateTranslator(
            simulation_config, self.geometry_utils
        )
        self.result_processor = ResultProcessor(simulation_config)
        self.sensor_processor = SensorProcessor(simulation_config)

        self._current_scene_dir = None
        self._current_scene_description = None

    def is_available(self) -> bool:
        """Check if Eradiate dependencies are available.

        Returns:
            True if Eradiate is available, False otherwise
        """
        return ERADIATE_AVAILABLE

    @property
    def supported_platforms(self) -> List[str]:
        """Get list of supported platform types.

        Returns:
            List of platform type strings
        """
        return ["satellite", "uav", "ground"]

    @property
    def supported_measurements(self) -> List[str]:
        """Get list of supported measurement types.

        Returns:
            List of measurement type strings
        """
        return [
            "hdrf",
            "hcrf",
            "brf",
            "bhr",
            "boa_irradiance",
            "pixel_hdrf",
            "pixel_brf",
            "pixel_bhr",
        ]

    def validate_configuration(self) -> List[str]:
        """Validate configuration for Eradiate backend.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate_configuration()

        if not self.is_available():
            errors.append(
                "Eradiate is not available. Install with: pip install eradiate[kernel]"
            )

        if self._eradiate_mode not in VALID_ERADIATE_MODES:
            errors.append(
                f"Invalid Eradiate mode '{self._eradiate_mode}'. "
                f"Valid modes: {sorted(VALID_ERADIATE_MODES)}"
            )

        for sensor in self.simulation_config.sensors:
            if sensor.platform_type == PlatformType.SATELLITE:
                validation_error = self._validate_satellite_sensor(sensor)
                if validation_error:
                    errors.append(validation_error)
            elif sensor.platform_type == PlatformType.GROUND:
                if sensor.instrument not in [
                    GroundInstrumentType.HYPSTAR,
                    GroundInstrumentType.PERSPECTIVE_CAMERA,
                    GroundInstrumentType.PYRANOMETER,
                    GroundInstrumentType.FLUX_METER,
                    GroundInstrumentType.DHP_CAMERA,
                    GroundInstrumentType.RADIANCEMETER,
                ]:
                    errors.append(
                        f"Ground sensor {sensor.id} instrument type {sensor.instrument} is not supported"
                    )

        return errors

    def _validate_satellite_sensor(self, sensor) -> Optional[str]:
        """Validate satellite sensor platform/instrument/band combination.

        Args:
            sensor: Satellite sensor configuration

        Returns:
            Error message if invalid, None if valid
        """
        from ...config import (
            INSTRUMENT_BANDS,
            PLATFORM_INSTRUMENTS,
            SatelliteInstrument,
            SatellitePlatform,
        )

        if sensor.platform == SatellitePlatform.CUSTOM:
            if sensor.srf is None:
                return "Custom platform requires explicit SRF configuration"
            return None

        try:
            platform_enum = SatellitePlatform(sensor.platform)
            instrument_enum = SatelliteInstrument(sensor.instrument)
        except ValueError as e:
            return f"Invalid platform or instrument: {e}"

        valid_instruments = PLATFORM_INSTRUMENTS.get(platform_enum, [])
        if instrument_enum not in valid_instruments:
            return (
                f"Platform '{sensor.platform}' does not support instrument '{sensor.instrument}'. "
                f"Valid instruments: {[inst.value for inst in valid_instruments]}"
            )

        band_enum_class = INSTRUMENT_BANDS.get(instrument_enum)
        if band_enum_class is not None:
            try:
                band_enum_class(sensor.band)
            except ValueError:
                valid_bands = [band.value for band in band_enum_class]
                return (
                    f"Instrument '{sensor.platform}/{sensor.instrument}' does not support band '{sensor.band}'. "
                    f"Valid bands: {valid_bands}"
                )

        return None

    def run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: Optional[UPath] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run Eradiate simulation from scene description.

        Automatically detects measurement types and executes appropriate workflow:
        - BRF: No atmosphere
        - HDRF/HCRF: With atmosphere, uses BOA irradiance from white disk
        - Standard: Regular radiance measurements

        Args:
            scene_description: Scene description from s2gos_generator
            scene_dir: Directory containing scene assets
            output_dir: Output directory (defaults to scene_dir/eradiate_renders)
            **kwargs: Additional options (plot_image, id_to_plot, etc.)

        Returns:
            xarray Dataset containing all simulation results
        """
        if not self.is_available():
            raise RuntimeError("Eradiate is not available")

        if output_dir is None:
            output_dir = scene_dir / "eradiate_renders"

        output_dir = UPath(output_dir)
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        self._expand_pixel_measurement_configs(scene_description, scene_dir)
        self._auto_generate_sensors_for_measurements()

        self._log_simulation_summary()

        # Initialize processors
        brf_proc = BRFProcessor(self)
        bhr_proc = BHRProcessor(self)
        hdrf_proc = HDRFProcessor(self)
        irr_proc = IrradianceProcessor(self)

        # Separate BRF and BHR measurements from others (they have special workflows)
        brf_configs = brf_proc.get_brf_configs()
        bhr_configs = bhr_proc.get_bhr_configs()
        non_brf_measurements = [
            m
            for m in self.simulation_config.measurements
            if not isinstance(m, BRFConfig) and not isinstance(m, BHRConfig)
        ]

        # Extract BRF sensor IDs so we can exclude them from standard/HDRF workflow
        brf_sensor_ids = brf_proc.get_brf_sensor_ids()

        non_brf_sensor_ids = (
            {s.id for s in self.simulation_config.sensors if s.id not in brf_sensor_ids}
            if brf_sensor_ids
            else None
        )  # None means "all sensors"

        requires_hdrf = hdrf_proc.requires_hdrf()
        requires_irr = irr_proc.requires_irradiance()

        all_results = {}

        # Run BRF workflow if we have BRF measurements (NO atmosphere)
        if brf_configs:
            brf_results = self._run_brf_workflow(
                scene_description,
                scene_dir,
                output_dir,
                brf_proc,
                **kwargs,
            )
            all_results.update(brf_results)

        # Run BHR workflow if we have BHR measurements
        if bhr_configs:
            bhr_results = bhr_proc.execute_bhr_measurements(
                scene_description,
                scene_dir,
                output_dir / "derived",
                radiosity_dir=output_dir / "radiosity",
            )
            all_results.update(bhr_results)

        if non_brf_measurements or self.simulation_config.sensors:
            if requires_hdrf or requires_irr:
                hdrf_results = self._run_combined_workflow(
                    scene_description,
                    scene_dir,
                    output_dir,
                    irr_proc,
                    hdrf_proc,
                    sensor_ids=non_brf_sensor_ids,
                    **kwargs,
                )
                all_results.update(hdrf_results)
            else:
                logger.info("\nStandard workflow")
                standard_results = self._run_standard_simulation(
                    scene_description,
                    scene_dir,
                    output_dir,
                    sensor_ids=non_brf_sensor_ids,
                    **kwargs,
                )
                all_results.update(standard_results)

        logger.info(f"\n✓ Simulation complete: {len(all_results)} total datasets")
        return all_results

    def _run_standard_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        include_irradiance_measures: bool = True,
        sensor_ids: Optional[set] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run radiance simulation with atmosphere (used for plain sensor runs).

        This is NOT used for BRF (no atmosphere), that is handled by `_run_brf_workflow`.
        NOT used for BOA irradiance measurements, that is handled by `IrradianceProcessor`
        Atmosphere is built from scene description via `create_experiment(atmosphere="auto")`.

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory
            include_irradiance_measures: Whether to include irradiance measurements
            sensor_ids: Optional set of sensor IDs to include. If None, all sensors
                are included.
            **kwargs: Additional options

        Returns:
            Simulation results dataset
        """
        logger.info("=" * 60)
        logger.info("Running experiments")
        logger.info("=" * 60)
        experiment = self.create_experiment(
            scene_description,
            scene_dir,
            include_irradiance_measures,
            sensor_ids=sensor_ids,
        )

        all_saved_results = {}

        for i in range(len(experiment.measures)):
            measure_id = getattr(experiment.measures[i], "id", f"measure_{i}")
            logger.info(f"  Measure {i + 1}/{len(experiment.measures)}: {measure_id}")
            eradiate.run(experiment, measures=i)

            raw_dataset = experiment.results[measure_id]
            raw_dataset.attrs["output_dir"] = str(output_dir)

            sensor_config = self.eradiate_translator.get_sensor_by_id(measure_id)
            if sensor_config:
                post_processed_dataset = self.sensor_processor.process_sensor_result(
                    raw_dataset, sensor_config
                )

                if post_processed_dataset is not raw_dataset:
                    self.result_processor.save_result(
                        f"{measure_id}_raw_eradiate", raw_dataset, output_dir
                    )

                success = self.result_processor.save_result(
                    measure_id, post_processed_dataset, output_dir
                )
                if success:
                    all_saved_results[measure_id] = post_processed_dataset
            else:
                success = self.result_processor.save_result(
                    measure_id, raw_dataset, output_dir
                )
                if success:
                    all_saved_results[measure_id] = raw_dataset

        logger.debug(f"Successfully saved results: {list(all_saved_results.keys())}")

        return all_saved_results

    def _run_combined_workflow(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        irradiance_processor: IrradianceProcessor,
        hdrf_processor: HDRFProcessor,
        sensor_ids: Optional[set] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Execute combined workflow for BOA Irradiance, HDRF/HCRF measurements.

        Runs two simulations with atmosphere:
        1. Radiance simulation for all sensors
        2. BOA irradiance simulation using white reference disk(s)
        Then derives HDRF/HCRF from the ratio (see HDRFProcessor).

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory
            irradiance_processor: IrradianceProcessor instance
            sensor_ids: Optional set of sensor IDs to include. If None, all sensors
                are included. Use this to exclude BRF sensors when running alongside
                BRF workflow.
            **kwargs: Additional options

        Returns:
            Combined results dataset
        """
        logger.info("=" * 60)
        logger.info("Running radiance simulation Irradiance Measurements")
        logger.info("=" * 60)
        sensor_results = self._run_standard_simulation(
            scene_description,
            scene_dir,
            output_dir / "radiance",
            include_irradiance_measures=False,
            sensor_ids=sensor_ids,
            **kwargs,
        )

        irr_results, _ = irradiance_processor.execute_irradiance_measurements(
            scene_description, scene_dir, output_dir / "boa_irradiance"
        )

        combined_results = {**sensor_results, **irr_results}

        derived_output_dir = output_dir / "derived"

        derived_results = hdrf_processor.compute_h_reflectances(
            combined_results, derived_output_dir
        )

        for derived_name, dataset in derived_results.items():
            dataset.attrs["id"] = derived_name
            success = self.result_processor.save_result(
                derived_name, dataset, derived_output_dir, result_type="derived"
            )
            if not success:
                logger.warning(f"Failed to save derived measure '{derived_name}'")

        postprocessed_results = {**combined_results, **derived_results}
        logger.debug(f"Combined workflow results: {list(postprocessed_results.keys())}")
        logger.info(
            f"\n✓ Combined workflow complete: {len(postprocessed_results)} total datasets"
        )
        return postprocessed_results

    def _run_brf_workflow(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
        brf_proc: BRFProcessor,
        **kwargs,
    ) -> dict:
        """Execute BRF workflow without atmosphere.

        Only runs BRF radiance sensors - NOT satellite or HDRF sensors.
        Simpler than HDRF - just runs radiance measurements and computes BRF
        using TOA irradiance from results (no white disk needed).

        Uses incremental saving: runs one measure at a time, processes it,
        and saves immediately before moving to the next.

        Args:
            scene_description: Scene description
            scene_dir: Scene directory path
            output_dir: Output directory
            **kwargs: Additional options

        Returns:
            Dictionary of BRF results
        """
        logger.info("=" * 60)
        logger.info("Running BRF (no atmosphere)")
        logger.info("=" * 60)

        brf_sensor_ids = brf_proc.get_brf_sensor_ids()

        experiment = self.create_experiment(
            scene_description,
            scene_dir,
            include_irradiance_measures=False,
            atmosphere=None,  # NO ATMOSPHERE for BRF
            sensor_ids=brf_sensor_ids,
        )

        all_saved_results = {}
        radiance_output_dir = output_dir / "radiance"
        brf_output_dir = output_dir / "derived"

        for i, measure in enumerate(experiment.measures):
            measure_id = getattr(measure, "id", f"measure_{i}")
            logger.info(f"  Measure {i + 1}/{len(experiment.measures)}: {measure_id}")
            eradiate.run(experiment, measures=i)

            # Get raw result
            raw_dataset = experiment.results[measure_id]
            raw_dataset.attrs["output_dir"] = str(output_dir)

            # Post-process with sensor processor (apply Gaussian SRF if configured)
            sensor_config = self.eradiate_translator.get_sensor_by_id(measure_id)
            if sensor_config:
                post_processed_dataset = self.sensor_processor.process_sensor_result(
                    raw_dataset, sensor_config
                )

                if post_processed_dataset is not raw_dataset:
                    self.result_processor.save_result(
                        f"{measure_id}_raw_eradiate", raw_dataset, radiance_output_dir
                    )

                success = self.result_processor.save_result(
                    measure_id, post_processed_dataset, radiance_output_dir
                )
                if success:
                    all_saved_results[measure_id] = post_processed_dataset
            else:
                success = self.result_processor.save_result(
                    measure_id, raw_dataset, radiance_output_dir
                )
                if success:
                    all_saved_results[measure_id] = raw_dataset

        # After all radiance measures are collected, compute BRF measurements
        logger.info("\nComputing BRF measurements from radiance...")
        brf_results = brf_proc.compute_all_brf_measurements(
            all_saved_results, output_dir=brf_output_dir
        )

        all_results = {**all_saved_results, **brf_results}
        logger.info(f"\nBRF workflow complete: {len(brf_results)} BRF measurements")
        return all_results

    def create_experiment(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
        atmosphere: Optional[str] = "auto",
        sensor_ids: Optional[set] = None,
        measures: Optional[list] = None,
        irradiance_disk_coords: Optional[Dict[str, tuple]] = None,
    ):
        """Create Eradiate experiment from scene description.

        Args:
            scene_description: Scene description with all scene data
            scene_dir: Base directory for resolving file paths
            include_irradiance_measures: Whether to include irradiance measurements
            atmosphere: Atmosphere option:
                - "auto": Create atmosphere from scene config (default)
                - None: No atmosphere (for BRF measurements)
            sensor_ids: Optional set of sensor IDs to include. If None, all sensors
                are included. Use this to filter sensors for specific workflows
                (e.g., only BRF sensors for BRF workflow).
            measures: Pre-built Eradiate measure dicts. When provided,
                translate_sensors() is bypassed and these measures are used directly.
                include_irradiance_measures and sensor_ids are ignored.
                Use this when the caller owns measure construction (e.g. BHR distant_flux).
            irradiance_disk_coords: Optional mapping of IrradianceConfig ID to
                (x, y, z) disk coordinates. Passed through to translate_sensors()
                when measures is None.

        Returns:
            AtmosphereExperiment configured for the scene
        """
        self._current_scene_dir = scene_dir
        self._current_scene_description = scene_description

        self._validate_object_materials(scene_description)
        kdict, kpmap = self.surface_builder.translate_materials(
            scene_description, scene_dir
        )

        hamster_data_dict = self._get_hamster_data_for_scene(
            scene_description, scene_dir
        )
        self.surface_builder.add_hamster_materials(
            kdict, kpmap, scene_description, hamster_data_dict
        )

        hamster_available = hamster_data_dict is not None
        kdict.update(
            self.surface_builder.create_target_surface(
                scene_description, scene_dir, hamster_available
            )
        )

        if scene_description.buffer:
            kdict.update(
                self.surface_builder.create_buffer_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        if scene_description.background:
            kdict.update(
                self.surface_builder.create_background_surface(
                    scene_description, scene_dir, hamster_available
                )
            )

        self.surface_builder.add_scene_objects(kdict, scene_description, scene_dir)

        # Create atmosphere (or use None for BRF measurements)
        if atmosphere == "auto":
            # TODO: This needs to be more fleshed out and tested
            # Use simple atmosphere for mono mode (debugging), otherwise use scene atmosphere
            if self._eradiate_mode == "mono":
                logger.info(
                    "  Using simple mono atmosphere (GECKO + US Standard) for fast debugging"
                )
                atmosphere_obj = self.atmosphere_builder.create_simple_mono_atmosphere()
            else:
                atmosphere_obj = self.atmosphere_builder.create_atmosphere_from_config(
                    scene_description
                )
        else:
            atmosphere_obj = atmosphere  # None for BRF

        illumination = self.eradiate_translator.translate_illumination()

        if measures is None:
            measures = self.eradiate_translator.translate_sensors(
                scene_description,
                scene_dir,
                include_irradiance_measures,
                sensor_ids,
                irradiance_disk_coords=irradiance_disk_coords,
            )

        logger.debug(
            f"Experiment measures: {[m.get('id', i) for i, m in enumerate(measures)]}"
        )

        # For mono mode with simple atmosphere, use fixed TOA
        if self._eradiate_mode == "mono":
            geometry = {
                "type": "plane_parallel",
                "toa_altitude": 120000.0,
            }
        else:
            geometry = self.atmosphere_builder.create_geometry_from_atmosphere(
                scene_description
            )

        self.surface_builder.validate_material_ids(kdict, scene_description)

        return AtmosphereExperiment(
            geometry=geometry,
            atmosphere=atmosphere_obj,
            surface=None,  # We set this up through the expert interface (kdict, kpmap)
            illumination=illumination,
            measures=measures,
            kdict=kdict,
            kpmap=kpmap,
        )

    def _log_simulation_summary(self) -> None:
        """Log a clear simulation summary with measurement type breakdown."""
        from collections import Counter

        # Count measurement types
        meas_types = Counter(
            type(m).__name__ for m in self.simulation_config.measurements
        )
        meas_summary = ", ".join(
            f"{count} {name}" for name, count in meas_types.items()
        )

        logger.info(f"Simulation: {self.simulation_config.name}")
        logger.info(f"  Sensors: {len(self.simulation_config.sensors)}")
        logger.info(f"  Measurements: {meas_summary if meas_summary else 'none'}")

    def _validate_object_materials(self, scene_description: SceneDescription) -> None:
        """Validate that all object materials use string references and exist.

        Args:
            scene_description: Scene description with objects

        Raises:
            ValueError: If any object has invalid material reference
        """
        if not scene_description.objects:
            return

        for i, obj in enumerate(scene_description.objects):
            if "material" not in obj:
                continue

            material = obj["material"]

            if not isinstance(material, str):
                raise ValueError(
                    f"Object {i} has invalid material type {type(material).__name__}. "
                    "Only string references are supported. "
                    "Define materials in the scene's material library."
                )

            if material not in scene_description.materials:
                available = list(scene_description.materials.keys())
                raise ValueError(
                    f"Object {i} references unknown material '{material}'. "
                    f"Available materials: {available}"
                )

    def _expand_pixel_measurement_configs(
        self,
        scene_description: Optional[SceneDescription] = None,
        scene_dir: Optional[UPath] = None,
    ) -> None:
        """Expand PixelHDRFConfig and PixelBRFConfig into individual measurements.

        For each pixel in a pixel measurement config, creates the corresponding
        expanded config (HDRFConfig or BRFConfig) at the scene coordinates.

        Args:
            scene_description: Scene description for terrain lookup (optional)
            scene_dir: Scene directory for terrain data (optional)
        """
        from s2gos_utils.coordinates import CoordinateSystem, pixel_to_scene_xy

        pixel_configs = [
            m
            for m in self.simulation_config.measurements
            if isinstance(m, (PixelHDRFConfig, PixelBRFConfig, PixelBHRConfig))
        ]

        if not pixel_configs:
            return

        for pixel_config in pixel_configs:
            sensor = self.simulation_config.get_sensor(pixel_config.satellite_sensor_id)
            if sensor is None:
                config_type = type(pixel_config).__name__
                raise ValueError(
                    f"{config_type} '{pixel_config.id}' references unknown sensor "
                    f"'{pixel_config.satellite_sensor_id}'"
                )

            # Compute scene bounds from satellite sensor geometry
            coord_sys = CoordinateSystem(
                sensor.target_center_lat, sensor.target_center_lon
            )
            if isinstance(sensor.target_size_km, (int, float)):
                width_km = height_km = float(sensor.target_size_km)
            else:
                width_km, height_km = sensor.target_size_km

            bounds = coord_sys.create_rectangle(
                sensor.target_center_lat, sensor.target_center_lon, width_km, height_km
            )

            srf = pixel_config.srf if pixel_config.srf is not None else sensor.srf

            # Get pixel size for BRF rectangle targets
            pixel_size_x, pixel_size_y = sensor.pixel_size_m

            # Expand each pixel to an individual measurement config
            for row, col in pixel_config.pixel_indices:
                x, y = pixel_to_scene_xy(row, col, bounds, sensor.film_resolution)

                # Query terrain elevation if scene_description is provided
                target_z = pixel_config.height_offset_m
                if scene_description is not None and scene_dir is not None:
                    try:
                        terrain_z = self.geometry_utils.query_terrain_elevation(
                            scene_description, scene_dir, x, y
                        )
                        target_z = terrain_z + pixel_config.height_offset_m
                    except Exception as e:
                        logger.warning(
                            f"Could not query terrain at ({x:.1f}, {y:.1f}): {e}. "
                            f"Using height_offset_m={pixel_config.height_offset_m} directly."
                        )

                expanded = self._create_expanded_measurement(
                    pixel_config,
                    row,
                    col,
                    x,
                    y,
                    srf,
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    target_z=target_z,
                )
                self.simulation_config.measurements.append(expanded)
                logger.debug(
                    f"Expanded pixel ({row}, {col}) -> {type(expanded).__name__} "
                    f"'{expanded.id}' at scene ({x:.1f}, {y:.1f}, z={target_z:.1f})"
                )

            self.simulation_config.measurements.remove(pixel_config)

    def _create_expanded_measurement(
        self,
        pixel_config,
        row: int,
        col: int,
        x: float,
        y: float,
        srf,
        pixel_size_x: float = 10.0,
        pixel_size_y: float = 10.0,
        target_z: float = 0.0,
    ):
        """Create appropriate measurement config from pixel coordinates.

        Args:
            pixel_config: PixelHDRFConfig, PixelBRFConfig, or PixelBHRConfig instance
            row: Pixel row index
            col: Pixel column index
            x: Scene x coordinate (pixel center)
            y: Scene y coordinate (pixel center)
            srf: Spectral response function
            pixel_size_x: Pixel width in meters (for BRF rectangle targets)
            pixel_size_y: Pixel height in meters (for BRF rectangle targets)
            target_z: Target z-coordinate (terrain + height_offset for BRF)

        Returns:
            HDRFConfig, BRFConfig, or BHRConfig instance
        """
        if isinstance(pixel_config, PixelHDRFConfig):
            # Create rectangle target centered on pixel at specified altitude
            rectangle_target = RectangleTarget.from_center_and_size(
                cx=x,
                cy=y,
                width=pixel_size_x,
                height=pixel_size_y,
                z=target_z,
            )

            return HDRFConfig(
                id=f"{pixel_config.id}_r{row}_c{col}",
                instrument="radiancemeter",
                viewing=DistantViewing(
                    target=rectangle_target,
                    direction=[0, 0, 1],
                    ray_offset=5.0,
                    terrain_relative_height=False,  # z already computed
                ),
                location=HemisphericalMeasurementLocation(
                    target_x=x,
                    target_y=y,
                    target_z=0,
                    height_offset_m=pixel_config.height_offset_m,
                    terrain_relative_height=True,
                ),
                reference_height_offset_m=pixel_config.height_offset_m,
                srf=srf,
                samples_per_pixel=pixel_config.samples_per_pixel,
                terrain_relative_height=False,
            )
        elif isinstance(pixel_config, PixelBHRConfig):
            rectangle_target = RectangleTarget.from_center_and_size(
                cx=x,
                cy=y,
                width=pixel_size_x,
                height=pixel_size_y,
                z=target_z,
            )
            return BHRConfig(
                id=f"{pixel_config.id}_r{row}_c{col}",
                target_x=x,
                target_y=y,
                target_z=0,
                height_offset_m=pixel_config.height_offset_m,
                terrain_relative_height=True,
                srf=srf,
                samples_per_pixel=pixel_config.samples_per_pixel,
                reference_height_offset_m=pixel_config.reference_height_offset_m,
                viewing=DistantViewing(
                    target=rectangle_target,
                    direction=[0, 0, 1],
                    ray_offset=5.0,
                    terrain_relative_height=False,  # z already computed
                ),
            )
        else:  # PixelBRFConfig
            rectangle_target = RectangleTarget.from_center_and_size(
                cx=x,
                cy=y,
                width=pixel_size_x,
                height=pixel_size_y,
                z=target_z,
            )

            return BRFConfig(
                id=f"{pixel_config.id}_r{row}_c{col}",
                viewing=DistantViewing(
                    target=rectangle_target,
                    direction=[0, 0, 1],
                    ray_offset=5.0,
                    terrain_relative_height=False,  # z is already computed
                ),
                srf=srf,
                samples_per_pixel=pixel_config.samples_per_pixel,
                terrain_relative_height=False,
            )

    def _auto_generate_sensors_for_measurements(self) -> List[str]:
        """Auto-generate sensors for measurements that require them.

        Returns:
            List of generated sensor IDs
        """
        generated_sensor_ids = []

        for measurement in self.simulation_config.measurements:
            if (
                isinstance(measurement, HDRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                radiance_sensor = (
                    self.eradiate_translator.generate_radiance_sensor_for_hdrf(
                        measurement
                    )
                )
                irradiance_measurement = (
                    self.eradiate_translator.generate_irradiance_measurement_for_hdrf(
                        measurement
                    )
                )

                self.simulation_config.sensors.append(radiance_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = radiance_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(radiance_sensor.id)
                logging.info(
                    f"Auto-generated radiance sensor '{radiance_sensor.id}' and "
                    f"irradiance measurement '{irradiance_measurement.id}' for HDRF '{measurement.id}'"
                )

            elif (
                isinstance(measurement, HCRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                camera_sensor = (
                    self.eradiate_translator.generate_camera_sensor_for_hcrf(
                        measurement
                    )
                )
                irradiance_measurement = (
                    self.eradiate_translator.generate_irradiance_measurement_for_hcrf(
                        measurement
                    )
                )

                self.simulation_config.sensors.append(camera_sensor)
                self.simulation_config.measurements.append(irradiance_measurement)

                measurement.radiance_sensor_id = camera_sensor.id
                measurement.irradiance_measurement_id = irradiance_measurement.id

                generated_sensor_ids.append(camera_sensor.id)
                logging.info(
                    f"Auto-generated camera sensor '{camera_sensor.id}' and "
                    f"irradiance measurement '{irradiance_measurement.id}' for HCRF '{measurement.id}'"
                )

            elif (
                isinstance(measurement, BRFConfig)
                and measurement.radiance_sensor_id is None
            ):
                # BRF only needs radiance sensor (no irradiance - uses TOA from results)
                radiance_sensor = (
                    self.eradiate_translator.generate_radiance_sensor_for_brf(
                        measurement
                    )
                )

                self.simulation_config.sensors.append(radiance_sensor)
                measurement.radiance_sensor_id = radiance_sensor.id

                generated_sensor_ids.append(radiance_sensor.id)
                logging.info(
                    f"Auto-generated radiance sensor '{radiance_sensor.id}' for BRF '{measurement.id}'"
                )

        return generated_sensor_ids

    def _get_material_ids_from_scene(
        self, scene_description: SceneDescription
    ) -> List[str]:
        """Get material IDs from SceneDescription metadata.

        Args:
            scene_description: SceneDescription object

        Returns:
            List of material IDs in texture index order
        """
        material_indices = scene_description.material_indices

        material_ids = []
        for texture_index in sorted(material_indices.keys(), key=int):
            material_name = material_indices[texture_index]
            material_ids.append(material_name)

        return material_ids

    def _get_hamster_data_for_scene(
        self, scene_description: SceneDescription, scene_dir: UPath
    ) -> Optional[dict]:
        """Load HAMSTER albedo data from zarr files.

        Args:
            scene_description: Scene description with HAMSTER data paths
            scene_dir: Directory containing scene files

        Returns:
            Dict with HAMSTER albedo DataArrays for each surface, or None
        """
        try:
            hamster_data_files = {}

            if scene_description.target and "hamster_data" in scene_description.target:
                hamster_data_files["target"] = scene_description.target["hamster_data"]

            if scene_description.buffer and "hamster_data" in scene_description.buffer:
                hamster_data_files["buffer"] = scene_description.buffer["hamster_data"]

            if (
                scene_description.background
                and "hamster_data" in scene_description.background
            ):
                hamster_data_files["background"] = scene_description.background[
                    "hamster_data"
                ]

            if not hamster_data_files:
                logging.info("No HAMSTER data files found in scene description")
                return None

            hamster_data = {}
            base_path = scene_dir

            import xarray as xr
            from s2gos_utils.io.paths import exists

            for area, relative_path in hamster_data_files.items():
                file_path = base_path / relative_path
                if not exists(file_path):
                    logging.warning(
                        f"HAMSTER data file not found: {file_path}, skipping {area}"
                    )
                    continue

                try:
                    dataset = xr.open_zarr(file_path)
                    data_vars = list(dataset.data_vars.keys())
                    if not data_vars:
                        logging.warning(
                            f"No data variables in HAMSTER file: {file_path}"
                        )
                        continue

                    albedo_data = dataset[data_vars[0]]
                    hamster_data[area] = albedo_data
                    logging.info(
                        f"Loaded HAMSTER data for {area} from {file_path}: {albedo_data.sizes}"
                    )

                except Exception as e:
                    logging.warning(
                        f"Failed to load HAMSTER data from {file_path}: {e}"
                    )
                    continue

            if hamster_data:
                logging.info(
                    f"Successfully loaded HAMSTER data for {len(hamster_data)} surface areas"
                )
                return hamster_data
            else:
                logging.warning("No HAMSTER data could be loaded")
                return None

        except Exception as e:
            logging.warning(f"Could not load HAMSTER data: {e}")
            return None
