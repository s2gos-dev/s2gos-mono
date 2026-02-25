"""Simulation utility for Pixel HDRF measurements using Sentinel-2 satellite sensors."""

from datetime import datetime

from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import (
    DirectionalIllumination,
    SimulationConfig,
    create_chime_sensor,
)
from s2gos_utils import PathLike, SceneDescription
from upath import UPath


def pixel_hdrf_simulation_config(
    scene_name: str,
    target_lat: float,
    target_lon: float,
    target_size: float,
    gmt_hour: float,
    spp: int = 8,
    config_output_dir: PathLike | None = None,
    film_resolution: tuple[int, int] = (1000, 1000),
    pixel_indices: list[tuple[int, int]] | None = None,
    pixel_samples_per_pixel: int = 256,
    srf_wmin: float = 380.0,
    srf_wmax: float = 1600.0,
    observation_date: datetime | None = None,
) -> PathLike | None:
    """Create simulation config with Sentinel-2A MSI bands 2,3,4 and PixelHDRF measurements.

    Args:
        scene_name: Name of the scene for output files
        target_lat: Target center latitude in WGS84
        target_lon: Target center longitude in WGS84
        target_size: Target area size in km (square area)
        gmt_hour: Hour of observation in GMT
        spp: Samples per pixel for satellite sensors (default: 8)
        config_output_dir: Optional output directory for config file
        film_resolution: Film resolution for satellite sensors (default: (1000, 1000))
        pixel_indices: List of (row, col) pixel indices to measure.
            If None, defaults to [(500, 500), (912, 687)].
        pixel_samples_per_pixel: Samples per pixel for pixel measurements (default: 256)
        srf_wmin: Minimum wavelength for SRF in nm (default: 380.0)
        srf_wmax: Maximum wavelength for SRF in nm (default: 1600.0)
        observation_date: Date/time for illumination calculation.
            If None, uses datetime(2024, 1, 1, gmt_hour, 0, 0).

    Returns:
        Path to the saved simulation config JSON file
    """
    print("\n")
    print("=" * 60)
    print("Configuring Pixel HDRF simulation...")

    # Apply defaults
    if pixel_indices is None:
        pixel_indices = [(500, 500), (912, 687), (415, 475), (400, 475)]

    if observation_date is None:
        observation_date = datetime(2024, 12, 21, int(gmt_hour), 0, 0)

    # Sentinel-2A MSI sensors for bands 2, 3, 4
    sensors = [
        # UAVSensor(
        #     id="uav_rgb_camera_hyp",
        #     instrument=UAVInstrumentType.PERSPECTIVE_CAMERA,
        #     viewing=LookAtViewing(
        #         origin=[0.019474, 8.30517, 24.0783 * 4.5],
        #         target=[0.019474, 8.30517, 0.0],
        #         up=[0, 1, 0],
        #         relative_to_asset="only_tower_v0_1.xml",
        #     ),
        #     srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
        #     fov=40,
        #     resolution=[2000, 2000],
        #     samples_per_pixel=4,
        # ),
        # UAVSensor(
        #     id="uav_rgb_camera_hyp",
        #     instrument=UAVInstrumentType.PERSPECTIVE_CAMERA,
        #     viewing=LookAtViewing(
        #         origin=[0, 0, 20000],
        #         target=[0, 0, 0.0],
        #         up=[0, 1, 0],
        #     ),
        #     srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
        #     fov=40,
        #     resolution=[2000, 2000],
        #     samples_per_pixel=4,
        # ),
        # SatelliteSensor(
        #     id="s2a_b2",
        #     platform=SatellitePlatform.SENTINEL_2A,
        #     instrument=SatelliteInstrument.MSI,
        #     band="2",
        #     target_center_lat=target_lat,
        #     target_center_lon=target_lon,
        #     target_size_km=target_size,
        #     film_resolution=film_resolution,
        #     viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        #     samples_per_pixel=spp,
        # ),
        # SatelliteSensor(
        #     id="s2a_b3",
        #     platform=SatellitePlatform.SENTINEL_2A,
        #     instrument=SatelliteInstrument.MSI,
        #     band="3",
        #     target_center_lat=target_lat,
        #     target_center_lon=target_lon,
        #     target_size_km=target_size,
        #     film_resolution=film_resolution,
        #     viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        #     samples_per_pixel=4,
        # ),
        # SatelliteSensor(
        #     id="s2a_b4",
        #     platform=SatellitePlatform.SENTINEL_2A,
        #     instrument=SatelliteInstrument.MSI,
        #     band="4",
        #     target_center_lat=target_lat,
        #     target_center_lon=target_lon,
        #     target_size_km=target_size,
        #     film_resolution=film_resolution,
        #     viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        #     samples_per_pixel=spp,
        # ),
        # CHIME hyperspectral sensor (30m resolution)
        # Film resolution scaled to match 30m vs S2's 10m (1/3 of S2 resolution)
        create_chime_sensor(
            target_center_lat=target_lat,
            target_center_lon=target_lon,
            target_size_km=target_size,
            zenith=3,
            samples_per_pixel=spp,
        ),
    ]

    # PixelHDRF and PixelBRF at specified pixels
    measurements = [
        # PixelHDRFConfig(
        #     id="center_pixels_hdrf",
        #     satellite_sensor_id="s2a_b3",
        #     pixel_indices=pixel_indices,
        #     height_offset_m=35,
        #     samples_per_pixel=pixel_samples_per_pixel,
        #     srf=SpectralResponse(type="uniform", wmin=srf_wmin, wmax=srf_wmax),
        # ),
        # PixelBHRConfig(
        #     id="center_pixels_bhr",
        #     satellite_sensor_id="s2a_b2",
        #     pixel_indices=pixel_indices,
        #     height_offset_m=0.5,
        #     samples_per_pixel=pixel_samples_per_pixel,
        #     srf=SpectralResponse(type="uniform", wmin=srf_wmin, wmax=srf_wmax),
        # ),
        # PixelBRFConfig(
        #     id="center_pixels_brf",
        #     satellite_sensor_id="s2a_b2",
        #     pixel_indices=pixel_indices,
        #     height_offset_m=10,
        #     samples_per_pixel=pixel_samples_per_pixel,
        #     srf=SpectralResponse(type="uniform", wmin=srf_wmin, wmax=srf_wmax),
        # ),
    ]

    simulation_config = SimulationConfig(
        name="pixel_hdrf_simulation",
        description="Pixel HDRF measurement using Sentinel-2A MSI bands 2, 3, 4",
        illumination=DirectionalIllumination.from_date_and_location(
            observation_date,
            target_lat,
            target_lon,
            "coddington_2022-1_nm",
        ),
        sensors=sensors,
        # measurements=measurements,
        backend_hints={"eradiate": {"mode": "ckd"}},
    )

    print("Simulation configured:")
    print(f"  Sensors: {len(simulation_config.sensors)}")
    for i, sensor in enumerate(simulation_config.sensors):
        platform = sensor.platform_type.value
        instrument = getattr(sensor, "instrument", "N/A")
        if hasattr(instrument, "value"):
            instrument = instrument.value
        print(f"    {i + 1}. {sensor.id} ({platform}/{instrument})")

    print(f"  Measurements: {len(simulation_config.measurements)}")
    for i, measurement in enumerate(simulation_config.measurements):
        print(f"    {i + 1}. {measurement.id} ({measurement.type})")

    # Save simulation configuration
    config_filename = f"{scene_name}_pixel_hdrf_sim_config.json"

    if config_output_dir is None:
        config_dir = UPath("./sim_config")
    else:
        config_dir = UPath(config_output_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_filename

    simulation_config.to_json(config_path)
    print(f"  Saved: {config_path}")
    return config_path


def simulation_from_config(
    scene_description_path: PathLike,
    config: SimulationConfig,
    simulation_output_dir: PathLike | None = None,
) -> PathLike | None:
    """Run simulation from a scene description and simulation config.

    Args:
        scene_description_path: Path to the scene description YAML file
        config: SimulationConfig object with sensors and measurements
        simulation_output_dir: Optional output directory for simulation results

    Returns:
        Path to the simulation output directory
    """
    print("\n")
    print("=" * 60)
    print("Simulating Pixel HDRF observation...")

    scene_description_path = UPath(scene_description_path)
    scene_description = SceneDescription.load_yaml(scene_description_path)

    # Set output directory
    if simulation_output_dir is None:
        simulation_output_dir = UPath(f"./sim_output/{scene_description.name}")

    # Run simulation if Eradiate is available
    if ERADIATE_AVAILABLE and scene_description:
        print("\nRunning Pixel HDRF simulation...")

        # Get available materials for validation
        if hasattr(scene_description, "materials"):
            available_materials = list(scene_description.materials.keys())
            objects_to_check = scene_description.objects
            print(f"Available materials: {available_materials}")

            # Check objects for invalid material references
            material_issues = []
            for i, obj in enumerate(objects_to_check):
                if "material" in obj:
                    mat_ref = obj["material"]
                    if mat_ref not in available_materials:
                        material_issues.append(
                            f"Object {i} ({obj.get('object_id', 'unnamed')}) "
                            f"references unknown material '{mat_ref}'"
                        )

            if material_issues:
                print("Material validation found issues:")
                for issue in material_issues:
                    print(f"  - {issue}")
            else:
                print(
                    f"All {len(objects_to_check)} object material references are valid"
                )

        # Run the simulation
        simulator = EradiateBackend(config)
        simulator.run_simulation(
            scene_description,
            scene_dir=scene_description_path.parent,
            output_dir=simulation_output_dir,
            # plot_image=True,
            # id_to_plot=["uav_rgb_camera_hyp"],
        )
        print("Pixel HDRF simulation completed successfully!")
    else:
        print("\nSimulation skipped")
        if not ERADIATE_AVAILABLE:
            print("  Eradiate not available")
        if not scene_description:
            print("  Scene generation failed")

    # Summary
    print("\n" + "=" * 60)
    print(f"Output directory: {simulation_output_dir}")

    return simulation_output_dir
