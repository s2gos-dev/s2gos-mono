"""Simulation utility for MTR."""

from datetime import datetime

from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import (
    AngularFromOriginViewing,
    AngularViewing,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
    SpectralResponse,
    create_chime_sensor,
)
from s2gos_utils import PathLike, SceneDescription
from upath import UPath


def simulation_config(
    scene_name: str,
    target_lat: float,
    target_lon: float,
    target_size: float,
    gmt_hour: float,
    spp: int = 8,
    season: str = "december",
    sensor_groups: list[str] | str | None = None,
    config_output_dir: PathLike | None = None,
    film_resolution: tuple[int, int] = (1000, 1000),
    pixel_indices: list[tuple[int, int]] | None = None,
) -> PathLike | None:
    """Create simulation config with seasonal date support and configurable sensors.

    Args:
        scene_name: Name of the scene for output files
        target_lat: Target center latitude in WGS84
        target_lon: Target center longitude in WGS84
        target_size: Target area size in km (square area)
        gmt_hour: Hour of observation in GMT
        spp: Samples per pixel for satellite sensors (default: 8)
        season: Season for simulation - 'december' (summer) or 'june' (winter).
                Determines default observation date if not specified.
        sensor_groups: Sensor groups to include. Options: "chime", "sentinel2", "hypstar".
                      Can be a single string or list of strings. Default: ["chime"].
        config_output_dir: Optional output directory for config file
        film_resolution: Film resolution for satellite sensors (default: (1000, 1000))
        pixel_indices: List of (row, col) pixel indices to measure.
            If None, defaults to [(500, 500), (912, 687), (415, 475), (400, 475)].

    Returns:
        Path to the saved simulation config JSON file
    """
    print("\n")
    print("=" * 60)
    print(f"Configuring Pixel HDRF simulation for season: {season}...")

    # Apply defaults
    if pixel_indices is None:
        pixel_indices = [(500, 500), (912, 687), (415, 475), (400, 475)]

    if sensor_groups is None:
        sensor_groups = ["chime"]

    if isinstance(sensor_groups, str):
        sensor_groups = [sensor_groups]

    if season == "june":
        observation_date = datetime(2024, 6, 21, int(gmt_hour), 0, 0)
    else:  # december
        observation_date = datetime(2024, 12, 21, int(gmt_hour), 0, 0)

    # Build sensors list based on requested groups
    sensors = []
    measurements = []

    # Build CHIME sensor
    if "chime" in sensor_groups:
        sensors.append(
            create_chime_sensor(
                target_center_lat=target_lat,
                target_center_lon=target_lon,
                target_size_km=target_size,
                zenith=3,
                samples_per_pixel=spp,
            )
        )

    # Build Sentinel-2 sensors
    if "sentinel2" in sensor_groups:
        sentinel_bands = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "8a",
            "10",
            "11",
            "12",
        ]
        for band in sentinel_bands:
            sensors.append(
                SatelliteSensor(
                    id=f"s2a_b{band}",
                    platform=SatellitePlatform.SENTINEL_2A,
                    instrument=SatelliteInstrument.MSI,
                    band=band,
                    target_center_lat=target_lat,
                    target_center_lon=target_lon,
                    target_size_km=target_size,
                    film_resolution=film_resolution,
                    viewing=AngularViewing(zenith=3.0, azimuth=0.0),
                    samples_per_pixel=spp,
                )
            )

    # Build HYPSTAR ground sensor
    if "hypstar" in sensor_groups:
        sensors.append(
            GroundSensor(
                id="hypstar",
                instrument=GroundInstrumentType.HYPSTAR,
                viewing=AngularFromOriginViewing(
                    origin=[0.019474, 8.30517, 32.987],
                    zenith=177,
                    azimuth=180,
                    up=[0, 1, 0],
                    relative_to_asset="only_tower_v0_1.xml",
                ),
                fov=5.0,
                resolution=[64 * 2, 64 * 2],
                samples_per_pixel=spp,
                srf=SpectralResponse(type="uniform", wmin=380, wmax=1680),
                hypstar_post_processing=HypstarPostProcessingConfig(
                    apply_circular_mask=True,
                    generate_rgb_image=True,
                    rgb_wavelengths=(660.0, 550.0, 440.0),
                    rgb_brightness_factor=1.8,
                    apply_srf=True,
                    fwhm_vnir_nm=3.0,
                    fwhm_swir_nm=10.0,
                    spatial_averaging=True,
                    real_reference_file="/home/gonzalezm/test/test2/s2gos-apps/hypstar_sim/HYPERNETS_L_GHNA_L2A_REF_20220517T0743_20230424T0625_v1.0.nc",
                    wavelength_variable="wavelength",
                ),
            )
        )
        measurements.extend(
            [
                IrradianceConfig(
                    id="hypstar_irradiance",
                    location=HemisphericalMeasurementLocation(
                        target_x=-220,
                        target_y=850,
                        target_z=4,  # it is irrelevant
                        height_offset_m=45,
                        terrain_relative_height=True,
                        srf=SpectralResponse(type="uniform", wmin=380, wmax=1680),
                        samples_per_pixel=spp,
                    ),
                    samples_per_pixel=spp,
                ),
                HDRFConfig(
                    id="hypstar_hcrf",
                    radiance_sensor_id="hypstar",
                    irradiance_measurement_id="hypstar_irradiance",
                ),
            ]
        )
    simulation_config = SimulationConfig(
        name="mtr_simulation",
        description="Simulation configuration for MTR demo",
        illumination=DirectionalIllumination.from_date_and_location(
            observation_date,
            target_lat,
            target_lon,
            "coddington_2022-1_nm",
        ),
        sensors=sensors,
        measurements=measurements,
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
