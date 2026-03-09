#!/usr/bin/env python3
"""PNP GUI Demo – Patagonia National Park generation and simulation demo for the GUI.

This module provides two processors for the PNP GUI demo:
1. pnp_gui/generation - Creates scene with seasonal variations
2. pnp_gui/simulation - Runs simulation with configurable observation types

The demo showcases:
- Seasonal variations (December=summer, June=winter in Patagonia)
- Multiple observation types (CHIME, MSI, HYPSTAR, RGB camera, satellite HDRF)
- Fixed demonstration dates (21st of month), flexible time
"""

import enum
from datetime import datetime
from typing import Annotated

from pydantic import Field
from s2gos_utils.coordinates import CoordinateSystem

from s2gos_apps.registry import registry

# ============================================================================
# Constants
# ============================================================================

PNP_LAT = -46.9097
PNP_LON = -72.4500
PNP_SIZE_KM = 10.0
TOWER_COORDS = (-220.0, 850.0)  # meters in scene coordinates
coord_system = CoordinateSystem(PNP_LAT, PNP_LON)

# ============================================================================
# Enums
# ============================================================================


class Month(enum.StrEnum):
    """Month selection for PNP GUI demo (limited to solstice dates)."""

    DECEMBER = "december"  # Summer in Patagonia
    JUNE = "june"  # Winter in Patagonia


class ObservationType(enum.StrEnum):
    """Available observation types for PNP GUI demo."""

    CHIME = "chime"
    MSI = "msi"
    SATELLITE_PIXEL_HDRF = "satellite_pixel_hdrf"
    HYPSTAR = "hypstar"
    RGB_CAMERA = "rgb_camera"


# ============================================================================
# Observation Configuration Models
# ============================================================================
# Note: All observations now use fixed configurations (no parameters needed)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_seasonal_config(month: Month) -> dict:
    """Return seasonal configuration (materials, assets, dates, snow settings)."""
    if month == Month.JUNE:
        # Winter configuration
        return {
            "material_config": "materials_winter.json",
            "tree_assets": [
                "tls_tree_38_winter.xml",
                "tls_tree_71_winter.xml",
                "tls_tree_165_winter.xml",
                "tls_tree_228_winter.xml",
                "tls_tree_290_winter.xml",
                "tls_tree_300_winter.xml",
                "tls_tree_336_winter.xml",
            ],
            "shrub_asset": "tls_tree_336_winter.xml",
            "thermoprops_date": "2020-06-21",
            "apply_snow": True,
            "observation_month": 6,
            "observation_day": 21,
        }
    else:  # DECEMBER
        # Summer configuration
        return {
            "material_config": "materials.json",
            "tree_assets": [
                "tls_tree_38_prospect.xml",
                "tls_tree_71_prospect.xml",
                "tls_tree_165_prospect.xml",
                "tls_tree_228_prospect.xml",
                "tls_tree_290_prospect.xml",
                "tls_tree_300_prospect.xml",
                "tls_tree_336_prospect.xml",
            ],
            "shrub_asset": "tls_tree_336_prospect.xml",
            "thermoprops_date": "2020-12-21",
            "apply_snow": False,
            "observation_month": 12,
            "observation_day": 21,
        }


# ============================================================================
# Processor 1: Scene Generation
# ============================================================================


@registry.process(id="pnp_gui/generation", title="Scene Generation Demo")
def pnp_gui_generation(
    month: Annotated[
        Month,
        Field(
            default=Month.DECEMBER,
            description="Month for simulation (December=summer, June=winter in Patagonia)",
            title="Month",
        ),
    ],
    random_seed: Annotated[
        int,
        Field(
            default=13,
            description="RNG seed for vegetation placement",
            title="Vegetation RNG seed",
        ),
    ],
    scene_name: Annotated[
        str,
        Field(
            ...,
            description="Name of scene",
            title="Scene name",
        ),
    ] = None,
) -> str | None:
    """Generate 3D scene for PNP GUI demo with seasonal variations.

    This processor:
    1. Creates a scene generation configuration based on season/month
    2. Immediately runs the generation pipeline
    3. Returns path to the generated scene description YAML

    The scene includes:
    - PNP location with 10km target area
    - Seasonal vegetation (summer/winter variants)
    - Heterogeneous atmosphere with aerosol layer
    - Tower XML scene at fixed location
    - Optional snow cover (June only)

    Args:
        month: Month for simulation (controls seasonal variations)
        random_seed: Random seed for reproducible vegetation placement
        scene_name: Name of scene

    Returns:
        Path to generated scene description YAML file, or None if validation fails
    """
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        AerosolDataset,
        BackgroundConfig,
        BufferConfig,
        ExponentialDistribution,
        MolecularAtmosphereConfig,
        ParticleLayerConfig,
        SnowConfig,
        ThermophysicalConfig,
        VegetationPlacementConfig,
        VegetationSpecies,
        XmlSceneConfig,
        create_scene_config,
    )
    from upath import UPath

    from s2gos_apps.gen_util import generation_from_config

    print("\n")
    print("=" * 60)
    print("PNP GUI DEMO - SCENE GENERATION")
    print("=" * 60)
    print(f"Season: {month.value}")
    print(f"Random seed: {random_seed}")
    print()

    # Get seasonal configuration
    seasonal = _get_seasonal_config(month)
    gen_path = UPath(f"./{scene_name}/gen_output")
    gen_path.mkdir(parents=True, exist_ok=True)
    # Create basic configuration
    config = create_scene_config(
        scene_name=scene_name,
        center_lat=PNP_LAT,
        center_lon=PNP_LON,
        aoi_size_km=PNP_SIZE_KM,
        output_dir=gen_path,
        data_overrides={"material_config_path": seasonal["material_config"]},
        target_resolution_m=10.0,
        description=f"PNP GUI demo scene - {month.value}",
    )

    config.buffer = BufferConfig(size_km=30.0, resolution_m=60.0)
    config.background = BackgroundConfig(
        size_km=120.0, resolution_m=350.0, elevation=0.0
    )
    config.processing.flatten_dem = False

    # Configure multi-species vegetation placement
    config.vegetation_placement = VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [  # Treecover
                VegetationSpecies(
                    name="trees",
                    asset_xml_paths=seasonal["tree_assets"],
                    density_per_hectare=1067 // 4,  # from dataset
                    scale_min=0.8,
                    scale_max=1.15,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=[seasonal["shrub_asset"]],
                    density_per_hectare=40.0,
                    scale_min=0.4,
                    scale_max=0.8,
                )
            ],
        },
        density_variation=0.5,
        min_spacing=0.1,
        max_instances_per_pixel=2500,
        spillover_max_distance_m=50.0,
        spillover_compatibility={
            30: 0.9,  # High spillover into grassland
            20: 0.5,  # Moderate spillover into shrubland
            60: 0.5,
            100: 0.5,
        },
    )

    # Configure atmosphere
    thermoprops = ThermophysicalConfig(
        identifier=None,
        thermoprops_file=UPath(
            f"PNP/timeseries_ms_{seasonal['thermoprops_date']}_v1.nc"
        ),
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=thermoprops,
        # thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
        absorption_database=AbsorptionDatabase.MONOTROPA,
        has_absorption=True,
        has_scattering=True,
    )

    aerosol = ParticleLayerConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.1,
        altitude_bottom=600,
        altitude_top=1600,
        distribution=ExponentialDistribution(),
    )

    config.set_atmosphere_heterogeneous(
        molecular_config=molecular_config, particle_layers=[aerosol]
    )

    # Apply seasonal snow settings
    if seasonal["apply_snow"]:
        config.snow = SnowConfig(season_month="june", thermoprops=thermoprops)

    # Add tower XML scene
    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path="only_tower_v0_1.xml",
            base_coordinate=TOWER_COORDS,
            coord_type="scene",
            elevation_offset=0.1,
        )
    )

    config.random_seed = random_seed

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"
    config_path = gen_path / config_filename
    config.to_json(config_path)

    # Run generation pipeline
    print("\n" + "=" * 60)
    print("Running scene generation pipeline...")
    print("=" * 60)

    scene_path = generation_from_config(config)

    if scene_path:
        print("\n" + "=" * 60)
        print("SCENE GENERATION COMPLETE")
        print("=" * 60)
        print(f"Scene description: {scene_path}")
        print()

    return scene_path


# ============================================================================
# Processor 2: Simulation
# ============================================================================


@registry.process(id="pnp_gui/simulation", title="Simulation Demo")
def pnp_gui_simulation(
    scene_name: Annotated[
        str,
        Field(
            ...,
            description="Scene to sue for simulation",
            title="Scene name",
        ),
    ],
    month: Annotated[
        Month,
        Field(
            default=Month.DECEMBER,
            description="Month for simulation (December=summer, June=winter)",
            title="Month",
        ),
    ],
    hour_utc: Annotated[
        float,
        Field(
            ...,
            description="Hour of observation in UTC (0-23) of the 21st of the chosen month, affects sun position ",
            title="Hour (UTC)",
        ),
    ],
    observation: Annotated[
        ObservationType,
        Field(..., description="Observation type (enum value)", title="Observation"),
    ],
    spp: Annotated[
        int,
        Field(
            ...,
            description="Samples per pixel for Monte Carlo simulation",
            title="Samples per pixel",
        ),
    ] = 8,
    sim_name: Annotated[
        str,
        Field(
            ...,
            description="Simulation run name",
            title="Name of run",
        ),
    ] = None,
) -> str | None:
    """Run simulation for PNP GUI demo with configurable observation types.

    This processor:
    1. Creates a simulation configuration based on observation type
    2. Immediately runs the simulation
    3. Returns path to the simulation output directory

    Supported observation types:
    - CHIME: Hyperspectral satellite sensor
    - MSI: Sentinel-2 multispectral sensor (configurable bands)
    - HYPSTAR: Ground-based hyperspectral sensor with HCRF processing
    - RGB_CAMERA: Perspective camera viewing tower from configurable position
    - SATELLITE_PIXEL_HDRF: [PLACEHOLDER - To be implemented]

    Args:
        scene_name: Name  of scene to be used
        month: Month for simulation (determines observation date)
        hour_utc: Hour of observation in UTC
        observation: Observation type configuration
        spp: Samples per pixel for Monte Carlo simulation
        sim_name: Name of simulation run

    Returns:
        Path to simulation output directory, or None if observation type
        is not yet implemented or simulation fails
    """
    from s2gos_simulator.config import (
        AngularFromOriginViewing,
        AngularViewing,
        DirectionalIllumination,
        GroundInstrumentType,
        GroundSensor,
        HDRFConfig,
        HemisphericalMeasurementLocation,
        IrradianceConfig,
        LookAtViewing,
        PixelHDRFConfig,
        PostProcessingOptions,
        SatelliteInstrument,
        SatellitePlatform,
        SatelliteSensor,
        SimulationConfig,
        SpectralResponse,
        create_chime_sensor,
    )
    from upath import UPath

    from s2gos_apps.sim_util import simulation_from_config

    print("\n")
    print("=" * 60)
    print("PNP GUI DEMO - SIMULATION")
    print("=" * 60)

    print(f"Observation type: {observation}")
    print()

    # Determine observation date (fixed to 21st of month)
    seasonal = _get_seasonal_config(month)
    observation_date = datetime(
        2024,
        seasonal["observation_month"],
        seasonal["observation_day"],
        int(hour_utc),
        0,
        0,
    )
    eradiate_mode = "ckd"

    # Build sensors and measurements based on observation type
    sensors = []
    measurements = []

    if observation == ObservationType.CHIME:
        sensors.append(
            create_chime_sensor(
                target_center_lat=PNP_LAT,
                target_center_lon=PNP_LON,
                target_size_km=PNP_SIZE_KM,
                zenith=3.0,
                samples_per_pixel=spp,
            )
        )

    elif observation == ObservationType.MSI:
        bands = ["1", "2", "3", "4", "5", "6", "7", "8", "8a", "9", "10", "11", "12"]
        for band in bands:
            sensors.append(
                SatelliteSensor(
                    id=f"s2a_b{band}",
                    platform=SatellitePlatform.SENTINEL_2A,
                    instrument=SatelliteInstrument.MSI,
                    band=band,
                    target_center_lat=PNP_LAT,
                    target_center_lon=PNP_LON,
                    target_size_km=PNP_SIZE_KM,
                    film_resolution=(1000, 1000),
                    viewing=AngularViewing(zenith=3.0, azimuth=0.0),
                    samples_per_pixel=spp,
                )
            )

    elif observation == ObservationType.SATELLITE_PIXEL_HDRF:
        sensors.append(
            create_chime_sensor(
                sensor_id="chime_for_pixel_hdrf",
                target_center_lat=PNP_LAT,
                target_center_lon=PNP_LON,
                target_size_km=PNP_SIZE_KM,
                zenith=3.0,
                samples_per_pixel=spp,
                for_reference_only=True,
            )
        )
        # 3x3 around tower pixel
        measurements.append(
            PixelHDRFConfig(
                id="chime_pixel_HDRF",
                satellite_sensor_id="chime_for_pixel_hdrf",
                pixel_indices=[
                    # (137, 158),
                    # (137, 159),
                    # (137, 160),
                    # (138, 158),
                    (138, 159),
                    # (138, 160),
                    # (139, 158),
                    # (139, 159),
                    # (139, 160),
                ],
                height_offset_m=45,
                samples_per_pixel=4,
                srf=SpectralResponse(type="uniform", wmin=400, wmax=2500),
            )
        )

    elif observation == ObservationType.HYPSTAR:
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
                resolution=[128, 128],
                samples_per_pixel=spp,
                srf=SpectralResponse(type="uniform", wmin=380, wmax=1680),
                post_processing=PostProcessingOptions(
                    apply_circular_mask=True,
                    generate_rgb_image=True,
                    rgb_wavelengths=(660.0, 550.0, 440.0),
                    rgb_brightness_factor=1.8,
                    apply_srf=True,
                    spatial_averaging=True,
                ),
            )
        )

        # Add irradiance measurement and HDRF config
        measurements.extend(
            [
                IrradianceConfig(
                    id="hypstar_irradiance",
                    location=HemisphericalMeasurementLocation(
                        target_x=TOWER_COORDS[0],
                        target_y=TOWER_COORDS[1],
                        target_z=4,
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

    elif observation == ObservationType.RGB_CAMERA:
        eradiate_mode = "mono"

        camera_x, camera_y = coord_system.latlon_to_scene(
            PNP_LAT - 0.0010, PNP_LON + 0.0015
        )

        sensors.append(
            GroundSensor(
                id="nice_rgb_camera",
                instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
                viewing=LookAtViewing(
                    origin=[0, 0, 39000.0],
                    target=[0, 0, 15],
                    up=[0, 1, 0],
                    relative_to_asset="only_tower_v0_1.xml",
                ),
                srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
                fov=50.0,
                resolution=[1280, 1280],
                samples_per_pixel=32,
                post_processing=PostProcessingOptions(
                    generate_rgb_image=True,
                    rgb_wavelengths=(660.0, 550.0, 440.0),
                    rgb_brightness_factor=1.8,
                ),
            )
        )

    # Create simulation configuration
    simulation_config = SimulationConfig(
        name=f"pnp_gui_{observation}",
        description=f"PNP GUI demo simulation - {observation}",
        illumination=DirectionalIllumination.from_date_and_location(
            observation_date,
            PNP_LAT,
            PNP_LON,
            "coddington_2022-1_nm",
        ),
        sensors=sensors,
        measurements=measurements,
        backend_hints={"eradiate": {"mode": eradiate_mode}},
    )

    # Save simulation configuration
    config_filename = f"pnp_gui_{observation}_sim_config.json"

    sim_output_dir = UPath(f"./{scene_name}/sim_output")
    sim_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = sim_output_dir / config_filename

    simulation_config.to_json(config_path)

    # Run simulation
    print("\n" + "=" * 60)
    print("Running simulation...")
    print("=" * 60)

    output_path = simulation_from_config(
        UPath(f"./{scene_name}/gen_output/{scene_name}/{scene_name}.yml"),
        simulation_config,
        UPath(f"./{scene_name}/sim_output"),
    )

    if output_path:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_path}")
        print()

    return str(output_path)
