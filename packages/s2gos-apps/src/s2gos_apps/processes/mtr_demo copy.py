#!/usr/bin/env python3
"""MTR Demo Processors - Self-contained generation and simulation workflows.

This module provides two processors for the MTR (Multi-Temporal Radiometric) demo:
1. mtr_demo/generation - Creates scene with seasonal variations
2. mtr_demo/simulation - Runs simulation with configurable observation types

The demo showcases:
- Seasonal variations (December=summer, June=winter in Patagonia)
- Multiple observation types (CHIME, MSI, HYPSTAR, RGB camera, satellite HDRF)
- Fixed demonstration dates (21st of month)
- UTC time specification
"""

import enum
from datetime import datetime
from typing import Annotated, List, Literal, Tuple

import numpy as np
from pydantic import BaseModel, Field
from s2gos_generator.core.config import (
    AbsorptionDatabase,
    AerosolDataset,
    ExponentialDistribution,
    MolecularAtmosphereConfig,
    ParticleLayerConfig,
    ThermophysicalConfig,
    VegetationPlacementConfig,
    VegetationSpecies,
    XmlSceneConfig,
    create_scene_config,
)
from s2gos_simulator.config import (
    AngularFromOriginViewing,
    AngularViewing,
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    HDRFConfig,
    HemisphericalMeasurementLocation,
    HypstarPostProcessingConfig,
    IrradianceConfig,
    LookAtViewing,
    SatelliteInstrument,
    SatellitePlatform,
    SatelliteSensor,
    SimulationConfig,
    SpectralResponse,
    UAVInstrumentType,
    UAVSensor,
    create_chime_sensor,
)
from s2gos_utils.typing import PathLike
from upath import UPath

from s2gos_apps.gen_util import generation_from_config
from s2gos_apps.registry import registry
from s2gos_apps.sim_util_mtr import simulation_from_config

# ============================================================================
# Constants
# ============================================================================

PNP_LAT = -46.917
PNP_LON = -72.450
PNP_SIZE_KM = 10.0
TOWER_COORDS = (-220.0, 850.0)  # meters in scene coordinates

# ============================================================================
# Enums
# ============================================================================


class Month(enum.StrEnum):
    """Month selection for MTR demo (limited to solstice dates)."""

    DECEMBER = "december"  # Summer in Patagonia
    JUNE = "june"  # Winter in Patagonia


class ObservationType(enum.StrEnum):
    """Available observation types for MTR demo."""

    CHIME = "chime"
    MSI = "msi"
    SATELLITE_HDRF = "satellite_hdrf"
    HYPSTAR = "hypstar"
    RGB_CAMERA = "rgb_camera"


# ============================================================================
# Observation Configuration Models
# ============================================================================


class ChimeSensorConfig(BaseModel):
    """CHIME sensor observation configuration."""

    type: Literal[ObservationType.CHIME] = ObservationType.CHIME
    zenith: Annotated[
        float,
        Field(
            default=3.0,
            ge=0.0,
            le=90.0,
            description="Viewing zenith angle in degrees",
        ),
    ]


class MsiSensorConfig(BaseModel):
    """MSI (Sentinel-2) sensor observation configuration."""

    type: Literal[ObservationType.MSI] = ObservationType.MSI
    zenith: Annotated[
        float,
        Field(
            default=3.0,
            ge=0.0,
            le=90.0,
            description="Viewing zenith angle in degrees",
        ),
    ]
    bands: Annotated[
        List[str],
        Field(
            default=["2", "3", "4", "8", "11", "12"],
            description="MSI bands to simulate",
        ),
    ]


class SatelliteHdrfConfig(BaseModel):
    """HDRF measurement in satellite projection (3x3 pixels around tower).

    NOTE: This observation type is a placeholder for future implementation.
    """

    type: Literal[ObservationType.SATELLITE_HDRF] = ObservationType.SATELLITE_HDRF


class HypstarObservation(BaseModel):
    """Hypstar sensor with FOV view and HCRF processing."""

    type: Literal[ObservationType.HYPSTAR] = ObservationType.HYPSTAR
    fov: Annotated[
        float,
        Field(default=5.0, ge=1.0, le=180.0, description="Field of view in degrees"),
    ]
    resolution: Annotated[
        Tuple[int, int], Field(default=(128, 128), description="Sensor resolution")
    ]


class RgbCameraConfig(BaseModel):
    """RGB perspective camera viewing the tower from fixed position."""

    type: Literal[ObservationType.RGB_CAMERA] = ObservationType.RGB_CAMERA
    distance_m: Annotated[
        float,
        Field(default=100.0, ge=10.0, description="Distance from tower in meters"),
    ]
    azimuth: Annotated[
        float,
        Field(
            default=45.0,
            ge=0.0,
            lt=360.0,
            description="Azimuth angle from tower (degrees, 0=North, 90=East)",
        ),
    ]
    elevation_angle: Annotated[
        float,
        Field(
            default=10.0,
            ge=-30.0,
            le=60.0,
            description="Elevation angle (degrees above horizon)",
        ),
    ]
    fov: Annotated[
        float,
        Field(default=40.0, ge=10.0, le=120.0, description="Field of view (degrees)"),
    ]
    resolution: Annotated[
        Tuple[int, int], Field(default=(2000, 2000), description="Image resolution")
    ]


# ============================================================================
# Helper Functions
# ============================================================================


def _get_seasonal_config(month: Month) -> dict:
    """Return season-specific configuration dictionary.

    Args:
        month: Month enum (DECEMBER or JUNE)

    Returns:
        Dictionary with seasonal configuration:
        - material_config: Path to material configuration file
        - tree_assets: List of tree XML asset paths
        - shrub_asset: Shrub XML asset path
        - thermoprops_date: Date string for thermophysical properties
        - apply_snow: Whether to apply snow
        - observation_date_template: Date for simulation (day 21)
    """
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


@registry.process(id="mtr_demo/generation")
def mtr_demo_generation(
    month: Annotated[
        Month,
        Field(
            default=Month.DECEMBER,
            description="Month for simulation (December=summer, June=winter in Patagonia)",
        ),
    ],
    hour_utc: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            lt=24.0,
            description="Hour of observation in UTC (0-23)",
        ),
    ],
    random_seed: Annotated[
        int, Field(default=42, description="RNG seed for vegetation placement")
    ],
    config_output_dir: Annotated[
        PathLike | None,
        Field(default=None, description="Generation configuration output directory"),
    ] = None,
    scene_output_dir: Annotated[
        PathLike | None,
        Field(default=None, description="Scene description output directory"),
    ] = None,
) -> PathLike | None:
    """Generate 3D scene for MTR demo with seasonal variations.

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
        hour_utc: Hour of observation in UTC (used for file naming)
        random_seed: Random seed for reproducible vegetation placement
        config_output_dir: Optional directory for generation config JSON
        scene_output_dir: Optional directory for scene description YAML

    Returns:
        Path to generated scene description YAML file, or None if validation fails
    """
    print("\n")
    print("=" * 60)
    print("MTR DEMO - SCENE GENERATION")
    print("=" * 60)
    print(f"Season: {month.value}")
    print(f"Observation time: {hour_utc:02.0f}:00 UTC")
    print(f"Random seed: {random_seed}")
    print()

    # Get seasonal configuration
    seasonal = _get_seasonal_config(month)
    scene_name = f"pnp_mtr_demo_{month.value}_{int(hour_utc):02d}utc"

    print(f"Configuring scene: {scene_name}")
    print(f"  Material config: {seasonal['material_config']}")
    print(f"  Tree variants: {len(seasonal['tree_assets'])} assets")
    print(f"  Apply snow: {seasonal['apply_snow']}")

    # Create basic configuration
    material_config_path = (
        f"/home/gonzalezm/test/test2/s2gos-apps/packages/s2gos-generator/"
        f"resources/data/{seasonal['material_config']}"
    )

    config = create_scene_config(
        scene_name=scene_name,
        center_lat=PNP_LAT,
        center_lon=PNP_LON,
        aoi_size_km=PNP_SIZE_KM,
        output_dir=UPath("./gen_output")
        if scene_output_dir is None
        else scene_output_dir,
        data_overrides={"material_config_path": material_config_path},
        target_resolution_m=10.0,
        description=f"PNP MTR demo scene - {month.value}",
    )

    # Disable buffer and background for demo
    config.enable_buffer = False
    config.buffer_size_km = 30.0
    config.buffer_resolution_m = 60.0

    config.enable_background = False
    config.background_elevation = 0.0
    config.background_size_km = 120
    config.background_resolution_m = 350.0
    config.processing.flatten_dem = False

    # Configure multi-species vegetation placement
    print("\nConfiguring vegetation placement...")
    config.vegetation_placement = VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [  # Treecover
                VegetationSpecies(
                    name="trees",
                    asset_xml_paths=seasonal["tree_assets"],
                    density_per_hectare=1067,  # from dataset
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
    print("Configuring atmosphere...")
    thermoprops = ThermophysicalConfig(
        identifier=None,
        thermoprops_file=UPath(
            f"/home/gonzalezm/test/test2/s2gos-apps/example/PNP/"
            f"timeseries_ms_{seasonal['thermoprops_date']}_v1.nc"
        ),
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=thermoprops,
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
        print("Applying snow cover (winter season)...")
        config.apply_seasonal_snow = True
        config.snow_season_month = "june"
        config.snow_thermoprops = thermoprops

    # Add tower XML scene
    print(f"Adding tower at scene coordinates: {TOWER_COORDS}")
    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path="only_tower_v0_1.xml",
            base_coordinate=TOWER_COORDS,
            coord_type="scene",
            elevation_offset=0.1,
        )
    )

    config.random_seed = random_seed

    # Validate configuration
    print("\nValidating configuration...")
    errors = config.validate_configuration()
    if errors:
        print(f"Configuration errors: {errors}")
        return None
    else:
        print("Configuration validation passed")

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"

    if config_output_dir is None:
        config_dir = UPath("./gen_config")
    else:
        config_dir = UPath(config_output_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_filename

    config.to_json(config_path)
    print(f"Configuration saved: {config_path}")

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


@registry.process(id="mtr_demo/simulation")
def mtr_demo_simulation(
    scene_description_path: Annotated[
        PathLike, Field(..., description="Path to scene description YAML file")
    ],
    month: Annotated[
        Month,
        Field(
            default=Month.DECEMBER,
            description="Month for simulation (December=summer, June=winter)",
        ),
    ],
    hour_utc: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            lt=24.0,
            description="Hour of observation in UTC (0-23)",
        ),
    ],
    observation: Annotated[
        ChimeSensorConfig
        | MsiSensorConfig
        | SatelliteHdrfConfig
        | HypstarObservation
        | RgbCameraConfig,
        Field(..., description="Observation type configuration"),
    ],
    spp: Annotated[
        int,
        Field(
            default=8,
            ge=1,
            le=1024,
            description="Samples per pixel for Monte Carlo simulation",
        ),
    ] = 8,
    config_output_dir: Annotated[
        PathLike | None,
        Field(default=None, description="Simulation configuration output directory"),
    ] = None,
    simulation_output_dir: Annotated[
        PathLike | None,
        Field(default=None, description="Simulation output directory"),
    ] = None,
) -> PathLike | None:
    """Run simulation for MTR demo with configurable observation types.

    This processor:
    1. Creates a simulation configuration based on observation type
    2. Immediately runs the simulation
    3. Returns path to the simulation output directory

    Supported observation types:
    - CHIME: Hyperspectral satellite sensor
    - MSI: Sentinel-2 multispectral sensor (configurable bands)
    - HYPSTAR: Ground-based hyperspectral sensor with HCRF processing
    - RGB_CAMERA: Perspective camera viewing tower from configurable position
    - SATELLITE_HDRF: [PLACEHOLDER - To be implemented]

    Args:
        scene_description_path: Path to scene YAML from generation step
        month: Month for simulation (determines observation date)
        hour_utc: Hour of observation in UTC
        observation: Observation type configuration (mutually exclusive)
        spp: Samples per pixel for Monte Carlo simulation
        config_output_dir: Optional directory for simulation config JSON
        simulation_output_dir: Optional directory for simulation outputs

    Returns:
        Path to simulation output directory, or None if observation type
        is not yet implemented or simulation fails
    """
    print("\n")
    print("=" * 60)
    print("MTR DEMO - SIMULATION")
    print("=" * 60)
    print(f"Scene: {scene_description_path}")
    print(f"Season: {month.value}")
    print(f"Observation time: {hour_utc:02.0f}:00 UTC")
    print(f"Observation type: {observation.type}")
    print(f"Samples per pixel: {spp}")
    print()

    # Check for placeholder observation type
    if observation.type == ObservationType.SATELLITE_HDRF:
        print("=" * 60)
        print("SATELLITE HDRF (3x3 pixels around tower)")
        print("Status: TO BE IMPLEMENTED")
        print()
        print("This observation type requires implementation of:")
        print("  - Pixel coordinate calculation for tower location")
        print("  - 3x3 grid generation around tower pixel")
        print("  - Multiple HDRF measurements creation")
        print("=" * 60)
        return None

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

    print(f"Observation date: {observation_date}")
    print()

    # Build sensors and measurements based on observation type
    sensors = []
    measurements = []

    if observation.type == ObservationType.CHIME:
        print("Configuring CHIME sensor...")
        print(f"  Zenith angle: {observation.zenith}°")
        sensors.append(
            create_chime_sensor(
                target_center_lat=PNP_LAT,
                target_center_lon=PNP_LON,
                target_size_km=PNP_SIZE_KM,
                zenith=observation.zenith,
                samples_per_pixel=spp,
            )
        )

    elif observation.type == ObservationType.MSI:
        print("Configuring Sentinel-2 MSI sensors...")
        print(f"  Zenith angle: {observation.zenith}°")
        print(f"  Bands: {', '.join(observation.bands)}")
        for band in observation.bands:
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
                    viewing=AngularViewing(zenith=observation.zenith, azimuth=0.0),
                    samples_per_pixel=spp,
                )
            )

    elif observation.type == ObservationType.HYPSTAR:
        print("Configuring HYPSTAR sensor...")
        print(f"  FOV: {observation.fov}°")
        print(f"  Resolution: {observation.resolution}")
        print(f"  Tower location: {TOWER_COORDS}")

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
                fov=observation.fov,
                resolution=list(observation.resolution),
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

    elif observation.type == ObservationType.RGB_CAMERA:
        print("Configuring RGB camera viewing tower...")
        print(f"  Distance: {observation.distance_m} m")
        print(f"  Azimuth: {observation.azimuth}°")
        print(f"  Elevation: {observation.elevation_angle}°")
        print(f"  FOV: {observation.fov}°")
        print(f"  Resolution: {observation.resolution}")

        # Calculate camera position from tower
        azimuth_rad = np.deg2rad(observation.azimuth)
        elevation_rad = np.deg2rad(observation.elevation_angle)

        # Camera offset from tower in scene coordinates
        # Azimuth: 0=North(+Y), 90=East(+X), 180=South(-Y), 270=West(-X)
        camera_x = TOWER_COORDS[0] + observation.distance_m * np.sin(
            azimuth_rad
        ) * np.cos(elevation_rad)
        camera_y = TOWER_COORDS[1] + observation.distance_m * np.cos(
            azimuth_rad
        ) * np.cos(elevation_rad)
        camera_z = observation.distance_m * np.sin(elevation_rad)

        print(f"  Camera position: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")
        print(f"  Target (tower): {TOWER_COORDS} + 30m height")

        sensors.append(
            UAVSensor(
                id="rgb_camera_tower_view",
                instrument=UAVInstrumentType.PERSPECTIVE_CAMERA,
                viewing=LookAtViewing(
                    origin=[camera_x, camera_y, camera_z],
                    target=[TOWER_COORDS[0], TOWER_COORDS[1], 30.0],  # Tower at ~30m
                    up=[0, 0, 1],
                ),
                srf=SpectralResponse(
                    type="delta", wavelengths=[440.0, 550.0, 660.0]
                ),  # B, G, R
                fov=observation.fov,
                resolution=list(observation.resolution),
                samples_per_pixel=spp,
            )
        )

    # Create simulation configuration
    print("\nCreating simulation configuration...")
    simulation_config = SimulationConfig(
        name=f"mtr_demo_{observation.type}",
        description=f"MTR demo simulation - {observation.type}",
        illumination=DirectionalIllumination.from_date_and_location(
            observation_date,
            PNP_LAT,
            PNP_LON,
            "coddington_2022-1_nm",
        ),
        sensors=sensors,
        measurements=measurements,
        backend_hints={"eradiate": {"mode": "ckd"}},
    )

    print(f"  Sensors configured: {len(simulation_config.sensors)}")
    for i, sensor in enumerate(simulation_config.sensors):
        print(f"    {i + 1}. {sensor.id}")

    print(f"  Measurements configured: {len(simulation_config.measurements)}")
    for i, measurement in enumerate(simulation_config.measurements):
        print(f"    {i + 1}. {measurement.id}")

    # Save simulation configuration
    config_filename = f"mtr_demo_{observation.type}_sim_config.json"

    if config_output_dir is None:
        config_dir = UPath("./sim_config")
    else:
        config_dir = UPath(config_output_dir)

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_filename

    simulation_config.to_json(config_path)
    print(f"  Configuration saved: {config_path}")

    # Run simulation
    print("\n" + "=" * 60)
    print("Running simulation...")
    print("=" * 60)

    output_path = simulation_from_config(
        scene_description_path,
        simulation_config,
        simulation_output_dir,
    )

    if output_path:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_path}")
        print()

    return output_path
