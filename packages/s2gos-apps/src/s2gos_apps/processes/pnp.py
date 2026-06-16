#!/usr/bin/env python3
import logging
from typing import Annotated

from pydantic import Field
from s2gos_generator.core.config import (
    AerosolDataset,
    ExponentialDistribution,
    ParticleLayerConfig,
    VegetationPlacementConfig,
    VegetationSpecies,
    XmlSceneConfig,
)
from s2gos_utils.io import PathRef, mkdir

from s2gos_apps.registry import registry

logger = logging.getLogger(__name__)


@registry.process(id="pnp-generation-config", title="PNP Generation Config")
def generation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    random_seed: Annotated[
        int, Field(..., description="RNG seed, mostly for vegetation")
    ],
    config_output_dir: Annotated[
        PathRef,
        Field(description="Generation configuration output directory."),
    ] = PathRef("./gen_config"),
    scene_output_dir: Annotated[
        PathRef,
        Field(description="Scene description output directiory."),
    ] = PathRef("./gen_output"),
) -> PathRef:
    """
    Create the scene confifuration corresponding the PNP scene.
    """
    from s2gos_generator import create_scene_config
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        MolecularAtmosphereConfig,
        ThermophysicalConfig,
    )

    logger.info("Configuring generation...")

    # Create basic configuration using defaults
    config = create_scene_config(
        scene_name=scene_name,
        center_lat=target_lat,
        center_lon=target_lon,
        aoi_size_km=target_size,
        output_dir=scene_output_dir,
        target_resolution_m=10.0,
        description="PNP cite and surroundings",
    )

    config.processing.flatten_dem = False

    # Configure multi-species vegetation placement with trees and shrubs
    config.vegetation_placement = VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [  # Treecover
                VegetationSpecies(
                    name="trees",
                    asset_xml_paths=[
                        "tls_tree_38_prospect.xml",
                        "tls_tree_71_prospect.xml",
                        "tls_tree_165_prospect.xml",
                        "tls_tree_228_prospect.xml",
                        "tls_tree_290_prospect.xml",
                        "tls_tree_300_prospect.xml",
                        "tls_tree_336_prospect.xml",
                    ],
                    density_per_hectare=1067,  # from dataset
                    scale_min=0.8,
                    scale_max=1.15,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=["tls_tree_336_prospect.xml"],
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

    thermoprops = ThermophysicalConfig(
        identifier=None,
        thermoprops_file=PathRef("timeseries_ms_2020-12-21_v1.nc"),
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

    config.set_atmosphere_molecular(molecular_config=molecular_config)

    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path="only_tower_v0_1.xml",
            base_coordinate=(-220, 850),
            coord_type="scene",
            elevation_offset=0.1,
        )
    )
    config.random_seed = random_seed

    logger.info("Basic configuration created")
    logger.info("Configuration validation passed")

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"

    mkdir(config_output_dir)
    config_path = config_output_dir / config_filename

    config.to_json(config_path)

    return config_path


@registry.process(id="pnp-simulation-config", title="PNP Simulation Config")
def simulation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    gmt_hour: Annotated[
        float, Field(..., description="Hour of observation at target in GMT time.")
    ],
    spp: Annotated[int, Field(..., description="Number of Monte Carlo samples.")] = 8,
    config_output_dir: Annotated[
        PathRef,
        Field(description="Simulation configuration output directiory."),
    ] = PathRef("./sim_config"),
) -> PathRef:
    from s2gos_apps.sim_util import simulation_config

    return simulation_config(
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp,
        config_output_dir,
    )
