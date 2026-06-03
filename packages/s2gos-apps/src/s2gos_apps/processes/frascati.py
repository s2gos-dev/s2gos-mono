#!/usr/bin/env python3
import logging
import os
from typing import Annotated

logger = logging.getLogger(__name__)

from pydantic import Field
from s2gos_utils.io import PathRef

from s2gos_apps.registry import registry


@registry.process(id="frascati-generation-config", title="Frascati Generation Config")
def generation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    config_output_dir: Annotated[
        PathRef | None,
        Field(..., description="Generation configuration output directory."),
    ] = None,
    scene_output_dir: Annotated[
        PathRef | None,
        Field(..., description="Scene description output directiory."),
    ] = None,
) -> PathRef | None:
    """
    Create the scene confifuration corresponding the Frascati scene.
    """
    from s2gos_generator import create_scene_config
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        BackgroundConfig,
        BufferConfig,
        MolecularAtmosphereConfig,
        ThermophysicalConfig,
        VegetationPlacementConfig,
        VegetationSpecies,
    )

    # Enforce PathRef type
    config_output_dir = (
        PathRef(config_output_dir) if config_output_dir is not None else None
    )
    scene_output_dir = (
        PathRef(scene_output_dir) if scene_output_dir is not None else None
    )

    logger.info("Configuring generation...")

    # Create basic configuration using defaults
    config = create_scene_config(
        scene_name=scene_name,
        center_lat=target_lat,
        center_lon=target_lon,
        aoi_size_km=target_size,
        output_dir=PathRef("./gen_output")
        if scene_output_dir is None
        else scene_output_dir,
        target_resolution_m=10.0,
        description="Frascati city and surroundings",
    )

    config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)
    config.background = BackgroundConfig(
        size_km=150.0, resolution_m=200.0, elevation=0.0
    )

    # Configure multi-species vegetation placement with trees and shrubs
    config.vegetation_placement = VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [  # Treecover
                VegetationSpecies(
                    name="trees",
                    asset_xml_paths=[
                        "tls_tree_25.xml",
                        "tls_tree_71.xml",
                        "tls_tree_165.xml",
                        "tls_tree_228.xml",
                        "tls_tree_290.xml",
                        "tls_tree_300.xml",
                        "tls_tree_336.xml",
                    ],  # Single asset in list
                    # For multiple variants with uniform distribution:
                    # asset_xml_paths=["tree1.xml", "tree2.xml", "tree3.xml"]
                    # For weighted distribution:
                    # asset_xml_paths={"tree_mature.xml": 5.0, "tree_young.xml": 2.0, "tree_old.xml": 1.0}
                    density_per_hectare=450.0,  # Moderate forest density
                    scale_min=0.8,
                    scale_max=1.4,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=["tls_tree_336.xml"],  # Single asset in list
                    density_per_hectare=40.0,
                    scale_min=0.4,
                    scale_max=0.8,
                )
            ],
        },
        density_variation=0.5,
        min_spacing=0.1,
        max_instances_per_pixel=2000,
        spillover_max_distance_m=50.0,
        spillover_compatibility={  # Optional: override global
            30: 0.9,  # High spillover into grassland
            20: 0.5,  # Moderate spillover into shrubland
            60: 0.5,
            100: 0.5,
        },
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
        has_scattering=True,
    )

    config.set_atmosphere_molecular(molecular_config)

    logger.info("Basic configuration created")
    logger.info("Configuration validation passed")

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"

    if config_output_dir is None:
        if not os.path.exists("./gen_config"):
            os.mkdir("./gen_config")

        config_path = PathRef(f"./gen_config/{config_filename}")
    else:
        config_output_dir = PathRef(config_output_dir)
        if not config_output_dir.upath.exists():
            config_output_dir.upath.mkdir(parents=True, exist_ok=True)

        config_path = config_output_dir / config_filename

    config.to_json(config_path)

    return config_path


@registry.process(id="frascati-simulation-config", title="Frascati Simulation Config")
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
        PathRef | None,
        Field(..., description="Simulation configuration output directiory."),
    ] = None,
) -> PathRef | None:
    from s2gos_apps.sim_util import simulation_config

    logger.debug("config_output_dir raw: %r", config_output_dir)
    input_ref = PathRef(config_output_dir) if config_output_dir is not None else None
    cid = input_ref.cid if input_ref is not None else None
    config_output_upath = input_ref.upath if input_ref is not None else None
    logger.debug("config_output_upath=%r  cid=%r", config_output_upath, cid)

    config_path = simulation_config(
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp,
        config_output_upath,
    )
    logger.debug("config_path=%r", config_path)
    if config_path is not None:
        result = PathRef(value=str(config_path), cid=cid)
        logger.debug("returning PathRef(value=%r, cid=%r)", result.value, result.cid)
        return result
    return None
