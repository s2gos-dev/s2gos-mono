#!/usr/bin/env python3
import os
from typing import Annotated

from pydantic import Field
from s2gos_generator.core.config import (
    AerosolDataset,
    ExponentialDistribution,
    ParticleLayerConfig,
    SnowConfig,
    VegetationPlacementConfig,
    VegetationSpecies,
    XmlSceneConfig,
)
from s2gos_utils.typing import PathLike

from s2gos_apps.registry import registry


@registry.process(id="pnp/generation_config_seasonal")
def generation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    random_seed: Annotated[
        int, Field(..., description="RNG seed, mostly for vegetation")
    ],
    season: Annotated[
        str,
        Field(..., description="Season/month: 'december' (summer) or 'june' (winter)"),
    ],
    config_output_dir: Annotated[
        PathLike | None,
        Field(..., description="Generation configuration output directory."),
    ] = None,
    scene_output_dir: Annotated[
        PathLike | None,
        Field(..., description="Scene description output directiory."),
    ] = None,
) -> PathLike | None:
    """
    Create the scene configuration corresponding the PNP scene with seasonal variations.

    Args:
        season: Season for simulation - 'december' (summer) or 'june' (winter).
                Controls material config, tree assets, thermoprops, and snow settings.
    """
    from s2gos_generator import create_scene_config
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        MolecularAtmosphereConfig,
        ThermophysicalConfig,
    )
    from upath import UPath

    print("\n")
    print("=" * 60)
    print(f"Configuring generation for season: {season}...")

    # Determine season-specific configuration
    if season == "june":
        material_config = "materials_winter.json"
        tree_assets = [
            "tls_tree_38_winter.xml",
            "tls_tree_71_winter.xml",
            "tls_tree_165_winter.xml",
            "tls_tree_228_winter.xml",
            "tls_tree_290_winter.xml",
            "tls_tree_300_winter.xml",
            "tls_tree_336_winter.xml",
        ]
        shrub_asset = "tls_tree_336_winter.xml"
        thermoprops_date = "2020-06-21"
        apply_snow = True
    else:  # december
        material_config = "materials.json"
        tree_assets = [
            "tls_tree_38_prospect.xml",
            "tls_tree_71_prospect.xml",
            "tls_tree_165_prospect.xml",
            "tls_tree_228_prospect.xml",
            "tls_tree_290_prospect.xml",
            "tls_tree_300_prospect.xml",
            "tls_tree_336_prospect.xml",
        ]
        shrub_asset = "tls_tree_336_prospect.xml"
        thermoprops_date = "2020-12-21"
        apply_snow = False

    # Create basic configuration using defaults
    config = create_scene_config(
        scene_name=scene_name,
        center_lat=target_lat,
        center_lon=target_lon,
        aoi_size_km=target_size,
        output_dir=UPath("./gen_output")
        if scene_output_dir is None
        else scene_output_dir,
        data_overrides={
            "material_config_path": f"/home/gonzalezm/test/test2/s2gos-apps/packages/s2gos-generator/resources/data/{material_config}"
        },
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
                    asset_xml_paths=tree_assets,
                    density_per_hectare=1067,  # from dataset
                    scale_min=0.8,
                    scale_max=1.15,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=[shrub_asset],
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
        spillover_compatibility={  # Optional: override global
            30: 0.9,  # High spillover into grassland
            20: 0.5,  # Moderate spillover into shrubland
            60: 0.5,
            100: 0.5,
        },
    )

    thermoprops = ThermophysicalConfig(
        identifier=None,
        thermoprops_file=UPath(
            f"/home/gonzalezm/test/test2/s2gos-apps/example/PNP/timeseries_ms_{thermoprops_date}_v1.nc"
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

    config.set_atmosphere_molecular(molecular_config=molecular_config)

    # Apply seasonal snow settings
    if apply_snow:
        config.snow = SnowConfig(season_month="june", thermoprops=thermoprops)

    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path="only_tower_v0_1.xml",
            base_coordinate=(-220, 850),
            coord_type="scene",
            elevation_offset=0.1,
        )
    )
    config.random_seed = random_seed

    print("Basic configuration created")

    # Validate configuration
    errors = config.validate_configuration()
    if errors:
        print(f"Configuration errors: {errors}")
        return None
    else:
        print("Configuration validation passed")

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"

    if config_output_dir is None:
        if not os.path.exists("./gen_config"):
            os.mkdir("./gen_config")

        config_path = UPath(f"./gen_config/{config_filename}")
    else:
        if not os.path.exists(UPath(config_output_dir)):
            os.mkdir(UPath(config_output_dir))

        config_path = UPath(config_output_dir) / config_filename

    config.to_json(config_path)

    return config_path


@registry.process(id="pnp/simulation_config_seasonal")
def simulation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    gmt_hour: Annotated[
        float, Field(..., description="Hour of observation at target in GMT time.")
    ],
    season: Annotated[
        str,
        Field(..., description="Season/month: 'december' (summer) or 'june' (winter)"),
    ],
    spp: Annotated[int, Field(..., description="Number of Monte Carlo samples.")] = 8,
    config_output_dir: Annotated[
        PathLike | None,
        Field(..., description="Simulation configuration output directiory."),
    ] = None,
) -> PathLike | None:
    from s2gos_apps.sim_util_seasonal import simulation_config

    config_path = simulation_config(
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp=spp,
        season=season,
        config_output_dir=config_output_dir,
    )
    return config_path
