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


@registry.process(id="pnp/generation_config")
def generation_configs(
    scene_name: Annotated[str, Field(..., description="Scene id name.")],
    target_lat: Annotated[float, Field(..., description="Target's center latitude.")],
    target_lon: Annotated[float, Field(..., description="Target's center longitude.")],
    target_size: Annotated[float, Field(..., description="Target's size in [km].")],
    random_seed: Annotated[
        int, Field(..., description="RNG seed, mostly for vegetation")
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
    Create the scene confifuration corresponding the PNP scene.
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
    print("Configuring generation...")

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
            "material_config_path": "/home/gonzalezm/test/test2/s2gos-apps/packages/s2gos-generator/resources/data/materials_winter.json"
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
                    asset_xml_paths=[
                        "tls_tree_38_winter.xml",
                        "tls_tree_71_winter.xml",
                        "tls_tree_165_winter.xml",
                        "tls_tree_228_winter.xml",
                        "tls_tree_290_winter.xml",
                        "tls_tree_300_winter.xml",
                        "tls_tree_336_winter.xml",
                    ],  # Single asset in list
                    # For multiple variants with uniform distribution:
                    # asset_xml_paths=["tree1.xml", "tree2.xml", "tree3.xml"]
                    # For weighted distribution:
                    # asset_xml_paths={"tree_mature.xml": 5.0, "tree_young.xml": 2.0, "tree_old.xml": 1.0}
                    density_per_hectare=1067,  # from dataset
                    scale_min=0.8,
                    scale_max=1.15,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=["tls_tree_336_winter.xml"],  # Single asset in list
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

    # config.vegetation_placement = VegetationPlacementConfig(
    #     enabled=True,
    #     landcover_species_mapping={
    #         10: [  # Treecover
    #             VegetationSpecies(
    #                 name="trees",
    #                 asset_xml_paths=[
    #                     "tls_tree_38_bilambertian.xml",
    #                 ],  # Single asset in list
    #                 # For multiple variants with uniform distribution:
    #                 # asset_xml_paths=["tree1.xml", "tree2.xml", "tree3.xml"]
    #                 # For weighted distribution:
    #                 # asset_xml_paths={"tree_mature.xml": 5.0, "tree_young.xml": 2.0, "tree_old.xml": 1.0}
    #                 density_per_hectare=1067 // 3,  # from dataset
    #                 scale_min=0.8,
    #                 scale_max=1.15,
    #             )
    #         ],
    #         20: [  # Shrubland
    #             VegetationSpecies(
    #                 name="shrubs",
    #                 asset_xml_paths=["tls_tree_336.xml"],  # Single asset in list
    #                 density_per_hectare=40.0,
    #                 scale_min=0.4,
    #                 scale_max=0.8,
    #             )
    #         ],
    #     },
    #     density_variation=0.5,
    #     min_spacing=0.1,
    #     max_instances_per_pixel=2500,
    #     spillover_max_distance_m=50.0,
    #     spillover_compatibility={  # Optional: override global
    #         30: 0.9,  # High spillover into grassland
    #         20: 0.5,  # Moderate spillover into shrubland
    #         60: 0.5,
    #         100: 0.5,
    #     },
    # )

    thermoprops = ThermophysicalConfig(
        identifier=None,
        thermoprops_file=UPath(
            "/home/gonzalezm/test/test2/s2gos-apps/example/PNP/timeseries_ms_2020-06-21_v1.nc"
        ),
    )

    # print("Added XML scene to configuration - assets and materials will be loaded automatically")
    molecular_config = MolecularAtmosphereConfig(
        # thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
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
    config.snow = SnowConfig(season_month="june", thermoprops=thermoprops)
    # config.month = "january"
    # config.terrain_texture = TerrainTextureConfig(
    #     month="july",
    #     overrides=[
    #         MonthlyMaterialOverride(
    #             landcover_class=10,
    #             base_material="treecover",
    #             july="rough_aluminum"
    #         )
    #     ]
    # )

    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path="only_tower_v0_1.xml",
            base_coordinate=(-220, 850),
            coord_type="scene",
            elevation_offset=0.1,
        )
    )
    config.random_seed = random_seed
    # config.vegetation_exclusion_zones = [
    #     VegetationExclusionZone(
    #         zone_id="tower",
    #         geometry=BoxGeometry(
    #             center=(-220, 858), coord_type="scene", width=50, height=50
    #         ),
    #     )
    # ]

    # config.vegetation_exclusion_zones = [
    #     VegetationExclusionZone(
    #         zone_id="tower",
    #         geometry=CircleGeometry(center=(-220, 850), coord_type="scene", radius=25),
    #     )
    # ]

    # config.vegetation_exclusion_zones = [
    #     VegetationExclusionZone(zone_id="tower", geometry=BoxGeometry(center=(-46.900864, -72.452897), width=1000, height=1000))
    # ]

    # config.vegetation_exclusion_zones = [
    #     VegetationExclusionZone(zone_id="tower", geometry=PolygonGeometry(coordinates=[
    #         (-46.88608227959665, -72.4499488525779),
    #         (-46.90399860635168, -72.45826733305336),
    #         (-46.90923197145512, -72.48291021533309),
    #         (-46.91302438392444, -72.45660638286215),
    #         (-46.92848134325272, -72.47525179675789),
    #         (-46.91886113779319, -72.44996848231473),
    #         (-46.92445434645551, -72.43178991149672),
    #         (-46.91274564314465, -72.44267178571985),
    #         (-46.90806316892585, -72.42415249644807),
    #         (-46.90458100914423, -72.44504736980562),
    #         (-46.88608227959665, -72.4499488525779)
    #     ]))
    # ]
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


@registry.process(id="pnp/simulation_config")
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
        PathLike | None,
        Field(..., description="Simulation configuration output directiory."),
    ] = None,
) -> PathLike | None:
    from s2gos_apps.sim_util import simulation_config

    config_path = simulation_config(
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp,
        config_output_dir,
    )
    return config_path


if __name__ == "__main__":
    scene_name = "pnp"
    target_lat = -46.917
    target_lon = -72.450
    target_size = 10
    gmt_hour = 14
    spp = 4
