from datetime import datetime
from s2gos_apps.sim_util import top_down_perspective_sensor

from s2gos_generator.core.config import (
    AerosolDataset,
    BuildingsConfig,
    ExponentialDistribution,
    MolecularAtmosphereConfig,
    ParticleLayerConfig,
    ThermophysicalConfig,
    VegetationPlacementConfig,
    VegetationSpecies,
    WaysConfig,
    WaterConfig,
    create_scene_config,
)
from s2gos_generator.core.pipeline import SceneGenerationPipeline
from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import DirectionalIllumination, SimulationConfig
from s2gos_simulator.config.sensors import (
    GroundInstrumentType,
    GroundSensor,
    PostProcessingOptions, 
)
from s2gos_simulator.config.spectral import SpectralResponse
from s2gos_simulator.config.viewing import LookAtViewing
from s2gos_utils.io.resolver import resolver
from upath import UPath

LAT = 41.8210
LON = 12.5700
AOI_KM = 20.0
RESOLUTION_M = 5.0
OUTPUT_DIR = "./examples/outputs/frascati_demo_output"
FOV = 50
SPP = 10
TOWER_COORDS = (-220.0, 850.0)  # meters in scene coordinates
RANDOM_SEED = 13
SCENE_NAME = "frascati"
# Data Dependencies
PATHS = {
    "hamster": "DOY181_Frascati.nc",
    "thermo": "timeseries_ms_2022-05-02_v1.nc",
    "aerosol": "D5_aerosol_model_v5_gobabeb_ert.nc",
    "rpv": "RPV_gobabeb.nc",
    "cams": "Gobabeb.nc",
    "kinne": "altitude_t.nc",
}

resolved_paths = {k: str(resolver.resolve(v, strict=True)) for k, v in PATHS.items()}

print(resolved_paths)
# ── Generation ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Scene Generation")
print("=" * 60)


config = create_scene_config(
    scene_name=SCENE_NAME,
    center_lat=LAT,
    center_lon=LON,
    aoi_size_km=20.0,
    dem_resolution_m=20.0,
    landcover_resolution_m=10.0,
    output_dir=UPath(OUTPUT_DIR) / "scene",
    description="HYPSTAR validation scene",
)

# config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)
# config.background = BackgroundConfig(size_km=100.0, resolution_m=200.0)



config.ways = WaysConfig()
config.water = WaterConfig()
config.texture_resolution_m = 3.0

config.buildings = BuildingsConfig(
    material={"concrete": 0.5, "brick": 0.1, "cement_cinder": 0.4},
    pitched_roof_proportion=0.95,
    roof_material="shingle",
)


# )

config.set_atmosphere_heterogeneous(
    molecular_config=MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
    ),
    particle_layers=[
        ParticleLayerConfig(
            aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
            optical_thickness=0.1,
            altitude_bottom=0.0,
            altitude_top=10_000.0,
            distribution=ExponentialDistribution(scale=1000.0),
        )
    ],
)

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
                density_per_hectare=1067 // 2,
                scale_min=0.8,
                scale_max=1.15,
            )
        ],
        20: [  # Shrubland
            VegetationSpecies(
                name="shrubs",
                asset_xml_paths=["tls_tree_336_prospect.xml"],
                density_per_hectare=100.0,
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
        30: 0.9,
        20: 0.5,
        60: 0.5,
        100: 0.5,
    },
    random_seed=RANDOM_SEED,
)

pipeline = SceneGenerationPipeline(config)
path = pipeline.visualize_dag()  # saves <output_dir>/pnp_demo_summer_dag.png
scene_description = pipeline.run()
print(path)
print(f"Scene generated: {config.scene_output_dir}")

print()
print("=" * 60)
print("Step 2: Simulation")
print("=" * 60)

if not ERADIATE_AVAILABLE:
    print("Eradiate not available — skipping simulation.")
else:
    sim_config = SimulationConfig(
        name="frascati_sim",
        illumination=DirectionalIllumination.from_date_and_location(
            datetime(2024, 6, 21, 10, 0, 0), LAT, LON
        ),
        sensors=[
            # GroundSensor(
            #     id="scene_camera",
            #     instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            #     viewing=LookAtViewing(
            #         origin=[0.0, -5000.0, 1500], target=[0.0, 0.0, 0.0], up=[0, 0, 1]
            #     ),
            #     srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
            #     fov=45,
            #     resolution=[1920, 1080],
            #     samples_per_pixel=4,
            #     post_processing=PostProcessingOptions(
            #         apply_srf=False, generate_rgb_image=True
            #     ),
            # ),
            # GroundSensor(
            #     id="zoom_sensor",
            #     instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            #     viewing=LookAtViewing(
            #         origin=[7500.0, -5500.0, 1100], target=[7500.0, -5500.0, 0], up=[0, 1, 0]
            #     ),
            #     srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
            #     fov=45,
            #     resolution=[1920, 1080],
            #     samples_per_pixel=4,
            #     post_processing=PostProcessingOptions(
            #         apply_srf=False, generate_rgb_image=True
            #     ),
            # ),
            #             GroundSensor(
            #     id="zoom_sensor_2",
            #     instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            #     viewing=LookAtViewing(
            #         origin=[6000.0, -4000.0, 1100], target=[6000.0, -4000.0, 0], up=[0, 1, 0]
            #     ),
            #     srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
            #     fov=45,
            #     resolution=[1920, 1080],
            #     samples_per_pixel=4,
            #     post_processing=PostProcessingOptions(
            #         apply_srf=False, generate_rgb_image=True
            #     ),
            # ),
            top_down_perspective_sensor(
                target_size=AOI_KM, fov=FOV, resolution=[1920, 1920], spp=32
            )
        ],
        backend_hints={"eradiate": {"mode": "mono"}},
    )

    output_dir = UPath(OUTPUT_DIR) / "sim_output"
    backend = EradiateBackend(sim_config)
    backend.run_simulation(
        scene_description,
        scene_dir=config.scene_output_dir,
        output_dir=output_dir,
        plot_image=True,
        id_to_plot="rgb_camera",
    )
    print(f"Simulation output: {output_dir}")
