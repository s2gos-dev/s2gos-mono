"""San Rossore pine plot: DHP hemispherical images, RGB context views and PAR flux.

Generates the scene once, then runs eradiate twice: the fisheye grid and RGB cameras in
mono (their delta SRFs need it), the PAR flux in ckd (the 400-700 nm band needs it).

Setup: see SETUP.md. Run: pixi run --frozen -e dev python examples/san_rossore_icos.py
"""

from datetime import datetime

from s2gos_apps.sim_util import top_down_perspective_sensor
from s2gos_generator.core.config import (
    AerosolDataset,
    BoxGeometry,
    BufferConfig,
    BuildingsConfig,
    ExclusionZone,
    MeshRefinementConfig,
    MolecularAtmosphereConfig,
    Month,
    ParticleLayerConfig,
    SnowConfig,
    ThermophysicalConfig,
    UniformDistribution,
    VegetationPlacementConfig,
    VegetationSpecies,
    WaysConfig,
    XmlSceneConfig,
    create_scene_config,
)
from s2gos_generator.core.config.vegetation import WayExclusionConfig
from s2gos_generator.core.pipeline import SceneGenerationPipeline
from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import (
    DirectionalIllumination,
    FisheyeOptions,
    FluxConfig,
    GroundInstrumentType,
    GroundSensor,
    HemisphericalMeasurementLocation,
    LookAtViewing,
    PostProcessingOptions,
    SimulationConfig,
    SpectralResponse,
)
from s2gos_utils.coordinates import CoordinateSystem
from upath import UPath

CENTER_LAT, CENTER_LON = 43.7320, 10.3500
AOI_KM = 15.0
BUFFER_KM = 30.0
RESOLUTION_M = 10.0
OUTPUT_DIR = "./san_rossore_icos"
OBS_DATE = datetime(2024, 6, 21, 12, 0, 0)
RANDOM_SEED = 13

PLOT_LONLAT = (10.290910, 43.732022)
FISHEYE_LATLON = (43.732105, 10.290495)

# Calibrated DHP lens: rho(theta) = A*theta + B*theta**2, optical centre and image
# circle in pixels at CALIBRATION_RESOLUTION. Pixels outside the image circle come
# back masked.
CALIBRATION_RESOLUTION = (6000, 4000)
DHP_LENS = FisheyeOptions(
    projection_model="polynomial",
    lens_coefficients=[0.7195, -0.0637],  # Lens A, Lens B
    center_x=3017,  # Lens X
    center_y=1989,  # Lens Y
    image_circle_radius=1643,  # Maximum Radius
    calibration_resolution=CALIBRATION_RESOLUTION,
)

# Render cost: 1.0 -> 6000x4000 (full), 0.25 -> 1500x1000 (preview), 0.1 -> smoke test.
FISHEYE_SCALE = 0.5
FISHEYE_SPP = 4
PAR_SPP = 4
CAMERA_SPP = 16  # the two RGB context views
FISHEYE_RESOLUTION = [
    round(CALIBRATION_RESOLUTION[0] * FISHEYE_SCALE),
    round(CALIBRATION_RESOLUTION[1] * FISHEYE_SCALE),
]

# The nine DHP positions: (number, name, east offset, north offset) in scene metres.
DHP_OFFSETS = [
    (1, "center", 0.0, 0.0),
    (2, "north", 0.0, 21.21),
    (3, "north_east", 15.00, 15.00),
    (4, "east", 21.21, 0.0),
    (5, "south_east", 15.00, -15.00),
    (6, "south", 0.0, -21.21),
    (7, "south_west", -15.00, -15.00),
    (8, "west", -21.21, 0.0),
    (9, "north_west", -15.00, 15.00),
]

FOREST_XML = "sr_rayshade_static_ellipsoids.xml"

WYTHAM_TREES = [
    f"tree_{tid}.xml"
    for tid in (
        "8109",
        "8110",
        "654b",
        "1713",
        "8193",
        "8075a",
        "8057b",
        "8056",
        "2024b",
        "1016",
        "46",
        "118",
        "8149b",
        "8097",
    )
]
WYTHAM_SHRUB = "tree_2024b.xml"

# ── Generation ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Scene Generation")
print("=" * 60)

config = create_scene_config(
    scene_name="san_rossore_icos",
    center_lat=CENTER_LAT,
    center_lon=CENTER_LON,
    aoi_size_km=AOI_KM,
    output_dir=OUTPUT_DIR,
    data_overrides={"material_config_path": "materials.json"},
    dem_resolution_m=RESOLUTION_M,
    landcover_resolution_m=RESOLUTION_M,
)
config.snow = SnowConfig(season_month=Month.JUNE)
config.ways = WaysConfig(
    enabled=True,
    source="overpass",
    mesh_gradient_threshold=0.3,
)
config.buildings = BuildingsConfig(
    material={"concrete": 0.5, "brick": 0.1, "cement_cinder": 0.4},
    pitched_roof_proportion=0.95,
    roof_material="shingle",
    base_skirt_m=4.0,
)
config.mesh_refinement = MeshRefinementConfig(
    decimation_depth=2,
    decimation_tolerance_m=0.1,
)
config.texture_resolution_m = 0.5
config.buffer = BufferConfig(size_km=BUFFER_KM)

config.xml_scenes.append(
    XmlSceneConfig(
        xml_path="san_rossore_tower.xml",
        base_coordinate=PLOT_LONLAT,
        coord_type="geographic",
        elevation_offset=0.1,
        rotation_z=180,
    )
)

config.vegetation_placement = VegetationPlacementConfig(
    enabled=True,
    landcover_species_mapping={
        10: [  # Treecover
            VegetationSpecies(
                name="trees",
                asset_xml_paths=WYTHAM_TREES,
                density_per_hectare=155,
                scale_min=0.8,
                scale_max=1.4,
            )
        ],
        20: [  # Shrubland
            VegetationSpecies(
                name="shrubs",
                asset_xml_paths=[WYTHAM_SHRUB],
                density_per_hectare=40.0,
                scale_min=0.4,
                scale_max=0.8,
            )
        ],
    },
    way_exclusion=WayExclusionConfig(enabled=True, buffer_m=1.2),
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

# The rayshade forest, placed at the plot.
config.xml_scenes.append(
    XmlSceneConfig(
        xml_path=FOREST_XML,
        base_coordinate=PLOT_LONLAT,  # (lon, lat)
        coord_type="geographic",
        elevation_offset=0.0,
        fix_blender_coords=False,
    )
)

# Keep procedural vegetation and buildings out of the measured plot.
config.exclusion_zones = [
    ExclusionZone(
        zone_id="pov_forest_plot",
        geometry=BoxGeometry(
            center=PLOT_LONLAT,  # (lon, lat)
            coord_type="geographic",
            width=100.0,
            height=100.0,
        ),
    )
]

config.set_atmosphere_heterogeneous(
    molecular_config=MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(
            identifier="afgl_1986-us_standard",
        ),
    ),
    particle_layers=[
        ParticleLayerConfig(
            aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
            optical_thickness=0.1,
            altitude_bottom=1000.0,
            altitude_top=1500.0,
            distribution=UniformDistribution(),
        )
    ],
)
config.atmosphere.boa = 0.0

pipeline = SceneGenerationPipeline(config)
pipeline.visualize_dag()
scene_description = pipeline.run()
print(f"Scene generated: {config.scene_output_dir}")
print(f"  objects: {len(scene_description.objects)}")

# ── Simulation ────────────────────────────────────────────────────────────────
if not ERADIATE_AVAILABLE:
    print("\nEradiate not available — skipping simulation.")
else:
    # Sensors are placed in scene metres (east, north).
    coords = CoordinateSystem(CENTER_LAT, CENTER_LON)
    fx, fy = coords.latlon_to_scene(*FISHEYE_LATLON)
    px, py = coords.latlon_to_scene(PLOT_LONLAT[1], PLOT_LONLAT[0])

    illumination = DirectionalIllumination.from_date_and_location(
        OBS_DATE, CENTER_LAT, CENTER_LON
    )
    output_dir = UPath(OUTPUT_DIR) / "sim_output"

    print()
    print("=" * 60)
    print("Step 2: DHP images and RGB context views (mono)")
    print("=" * 60)
    print(f"DHP grid anchored at ({fx:.1f}, {fy:.1f}) m")
    print(f"Plot centre at ({px:.1f}, {py:.1f}) m")

    top_down = top_down_perspective_sensor(AOI_KM, 45, CAMERA_SPP, [1280, 1280])
    top_down.id = "top_down"

    visuals_config = SimulationConfig(
        name="san_rossore_icos_visuals",
        illumination=illumination,
        sensors=[
            *(
                GroundSensor(
                    id=f"fisheye_{dhp_no}_{name}",
                    instrument=GroundInstrumentType.FISHEYE_CAMERA,
                    viewing=LookAtViewing(
                        origin=[fx + dx + 3.2, fy + dy + 1, 0.9],
                        target=[fx + dx + 3.2, fy + dy + 1, 500.0],  # straight up
                        up=[0, 1, 0],
                        terrain_relative_height=True,  # origin z is above the DEM
                    ),
                    srf=SpectralResponse(
                        type="delta", wavelengths=[440.0, 550.0, 660.0]
                    ),
                    fisheye=DHP_LENS,
                    resolution=FISHEYE_RESOLUTION,
                    samples_per_pixel=FISHEYE_SPP,
                    # Black outside the image circle, like a real DHP frame.
                    post_processing=PostProcessingOptions(
                        apply_srf=False,
                        generate_rgb_image=True,
                        apply_circular_mask=True,
                        rgb_background="black",
                    ),
                )
                for dhp_no, name, dx, dy in DHP_OFFSETS
            ),
            # Isometric view of the plot from the south-west.
            GroundSensor(
                id="isometric",
                instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
                viewing=LookAtViewing(
                    origin=[px - 120.0, py - 120.0, 120.0],
                    target=[px, py, 10.0],
                    up=[0, 0, 1],
                    terrain_relative_height=True,
                ),
                srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
                fov=40,
                resolution=[1280, 720],
                samples_per_pixel=CAMERA_SPP,
                post_processing=PostProcessingOptions(
                    apply_srf=False, generate_rgb_image=True
                ),
            ),
            # Nadir view of the whole AOI, centred on the scene origin.
            top_down,
        ],
        backend_hints={"eradiate": {"mode": "mono"}},
    )

    # Each backend sets the eradiate mode on construction: build one, run it, then the next.
    EradiateBackend(visuals_config).run_simulation(
        scene_description,
        scene_dir=config.scene_output_dir,
        output_dir=output_dir / "visuals",
    )
    print(f"DHP images and context views: {output_dir / 'visuals'}")

    print()
    print("=" * 60)
    print("Step 3: PAR flux above the plot (ckd)")
    print("=" * 60)

    # Flux collectors just over the canopy: facing up reads incoming PAR, facing down
    # the PAR reflected by the stand. height_offset_m is above the DEM.
    par_location = HemisphericalMeasurementLocation(
        target_x=fx + 10,
        target_y=fy + 10,
        target_z=24,
        height_offset_m=24,
        terrain_relative_height=True,
        srf=SpectralResponse(type="uniform", wmin=400.0, wmax=700.0),
        samples_per_pixel=PAR_SPP,
    )
    par_config = SimulationConfig(
        name="san_rossore_icos_par",
        illumination=illumination,
        measurements=[
            FluxConfig.upward("par_in", par_location, samples_per_pixel=PAR_SPP),
            FluxConfig.downward("par_out", par_location, samples_per_pixel=PAR_SPP),
        ],
        backend_hints={"eradiate": {"mode": "ckd"}},
    )

    EradiateBackend(par_config).run_simulation(
        scene_description,
        scene_dir=config.scene_output_dir,
        output_dir=output_dir / "par",
    )
    print(f"PAR flux: {output_dir / 'par'}")
