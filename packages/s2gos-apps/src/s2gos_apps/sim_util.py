import os
from datetime import datetime

import numpy as np
from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import (
    DirectionalIllumination,
    GroundInstrumentType,
    GroundSensor,
    LookAtViewing,
    PostProcessingOptions,
    SimulationConfig,
    SpectralResponse,
)
from s2gos_utils import SceneDescription
from upath import UPath


def top_down_perspective_sensor(target_size, fov, spp):
    distance = (target_size / 2.0) / np.tan(np.deg2rad(fov / 2))

    return GroundSensor(
        id="camera",
        instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
        viewing=LookAtViewing(
            origin=[0.0, 0.0, distance * 1000], target=[0.0, 0.0, 0.0], up=[0, 1, 0]
        ),
        srf=SpectralResponse(type="delta", wavelengths=[440.0, 550.0, 660.0]),
        fov=fov,
        resolution=[512, 512],
        samples_per_pixel=spp,
        post_processing=PostProcessingOptions(apply_srf=False, generate_rgb_image=True),
    )


def simulation_config(
    scene_name: str,
    target_lat: float,
    target_lon: float,
    target_size: float,
    gmt_hour: float,
    spp: int = 8,
    config_output_dir: UPath | None = None,
) -> UPath | None:
    """Expand core parameters to a full simulation config."""
    # Step 3: Configure simulation with enhanced sensors

    config_output_dir = (
        UPath(config_output_dir) if config_output_dir is not None else None
    )

    print("\n")
    print("=" * 60)
    print("Configuring simulation...")

    # create top down sensor
    fov = 50
    sensors = [
        top_down_perspective_sensor(target_size, fov, spp),
    ]

    simulation_config = SimulationConfig(
        name="config_simulation",
        description="Simulation using scene configuration with both sensors and radiative quantities",
        illumination=DirectionalIllumination.from_date_and_location(
            datetime(2024, 1, 1, int(gmt_hour), 0, 0),
            target_lat,
            target_lon,
            "coddington_2022-1_nm",
        ),
        sensors=sensors,
        backend_hints={"eradiate": {"mode": "mono"}},
    )

    print("Simulation configured:")
    print(f"  Sensors: {len(simulation_config.sensors)}")
    for i, sensor in enumerate(simulation_config.sensors):
        platform = sensor.platform_type.value
        instrument = getattr(sensor, "instrument", "N/A")
        if hasattr(instrument, "value"):
            instrument = instrument.value
        print(f"    {i + 1}. {sensor.id} ({platform}/{instrument})")

    # Save simulation configuration
    config_filename = f"{scene_name}_sim_config.json"

    if config_output_dir is None:
        if not os.path.exists("./sim_config"):
            os.mkdir("./sim_config")

        config_path = UPath(f"./sim_config/{config_filename}")
    else:
        if not config_output_dir.exists():
            config_output_dir.mkdir()

        config_path = config_output_dir / config_filename

    simulation_config.to_json(config_path)
    print("  Saved: simulation_config.json")
    return config_path


def simulation_from_config(
    scene_description_path: UPath,
    config: SimulationConfig,
    simulation_output_dir: UPath | None = None,
) -> UPath | None:
    print("\n")
    print("=" * 60)
    print("Simulating observation...")

    scene_description_path = UPath(scene_description_path)
    scene_description = SceneDescription.load_yaml(scene_description_path)

    # Generate schema for reference
    if simulation_output_dir is None:
        simulation_output_dir = UPath(f"./sim_output/{scene_description.name}")

    # Step 4: Run simulation (if available)
    if ERADIATE_AVAILABLE and scene_description:
        print("\nValidating materials and running simulation...")

        available_materials = list(scene_description.materials.keys())
        objects_to_check = scene_description.objects

        if available_materials:
            print(f"Available materials: {available_materials}")

            # Check objects for invalid material references
            material_issues = []
            for i, obj in enumerate(objects_to_check):
                if "material" in obj:
                    mat_ref = obj["material"]
                    if mat_ref not in available_materials:
                        material_issues.append(
                            f"Object {i} ({obj.get('object_id', 'unnamed')}) references unknown material '{mat_ref}'"
                        )

            if material_issues:
                print("WARNING: Material validation found issues:")
                for issue in material_issues:
                    print(f"  - {issue}")
                print(
                    "Note: Material validation issues found, but simulation may still work"
                )
            else:
                print(
                    f"OK: All {len(objects_to_check)} object material references are valid"
                )
        else:
            print("Skipping detailed material validation - using scene file as-is")

        scene_input = scene_description

        if scene_input:
            simulator = EradiateBackend(config)
            simulator.run_simulation(
                scene_input,
                scene_dir=scene_description_path.parent,
                output_dir=simulation_output_dir,
                plot_image=True,
            )
            print("Simulation completed successfully!")
        else:
            print("Skipping simulation - no valid scene description available")
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
