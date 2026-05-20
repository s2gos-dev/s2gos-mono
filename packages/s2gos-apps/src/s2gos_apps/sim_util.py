import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

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

    print(f"  config_output_dir: {config_output_dir!r}")
    if config_output_dir is None:
        if not os.path.exists("./sim_config"):
            os.mkdir("./sim_config")

        config_path = UPath(f"./sim_config/{config_filename}")
    else:
        if not config_output_dir.exists():
            config_output_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_output_dir / config_filename

    simulation_config.to_json(config_path)
    print(f"  Saved: {config_path}")
    return config_path


_MITSUBA_EXTENSIONS = {".ply", ".png", ".bmp", ".exr", ".xml", ".obj"}


def _download_scene_assets(s3_scene_dir: UPath, local_dir: Path) -> None:
    """Download scene assets required by Mitsuba from S3 to a local directory.

    Skips zarr datasets and YAML/JSON files — only mesh and texture files
    that Mitsuba's file resolver needs are copied.

    Uses fs.find() + fs.open() directly on the authenticated filesystem so
    child paths cannot silently lose credentials (unlike rglob).
    """
    print(f"  Downloading scene assets from {s3_scene_dir} → {local_dir}")
    downloaded = 0
    fs = s3_scene_dir.fs
    s3_prefix = s3_scene_dir.path.rstrip("/")
    for file_path_str in fs.find(s3_prefix):
        suffix = Path(file_path_str).suffix.lower()
        if suffix not in _MITSUBA_EXTENSIONS:
            continue
        rel = file_path_str[len(s3_prefix):].lstrip("/")
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(file_path_str, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        downloaded += 1
    print(f"  Downloaded {downloaded} scene asset(s)")


def simulation_from_config(
    scene_description_path: UPath,
    config: SimulationConfig,
    simulation_output_dir: UPath | None = None,
) -> UPath | None:
    print("\n")
    print("=" * 60)
    print("Simulating observation...")

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

            # Mitsuba can only load mesh/texture files from the local filesystem.
            # If the scene is on S3, download assets to a temp directory first.
            s3_scene_dir = scene_description_path.parent
            tmp_dir = None
            if getattr(s3_scene_dir, "protocol", None) in ("s3", "s3a"):
                tmp_dir = tempfile.mkdtemp(prefix="s2gos_scene_")
                local_scene_dir = Path(tmp_dir)
                _download_scene_assets(s3_scene_dir, local_scene_dir)
                effective_scene_dir = UPath(tmp_dir)
            else:
                effective_scene_dir = s3_scene_dir

            try:
                simulator.run_simulation(
                    scene_input,
                    scene_dir=effective_scene_dir,
                    output_dir=simulation_output_dir,
                    plot_image=True,
                )
            finally:
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

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
