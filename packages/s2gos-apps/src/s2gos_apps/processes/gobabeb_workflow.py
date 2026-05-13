#!/usr/bin/env python3
import os
from typing import Annotated

from procodile import FromMain, FromStep
from pydantic import Field
from upath import UPath

from s2gos_utils.io import PathRef

from s2gos_apps.registry import registry


@registry.main(
    id="gobabeb_generation_simulation_workflow",
    title="Gobabeb Generation and Simulation Workflow",
    outputs={
        "scene_name": Field(..., description="Scene id name."),
        "target_lat": Field(..., description="Target's center latitude."),
        "target_lon": Field(..., description="Target's center longitude."),
        "target_size": Field(..., description="Target's size in [km]."),
        "gmt_hour": Field(),
        "spp": Field(),
        "config_path": Field(
            title="PathRef class with generated config as output"
        ),
        "scene_output_dir": Field(),
        "config_output_dir_simulation": Field(),
        "output_dir_simulation": Field(),
    },
)
def gobabeb_generation_simulation_workflow(
    scene_name: Annotated[str, Field(default="gobabeb", description="Scene id name.")],
    target_lat: Annotated[
        float, Field(default=-23.6015417, description="Target's center latitude.")
    ],
    target_lon: Annotated[
        float, Field(default=15.1258696, description="Target's center longitude.")
    ],
    target_size: Annotated[
        float, Field(default=10.0, description="Target's size in [km].")
    ],
    gmt_hour: Annotated[
        float, Field(default=9.0, description="Hour of observation at target in GMT time.")
    ],
    spp: Annotated[int, Field(..., description="Number of Monte Carlo samples.")] = 8,
    config_output_dir_generation: Annotated[
        PathRef | None,
        Field(description="Generation configuration output directory."),
    ] = PathRef("/mnt/s2gos-output/gen_config"),
    scene_output_dir_generation: Annotated[
        PathRef | None,
        Field(description="Scene description output directory."),
    ] = PathRef("/mnt/s2gos-output/gen_output"),
    config_output_dir_simulation: Annotated[
        PathRef | None,
        Field(description="Simulation configuration output directory."),
    ] = PathRef("/mnt/s2gos-output/sim_config"),
    output_dir_simulation: Annotated[
        PathRef | None,
        Field(description="Simulation output directory."),
    ] = PathRef("/mnt/s2gos-output/sim_output"),
) -> tuple[
    str,
    float,
    float,
    float,
    float,
    int,
    PathRef | None,
    PathRef | None,
    PathRef | None,
    PathRef | None,
]:
    """
    Create the scene configuration corresponding to the Gobabeb scene.
    """
    from s2gos_generator import create_scene_config
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        BackgroundConfig,
        BufferConfig,
        MolecularAtmosphereConfig,
        ThermophysicalConfig,
    )

    # Enforce PathRef type
    config_output_dir = (
        PathRef(config_output_dir_generation) if config_output_dir_generation is not None else None
    )
    scene_output_dir = (
        PathRef(scene_output_dir_generation) if scene_output_dir_generation is not None else None
    )

    print("\n")
    print("=" * 60)
    print("Configuring generation...")

    config = create_scene_config(
        scene_name=scene_name,
        center_lat=target_lat,
        center_lon=target_lon,
        aoi_size_km=target_size,
        output_dir=PathRef("./gen_output")
        if scene_output_dir is None
        else scene_output_dir,
        target_resolution_m=10.0,
        description="Gobabeb PICS with HyperNET mast.",
    )

    config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)
    config.background = BackgroundConfig(
        size_km=150.0, resolution_m=200.0, elevation=0.0
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
        has_scattering=True,
    )

    config.set_atmosphere_molecular(molecular_config)

    print("Basic configuration created")
    print("Configuration validation passed")

    config_filename = f"{config.scene_name}_gen_config.json"

    if config_output_dir is None:
        if not os.path.exists("./gen_config"):
            os.mkdir("./gen_config")

        config_path = PathRef(f"./gen_config/{config_filename}")
    else:
        config_output_dir = PathRef(config_output_dir)
        if not config_output_dir.upath.exists():
            config_output_dir.upath.mkdir()

        config_path = config_output_dir / config_filename

    config.to_json(config_path)

    return (
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp,
        config_path,
        scene_output_dir,
        config_output_dir_simulation,
        output_dir_simulation,
    )


@gobabeb_generation_simulation_workflow.step(
    id="gobabeb-generation",
    title="Generate Scenes for Gobabeb",
    inputs={
        "config_path": FromMain(output="config_path"),
    },
    outputs={
        "scene_description_path": Field()
    }
)
def gobabeb_generation(config_path: PathRef) -> PathRef:
    from s2gos_apps.processes.common.generation import generation
    return generation(config_path)


@gobabeb_generation_simulation_workflow.step(
    id="gobabeb-simulation-config",
    title="Gobabeb Simulation Config",
    inputs={
        "scene_name": FromMain(output="scene_name"),
        "target_lat": FromMain(output="target_lat"),
        "target_lon": FromMain(output="target_lon"),
        "target_size": FromMain(output="target_size"),
        "gmt_hour": FromMain(output="gmt_hour"),
        "spp": FromMain(output="spp"),
        "config_output_dir": FromMain(output="config_output_dir_simulation"),
    },
    outputs={
        "config_path": Field(
            title="PathRef class with simulated config as output",
        )
    }
)
def gobabeb_simulation_config(
    scene_name: str,
    target_lat: float,
    target_lon: float,
    target_size: float,
    gmt_hour: float,
    spp: int,
    config_output_dir: PathRef | None,
) -> PathRef | None:
    from s2gos_apps.sim_util import simulation_config

    config_output_dir = (
        PathRef(config_output_dir).upath if config_output_dir is not None else None
    )

    config_path = simulation_config(
        scene_name,
        target_lat,
        target_lon,
        target_size,
        gmt_hour,
        spp,
        config_output_dir,
    )
    return PathRef(config_path) if config_path is not None else None


@gobabeb_generation_simulation_workflow.step(
    id="gobabeb-simulation",
    inputs={
        "scene_description_path": FromStep(step_id="gobabeb-generation",
                                           output="scene_description_path"),
        "config_path": FromStep(step_id="gobabeb-simulation-config",
                                output="config_path"),
        "simulation_output_dir": FromMain(output="output_dir_simulation"),
    },
    outputs={
        "simulation_path": Field()
    }
)
def gobabeb_simulation(
    scene_description_path: PathRef,
    config_path: PathRef,
    simulation_output_dir: PathRef | None,
) -> UPath:
    from s2gos_apps.processes.common.simulation import simulation
    return simulation(scene_description_path, config_path, simulation_output_dir)
