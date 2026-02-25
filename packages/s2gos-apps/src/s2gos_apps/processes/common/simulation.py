from typing import Annotated

from pydantic import Field
from s2gos_utils.typing import PathLike

from s2gos_apps.registry import registry


@registry.process(id="common/simulation")
def simulation(
    scene_description_path: Annotated[
        PathLike, Field(..., description="Path to scene description yaml file.")
    ],
    config_path: Annotated[
        PathLike,
        Field(..., description="Path to the simulation configuration JSON file."),
    ],
    simulation_output_dir: Annotated[
        PathLike | None,
        Field(..., description="Path to the simulation output directory."),
    ] = None,
) -> PathLike | None:
    """
    General process to simulate observations from 3D scene descriptions and
    simulation configurations.
    """
    from s2gos_simulator.config import SimulationConfig

    from s2gos_apps.sim_util import simulation_from_config

    config = SimulationConfig.from_json(config_path)
    return simulation_from_config(scene_description_path, config, simulation_output_dir)
