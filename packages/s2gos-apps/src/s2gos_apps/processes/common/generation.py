from typing import Annotated

from pydantic import Field
from s2gos_utils.typing import PathLike

from s2gos_apps.registry import registry


@registry.process(id="common/generation")
def generation(
    config_path: Annotated[
        PathLike, Field(..., description="Path to the configuration JSON file.")
    ],
) -> PathLike | None:
    """General process to generate a 3D scene from a scene configuration."""
    from s2gos_generator import SceneGenConfig

    from s2gos_apps.gen_util import generation_from_config

    config = SceneGenConfig.from_json(config_path)
    scene_descr_path = generation_from_config(config)

    return scene_descr_path
