from typing import Annotated

from pydantic import Field
from s2gos_utils.io.paths import PathRef

from s2gos_apps.registry import registry


@registry.process(id="common-generation", title="Scene Generation")
def generation(
    config_path: Annotated[
        PathRef, Field(..., description="Path to the configuration JSON file.")
    ],
) -> PathRef:
    """General process to generate a 3D scene from a scene configuration."""
    from s2gos_generator import SceneGenConfig

    from s2gos_apps.gen_util import generation_from_config

    config = SceneGenConfig.from_json(config_path)
    return generation_from_config(config)
