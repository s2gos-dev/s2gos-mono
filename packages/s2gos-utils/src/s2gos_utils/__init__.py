from .setting import load_config, settings

# Start by loading the config. Currently required so that other imports work.
# This is code smell.. MaterialConfigLoader is also initialized globally.
# Might need to consider better syster management or proper lazy loading.
load_config()


from .io.paths import (
    PathRef,
    exists,
    is_remote_path,
    mkdir,
    normalize_path,
    open_file,
    optional_str,
    read_json,
    read_yaml,
)
from .io.resolver import FileResolver, resolver
from .scene.description import SceneDescription
from .scene.materials import (
    Material,
    MaterialConfigLoader,
    get_landcover_mapping,
    load_materials,
)
from .typing import PathLike
from .versioning import (
    check_version_compatibility,
    get_package_version,
    get_version_info,
    parse_version,
    validate_config_version,
    version_stamp,
)

__all__ = [
    "PathRefSceneDescription",
    "Material",
    "MaterialConfigLoader",
    "load_materials",
    "get_landcover_mapping",
    "open_file",
    "exists",
    "read_json",
    "read_yaml",
    "normalize_path",
    "is_remote_path",
    "mkdir",
    "optional_str",
    "FileResolver",
    "resolver",
    "PathLike",
    "check_version_compatibility",
    "get_version_info",
    "parse_version",
    "validate_config_version",
    "version_stamp",
    "get_package_version",
]
