import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

from s2gos_utils.io.paths import optional_str
from upath import UPath


@dataclass
class SceneAssets:
    """Container for generated scene assets."""

    dem_file: Optional[UPath] = None
    landcover_file: Optional[UPath] = None
    mesh_file: Optional[UPath] = None
    selection_texture_file: Optional[UPath] = None
    preview_texture_file: Optional[UPath] = None
    config_file: Optional[UPath] = None
    scene_description_file: Optional[UPath] = None

    buffer_dem_file: Optional[UPath] = None
    buffer_landcover_file: Optional[UPath] = None
    buffer_mesh_file: Optional[UPath] = None
    buffer_selection_texture_file: Optional[UPath] = None
    buffer_preview_texture_file: Optional[UPath] = None

    background_landcover_file: Optional[UPath] = None
    background_selection_texture_file: Optional[UPath] = None
    background_preview_texture_file: Optional[UPath] = None

    vegetation_objects_file: Optional[UPath] = None
    user_assets_file: Optional[UPath] = None
    hamster_paths_file: Optional[UPath] = None
    ways_file: Optional[UPath] = None
    buildings_objects_file: Optional[UPath] = None

    sentinel2_file: Optional[UPath] = None
    matched_materials_file: Optional[UPath] = None

    def to_dict(self) -> Dict:
        """Convert assets to dictionary."""
        return {k: optional_str(v) for k, v in dataclasses.asdict(self).items()}
