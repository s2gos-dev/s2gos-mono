"""Selection-texture generation: base material texture, paint overlays, and snow seasonality.

`terrain_material` generates the base selection texture from land cover (with `snow` as its
seasonal input); `overlays` provides the `apply_*` primitives that paint material regions and
roads onto that texture. All three read/write the same selection-texture artifact.
"""

from .overlays import apply_region_materials, apply_ways, apply_ways_to_preview
from .snow import calculate_snow_probability_map
from .terrain_material import TerrainMaterialGenerator

__all__ = [
    "TerrainMaterialGenerator",
    "apply_region_materials",
    "apply_ways",
    "apply_ways_to_preview",
    "calculate_snow_probability_map",
]
