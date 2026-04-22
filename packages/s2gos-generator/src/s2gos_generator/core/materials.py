"""Shared material index utilities for the generator pipeline."""

from __future__ import annotations

from s2gos_utils.scene.materials import get_landcover_mapping

from .context import SceneResourceContext


def landcover_material_to_index(material_config_path) -> dict[str, int]:
    """Return {material_name: texture_index} for all landcover classes.

    Indices are derived from the position in the landcover_mapping from the
    materials config — matching exactly what TerrainMaterialGenerator assigns.
    """
    return {
        material_name: idx
        for idx, material_name in enumerate(
            get_landcover_mapping(material_config_path).values()
        )
    }


def build_material_index_map(ctx: SceneResourceContext) -> dict[str, int]:
    """Build the complete {material_name: texture_index} map.

    Landcover indices (0–N) come from the materials config and match
    TerrainMaterialGenerator. Overlay indices (N+1, N+2, ...) are
    assigned only to materials NOT already covered by landcover,
    ensuring the index sequence is always contiguous.
    """
    landcover_map = landcover_material_to_index(
        ctx.config.data_sources.material_config_path.upath
    )

    overlay_candidates = set(r.material_name for r in ctx.config.material_regions)
    if ctx.config.roads is not None and ctx.config.roads.enabled:
        roads_cfg = ctx.config.roads
        overlay_candidates.add(roads_cfg.default_material)
        overlay_candidates.update(roads_cfg.surface_material_mapping.values())

    assert landcover_map, "landcover_map must not be empty"
    next_index = max(landcover_map.values()) + 1
    new_materials = sorted(
        name for name in overlay_candidates if name not in landcover_map
    )
    overlay_map = {name: next_index + i for i, name in enumerate(new_materials)}

    return {**landcover_map, **overlay_map}
