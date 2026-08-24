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


def way_material_candidates(ways_cfg) -> set[str]:
    """Return every material name a way can end up painted with.

    Six sources feed way material resolution (see ``processors/ways.py``
    ``_get_road_material`` / ``_get_railway_material``), and a name reachable
    through any of them must receive a texture index.
    """
    candidates = {ways_cfg.default_material, ways_cfg.default_railway_material}
    candidates.update(ways_cfg.DEFAULT_SURFACE_MATERIALS.values())
    candidates.update(d.default_material for d in ways_cfg.ROAD_TYPE_TABLE.values())
    candidates.update(d.default_material for d in ways_cfg.RAILWAY_TYPE_TABLE.values())
    for overrides in (ways_cfg.road_overrides, ways_cfg.railway_overrides):
        candidates.update(
            ov.default_material
            for ov in overrides.values()
            if ov.default_material is not None
        )
    return candidates


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
    if ctx.config.ways is not None and ctx.config.ways.enabled:
        overlay_candidates |= way_material_candidates(ctx.config.ways)

    assert landcover_map, "landcover_map must not be empty"
    next_index = max(landcover_map.values()) + 1
    new_materials = sorted(
        name for name in overlay_candidates if name not in landcover_map
    )
    overlay_map = {name: next_index + i for i, name in enumerate(new_materials)}

    return {**landcover_map, **overlay_map}


# Selection textures are 8-bit grayscale PNGs, so material indices must fit in a byte.
MAX_MATERIAL_INDEX = 255


def allocate_matched_indices(
    base_index_map: dict[str, int],
    matched_material_ids: list[str],
) -> dict[str, int]:
    """Assign texture indices to spectrally-matched materials.

    Deduplicates by material id (one index per unique material): ids already
    present in ``base_index_map`` reuse their existing index, while genuinely new
    ids receive contiguous indices above ``max(base_index_map)``. New ids are
    sorted, so the allocation is deterministic — the texture step writes it to the
    sidecar and the same inputs always reproduce the same indices (stable cache
    fingerprint).

    Args:
        base_index_map: ``{material_name: index}`` for landcover + overlay
            materials, as returned by :func:`build_material_index_map`.
        matched_material_ids: Material ids chosen by spectral matching (may repeat).

    Returns:
        ``{material_id: index}`` covering every unique id in
        ``matched_material_ids``.

    Raises:
        ValueError: If allocation would exceed :data:`MAX_MATERIAL_INDEX`.
    """
    unique_ids = sorted(set(matched_material_ids))
    next_index = (max(base_index_map.values()) + 1) if base_index_map else 0

    result: dict[str, int] = {}
    for mid in unique_ids:
        if mid in base_index_map:
            result[mid] = base_index_map[mid]
        else:
            result[mid] = next_index
            next_index += 1

    if result:
        highest = max(result.values())
        if highest > MAX_MATERIAL_INDEX:
            raise ValueError(
                f"Spectral material index {highest} exceeds the 8-bit selection "
                f"texture limit ({MAX_MATERIAL_INDEX}). Reduce clusters_per_class or "
                f"the number of diversified landcover classes."
            )

    return result
