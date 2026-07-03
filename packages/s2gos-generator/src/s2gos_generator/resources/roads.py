"""Road infrastructure resource — fetches OSM road data and saves a sidecar."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..core.context import SceneResourceContext
from ..processors.roads import (
    RoadsFetchError,
    fetch_osm_data,
    parse_roads,
    roads_to_sidecar,
)


def process_target_roads(ctx: SceneResourceContext) -> Optional[Path]:
    """Fetch/load road data, parse it to segments, and save the sidecar JSON."""
    roads_cfg = ctx.config.roads
    if roads_cfg is None or not roads_cfg.enabled:
        return None

    bbox_west, bbox_south, bbox_east, bbox_north = ctx.target_aoi_polygon.bounds

    try:
        osm_data = fetch_osm_data(
            roads_cfg, bbox_south, bbox_west, bbox_north, bbox_east
        )
    except RoadsFetchError as exc:
        logging.error("Road fetch failed, skipping roads: %s", exc)
        return None
    if osm_data is None:
        logging.warning("No road data available — skipping roads")
        return None

    parsed = parse_roads(
        osm_data, roads_cfg, ctx.coordinate_system, ctx.target_scene_bounds
    )
    if not parsed:
        logging.info("No roads found in AOI — skipping")
        return None

    sidecar_path = ctx.data_dir / "roads.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(roads_to_sidecar(parsed), f)
    ctx.assets.roads_file = sidecar_path

    materials = sorted({r.material for r in parsed})
    logging.info(
        "Roads sidecar saved: %s (%d segments, %d materials: %s)",
        sidecar_path,
        len(parsed),
        len(materials),
        ", ".join(materials),
    )
    return sidecar_path
