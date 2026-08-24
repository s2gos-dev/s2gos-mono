"""Way infrastructure resource — fetches OSM way (road and railway) data and saves a sidecar."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..core.context import SceneResourceContext
from ..processors.ways import (
    WaysFetchError,
    fetch_osm_data,
    parse_ways,
    ways_to_sidecar,
)


def process_target_ways(ctx: SceneResourceContext) -> Optional[Path]:
    """Fetch/load way data, parse it to segments, and save the sidecar JSON."""
    ways_cfg = ctx.config.ways
    if ways_cfg is None or not ways_cfg.enabled:
        return None

    bbox_west, bbox_south, bbox_east, bbox_north = ctx.target_aoi_polygon.bounds

    try:
        osm_data = fetch_osm_data(
            ways_cfg, bbox_south, bbox_west, bbox_north, bbox_east
        )
    except WaysFetchError as exc:
        logging.error("Way fetch failed, skipping ways: %s", exc)
        return None
    if osm_data is None:
        logging.warning("No way data available — skipping ways")
        return None

    parsed = parse_ways(
        osm_data, ways_cfg, ctx.coordinate_system, ctx.target_scene_bounds
    )
    if not parsed:
        logging.info("No ways found in AOI — skipping")
        return None

    sidecar_path = ctx.data_dir / "ways.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(ways_to_sidecar(parsed), f)
    ctx.assets.ways_file = sidecar_path

    materials = sorted({w.material for w in parsed})
    logging.info(
        "Ways sidecar saved: %s (%d segments, %d materials: %s)",
        sidecar_path,
        len(parsed),
        len(materials),
        ", ".join(materials),
    )
    return sidecar_path
