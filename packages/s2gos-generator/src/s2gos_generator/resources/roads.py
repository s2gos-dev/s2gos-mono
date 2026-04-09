"""Road infrastructure resource — fetches OSM road data and produces road polygons."""

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from shapely.geometry import LineString, MultiPolygon, box, mapping
from shapely.ops import unary_union

from ..core.context import SceneResourceContext

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAX_RETRIES = 3
OVERPASS_RETRY_DELAY_S = 30


def _fetch_overpass(bbox_south, bbox_west, bbox_north, bbox_east) -> Optional[dict]:
    """Fetch road data from Overpass API with retry on 429/5xx errors."""
    query = (
        f'[out:json];way["highway"]'
        f"({bbox_south},{bbox_west},{bbox_north},{bbox_east});"
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    for attempt in range(1, OVERPASS_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 504) and attempt < OVERPASS_MAX_RETRIES:
                logging.warning(
                    "Overpass API returned %s, retrying in %ds (attempt %d/%d)",
                    exc.code,
                    OVERPASS_RETRY_DELAY_S,
                    attempt,
                    OVERPASS_MAX_RETRIES,
                )
                time.sleep(OVERPASS_RETRY_DELAY_S)
            else:
                logging.warning("Overpass API request failed: %s", exc)
                return None
        except Exception as exc:
            logging.warning("Overpass API request failed: %s", exc)
            return None
    return None


def _parse_osm_width(width_str: str) -> Optional[float]:
    """Parse OSM width tag value. Handles '5', '5.5', '5 m', '5.5m' formats."""
    s = width_str.strip().lower()

    if s.endswith("m"):
        s = s[:-1].strip()

    try:
        return float(s)
    except ValueError:
        return None


def _get_road_width(tags: dict, hw_type: str, roads_cfg) -> float:
    """Determine road width checking user overrides, OSM tags, and defaults."""

    # 1. Total width override
    if hw_type in roads_cfg.total_width_overrides:
        return roads_cfg.total_width_overrides[hw_type]

    # 2. OSM Tag Explicit Width
    osm_width_str = tags.get("width")
    if osm_width_str:
        osm_width = _parse_osm_width(osm_width_str)
        if osm_width is not None:
            return osm_width

    # 3. Calculated Lane-based Width
    lanes_str = tags.get("lanes")
    if lanes_str is not None and lanes_str.isdigit():
        lanes = int(lanes_str)
    else:
        fallback_lanes = 1 if tags.get("oneway") == "yes" else 2
        lanes = roads_cfg.lane_count_mapping.get(hw_type, fallback_lanes)

    lane_width = roads_cfg.lane_width_mapping.get(
        hw_type, roads_cfg.default_lane_width_m
    )

    return lanes * lane_width + 2 * roads_cfg.default_shoulder_m


def _get_road_material(tags: dict, roads_cfg) -> str:
    """Determine road material from OSM surface tag, falling back to default."""
    surface_tag = tags.get("surface")
    if not surface_tag:
        return roads_cfg.default_material

    return roads_cfg.surface_material_mapping.get(
        surface_tag, roads_cfg.default_material
    )


def _parse_roads(
    osm_data: dict,
    roads_cfg,
    coordinate_system,
    scene_bounds,
) -> dict[str, list]:
    """Parse OSM road ways into buffered polygons, grouped by material."""
    elements = osm_data.get("elements", [])
    roads_by_material: dict[str, list] = {}

    for element in elements:
        if element.get("type") != "way":
            continue

        tags = element.get("tags", {})
        hw_type = tags.get("highway")

        if hw_type is None:
            continue
        if (
            roads_cfg.highway_types is not None
            and hw_type not in roads_cfg.highway_types
        ):
            continue

        geometry = element.get("geometry")
        if not geometry or len(geometry) < 2:
            continue

        scene_coords = []
        for node in geometry:
            lat, lon = node.get("lat"), node.get("lon")
            if lat is None or lon is None:
                continue
            x, y = coordinate_system.latlon_to_scene(lat, lon)
            scene_coords.append((x, y))

        if len(scene_coords) < 2:
            continue

        width = _get_road_width(tags, hw_type, roads_cfg)
        road_poly = LineString(scene_coords).buffer(width / 2, cap_style="flat")

        clipped = road_poly.intersection(scene_bounds)
        if clipped.is_empty:
            continue

        material = _get_road_material(tags, roads_cfg)

        if material not in roads_by_material:
            roads_by_material[material] = []

        roads_by_material[material].append(clipped)

    return roads_by_material


def _merge_polygons(polygons: list) -> list:
    """Merge overlapping polygons into a minimal set."""
    merged = unary_union(polygons)
    if isinstance(merged, MultiPolygon):
        return list(merged.geoms)
    return [merged]


def _fetch_osm_data(
    roads_cfg, bbox_south, bbox_west, bbox_north, bbox_east
) -> Optional[dict]:
    """Fetch or load OSM road data based on config source."""
    if roads_cfg.source == "overpass":
        logging.info(
            "Fetching roads from Overpass API: bbox=(%.4f, %.4f, %.4f, %.4f)",
            bbox_south,
            bbox_west,
            bbox_north,
            bbox_east,
        )
        return _fetch_overpass(bbox_south, bbox_west, bbox_north, bbox_east)

    if roads_cfg.source == "file":
        logging.info("Loading roads from file: %s", roads_cfg.file_path)
        try:
            with open(roads_cfg.file_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logging.warning("Failed to load road data file: %s", exc)
            return None

    logging.warning("Unknown road data source: %s", roads_cfg.source)
    return None


def process_target_roads(ctx: SceneResourceContext) -> Optional[Path]:
    """Fetch/load road data, convert to polygons, save sidecar JSON."""
    config = ctx.config
    roads_cfg = config.roads

    if roads_cfg is None or not roads_cfg.enabled:
        return None

    logging.info("=== Processing Roads ===")

    bbox_west, bbox_south, bbox_east, bbox_north = ctx.target_aoi_polygon.bounds

    osm_data = _fetch_osm_data(roads_cfg, bbox_south, bbox_west, bbox_north, bbox_east)
    if osm_data is None:
        logging.warning("No road data available — skipping roads")
        return None

    half_size_m = (config.location.aoi_size_km * 1000) / 2
    scene_bounds = box(-half_size_m, -half_size_m, half_size_m, half_size_m)

    roads_by_material = _parse_roads(
        osm_data,
        roads_cfg,
        ctx.coordinate_system,
        scene_bounds,
    )

    if not roads_by_material:
        logging.info("No roads found in AOI — skipping")
        return None

    merged_by_material = {}
    for material_name in sorted(roads_by_material):
        raw = roads_by_material[material_name]
        merged = _merge_polygons(raw)
        merged_by_material[material_name] = merged
        logging.info(
            "Roads [%s]: %d raw polygons → %d merged",
            material_name,
            len(raw),
            len(merged),
        )

    sidecar = {
        "road_layers": [
            {
                "material_name": name,
                "polygons": [mapping(p) for p in polys],
            }
            for name, polys in sorted(merged_by_material.items())
        ]
    }
    sidecar_path = ctx.data_dir / "roads.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(sidecar, f)

    ctx.assets.roads_file = sidecar_path

    total = sum(len(p) for p in merged_by_material.values())
    logging.info(
        "Roads sidecar saved: %s (%d polygons, %d materials)",
        sidecar_path,
        total,
        len(merged_by_material),
    )
    return sidecar_path
