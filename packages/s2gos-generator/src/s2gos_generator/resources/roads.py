"""Road infrastructure resource — fetches OSM road data and produces road polygons."""

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from shapely.geometry import LineString, MultiPolygon, mapping
from shapely.ops import unary_union

from ..core.config.roads import DEFAULT_ROAD_WIDTH_FALLBACK, DEFAULT_ROAD_WIDTHS
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


def _parse_roads(
    osm_data: dict,
    highway_types: Optional[list[str]],
    width_overrides: dict[str, float],
    coordinate_system,
    scene_bounds_polygon,
) -> list:
    """Parse OSM elements into buffered road polygons in scene coordinates.

    Returns list of (polygon, highway_type) tuples.
    """
    elements = osm_data.get("elements", [])
    road_polygons = []

    for element in elements:
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        hw_type = tags.get("highway")
        if hw_type is None:
            continue

        if highway_types is not None and hw_type not in highway_types:
            continue

        geometry = element.get("geometry")
        if not geometry or len(geometry) < 2:
            continue

        # Convert lat/lon nodes to scene coordinates
        scene_coords = []
        for node in geometry:
            lat, lon = node.get("lat"), node.get("lon")
            if lat is None or lon is None:
                continue
            x, y = coordinate_system.latlon_to_scene(lat, lon)
            scene_coords.append((x, y))

        if len(scene_coords) < 2:
            continue

        line = LineString(scene_coords)

        # Determine road width
        width = width_overrides.get(
            hw_type, DEFAULT_ROAD_WIDTHS.get(hw_type, DEFAULT_ROAD_WIDTH_FALLBACK)
        )
        road_poly = line.buffer(width / 2, cap_style="flat")

        # Clip to scene bounds
        clipped = road_poly.intersection(scene_bounds_polygon)
        if clipped.is_empty:
            continue

        road_polygons.append(clipped)

    return road_polygons


def process_target_roads(ctx: SceneResourceContext) -> Optional[Path]:
    """Fetch/load road data, convert to polygons, save sidecar JSON.

    Args:
        ctx: Scene resource context

    Returns:
        Path to roads sidecar JSON, or None if roads disabled or no data
    """
    config = ctx.config
    roads_cfg = config.roads

    if roads_cfg is None or not roads_cfg.enabled:
        return None

    logging.info("=== Processing Roads ===")

    # Get AOI bounding box in WGS84
    aoi_polygon = ctx.target_aoi_polygon  # WGS84 Shapely polygon
    bounds = (
        aoi_polygon.bounds
    )  # (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)
    bbox_west, bbox_south, bbox_east, bbox_north = bounds

    # Fetch road data
    if roads_cfg.source == "overpass":
        logging.info(
            "Fetching roads from Overpass API: bbox=(%.4f, %.4f, %.4f, %.4f)",
            bbox_south,
            bbox_west,
            bbox_north,
            bbox_east,
        )
        osm_data = _fetch_overpass(bbox_south, bbox_west, bbox_north, bbox_east)
    elif roads_cfg.source == "file":
        logging.info("Loading roads from file: %s", roads_cfg.file_path)
        try:
            with open(roads_cfg.file_path, "r") as f:
                osm_data = json.load(f)
        except Exception as exc:
            logging.warning("Failed to load road data file: %s", exc)
            return None
    else:
        logging.warning("Unknown road data source: %s", roads_cfg.source)
        return None

    if osm_data is None:
        logging.warning("No road data available — skipping roads")
        return None

    # Build scene-coordinate bounding box for clipping
    half_size_m = (config.location.aoi_size_km * 1000) / 2
    from shapely.geometry import box

    scene_bounds_polygon = box(-half_size_m, -half_size_m, half_size_m, half_size_m)

    # Parse and buffer roads
    road_polygons = _parse_roads(
        osm_data,
        roads_cfg.highway_types,
        roads_cfg.width_overrides,
        ctx.coordinate_system,
        scene_bounds_polygon,
    )

    if not road_polygons:
        logging.info("No roads found in AOI — skipping")
        return None

    # Union overlapping road polygons
    merged = unary_union(road_polygons)

    # Normalize to list of polygons
    if isinstance(merged, MultiPolygon):
        polygon_list = list(merged.geoms)
    else:
        polygon_list = [merged]

    logging.info(
        "Roads: %d individual polygons merged into %d",
        len(road_polygons),
        len(polygon_list),
    )

    # Save sidecar JSON
    sidecar = {
        "material_name": roads_cfg.material_name,
        "road_count": len(polygon_list),
        "polygons": [mapping(p) for p in polygon_list],
    }
    sidecar_path = ctx.data_dir / "roads.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(sidecar, f)

    # Store on context
    ctx.assets.roads_file = sidecar_path
    ctx._road_geometries = polygon_list

    logging.info(
        "Roads sidecar saved: %s (%d polygons)", sidecar_path, len(polygon_list)
    )
    return sidecar_path
