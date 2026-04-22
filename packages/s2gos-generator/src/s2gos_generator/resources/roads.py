"""Road infrastructure resource — fetches OSM road data and produces road polygons."""

import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, mapping

from .._version import get_version
from ..core.context import SceneResourceContext

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAX_RETRIES = 5
OVERPASS_RETRY_DELAY_S = 15
OVERPASS_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def _is_transient_urlerror(exc: urllib.error.URLError) -> bool:
    """Return True if a URLError wraps a known transient transport failure."""
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (ConnectionResetError, socket.timeout, ssl.SSLEOFError))


def _fetch_overpass(bbox_south, bbox_west, bbox_north, bbox_east) -> Optional[dict]:
    """Fetch road data from Overpass API with retry on transient failures."""
    query = (
        f'[out:json];way["highway"]'
        f"({bbox_south},{bbox_west},{bbox_north},{bbox_east});"
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    user_agent = f"s2gos-generator/{get_version()}"

    for attempt in range(1, OVERPASS_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OVERPASS_URL,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": user_agent,
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in OVERPASS_RETRYABLE_STATUS and attempt < OVERPASS_MAX_RETRIES:
                logging.info(
                    "Overpass API returned %s, retrying in %ds (attempt %d/%d)",
                    exc.code,
                    OVERPASS_RETRY_DELAY_S,
                    attempt,
                    OVERPASS_MAX_RETRIES,
                )
                time.sleep(OVERPASS_RETRY_DELAY_S)
            else:
                logging.warning(
                    "Overpass API request failed (%s): %s",
                    type(exc).__name__,
                    exc,
                )
                return None
        except urllib.error.URLError as exc:
            if _is_transient_urlerror(exc) and attempt < OVERPASS_MAX_RETRIES:
                logging.info(
                    "Overpass transport error %r, retrying in %ds (attempt %d/%d)",
                    exc.reason,
                    OVERPASS_RETRY_DELAY_S,
                    attempt,
                    OVERPASS_MAX_RETRIES,
                )
                time.sleep(OVERPASS_RETRY_DELAY_S)
            else:
                logging.warning(
                    "Overpass API request failed (%s): %s",
                    type(exc).__name__,
                    exc,
                )
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


def _get_road_width(
    tags: dict,
    hw_type: str,
    total_width_overrides: dict,
    lane_count_map: dict,
    lane_width_map: dict,
    default_lane_width_m: float,
    default_shoulder_m: float,
) -> float:
    """Determine road width checking user overrides, OSM tags, and defaults."""

    # 1. Total width override
    if hw_type in total_width_overrides:
        return total_width_overrides[hw_type]

    # 2. OSM Tag Explicit Width
    osm_width_str = tags.get("width")
    if osm_width_str:
        osm_width = _parse_osm_width(osm_width_str)
        if osm_width is not None:
            return osm_width

    # 3. Calculated Lane-based Width
    lanes_str = tags.get("lanes")
    lanes: Optional[int] = None
    if lanes_str is not None:
        try:
            lanes = int(lanes_str)
        except ValueError:
            logging.debug(
                "Ignoring non-integer OSM lanes tag %r on highway=%s",
                lanes_str,
                hw_type,
            )
    if lanes is None:
        fallback_lanes = 1 if tags.get("oneway") == "yes" else 2
        lanes = lane_count_map.get(hw_type, fallback_lanes)

    lane_width = lane_width_map.get(hw_type, default_lane_width_m)

    return lanes * lane_width + 2 * default_shoulder_m


def _get_road_material(
    tags: dict,
    hw_type: str,
    surface_material_map: dict,
    road_type_table: dict,
    default_material: str,
) -> str:
    """Determine road material from OSM surface tag, with per-highway-type fallback."""
    surface_tag = tags.get("surface")
    if surface_tag:
        return surface_material_map.get(surface_tag, default_material)
    hw_defaults = road_type_table.get(hw_type)
    if hw_defaults is not None:
        return hw_defaults.default_material
    return default_material


@dataclass(slots=True)
class Road:
    """A single road segment in scene coordinates."""

    centerline: LineString
    width: float  # meters, full road width
    material: str


def _parse_roads(
    osm_data: dict,
    roads_cfg,
    coordinate_system,
    scene_bounds,
) -> list[Road]:
    """Parse OSM road ways into Road segments, clipped to scene bounds."""
    elements = osm_data.get("elements", [])
    roads: list[Road] = []

    # Resolve once — avoids rebuilding merged dicts per road segment
    total_width_overrides = roads_cfg.total_width_overrides
    lane_count_map = roads_cfg.lane_count_mapping
    lane_width_map = roads_cfg.lane_width_mapping
    surface_material_map = roads_cfg.surface_material_mapping
    default_material = roads_cfg.default_material
    default_lane_width_m = roads_cfg.default_lane_width_m
    default_shoulder_m = roads_cfg.default_shoulder_m
    road_type_table = roads_cfg.ROAD_TYPE_TABLE

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

        centerline = LineString(scene_coords)
        width = _get_road_width(
            tags,
            hw_type,
            total_width_overrides,
            lane_count_map,
            lane_width_map,
            default_lane_width_m,
            default_shoulder_m,
        )
        road_poly = centerline.buffer(width / 2, cap_style="flat")

        clipped_poly = road_poly.intersection(scene_bounds)
        if clipped_poly.is_empty:
            continue

        if not isinstance(clipped_poly, (Polygon, MultiPolygon)):
            continue

        clipped_cl = centerline.intersection(scene_bounds)
        if clipped_cl.is_empty:
            clipped_cl = centerline

        material = _get_road_material(
            tags, hw_type, surface_material_map, road_type_table, default_material
        )

        # A road that re-enters the AOI after leaving produces a MultiLineString.
        # Decompose into one Road per component so RoadFlattenOperation always
        # receives a plain LineString
        if isinstance(clipped_cl, MultiLineString):
            for component in clipped_cl.geoms:
                if not component.is_empty:
                    roads.append(
                        Road(centerline=component, width=width, material=material)
                    )
        else:
            roads.append(Road(centerline=clipped_cl, width=width, material=material))

    return roads


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
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
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

    parsed = _parse_roads(
        osm_data,
        roads_cfg,
        ctx.coordinate_system,
        ctx.target_scene_bounds,
    )

    if not parsed:
        logging.info("No roads found in AOI — skipping")
        return None

    roads_by_material: dict[str, list[Road]] = {}
    for road in parsed:
        roads_by_material.setdefault(road.material, []).append(road)

    for material_name, road_list in sorted(roads_by_material.items()):
        logging.info("Roads [%s]: %d segment(s)", material_name, len(road_list))

    sidecar = {
        "road_layers": [
            {
                "material_name": material,
                "roads": [
                    {"centerline": mapping(r.centerline), "width": r.width}
                    for r in road_list
                ],
            }
            for material, road_list in sorted(roads_by_material.items())
        ],
    }
    sidecar_path = ctx.data_dir / "roads.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(sidecar, f)

    ctx.assets.roads_file = sidecar_path

    logging.info(
        "Roads sidecar saved: %s (%d segments, %d materials)",
        sidecar_path,
        len(parsed),
        len(roads_by_material),
    )
    return sidecar_path


def build_road_terraform_operations(
    ctx: SceneResourceContext,
    dem_data,
    refinement_cfg,
) -> list:
    """Build RoadFlattenOperation instances from road sidecar + DEM gradient filter.

    Returns an empty list if no road data is available or all segments are
    filtered out by the gradient threshold. Caller is responsible for checking
    that roads are enabled.
    """
    from ..assets.terraforming import GradientFilter
    from ..assets.terrain_mesh import _extract_dem

    roads_cfg = ctx.config.roads
    all_roads = ctx.roads

    if not all_roads:
        logging.warning("No road segments in sidecar — falling back to uniform mesh")
        return []

    centerlines = [r.centerline for r in all_roads]
    half_widths = [r.width / 2.0 for r in all_roads]

    x, y, elev = _extract_dem(dem_data)

    gf = GradientFilter(elev, x, y)
    operations = gf.build_operations(
        centerlines,
        half_widths,
        refinement_cfg.transition_buffer_m,
        threshold=roads_cfg.mesh_gradient_threshold,
        thin_road_bypass_m=roads_cfg.mesh_thin_road_bypass_m,
    )

    if operations:
        logging.info(
            "Built %d terraform operation(s) from %d road segment(s)",
            len(operations),
            len(centerlines),
        )

    return operations
