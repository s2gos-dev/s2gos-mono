"""Road algorithms: OSM/Overpass fetching, parsing, width/material resolution,
sidecar (de)serialization, and terrain-flatten operation building."""

import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

from shapely.geometry import LineString, MultiLineString, mapping, shape

from .terrain_mesh import GradientFilter, extract_dem
from .._version import get_version

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAX_RETRIES = 5
OVERPASS_RETRY_DELAY_S = 15
OVERPASS_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class RoadsFetchError(RuntimeError):
    """Raised when the Overpass API fails on every retry attempt."""


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
                raise RoadsFetchError(
                    f"Overpass API request failed ({type(exc).__name__}): {exc}"
                )
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
                raise RoadsFetchError(
                    f"Overpass API request failed ({type(exc).__name__}): {exc}"
                )


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
    highway_overrides: dict,
    road_type_table: dict,
    default_lane_width_m: float,
    default_shoulder_m: float,
) -> float:
    """Determine road width checking user overrides, OSM tags, and defaults.

    Resolution order: override total_width -> OSM width tag -> lane-based calculation.
    """
    override = highway_overrides.get(hw_type)

    if override is not None and override.total_width_m is not None:
        return override.total_width_m

    osm_width_str = tags.get("width")
    if osm_width_str:
        osm_width = _parse_osm_width(osm_width_str)
        if osm_width is not None:
            return osm_width

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
        hw_defaults = road_type_table.get(hw_type)
        lanes = (
            (
                override.lane_count
                if override is not None and override.lane_count is not None
                else None
            )
            or (hw_defaults.lane_count if hw_defaults is not None else None)
            or fallback_lanes
        )

    hw_defaults = road_type_table.get(hw_type)
    lane_width = (
        (
            override.lane_width_m
            if override is not None and override.lane_width_m is not None
            else None
        )
        or (hw_defaults.lane_width_m if hw_defaults is not None else None)
        or default_lane_width_m
    )

    return lanes * lane_width + 2 * default_shoulder_m


def _get_road_material(
    tags: dict,
    hw_type: str,
    highway_overrides: dict,
    road_type_table: dict,
    default_surface_materials: dict,
    default_material: str,
) -> str:
    """Determine road material from OSM surface tag, with per-highway-type fallback.

    Resolution order: OSM surface tag -> override default_material -> type-table default.
    """
    surface_tag = tags.get("surface")
    if surface_tag:
        return default_surface_materials.get(surface_tag, default_material)

    override = highway_overrides.get(hw_type)
    if override is not None and override.default_material is not None:
        return override.default_material

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


def parse_roads(
    osm_data: dict,
    roads_cfg,
    coordinate_system,
    scene_bounds,
) -> list[Road]:
    """Parse OSM road ways into Road segments, clipped to scene bounds."""
    elements = osm_data.get("elements", [])
    roads: list[Road] = []

    # Config values pulled into locals before the per-element loop.
    highway_overrides = roads_cfg.highway_overrides
    road_type_table = roads_cfg.ROAD_TYPE_TABLE
    surface_materials = roads_cfg.DEFAULT_SURFACE_MATERIALS
    default_material = roads_cfg.default_material
    default_lane_width_m = roads_cfg.default_lane_width_m
    default_shoulder_m = roads_cfg.default_shoulder_m

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
        clipped_cl = centerline.intersection(scene_bounds)
        if clipped_cl.is_empty:
            continue

        width = _get_road_width(
            tags,
            hw_type,
            highway_overrides,
            road_type_table,
            default_lane_width_m,
            default_shoulder_m,
        )

        material = _get_road_material(
            tags,
            hw_type,
            highway_overrides,
            road_type_table,
            surface_materials,
            default_material,
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


def fetch_osm_data(
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


def roads_to_sidecar(roads: list[Road]) -> dict:
    """Serialize road segments to the roads-sidecar structure."""
    roads_by_material: dict[str, list[Road]] = {}
    for road in roads:
        roads_by_material.setdefault(road.material, []).append(road)

    return {
        "version": 1,
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


def roads_from_sidecar(data: dict) -> list[Road]:
    """Reconstruct road segments from the roads-sidecar structure.

    Returns an empty list for an unrecognised schema version.
    """
    version = data.get("version", 1)
    if version != 1:
        logging.warning("Unknown road sidecar version %s; skipping roads", version)
        return []

    roads: list[Road] = []
    for layer in data.get("road_layers", []):
        material = layer["material_name"]
        for r in layer.get("roads", []):
            roads.append(
                Road(
                    centerline=shape(r["centerline"]),
                    width=r["width"],
                    material=material,
                )
            )
    return roads


def build_road_terraform_operations(
    roads: list[Road],
    dem_data,
    *,
    transition_buffer_m: float,
    gradient_threshold: float,
    thin_road_skip_m: float,
) -> list:
    """Build terrain-flatten operations for roads using a DEM gradient filter.

    Returns an empty list when there are no road segments or all are filtered
    out by the gradient threshold.
    """
    if not roads:
        logging.warning("No road segments in sidecar — falling back to uniform mesh")
        return []

    centerlines = [r.centerline for r in roads]
    half_widths = [r.width / 2.0 for r in roads]

    x, y, elev = extract_dem(dem_data)
    gf = GradientFilter(elev, x, y)
    operations = gf.build_operations(
        centerlines,
        half_widths,
        transition_buffer_m,
        threshold=gradient_threshold,
        thin_road_skip_m=thin_road_skip_m,
    )

    if operations:
        logging.info(
            "Built %d terraform operation(s) from %d road segment(s)",
            len(operations),
            len(centerlines),
        )
    return operations
