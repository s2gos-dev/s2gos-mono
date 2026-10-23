"""Way and railways algorithms: OSM/Overpass fetching, parsing, width/material resolution,
sidecar (de)serialization, and terrain-flatten operation building."""

import logging
from dataclasses import dataclass
from typing import Optional

from shapely.geometry import LineString, MultiLineString, mapping, shape

from .osm import load_osm_data, parse_osm_width
from .terrain_mesh import GradientFilter, extract_dem


def _get_road_width(
    tags: dict,
    hw_type: str,
    road_overrides: dict,
    road_type_table: dict,
    default_lane_width_m: float,
    default_shoulder_m: float,
) -> float:
    """Determine road width checking user overrides, OSM tags, and defaults.

    Resolution order: override total_width -> OSM width tag -> lane-based calculation.
    """
    override = road_overrides.get(hw_type)

    if override is not None and override.total_width_m is not None:
        return override.total_width_m

    osm_width_str = tags.get("width")
    if osm_width_str:
        osm_width = parse_osm_width(osm_width_str)
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


def _get_railway_width(
    tags: dict,
    rail_type: str,
    railway_overrides: dict,
    railway_type_table: dict,
    default_track_width_m: float,
) -> float:
    """Determine railway corridor width from overrides, OSM tags, or defaults.
    Resolution order: override total_width -> OSM width tag -> track-count based default.
    """
    override = railway_overrides.get(rail_type)
    if override is not None and override.total_width_m is not None:
        return override.total_width_m

    osm_width_str = tags.get("width")
    if osm_width_str:
        osm_width = parse_osm_width(osm_width_str)
        if osm_width is not None:
            return osm_width

    tracks_str = tags.get("tracks")
    rail_defaults = railway_type_table.get(rail_type)
    if override is not None and override.track_count is not None:
        tracks = override.track_count
    elif rail_defaults is not None:
        tracks = rail_defaults.track_count
    else:
        tracks = 1
    if tracks_str is not None:
        try:
            tracks = int(tracks_str)
        except ValueError:
            logging.debug(
                "Ignoring non-integer OSM tracks tag %r on railway=%s",
                tracks_str,
                rail_type,
            )

    if override is not None and override.track_width_m is not None:
        track_width = override.track_width_m
    else:
        rail_defaults = railway_type_table.get(rail_type)
        track_width = (
            rail_defaults.track_width_m
            if rail_defaults is not None
            else default_track_width_m
        )
    return tracks * track_width


def _get_road_material(
    tags: dict,
    hw_type: str,
    road_overrides: dict,
    road_type_table: dict,
    default_surface_materials: dict,
    default_material: str,
) -> str:
    """Determine road material from OSM surface tag, with per-road-type fallback.

    Resolution order: OSM surface tag -> override default_material -> type-table default.
    """
    surface_tag = tags.get("surface")
    if surface_tag:
        return default_surface_materials.get(surface_tag, default_material)

    override = road_overrides.get(hw_type)
    if override is not None and override.default_material is not None:
        return override.default_material

    hw_defaults = road_type_table.get(hw_type)
    if hw_defaults is not None:
        return hw_defaults.default_material

    return default_material


def _get_railway_material(
    tags: dict,
    rail_type: str,
    railway_overrides: dict,
    railway_type_table: dict,
    default_surface_materials: dict,
    default_railway_material: str,
) -> str:
    """Determine railway material from OSM surface tag, with per-railway-type fallback.

    Resolution order: OSM surface tag -> override default_material -> type-table default ->
    global default (`gravel_road`).
    """
    surface_tag = tags.get("surface")
    if surface_tag:
        return default_surface_materials.get(surface_tag, default_railway_material)

    override = railway_overrides.get(rail_type)
    if override is not None and override.default_material is not None:
        return override.default_material

    rail_defaults = railway_type_table.get(rail_type)
    if rail_defaults is not None:
        return rail_defaults.default_material

    return default_railway_material


@dataclass(slots=True)
class Way:
    """A single road or railway segment in scene coordinates."""

    centerline: LineString
    width: float  # meters, full way width
    material: str
    kind: str = "unknown"  # "road" or "railway"; "unknown" for pre-version-2 sidecars


def parse_ways(
    osm_data: dict,
    ways_cfg,
    coordinate_system,
    scene_bounds,
) -> list[Way]:
    """Parse OSM ways (roads and railways) into Way segments, clipped to scene bounds.

    Tunnels (any ``tunnel`` value other than "no") are skipped.
    """
    elements = osm_data.get("elements", [])
    ways: list[Way] = []

    # Config values pulled into locals before the per-element loop.
    road_overrides = ways_cfg.road_overrides
    road_type_table = ways_cfg.ROAD_TYPE_TABLE
    surface_materials = ways_cfg.DEFAULT_SURFACE_MATERIALS
    default_material = ways_cfg.default_material
    default_lane_width_m = ways_cfg.default_lane_width_m
    default_shoulder_m = ways_cfg.default_shoulder_m
    railway_overrides = ways_cfg.railway_overrides
    railway_type_table = ways_cfg.RAILWAY_TYPE_TABLE
    default_track_width_m = ways_cfg.default_track_width_m
    default_railway_material = ways_cfg.default_railway_material

    for element in elements:
        if element.get("type") != "way":
            continue

        tags = element.get("tags", {})
        hw_type = tags.get("highway")
        rail_type = tags.get("railway")
        if hw_type is None and rail_type is None:
            continue
        if tags.get("tunnel", "no") != "no":
            continue
        if hw_type is not None and (
            ways_cfg.road_types is not None and hw_type not in ways_cfg.road_types
        ):
            continue
        if rail_type is not None and (
            getattr(ways_cfg, "railway_types", None) is not None
            and rail_type not in ways_cfg.railway_types
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

        if hw_type is not None:
            kind = "road"
            width = _get_road_width(
                tags,
                hw_type,
                road_overrides,
                road_type_table,
                default_lane_width_m,
                default_shoulder_m,
            )
            material = _get_road_material(
                tags,
                hw_type,
                road_overrides,
                road_type_table,
                surface_materials,
                default_material,
            )
        else:
            kind = "railway"
            width = _get_railway_width(
                tags,
                rail_type,
                railway_overrides,
                railway_type_table,
                default_track_width_m,
            )
            material = _get_railway_material(
                tags,
                rail_type,
                railway_overrides,
                railway_type_table,
                surface_materials,
                default_railway_material,
            )
        # A way that re-enters the AOI after leaving produces a MultiLineString.
        # Decompose into one Way per component so WayFlattenOperation always
        # receives a plain LineString
        if isinstance(clipped_cl, MultiLineString):
            for component in clipped_cl.geoms:
                if not component.is_empty:
                    ways.append(
                        Way(
                            centerline=component,
                            width=width,
                            material=material,
                            kind=kind,
                        )
                    )
        else:
            ways.append(
                Way(centerline=clipped_cl, width=width, material=material, kind=kind)
            )

    return ways


def fetch_osm_data(
    ways_cfg, bbox_south, bbox_west, bbox_north, bbox_east
) -> Optional[dict]:
    """Fetch or load OSM way (road and railway) data based on config source."""
    bbox = f"{bbox_south},{bbox_west},{bbox_north},{bbox_east}"
    query = (
        f'[out:json];(way["highway"]({bbox});way["railway"]({bbox}););out body geom;'
    )
    return load_osm_data(ways_cfg, query, "ways")


def ways_to_sidecar(ways: list[Way]) -> dict:
    """Serialize way segments to the ways-sidecar structure."""
    ways_by_material: dict[str, list[Way]] = {}
    for way in ways:
        ways_by_material.setdefault(way.material, []).append(way)

    return {
        "version": 2,
        "way_layers": [
            {
                "material_name": material,
                "ways": [
                    {
                        "centerline": mapping(w.centerline),
                        "width": w.width,
                        "kind": w.kind,
                    }
                    for w in way_list
                ],
            }
            for material, way_list in sorted(ways_by_material.items())
        ],
    }


def ways_from_sidecar(data: dict) -> list[Way]:
    """Reconstruct way segments from the ways-sidecar structure.

    Version 1 sidecars predate the per-way ``kind`` field ("road"/"railway"); ways
    loaded from one get ``kind="unknown"``. Returns an empty list for an
    unrecognised schema version.
    """
    version = data.get("version", 1)
    if version not in (1, 2):
        logging.warning("Unknown way sidecar version %s; skipping ways", version)
        return []

    ways: list[Way] = []
    for layer in data.get("way_layers", []):
        material = layer["material_name"]
        for w in layer.get("ways", []):
            ways.append(
                Way(
                    centerline=shape(w["centerline"]),
                    width=w["width"],
                    material=material,
                    kind=w.get("kind", "unknown"),
                )
            )
    return ways


def build_way_terraform_operations(
    ways: list[Way],
    dem_data,
    *,
    transition_buffer_m: float,
    gradient_threshold: float,
    thin_way_skip_m: float,
) -> list:
    """Build terrain-flatten operations for ways using a DEM gradient filter.

    Returns an empty list when there are no way segments or all are filtered
    out by the gradient threshold.
    """
    if not ways:
        logging.warning("No way segments in sidecar — falling back to uniform mesh")
        return []

    centerlines = [w.centerline for w in ways]
    half_widths = [w.width / 2.0 for w in ways]

    x, y, elev = extract_dem(dem_data)
    gf = GradientFilter(elev, x, y)
    operations = gf.build_operations(
        centerlines,
        half_widths,
        transition_buffer_m,
        threshold=gradient_threshold,
        thin_way_skip_m=thin_way_skip_m,
    )

    if operations:
        logging.info(
            "Built %d terraform operation(s) from %d way segment(s)",
            len(operations),
            len(centerlines),
        )
    return operations
