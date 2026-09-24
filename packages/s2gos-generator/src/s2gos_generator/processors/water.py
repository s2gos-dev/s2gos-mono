"""Water body algorithms: OSM/Overpass fetching, parsing, landcover completion, sidecar (de)serialization, and terrain-flatten operation building."""

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

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

from .terrain_mesh import (
    GradientFilter,
    WaterFlattenOperation,
    WayFlattenOperation,
    compute_gradient,
    extract_dem,
)
from .._version import get_version

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MAX_RETRIES = 5
OVERPASS_RETRY_DELAY_S = 15
OVERPASS_RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# ESA WorldCover class for "Permanent water bodies".
LANDCOVER_WATER_CLASS = 80

# Linear waterway types treated as centerlines to buffer; natural=water/water=*/waterway=riverbank are areal instead.
LINEAR_WATERWAY_TYPES = {"river", "canal", "stream", "drain", "ditch", "weir"}

# MAD multiplier scaling centerline-smoothing outlier rejection to local terrain variability.
OUTLIER_MAD_MULTIPLIER = 4.0


class WaterFetchError(RuntimeError):
    """Raised when the Overpass API fails on every retry attempt for water data."""


def _is_transient_urlerror(exc: urllib.error.URLError) -> bool:
    """Return True if a URLError wraps a known transient transport failure."""
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (ConnectionResetError, socket.timeout, ssl.SSLEOFError))


def _fetch_overpass_water(
    bbox_south, bbox_west, bbox_north, bbox_east
) -> Optional[dict]:
    """Fetch water body data from Overpass API with retry on transient failures."""
    query = (
        "[out:json];"
        "("
        f'way["natural"="water"]({bbox_south},{bbox_west},{bbox_north},{bbox_east});'
        f'way["water"]({bbox_south},{bbox_west},{bbox_north},{bbox_east});'
        f'way["waterway"]({bbox_south},{bbox_west},{bbox_north},{bbox_east});'
        f'relation["natural"="water"]({bbox_south},{bbox_west},{bbox_north},{bbox_east});'
        f'relation["water"]({bbox_south},{bbox_west},{bbox_north},{bbox_east});'
        ");"
        "out geom;"
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
                raise WaterFetchError(
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
                raise WaterFetchError(
                    f"Overpass API request failed ({type(exc).__name__}): {exc}"
                )


def fetch_osm_data(
    water_cfg, bbox_south, bbox_west, bbox_north, bbox_east
) -> Optional[dict]:
    """Fetch or load OSM water data based on config source."""
    if water_cfg.source == "overpass":
        logging.info(
            "Fetching water bodies from Overpass API: bbox=(%.4f, %.4f, %.4f, %.4f)",
            bbox_south,
            bbox_west,
            bbox_north,
            bbox_east,
        )
        return _fetch_overpass_water(bbox_south, bbox_west, bbox_north, bbox_east)

    if water_cfg.source == "file":
        logging.info("Loading water data from file: %s", water_cfg.file_path)
        try:
            with open(water_cfg.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
            logging.warning("Failed to load water data file: %s", exc)
            return None

    logging.warning("Unknown water data source: %s", water_cfg.source)
    return None


def _parse_width_m(width_str: str) -> Optional[float]:
    """Parse an OSM width tag value. Handles '5', '5.5', '5 m', '5.5m' formats."""
    s = width_str.strip().lower()
    if s.endswith("m"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None


def _waterway_half_width(tags: dict, waterway_type: str, water_cfg) -> float:
    """Resolve a linear waterway's half-width: OSM width tag -> type table -> fallback."""
    width_str = tags.get("width")
    if width_str:
        width = _parse_width_m(width_str)
        if width is not None:
            return width / 2.0

    defaults = water_cfg.WATERWAY_TYPE_TABLE.get(waterway_type)
    if defaults is not None:
        return defaults.half_width_m

    return water_cfg.default_waterway_half_width_m


@dataclass(slots=True)
class WaterBody:
    """A single water body footprint in scene coordinates."""

    geometry: Polygon
    material: str
    anchor_xy: Optional[tuple[float, float]] = None
    # Set for linear-waterway-derived bodies (centerline midpoint); None for areal bodies.
    centerline: Optional[LineString] = None
    # Set only for linear bodies -- these flatten per-vertex via WayFlattenOperation; areal bodies use one WaterFlattenOperation reference elevation instead.
    half_width: Optional[float] = None


def _way_geometry_scene(
    element: dict, coordinate_system
) -> Optional[list[tuple[float, float]]]:
    """Convert an OSM element's 'geometry' node list to scene-coordinate points."""
    geometry = element.get("geometry")
    if not geometry or len(geometry) < 2:
        return None
    coords = []
    for node in geometry:
        lat, lon = node.get("lat"), node.get("lon")
        if lat is None or lon is None:
            continue
        x, y = coordinate_system.latlon_to_scene(lat, lon)
        coords.append((x, y))
    return coords if len(coords) >= 2 else None


def _split_polygon_result(geom, out: list[Polygon]) -> None:
    """Append Polygon components of a (possibly Multi/collection) clip result."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        out.append(geom)
    elif isinstance(geom, MultiPolygon):
        out.extend(g for g in geom.geoms if not g.is_empty)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            _split_polygon_result(g, out)
    # Stray Point/LineString slivers from clipping are silently ignored, same as parse_ways.


def parse_water_bodies(
    osm_data: dict, water_cfg, coordinate_system, scene_bounds
) -> list[WaterBody]:
    """Parse OSM water elements into WaterBody footprints, clipped to scene bounds."""
    elements = osm_data.get("elements", [])
    bodies: list[WaterBody] = []
    default_material = water_cfg.default_material

    # Ways that are outer members of a relation we assemble below are not
    # also emitted standalone (they'd double the polygon).
    relation_member_way_ids: set = set()
    for element in elements:
        if element.get("type") != "relation":
            continue
        tags = element.get("tags", {})
        if tags.get("natural") != "water" and not tags.get("water"):
            continue
        if water_cfg.exclude_intermittent and tags.get("intermittent") == "yes":
            continue
        for member in element.get("members", []):
            if member.get("type") == "way" and member.get("role") == "outer":
                relation_member_way_ids.add(member.get("ref"))

    # --- relations: assembled polygons (e.g. large lakes, sea) ---
    for element in elements:
        if element.get("type") != "relation":
            continue
        tags = element.get("tags", {})
        if tags.get("natural") != "water" and not tags.get("water"):
            continue
        if water_cfg.exclude_intermittent and tags.get("intermittent") == "yes":
            continue

        # Large water features often split their outer boundary across many way arcs, so stitch via linemerge then polygonize rather than force-closing each arc independently (which fabricates bogus wedge polygons).
        outer_lines = []
        for member in element.get("members", []):
            if member.get("role") != "outer" or not member.get("geometry"):
                continue
            coords = _way_geometry_scene(member, coordinate_system)
            if coords is None or len(coords) < 2:
                continue
            outer_lines.append(LineString(coords))

        if not outer_lines:
            continue

        stitched = linemerge(outer_lines)
        stitched_lines = (
            list(stitched.geoms)
            if stitched.geom_type == "MultiLineString"
            else [stitched]
        )
        outer_polys = [poly.buffer(0) for poly in polygonize(stitched_lines)]
        outer_polys = [poly for poly in outer_polys if not poly.is_empty]

        if not outer_polys:
            continue

        merged = unary_union(outer_polys)
        clipped = merged.intersection(scene_bounds)
        components: list[Polygon] = []
        _split_polygon_result(clipped, components)
        # One WaterBody per relation, unless clipping genuinely split it into disjoint pieces.
        for part in components:
            bodies.append(WaterBody(geometry=part, material=default_material))

    # --- ways ---
    for element in elements:
        if element.get("type") != "way":
            continue
        if element.get("id") in relation_member_way_ids:
            continue

        tags = element.get("tags", {})
        if water_cfg.exclude_intermittent and tags.get("intermittent") == "yes":
            continue
        coords = _way_geometry_scene(element, coordinate_system)
        if coords is None:
            continue

        is_closed = coords[0] == coords[-1] and len(coords) >= 4
        natural_water = tags.get("natural") == "water" or bool(tags.get("water"))
        is_riverbank = tags.get("waterway") == "riverbank"
        waterway_type = tags.get("waterway")

        if is_closed and (natural_water or is_riverbank):
            poly = Polygon(coords).buffer(0)
            if poly.is_empty:
                continue
            clipped = poly.intersection(scene_bounds)
            components: list[Polygon] = []
            _split_polygon_result(clipped, components)
            for part in components:
                bodies.append(WaterBody(geometry=part, material=default_material))
            continue

        if waterway_type and waterway_type in LINEAR_WATERWAY_TYPES | set(
            water_cfg.WATERWAY_TYPE_TABLE
        ):
            centerline = LineString(coords)
            clipped_cl = centerline.intersection(scene_bounds)
            if clipped_cl.is_empty:
                continue
            half_width = _waterway_half_width(tags, waterway_type, water_cfg)

            clipped_lines = (
                list(clipped_cl.geoms)
                if clipped_cl.geom_type == "MultiLineString"
                else [clipped_cl]
            )
            for line in clipped_lines:
                if line.is_empty or line.length == 0:
                    continue
                poly = line.buffer(half_width, cap_style="flat")
                if poly.is_empty:
                    continue
                mid = line.interpolate(0.5, normalized=True)
                bodies.append(
                    WaterBody(
                        geometry=poly,
                        material=default_material,
                        anchor_xy=(mid.x, mid.y),
                        centerline=line,
                        half_width=half_width,
                    )
                )
            continue

        # natural=water/riverbank on a non-closed way (malformed/edge-case
        # OSM data) -- buffer minimally rather than silently dropping it.
        if natural_water or is_riverbank:
            centerline = LineString(coords)
            clipped_cl = centerline.intersection(scene_bounds)
            if clipped_cl.is_empty:
                continue
            poly = clipped_cl.buffer(
                water_cfg.default_waterway_half_width_m, cap_style="flat"
            )
            if not poly.is_empty:
                bodies.append(WaterBody(geometry=poly, material=default_material))

    return bodies


def complete_with_landcover(
    osm_bodies: list[WaterBody], landcover_data, dem_data, water_cfg
) -> list[WaterBody]:
    """Add whole water bodies for landcover water-class components OSM entirely misses, never patching an OSM body's boundary."""
    from rasterio.features import rasterize
    from rasterio.features import shapes as raster_shapes
    from rasterio.transform import from_bounds
    from scipy.ndimage import label, map_coordinates

    from .terrain_texture.overlays import _lc_bounds

    landcover_data.load()
    lc_vals = landcover_data.values
    ny, nx = lc_vals.shape
    xmin, xmax, ymin, ymax, native_res = _lc_bounds(landcover_data)
    half_px = native_res / 2
    transform = from_bounds(
        xmin - half_px, ymin - half_px, xmax + half_px, ymax + half_px, nx, ny
    )

    lc_water = lc_vals == LANDCOVER_WATER_CLASS
    if not lc_water.any():
        return []

    osm_polys = [b.geometry for b in osm_bodies if not b.geometry.is_empty]
    if osm_polys:
        osm_raster = rasterize(
            [(geom, 1) for geom in osm_polys],
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        # rasterize() is north-up row-0; flip to match lc_vals' row order,
        # same convention as terrain_texture/overlays.py::apply_ways.
        osm_mask = np.flipud(osm_raster) > 0
    else:
        osm_mask = np.zeros((ny, nx), dtype=bool)

    lc_not_osm = lc_water & ~osm_mask
    if not lc_not_osm.any():
        return []

    dem_data.load()
    dem_x, dem_y, dem_elev = extract_dem(dem_data)
    gradient = compute_gradient(dem_elev, dem_x, dem_y)
    slope_deg = np.degrees(np.arctan(gradient))

    if slope_deg.shape != lc_vals.shape:
        lc_x = landcover_data.coords["x"].values
        lc_y = landcover_data.coords["y"].values
        dx = (dem_x[-1] - dem_x[0]) / (len(dem_x) - 1)
        dy = (dem_y[-1] - dem_y[0]) / (len(dem_y) - 1)
        xi = (lc_x - dem_x[0]) / dx
        yi = (lc_y - dem_y[0]) / dy
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        slope_deg = map_coordinates(
            slope_deg,
            np.vstack([yi_grid.ravel(), xi_grid.ravel()]),
            order=1,
            mode="nearest",
        ).reshape(lc_vals.shape)

    flat = lc_not_osm & (slope_deg < water_cfg.landcover_completion_max_slope_deg)
    if not flat.any():
        return []

    labeled, n_components = label(flat)
    pixel_area_m2 = native_res**2
    min_pixels = (
        water_cfg.landcover_completion_min_area_m2 / pixel_area_m2
        if pixel_area_m2 > 0
        else 0.0
    )

    sizes = np.bincount(labeled.ravel())
    keep_labels = [i for i in range(1, n_components + 1) if sizes[i] >= min_pixels]
    if not keep_labels:
        return []
    keep_mask = np.isin(labeled, keep_labels)

    # Undo the row-flip convention before vectorizing back to the same
    # north-up frame the (unflipped) `transform` describes.
    shapes_input = np.flipud(keep_mask).astype(np.uint8)

    new_bodies: list[WaterBody] = []
    for geom_dict, value in raster_shapes(
        shapes_input, mask=shapes_input.astype(bool), transform=transform
    ):
        if value != 1:
            continue
        poly = shape(geom_dict).buffer(0)
        if poly.is_empty:
            continue
        new_bodies.append(WaterBody(geometry=poly, material=water_cfg.default_material))

    if new_bodies:
        logging.info(
            "Landcover completion: %d additional water body/bodies from %d connected component(s) OSM missed",
            len(new_bodies),
            n_components,
        )
    return new_bodies


def water_bodies_to_sidecar(bodies: list[WaterBody]) -> dict:
    """Serialize water bodies to the water-sidecar structure."""
    bodies_by_material: dict[str, list[WaterBody]] = {}
    for body in bodies:
        bodies_by_material.setdefault(body.material, []).append(body)

    return {
        "version": 1,
        "water_layers": [
            {
                "material_name": material,
                "bodies": [
                    {
                        "geometry": mapping(b.geometry),
                        "anchor_xy": list(b.anchor_xy)
                        if b.anchor_xy is not None
                        else None,
                        "centerline": mapping(b.centerline)
                        if b.centerline is not None
                        else None,
                        "half_width": b.half_width,
                    }
                    for b in body_list
                ],
            }
            for material, body_list in sorted(bodies_by_material.items())
        ],
    }


def water_bodies_from_sidecar(data: dict) -> list[WaterBody]:
    """Reconstruct water bodies from the water-sidecar structure, returning an empty list for an unrecognised schema version."""
    version = data.get("version", 1)
    if version != 1:
        logging.warning(
            "Unknown water sidecar version %s; skipping water bodies", version
        )
        return []

    bodies: list[WaterBody] = []
    for layer in data.get("water_layers", []):
        material = layer["material_name"]
        for b in layer.get("bodies", []):
            anchor = b.get("anchor_xy")
            centerline = b.get("centerline")
            bodies.append(
                WaterBody(
                    geometry=shape(b["geometry"]),
                    material=material,
                    anchor_xy=tuple(anchor) if anchor is not None else None,
                    centerline=shape(centerline) if centerline is not None else None,
                    half_width=b.get("half_width"),
                )
            )
    return bodies


def _largest_polygon(geom) -> Optional[Polygon]:
    """Return the largest-area Polygon component of geom, or None if empty."""
    if geom is None or geom.is_empty:
        return None
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return geom


def _interior_sample_points(
    geometry: Polygon, erosion_m: float, n_points: int = 9
) -> np.ndarray:
    """Return candidate interior XY samples for reference-elevation estimation, eroding inward to avoid DEM bank-elevation bleed."""
    eroded = geometry.buffer(-erosion_m) if erosion_m > 0 else geometry
    target = _largest_polygon(eroded)
    if target is None:
        rp = geometry.representative_point()
        return np.array([[rp.x, rp.y]])

    minx, miny, maxx, maxy = target.bounds
    if maxx <= minx or maxy <= miny:
        rp = target.representative_point()
        return np.array([[rp.x, rp.y]])

    rng = np.random.default_rng(0)
    xs = rng.uniform(minx, maxx, size=n_points * 4)
    ys = rng.uniform(miny, maxy, size=n_points * 4)
    pts = shapely.points(xs, ys)
    inside = shapely.contains(target, pts)
    xy = np.column_stack([xs[inside], ys[inside]])
    if len(xy) == 0:
        rp = target.representative_point()
        return np.array([[rp.x, rp.y]])
    return xy[:n_points]


def _group_areal_bodies_by_adjacency(
    areal_bodies: list[WaterBody], touch_tolerance_m: float = 1.0
) -> list[list[WaterBody]]:
    """Group areal water bodies into clusters of touching/near-touching polygons so each cluster shares one flat reference elevation."""
    n = len(areal_bodies)
    if n == 0:
        return []

    geoms = [b.geometry for b in areal_bodies]
    tree = shapely.STRtree(geoms)
    buffered = [g.buffer(touch_tolerance_m) for g in geoms]

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, buf in enumerate(buffered):
        for j in tree.query(buf):
            j = int(j)
            if j <= i:
                continue
            if buf.intersects(geoms[j]):
                union(i, j)

    clusters: dict[int, list[WaterBody]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(areal_bodies[i])
    return list(clusters.values())


def _sample_dem_points(dem_x, dem_y, dem_elev, xy: np.ndarray) -> np.ndarray:
    """Bilinear-sample the DEM at (M, 2) scene-XY points."""
    from scipy.ndimage import map_coordinates

    dx = (dem_x[-1] - dem_x[0]) / (len(dem_x) - 1)
    dy = (dem_y[-1] - dem_y[0]) / (len(dem_y) - 1)
    x_idx = (xy[:, 0] - dem_x[0]) / dx
    y_idx = (xy[:, 1] - dem_y[0]) / dy
    return map_coordinates(dem_elev, np.vstack((y_idx, x_idx)), order=1, mode="nearest")


def _rolling_median(z: np.ndarray, window: int) -> np.ndarray:
    """Rolling median with reflect-padding, so an anomaly at an endpoint can't dominate its own window."""
    if window < 3 or len(z) < window:
        return z.copy()
    half = window // 2
    padded = np.pad(z, half, mode="reflect")
    return np.array([np.median(padded[i : i + window]) for i in range(len(z))])


def _rolling_mad(z: np.ndarray, window: int) -> np.ndarray:
    """Rolling median absolute deviation, reflect-padded and scaled to a normal-consistent spread estimate."""
    if window < 3 or len(z) < window:
        return np.zeros_like(z)
    half = window // 2
    padded = np.pad(z, half, mode="reflect")
    out = np.empty(len(z))
    for i in range(len(z)):
        w = padded[i : i + window]
        out[i] = np.median(np.abs(w - np.median(w)))
    return out * 1.4826


def _smoothed_centerline_profile(
    centerline: LineString,
    dem_x: np.ndarray,
    dem_y: np.ndarray,
    dem_elev: np.ndarray,
    sample_spacing_m: float,
    smooth_window_m: float,
    outlier_reject_m: float = 1.5,
):
    """Sample the DEM along a centerline and robustly smooth it: reject outliers against a MAD-scaled trend, then apply a rolling-mean low-pass, returning (distances, smoothed_z) or None if degenerate."""
    from scipy.ndimage import uniform_filter1d

    length = centerline.length
    if length <= 0:
        return None

    n_samples = max(2, int(length / sample_spacing_m) + 1)
    dists = np.linspace(0.0, length, n_samples)
    pts = [centerline.interpolate(d) for d in dists]
    xy = np.array([[p.x, p.y] for p in pts])
    raw_z = _sample_dem_points(dem_x, dem_y, dem_elev, xy)

    finite = np.isfinite(raw_z)
    if not finite.any():
        return None
    if not finite.all():
        raw_z = np.interp(dists, dists[finite], raw_z[finite])

    window = max(1, int(round(smooth_window_m / sample_spacing_m)))
    if window % 2 == 0:
        window += 1

    if outlier_reject_m > 0:
        # Detection uses a much wider window than the final smoothing pass, so a localized anomaly stays a small minority share of it.
        detection_window_m = max(smooth_window_m * 3.0, smooth_window_m + 200.0)
        detection_window = max(1, int(round(detection_window_m / sample_spacing_m)))
        if detection_window % 2 == 0:
            detection_window += 1
        trend = _rolling_median(raw_z, detection_window)
        local_mad = _rolling_mad(raw_z, detection_window)
        effective_reject_m = np.maximum(
            outlier_reject_m, OUTLIER_MAD_MULTIPLIER * local_mad
        )
        outlier = np.abs(raw_z - trend) > effective_reject_m
        cleaned = np.where(outlier, trend, raw_z)
    else:
        cleaned = raw_z
    # A rolling mean (not median) gives a continuous low-pass curve instead of a staircase; mode="mirror" matches _rolling_median's own padding.
    smoothed_z = uniform_filter1d(cleaned, size=window, mode="mirror")

    return dists, smoothed_z


def smooth_dem_along_linear_bodies(
    dem_data,
    linear_bodies: list[WaterBody],
    *,
    smooth_window_m: float = 150.0,
    sample_spacing_m: float = 20.0,
    margin_m: float = 10.0,
    outlier_reject_m: float = 1.5,
):
    """Return a copy of ``dem_data`` with elevation near linear water-body centerlines replaced by a robust, smoothed profile so raw DEM noise doesn't bake into the flattened channel; ``smooth_window_m`` <= 0 disables it."""
    if smooth_window_m <= 0 or not linear_bodies:
        return dem_data

    dem_x, dem_y, dem_elev = extract_dem(dem_data)
    pixel_pitch = (
        max(abs(dem_x[1] - dem_x[0]), abs(dem_y[1] - dem_y[0]))
        if len(dem_x) > 1 and len(dem_y) > 1
        else 0.0
    )
    effective_margin_m = max(margin_m, 2.0 * pixel_pitch)
    weighted_sum = np.zeros_like(dem_elev)
    weight_sum = np.zeros_like(dem_elev)

    xx, yy = np.meshgrid(dem_x, dem_y)
    ny, nx = dem_elev.shape

    for body in linear_bodies:
        cl = body.centerline
        if cl is None or cl.length <= 0:
            continue
        half_width = body.half_width or 0.0

        profile = _smoothed_centerline_profile(
            cl,
            dem_x,
            dem_y,
            dem_elev,
            sample_spacing_m,
            smooth_window_m,
            outlier_reject_m,
        )
        if profile is None:
            continue
        dists, smoothed_z = profile

        band = half_width + effective_margin_m
        minx, miny, maxx, maxy = cl.buffer(band).bounds
        xi0 = max(0, int(np.searchsorted(dem_x, minx)) - 1)
        xi1 = min(nx, int(np.searchsorted(dem_x, maxx)) + 2)
        yi0 = max(0, int(np.searchsorted(dem_y, miny)) - 1)
        yi1 = min(ny, int(np.searchsorted(dem_y, maxy)) + 2)
        if xi1 <= xi0 or yi1 <= yi0:
            continue

        sub_x = xx[yi0:yi1, xi0:xi1]
        sub_y = yy[yi0:yi1, xi0:xi1]
        sub_pts = shapely.points(sub_x.ravel(), sub_y.ravel())
        dist_to_cl = shapely.distance(cl, sub_pts).reshape(sub_x.shape)
        mask = dist_to_cl <= band
        if not mask.any():
            continue

        proj = shapely.line_locate_point(cl, sub_pts).reshape(sub_x.shape)
        band_smoothed_z = np.zeros(sub_x.shape)
        band_smoothed_z[mask] = np.interp(proj[mask], dists, smoothed_z)

        # Linear falloff weight: 1.0 at the centerline, ~0 at the band edge.
        w = np.zeros(sub_x.shape)
        w[mask] = np.clip(1.0 - dist_to_cl[mask] / band, 1e-3, 1.0)

        weighted_sum[yi0:yi1, xi0:xi1] += band_smoothed_z * w
        weight_sum[yi0:yi1, xi0:xi1] += w

    touched = weight_sum > 0
    if not touched.any():
        return dem_data

    smoothed_elev = dem_elev.copy()
    smoothed_elev[touched] = weighted_sum[touched] / weight_sum[touched]

    return dem_data.copy(data=smoothed_elev)


def build_water_terraform_operations(
    water_bodies: list[WaterBody],
    dem_data,
    *,
    transition_buffer_m: float,
    gradient_threshold: float = 0.0,
    thin_water_skip_m: float = 0.0,
    erosion_margin_m: float = 10.0,
    outlier_reject_m: float = 1.5,
    default_sea_level_m: float = 0.0,
    dem_resolution_m: float = 30.0,
    drop_m: float = 0.0,
):
    """Build terrain-flatten operations for water bodies: linear bodies get a per-vertex WayFlattenOperation (flat across, varying along the channel), areal bodies get one shared WaterFlattenOperation reference elevation per touching cluster; returns an empty list when there are none."""
    if not water_bodies:
        logging.warning("No water bodies in sidecar — skipping water flattening")
        return []

    dem_x, dem_y, dem_elev = extract_dem(dem_data)
    effective_erosion_m = max(erosion_margin_m, 1.5 * dem_resolution_m)

    gf = GradientFilter(dem_elev, dem_x, dem_y) if gradient_threshold > 0.0 else None

    operations: list = []
    n_filtered = 0
    n_merged_clusters = 0
    areal_bodies: list[WaterBody] = []

    for body in water_bodies:
        if body.centerline is not None and body.half_width is not None:
            if thin_water_skip_m > 0 and body.half_width * 2 < thin_water_skip_m:
                n_filtered += 1
                continue
            if gf is not None and not gf.exceeds_threshold(
                body.centerline, gradient_threshold
            ):
                n_filtered += 1
                continue
            operations.append(
                WayFlattenOperation(
                    body.centerline, body.half_width, transition_buffer_m
                )
            )
            continue

        if body.geometry.is_empty:
            continue
        areal_bodies.append(body)

    for cluster in _group_areal_bodies_by_adjacency(areal_bodies):
        geom = (
            cluster[0].geometry
            if len(cluster) == 1
            else unary_union([b.geometry for b in cluster])
        )
        if len(cluster) > 1:
            n_merged_clusters += 1

        minx, miny, maxx, maxy = geom.bounds
        width_m = min(maxx - minx, maxy - miny)
        if thin_water_skip_m > 0 and width_m < thin_water_skip_m:
            n_filtered += len(cluster)
            continue

        if gf is not None and not gf.exceeds_threshold(
            geom.boundary, gradient_threshold
        ):
            n_filtered += len(cluster)
            continue

        single_anchor = cluster[0].anchor_xy if len(cluster) == 1 else None
        sample_xy = (
            np.array([single_anchor])
            if single_anchor is not None
            else _interior_sample_points(geom, effective_erosion_m)
        )

        samples = _sample_dem_points(dem_x, dem_y, dem_elev, sample_xy)
        finite_mask = np.isfinite(samples)

        if finite_mask.any():
            finite_idx = np.nonzero(finite_mask)[0]
            finite_samples = samples[finite_idx]
            median = float(np.median(finite_samples))
            if outlier_reject_m > 0:
                keep = np.abs(finite_samples - median) <= outlier_reject_m
                if keep.any():
                    median = float(np.median(finite_samples[keep]))
            reference_z = median - drop_m
            best_local = int(np.argmin(np.abs(finite_samples - median)))
            anchor_xy = tuple(sample_xy[finite_idx[best_local]])
        else:
            reference_z = default_sea_level_m - drop_m
            anchor_xy = tuple(sample_xy[0])

        operations.append(
            WaterFlattenOperation(
                polygon=geom,
                buffer_m=transition_buffer_m,
                anchor_xy=anchor_xy,
                reference_z=reference_z,
            )
        )

    if n_merged_clusters:
        logging.info(
            "Water adjacency merge: %d cluster(s) of touching areal bodies "
            "flattened to one shared reference elevation each",
            n_merged_clusters,
        )
    if n_filtered:
        logging.info(
            "Water gradient/width filter: %d/%d body/bodies skipped",
            n_filtered,
            len(water_bodies),
        )
    if operations:
        logging.info(
            "Built %d water terraform operation(s) from %d water body/bodies",
            len(operations),
            len(water_bodies),
        )
    return operations
