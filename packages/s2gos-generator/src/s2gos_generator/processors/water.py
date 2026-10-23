"""Water bodies: OSM parsing, landcover completion, sidecar (de)serialization and terrain flattening."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

from .osm import load_osm_data, parse_osm_width
from .terrain_mesh import (
    GradientFilter,
    WaterFlattenOperation,
    WayFlattenOperation,
    compute_gradient,
    extract_dem,
)

LANDCOVER_WATER_CLASS = 80
LINEAR_WATERWAY_TYPES = {"river", "canal", "stream", "drain", "ditch", "weir"}
OUTLIER_MAD_MULTIPLIER = 4.0


def fetch_osm_data(
    water_cfg, bbox_south, bbox_west, bbox_north, bbox_east
) -> Optional[dict]:
    """Fetch or load OSM water data based on config source."""
    bbox = f"{bbox_south},{bbox_west},{bbox_north},{bbox_east}"
    query = (
        "[out:json];("
        f'way["natural"="water"]({bbox});'
        f'way["water"]({bbox});'
        f'way["waterway"]({bbox});'
        f'relation["natural"="water"]({bbox});'
        f'relation["water"]({bbox});'
        ");out geom;"
    )
    return load_osm_data(water_cfg, query, "water bodies")


@dataclass(slots=True)
class WaterBody:
    """A water body footprint in scene coordinates; linear waterways also carry ``centerline`` and ``half_width``."""

    geometry: Polygon
    material: str
    centerline: Optional[LineString] = None
    half_width: Optional[float] = None


def _polygon_parts(geom) -> list[Polygon]:
    """Non-empty Polygon parts of a possibly multi-part or collection geometry."""
    return [
        g
        for g in shapely.get_parts(shapely.get_parts(geom))
        if isinstance(g, Polygon) and not g.is_empty
    ]


def _way_geometry_scene(
    element: dict, coordinate_system
) -> Optional[list[tuple[float, float]]]:
    """Convert an OSM element's 'geometry' node list to scene-coordinate points."""
    coords = [
        coordinate_system.latlon_to_scene(node["lat"], node["lon"])
        for node in element.get("geometry") or []
        if node.get("lat") is not None and node.get("lon") is not None
    ]
    return coords if len(coords) >= 2 else None


def _rings_to_area(lines: list[LineString]):
    """Stitch open boundary arcs with linemerge and polygonize them into one area."""
    if not lines:
        return Polygon()
    return unary_union(
        [p.buffer(0) for p in polygonize(shapely.get_parts(linemerge(lines)))]
    )


def _is_water_relation(tags: dict, water_cfg) -> bool:
    return (tags.get("natural") == "water" or bool(tags.get("water"))) and not (
        water_cfg.exclude_intermittent and tags.get("intermittent") == "yes"
    )


def parse_water_bodies(
    osm_data: dict, water_cfg, coordinate_system, scene_bounds
) -> list[WaterBody]:
    """Parse OSM water elements into WaterBody footprints, clipped to scene bounds.

    Relation boundaries are stitched before polygonizing and inner rings
    (islands) become holes. Underground waterways are skipped.
    """
    elements = osm_data.get("elements", [])
    material = water_cfg.default_material
    bodies: list[WaterBody] = []
    relation_member_way_ids: set = set()

    for element in elements:
        if element.get("type") != "relation" or not _is_water_relation(
            element.get("tags", {}), water_cfg
        ):
            continue
        rings: dict[str, list[LineString]] = {"outer": [], "inner": []}
        for member in element.get("members", []):
            role = member.get("role")
            if role not in rings:
                continue
            if role == "outer" and member.get("type") == "way":
                relation_member_way_ids.add(member.get("ref"))
            coords = _way_geometry_scene(member, coordinate_system)
            if coords is not None:
                rings[role].append(LineString(coords))
        outer = _rings_to_area(rings["outer"])
        if not outer.is_empty:
            area = outer.difference(_rings_to_area(rings["inner"]))
            clipped = area.intersection(scene_bounds)
            bodies.extend(WaterBody(p, material) for p in _polygon_parts(clipped))

    for element in elements:
        if element.get("type") != "way" or element.get("id") in relation_member_way_ids:
            continue
        tags = element.get("tags", {})
        if water_cfg.exclude_intermittent and tags.get("intermittent") == "yes":
            continue
        if tags.get("tunnel", "no") != "no" or tags.get("location") == "underground":
            continue
        coords = _way_geometry_scene(element, coordinate_system)
        if coords is None:
            continue

        waterway_type = tags.get("waterway")
        is_areal = (
            tags.get("natural") == "water"
            or bool(tags.get("water"))
            or waterway_type == "riverbank"
        )
        if is_areal and coords[0] == coords[-1] and len(coords) >= 4:
            clipped = Polygon(coords).buffer(0).intersection(scene_bounds)
            bodies.extend(WaterBody(p, material) for p in _polygon_parts(clipped))
            continue

        clipped_cl = LineString(coords).intersection(scene_bounds)
        if clipped_cl.is_empty:
            continue

        if waterway_type in LINEAR_WATERWAY_TYPES:
            width = parse_osm_width(tags["width"]) if tags.get("width") else None
            half_width = (
                width / 2.0
                if width is not None
                else water_cfg.WATERWAY_HALF_WIDTH_M.get(
                    waterway_type, water_cfg.default_waterway_half_width_m
                )
            )
            for line in shapely.get_parts(clipped_cl):
                if line.is_empty or line.length == 0:
                    continue
                poly = line.buffer(half_width, cap_style="flat")
                if not poly.is_empty:
                    bodies.append(WaterBody(poly, material, line, half_width))
        elif is_areal:
            poly = clipped_cl.buffer(
                water_cfg.default_waterway_half_width_m, cap_style="flat"
            )
            if not poly.is_empty:
                bodies.append(WaterBody(poly, material))

    return bodies


def complete_with_landcover(
    osm_bodies: list[WaterBody], landcover_data, dem_data, water_cfg
) -> list[WaterBody]:
    """Add water bodies for flat landcover water areas OSM misses.

    Areas touching OSM water are only added above
    ``landcover_completion_linked_min_area_m2``; smaller ones are the landcover
    disagreeing with the OSM outline, larger ones are bodies OSM lacks (the sea).
    """
    from rasterio.features import rasterize
    from rasterio.features import shapes as raster_shapes
    from rasterio.transform import from_bounds
    from scipy.ndimage import binary_dilation, label

    from .terrain_texture.overlays import _lc_bounds

    landcover_data.load()
    lc_vals = landcover_data.values
    lc_water = lc_vals == LANDCOVER_WATER_CLASS
    if not lc_water.any():
        return []

    ny, nx = lc_vals.shape
    xmin, xmax, ymin, ymax, native_res = _lc_bounds(landcover_data)
    half_px = native_res / 2
    transform = from_bounds(
        xmin - half_px, ymin - half_px, xmax + half_px, ymax + half_px, nx, ny
    )

    osm_polys = [b.geometry for b in osm_bodies if not b.geometry.is_empty]
    osm_mask = np.zeros((ny, nx), dtype=bool)
    if osm_polys:
        osm_mask = (
            np.flipud(
                rasterize(
                    [(geom, 1) for geom in osm_polys],
                    out_shape=(ny, nx),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )
            )
            > 0
        )

    lc_not_osm = lc_water & ~osm_mask
    labeled_water, _ = label(lc_not_osm)
    linked = np.unique(labeled_water[binary_dilation(osm_mask & lc_water) & lc_not_osm])
    linked_area_m2 = np.bincount(labeled_water.ravel())[linked] * native_res**2
    small_linked = linked[
        (linked > 0)
        & (linked_area_m2 < water_cfg.landcover_completion_linked_min_area_m2)
    ]
    linked_drop = np.isin(labeled_water, small_linked)
    lc_not_osm &= ~linked_drop
    if not lc_not_osm.any():
        return []

    dem_x, dem_y, dem_elev = extract_dem(dem_data)
    slope_deg = np.degrees(np.arctan(compute_gradient(dem_elev, dem_x, dem_y)))
    if slope_deg.shape != lc_vals.shape:
        xx, yy = np.meshgrid(
            landcover_data.coords["x"].values, landcover_data.coords["y"].values
        )
        slope_deg = _sample_dem_points(
            dem_x, dem_y, slope_deg, np.column_stack((xx.ravel(), yy.ravel()))
        ).reshape(lc_vals.shape)

    flat = lc_not_osm & (slope_deg < water_cfg.landcover_completion_max_slope_deg)
    labeled, _ = label(flat)
    sizes = np.bincount(labeled.ravel())
    min_pixels = water_cfg.landcover_completion_min_area_m2 / native_res**2
    keep_labels = np.nonzero(sizes >= min_pixels)[0]
    keep_mask = np.isin(labeled, keep_labels[keep_labels > 0])

    px_km2 = native_res**2 / 1e6
    logging.info(
        "Landcover water outside OSM: %.3f km^2 added; dropped %.3f km^2 touching OSM, "
        "%.3f km^2 too steep, %.3f km^2 too small",
        keep_mask.sum() * px_km2,
        linked_drop.sum() * px_km2,
        (lc_not_osm & ~flat).sum() * px_km2,
        (flat & ~keep_mask).sum() * px_km2,
    )
    if not keep_mask.any():
        return []

    shapes_input = np.flipud(keep_mask).astype(np.uint8)
    return [
        WaterBody(poly, water_cfg.default_material)
        for geom_dict, value in raster_shapes(
            shapes_input, mask=shapes_input.astype(bool), transform=transform
        )
        if value == 1 and not (poly := shape(geom_dict).buffer(0)).is_empty
    ]


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
    """Reconstruct water bodies from the water-sidecar structure."""
    version = data.get("version", 1)
    if version != 1:
        logging.warning(
            "Unknown water sidecar version %s; skipping water bodies", version
        )
        return []

    return [
        WaterBody(
            geometry=shape(b["geometry"]),
            material=layer["material_name"],
            centerline=shape(b["centerline"]) if b.get("centerline") else None,
            half_width=b.get("half_width"),
        )
        for layer in data.get("water_layers", [])
        for b in layer.get("bodies", [])
    ]


def _interior_sample_points(
    geometry: Polygon, erosion_m: float, dem_x: np.ndarray, dem_y: np.ndarray
) -> np.ndarray:
    """DEM cell centres inside ``geometry`` eroded by ``erosion_m``, away from the banks.

    Falls back to the cells inside the uneroded geometry, then to one interior point.
    """
    eroded = geometry.buffer(-erosion_m) if erosion_m > 0 else geometry
    for target in (eroded, geometry):
        if target.is_empty:
            continue
        minx, miny, maxx, maxy = target.bounds
        xx, yy = np.meshgrid(
            dem_x[(dem_x >= minx) & (dem_x <= maxx)],
            dem_y[(dem_y >= miny) & (dem_y <= maxy)],
        )
        xx, yy = xx.ravel(), yy.ravel()
        inside = shapely.contains_xy(target, xx, yy)
        if inside.any():
            return np.column_stack([xx[inside], yy[inside]])
    rp = geometry.representative_point()
    return np.array([[rp.x, rp.y]])


def _group_areal_bodies_by_adjacency(
    areal_bodies: list[WaterBody], touch_tolerance_m: float = 1.0
) -> list[list[WaterBody]]:
    """Group areal water bodies into clusters of touching polygons."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(areal_bodies)
    if n == 0:
        return []
    geoms = [b.geometry for b in areal_bodies]
    i, j = shapely.STRtree(geoms).query(
        shapely.buffer(geoms, touch_tolerance_m), predicate="intersects"
    )
    i, j = i[j > i], j[j > i]
    adjacency = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    n_clusters, labels = connected_components(adjacency, directed=False)
    return [
        [areal_bodies[k] for k in np.nonzero(labels == c)[0]] for c in range(n_clusters)
    ]


def _sample_dem_points(dem_x, dem_y, dem_elev, xy: np.ndarray) -> np.ndarray:
    """Bilinear-sample the DEM at (M, 2) scene-XY points."""
    from scipy.ndimage import map_coordinates

    dx = (dem_x[-1] - dem_x[0]) / (len(dem_x) - 1)
    dy = (dem_y[-1] - dem_y[0]) / (len(dem_y) - 1)
    x_idx = (xy[:, 0] - dem_x[0]) / dx
    y_idx = (xy[:, 1] - dem_y[0]) / dy
    return map_coordinates(dem_elev, np.vstack((y_idx, x_idx)), order=1, mode="nearest")


def _odd_window(length_m: float, spacing_m: float) -> int:
    window = max(1, int(round(length_m / spacing_m)))
    return window + 1 if window % 2 == 0 else window


def _smoothed_centerline_profile(
    centerline: LineString,
    dem_x: np.ndarray,
    dem_y: np.ndarray,
    dem_elev: np.ndarray,
    sample_spacing_m: float,
    smooth_window_m: float,
    outlier_reject_m: float = 1.5,
):
    """Sample the DEM along a centerline and return ``(distances, smoothed_z)``, or None.

    Outliers from a wide rolling median are replaced by it before a rolling mean.
    """
    from scipy.ndimage import uniform_filter1d

    length = centerline.length
    if length <= 0:
        return None

    dists = np.linspace(0.0, length, max(2, int(length / sample_spacing_m) + 1))
    xy = shapely.get_coordinates(shapely.line_interpolate_point(centerline, dists))
    raw_z = _sample_dem_points(dem_x, dem_y, dem_elev, xy)

    finite = np.isfinite(raw_z)
    if not finite.any():
        return None
    if not finite.all():
        raw_z = np.interp(dists, dists[finite], raw_z[finite])

    cleaned = raw_z
    detection_window = _odd_window(
        max(smooth_window_m * 3.0, smooth_window_m + 200.0), sample_spacing_m
    )
    if outlier_reject_m > 0 and 3 <= detection_window <= len(raw_z):
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(raw_z, detection_window // 2, mode="reflect"), detection_window
        )
        trend = np.median(windows, axis=1)
        local_mad = np.median(np.abs(windows - trend[:, None]), axis=1) * 1.4826
        reject_m = np.maximum(outlier_reject_m, OUTLIER_MAD_MULTIPLIER * local_mad)
        cleaned = np.where(np.abs(raw_z - trend) > reject_m, trend, raw_z)

    window = _odd_window(smooth_window_m, sample_spacing_m)
    return dists, uniform_filter1d(cleaned, size=window, mode="mirror")


def smooth_dem_along_linear_bodies(
    dem_data,
    linear_bodies: list[WaterBody],
    *,
    smooth_window_m: float = 150.0,
    sample_spacing_m: float = 20.0,
    margin_m: float = 10.0,
    outlier_reject_m: float = 1.5,
):
    """Return ``dem_data`` with elevation along linear waterways replaced by a smoothed centerline profile.

    Overlapping bands are blended with weights falling linearly from the centerline.
    """
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

        band = (body.half_width or 0.0) + effective_margin_m
        minx, miny, maxx, maxy = cl.buffer(band).bounds
        xi0 = max(0, int(np.searchsorted(dem_x, minx)) - 1)
        xi1 = min(nx, int(np.searchsorted(dem_x, maxx)) + 2)
        yi0 = max(0, int(np.searchsorted(dem_y, miny)) - 1)
        yi1 = min(ny, int(np.searchsorted(dem_y, maxy)) + 2)
        if xi1 <= xi0 or yi1 <= yi0:
            continue

        sub_x = xx[yi0:yi1, xi0:xi1]
        sub_pts = shapely.points(sub_x.ravel(), yy[yi0:yi1, xi0:xi1].ravel())
        dist_to_cl = shapely.distance(cl, sub_pts).reshape(sub_x.shape)
        mask = dist_to_cl <= band
        if not mask.any():
            continue

        proj = shapely.line_locate_point(cl, sub_pts).reshape(sub_x.shape)
        band_smoothed_z = np.zeros(sub_x.shape)
        band_smoothed_z[mask] = np.interp(proj[mask], dists, smoothed_z)
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


def _sloped_surface(
    geom, dem_x, dem_y, dem_elev, *, smoothing_m, tolerance_m, max_deviation_m
):
    """Water surface over ``geom`` as an (M, 2) XY -> z sampler, or None when the body is level.

    The DEM is Gaussian-averaged over water pixels only, so banks don't bleed
    in. The body is level when that average, at three times ``smoothing_m``,
    varies by at most ``tolerance_m``. Otherwise the surface is the average at
    ``smoothing_m``, clamped to ``max_deviation_m`` of the DEM and extended past
    the shoreline from the nearest water pixel.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    res = max(abs(dem_x[1] - dem_x[0]), abs(dem_y[1] - dem_y[0]))
    pad = 3 * 3 * smoothing_m + res
    minx, miny, maxx, maxy = geom.bounds
    xs = np.nonzero((dem_x >= minx - pad) & (dem_x <= maxx + pad))[0]
    ys = np.nonzero((dem_y >= miny - pad) & (dem_y <= maxy + pad))[0]
    if len(xs) < 2 or len(ys) < 2:
        return None
    cx, cy = dem_x[xs[0] : xs[-1] + 1], dem_y[ys[0] : ys[-1] + 1]
    z = dem_elev[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
    xx, yy = np.meshgrid(cx, cy)

    mask = shapely.contains_xy(geom, xx, yy) & np.isfinite(z)
    if not mask.any():
        mask = shapely.contains_xy(geom.buffer(res), xx, yy) & np.isfinite(z)
    if not mask.any():
        return None

    weights = mask.astype(np.float64)
    masked_z = np.where(mask, z, 0.0)

    def water_average(scale_m: float) -> np.ndarray:
        sigma = scale_m / res
        return gaussian_filter(masked_z, sigma) / np.maximum(
            gaussian_filter(weights, sigma), 1e-12
        )

    level = water_average(3 * smoothing_m)[mask]
    if np.percentile(level, 95) - np.percentile(level, 5) <= tolerance_m:
        return None

    surface = np.clip(
        water_average(smoothing_m), z - max_deviation_m, z + max_deviation_m
    )
    _, nearest = distance_transform_edt(~mask, return_indices=True)
    surface = surface[tuple(nearest)]
    return lambda xy: _sample_dem_points(cx, cy, surface, xy)


def build_water_terraform_operations(
    water_bodies: list[WaterBody],
    dem_data,
    *,
    transition_buffer_m: float,
    gradient_threshold: float = 0.0,
    thin_water_skip_m: float = 0.0,
    erosion_margin_m: float = 10.0,
    outlier_reject_m: float = 1.5,
    dem_resolution_m: float = 30.0,
    shore_flat_margin_m: float = 0.0,
    sloped_tolerance_m: float = 3.0,
    sloped_smoothing_m: float = 50.0,
    sloped_max_deviation_m: float = 5.0,
):
    """Build terrain-flatten operations for water bodies.

    Linear bodies get a :class:`WayFlattenOperation`; areal bodies get one
    :class:`WaterFlattenOperation` per cluster of touching bodies, extended
    ``shore_flat_margin_m`` past the shoreline. A level cluster is flat at the
    robust median of interior DEM samples, a sloped one follows
    :func:`_sloped_surface`.
    """
    if not water_bodies:
        logging.warning("No water bodies in sidecar — skipping water flattening")
        return []

    dem_x, dem_y, dem_elev = extract_dem(dem_data)
    effective_erosion_m = max(erosion_margin_m, 1.5 * dem_resolution_m)
    gf = GradientFilter(dem_elev, dem_x, dem_y) if gradient_threshold > 0.0 else None

    def keep(width_m: float, line) -> bool:
        if thin_water_skip_m > 0 and width_m < thin_water_skip_m:
            return False
        return gf is None or gf.exceeds_threshold(line, gradient_threshold)

    operations: list = []
    areal_bodies: list[WaterBody] = []
    for body in water_bodies:
        if body.centerline is not None and body.half_width is not None:
            if keep(body.half_width * 2, body.centerline):
                operations.append(
                    WayFlattenOperation(
                        body.centerline, body.half_width, transition_buffer_m
                    )
                )
        elif not body.geometry.is_empty:
            areal_bodies.append(body)

    for cluster in _group_areal_bodies_by_adjacency(areal_bodies):
        geom = (
            cluster[0].geometry
            if len(cluster) == 1
            else unary_union([b.geometry for b in cluster])
        )
        if not keep(2 * geom.area / geom.length, geom.boundary):
            continue

        samples = _sample_dem_points(
            dem_x,
            dem_y,
            dem_elev,
            _interior_sample_points(geom, effective_erosion_m, dem_x, dem_y),
        )
        samples = samples[np.isfinite(samples)]
        reference_z = 0.0
        if samples.size:
            reference_z = float(np.median(samples))
            inliers = samples[np.abs(samples - reference_z) <= outlier_reject_m]
            if outlier_reject_m > 0 and inliers.size:
                reference_z = float(np.median(inliers))

        surface = _sloped_surface(
            geom,
            dem_x,
            dem_y,
            dem_elev,
            smoothing_m=sloped_smoothing_m,
            tolerance_m=sloped_tolerance_m,
            max_deviation_m=sloped_max_deviation_m,
        )
        operations.append(
            WaterFlattenOperation(
                geom.buffer(shore_flat_margin_m) if shore_flat_margin_m > 0 else geom,
                transition_buffer_m,
                reference_z,
                surface,
            )
        )

    logging.info(
        "Built %d water terraform operation(s) from %d water body/bodies",
        len(operations),
        len(water_bodies),
    )
    return operations
