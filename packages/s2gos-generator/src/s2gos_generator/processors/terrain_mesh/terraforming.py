from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .error_pyramid import DemErrorPyramid

import numpy as np
import shapely
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString, MultiLineString
from shapely.strtree import STRtree


@runtime_checkable
class TerraformOperation(Protocol):
    """A terrain modification applied to mesh vertices.

    Implementations must provide:
    - ``influence_zone``: the spatial region this operation may alter.
    - ``apply()``: modifies ``vertices`` in-place and returns the array.
    - ``apply_to_subset()``: same, but for a pre-filtered vertex subset (used
      by :func:`apply_way_flatten_batch` for the STRtree fast path).
    """

    @property
    def influence_zone(self) -> shapely.Geometry:
        """Region where this operation may modify vertices."""
        ...

    def apply(
        self,
        vertices: np.ndarray,
        elevation_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Apply terrain modification.

        Args:
            vertices: (N, 3) array of mesh vertices — may be modified in-place.
            elevation_fn: maps (M, 2) XY coordinates -> elevation array of length M.

        Returns:
            Modified vertices array (may be the same object).
        """
        ...

    def apply_to_subset(
        self,
        vertices: np.ndarray,
        vertex_indices: np.ndarray,
        inside_points,  # shapely geometry array
        elevation_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Apply modification for a pre-filtered subset of vertices (in-place).

        Called by :func:`apply_way_flatten_batch` to avoid re-creating shapely
        points per operation when many operations share the same vertex array.

        Args:
            vertices:       Full (N, 3) vertex array — modified in-place.
            vertex_indices: Indices into ``vertices`` that fall inside this
                            operation's influence zone.
            inside_points:  Shapely point array corresponding to ``vertex_indices``.
            elevation_fn:   maps (M, 2) XY coordinates -> elevation array of length M.
        """
        ...


class WayFlattenOperation:
    """Flatten terrain cross-slope under a single road segment.

    For each vertex inside the influence zone the elevation is blended toward
    the elevation of the nearest point on the road centerline:

    * Within ``half_width`` of the centerline -> fully flattened (alpha = 1).
    * Between ``half_width`` and ``half_width + buffer_m`` -> linearly blended.
    * Beyond that -> unchanged.
    """

    def __init__(
        self,
        centerline: LineString | MultiLineString,
        half_width: float,
        buffer_m: float,
    ) -> None:
        self._centerline = centerline
        self._half_width = half_width
        self._buffer_m = buffer_m
        self._influence_zone: shapely.Geometry = shapely.buffer(
            centerline, half_width + buffer_m, cap_style="flat"
        )
        # Prepare once at construction — prepare is idempotent and fast
        shapely.prepare(self._influence_zone)

    @property
    def influence_zone(self) -> shapely.Geometry:
        return self._influence_zone

    def apply(
        self,
        vertices: np.ndarray,
        elevation_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Apply flattening.  For single-op use; prefer :func:`apply_way_flatten_batch`
        when applying multiple operations to avoid recreating shapely points per road."""
        xy = vertices[:, :2]
        points = shapely.points(xy[:, 0], xy[:, 1])
        inside_mask = shapely.intersects(self._influence_zone, points)
        inside_indices = np.nonzero(inside_mask)[0]

        if inside_indices.size == 0:
            return vertices

        self.apply_to_subset(
            vertices, inside_indices, points[inside_indices], elevation_fn
        )
        return vertices

    def apply_to_subset(
        self,
        vertices: np.ndarray,
        vertex_indices: np.ndarray,
        inside_points,  # shapely geometry array
        elevation_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Apply flattening for a pre-filtered subset of vertices (in-place).

        Called by both :meth:`apply` and :func:`apply_way_flatten_batch`.
        """
        nearest_xy, alpha, mask_apply = self._geometry_and_alpha(
            vertex_indices, inside_points
        )
        if not mask_apply.any():
            return
        ref_z = elevation_fn(nearest_xy)
        apply_indices = vertex_indices[mask_apply]
        a = alpha[mask_apply]
        vertices[apply_indices, 2] = (
            a * ref_z[mask_apply] + (1.0 - a) * vertices[apply_indices, 2]
        )

    def _geometry_and_alpha(
        self,
        vertex_indices: np.ndarray,
        inside_points,  # shapely geometry array
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nearest-centerline XY coords, blend weights, and apply mask.

        Returns:
            nearest_xy:  (M, 2) XY coordinates of the nearest point on the
                         centerline for each inside vertex (M = len(vertex_indices)).
            alpha:       (M,) blend weight in [0, 1].
            mask_apply:  (M,) boolean; True where alpha > 0.
        """
        cl = self._centerline
        dist_to_cl = shapely.distance(cl, inside_points)
        proj_dist = shapely.line_locate_point(cl, inside_points)
        nearest_pts = shapely.line_interpolate_point(cl, proj_dist)
        nearest_xy = shapely.get_coordinates(nearest_pts)

        hw = self._half_width
        alpha = np.zeros(len(vertex_indices), dtype=np.float64)
        alpha[dist_to_cl <= hw] = 1.0

        if self._buffer_m > 0:
            mask_blend = (dist_to_cl > hw) & (dist_to_cl <= hw + self._buffer_m)
            alpha[mask_blend] = 1.0 - (dist_to_cl[mask_blend] - hw) / self._buffer_m

        return nearest_xy, alpha, alpha > 0


def apply_way_flatten_batch(
    vertices: np.ndarray,
    operations: list[WayFlattenOperation],
    elevation_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply road-flatten operations efficiently with a single batched elevation sample.

    Args:
        vertices:     (N, 3) mesh vertex array — modified in-place.
        operations:   List of :class:`WayFlattenOperation` to apply.
        elevation_fn: ``(M, 2) XY -> (M,) Z`` elevation sampler.

    Returns:
        The same ``vertices`` array (modified in-place).
    """
    if not operations:
        return vertices

    xy = vertices[:, :2]
    points = shapely.points(xy[:, 0], xy[:, 1])
    buf_tree = STRtree([op.influence_zone for op in operations])
    pt_indices, buf_indices = buf_tree.query(points, predicate="intersects")

    if len(buf_indices) == 0:
        return vertices

    order = np.argsort(buf_indices, kind="stable")
    buf_sorted = buf_indices[order]
    pt_sorted = pt_indices[order]
    unique_bufs, first_idx = np.unique(buf_sorted, return_index=True)
    split_pts = np.split(pt_sorted, first_idx[1:])

    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_nearest_xy: list[np.ndarray] = []

    for op_idx, vertex_idx_arr in zip(unique_bufs, split_pts):
        op = operations[int(op_idx)]
        inside_pts = points[vertex_idx_arr]
        nearest_xy, alpha, mask_apply = op._geometry_and_alpha(
            vertex_idx_arr, inside_pts
        )
        pending.append((vertex_idx_arr, alpha, mask_apply))
        all_nearest_xy.append(nearest_xy)

    concat_xy = np.concatenate(all_nearest_xy)
    all_ref_z = elevation_fn(concat_xy) if len(concat_xy) else np.empty(0)

    z_offset = 0
    for (vertex_idx_arr, alpha, mask_apply), nearest_xy in zip(pending, all_nearest_xy):
        n_pts = len(nearest_xy)
        ref_z = all_ref_z[z_offset : z_offset + n_pts]
        z_offset += n_pts

        if not mask_apply.any():
            continue
        apply_indices = vertex_idx_arr[mask_apply]
        a = alpha[mask_apply]
        vertices[apply_indices, 2] = (
            a * ref_z[mask_apply] + (1.0 - a) * vertices[apply_indices, 2]
        )

    return vertices


def make_refinement_predicate(
    influence_zone: shapely.Geometry,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return a vectorised predicate for :class:`AdaptiveGrid.refine`.

    Args:
        influence_zone: Merged polygon/multipolygon covering all road buffers.

    Returns:
        ``predicate(xmin, ymin, xmax, ymax) -> bool[N]``
    """
    poly = influence_zone
    shapely.prepare(poly)
    px0, py0, px1, py1 = poly.bounds

    def predicate(
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
    ) -> np.ndarray:
        # AABB pre-filter: reject cells entirely outside the polygon bounding box
        possible = ~((xmax < px0) | (xmin > px1) | (ymax < py0) | (ymin > py1))
        result = np.zeros(len(xmin), dtype=bool)
        if possible.any():
            idx = np.nonzero(possible)[0]
            result[idx] = shapely.intersects(
                poly,
                shapely.box(xmin[idx], ymin[idx], xmax[idx], ymax[idx]),
            )
        return result

    return predicate


def make_roughness_predicate(
    pyramid: "DemErrorPyramid",
    tolerance_m: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Predicate returning True for cells whose max plane-residual exceeds tolerance_m.

    Args:
        pyramid:     Precomputed DEM error pyramid.
        tolerance_m: Max allowed plane-residual in metres before a cell is
                     subdivided. Replaces the old peak-to-peak tolerance;
                     typical values are 0.1–1.0 m depending on scene scale.
    """

    def predicate(
        xmin: np.ndarray, ymin: np.ndarray, xmax: np.ndarray, ymax: np.ndarray
    ) -> np.ndarray:
        return pyramid.query(xmin, ymin, xmax, ymax, tolerance_m)

    return predicate


def compute_gradient(
    elev: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute DEM gradient magnitude in m/m.

    Args:
        elev: 2-D elevation array, shape (ny, nx).
        x:    1-D x coordinates, length nx.
        y:    1-D y coordinates, length ny.

    Returns:
        Gradient magnitude array with the same shape as ``elev``.
    """
    dx_dem = (x[-1] - x[0]) / (len(x) - 1)
    dy_dem = (y[-1] - y[0]) / (len(y) - 1)
    grad_y, grad_x = np.gradient(elev, dy_dem, dx_dem)
    return np.sqrt(grad_x**2 + grad_y**2)


class GradientFilter:
    """Filter road segments by DEM gradient magnitude along the centerline.

    Roads whose max gradient is below ``threshold`` are skipped (flat terrain,
    flattening would be a no-op)
    Args:
        elev: 2-D elevation array, shape (ny, nx).
        x:    1-D x coordinates, length nx.
        y:    1-D y coordinates, length ny.
    """

    def __init__(self, elev: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        self._grad_mag = compute_gradient(elev, x, y)
        self._dx_dem = (x[-1] - x[0]) / (len(x) - 1)
        self._dy_dem = (y[-1] - y[0]) / (len(y) - 1)
        self._x0_dem = float(x[0])
        self._y0_dem = float(y[0])

    def exceeds_threshold(self, centerline: shapely.Geometry, threshold: float) -> bool:
        """Return True if the max gradient magnitude along *centerline* ≥ *threshold*.

        A threshold of 0.0 always returns True (disables filtering).
        """
        if threshold <= 0.0:
            return True

        coords = shapely.get_coordinates(centerline)
        if len(coords) < 2:
            return False

        x_idx = (coords[:, 0] - self._x0_dem) / self._dx_dem
        y_idx = (coords[:, 1] - self._y0_dem) / self._dy_dem
        grad = map_coordinates(
            self._grad_mag, np.vstack((y_idx, x_idx)), order=1, mode="nearest"
        )
        return float(grad.max()) >= threshold

    def build_operations(
        self,
        centerlines: list,
        half_widths: list[float],
        buffer_m: float,
        threshold: float,
        thin_road_skip_m: float = 0.0,
    ) -> list[WayFlattenOperation]:
        """Build a :class:`WayFlattenOperation` for each segment above threshold.

        Args:
            centerlines: Per-road centerline geometries.
            half_widths: Per-road half-widths in metres.
            buffer_m:    Blend-zone width outside road edge.
            threshold:   Gradient threshold (m/m).  0.0 keeps all segments.
            thin_road_skip_m: If > 0, roads narrower than this width (m) are
                skipped (excluded from flattening). Set to 0.0 to disable.

        Returns:
            Filtered list of :class:`WayFlattenOperation` objects.
        """
        operations: list[WayFlattenOperation] = []
        n_filtered = 0

        for cl, hw in zip(centerlines, half_widths):
            too_thin = thin_road_skip_m > 0 and hw * 2 < thin_road_skip_m
            if too_thin or not self.exceeds_threshold(cl, threshold):
                n_filtered += 1
            else:
                operations.append(WayFlattenOperation(cl, hw, buffer_m))

        if n_filtered:
            logging.info(
                "Gradient filter: %d/%d road segment(s) skipped "
                "(max gradient < %.4f m/m)",
                n_filtered,
                len(centerlines),
                threshold,
            )

        return operations
