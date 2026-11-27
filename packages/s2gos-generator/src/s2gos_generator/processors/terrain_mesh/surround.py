"""Buffer mesh that surrounds the target without overlapping it."""

import numpy as np
import trimesh
import xarray as xr

from .builder import extract_dem, grid_faces, make_elevation_fn


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    """Evenly spaced samples from lo to hi inclusive, at most step apart."""
    return np.linspace(lo, hi, max(int(np.ceil((hi - lo) / step)), 1) + 1)


def _boundary(vertices: np.ndarray, axis: int, value: float, along: int):
    """Sorted positions along one target edge and their elevations."""
    on_edge = np.isclose(vertices[:, axis], value, rtol=0.0, atol=1e-6)
    t, idx = np.unique(vertices[on_edge, along], return_index=True)
    return t, vertices[on_edge][idx, 2]


def _stitch(ia: np.ndarray, ta: np.ndarray, ib: np.ndarray, tb: np.ndarray):
    """Triangulate between two vertex rows that span the same parameter range."""
    faces = []
    i = j = 0
    while i < len(ia) - 1 or j < len(ib) - 1:
        if j == len(ib) - 1 or (i < len(ia) - 1 and ta[i + 1] <= tb[j + 1]):
            faces.append((ia[i], ib[j], ia[i + 1]))
            i += 1
        else:
            faces.append((ia[i], ib[j], ib[j + 1]))
            j += 1
    return faces


def _strip(inner_t, inner_z, regular_t, across, to_xy, elevation_fn):
    """One strip: an irregular inner row at ``across[0]`` and regular rows beyond."""
    t, s = np.meshgrid(regular_t, across[1:])
    grid_xy = to_xy(t.ravel(), s.ravel())
    inner_xy = to_xy(inner_t, np.full_like(inner_t, across[0]))

    vertices = np.vstack(
        [
            np.column_stack([inner_xy, inner_z]),
            np.column_stack([grid_xy, elevation_fn(grid_xy)]),
        ]
    )
    k = len(inner_t)
    seam = _stitch(np.arange(k), inner_t, k + np.arange(len(regular_t)), regular_t)
    faces = np.vstack([np.array(seam), k + grid_faces(len(regular_t), len(across) - 1)])
    return vertices, faces


def _orient_up(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip faces to point up and drop those with zero area in plan view."""
    a, b, c = (vertices[faces[:, i], :2] for i in range(3))
    u, v = b - a, c - a
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    faces = faces[cross != 0]
    flip = cross[cross != 0] < 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


def _assemble(parts, bounds, handle_nans: bool) -> trimesh.Trimesh:
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    vertices = np.vstack(vertices)
    faces = _orient_up(vertices, np.vstack(faces))

    if handle_nans:
        valid = ~np.isnan(vertices[:, 2])
        faces = faces[np.all(valid[faces], axis=1)]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    xmin, xmax, ymin, ymax = bounds
    uv = (mesh.vertices[:, :2] - [xmin, ymin]) / [xmax - xmin, ymax - ymin]
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.clip(uv, 0.0, 1.0))
    return mesh


def build_buffer_strips_mesh(
    dem_data: xr.DataArray,
    target_vertices: np.ndarray,
    handle_nans: bool = True,
) -> trimesh.Trimesh:
    """Mesh the buffer DEM as four strips sharing the target boundary vertices.

    The north and south strips span the full buffer width and cover the corners,
    while the east and west strips span only the height of the target.

    Args:
        dem_data: Buffer DEM, covering the target and extending past it on all sides.
        target_vertices: Vertices of the target mesh as saved. Boundary vertices are
            copied unchanged so the two meshes share their seam.
        handle_nans: Drop faces with a NaN elevation.

    Returns:
        Upward-facing mesh with UVs spanning the buffer DEM's pixel edges.

    Raises:
        ValueError: If the buffer DEM does not extend past the target on every side.
    """
    x, y, elev = extract_dem(dem_data)
    elevation_fn = make_elevation_fn(x, y, elev)
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    step = min(dx, dy)

    bounds = outer_x0, outer_x1, outer_y0, outer_y1 = (
        x.min() - dx / 2,
        x.max() + dx / 2,
        y.min() - dy / 2,
        y.max() + dy / 2,
    )
    inner_x0, inner_y0 = target_vertices[:, :2].min(axis=0)
    inner_x1, inner_y1 = target_vertices[:, :2].max(axis=0)
    if not (outer_x0 < inner_x0 < inner_x1 < outer_x1) or not (
        outer_y0 < inner_y0 < inner_y1 < outer_y1
    ):
        raise ValueError("Buffer DEM must extend beyond the target mesh on all sides")

    xs_w = _axis(outer_x0, inner_x0, step)
    xs_mid = _axis(inner_x0, inner_x1, step)
    xs_e = _axis(inner_x1, outer_x1, step)
    ys_s = _axis(outer_y0, inner_y0, step)
    ys_mid = _axis(inner_y0, inner_y1, step)
    ys_n = _axis(inner_y1, outer_y1, step)
    full_x = np.concatenate([xs_w[:-1], xs_mid, xs_e[1:]])

    def horizontal(t, s):
        return np.column_stack([t, s])

    def vertical(t, s):
        return np.column_stack([s, t])

    parts = []
    for y_edge, across in ((inner_y1, ys_n), (inner_y0, ys_s[::-1])):
        edge_t, edge_z = _boundary(target_vertices, 1, y_edge, 0)
        wing_w = horizontal(xs_w[:-1], np.full(len(xs_w) - 1, y_edge))
        wing_e = horizontal(xs_e[1:], np.full(len(xs_e) - 1, y_edge))
        inner_t = np.concatenate([xs_w[:-1], edge_t, xs_e[1:]])
        inner_z = np.concatenate([elevation_fn(wing_w), edge_z, elevation_fn(wing_e)])
        parts.append(_strip(inner_t, inner_z, full_x, across, horizontal, elevation_fn))

    for x_edge, across in ((inner_x1, xs_e), (inner_x0, xs_w[::-1])):
        edge_t, edge_z = _boundary(target_vertices, 0, x_edge, 1)
        parts.append(_strip(edge_t, edge_z, ys_mid, across, vertical, elevation_fn))

    return _assemble(parts, bounds, handle_nans)
