"""Buffer mesh that surrounds the target without overlapping it."""

from __future__ import annotations

import numpy as np
import trimesh
import xarray as xr

from .builder import _make_elevation_fn, extract_dem


def _axis(lo: float, hi: float, step: float) -> np.ndarray:
    return np.linspace(lo, hi, max(int(np.ceil((hi - lo) / step)), 1) + 1)


def _boundary(vertices: np.ndarray, axis: int, value: float, along: int):
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


def _grid_faces(nx: int, ny: int) -> np.ndarray:
    q = np.arange(nx * ny).reshape(ny, nx)[:-1, :-1].ravel()
    return np.vstack(
        [
            np.column_stack([q, q + 1, q + nx + 1]),
            np.column_stack([q, q + nx + 1, q + nx]),
        ]
    )


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
    faces = np.vstack(
        [np.array(seam), k + _grid_faces(len(regular_t), len(across) - 1)]
    )
    return vertices, faces


def _orient_up(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
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

    xmin, xmax, ymin, ymax = bounds
    uv = (mesh.vertices[:, :2] - [xmin, ymin]) / [xmax - xmin, ymax - ymin]
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.clip(uv, 0.0, 1.0))
    return mesh


def build_buffer_strips_mesh(
    dem_data: xr.DataArray,
    target_vertices: np.ndarray,
    handle_nans: bool = True,
) -> trimesh.Trimesh:
    """Mesh the buffer DEM as four strips sharing the target boundary vertices."""
    x, y, elev = extract_dem(dem_data)
    elevation_fn = _make_elevation_fn(x, y, elev)
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    step = min(dx, dy)

    bounds = X0, X1, Y0, Y1 = (
        x.min() - dx / 2,
        x.max() + dx / 2,
        y.min() - dy / 2,
        y.max() + dy / 2,
    )
    x0, y0 = target_vertices[:, :2].min(axis=0)
    x1, y1 = target_vertices[:, :2].max(axis=0)
    if not (X0 < x0 and x1 < X1 and Y0 < y0 and y1 < Y1):
        raise ValueError("Buffer DEM must extend beyond the target mesh on all sides")

    xs_w, xs_mid, xs_e = _axis(X0, x0, step), _axis(x0, x1, step), _axis(x1, X1, step)
    ys_s, ys_mid, ys_n = _axis(Y0, y0, step), _axis(y0, y1, step), _axis(y1, Y1, step)
    full_x = np.concatenate([xs_w[:-1], xs_mid, xs_e[1:]])

    def horizontal(t, s):
        return np.column_stack([t, s])

    def vertical(t, s):
        return np.column_stack([s, t])

    parts = []
    for y_edge, across in ((y1, ys_n), (y0, ys_s[::-1])):
        edge_t, edge_z = _boundary(target_vertices, 1, y_edge, 0)
        wing_w = horizontal(xs_w[:-1], np.full(len(xs_w) - 1, y_edge))
        wing_e = horizontal(xs_e[1:], np.full(len(xs_e) - 1, y_edge))
        inner_t = np.concatenate([xs_w[:-1], edge_t, xs_e[1:]])
        inner_z = np.concatenate([elevation_fn(wing_w), edge_z, elevation_fn(wing_e)])
        parts.append(_strip(inner_t, inner_z, full_x, across, horizontal, elevation_fn))

    for x_edge, across in ((x1, xs_e), (x0, xs_w[::-1])):
        edge_t, edge_z = _boundary(target_vertices, 0, x_edge, 1)
        parts.append(_strip(edge_t, edge_z, ys_mid, across, vertical, elevation_fn))

    return _assemble(parts, bounds, handle_nans)
