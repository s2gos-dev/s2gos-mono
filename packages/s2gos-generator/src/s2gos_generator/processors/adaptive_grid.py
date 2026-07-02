from __future__ import annotations

from typing import Callable

import numpy as np


def _build_mesh_templates() -> list[np.ndarray]:
    """Precompute the 16 triangulation patterns indexed by 4-bit neighbor-split mask.

    Mask bit layout: bit0=split_b, bit1=split_r, bit2=split_t, bit3=split_l.

    Each entry is an (n_tri, 3) int8 array of slot indices into the 9 canonical
    vertex slots per leaf:
        0=v00, 1=vS, 2=v10, 3=vE, 4=v11, 5=vN, 6=v01, 7=vW, 8=vC
    where v00/v10/v11/v01 are corners, vS/vE/vN/vW are edge midpoints (conditional),
    and vC is the cell center (present whenever any edge is split).
    """
    templates = []
    for m in range(16):
        split_b = bool(m & 1)
        split_r = bool(m & 2)
        split_t = bool(m & 4)
        split_l = bool(m & 8)

        if not (split_b or split_r or split_t or split_l):
            templates.append(np.array([[0, 2, 4], [0, 4, 6]], dtype=np.int8))
        else:
            boundary = [0]
            if split_b:
                boundary.append(1)
            boundary.append(2)
            if split_r:
                boundary.append(3)
            boundary.append(4)
            if split_t:
                boundary.append(5)
            boundary.append(6)
            if split_l:
                boundary.append(7)
            nb = len(boundary)
            tris = [[8, boundary[k], boundary[k + 1]] for k in range(nb - 1)]
            tris.append([8, boundary[-1], boundary[0]])
            templates.append(np.array(tris, dtype=np.int8))
    return templates


_MESH_TEMPLATES: list[np.ndarray] = _build_mesh_templates()


class AdaptiveGrid:
    """Quadtree over a regular DEM grid.

    Cells are stored as packed ``uint64`` integers:
        bits 63-60 : depth level
        bits 59-30 : column index i
        bits 29-0  : row index j
    """

    def __init__(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        max_depth: int,
    ):
        self._xmin = float(x_coords[0])
        self._xmax = float(x_coords[-1])
        self._ymin = float(y_coords[0])
        self._ymax = float(y_coords[-1])

        self._nx = int(len(x_coords) - 1)
        self._ny = int(len(y_coords) - 1)
        self.max_depth = max_depth

        if self._nx <= 0 or self._ny <= 0:
            raise ValueError("DEM must have at least 2 points in each dimension")

        dx_base = (self._xmax - self._xmin) / self._nx
        dy_base = (self._ymax - self._ymin) / self._ny

        self._dx_levels = [dx_base / (1 << L) for L in range(max_depth + 2)]
        self._dy_levels = [dy_base / (1 << L) for L in range(max_depth + 2)]
        self.limit_x = [self._nx << L for L in range(max_depth + 2)]
        self.limit_y = [self._ny << L for L in range(max_depth + 2)]

        ii, jj = np.meshgrid(
            np.arange(self._nx, dtype=np.uint64),
            np.arange(self._ny, dtype=np.uint64),
            indexing="ij",
        )
        self._leaves_arr = (ii.ravel() << np.uint64(30)) | jj.ravel()
        self._leaves: set[int] = set()
        self._internal: set[int] = set()

    def refine(
        self,
        predicate: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
        max_level: int | None = None,
    ) -> None:
        """Batch-subdivide leaves using the supplied spatial predicate."""
        limit = self.max_depth if max_level is None else max_level
        arr = (
            self._leaves_arr
            if not self._leaves
            else np.array(list(self._leaves), dtype=np.uint64)
        )

        for level in range(limit):
            lvl_uint = np.uint64(level)
            mask = (arr >> np.uint64(60)) == lvl_uint
            if not mask.any():
                continue

            current = arr[mask]
            kept_other = arr[~mask]

            i_arr = (current >> np.uint64(30)) & np.uint64(0x3FFFFFFF)
            j_arr = current & np.uint64(0x3FFFFFFF)

            dx = self._dx_levels[level]
            dy = self._dy_levels[level]
            xmin = self._xmin + i_arr.astype(float) * dx
            ymin = self._ymin + j_arr.astype(float) * dy
            xmax = xmin + dx
            ymax = ymin + dy

            intersects = predicate(xmin, ymin, xmax, ymax)

            to_keep = current[~intersects]
            to_split = current[intersects]

            if to_split.size == 0:
                arr = np.concatenate((kept_other, to_keep))
                continue

            split_i = i_arr[intersects] << np.uint64(1)
            split_j = j_arr[intersects] << np.uint64(1)

            c_level = np.uint64(level + 1) << np.uint64(60)
            shift30 = np.uint64(30)
            one = np.uint64(1)

            c00 = c_level | (split_i << shift30) | split_j
            c10 = c_level | ((split_i + one) << shift30) | split_j
            c01 = c_level | (split_i << shift30) | (split_j + one)
            c11 = c_level | ((split_i + one) << shift30) | (split_j + one)

            self._internal.update(to_split.tolist())

            arr = np.concatenate((kept_other, to_keep, c00, c10, c01, c11))

        self._leaves = set(arr.tolist())

    def balance(self) -> None:
        """Ensure 2:1 maximum-grade across all quadtree boundaries."""
        if not self._leaves:
            self._leaves = set(self._leaves_arr.tolist())
            self._internal = set()

        queue = list(self._internal)
        while queue:
            cell = queue.pop()
            lvl = cell >> 60
            i = (cell >> 30) & 0x3FFFFFFF
            j = cell & 0x3FFFFFFF

            neighbors = []
            if i > 0:
                neighbors.append((lvl, i - 1, j))
            if i + 1 < self.limit_x[lvl]:
                neighbors.append((lvl, i + 1, j))
            if j > 0:
                neighbors.append((lvl, i, j - 1))
            if j + 1 < self.limit_y[lvl]:
                neighbors.append((lvl, i, j + 1))

            for n_lvl, ni, nj in neighbors:
                n_cell = (n_lvl << 60) | (ni << 30) | nj
                # If neighbor is missing entirely, it means a coarser ancestor is holding the space
                if n_cell not in self._internal and n_cell not in self._leaves:
                    a_lvl, ai, aj = n_lvl, ni, nj
                    while a_lvl > 0:
                        a_lvl -= 1
                        ai >>= 1
                        aj >>= 1
                        ancestor = (a_lvl << 60) | (ai << 30) | aj
                        if ancestor in self._leaves:
                            # Split the offending coarse neighbor!
                            self._leaves.remove(ancestor)
                            self._internal.add(ancestor)
                            queue.append(ancestor)

                            c_lvl = a_lvl + 1
                            c_mask = c_lvl << 60
                            ci, cj = ai << 1, aj << 1
                            self._leaves.update(
                                [
                                    c_mask | (ci << 30) | cj,
                                    c_mask | ((ci + 1) << 30) | cj,
                                    c_mask | (ci << 30) | (cj + 1),
                                    c_mask | ((ci + 1) << 30) | (cj + 1),
                                ]
                            )

                            # Re-evaluate the original cell to see if neighbor requirements are met now
                            queue.append(cell)
                            break

    def to_mesh(
        self, elevation_fn: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Emit a crack-free triangle mesh from the current quadtree."""
        if self._leaves:
            leaves_arr = np.array(list(self._leaves), dtype=np.uint64)
        else:
            leaves_arr = self._leaves_arr

        n = len(leaves_arr)
        level_arr = (leaves_arr >> np.uint64(60)).astype(np.int64)
        i_arr = ((leaves_arr >> np.uint64(30)) & np.uint64(0x3FFFFFFF)).astype(np.int64)
        j_arr = (leaves_arr & np.uint64(0x3FFFFFFF)).astype(np.int64)

        step_arr = np.int64(1) << (self.max_depth - level_arr)
        x0 = i_arr * step_arr
        y0 = j_arr * step_arr
        x1 = x0 + step_arr
        y1 = y0 + step_arr
        cx = x0 + (step_arr >> 1)
        cy = y0 + (step_arr >> 1)

        can_split = level_arr < self.max_depth
        lvl_keys = level_arr.astype(np.uint64) << np.uint64(60)

        if self._internal:
            internals_arr = np.fromiter(
                self._internal, dtype=np.uint64, count=len(self._internal)
            )
        else:
            internals_arr = np.empty(0, dtype=np.uint64)

        limit_x_arr = np.array(self.limit_x, dtype=np.int64)
        limit_y_arr = np.array(self.limit_y, dtype=np.int64)
        lx = limit_x_arr[level_arr]
        ly = limit_y_arr[level_arr]

        def _split_flag(
            ni: np.ndarray, nj: np.ndarray, in_bounds: np.ndarray
        ) -> np.ndarray:
            """True for leaves whose same-level neighbor at (ni, nj) is in _internal."""
            keys = np.zeros(n, dtype=np.uint64)
            ib = in_bounds
            keys[ib] = (
                lvl_keys[ib]
                | ni[ib].astype(np.uint64) << np.uint64(30)
                | nj[ib].astype(np.uint64)
            )
            result = np.zeros(n, dtype=bool)
            if internals_arr.size > 0:
                result[ib] = np.isin(keys[ib], internals_arr)
            return result

        split_b = _split_flag(i_arr, j_arr - 1, can_split & (j_arr > 0))
        split_t = _split_flag(i_arr, j_arr + 1, can_split & (j_arr + 1 < ly))
        split_l = _split_flag(i_arr - 1, j_arr, can_split & (i_arr > 0))
        split_r = _split_flag(i_arr + 1, j_arr, can_split & (i_arr + 1 < lx))

        tmask = (
            split_b.astype(np.int32)
            | (split_r.astype(np.int32) << 1)
            | (split_t.astype(np.int32) << 2)
            | (split_l.astype(np.int32) << 3)
        )
        any_split = tmask > 0

        slot_vx = [x0, cx, x1, x1, x1, cx, x0, x0, cx]
        slot_vy = [y0, y0, y0, cy, y1, y1, y1, cy, cy]
        slot_active = [
            np.ones(n, dtype=bool),
            split_b,
            np.ones(n, dtype=bool),
            split_r,
            np.ones(n, dtype=bool),
            split_t,
            np.ones(n, dtype=bool),
            split_l,
            any_split,
        ]

        max_nx = self.limit_x[self.max_depth] + 1
        max_ny = self.limit_y[self.max_depth] + 1
        use_dense = max_nx * max_ny <= 32 * 1024 * 1024  # 128 MB cap (int32)

        if use_dense:
            vid_grid = np.full((max_nx, max_ny), -1, dtype=np.int32)
        else:
            vid_dict: dict[int, int] = {}

        scale_x = self._dx_levels[self.max_depth]
        scale_y = self._dy_levels[self.max_depth]
        xy_x_chunks: list[np.ndarray] = []
        xy_y_chunks: list[np.ndarray] = []
        next_vid = 0

        vertex_ids = np.full((9, n), -1, dtype=np.int32)

        for s in range(9):
            active = slot_active[s]
            if not active.any():
                continue

            vx = slot_vx[s][active]
            vy = slot_vy[s][active]

            keys_1d = vx.astype(np.int64) * max_ny + vy.astype(np.int64)
            uniq_keys, inv = np.unique(keys_1d, return_inverse=True)
            u_vx = (uniq_keys // max_ny).astype(np.intp)
            u_vy = (uniq_keys % max_ny).astype(np.intp)

            if use_dense:
                existing = vid_grid[u_vx, u_vy].copy()
            else:
                existing = np.array(
                    [vid_dict.get(int(k), -1) for k in uniq_keys], dtype=np.int32
                )

            new_mask = existing == -1
            if new_mask.any():
                n_new = int(new_mask.sum())
                new_ids = np.arange(next_vid, next_vid + n_new, dtype=np.int32)
                next_vid += n_new
                if use_dense:
                    vid_grid[u_vx[new_mask], u_vy[new_mask]] = new_ids
                else:
                    for idx, key in enumerate(uniq_keys[new_mask].tolist()):
                        vid_dict[key] = int(new_ids[idx])
                existing[new_mask] = new_ids
                xy_x_chunks.append(self._xmin + u_vx[new_mask] * scale_x)
                xy_y_chunks.append(self._ymin + u_vy[new_mask] * scale_y)

            vertex_ids[s, active] = existing[inv]

        all_faces: list[np.ndarray] = []
        for m in range(16):
            leaf_sel = tmask == m
            if not leaf_sel.any():
                continue
            t = _MESH_TEMPLATES[m].astype(np.intp)
            vids = vertex_ids[:, leaf_sel]
            faces_m = np.stack([vids[t[:, c]] for c in range(3)], axis=2)
            all_faces.append(faces_m.reshape(-1, 3))

        xy_arr = np.column_stack(
            (np.concatenate(xy_x_chunks), np.concatenate(xy_y_chunks))
        )
        z_arr = elevation_fn(xy_arr)
        vertices = np.column_stack((xy_arr, z_arr))
        faces = np.vstack(all_faces).astype(np.int32)
        return vertices, faces
