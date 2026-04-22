from __future__ import annotations

from typing import Callable

import numpy as np


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

        # Precompute constants by depth to bypass per-iteration calculations
        self._dx_levels = [dx_base / (1 << L) for L in range(max_depth + 2)]
        self._dy_levels = [dy_base / (1 << L) for L in range(max_depth + 2)]
        self.limit_x = [self._nx << L for L in range(max_depth + 2)]
        self.limit_y = [self._ny << L for L in range(max_depth + 2)]

        # Vectorised base-leaves construction (all cells at level 0)
        ii, jj = np.meshgrid(
            np.arange(self._nx, dtype=np.uint64),
            np.arange(self._ny, dtype=np.uint64),
            indexing="ij",
        )
        self._leaves_arr = (ii.ravel() << np.uint64(30)) | jj.ravel()
        self._leaves: set[int] = set()

    def refine(
        self,
        predicate: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
    ) -> None:
        """Batch-subdivide leaves using the supplied spatial predicate.

        Args:
            predicate: ``(xmin, ymin, xmax, ymax) → bool[N]`` — returns True
                for cells that should be subdivided.  All arguments are
                numpy arrays of the same length.
        """
        arr = self._leaves_arr

        for level in range(self.max_depth):
            lvl_uint = np.uint64(level)
            mask = (arr >> np.uint64(60)) == lvl_uint
            if not mask.any():
                break

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

            arr = np.concatenate((kept_other, to_keep, c00, c10, c01, c11))

        self._leaves = set(arr.tolist())

    def balance(self) -> None:
        """Ensure 2:1 maximum-grade across all quadtree boundaries."""
        if not self._leaves:
            self._leaves = set(self._leaves_arr.tolist())

        queue = {c for c in self._leaves if (c >> 60) > 0}

        while queue:
            cell = queue.pop()
            if cell not in self._leaves:
                continue

            level = cell >> 60
            i = (cell >> 30) & 0x3FFFFFFF
            j = cell & 0x3FFFFFFF

            needs_split = False
            finer = level + 1
            finer_mask = finer << 60
            lx = self.limit_x[level]
            ly = self.limit_y[level]
            lvl_mask = level << 60

            if i > 0 and (lvl_mask | ((i - 1) << 30) | j) not in self._leaves:
                bi, bj = (i - 1) << 1, j << 1
                if (
                    (finer_mask | (bi << 30) | bj) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | bj) in self._leaves
                    or (finer_mask | (bi << 30) | (bj + 1)) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | (bj + 1)) in self._leaves
                ):
                    needs_split = True

            if (
                not needs_split
                and i + 1 < lx
                and (lvl_mask | ((i + 1) << 30) | j) not in self._leaves
            ):
                bi, bj = (i + 1) << 1, j << 1
                if (
                    (finer_mask | (bi << 30) | bj) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | bj) in self._leaves
                    or (finer_mask | (bi << 30) | (bj + 1)) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | (bj + 1)) in self._leaves
                ):
                    needs_split = True

            # Bottom
            if (
                not needs_split
                and j > 0
                and (lvl_mask | (i << 30) | (j - 1)) not in self._leaves
            ):
                bi, bj = i << 1, (j - 1) << 1
                if (
                    (finer_mask | (bi << 30) | bj) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | bj) in self._leaves
                    or (finer_mask | (bi << 30) | (bj + 1)) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | (bj + 1)) in self._leaves
                ):
                    needs_split = True

            # Top
            if (
                not needs_split
                and j + 1 < ly
                and (lvl_mask | (i << 30) | (j + 1)) not in self._leaves
            ):
                bi, bj = i << 1, (j + 1) << 1
                if (
                    (finer_mask | (bi << 30) | bj) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | bj) in self._leaves
                    or (finer_mask | (bi << 30) | (bj + 1)) in self._leaves
                    or (finer_mask | ((bi + 1) << 30) | (bj + 1)) in self._leaves
                ):
                    needs_split = True

            if needs_split:
                self._leaves.remove(cell)
                bi, bj = i << 1, j << 1

                c00 = finer_mask | (bi << 30) | bj
                c10 = finer_mask | ((bi + 1) << 30) | bj
                c01 = finer_mask | (bi << 30) | (bj + 1)
                c11 = finer_mask | ((bi + 1) << 30) | (bj + 1)

                new_cells = (c00, c10, c01, c11)
                self._leaves.update(new_cells)
                queue.update(new_cells)

                if i > 0:
                    queue.add(lvl_mask | ((i - 1) << 30) | j)
                if i + 1 < lx:
                    queue.add(lvl_mask | ((i + 1) << 30) | j)
                if j > 0:
                    queue.add(lvl_mask | (i << 30) | (j - 1))
                if j + 1 < ly:
                    queue.add(lvl_mask | (i << 30) | (j + 1))

    def to_mesh(
        self, elevation_fn: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Emit vertices and faces for the current leaf set.

        Args:
            elevation_fn: ``(N, 2) XY → (N,) Z`` elevation sampler.

        Returns:
            ``(vertices, faces)`` — (N, 3) float64 and (M, 3) int32 arrays.
        """
        vert_map: dict[int, int] = {}
        v_get = vert_map.get
        xy_x: list[float] = []
        xy_y: list[float] = []
        x_app, y_app = xy_x.append, xy_y.append
        face_list: list[tuple[int, int, int]] = []
        face_add = face_list.append

        scale_x = self._dx_levels[self.max_depth]
        scale_y = self._dy_levels[self.max_depth]
        xmin, ymin = self._xmin, self._ymin

        def _get(vx: int, vy: int) -> int:
            k = (vx << 32) | vy
            idx = v_get(k)
            if idx is None:
                idx = len(vert_map)
                vert_map[k] = idx
                x_app(xmin + vx * scale_x)
                y_app(ymin + vy * scale_y)
            return idx

        for cell in self._leaves:
            level = cell >> 60
            i = (cell >> 30) & 0x3FFFFFFF
            j = cell & 0x3FFFFFFF

            step = 1 << (self.max_depth - level)
            x0, y0 = i * step, j * step
            x1, y1 = x0 + step, y0 + step

            finer = level + 1
            can_split = finer <= self.max_depth
            split_b = split_t = split_l = split_r = False

            if can_split:
                lx, ly = self.limit_x[level], self.limit_y[level]
                lvl_mask, finer_mask = level << 60, finer << 60

                if j > 0 and (lvl_mask | (i << 30) | (j - 1)) not in self._leaves:
                    bi, bj = i << 1, (j - 1) << 1
                    if (finer_mask | (bi << 30) | (bj + 1)) in self._leaves or (
                        finer_mask | ((bi + 1) << 30) | (bj + 1)
                    ) in self._leaves:
                        split_b = True

                if j + 1 < ly and (lvl_mask | (i << 30) | (j + 1)) not in self._leaves:
                    bi, bj = i << 1, (j + 1) << 1
                    if (finer_mask | (bi << 30) | bj) in self._leaves or (
                        finer_mask | ((bi + 1) << 30) | bj
                    ) in self._leaves:
                        split_t = True

                if i > 0 and (lvl_mask | ((i - 1) << 30) | j) not in self._leaves:
                    bi, bj = (i - 1) << 1, j << 1
                    if (finer_mask | ((bi + 1) << 30) | bj) in self._leaves or (
                        finer_mask | ((bi + 1) << 30) | (bj + 1)
                    ) in self._leaves:
                        split_l = True

                if i + 1 < lx and (lvl_mask | ((i + 1) << 30) | j) not in self._leaves:
                    bi, bj = (i + 1) << 1, j << 1
                    if (finer_mask | (bi << 30) | bj) in self._leaves or (
                        finer_mask | (bi << 30) | (bj + 1)
                    ) in self._leaves:
                        split_r = True

            v00, v10 = _get(x0, y0), _get(x1, y0)
            v11, v01 = _get(x1, y1), _get(x0, y1)

            if not (split_b or split_t or split_l or split_r):
                face_add((v00, v10, v11))
                face_add((v00, v11, v01))
            else:
                cx, cy = x0 + (step >> 1), y0 + (step >> 1)
                vc = _get(cx, cy)

                boundary = [v00]
                if split_b:
                    boundary.append(_get(cx, y0))
                boundary.append(v10)
                if split_r:
                    boundary.append(_get(x1, cy))
                boundary.append(v11)
                if split_t:
                    boundary.append(_get(cx, y1))
                boundary.append(v01)
                if split_l:
                    boundary.append(_get(x0, cy))

                for k in range(len(boundary) - 1):
                    face_add((vc, boundary[k], boundary[k + 1]))
                face_add((vc, boundary[-1], boundary[0]))

        xy_arr = np.column_stack((xy_x, xy_y))
        z_arr = elevation_fn(xy_arr)

        vertices = np.column_stack((xy_arr, z_arr))
        faces = np.array(face_list, dtype=np.int32)
        return vertices, faces
