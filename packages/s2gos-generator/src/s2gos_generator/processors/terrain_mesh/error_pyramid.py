from __future__ import annotations

import numpy as np


def _uniform_step(axis: np.ndarray, name: str) -> float:
    """Spacing of a uniformly spaced axis, rejecting anything else.

    The block arithmetic in this module is only correct on a uniform axis.
    """
    axis = np.asarray(axis, dtype=float)
    if len(axis) < 2:
        raise ValueError(f"{name} axis needs at least 2 samples, got {len(axis)}")
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    if step <= 0 or not np.allclose(np.diff(axis), step, rtol=0.0, atol=1e-6 * step):
        raise ValueError(f"{name} axis must be ascending and uniformly spaced")
    return float(step)


class DemErrorPyramid:
    """Precomputed per-level max plane-residual errors for adaptive-quadtree decimation.

    Level ``L`` holds, for every ``K x K`` block of DEM pixels (``K = 2 ** (D - L)``),
    the largest deviation of the DEM from the least-squares plane fitted to that block.
    Blocks tile from pixel 0, so a quadtree cell with index ``i`` at level ``L`` is
    described by block ``i``: both structures halve from the same origin.

    Errors are saturated top-down (parent >= max of children) so refinement decisions
    are monotone and no cracks arise from a coarse cell being called flat after its
    children were already refined.

    ``D`` (``decimation_depth``) is the number of refinement levels available, matching
    ``MeshRefinementConfig.decimation_depth``. At level ``D`` a block is one pixel and
    its residual is 0.

    The quadtree's base grid gains a short final cell whenever the DEM sample count is
    not stride-aligned, and no block describes it. :meth:`query` answers "subdivide" for
    such a cell, leaving that strip at native resolution.
    """

    def __init__(
        self,
        elev: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        decimation_depth: int,
    ) -> None:
        self._dx = _uniform_step(x, "x")
        self._dy = _uniform_step(y, "y")
        self._x_start = float(x[0])
        self._y_start = float(y[0])
        self._nx = elev.shape[1]
        self._ny = elev.shape[0]
        self._decimation_depth = decimation_depth

        self._levels: list[np.ndarray] = self._build(elev, decimation_depth)

    def query(
        self,
        xmin: np.ndarray,
        ymin: np.ndarray,
        level: int,
        tolerance_m: float,
    ) -> np.ndarray:
        """Vectorized predicate: True where a cell should be subdivided.

        A cell is subdivided when its max plane-residual exceeds ``tolerance_m``, or
        when no block describes it at all.
        """
        lvl_arr = self._levels[level]
        nh, nw = lvl_arr.shape
        block = 1 << (self._decimation_depth - level)

        # Round rather than floor: a cell corner is always a sample, give or take the
        # float error in the linspace that produced it.
        i_arr = np.rint((xmin - self._x_start) / self._dx).astype(np.int64) // block
        j_arr = np.rint((ymin - self._y_start) / self._dy).astype(np.int64) // block

        # The index clip only keeps the lookup legal, ``unmapped`` discards its result.
        unmapped = (i_arr < 0) | (i_arr >= nw) | (j_arr < 0) | (j_arr >= nh)
        residual = lvl_arr[np.clip(j_arr, 0, nh - 1), np.clip(i_arr, 0, nw - 1)].astype(
            np.float64
        )
        return unmapped | (residual > tolerance_m)

    def _build(self, elev: np.ndarray, D: int) -> list[np.ndarray]:
        levels: list[np.ndarray] = []
        for L in range(D + 1):
            K = 1 << (D - L)
            if K == 1:
                levels.append(np.zeros((self._ny, self._nx), dtype=np.float32))
            else:
                levels.append(self._compute_level(elev, K))

        for L in range(D - 1, -1, -1):
            child = levels[L + 1]
            parent = levels[L]
            ch = (child.shape[0] // 2) * 2
            cw = (child.shape[1] // 2) * 2
            if ch == 0 or cw == 0:
                continue
            child_max = np.maximum(
                np.maximum(child[:ch:2, :cw:2], child[1:ch:2, :cw:2]),
                np.maximum(child[:ch:2, 1:cw:2], child[1:ch:2, 1:cw:2]),
            )
            ph = min(parent.shape[0], child_max.shape[0])
            pw = min(parent.shape[1], child_max.shape[1])
            parent[:ph, :pw] = np.maximum(parent[:ph, :pw], child_max[:ph, :pw])

        return levels

    def _compute_level(self, elev: np.ndarray, K: int) -> np.ndarray:
        """Compute max plane residuals for non-overlapping K×K blocks."""
        ny, nx = elev.shape
        ny_cells = ny // K
        nx_cells = nx // K

        if ny_cells == 0 or nx_cells == 0:
            return np.zeros((max(1, ny_cells), max(1, nx_cells)), dtype=np.float32)

        elev_crop = elev[: ny_cells * K, : nx_cells * K]
        blocks = elev_crop.reshape(ny_cells, K, nx_cells, K).transpose(0, 2, 1, 3)

        n = K * K
        u = np.arange(K, dtype=np.float64)
        v = np.arange(K, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        uu_flat = uu.ravel()
        vv_flat = vv.ravel()

        sum_u = uu_flat.sum()
        sum_v = vv_flat.sum()
        sum_u2 = (uu_flat**2).sum()
        sum_v2 = (vv_flat**2).sum()
        sum_uv = (uu_flat * vv_flat).sum()

        A = np.array(
            [
                [sum_u2, sum_uv, sum_u],
                [sum_uv, sum_v2, sum_v],
                [sum_u, sum_v, n],
            ]
        )
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.zeros((ny_cells, nx_cells), dtype=np.float32)

        z_flat = blocks.reshape(ny_cells, nx_cells, n).astype(np.float64)
        sum_uz = (z_flat * uu_flat).sum(axis=2)
        sum_vz = (z_flat * vv_flat).sum(axis=2)
        sum_z = z_flat.sum(axis=2)

        rhs = np.stack([sum_uz, sum_vz, sum_z], axis=2)

        abc = rhs @ A_inv.T
        a = abc[:, :, 0:1]
        b = abc[:, :, 1:2]
        c = abc[:, :, 2:3]

        plane = a * uu_flat + b * vv_flat + c
        max_resid = np.abs(z_flat - plane).max(axis=2)

        return max_resid.astype(np.float32)
