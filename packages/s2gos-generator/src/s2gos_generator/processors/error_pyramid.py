from __future__ import annotations

import numpy as np


class DemErrorPyramid:
    """Precomputed per-level max plane-residual errors for adaptive-quadtree decimation.

    Each level L stores, for every cell of size K×K DEM pixels (K = 2^(D-L)),
    the maximum absolute deviation of the DEM from the least-squares plane
    fitted to that cell. Errors are saturated top-down (parent ≥ max of
    children) so top-down refinement decisions are monotone — no cracks arise
    from coarse cells being mis-classified as flat after their children were
    already refined.

    D (decimation_depth) is the number of refinement levels used for terrain
    decimation (matches ``MeshRefinementConfig.decimation_depth``). At the
    finest level D, cells are 1×1 DEM pixels and the residual is 0.
    """

    def __init__(
        self,
        elev: np.ndarray,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        decimation_depth: int,
    ) -> None:
        self._x0 = x0
        self._y0 = y0
        self._dx = dx
        self._dy = dy
        self._nx = elev.shape[1]
        self._ny = elev.shape[0]
        self._decimation_depth = decimation_depth

        self._x_start = min(x0, x0 + (self._nx - 1) * dx)
        self._y_start = min(y0, y0 + (self._ny - 1) * dy)

        self._levels: list[np.ndarray] = self._build(elev, decimation_depth)

    def query(
        self,
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
        tolerance_m: float,
    ) -> np.ndarray:
        """Vectorized predicate: True where max plane-residual > tolerance_m.

        Cell sizes are inferred from the world-coordinate extents so this
        method can be used directly as a ``refine`` predicate without knowing
        the current level explicitly.
        """
        D = self._decimation_depth
        K_float = abs(xmax[0] - xmin[0]) / abs(self._dx)
        K = max(1, int(round(K_float)))
        level = max(0, D - int(round(np.log2(max(1.0, float(K))))))

        lvl_arr = self._levels[level]
        nh, nw = lvl_arr.shape

        K_actual = 1 << (D - level)
        abs_dx = abs(self._dx)
        abs_dy = abs(self._dy)

        ix = (np.minimum(xmin, xmax) - self._x_start) / abs_dx
        iy = (np.minimum(ymin, ymax) - self._y_start) / abs_dy

        i_arr = np.clip(np.floor(ix / K_actual).astype(int), 0, nw - 1)
        j_arr = np.clip(np.floor(iy / K_actual).astype(int), 0, nh - 1)

        return lvl_arr[j_arr, i_arr].astype(np.float64) > tolerance_m

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
