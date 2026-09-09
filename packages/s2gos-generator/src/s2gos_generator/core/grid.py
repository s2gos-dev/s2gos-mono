"""Where a scene area's ground is.

Every scene raster is a :class:`SceneGrid`: uniform, at exactly the requested resolution,
and so slightly larger than the AOI when that resolution does not divide it. The caller
picks the registration, :meth:`SceneGrid.nodes` for point-like data such as elevation and
:meth:`SceneGrid.cell_centres` for area-like data such as a land cover class.

Only the terrain mesh spans the AOI exactly, and it is clipped to get there.

Every scene raster is stored ascending-y, so row 0 is the southernmost row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

#: Relative slack for resolution dividing the AOI evenly
_TOL = 1e-9


def aoi_to_uv(xy: np.ndarray, aoi_size_m: float) -> np.ndarray:
    """Map scene ``(x, y)`` metres to ``(u, v)`` over the AOI.

    ``u = 0`` at the AOI's western edge, ``v = 0`` at its southern edge.
    """
    return (np.asarray(xy, dtype=float) + aoi_size_m / 2.0) / aoi_size_m


@dataclass(frozen=True)
class SceneGrid:
    """A square regular grid centred on the scene origin.

    Args:
        size_m: The extent of *this grid*, which is not necessarily the AOI.
            :meth:`covering` reaches past it when the resolution does not divide
            the AOI evenly.
        n: Number of cells per axis.
    """

    size_m: float
    n: int

    def __post_init__(self) -> None:
        if not self.size_m > 0:
            raise ValueError(f"SceneGrid.size_m must be positive, got {self.size_m!r}")
        if self.n < 1:
            raise ValueError(f"SceneGrid.n must be at least 1, got {self.n!r}")

    @classmethod
    def covering(cls, aoi_size_m: float, resolution_m: float) -> "SceneGrid":
        """The smallest whole number of exact cells that covers the AOI.

        The cell size is exactly ``resolution_m``, so the extent bends instead. When
        the AOI divides evenly the grid **is** the AOI.
        """
        _check_resolution(resolution_m)
        n = max(1, math.ceil(aoi_size_m / resolution_m - _TOL))
        return cls(n * float(resolution_m), n)

    @property
    def resolution_m(self) -> float:
        """Cell size (``size_m / n``)."""
        return self.size_m / self.n

    @property
    def half_size_m(self) -> float:
        """Distance from the origin to this grid's boundary."""
        return self.size_m / 2.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the ground *this grid* covers, in metres."""
        h = self.half_size_m
        return (-h, -h, h, h)

    def cell_centres(self) -> np.ndarray:
        """``n`` ascending cell-centre coordinates, inset half a cell."""
        h, r = self.half_size_m, self.resolution_m
        return np.linspace(-h + r / 2.0, h - r / 2.0, self.n)

    def nodes(self) -> np.ndarray:
        """``n + 1`` ascending cell-corner coordinates, reaching :attr:`bounds`."""
        h = self.half_size_m
        return np.linspace(-h, h, self.n + 1)

    def world_to_uv(self, xy: np.ndarray) -> np.ndarray:
        """Map scene ``(x, y)`` metres to ``(u, v)`` over *this grid's* extent.

        Not the AOI's, for the mesh's mapping use :func:`aoi_to_uv`.
        """
        return aoi_to_uv(xy, self.size_m)

    def uv_window(self, aoi_size_m: float) -> Tuple[float, float, float, float]:
        """``(u0, v0, u1, v1)``: where a mesh spanning the AOI lands in this grid's UV.

        Identity when the grid is exactly the AOI, otherwise the interior rectangle the
        AOI occupies, which the renderer applies as ``to_uv``. Uses :func:`aoi_to_uv`,
        the same mapping the mesh's own UVs come from.
        """
        if aoi_size_m > self.size_m * (1.0 + _TOL):
            raise ValueError(
                f"AOI of {aoi_size_m} m does not fit inside a grid of {self.size_m} m"
            )
        half = aoi_size_m / 2.0
        (u0, v0), (u1, v1) = self.world_to_uv(np.array([[-half, -half], [half, half]]))
        return (float(u0), float(v0), float(u1), float(v1))

    def transform(self):
        """North-up rasterio affine over :attr:`bounds` (row 0 = north)."""
        from rasterio.transform import from_bounds

        xmin, ymin, xmax, ymax = self.bounds
        return from_bounds(xmin, ymin, xmax, ymax, self.n, self.n)

    def rasterize(
        self,
        shapes: Iterable[Any],
        *,
        all_touched: bool = False,
        fill: int = 0,
        dtype: Any = np.uint8,
    ) -> np.ndarray:
        """Burn ``shapes`` onto this grid, returned in scene row order.

        ``shapes`` is anything ``rasterio.features.rasterize`` accepts. rasterio emits
        row 0 = north, which is flipped here.
        """
        from rasterio.features import rasterize

        burned = rasterize(
            shapes,
            out_shape=(self.n, self.n),
            transform=self.transform(),
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )
        return np.flipud(burned)


def _check_resolution(resolution_m: float) -> None:
    if not resolution_m > 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m!r}")
