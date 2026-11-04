"""Fit of a real product's pixel grid around a target."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from s2gos_utils.coordinates import CoordinateSystem


class Grid(NamedTuple):
    """A footprint snapped onto a product's pixel grid. The matching real pixels are
    ``[first_row : first_row + rows, first_column : first_column + columns]``."""

    center_lat: float
    center_lon: float

    rotation: float
    """Clockwise rotation from north-up in degrees within [-180, 180), as
    `SatelliteSensor.grid_rotation`."""

    pixel_size_m: tuple[float, float]
    """Pixel (width, height) on the ground in meters: the spacing between columns,
    then between rows."""

    shape: tuple[int, int]
    """(columns, rows), as `SatelliteSensor.film_resolution`."""

    first_row: int
    """Product row of the footprint's first row."""

    first_column: int
    """Product column of the footprint's first column."""


def _heading(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vector[0], vector[1])) % 360.0)


def fit_grid(
    rows: np.ndarray,
    cols: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    size_m: tuple[float, float],
) -> Grid:
    """Snaps a footprint centered on (lat, lon) onto a product's pixel grid.

    The grid is the affine map from pixel index to local ground meters, fitted on
    pixel centers near the target. The footprint holds the whole number of pixels
    closest to `size_m`, and its center moves to the nearest position that puts its
    pixels on the product's.

    Args:
        rows: Product image row of each sample pixel center.
        cols: Product image column of each sample pixel center.
        lats: Latitude of each sample. These must be positions of the product grid
            itself, not a geolocation of what each pixel sees.
        lons: Longitude of each sample.
        lat: Target latitude (WGS84 decimal degrees).
        lon: Target longitude (WGS84 decimal degrees).
        size_m: Approximate footprint (width, height) in meters.

    Raises:
        ValueError: If the grid is mirrored or skewed, so a rotation of a north-up
            grid cannot represent it.
    """
    coords = CoordinateSystem(lat, lon)
    x, y = coords.latlon_to_scene(np.asarray(lats), np.asarray(lons))
    design = np.c_[np.ravel(cols), np.ravel(rows), np.ones(np.size(rows))]
    affine = np.linalg.lstsq(design, np.c_[np.ravel(x), np.ravel(y)], rcond=None)[0]
    column, row, origin = affine

    rotation = (_heading(column) + 90.0) % 360.0 - 180.0
    if abs((_heading(row) - _heading(column)) % 360.0 - 90.0) > 1.0:
        raise ValueError(
            f"Product grid with columns heading {_heading(column):.1f} deg and rows "
            f"{_heading(row):.1f} deg is not a rotated north-up grid"
        )

    spacing = (float(np.hypot(*column)), float(np.hypot(*row)))
    target = np.linalg.solve(np.c_[column, row], -origin)
    shape = tuple(max(1, round(size / step)) for size, step in zip(size_m, spacing))
    first = [int(round(t - (n - 1) / 2)) for t, n in zip(target, shape)]
    col_center, row_center = (f + (n - 1) / 2 for f, n in zip(first, shape))
    center_lat, center_lon = coords.scene_to_latlon(
        *(origin + col_center * column + row_center * row)
    )
    return Grid(
        float(center_lat),
        float(center_lon),
        rotation,
        spacing,
        shape,
        first[1],
        first[0],
    )
