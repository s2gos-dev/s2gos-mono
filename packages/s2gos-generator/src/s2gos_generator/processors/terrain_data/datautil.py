from typing import Optional

import numpy as np
import xarray as xr
from pyproj import Proj

from ...core.exceptions import RegridError


def regrid_to_projection(
    dataset: xr.Dataset,
    axis: np.ndarray,
    *,
    center_lat: float,
    center_lon: float,
    interpolation_method: str = "linear",
    fillna_value: Optional[float] = None,
    data_variable: Optional[str] = None,
) -> xr.Dataset:
    """Regrid dataset onto a scene axis using an oblique mercator projection.

    Args:
        dataset: Input xarray Dataset with lat/lon coordinates
        axis: 1-D ascending target coordinates in scene metres, used for both x and y
        center_lat: Center latitude for projection
        center_lon: Center longitude for projection
        interpolation_method: Interpolation method ("linear", "nearest", etc.)
        fillna_value: Value to fill NaN values with (optional)
        data_variable: Specific data variable to process for fillna (optional)

    Returns:
        Regridded xarray Dataset with oblique mercator coordinates

    Raises:
        RegridError: If regridding operation fails
    """
    try:
        axis = np.asarray(axis, dtype=float)
        target_x, target_y = axis, axis.copy()

        proj = Proj(
            f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        )

        ds = dataset.copy()
        if "y" in ds.dims and "x" in ds.dims:
            ds = ds.rename({"y": "lat", "x": "lon"})

        ys, xs = np.meshgrid(target_y, target_x, indexing="ij")
        lons, lats = proj(xs.ravel(), ys.ravel(), inverse=True)
        lonlats = xr.Dataset(
            {
                "lon": (("y", "x"), lons.reshape(xs.shape)),
                "lat": (("y", "x"), lats.reshape(xs.shape)),
            }
        )

        ds_regridded = ds.interp(
            lonlats,
            method=interpolation_method,
            kwargs={"bounds_error": False, "fill_value": None},
        )
        ds_regridded = ds_regridded.assign_coords({"y": target_y, "x": target_x})

        if fillna_value is not None and data_variable is not None:
            ds_regridded[data_variable] = ds_regridded[data_variable].fillna(
                fillna_value
            )

        ds_regridded = ds_regridded.drop_vars(["lon", "lat"], errors="ignore")

        return ds_regridded

    except Exception as e:
        n = len(np.atleast_1d(axis))
        raise RegridError(
            f"Failed to regrid dataset onto a {n}x{n} axis", "regridding", e
        )
