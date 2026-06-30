import logging
from typing import Optional, Union

import xarray as xr
from shapely.geometry import Polygon
from upath import UPath

from .base_processor import BaseTileProcessor
from ..dataset import Dataset, GeoTiffDEM, IndexedGeoTiff, Zarr


class DEMProcessor(BaseTileProcessor):
    """Finds, merges, and processes Copernicus GLO-30 DEM tiles for a given AOI."""

    def __init__(self, dataset: Dataset):
        """Initialize the DEM processor."""
        super().__init__(dataset)

    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        var_name = self.dataset.variable_name
        return var_name if var_name is not None else "elevation"

    @property
    def default_interpolation_method(self) -> str:
        """Default interpolation method for DEM regridding."""
        return "linear"

    @property
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (DEM data stays as float)."""
        return None

    @property
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for NaN values in DEM data."""
        return 0.0

    @property
    def use_context_manager(self) -> bool:
        """DEM processor uses context manager for file opening."""
        return True

    def generate_dem(
        self,
        aoi_polygon: Polygon,
        output_path: UPath,
        fillna_value: Optional[float] = 0.0,
        target_resolution_m: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        width_km: Optional[float] = None,
        height_km: Optional[float] = None,
        flatten_dem: bool = False,
    ) -> xr.Dataset:
        """Generate DEM data for the AOI."""

        if isinstance(self.dataset, IndexedGeoTiff):
            # Pass AOI to merge for early spatial filtering
            tile_paths = self.dataset.query(aoi_polygon)
            merged_dem = self._merge_tiles(
                tile_paths, aoi_polygon, fillna_value=fillna_value
            )
        elif isinstance(self.dataset, (Zarr, GeoTiffDEM)):
            merged_dem = self.dataset.open()
        else:
            raise NotImplementedError("This type of dataset is not supported for DEMs.")

        # Clip to exact AOI geometry
        clipped_dem = self._clip_to_aoi(merged_dem, aoi_polygon)

        if (
            target_resolution_m is not None
            and center_lat is not None
            and center_lon is not None
            and width_km is not None
            and height_km is not None
        ):
            # For a single GeoTIFF the file may not cover the whole (axis-aligned)
            # scene rectangle, so regrid without filling first to measure where
            # valid data actually lands on the scene grid, log it, then fill.
            is_geotiff = isinstance(self.dataset, GeoTiffDEM)
            clipped_dem = self._regrid_data(
                clipped_dem,
                target_resolution_m,
                center_lat,
                center_lon,
                width_km,
                height_km,
                None if is_geotiff else fillna_value,
            )
            if is_geotiff:
                self._log_regridded_coverage(clipped_dem, width_km, height_km)
                if fillna_value is not None:
                    var = self.data_variable_name
                    clipped_dem[var] = clipped_dem[var].fillna(fillna_value)

        if flatten_dem:
            clipped_dem[self.data_variable_name] = xr.zeros_like(
                clipped_dem[self.data_variable_name]
            )

        # Rename data variable to a predictable name
        clipped_dem = clipped_dem.rename({self.data_variable_name: "elevation"})

        self._save_dataset(clipped_dem, output_path)

        return clipped_dem

    def _log_regridded_coverage(
        self, regridded: xr.Dataset, width_km: float, height_km: float
    ) -> None:
        """Report where valid (non-NaN) data lands on the regridded scene grid.

        Called for a single GeoTIFF before NaN-filling, so the user can see how
        much of the axis-aligned scene rectangle the file actually covers and
        the extent of the NaN padding introduced by the reprojection.
        """
        da = regridded[self.data_variable_name]
        valid = da.notnull()
        total = int(valid.size)
        n_valid = int(valid.sum())
        if total == 0:
            return
        pct = 100.0 * n_valid / total

        if n_valid == 0:
            logging.warning(
                "GeoTIFF DEM covers none of the %.2f×%.2f km scene grid after "
                "reprojection; the whole scene will be NaN-filled.",
                width_km,
                height_km,
            )
            return

        xs = regridded["x"].values
        ys = regridded["y"].values
        cols = valid.any("y").values
        rows = valid.any("x").values
        xmin, xmax = float(xs[cols].min()), float(xs[cols].max())
        ymin, ymax = float(ys[rows].min()), float(ys[rows].max())
        logging.info(
            "GeoTIFF DEM on the %.2f×%.2f km scene grid: valid data covers "
            "%.2f×%.2f km (%.1f%% of cells), bbox x∈[%.0f, %.0f] m / "
            "y∈[%.0f, %.0f] m about centre; the remaining %.1f%% is NaN-filled "
            "(clip to the bbox to drop the padding).",
            width_km,
            height_km,
            (xmax - xmin) / 1000.0,
            (ymax - ymin) / 1000.0,
            pct,
            xmin,
            xmax,
            ymin,
            ymax,
            100.0 - pct,
        )
