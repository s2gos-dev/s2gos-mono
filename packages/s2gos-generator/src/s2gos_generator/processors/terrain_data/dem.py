from typing import Optional, Union

import xarray as xr
from shapely.geometry import Polygon
from upath import UPath

from .base_processor import BaseTileProcessor
from ...core.grid import SceneGrid
from ...dataset import Dataset, IndexedGeoTiff, Zarr


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
        grid: SceneGrid,
        center_lat: float,
        center_lon: float,
        fillna_value: Optional[float] = 0.0,
        flatten_dem: bool = False,
    ) -> xr.Dataset:
        """Generate DEM data on grid, sampled at its cell corners.

        Elevation is point-like, so it is sampled at :meth:`SceneGrid.nodes` rather
        than at cell centres.
        """

        source_aoi = self._source_aoi(aoi_polygon, grid.half_size_m)
        tile_paths = self.dataset.query(source_aoi)

        if isinstance(self.dataset, IndexedGeoTiff):
            # Pass AOI to merge for early spatial filtering
            merged_dem = self._merge_tiles(
                tile_paths, source_aoi, fillna_value=fillna_value
            )
        elif isinstance(self.dataset, Zarr):
            merged_dem = self.dataset.open()
        else:
            raise NotImplementedError("This type of dataset is not supported for DEMs.")

        clipped_dem = self._clip_to_aoi(merged_dem, source_aoi)

        clipped_dem = self._regrid_data(
            clipped_dem,
            grid.nodes(),
            center_lat,
            center_lon,
            fillna_value,
        )

        if flatten_dem:
            clipped_dem[self.data_variable_name] = xr.zeros_like(
                clipped_dem[self.data_variable_name]
            )

        clipped_dem = clipped_dem.rename({self.data_variable_name: "elevation"})

        self._save_dataset(clipped_dem, output_path)

        return clipped_dem
