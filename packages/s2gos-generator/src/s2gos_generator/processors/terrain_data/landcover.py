from typing import Optional, Union

import xarray as xr
from shapely.geometry import Polygon
from upath import UPath

from .base_processor import BaseTileProcessor
from ...core.grid import SceneGrid
from ...dataset import Dataset, IndexedGeoTiff, Zarr

ESA_CLASS_PERMANENT_WATER = 80


class LandCoverProcessor(BaseTileProcessor):
    """Finds, merges, and processes ESA WorldCover land cover tiles for a given AOI."""

    def __init__(self, dataset: Dataset):
        """Initialize the land cover processor."""
        super().__init__(dataset)

    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        var_name = self.dataset.variable_name
        return var_name if var_name is not None else "landcover"

    @property
    def default_interpolation_method(self) -> str:
        """Default interpolation method for land cover regridding."""
        return "nearest"

    @property
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (landcover uses uint8)."""
        return "uint8"

    @property
    def default_fill_value(self) -> Union[float, int]:
        """Fill for missing landcover, as an ESA class code, not a material index."""
        return ESA_CLASS_PERMANENT_WATER

    @property
    def use_context_manager(self) -> bool:
        """Landcover processor uses direct assignment for file opening."""
        return False

    def generate_landcover(
        self,
        aoi_polygon: Polygon,
        output_path: UPath,
        grid: SceneGrid,
        center_lat: float,
        center_lon: float,
    ) -> xr.Dataset:
        """Generate landcover data on grid, sampled at its cell centres.

        Args:
            aoi_polygon: Area of interest polygon
            output_path: Path where to save the processed landcover data
            grid: Target :class:`SceneGrid` for the regridded output
            center_lat: Center latitude for projection
            center_lon: Center longitude for projection

        Returns:
            Processed landcover dataset
        """

        source_aoi = self._source_aoi(aoi_polygon, grid.half_size_m)
        tile_paths = self.dataset.query(source_aoi)

        if isinstance(self.dataset, IndexedGeoTiff):
            # Pass AOI to merge for early spatial filtering
            merged_landcover = self._merge_tiles(tile_paths, source_aoi)
            merged_landcover = merged_landcover.persist()
        elif isinstance(self.dataset, Zarr):
            merged_landcover = self.dataset.open()
        else:
            raise NotImplementedError(
                "This type of dataset is not supported for landcovers."
            )

        clipped_landcover = self._clip_to_aoi(merged_landcover, source_aoi)

        clipped_landcover = self._regrid_data(
            clipped_landcover,
            grid.cell_centres(),
            center_lat,
            center_lon,
            fillna_value=self.default_fill_value,
        )

        clipped_landcover = clipped_landcover.rename(
            {self.data_variable_name: "landcover"}
        )

        self._save_dataset(clipped_landcover, output_path)

        return clipped_landcover


ESA_LANDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
