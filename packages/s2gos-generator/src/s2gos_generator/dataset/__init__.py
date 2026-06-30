from dynaconf.utils.boxing import DynaBox

from .dataset import Dataset
from .geotiff_dem import GeoTiffDEM
from .indexed_geotiff import IndexedGeoTiff
from .zarr import Zarr

__all__ = [
    "dataset_factory",
    "Dataset",
    "GeoTiffDEM",
    "IndexedGeoTiff",
    "Zarr",
]


# key: str = <dataset type>
# value: Class = <dataset class>, should implement a from_settings function.
# Note: Consider moving to a fully fledge factory if we need external Dataset extension.
_id_to_class = {
    "indexed-geotiff": IndexedGeoTiff,
    "zarr": Zarr,
    "geotiff-dem": GeoTiffDEM,
}


def dataset_factory(settings: dict | DynaBox, name=None) -> Dataset:
    try:
        dataset_class = _id_to_class[settings["type"]]
    except KeyError:
        raise ValueError(
            f"This dataset settings type does not exist: {settings['type']}"
        )
    return dataset_class.from_settings(settings, name)
