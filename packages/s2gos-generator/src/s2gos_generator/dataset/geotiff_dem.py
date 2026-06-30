"""Single-GeoTIFF DEM dataset.

Wraps a single Digital Elevation Model GeoTIFF (in any CRS). Unlike
:class:`IndexedGeoTiff`, there is no tile index and no spatial query: the file
*is* the scene. Its footprint, centre and native resolution can be read with
:meth:`geo_extent` to drive scene generation, and :meth:`open` reprojects it to
WGS84 so it flows through the same clip + regrid path as every other DEM source.
"""

from typing import Any

import rioxarray as rxr
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, field_validator
from s2gos_utils.io import PathRef, resolver
from s2gos_utils.setting import to_pathref
from shapely import Polygon

from .dataset import Dataset


class GeoTiffDEM(Dataset):
    """Single GeoTIFF DEM of arbitrary CRS and extent.

    Attributes:
        path: Path to the GeoTIFF on disk (or any fsspec-supported location).
        variable_name: Name the elevation variable is exposed under (the DEM
            processor renames it to ``"elevation"``).
        crs: Fallback CRS used only when the file itself carries no CRS.
    """

    path: PathRef = Field()
    variable_name: str = Field(default="elevation")

    @classmethod
    def from_settings(cls, settings: DynaBox | dict, name: str):
        return cls(
            name=name,
            crs=settings.get("crs", "EPSG:4326"),
            path=to_pathref(settings["path"]),
            variable_name=settings.get("variable_name", "elevation"),
        )

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files exist."""
        path = resolver.resolve(v)
        if not path.exists() and path.protocol == "file":
            raise ValueError(f"Path does not exist: {v}")
        return v

    def _open_rioxarray(self) -> xr.DataArray:
        """Open the GeoTIFF as a single-band DataArray with a known CRS."""
        da = rxr.open_rasterio(str(resolver.resolve(self.path)), masked=True)
        if da.rio.crs is None:
            da = da.rio.write_crs(self.crs)
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        return da

    def open(self, path: PathRef | None = None, **kwargs: Any) -> xr.Dataset:
        """Open the DEM, reprojected to WGS84, as an ``xarray.Dataset``.

        The ``path`` argument is accepted for interface compatibility but
        ignored. The returned dataset has ``x``/``y`` dimensions carrying
        longitude/latitude in EPSG:4326 and a single elevation variable — the
        shape the clip/regrid pipeline already consumes.
        """
        da = self._open_rioxarray().rio.reproject("EPSG:4326")
        return da.rename(self.variable_name).to_dataset()

    def query(self, polygon: Polygon, **kwargs: Any) -> list[PathRef]:
        """Return the single DEM path (the file defines its own extent)."""
        return [self.path]

    def geo_extent(self) -> dict:
        """Derive scene geometry from the DEM's footprint.

        Reads the native bounds + CRS, finds the WGS84 centre, then projects the
        footprint into a scene-local oblique-mercator frame at that centre to get
        an axis-aligned ``width_km`` x ``height_km`` extent. Native resolution is
        the effective metres-per-pixel of that extent.

        Returns:
            Dict with ``center_lat``, ``center_lon``, ``width_km``, ``height_km``
            and ``native_resolution_m``.
        """
        from pyproj import Transformer
        from s2gos_utils.coordinates import CoordinateSystem

        da = self._open_rioxarray()
        src_crs = da.rio.crs
        left, bottom, right, top = da.rio.bounds()
        height_px, width_px = da.rio.shape

        to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs84.transform(
            (left + right) / 2, (bottom + top) / 2
        )

        # Project the four corners into a scene frame centred on the DEM and take
        # the axis-aligned bounding box as the rectangular extent.
        coords = CoordinateSystem(center_lat=center_lat, center_lon=center_lon)
        corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
        xs, ys = [], []
        for cx, cy in corners:
            lon, lat = to_wgs84.transform(cx, cy)
            sx, sy = coords.latlon_to_scene(lat, lon)
            xs.append(sx)
            ys.append(sy)

        width_m = max(xs) - min(xs)
        height_m = max(ys) - min(ys)
        native_resolution_m = min(width_m / width_px, height_m / height_px)

        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "width_km": width_m / 1000.0,
            "height_km": height_m / 1000.0,
            "native_resolution_m": native_resolution_m,
        }
