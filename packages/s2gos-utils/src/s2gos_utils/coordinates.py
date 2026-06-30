import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from upath import UPath


class CoordinateSystem:
    """
    Coordinate system for precise geometric operations.

    Uses oblique mercator projection centered on a reference point for accurate
    distance and area calculations within typical scene sizes (~10-200 km).

    The coordinate system is cached for performance - create once, use many times.
    """

    def __init__(self, center_lat: float, center_lon: float):
        """
        Initialize scene coordinate system centered at given location.

        Args:
            center_lat: Scene center latitude in WGS84 decimal degrees
            center_lon: Scene center longitude in WGS84 decimal degrees
        """
        self.center_lat = center_lat
        self.center_lon = center_lon

        self.wgs84_crs = CRS("EPSG:4326")
        self.scene_crs = CRS(
            f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        )

        self._to_scene_transformer = Transformer.from_crs(
            self.wgs84_crs, self.scene_crs, always_xy=True
        )
        self._from_scene_transformer = Transformer.from_crs(
            self.scene_crs, self.wgs84_crs, always_xy=True
        )

        self._center_x, self._center_y = self._to_scene_transformer.transform(
            center_lon, center_lat
        )

        logging.debug(
            f"Created CoordinateSystem for ({center_lat:.6f}, {center_lon:.6f})"
        )

    def latlon_to_scene(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert WGS84 lat/lon to scene-local coordinates in meters.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Tuple of (x, y) coordinates in meters relative to scene center
        """
        target_x, target_y = self._to_scene_transformer.transform(lon, lat)
        return (target_x - self._center_x, target_y - self._center_y)

    def geodataframe_to_scene_local(self, gdf):
        """Reproject a GeoDataFrame's geometries to scene-local meters.

        Equivalent to ``latlon_to_scene`` for whole geometries: reproject to the
        scene oblique-mercator CRS, then shift so the scene center is the origin.
        Returns a copy; the input is left untouched.

        Args:
            gdf: GeoDataFrame in any CRS.

        Returns:
            A new GeoDataFrame with geometries in scene-local coordinates (meters
            relative to the scene center).
        """
        gdf = gdf.to_crs(self.scene_crs).copy()
        gdf.geometry = gdf.geometry.translate(-self._center_x, -self._center_y)
        return gdf

    def scene_to_latlon(
        self,
        x: Union[float, "np.ndarray"],
        y: Union[float, "np.ndarray"],
    ) -> Tuple[Union[float, "np.ndarray"], Union[float, "np.ndarray"]]:
        """
        Convert scene-local coordinates back to WGS84 lat/lon.

        Accepts scalar floats or NumPy arrays (e.g. a meshgrid of scene coords).

        Args:
            x: X coordinate(s) in meters relative to scene center
            y: Y coordinate(s) in meters relative to scene center

        Returns:
            Tuple of (lat, lon) in decimal degrees
        """
        absolute_x = self._center_x + x
        absolute_y = self._center_y + y
        lon, lat = self._from_scene_transformer.transform(absolute_x, absolute_y)
        return (lat, lon)

    def create_rectangle(
        self, center_lat: float, center_lon: float, width_km: float, height_km: float
    ) -> Dict[str, float]:
        """
        Create rectangular bounds centered at specified coordinates.

        Args:
            center_lat: Rectangle center latitude in degrees
            center_lon: Rectangle center longitude in degrees
            width_km: Rectangle width in kilometers
            height_km: Rectangle height in kilometers

        Returns:
            Dictionary with 'xmin', 'xmax', 'ymin', 'ymax' in scene coordinates (meters)
        """
        center_x, center_y = self.latlon_to_scene(center_lat, center_lon)

        half_width_m = (width_km * 1000) / 2
        half_height_m = (height_km * 1000) / 2

        return {
            "xmin": center_x - half_width_m,
            "xmax": center_x + half_width_m,
            "ymin": center_y - half_height_m,
            "ymax": center_y + half_height_m,
        }

    def create_scene_polygon(
        self, width_km: float, height_km: Optional[float] = None
    ) -> Polygon:
        """
        Create a rectangular polygon in WGS84 coordinates centered on scene.

        Args:
            width_km: Width (x extent) of the rectangle in kilometers.
            height_km: Height (y extent) in kilometers. Defaults to ``width_km``,
                which yields a square (the historical behaviour).

        Returns:
            Rectangular polygon in WGS84 coordinates
        """
        if height_km is None:
            height_km = width_km

        half_width_m = (width_km * 1000) / 2
        half_height_m = (height_km * 1000) / 2

        corners_scene = [
            (-half_width_m, -half_height_m),
            (half_width_m, -half_height_m),
            (half_width_m, half_height_m),
            (-half_width_m, half_height_m),
        ]

        corners_latlon = []
        for x, y in corners_scene:
            lat, lon = self.scene_to_latlon(x, y)
            corners_latlon.append((lon, lat))  # Shapely uses (lon, lat) order

        return Polygon(corners_latlon)

    def create_polygon_from_coordinates(
        self, coordinates: list[Tuple[float, float]]
    ) -> Polygon:
        """
        Create polygon from list of lat/lon coordinate pairs.

        Args:
            coordinates: List of (lat, lon) tuples in decimal degrees

        Returns:
            Polygon in WGS84 coordinates
        """
        shapely_coords = [(lon, lat) for lat, lon in coordinates]
        return Polygon(shapely_coords)

    def query_height_from_dem(self, lat: float, lon: float, dem_path: UPath) -> float:
        """
        Query elevation from DEM dataset at specific coordinate.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            dem_path: Path to DEM zarr dataset

        Returns:
            Elevation value in meters

        Raises:
            ValueError: If coordinate is outside DEM bounds or DEM cannot be read
        """
        try:
            dem_dataset = xr.open_zarr(dem_path)
            elevation_data = dem_dataset["elevation"]

            scene_x, scene_y = self.latlon_to_scene(lat, lon)

            elevation = elevation_data.sel(
                x=scene_x, y=scene_y, method="nearest"
            ).values

            if hasattr(elevation, "item"):
                elevation = float(elevation.item())
            else:
                elevation = float(elevation)

            return elevation

        except (KeyError, IndexError, FileNotFoundError) as e:
            raise ValueError(
                f"Could not query elevation at ({lat:.6f}, {lon:.6f}). "
                f"Coordinate may be outside DEM bounds or DEM file not accessible. "
                f"Error: {e}"
            )

    def query_height_from_mesh(
        self, lat: float, lon: float, mesh_path: UPath
    ) -> Optional[float]:
        """
        Query height from 3D terrain mesh using ray-casting.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            mesh_path: Path to terrain mesh file (PLY, OBJ, etc.)

        Returns:
            Height in meters, or None if no intersection found
        """
        try:
            import trimesh
        except ImportError:
            logging.warning("trimesh not available for mesh-based height queries")
            return None

        try:
            mesh = trimesh.load(str(mesh_path))

            scene_x, scene_y = self.latlon_to_scene(lat, lon)

            # Create ray from high above, pointing down
            ray_origin = np.array([scene_x, scene_y, 10000.0])
            ray_direction = np.array([0.0, 0.0, -1.0])

            # Perform ray-mesh intersection
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=[ray_origin], ray_directions=[ray_direction]
            )

            if len(locations) > 0:
                # Return highest intersection (closest to ray origin)
                heights = locations[:, 2]
                return float(np.max(heights))
            else:
                return None

        except Exception as e:
            logging.warning(f"Failed to query height from mesh {mesh_path}: {e}")
            return None

    def query_height(
        self,
        lat: float,
        lon: float,
        dem_path: Optional[UPath] = None,
        mesh_path: Optional[UPath] = None,
        prefer_mesh: bool = True,
    ) -> Optional[float]:
        """
        Query height at coordinate, trying mesh first, then DEM.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            dem_path: Optional path to DEM dataset
            mesh_path: Optional path to terrain mesh
            prefer_mesh: Whether to try mesh before DEM

        Returns:
            Height in meters, or None if no data available
        """
        if prefer_mesh and mesh_path:
            height = self.query_height_from_mesh(lat, lon, mesh_path)
            if height is not None:
                return height

        if dem_path:
            try:
                return self.query_height_from_dem(lat, lon, dem_path)
            except ValueError:
                pass

        if not prefer_mesh and mesh_path:
            height = self.query_height_from_mesh(lat, lon, mesh_path)
            if height is not None:
                return height

        return None

    def spherical_to_cartesian(
        self,
        zenith_deg: float,
        azimuth_deg: float,
        distance: float,
        center_point: Tuple[float, float, float] = (0, 0, 0),
    ) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian position in scene coordinates.

        Args:
            zenith_deg: Zenith angle in degrees (0=down/nadir, 90=horizontal, 180=up/zenith)
            azimuth_deg: Azimuth angle in degrees (0=East, 90=North, 180=West, 270=South)
            distance: Distance from center point in meters
            center_point: Center point (x, y, z) in scene coordinates (default: origin)

        Returns:
            Tuple of (x, y, z) position in scene coordinates
        """
        zen_rad = np.radians(zenith_deg)
        az_rad = np.radians(azimuth_deg)

        dx = distance * np.sin(zen_rad) * np.cos(az_rad)
        dy = distance * np.sin(zen_rad) * np.sin(az_rad)
        dz = distance * np.cos(zen_rad)

        return (
            center_point[0] + dx,
            center_point[1] + dy,
            center_point[2] + dz,
        )

    @property
    def center_coordinates(self) -> Tuple[float, float]:
        """Get scene center coordinates as (lat, lon) tuple."""
        return (self.center_lat, self.center_lon)

    @property
    def projection_info(self) -> Dict[str, str]:
        """Get information about the coordinate projection."""
        return {
            "projection": "Oblique Mercator",
            "center_lat": f"{self.center_lat:.6f}",
            "center_lon": f"{self.center_lon:.6f}",
            "proj_string": f"+proj=omerc +lat_0={self.center_lat} +lonc={self.center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m",
            "use_case": "Scene-local coordinate operations with preserved distances and areas",
        }


def bounds_overlap(bounds1: Dict[str, float], bounds2: Dict[str, float]) -> bool:
    """
    Check if two rectangular bounds overlap.

    Args:
        bounds1: First bounds dictionary with 'xmin', 'xmax', 'ymin', 'ymax'
        bounds2: Second bounds dictionary with 'xmin', 'xmax', 'ymin', 'ymax'

    Returns:
        True if bounds overlap, False otherwise
    """
    return not (
        bounds1["xmax"] < bounds2["xmin"]
        or bounds1["xmin"] > bounds2["xmax"]
        or bounds1["ymax"] < bounds2["ymin"]
        or bounds1["ymin"] > bounds2["ymax"]
    )


def calculate_pixel_size(
    target_size_km: Union[float, Tuple[float, float]], film_resolution: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Calculate pixel size in meters for given target area and film resolution.

    Args:
        target_size_km: Target area size. Float for square (km), tuple for rectangular (width_km, height_km)
        film_resolution: Film resolution as (width_pixels, height_pixels)

    Returns:
        Tuple of (pixel_size_x_m, pixel_size_y_m) in meters per pixel
    """
    if isinstance(target_size_km, (int, float)):
        # Square area
        width_km = height_km = target_size_km
    else:
        # Rectangular area
        width_km, height_km = target_size_km

    pixel_size_x = (width_km * 1000) / film_resolution[0]
    pixel_size_y = (height_km * 1000) / film_resolution[1]

    return pixel_size_x, pixel_size_y


def pixel_to_scene_xy(
    row: int,
    col: int,
    bounds: Dict[str, float],
    resolution: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Convert pixel (row, col) indices to scene (x, y) coordinates at pixel center.

    Pixel indexing convention:
    - Row 0 is at the top (ymax), increasing row goes south (decreasing y)
    - Col 0 is at the left (xmin), increasing col goes east (increasing x)
    - Returns coordinates at pixel CENTER (offset by 0.5 pixels)

    Args:
        row: Pixel row index (0 = top/north)
        col: Pixel column index (0 = left/west)
        bounds: Scene bounds dict with 'xmin', 'xmax', 'ymin', 'ymax' in meters
        resolution: Film resolution as (width_pixels, height_pixels)

    Returns:
        Tuple of (x, y) scene coordinates in meters at pixel center
    """

    width, height = resolution
    x = bounds["xmin"] + ((col + 0.5) / width) * (bounds["xmax"] - bounds["xmin"])
    y = bounds["ymax"] - ((row + 0.5) / height) * (bounds["ymax"] - bounds["ymin"])

    return x, y
