"""Terrain elevation querying utilities for scene simulations.

This module provides a unified interface for querying terrain elevation from
DEM (Digital Elevation Model) data, eliminating code duplication across
different processors and backends.
"""

import logging
from typing import Optional, Tuple

import xarray as xr
from s2gos_utils.io.paths import UPath
from s2gos_utils.scene.description import SceneDescription
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class TerrainQuery:
    """Unified terrain elevation querying for simulation backends.

    This class provides methods to:
    - Find DEM files in scene directories
    - Query elevation at geographic coordinates (lat/lon)
    - Query elevation at scene coordinates (x, y)
    - Validate coordinate bounds
    """

    def __init__(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ):
        """Initialize terrain query for a specific scene.

        Args:
            scene_description: Scene description containing metadata
            scene_dir: Scene directory containing DEM data
        """
        self.scene_description = scene_description
        self.scene_dir = scene_dir
        self._dem_path: Optional[UPath] = None
        self._dem_data: Optional[xr.DataArray] = None

    def find_dem_file(self) -> Optional[UPath]:
        """Find DEM file in scene data directory.

        Searches for DEM files in the following order:
        1. dem_{scene_name}_{resolution}m.zarr
        2. dem_{scene_name}.zarr

        Returns:
            Path to DEM file or None if not found
        """
        if self._dem_path is not None:
            return self._dem_path

        scene_name = self.scene_description.name
        resolution = self.scene_description.resolution_m

        # Try resolution-specific DEM first
        dem_path = self.scene_dir / "data" / f"dem_{scene_name}_{resolution}m.zarr"
        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            self._dem_path = dem_path
            return dem_path

        # Fall back to generic DEM
        dem_path = self.scene_dir / "data" / f"dem_{scene_name}.zarr"
        if dem_path.exists():
            logger.debug(f"Found DEM at: {dem_path}")
            self._dem_path = dem_path
            return dem_path

        logger.warning(
            f"DEM not found in {self.scene_dir / 'data'}. "
            f"Tried: dem_{scene_name}_{resolution}m.zarr, dem_{scene_name}.zarr"
        )
        return None

    def query_elevation_at_scene_coords(
        self,
        x: float,
        y: float,
        raise_on_error: bool = True,
    ) -> float:
        """Query terrain elevation at scene coordinates (x, y) in meters.

        Args:
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters
            raise_on_error: If True, raise exceptions. If False, return 0.0 on error.

        Returns:
            Elevation in meters at (x, y)

        Raises:
            FileNotFoundError: If DEM file not found (only if raise_on_error=True)
        """
        dem_path = self.find_dem_file()

        if dem_path is None:
            msg = (
                f"DEM file not found in {self.scene_dir / 'data'}. "
                f"Cannot query terrain elevation."
            )
            if raise_on_error:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg + " Using 0.0m.")
                return 0.0

        try:
            # Load DEM data
            with xr.open_zarr(dem_path) as dem_ds:
                dem_data = dem_ds["elevation"]

                interpolator = RegularGridInterpolator(
                    (dem_data.y.values, dem_data.x.values),
                    dem_data.values,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )

                elevation = float(interpolator([(y, x)])[0])

            logger.debug(
                f"DEM elevation at scene coords ({x:.1f}m, {y:.1f}m): {elevation:.2f}m"
            )
            return elevation

        except Exception as e:
            msg = f"DEM query failed at ({x}, {y}): {e}"
            if raise_on_error:
                raise RuntimeError(msg) from e
            else:
                logger.warning(msg + " Using 0.0m.")
                return 0.0

    def query_elevation_at_geographic_coords(
        self,
        lat: float,
        lon: float,
        raise_on_error: bool = False,
    ) -> float:
        """Query terrain elevation at geographic coordinates (lat, lon).

        This method handles coordinate transformation from lat/lon to scene
        coordinates before querying the DEM.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            raise_on_error: If True, raise exceptions. If False, return 0.0 on error.

        Returns:
            Elevation in meters at (lat, lon)
        """
        # Transform lat/lon to scene coordinates
        try:
            x, y = self.latlon_to_scene(lat, lon)
            logger.debug(
                f"Geographic coords ({lat:.6f}°, {lon:.6f}°) -> "
                f"scene coords ({x:.1f}m, {y:.1f}m)"
            )
        except Exception as e:
            msg = f"Coordinate transformation failed for ({lat}, {lon}): {e}"
            if raise_on_error:
                raise RuntimeError(msg) from e
            else:
                logger.warning(msg + " Using 0.0m elevation.")
                return 0.0

        return self.query_elevation_at_scene_coords(x, y, raise_on_error=raise_on_error)

    def latlon_to_scene(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert geographic coordinates to scene coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple of (x, y) scene coordinates in meters

        Raises:
            ValueError: If coordinate transformation fails
        """
        from s2gos_utils.coordinates import CoordinateSystem

        center_lat = self.scene_description.location["center_lat"]
        center_lon = self.scene_description.location["center_lon"]

        coord_system = CoordinateSystem(center_lat, center_lon)
        x, y = coord_system.latlon_to_scene(lat, lon)

        return x, y

    def validate_coordinate_bounds(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 50.0,
    ) -> bool:
        """Validate that coordinates are within acceptable bounds.

        The local tangent plane approximation used for coordinate transformation
        breaks down at large distances (typically >50km from scene center).

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            max_distance_km: Maximum allowed distance from scene center in km

        Returns:
            True if coordinates are valid, False otherwise
        """
        try:
            x, y = self.latlon_to_scene(lat, lon)
            distance_m = (x**2 + y**2) ** 0.5
            distance_km = distance_m / 1000.0

            if distance_km > max_distance_km:
                logger.warning(
                    f"Coordinate ({lat}, {lon}) is {distance_km:.1f}km from scene center. "
                    f"Local tangent plane approximation may be inaccurate (>{max_distance_km}km limit)."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Coordinate validation failed: {e}")
            return False

    def get_dem_bounds(self) -> Optional[dict]:
        """Get DEM spatial bounds in scene coordinates.

        Returns:
            Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' or None if DEM not found
        """
        dem_path = self.find_dem_file()
        if dem_path is None:
            return None

        try:
            with xr.open_zarr(dem_path) as dem_ds:
                dem_data = dem_ds["elevation"]
                return {
                    "x_min": float(dem_data.x.min()),
                    "x_max": float(dem_data.x.max()),
                    "y_min": float(dem_data.y.min()),
                    "y_max": float(dem_data.y.max()),
                }
        except Exception as e:
            logger.error(f"Failed to get DEM bounds: {e}")
            return None

    def validate_scene_coordinates(
        self,
        x: float,
        y: float,
    ) -> bool:
        """Validate that scene coordinates are within DEM bounds.

        Args:
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters

        Returns:
            True if coordinates are within DEM bounds, False otherwise
        """
        bounds = self.get_dem_bounds()
        if bounds is None:
            logger.warning("Cannot validate coordinates - DEM not found")
            return False

        if not (bounds["x_min"] <= x <= bounds["x_max"]):
            logger.warning(
                f"X coordinate {x}m outside DEM bounds "
                f"[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}]m"
            )
            return False

        if not (bounds["y_min"] <= y <= bounds["y_max"]):
            logger.warning(
                f"Y coordinate {y}m outside DEM bounds "
                f"[{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]m"
            )
            return False

        return True

    def resolve_height(
        self,
        x: float,
        y: float,
        z: float,
        terrain_relative: bool,
    ) -> float:
        """Resolve z-coordinate to absolute height.

        Args:
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters
            z: Height value in meters
            terrain_relative: If True, z is offset above terrain elevation.
                              If False, z is absolute elevation.

        Returns:
            Absolute height in meters
        """
        if not terrain_relative:
            return z

        terrain_elev = self.query_elevation_at_scene_coords(x, y, raise_on_error=False)
        return terrain_elev + z

    def resolve_origin_target(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        target_x: Optional[float],
        target_y: Optional[float],
        target_z: Optional[float],
        origin_terrain_relative: bool,
        target_terrain_relative: bool = False,
    ) -> Tuple[list, Optional[list]]:
        """Resolve origin and target to absolute coordinates.

        Args:
            origin_x: Origin x-coordinate in meters
            origin_y: Origin y-coordinate in meters
            origin_z: Origin z-coordinate in meters
            target_x: Target x-coordinate in meters (None if no target)
            target_y: Target y-coordinate in meters (None if no target)
            target_z: Target z-coordinate in meters (None if no target)
            origin_terrain_relative: If True, origin z is offset above terrain
            target_terrain_relative: If True, target z is offset above terrain

        Returns:
            Tuple of (origin, target) where:
            - origin is [x, y, z] with absolute z
            - target is [x, y, z] with absolute z, or None if no target specified
        """
        abs_origin_z = self.resolve_height(
            origin_x, origin_y, origin_z, origin_terrain_relative
        )
        origin = [origin_x, origin_y, abs_origin_z]

        target = None
        if target_x is not None and target_y is not None:
            t_z = target_z if target_z is not None else 0.0
            abs_target_z = self.resolve_height(
                target_x, target_y, t_z, target_terrain_relative
            )
            target = [target_x, target_y, abs_target_z]

        return origin, target

    def resolve_viewing_geometry(
        self,
        viewing,
        terrain_relative: bool,
    ) -> Tuple[list, Optional[list]]:
        """Resolve viewing geometry to absolute coordinates.

        Convenience method that extracts origin/target from viewing geometry
        objects and resolves them to absolute coordinates.

        Args:
            viewing: LookAtViewing or AngularFromOriginViewing object
            terrain_relative: Whether to apply terrain-relative adjustment

        Returns:
            Tuple of (origin, target) with absolute z-coordinates

        Note:
            For AngularFromOriginViewing, target is None (direction computed from angles).
            For LookAtViewing, both origin and target are resolved.
        """
        from .config import AngularFromOriginViewing, LookAtViewing

        if isinstance(viewing, LookAtViewing):
            return self.resolve_origin_target(
                origin_x=viewing.origin[0],
                origin_y=viewing.origin[1],
                origin_z=viewing.origin[2],
                target_x=viewing.target[0],
                target_y=viewing.target[1],
                target_z=viewing.target[2],
                origin_terrain_relative=terrain_relative,
                target_terrain_relative=terrain_relative,
            )
        elif isinstance(viewing, AngularFromOriginViewing):
            abs_origin_z = self.resolve_height(
                viewing.origin[0],
                viewing.origin[1],
                viewing.origin[2],
                terrain_relative,
            )
            origin = [viewing.origin[0], viewing.origin[1], abs_origin_z]
            return origin, None  # Target computed from angles
        else:
            # Passthrough for unsupported viewing types
            logger.warning(
                f"Unsupported viewing type for terrain resolution: {type(viewing)}"
            )
            if hasattr(viewing, "origin"):
                return list(viewing.origin), getattr(viewing, "target", None)
            return [0, 0, 0], None
