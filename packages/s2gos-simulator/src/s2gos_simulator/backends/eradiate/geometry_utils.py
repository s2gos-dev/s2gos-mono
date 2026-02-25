"""Geometry and viewing utilities for Eradiate backend."""

import logging
from typing import Optional, Union

import numpy as np
from s2gos_utils.scene import SceneDescription
from upath import UPath

from ...config import AngularFromOriginViewing, LookAtViewing


def sanitize_sensor_id(sensor_id: str) -> str:
    """Sanitize sensor ID for Eradiate kernel compatibility.

    Replaces dots with underscores to prevent Eradiate from interpreting
    them as nested dictionary path separators.

    Example:
        'hypstar_wl500.3nm' → 'hypstar_wl500_3nm'

    Args:
        sensor_id: Original sensor ID

    Returns:
        Sanitized sensor ID safe for Eradiate kernel
    """
    return sensor_id.replace(".", "_") if sensor_id else sensor_id


class GeometryUtils:
    """Utilities for handling geometry, viewing angles, and terrain queries."""

    def __init__(self):
        """Initialize geometry utilities."""
        pass

    def query_terrain_elevation(
        self, scene_description: SceneDescription, scene_dir: UPath, x: float, y: float
    ) -> float:
        """Query terrain elevation at scene coordinates (x, y) in meters.

        Args:
            scene_description: Scene description containing metadata
            scene_dir: Scene directory containing DEM data
            x: Scene x-coordinate in meters
            y: Scene y-coordinate in meters

        Returns:
            Elevation in meters at (x, y)

        Raises:
            FileNotFoundError: If DEM file not found
        """
        from s2gos_simulator.terrain_query import TerrainQuery

        terrain_query = TerrainQuery(scene_description, scene_dir)
        return terrain_query.query_elevation_at_scene_coords(x, y, raise_on_error=True)

    def adjust_origin_target_for_terrain(
        self,
        viewing,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> tuple[list[float], Optional[list[float]]]:
        """Adjust origin and optionally target for terrain-relative positioning.

        Checks viewing geometry's terrain_relative_height flag and adjusts
        z-coordinates relative to terrain elevation from DEM.

        Args:
            viewing: Viewing geometry (LookAtViewing, AngularFromOriginViewing, etc.)
            scene_description: Scene description for terrain queries
            scene_dir: Scene directory for DEM access

        Returns:
            Tuple of (adjusted_origin, adjusted_target_or_None)
            - adjusted_origin: Origin with z adjusted for terrain elevation if flag set
            - adjusted_target: Target adjusted for terrain if LookAtViewing with flag set, else None
        """
        if not getattr(viewing, "terrain_relative_height", False):
            origin = list(viewing.origin)
            target = list(viewing.target) if hasattr(viewing, "target") else None
            return origin, target

        origin = list(viewing.origin)
        x, y, z_offset = origin
        terrain_elevation = self.query_terrain_elevation(
            scene_description, scene_dir, x, y
        )
        absolute_z = terrain_elevation + z_offset
        origin[2] = absolute_z

        logging.debug(
            f"Viewing origin: terrain={terrain_elevation:.2f}m, "
            f"offset={z_offset:.2f}m, final_z={absolute_z:.2f}m"
        )

        target = None
        if isinstance(viewing, LookAtViewing):
            target = list(viewing.target)
            target_x, target_y, target_z_offset = target
            target_terrain_elevation = self.query_terrain_elevation(
                scene_description, scene_dir, target_x, target_y
            )
            target_absolute_z = target_terrain_elevation + target_z_offset
            target[2] = target_absolute_z

            logging.debug(
                f"Viewing target: terrain={target_terrain_elevation:.2f}m, "
                f"offset={target_z_offset:.2f}m, final_z={target_absolute_z:.2f}m"
            )

        return origin, target

    def calculate_target_from_angles(
        self, view: AngularFromOriginViewing
    ) -> tuple[list[float], list[float]]:
        """Calculate a target point and direction vector from angular viewing.

        Args:
            view: AngularFromOriginViewing object with origin, zenith, and azimuth

        Returns:
            A tuple containing (target_position, direction_vector).
        """
        zen_rad = np.deg2rad(view.zenith)
        az_rad = np.deg2rad(view.azimuth)

        direction = np.array(
            [
                np.sin(zen_rad) * np.cos(az_rad),
                np.sin(zen_rad) * np.sin(az_rad),
                np.cos(zen_rad),
            ]
        )

        origin_vec = np.array(view.origin)
        target_vec = origin_vec + direction * 1000.0

        return target_vec.tolist(), direction.tolist()

    def viewing_angles_to_direction(self, zenith: float, azimuth: float) -> list[float]:
        """Convert viewing angles to direction vector for distant measure.

        Args:
            zenith: Viewing zenith angle in degrees (0° = nadir, 90° = horizon)
            azimuth: Viewing azimuth angle in degrees (0° = North, 90° = East)

        Returns:
            Direction vector [x, y, z] pointing from distant sensor toward scene.
            Uses coordinate system: x = East, y = North, z = Up.
        """
        zen_rad = np.deg2rad(zenith)
        az_rad = np.deg2rad(azimuth)

        direction = [
            np.sin(zen_rad) * np.sin(az_rad),
            np.sin(zen_rad) * np.cos(az_rad),
            -np.cos(zen_rad),
        ]

        return direction

    def resolve_viewing_geometry(
        self,
        config,
        viewing: Union[LookAtViewing, AngularFromOriginViewing],
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> tuple[list[float], list[float]]:
        """Resolve viewing geometry to origin and target coordinates.

        Applies terrain-relative adjustment if viewing geometry has the flag set.

        Args:
            config: Measurement config (for future use)
            viewing: Viewing geometry specification
            scene_description: Scene description for terrain elevation queries
            scene_dir: Path to scene directory containing DEM

        Returns:
            Tuple of (origin, target) as lists of [x, y, z] coordinates
        """
        origin, target = self.adjust_origin_target_for_terrain(
            viewing, scene_description, scene_dir
        )

        if isinstance(viewing, AngularFromOriginViewing) and target is None:
            target, _ = self.calculate_target_from_angles(viewing)

        return origin, target


def apply_asset_relative_transform(
    relative_position: list[float],
    asset_transform: "AssetTransform",
) -> list[float]:
    """Transform position from asset local space to scene space.

    Applies rotation, scale, and translation from asset transform.

    Args:
        relative_position: [x, y, z] offset in asset's local coordinate system
        asset_transform: Asset's transform (position, rotation, scale)

    Returns:
        Absolute position [x, y, z] in scene coordinates
    """
    from scipy.spatial.transform import Rotation

    rel_pos = np.array(relative_position, dtype=float)

    rx, ry, rz = asset_transform.rotation
    rotation = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)

    rotated = rotation.apply(rel_pos)
    scaled = rotated * asset_transform.scale
    absolute = scaled + np.array(asset_transform.position)

    return absolute.tolist()
