"""Scene-specific resource context for pipeline execution."""

import logging
from typing import Dict, List, Optional

from upath import UPath

from .assets import SceneAssets
from .config import SceneGenConfig


class SceneResourceContext:
    """Resource context for scene generation pipeline execution."""

    def __init__(
        self,
        config: SceneGenConfig,
        combined_user_assets: List = None,
        additional_material_libraries: List = None,
        **kwargs,
    ):
        """Initialize scene resource context.

        Args:
            config: Scene generation configuration
            combined_user_assets: List combining config + XML assets
            additional_material_libraries: Extra material libraries from XML
        """

        # Core configuration
        self.config = config
        self.dependency_outputs: Dict[str, UPath | None] = {}
        self.kwargs = kwargs

        # Asset management
        self.config_assets = list(config.user_assets)
        self.xml_assets = (
            combined_user_assets[len(config.user_assets) :]
            if combined_user_assets
            else []
        )

        # Direct computed properties from config
        self.output_dir = config.scene_output_dir.upath
        self.data_dir = config.data_dir.upath
        self.meshes_dir = config.meshes_dir.upath
        self.textures_dir = config.textures_dir.upath
        self.scene_name = config.scene_name
        self.center_lat = config.location.center_lat
        self.center_lon = config.location.center_lon
        self.aoi_size_km = config.location.aoi_size_km
        self.target_resolution_m = config.target_resolution_m

        # Scene-specific data
        self.assets = SceneAssets()
        self.additional_material_libraries = additional_material_libraries or []
        self.scene_description: Optional[object] = None

        # AOI polygon storage for geometric operations
        self._target_aoi_polygon: Optional[object] = None
        self._buffer_aoi_polygon: Optional[object] = None
        self._background_aoi_polygon: Optional[object] = None

        self._coord_system: Optional[object] = None
        self._exclusion_zone_geometries: Optional[list] = None
        self._road_geometries: Optional[list] = None

    @property
    def user_assets(self):
        """Get combined user assets (config + XML assets)."""
        return self.config_assets + self.xml_assets

    @property
    def has_buffer(self) -> bool:
        """Check if buffer processing is enabled."""
        return self.config.buffer is not None

    @property
    def has_background(self) -> bool:
        """Check if background processing is enabled."""
        return self.config.background is not None

    @property
    def has_hamster(self) -> bool:
        """Check if HAMSTER data integration is enabled."""
        return self.config.hamster is not None and self.config.hamster.enabled

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        """Get cached coordinate system for this scene.

        Returns:
            CoordinateSystem instance for this scene's center location
        """
        if self._coord_system is None:
            from s2gos_utils.coordinates import CoordinateSystem

            self._coord_system = CoordinateSystem(
                center_lat=self.center_lat, center_lon=self.center_lon
            )
        return self._coord_system

    @property
    def target_aoi_polygon(self):
        """Lazy AOI polygon for the target area."""
        if self._target_aoi_polygon is None:
            self._target_aoi_polygon = self.coordinate_system.create_scene_polygon(
                self.aoi_size_km
            )
            corners = list(self._target_aoi_polygon.exterior.coords[:-1])
            logging.info("AOI corners (lon, lat):")
            for i, (lon, lat) in enumerate(corners):
                logging.info("  Corner %d: (%.6f, %.6f)", i + 1, lat, lon)
            logging.info(
                "AOI polygon: %.1fkm × %.1fkm at (%.6f, %.6f)",
                self.aoi_size_km,
                self.aoi_size_km,
                self.center_lat,
                self.center_lon,
            )
        return self._target_aoi_polygon

    @property
    def buffer_aoi_polygon(self):
        """Lazy AOI polygon for the buffer area, or None if unconfigured."""
        if self._buffer_aoi_polygon is None and self.config.buffer is not None:
            self._buffer_aoi_polygon = self.coordinate_system.create_scene_polygon(
                self.config.buffer.size_km
            )
        return self._buffer_aoi_polygon

    @property
    def background_aoi_polygon(self):
        """Lazy AOI polygon for the background area, or None if unconfigured."""
        if self._background_aoi_polygon is None and self.config.background is not None:
            self._background_aoi_polygon = self.coordinate_system.create_scene_polygon(
                self.config.background.size_km
            )
        return self._background_aoi_polygon

    @property
    def road_exclusion_geometries(self) -> list:
        """Road-based exclusion zone geometries in scene coordinates."""
        roads_cfg = self.config.roads
        if roads_cfg is None or not roads_cfg.exclude_vegetation:
            return []

        # Load road geometries from sidecar file if not already set
        polygons = self._road_geometries
        if polygons is None and self.assets.roads_file is not None:
            try:
                import json

                from shapely.geometry import shape

                with open(str(self.assets.roads_file), "r") as f:
                    data = json.load(f)
                polygons = [shape(p) for p in data.get("polygons", [])]
            except Exception as exc:
                logging.warning("Failed to load road geometries from sidecar: %s", exc)
                return []

        if not polygons:
            return []

        buffer_m = roads_cfg.vegetation_buffer_m
        result = []
        for i, poly in enumerate(polygons):
            buffered = poly.buffer(buffer_m) if buffer_m > 0 else poly
            result.append({"source": f"road_{i}", "geometry": buffered})
        return result

    @property
    def exclusion_zone_geometries(self) -> list:
        """Exclusion zone geometries in scene coordinates, derived from config."""
        if self._exclusion_zone_geometries is None:
            self._exclusion_zone_geometries = _build_exclusion_zone_geometries(
                self.config,
                self.coordinate_system,
            )
            self._exclusion_zone_geometries.extend(self.road_exclusion_geometries)
        return self._exclusion_zone_geometries


def _build_exclusion_zone_geometries(config, coordinate_system) -> list:
    """Convert exclusion zone config objects to scene-coordinate Shapely geometries."""
    from shapely.geometry import Point, Polygon, box

    from .config import BoxGeometry, CircleGeometry, PolygonGeometry

    def _to_scene(coord, coord_type):
        if coord_type == "geographic":
            lon, lat = coord
            return coordinate_system.latlon_to_scene(lat, lon)
        return tuple(coord)

    def _make_zone_geometry(x, y, zone_spec):
        if isinstance(zone_spec, (int, float)):
            return Point(x, y).buffer(zone_spec)
        else:
            width, height = zone_spec
            hw, hh = width / 2, height / 2
            return box(x - hw, y - hh, x + hw, y + hh)

    result = []

    # Config-level vegetation exclusion zones
    for zone in config.vegetation_exclusion_zones:
        try:
            g = zone.geometry
            if isinstance(g, CircleGeometry):
                x, y = _to_scene(g.center, g.coord_type)
                geometry = Point(x, y).buffer(g.radius)
            elif isinstance(g, BoxGeometry):
                x, y = _to_scene(g.center, g.coord_type)
                hw, hh = g.width / 2, g.height / 2
                geometry = box(x - hw, y - hh, x + hw, y + hh)
            elif isinstance(g, PolygonGeometry):
                if g.coord_type == "geographic":
                    coords = [
                        coordinate_system.latlon_to_scene(lat, lon)
                        for lon, lat in g.coordinates
                    ]
                else:
                    coords = list(g.coordinates)
                geometry = Polygon(coords)
            else:
                logging.warning("Unknown geometry type for zone '%s'", zone.zone_id)
                continue
            result.append({"source": f"zone_{zone.zone_id}", "geometry": geometry})
            logging.info("Processed exclusion zone '%s'", zone.zone_id)
        except Exception as e:
            logging.warning(
                "Failed to process exclusion zone '%s': %s", zone.zone_id, e
            )

    # Per-asset exclusion zones
    for asset in config.user_assets:
        if asset.exclusion_zone is None:
            continue
        x, y = _to_scene(asset.coordinate, asset.coord_type)
        geometry = _make_zone_geometry(x, y, asset.exclusion_zone)
        result.append({"source": f"asset_{asset.object_id}", "geometry": geometry})
        logging.info("Processed asset exclusion zone for '%s'", asset.object_id)

    # Per-XML-scene exclusion zones
    for xml_scene in config.xml_scenes:
        if xml_scene.exclusion_zone is None:
            continue
        x, y = _to_scene(xml_scene.base_coordinate, xml_scene.coord_type)
        geometry = _make_zone_geometry(x, y, xml_scene.exclusion_zone)
        source = xml_scene.object_id_prefix or xml_scene.xml_path.upath.stem
        result.append({"source": f"xml_scene_{source}", "geometry": geometry})
        logging.info("Processed XML scene exclusion zone for '%s'", source)

    return result
