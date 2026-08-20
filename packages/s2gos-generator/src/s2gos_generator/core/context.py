"""Scene-specific resource context for pipeline execution."""

import json
import logging
from typing import Dict, List, Optional

from shapely.geometry import box
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
        self.dem_resolution_m = config.dem_resolution_m
        self.landcover_resolution_m = config.landcover_resolution_m

        # Scene-specific data
        self.assets = SceneAssets()
        self.additional_material_libraries = additional_material_libraries or []
        self.scene_description: Optional[object] = None

        # AOI polygon storage for geometric operations
        self._target_aoi_polygon: Optional[object] = None
        self._target_scene_bounds: Optional[object] = None
        self._buffer_aoi_polygon: Optional[object] = None
        self._background_aoi_polygon: Optional[object] = None

        self._coord_system: Optional[object] = None
        self._exclusion_zone_geometries: Optional[list] = None
        self._ways: Optional[list] = None
        self._way_polygons_by_material: Optional[dict] = None
        self._matched_materials: Optional[dict] = None

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
        return self._target_aoi_polygon

    @property
    def target_scene_bounds(self):
        """Lazy axis-aligned scene-coord clip bounds for the target AOI."""
        if self._target_scene_bounds is None:
            half = (self.aoi_size_km * 1000) / 2
            self._target_scene_bounds = box(-half, -half, half, half)
        return self._target_scene_bounds

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

    def _load_ways_from_sidecar(self) -> list:
        from ..processors.ways import ways_from_sidecar

        if self.assets.ways_file is None:
            return []
        try:
            with open(str(self.assets.ways_file), "r") as f:
                data = json.load(f)
            return ways_from_sidecar(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logging.warning("Failed to load ways from sidecar: %s", exc)
            return []

    @property
    def ways(self) -> list:
        """All way segments, lazily loaded from the ways sidecar."""
        if self._ways is None:
            self._ways = self._load_ways_from_sidecar()
        return self._ways

    @property
    def way_polygons_by_material(self) -> dict:
        """Merged way footprints per material, derived from the ways list.

        Computed once and cached. Each value is the unary_union of all buffered
        centerlines for that material — the geometry the texture painter needs,
        without storing it redundantly in the sidecar.
        """
        if self._way_polygons_by_material is None:
            from shapely.ops import unary_union

            by_mat: dict[str, list] = {}
            for way in self.ways:
                poly = way.centerline.buffer(way.width / 2, cap_style="flat")
                by_mat.setdefault(way.material, []).append(poly)
            self._way_polygons_by_material = {
                mat: unary_union(polys) for mat, polys in by_mat.items()
            }
        return self._way_polygons_by_material

    def _load_matched_materials_sidecar(self) -> dict:
        from ..processors.spectral.diversify import matched_materials_from_sidecar

        if self.assets.matched_materials_file is None:
            return {}
        try:
            with open(str(self.assets.matched_materials_file), "r") as f:
                data = json.load(f)
            return matched_materials_from_sidecar(data)
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Failed to load matched materials sidecar: %s", exc)
            return {}

    @property
    def matched_materials(self) -> dict:
        """Spectral-matching result, lazily loaded."""
        if self._matched_materials is None:
            self._matched_materials = self._load_matched_materials_sidecar()
        return self._matched_materials

    @property
    def exclusion_zone_geometries(self) -> list:
        """Exclusion zone geometries in scene coordinates, derived from config."""
        if self._exclusion_zone_geometries is None:
            self._exclusion_zone_geometries = _build_exclusion_zone_geometries(
                self.config,
                self.coordinate_system,
            )
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
