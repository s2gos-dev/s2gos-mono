from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class ResolvedExclusionZone:
    """An exclusion zone in scene coordinates."""

    source: str
    geometry: BaseGeometry
    excludes: frozenset[str]


def _to_scene(coord, coord_type, coordinate_system):
    if coord_type == "geographic":
        lon, lat = coord
        return coordinate_system.latlon_to_scene(lat, lon)
    return tuple(coord)


def _zone_geometry(zone, coordinate_system) -> BaseGeometry:
    from shapely.geometry import Point, Polygon, box

    from ..core.config import BoxGeometry, CircleGeometry, PolygonGeometry

    g = zone.geometry
    if isinstance(g, CircleGeometry):
        x, y = _to_scene(g.center, g.coord_type, coordinate_system)
        return Point(x, y).buffer(g.radius)
    if isinstance(g, BoxGeometry):
        x, y = _to_scene(g.center, g.coord_type, coordinate_system)
        hw, hh = g.width / 2, g.height / 2
        return box(x - hw, y - hh, x + hw, y + hh)
    if isinstance(g, PolygonGeometry):
        return Polygon(
            [_to_scene(c, g.coord_type, coordinate_system) for c in g.coordinates]
        )
    raise TypeError(f"Unknown geometry type for zone '{zone.zone_id}': {type(g)}")


def _object_zone_geometry(x, y, zone) -> BaseGeometry:
    from shapely.geometry import Point, box

    if zone.radius is not None:
        return Point(x, y).buffer(zone.radius)
    hw, hh = zone.size[0] / 2, zone.size[1] / 2
    return box(x - hw, y - hh, x + hw, y + hh)


def resolve_exclusion_zones(config, coordinate_system) -> list[ResolvedExclusionZone]:
    """Collect every exclusion zone in ``config`` as scene-coordinate geometry.

    Sources: ``config.exclusion_zones``, plus the ``exclusion_zone`` shorthand on
    ``user_assets`` and ``xml_scenes``. A zone that fails to convert is skipped
    with a warning.
    """
    result: list[ResolvedExclusionZone] = []

    def _add(source, make_geometry, excludes):
        try:
            geometry = make_geometry()
        except Exception as e:
            logging.warning("Failed to process exclusion zone '%s': %s", source, e)
            return
        result.append(ResolvedExclusionZone(source, geometry, frozenset(excludes)))
        logging.info("Processed exclusion zone '%s'", source)

    for zone in config.exclusion_zones:
        _add(
            f"zone_{zone.zone_id}",
            lambda z=zone: _zone_geometry(z, coordinate_system),
            zone.excludes,
        )

    for asset in config.user_assets:
        if asset.exclusion_zone is None:
            continue
        x, y = _to_scene(asset.coordinate, asset.coord_type, coordinate_system)
        _add(
            f"asset_{asset.object_id}",
            lambda z=asset.exclusion_zone, x=x, y=y: _object_zone_geometry(x, y, z),
            asset.exclusion_zone.excludes,
        )

    for xml_scene in config.xml_scenes:
        if xml_scene.exclusion_zone is None:
            continue
        x, y = _to_scene(
            xml_scene.base_coordinate, xml_scene.coord_type, coordinate_system
        )
        name = xml_scene.object_id_prefix or xml_scene.xml_path.upath.stem
        _add(
            f"xml_scene_{name}",
            lambda z=xml_scene.exclusion_zone, x=x, y=y: _object_zone_geometry(x, y, z),
            xml_scene.exclusion_zone.excludes,
        )

    return result


def exclusion_mask(
    geoms: Sequence[BaseGeometry] | np.ndarray,
    zones: Iterable[ResolvedExclusionZone],
    *,
    label: str = "Exclusion",
) -> np.ndarray:
    """Boolean keep-mask: False where a geometry intersects any zone.

    ``intersects`` is boundary-safe (a point on a zone edge, or a footprint
    straddling it, is excluded). Empty inputs keep everything.
    """
    from shapely.strtree import STRtree

    geoms = np.asarray(geoms, dtype=object)
    zones = list(zones)
    keep = np.ones(len(geoms), dtype=bool)
    if len(geoms) == 0 or not zones:
        return keep

    tree = STRtree([z.geometry for z in zones])
    hit_idx, _ = tree.query(geoms, predicate="intersects")
    keep[hit_idx] = False

    excluded = int((~keep).sum())
    logging.info(
        "%s exclusion: kept %d, excluded %d (%.1f%%) across %d zone(s)",
        label,
        len(geoms) - excluded,
        excluded,
        100.0 * excluded / len(geoms),
        len(zones),
    )
    return keep
