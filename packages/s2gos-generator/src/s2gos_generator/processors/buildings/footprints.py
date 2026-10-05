"""Building-footprint sidecar (de)serialization.

The buildings resource extrudes footprints into meshes but the footprint polygons
themselves are also needed downstream (vegetation placement excludes trees that
land on a building). We persist the scene-local footprints to a small JSON
sidecar so a ``target_buildings`` cache hit can restore them without re-reading
the GPKG tiles.
"""

from __future__ import annotations

import logging

from shapely.geometry import mapping, shape

SIDECAR_VERSION = 1


def footprints_to_sidecar(gdf) -> dict:
    """Serialize a scene-local footprint GeoDataFrame to the sidecar structure.

    Empty and missing geometries are dropped.
    """
    geometries = [
        mapping(geom) for geom in gdf.geometry if geom is not None and not geom.is_empty
    ]
    return {"version": SIDECAR_VERSION, "footprints": geometries}


def footprints_from_sidecar(data: dict) -> list:
    """Reconstruct footprint geometries from the sidecar structure.

    Returns an empty list for an unrecognised schema version.
    """
    version = data.get("version", SIDECAR_VERSION)
    if version != SIDECAR_VERSION:
        logging.warning(
            "Unknown building-footprint sidecar version %s; skipping footprints",
            version,
        )
        return []
    return [shape(g) for g in data.get("footprints", [])]
