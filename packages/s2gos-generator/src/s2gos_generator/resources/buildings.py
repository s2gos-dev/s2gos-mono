"""Building footprint resource — clips GPKG building footprints to the AOI,
extrudes each one to a height parsed from a taxonomy column, places it on the
DEM, and emits a YAML sidecar referenced by the scene description.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import trimesh
import xarray as xr
import yaml
from s2gos_utils.io.paths import open_file
from shapely.geometry import MultiPolygon, Polygon

from ..assets.terrain_mesh import _make_elevation_fn, extract_dem
from ..core.context import SceneResourceContext


def _parse_taxonomy_height(taxonomy_str, story_height: float, default: float) -> float:
    """Parse a string like 'HHT:35+H:10' into meters."""
    if not isinstance(taxonomy_str, str) or not taxonomy_str or taxonomy_str == "nan":
        return default

    parts = taxonomy_str.split("+")

    for part in parts:
        if part.startswith("HHT:"):
            try:
                return float(part.split(":")[1])
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("H:"):
            try:
                return float(part.split(":")[1]) * story_height
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("HAPP:"):
            try:
                return float(part.split(":")[1]) * story_height
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("HBET:"):
            try:
                s_min, s_max = part.split(":")[1].split("-")
                return ((float(s_min) + float(s_max)) / 2.0) * story_height
            except (ValueError, IndexError):
                pass

    return default


def _load_and_clip(
    file_paths: list[Path], layer: str, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Read each GPKG with a bbox prefilter and concat into a single frame."""
    frames = []
    for p in file_paths:
        gdf = gpd.read_file(str(p), layer=layer, engine="pyogrio", bbox=bbox)
        if not gdf.empty:
            frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(geometry=[])
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def _make_dem_elev_fn(dem_path):
    """Open the DEM zarr and return a fast bilinear elevation sampler."""
    dem = xr.open_zarr(dem_path)["elevation"]
    x, y, elev = extract_dem(dem)
    return _make_elevation_fn(x, y, elev)


def _build_one_building(
    geometry, height: float, skirt: float, base_z: float
) -> Optional[trimesh.Trimesh]:
    """Extrude a footprint into a mesh with world-space coordinates."""
    polys = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]

    meshes: list[trimesh.Trimesh] = []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        try:
            m = trimesh.creation.extrude_polygon(poly, height=height + skirt)
            meshes.append(m)
        except Exception as exc:
            logging.debug("Skipping un-extrudable polygon: %s", exc)

    if not meshes:
        return None
    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)

    mesh.apply_translation([0.0, 0.0, base_z - skirt])
    return mesh


def _resolve_material_distribution(
    material: Union[str, dict[str, float]],
) -> tuple[list[str], Optional[np.ndarray]]:
    """Return (names, normalized_weights). weights is None when there is a
    single material."""
    if isinstance(material, str):
        return [material], None
    names = list(material.keys())
    weights = np.array([material[n] for n in names], dtype=float)
    weights = weights / weights.sum()
    return names, weights


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name)


def process_target_buildings(ctx: SceneResourceContext) -> Optional[Path]:
    """Clip GPKG footprints to the AOI, extrude each, and emit a YAML sidecar
    of per-building scene objects placed on top of the DEM. When the building
    material is given as a {name: weight} mapping, each building is assigned
    one material by weighted random draw and buildings are grouped into one
    mesh per material."""
    cfg = ctx.config.buildings
    if cfg is None or not cfg.enabled or not cfg.file_paths:
        return None

    target_dem_path = ctx.dependency_outputs.get("target_dem")
    if target_dem_path is None:
        raise RuntimeError("Target DEM not available for building elevation queries")

    bbox_west, bbox_south, bbox_east, bbox_north = ctx.target_aoi_polygon.bounds
    bbox = (bbox_west, bbox_south, bbox_east, bbox_north)

    gdf = _load_and_clip(cfg.file_paths, cfg.layer_name, bbox)
    if gdf.empty:
        logging.info("No buildings found in AOI — skipping")
        return None

    cs = ctx.coordinate_system
    gdf = gdf.to_crs(cs.scene_crs)
    # to_crs gives raw omerc coords; shift to scene-local (same as latlon_to_scene)
    gdf.geometry = gdf.geometry.translate(-cs._center_x, -cs._center_y)
    elev_fn = _make_dem_elev_fn(target_dem_path)

    names, weights = _resolve_material_distribution(cfg.material)
    rng = np.random.default_rng(cfg.material_seed)

    meshes_by_material: dict[str, list[trimesh.Trimesh]] = {n: [] for n in names}
    skipped = 0
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        taxonomy = (
            row.get(cfg.height_column) if cfg.height_column in gdf.columns else None
        )
        height_m = _parse_taxonomy_height(
            taxonomy, cfg.story_height_m, cfg.default_height_m
        )

        cx, cy = float(geom.centroid.x), float(geom.centroid.y)
        base_z = float(elev_fn(np.array([[cx, cy]]))[0]) + cfg.elevation_offset_m

        mesh = _build_one_building(geom, height_m, cfg.base_skirt_m, base_z)
        if mesh is None:
            skipped += 1
            continue

        chosen = names[0] if weights is None else str(rng.choice(names, p=weights))
        meshes_by_material[chosen].append(mesh)

    total = sum(len(m) for m in meshes_by_material.values())
    if total == 0:
        logging.info("No buildings extruded successfully — skipping")
        return None

    single_material = weights is None
    objects: list[dict] = []
    for material_name, meshes in meshes_by_material.items():
        if not meshes:
            continue
        combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        if single_material:
            ply_path = ctx.meshes_dir / "buildings.ply"
            obj_id = cfg.object_id_prefix
        else:
            ply_path = ctx.meshes_dir / f"buildings_{_safe_name(material_name)}.ply"
            obj_id = f"{cfg.object_id_prefix}_{_safe_name(material_name)}"
        combined.export(str(ply_path))
        objects.append(
            {
                "id": obj_id,
                "mesh": str(ply_path.relative_to(ctx.output_dir)),
                "position": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "rotation": [0.0, 0.0, 0.0],
                "material": material_name,
                "face_normals": True,
            }
        )

    counts = ", ".join(
        f"{n}={len(meshes_by_material[n])}" for n in names if meshes_by_material[n]
    )
    logging.info(
        "Buildings meshes saved: %d buildings (%s), %d skipped",
        total,
        counts,
        skipped,
    )

    sidecar_path = ctx.data_dir / "buildings.yml"
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    with open_file(sidecar_path, "w") as f:
        yaml.safe_dump(
            {"objects": objects},
            f,
            default_flow_style=False,
            indent=2,
        )

    ctx.assets.buildings_objects_file = sidecar_path
    return sidecar_path
