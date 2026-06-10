"""Building footprint resource — clips GPKG building footprints to the AOI,
extrudes each one to a height places it on the DEM, and emits a YAML sidecar
referenced by the scene description. Footprints are merged into one mesh per material
(plus one roof mesh, if hipped roof) rather than emitted as individual objects.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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

from ..assets.building_roof import build_hip_roof, compute_pitched_geometry
from ..assets.terrain_mesh import _make_elevation_fn, extract_dem
from ..core.context import SceneResourceContext


def _parse_height(value, story_height: float, default: float) -> tuple[float, bool]:
    """Resolve a building height in meters from a GPKG height-column value.

    Accepts either a numeric height (used directly as meters) or  taxonomy string.
    For more info see the Open Building Map dataset's paper.

    Returns ``(height_m, parsed)`` where ``parsed`` is ``False`` when the value
    carried no usable height and ``default`` was substituted, so the caller can
    count and report silent fallbacks.
    """
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        if np.isfinite(value) and value > 0:
            return float(value), True
        return default, False

    if not isinstance(value, str) or not value or value == "nan":
        return default, False

    parts = value.split("+")

    for part in parts:
        if part.startswith("HHT:"):
            try:
                return float(part.split(":")[1]), True
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("H:"):
            try:
                return float(part.split(":")[1]) * story_height, True
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("HAPP:"):
            try:
                return float(part.split(":")[1]) * story_height, True
            except (ValueError, IndexError):
                pass
    for part in parts:
        if part.startswith("HBET:"):
            try:
                s_min, s_max = part.split(":")[1].split("-")
                return ((float(s_min) + float(s_max)) / 2.0) * story_height, True
            except (ValueError, IndexError):
                pass

    return default, False


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


@dataclass
class _BuildingTask:
    """One building's worth of pre-computed inputs, marshalled to a worker.

    All RNG draws (material assignment, pitched-vs-flat selection) are made in
    the main thread before constructing the task, so per-building results stay
    deterministic regardless of worker count or scheduling order.
    """

    idx: int
    geom: Union[Polygon, MultiPolygon]
    height_m: float
    base_z: float
    skirt_m: float
    material_name: str
    pitched: bool
    pitch_deg: float
    target_roof_height: float


@dataclass
class _BuildingResult:
    idx: int
    material_name: str
    wall_mesh: Optional[trimesh.Trimesh]
    roof_mesh: Optional[trimesh.Trimesh]
    pitched_attempted: bool
    pitched_succeeded: bool


def _process_one_building(task: _BuildingTask) -> _BuildingResult:
    """Worker-side: build the wall mesh and (optionally) the hip-roof mesh.
    Pure function — no shared state — safe to call from a subprocess."""
    geom = task.geom
    wall_mesh: Optional[trimesh.Trimesh] = None
    roof_mesh: Optional[trimesh.Trimesh] = None
    pitched_attempted = task.pitched
    pitched_succeeded = False

    roof_info = None
    if task.pitched:
        try:
            roof_info = compute_pitched_geometry(
                total_height=task.height_m,
                pitch_deg=task.pitch_deg,
                target_roof_height=task.target_roof_height,
            )
        except Exception as exc:
            logging.debug("Building %d: pitched geometry failed: %s", task.idx, exc)
            roof_info = None

    if roof_info is None:
        wall_mesh = _build_one_building(geom, task.height_m, task.skirt_m, task.base_z)
    else:
        eaves_z = task.base_z + roof_info["eaves_z_offset"]
        apex_z = task.base_z + roof_info["apex_z_offset"]
        wall_mesh = _build_one_building(
            geom, roof_info["eaves_z_offset"], task.skirt_m, task.base_z
        )
        try:
            roof_mesh = build_hip_roof(geom, eaves_z, apex_z, roof_info["pitch_deg"])
        except Exception as exc:
            logging.debug(
                "Building %d: hip-roof build failed, using flat: %s", task.idx, exc
            )
            roof_mesh = None
        if roof_mesh is None:
            wall_mesh = _build_one_building(
                geom, task.height_m, task.skirt_m, task.base_z
            )
        else:
            pitched_succeeded = True

    return _BuildingResult(
        idx=task.idx,
        material_name=task.material_name,
        wall_mesh=wall_mesh,
        roof_mesh=roof_mesh,
        pitched_attempted=pitched_attempted,
        pitched_succeeded=pitched_succeeded,
    )


def _process_chunk(tasks: list[_BuildingTask]) -> list[_BuildingResult]:
    """Worker entrypoint: process a batch of buildings to amortize IPC overhead."""
    return [_process_one_building(t) for t in tasks]


def process_target_buildings(ctx: SceneResourceContext) -> Optional[Path]:
    """Clip GPKG footprints to the AOI, extrude each, and emit a YAML sidecar
    placed on top of the DEM. Footprints are grouped into one mesh per material
    (plus one roof mesh). When the building material is given as a {name: weight}
    mapping, each building is assigned one material by weighted random draw before
    grouping."""
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

    gdf = ctx.coordinate_system.geodataframe_to_scene_local(gdf)
    elev_fn = _make_dem_elev_fn(target_dem_path)

    names, weights = _resolve_material_distribution(cfg.material)
    rng = np.random.default_rng(cfg.material_seed)
    roof_rng = np.random.default_rng(cfg.roof_seed)

    meshes_by_material: dict[str, list[trimesh.Trimesh]] = {n: [] for n in names}
    roof_meshes: list[trimesh.Trimesh] = []
    skipped = 0
    pitched_count = 0
    flat_fallback = 0
    unparsed_height = 0

    has_height_col = cfg.height_column in gdf.columns
    if not has_height_col:
        logging.warning(
            "Buildings height_column %r not found in GPKG (columns: %s); "
            "every building will use default_height_m=%.1f m",
            cfg.height_column,
            list(gdf.columns),
            cfg.default_height_m,
        )

    # Pass 1 (sequential, deterministic): parse heights, sample DEM elevation,
    # draw all RNG outcomes (material + pitched/flat), and build a list of
    # self-contained per-building tasks. Doing all RNG draws here keeps results
    # reproducible regardless of how the worker pool schedules things.
    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    skipped += int((~valid_mask).sum())
    gdf_valid = gdf[valid_mask]
    n = len(gdf_valid)

    if n:
        centroids = gdf_valid.geometry.centroid
        xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
        base_zs = elev_fn(xy) + cfg.elevation_offset_m
    else:
        base_zs = np.empty(0)

    us = roof_rng.random(n)
    if weights is None:
        material_names = [names[0]] * n
    else:
        material_names = [str(m) for m in rng.choice(names, size=n, p=weights)]

    tasks: list[_BuildingTask] = []
    for (idx, row), base_z, u, material_name in zip(
        gdf_valid.iterrows(), base_zs, us, material_names
    ):
        geom = row.geometry
        raw_height = row.get(cfg.height_column) if has_height_col else None
        height_m, parsed = _parse_height(
            raw_height, cfg.story_height_m, cfg.default_height_m
        )
        if not parsed:
            unparsed_height += 1

        eligible = (
            cfg.pitched_roof_proportion > 0.0
            and (
                cfg.pitched_roof_min_area_m2 is None
                or geom.area >= cfg.pitched_roof_min_area_m2
            )
            and (
                cfg.pitched_roof_min_height_m is None
                or height_m >= cfg.pitched_roof_min_height_m
            )
        )
        pitched = eligible and u < cfg.pitched_roof_proportion
        tasks.append(
            _BuildingTask(
                idx=int(idx),
                geom=geom,
                height_m=height_m,
                base_z=float(base_z),
                skirt_m=cfg.base_skirt_m,
                material_name=material_name,
                pitched=pitched,
                pitch_deg=cfg.roof_pitch_deg,
                target_roof_height=cfg.roof_height_m,
            )
        )

    if has_height_col and unparsed_height:
        logging.info(
            "Buildings height: %d/%d footprints had no usable value in %r and "
            "fell back to default_height_m=%.1f m",
            unparsed_height,
            len(tasks),
            cfg.height_column,
            cfg.default_height_m,
        )

    # Pass 2: parallel mesh construction.
    workers = (
        cfg.roof_workers
        if cfg.roof_workers is not None
        else max((os.cpu_count() or 2) // 2, 1)
    )
    if workers > 1 and len(tasks) >= 64:
        chunk_size = max(8, (len(tasks) + workers - 1) // workers)
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        results: list[_BuildingResult] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk_results in ex.map(_process_chunk, chunks):
                results.extend(chunk_results)
        results.sort(key=lambda r: r.idx)
    else:
        results = [_process_one_building(t) for t in tasks]

    # Pass 3 (sequential): assemble results into the per-material mesh lists.
    for r in results:
        if r.wall_mesh is None:
            skipped += 1
            continue
        if r.pitched_attempted and not r.pitched_succeeded:
            flat_fallback += 1
        if r.pitched_succeeded:
            pitched_count += 1
        meshes_by_material[r.material_name].append(r.wall_mesh)
        if r.roof_mesh is not None:
            roof_meshes.append(r.roof_mesh)

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

    if roof_meshes:
        combined_roof = (
            roof_meshes[0]
            if len(roof_meshes) == 1
            else trimesh.util.concatenate(roof_meshes)
        )
        roof_safe = _safe_name(cfg.roof_material)
        roof_ply_path = ctx.meshes_dir / f"roofs_{roof_safe}.ply"
        combined_roof.export(str(roof_ply_path))
        objects.append(
            {
                "id": f"{cfg.object_id_prefix}_roof_{roof_safe}",
                "mesh": str(roof_ply_path.relative_to(ctx.output_dir)),
                "position": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "rotation": [0.0, 0.0, 0.0],
                "material": cfg.roof_material,
                "face_normals": True,
            }
        )

    counts = ", ".join(
        f"{n}={len(meshes_by_material[n])}" for n in names if meshes_by_material[n]
    )
    logging.info(
        "Buildings meshes saved: %d buildings (%s), %d pitched, %d flat-fallback, %d skipped",
        total,
        counts,
        pitched_count,
        flat_fallback,
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
