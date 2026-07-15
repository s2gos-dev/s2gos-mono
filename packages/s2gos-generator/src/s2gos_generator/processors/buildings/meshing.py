"""Building mesh construction primitives.

Mesh computation for building footprints: height-taxonomy
parsing, per-building extrusion, hip-roof creation, weighted material assignment,
and the parallel mesh-build pipeline.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
import trimesh
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from .roof import build_hip_roof, compute_pitched_geometry
from ..terrain_mesh.builder import _make_elevation_fn, extract_dem


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


def quadkeys_for_bbox(bbox: tuple[float, float, float, float], zoom: int) -> set[str]:
    """Quadkeys of every tile at ``zoom`` overlapping a WGS84 lon/lat bbox."""
    return {mercantile.quadkey(t) for t in mercantile.tiles(*bbox, zoom)}


def select_tile_files(
    tile_dir: Path,
    bbox: tuple[float, float, float, float],
    index_csv: str,
) -> list[Path]:
    """Pick the building tiles overlapping ``bbox`` from a quadkey-indexed dir.

    Reads the index CSV (quadkey -> filename), figures out which tiles the AOI
    touches via :func:`quadkeys_for_bbox`, and returns the matching files that are
    actually present in ``tile_dir``. Index-listed tiles that are missing
    are warned about and skipped.
    """
    index = pd.read_csv(tile_dir / index_csv, dtype={"quadkey": str})

    zoom = len(index["quadkey"].iloc[0])

    want = quadkeys_for_bbox(bbox, zoom)
    selected = index[index["quadkey"].isin(want)]

    present, missing = [], []
    for filename in selected["filename"]:
        path = tile_dir / filename
        (present if path.exists() else missing).append(path)

    if missing:
        logging.warning(
            "%d building tile(s) overlap the AOI but are not present in %s: %s",
            len(missing),
            tile_dir,
            ", ".join(p.name for p in missing),
        )
    logging.info(
        "buildings: %d tiles overlap AOI, %d present locally",
        len(selected),
        len(present),
    )
    return sorted(present)


def load_building_footprints(
    file_paths: list[Path], layer: str, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Read each GPKG with a bbox prefilter and concat into a single frame."""
    frames = []
    for p in file_paths:
        gdf = gpd.read_file(str(p), layer=layer, bbox=bbox)
        if not gdf.empty:
            frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(geometry=[])
    if not frames[0].crs.is_geographic:
        raise ValueError(
            f"Building tiles must be in a geographic CRS for the WGS84 bbox "
            f"prefilter to be valid; got {frames[0].crs}."
        )
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def make_dem_elevation_sampler(dem_path) -> Callable[[np.ndarray], np.ndarray]:
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


def _concat_vf(
    parts: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (vertices, faces) pairs, offsetting face indices."""
    if len(parts) == 1:
        return parts[0]
    vert_counts = [len(v) for v, _ in parts]
    offsets = np.cumsum([0, *vert_counts[:-1]])
    vertices = np.vstack([v for v, _ in parts])
    faces = np.vstack([f.astype(np.int64) + off for (_, f), off in zip(parts, offsets)])
    return vertices, faces


@dataclass
class _ChunkResult:
    """Aggregated mesh arrays for one chunk of buildings."""

    buildings: dict[str, tuple[np.ndarray, np.ndarray]]
    building_counts: dict[str, int]
    roof: Optional[tuple[np.ndarray, np.ndarray]]
    pitched: int
    flat_fallback: int
    skipped: int


def _process_chunk(tasks: list[_BuildingTask]) -> _ChunkResult:
    """Worker entrypoint: build a batch of buildings and aggregate the results
    into per-material (vertices, faces) arrays."""

    building_parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    roof_parts: list[tuple[np.ndarray, np.ndarray]] = []
    pitched = 0
    flat_fallback = 0
    skipped = 0

    for task in tasks:
        r = _process_one_building(task)
        if r.wall_mesh is None:
            skipped += 1
            continue
        if r.pitched_attempted and not r.pitched_succeeded:
            flat_fallback += 1
        if r.pitched_succeeded:
            pitched += 1
        building_parts.setdefault(r.material_name, []).append(
            (r.wall_mesh.vertices, r.wall_mesh.faces)
        )
        if r.roof_mesh is not None:
            roof_parts.append((r.roof_mesh.vertices, r.roof_mesh.faces))

    return _ChunkResult(
        buildings={mat: _concat_vf(parts) for mat, parts in building_parts.items()},
        building_counts={mat: len(parts) for mat, parts in building_parts.items()},
        roof=_concat_vf(roof_parts) if roof_parts else None,
        pitched=pitched,
        flat_fallback=flat_fallback,
        skipped=skipped,
    )


@dataclass
class BuildingMeshStats:
    """Per-run counts gathered while building footprint meshes."""

    total: int
    per_material_counts: dict[str, int]
    pitched: int
    flat_fallback: int
    skipped: int
    unparsed_height: int


@dataclass
class BuildingMeshes:
    """Combined building geometry, grouped one mesh per material plus one roof mesh."""

    material_meshes: dict[str, trimesh.Trimesh]
    roof_mesh: Optional[trimesh.Trimesh]
    single_material: bool
    stats: BuildingMeshStats


def _build_tasks(
    gdf_valid: gpd.GeoDataFrame,
    base_zs: np.ndarray,
    cfg: "BuildingsConfig",
    has_height_col: bool,
) -> tuple[list[_BuildingTask], list[str], Optional[np.ndarray], int]:
    """Parse heights, draw all RNG outcomes (material + pitched/flat), and
    build a list of self-contained per-building tasks. Doing all RNG draws
    here keeps results reproducible regardless of how the worker pool later
    schedules things."""

    names, weights = _resolve_material_distribution(cfg.material)
    rng = np.random.default_rng(cfg.material_seed)
    roof_rng = np.random.default_rng(cfg.roof_seed)

    n = len(gdf_valid)
    us = roof_rng.random(n)
    if weights is None:
        material_names = [names[0]] * n
    else:
        material_names = [str(m) for m in rng.choice(names, size=n, p=weights)]

    indices = gdf_valid.index.to_numpy()
    geoms = gdf_valid.geometry.to_numpy()
    if has_height_col:
        raw_heights = gdf_valid[cfg.height_column].to_numpy()
    else:
        raw_heights = itertools.repeat(None)
    if cfg.pitched_roof_proportion > 0.0 and cfg.pitched_roof_min_area_m2 is not None:
        areas = gdf_valid.geometry.area.to_numpy()
    else:
        areas = itertools.repeat(0.0)

    tasks: list[_BuildingTask] = []
    unparsed_height = 0
    for idx, geom, raw_height, area, base_z, u, material_name in zip(
        indices, geoms, raw_heights, areas, base_zs, us, material_names
    ):
        height_m, parsed = _parse_height(
            raw_height, cfg.story_height_m, cfg.default_height_m
        )
        if not parsed:
            unparsed_height += 1

        eligible = (
            cfg.pitched_roof_proportion > 0.0
            and (
                cfg.pitched_roof_min_area_m2 is None
                or area >= cfg.pitched_roof_min_area_m2
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

    return tasks, names, weights, unparsed_height


_MAX_CHUNK_SIZE = 16384


def _run_tasks(
    tasks: list[_BuildingTask], cfg: "BuildingsConfig"
) -> list[_ChunkResult]:
    """Build per-building meshes, in parallel for large batches."""
    workers = (
        cfg.roof_workers
        if cfg.roof_workers is not None
        else max((os.cpu_count() or 2) // 2, 1)
    )
    if workers > 1 and len(tasks) >= 64:
        chunk_size = max(8, (len(tasks) + workers - 1) // workers)
        chunk_size = min(chunk_size, _MAX_CHUNK_SIZE)
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_process_chunk, chunks))
    return [_process_chunk(tasks)]


def build_meshes(
    gdf: gpd.GeoDataFrame,
    elev_fn: Callable[[np.ndarray], np.ndarray],
    cfg: "BuildingsConfig",
) -> BuildingMeshes:
    """Build combined building meshes from scene-local footprints.

    ``gdf`` must already be reprojected to scene-local meters. Each footprint is
    placed on the DEM (its centroid sampled via ``elev_fn``), extruded to a parsed
    height, optionally given a hip roof, and assigned a material. Footprints are
    grouped into one combined mesh per material (plus one combined roof mesh). When
    ``cfg.material`` is a ``{name: weight}`` mapping, each building is assigned one
    material by weighted random draw before grouping.
    """
    names, _ = _resolve_material_distribution(cfg.material)
    single_material = isinstance(cfg.material, str)

    has_height_col = cfg.height_column in gdf.columns
    if not has_height_col:
        logging.warning(
            "Buildings height_column %r not found in GPKG (columns: %s); "
            "every building will use default_height_m=%.1f m",
            cfg.height_column,
            list(gdf.columns),
            cfg.default_height_m,
        )

    valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    skipped = int((~valid_mask).sum())
    gdf_valid = gdf[valid_mask]
    n = len(gdf_valid)

    if n:
        centroids = gdf_valid.geometry.centroid
        xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
        base_zs = elev_fn(xy) + cfg.elevation_offset_m
    else:
        base_zs = np.empty(0)

    tasks, names, _weights, unparsed_height = _build_tasks(
        gdf_valid, base_zs, cfg, has_height_col
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

    chunk_results = _run_tasks(tasks, cfg)

    pitched_count = sum(c.pitched for c in chunk_results)
    flat_fallback = sum(c.flat_fallback for c in chunk_results)
    skipped += sum(c.skipped for c in chunk_results)

    material_meshes: dict[str, trimesh.Trimesh] = {}
    per_material_counts: dict[str, int] = {}
    for material_name in names:
        parts = [
            c.buildings[material_name]
            for c in chunk_results
            if material_name in c.buildings
        ]
        if not parts:
            continue
        per_material_counts[material_name] = sum(
            c.building_counts[material_name]
            for c in chunk_results
            if material_name in c.building_counts
        )
        vertices, faces = _concat_vf(parts)
        material_meshes[material_name] = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        )

    roof_mesh = None
    roof_parts = [c.roof for c in chunk_results if c.roof is not None]
    if roof_parts:
        vertices, faces = _concat_vf(roof_parts)
        roof_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    total = sum(per_material_counts.values())
    stats = BuildingMeshStats(
        total=total,
        per_material_counts=per_material_counts,
        pitched=pitched_count,
        flat_fallback=flat_fallback,
        skipped=skipped,
        unparsed_height=unparsed_height,
    )
    return BuildingMeshes(
        material_meshes=material_meshes,
        roof_mesh=roof_mesh,
        single_material=single_material,
        stats=stats,
    )
