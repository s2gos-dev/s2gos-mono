"""Building footprint resource — clips GPKG building footprints to the AOI,
extrudes each one to a height and places it on the DEM, and emits a YAML sidecar
referenced by the scene description. Footprints are merged into one mesh per material
(plus one roof mesh, if hipped roof) rather than emitted as individual objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from s2gos_utils.io.paths import open_file

from ..assets.buildings import (
    _safe_name,
    build_meshes,
    load_building_footprints,
    make_dem_elevation_sampler,
)
from ..core.context import SceneResourceContext


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

    bbox = ctx.target_aoi_polygon.bounds

    gdf = load_building_footprints(cfg.file_paths, cfg.layer_name, bbox)
    if gdf.empty:
        logging.info("No buildings found in AOI — skipping")
        return None

    gdf = ctx.coordinate_system.geodataframe_to_scene_local(gdf)
    elev_fn = make_dem_elevation_sampler(target_dem_path)

    result = build_meshes(gdf, elev_fn, cfg)
    stats = result.stats
    if stats.total == 0:
        logging.info("No buildings extruded successfully — skipping")
        return None

    objects: list[dict] = []
    for material_name, mesh in result.material_meshes.items():
        if result.single_material:
            ply_path = ctx.meshes_dir / "buildings.ply"
            obj_id = cfg.object_id_prefix
        else:
            ply_path = ctx.meshes_dir / f"buildings_{_safe_name(material_name)}.ply"
            obj_id = f"{cfg.object_id_prefix}_{_safe_name(material_name)}"
        mesh.export(str(ply_path))
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

    if result.roof_mesh is not None:
        roof_safe = _safe_name(cfg.roof_material)
        roof_ply_path = ctx.meshes_dir / f"roofs_{roof_safe}.ply"
        result.roof_mesh.export(str(roof_ply_path))
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

    counts = ", ".join(f"{n}={c}" for n, c in stats.per_material_counts.items())
    logging.info(
        "Buildings meshes saved: %d buildings (%s), %d pitched, %d flat-fallback, %d skipped",
        stats.total,
        counts,
        stats.pitched,
        stats.flat_fallback,
        stats.skipped,
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
