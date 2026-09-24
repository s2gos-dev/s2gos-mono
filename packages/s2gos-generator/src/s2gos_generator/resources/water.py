"""Water body resource — fetches OSM water data, completes with landcover, saves a sidecar."""

import json
import logging
from pathlib import Path
from typing import Optional

import xarray as xr
from s2gos_utils.io.paths import expand_mapper

from ..core.context import SceneResourceContext
from ..processors.water import (
    WaterFetchError,
    complete_with_landcover,
    fetch_osm_data,
    parse_water_bodies,
    water_bodies_to_sidecar,
)


def process_target_water(ctx: SceneResourceContext) -> Optional[Path]:
    """Fetch/load OSM water data, optionally complete with landcover, and save the sidecar JSON."""
    water_cfg = ctx.config.water
    if water_cfg is None or not water_cfg.enabled:
        return None

    bbox_west, bbox_south, bbox_east, bbox_north = ctx.target_aoi_polygon.bounds

    try:
        osm_data = fetch_osm_data(
            water_cfg, bbox_south, bbox_west, bbox_north, bbox_east
        )
    except WaterFetchError as exc:
        logging.error("Water fetch failed, skipping water bodies: %s", exc)
        return None

    if osm_data is None:
        logging.warning(
            "No water data available from source — proceeding with landcover only"
        )
        osm_bodies = []
    else:
        osm_bodies = parse_water_bodies(
            osm_data, water_cfg, ctx.coordinate_system, ctx.target_scene_bounds
        )

    all_bodies = list(osm_bodies)

    if water_cfg.landcover_completion:
        landcover_path = ctx.dependency_outputs.get("target_landcover")
        dem_path = ctx.dependency_outputs.get("target_dem")
        if landcover_path is None or dem_path is None:
            logging.warning(
                "Landcover completion requested but landcover/DEM data unavailable — skipping completion"
            )
        else:
            lc_dataset = xr.open_zarr(expand_mapper(landcover_path))
            landcover_data = lc_dataset[list(lc_dataset.data_vars)[0]]
            dem_dataset = xr.open_zarr(expand_mapper(dem_path))
            dem_data = dem_dataset["elevation"]
            extra_bodies = complete_with_landcover(
                osm_bodies, landcover_data, dem_data, water_cfg
            )
            all_bodies.extend(extra_bodies)

    if not all_bodies:
        logging.info("No water bodies found in AOI — skipping")
        return None

    sidecar_path = ctx.data_dir / "water_bodies.json"
    with open(str(sidecar_path), "w") as f:
        json.dump(water_bodies_to_sidecar(all_bodies), f)
    ctx.assets.water_file = sidecar_path

    materials = sorted({b.material for b in all_bodies})
    logging.info(
        "Water bodies sidecar saved: %s (%d bodies: %d OSM + %d landcover-completed, %d materials: %s)",
        sidecar_path,
        len(all_bodies),
        len(osm_bodies),
        len(all_bodies) - len(osm_bodies),
        len(materials),
        ", ".join(materials),
    )
    return sidecar_path
