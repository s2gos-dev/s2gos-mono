#!/usr/bin/env python3
import logging
import math
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Annotated

from gavicore.models import Link
from procodile import FromMain, FromStep
from pydantic import BaseModel, Field
from s2gos_utils.io import PathRef
from upath import UPath

from s2gos_apps.registry import registry

logger = logging.getLogger(__name__)

OUTPUT_BUCKET = "s3://s2gos-output"
OUTPUT_CREDENTIAL_ID = "s3ovh"


class TargetInfo(BaseModel):
    scene_name: str
    lat: float
    lon: float
    size: float
    gmt_hour: float
    spp: int


def target_from_bbox(target_bbox: list[float]) -> tuple[float, float, float]:
    """Return the center and enclosing square size in km for a WGS84 bbox."""
    west, south, east, north = target_bbox
    if not -180.0 <= west < east <= 180.0:
        raise ValueError("Target area longitude bounds must be within -180..180.")
    if not -90.0 <= south < north <= 90.0:
        raise ValueError("Target area latitude bounds must be within -90..90.")

    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    earth_radius_km = 6371.0088

    def great_circle_distance_km(
        lat_a: float, lon_a: float, lat_b: float, lon_b: float
    ) -> float:
        lat_delta = math.radians(lat_b - lat_a)
        lon_delta = math.radians(lon_b - lon_a)
        haversine = (
            math.sin(lat_delta / 2.0) ** 2
            + math.cos(math.radians(lat_a))
            * math.cos(math.radians(lat_b))
            * math.sin(lon_delta / 2.0) ** 2
        )
        return 2.0 * earth_radius_km * math.asin(math.sqrt(haversine))

    width_km = great_circle_distance_km(center_lat, west, center_lat, east)
    height_km = great_circle_distance_km(south, center_lon, north, center_lon)
    return center_lat, center_lon, max(width_km, height_km)


def s3_output_path(relative_path: str) -> PathRef:
    """Resolve a user-facing, bucket-relative output directory to an S3 path."""
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts or "://" in relative_path:
        raise ValueError("Output directories must be non-empty relative paths.")
    return PathRef(f"{OUTPUT_BUCKET}/{path.as_posix()}", cid=OUTPUT_CREDENTIAL_ID)


def gmt_hour_from_observation_time(observation_time: datetime) -> float:
    """Convert an ISO date-time input to the GMT hour expected by sim_util."""
    if observation_time.tzinfo is not None:
        observation_time = observation_time.astimezone(timezone.utc)
    return (
        observation_time.hour
        + observation_time.minute / 60.0
        + observation_time.second / 3600.0
        + observation_time.microsecond / 3_600_000_000.0
    )


@registry.main(
    id="frascati_generation_simulation_workflow",
    title="Frascati generation and simulation",
    outputs={
        "target_info": Field(
            ...,
            description=(
                "Target scene parameters (scene name, centre lat/lon, size, "
                "observation time, sample count) as an inline JSON document."
            ),
        ),
        "config_path": Field(title="PathRef class with generated config as output"),
        "scene_output_dir": Field(),
        "config_output_dir_simulation": Field(),
        "output_dir_simulation": Field(),
    },
)
def frascati_generation_simulation_workflow(
    scene_name: Annotated[
        str,
        Field(
            default="frascati",
            description="Identifier used for the generated scene and configuration files.",
            json_schema_extra={"x-ui-order": 10},
        ),
    ],
    target_bbox: Annotated[
        list[float],
        Field(
            default=[12.6206, 41.7631, 12.7414, 41.8529],
            min_length=4,
            max_length=4,
            title="Target area",
            description=(
                "Draw the target area as [west, south, east, north] in WGS84."
            ),
            json_schema_extra={"x-ui-widget": "map", "x-ui-order": 20},
        ),
    ],
    observation_time: Annotated[
        datetime,
        Field(
            default=datetime(2024, 1, 1, 11, 0, 0),
            title="Observation date and time (UTC)",
            description="The time component is passed to the simulator as its GMT hour.",
            json_schema_extra={"x-ui-order": 30},
        ),
    ],
    spp: Annotated[
        int,
        Field(
            default=8,
            ge=1,
            description="Number of Monte Carlo samples per pixel.",
            json_schema_extra={"x-ui-order": 40, "x-ui-advanced": True},
        ),
    ],
    config_output_dir_generation: Annotated[
        str,
        Field(
            default="frascati/gen_config",
            description="Directory for generation configurations, relative to the processing S3 bucket.",
            min_length=1,
            json_schema_extra={"x-ui-order": 100, "x-ui-advanced": True},
        ),
    ],
    scene_output_dir_generation: Annotated[
        str,
        Field(
            default="frascati/gen_output",
            description="Directory for generated scene descriptions, relative to the processing S3 bucket.",
            min_length=1,
            json_schema_extra={"x-ui-order": 110, "x-ui-advanced": True},
        ),
    ],
    config_output_dir_simulation: Annotated[
        str,
        Field(
            default="frascati/sim_config",
            description="Directory for simulation configurations, relative to the processing S3 bucket.",
            min_length=1,
            json_schema_extra={"x-ui-order": 120, "x-ui-advanced": True},
        ),
    ],
    output_dir_simulation: Annotated[
        str,
        Field(
            default="frascati/sim_output",
            description="Directory for simulation outputs, relative to the processing S3 bucket.",
            min_length=1,
            json_schema_extra={"x-ui-order": 130, "x-ui-advanced": True},
        ),
    ],
) -> tuple[
    TargetInfo,
    PathRef | None,
    PathRef | None,
    PathRef | None,
    PathRef | None,
]:
    """
    Create the scene configuration corresponding the Frascati scene.
    """
    from s2gos_generator import create_scene_config
    from s2gos_generator.core.config import (
        AbsorptionDatabase,
        BackgroundConfig,
        BufferConfig,
        MolecularAtmosphereConfig,
        ThermophysicalConfig,
        VegetationPlacementConfig,
        VegetationSpecies,
    )

    # Enforce PathRef type
    target_lat, target_lon, target_size = target_from_bbox(target_bbox)
    gmt_hour = gmt_hour_from_observation_time(observation_time)
    config_output_dir = s3_output_path(config_output_dir_generation)
    scene_output_dir = s3_output_path(scene_output_dir_generation)
    simulation_config_output_dir = s3_output_path(config_output_dir_simulation)
    simulation_output_dir = s3_output_path(output_dir_simulation)

    logger.info("Configuring generation...")

    # Create basic configuration using defaults
    config = create_scene_config(
        scene_name=scene_name,
        center_lat=target_lat,
        center_lon=target_lon,
        aoi_size_km=target_size,
        output_dir=scene_output_dir,
        target_resolution_m=10.0,
        description="Frascati city and surroundings",
    )

    config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)
    config.background = BackgroundConfig(
        size_km=150.0, resolution_m=200.0, elevation=0.0
    )

    # Configure multi-species vegetation placement with trees and shrubs
    config.vegetation_placement = VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [  # Treecover
                VegetationSpecies(
                    name="trees",
                    asset_xml_paths=[
                        "tls_tree_25.xml",
                        "tls_tree_71.xml",
                        "tls_tree_165.xml",
                        "tls_tree_228.xml",
                        "tls_tree_290.xml",
                        "tls_tree_300.xml",
                        "tls_tree_336.xml",
                    ],  # Single asset in list
                    # For multiple variants with uniform distribution:
                    # asset_xml_paths=["tree1.xml", "tree2.xml", "tree3.xml"]
                    # For weighted distribution:
                    # asset_xml_paths={"tree_mature.xml": 5.0, "tree_young.xml": 2.0, "tree_old.xml": 1.0}
                    density_per_hectare=450.0,  # Moderate forest density
                    scale_min=0.8,
                    scale_max=1.4,
                )
            ],
            20: [  # Shrubland
                VegetationSpecies(
                    name="shrubs",
                    asset_xml_paths=["tls_tree_336.xml"],  # Single asset in list
                    density_per_hectare=40.0,
                    scale_min=0.4,
                    scale_max=0.8,
                )
            ],
        },
        density_variation=0.5,
        min_spacing=0.1,
        max_instances_per_pixel=2000,
        spillover_max_distance_m=50.0,
        spillover_compatibility={  # Optional: override global
            30: 0.9,  # High spillover into grassland
            20: 0.5,  # Moderate spillover into shrubland
            60: 0.5,
            100: 0.5,
        },
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
        has_scattering=True,
    )

    config.set_atmosphere_molecular(molecular_config)

    logger.info("Basic configuration created")
    logger.info("Configuration validation passed")

    # Save generation config file
    config_filename = f"{config.scene_name}_gen_config.json"

    if not config_output_dir.upath.exists():
        config_output_dir.upath.mkdir(parents=True, exist_ok=True)
    config_path = config_output_dir / config_filename

    config.to_json(config_path)

    return (
        TargetInfo(
            scene_name=scene_name,
            lat=target_lat,
            lon=target_lon,
            size=target_size,
            gmt_hour=gmt_hour,
            spp=spp,
        ),
        config_path,
        scene_output_dir,
        simulation_config_output_dir,
        simulation_output_dir,
    )


@frascati_generation_simulation_workflow.step(
    id="frascati-generation",
    title="Generate Scenes for Frascati",
    inputs={
        "config_path": FromMain(output="config_path"),
    },
    outputs={"scene_description_path": Field()},
)
def frascati_generation(config_path: PathRef) -> PathRef:
    from s2gos_apps.processes.common.generation import generation

    logger.debug("config_path raw: %r", config_path)
    input_ref = PathRef(config_path) if config_path is not None else None
    cid = input_ref.cid if input_ref is not None else None

    result = generation(config_path=config_path)

    # generation() is a Workflow; it wraps its return as {"return_value": upath}
    if isinstance(result, dict) and "return_value" in result:
        result = result["return_value"]

    logger.debug("scene_description_path=%r  cid=%r", result, cid)
    if result is not None:
        return PathRef(str(result), cid=cid)
    return None


@frascati_generation_simulation_workflow.step(
    id="frascati-simulation-config",
    title="Frascati Simulation Config",
    inputs={
        "target_info": FromMain(output="target_info"),
        "config_output_dir": FromMain(output="config_output_dir_simulation"),
    },
    outputs={
        "config_path": Field(
            title="PathRef class with simulated config as output",
        )
    },
)
def simulation_configs(
    target_info: TargetInfo,
    config_output_dir: PathRef | None,
) -> PathRef | None:
    from s2gos_apps.sim_util import simulation_config

    logger.debug("config_output_dir raw: %r", config_output_dir)
    input_ref = PathRef(config_output_dir) if config_output_dir is not None else None
    cid = input_ref.cid if input_ref is not None else None
    config_output_upath = input_ref.upath if input_ref is not None else None
    logger.debug("config_output_upath=%r  cid=%r", config_output_upath, cid)

    config_path = simulation_config(
        target_info.scene_name,
        target_info.lat,
        target_info.lon,
        target_info.size,
        target_info.gmt_hour,
        target_info.spp,
        config_output_upath,
    )
    logger.debug("config_path=%r", config_path)
    if config_path is not None:
        result = PathRef(str(config_path), cid=cid)
        logger.debug("returning PathRef(href=%r, cid=%r)", result.href, result.cid)
        return result
    return None


@frascati_generation_simulation_workflow.step(
    id="frascati-simulation",
    inputs={
        "scene_description_path": FromStep(
            step_id="frascati-generation", output="scene_description_path"
        ),
        "config_path": FromStep(
            step_id="frascati-simulation-config", output="config_path"
        ),
        "simulation_output_dir": FromMain(output="output_dir_simulation"),
    },
    outputs={"simulation_path": Field()},
)
def frascati_simulation(
    scene_description_path: PathRef,
    config_path: PathRef,
    simulation_output_dir: PathRef | None,
) -> Link:
    from s2gos_apps.processes.common.simulation import simulation

    logger.debug("scene_description_path raw: %r", scene_description_path)
    logger.debug("config_path raw: %r", config_path)
    logger.debug("simulation_output_dir raw: %r", simulation_output_dir)

    def _to_upath(v, key=None):
        # Procodile may pass step output as the full dict {"output_key": value}
        # or as a PathRef-shaped dict {"href": "...", "x-cid": ...}, or as a
        # PathRef instance, UPath, or plain string.
        if isinstance(v, dict):
            if "href" in v:
                return PathRef(v).upath
            if "value" in v:  # Compatibility with older serialized PathRefs.
                return PathRef(v["value"], cid=v.get("cid")).upath
            if key and key in v:
                return _to_upath(v[key])
        if hasattr(v, "upath"):
            return v.upath
        return UPath(str(v)) if v is not None else None

    sdp = _to_upath(scene_description_path, "scene_description_path")
    cp = _to_upath(config_path, "config_path")
    sod = _to_upath(simulation_output_dir, "simulation_output_dir")
    logger.debug("resolved: scene_description_path=%r", sdp)
    logger.debug("resolved: config_path=%r", cp)
    logger.debug("resolved: simulation_output_dir=%r", sod)

    result = simulation(
        scene_description_path=sdp,
        config_path=cp,
        simulation_output_dir=sod,
    )
    result_path = PathRef(result)
    return Link(
        href=result_path.href,
        rel="output",
        title="Frascati simulation output",
    )
