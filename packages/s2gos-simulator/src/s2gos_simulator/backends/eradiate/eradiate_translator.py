"""Sensor and measurement translation for Eradiate backend.

This module handles translation of generic sensor configurations to Eradiate-specific
measure definitions.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .geometry_utils import apply_asset_relative_transform, sanitize_sensor_id
from ...config import (
    AngularFromOriginViewing,
    BHRConfig,
    BRFConfig,
    ConstantIllumination,
    DirectionalIllumination,
    DistantViewing,
    GroundInstrumentType,
    GroundSensor,
    HCRFConfig,
    HDRFConfig,
    HemisphericalViewing,
    IrradianceConfig,
    LookAtViewing,
    RectangleTarget,
    SatelliteSensor,
    UAVInstrumentType,
    UAVSensor,
)
from ...config.sensors import PostProcessingOptions

try:
    from eradiate.units import unit_registry as ureg

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class _AssetTransform(NamedTuple):
    """Internal: Transform information for a positioned asset.

    Used to resolve sensor positions specified relative to assets.
    Rotation contains USER-SPECIFIED rotations only, NOT including
    the internal Blender coordinate fix (90° X rotation).
    """

    object_id: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: float


class EradiateTranslator:
    """
    Translator from simulation config to Eraidate format
    for illumination, sensors, measurements, and derived quantities.
    """

    def __init__(self, simulation_config, geometry_utils):
        """Initialize sensor translator.

        Args:
            simulation_config: SimulationConfig object
            geometry_utils: GeometryUtils instance for coordinate operations
        """
        self.simulation_config = simulation_config
        self.geometry_utils = geometry_utils
        self._current_scene_description = None
        self._current_scene_dir = None

    def translate_illumination(self) -> Dict[str, Any]:
        """Translate generic illumination to Eradiate format.

        Returns:
            Illumination configuration dictionary for Eradiate
        """
        illumination = self.simulation_config.illumination

        if isinstance(illumination, DirectionalIllumination):
            return {
                "type": "directional",
                "id": illumination.id,
                "zenith": illumination.zenith * ureg.deg,
                "azimuth": illumination.azimuth * ureg.deg,
                "irradiance": {
                    "type": "solar_irradiance",
                    "dataset": illumination.irradiance_dataset,
                },
            }
        elif isinstance(illumination, ConstantIllumination):
            return {
                "type": "constant",
                "id": illumination.id,
                "radiance": illumination.radiance,
            }
        else:
            raise ValueError(f"Unsupported illumination type: {type(illumination)}")

    def _extract_asset_transforms(
        self, scene_description: SceneDescription
    ) -> Dict[str, _AssetTransform]:
        """Extract asset transforms from scene description (backend-internal).

        Reads blender_fix metadata to reverse coordinate fix and get user rotations.

        Args:
            scene_description: SceneDescription with positioned assets

        Returns:
            Dict mapping object_id to _AssetTransform
        """
        asset_transforms = {}

        for obj in scene_description.objects:
            if not isinstance(obj, dict):
                continue
            if "position" not in obj or "rotation" not in obj or "scale" not in obj:
                continue
            if obj.get("type") in ["vegetation_collection", "shapegroup"]:
                continue

            final_rotation = obj["rotation"]
            blender_fix = obj.get("blender_fix", False)

            if blender_fix:
                user_rotation_x = final_rotation[0] - 90.0
                user_rotation_z = final_rotation[1]
                user_rotation_y = -final_rotation[2]
                user_rotation = (user_rotation_x, user_rotation_y, user_rotation_z)
            else:
                user_rotation = tuple(final_rotation)

            asset_transforms[obj["id"]] = _AssetTransform(
                object_id=obj["id"],
                position=tuple(obj["position"]),
                rotation=user_rotation,
                scale=obj.get("scale", 1.0),
            )

        return asset_transforms

    def _resolve_asset_reference(
        self, reference: str, asset_transforms: Dict[str, _AssetTransform]
    ) -> str:
        """Resolve user-friendly asset reference to actual object ID.

        Handles extensions and prefix matching for intuitive asset referencing.

        Args:
            reference: User reference like "tower_v0_1.xml", "my_asset.ply", or "exact_id"
            asset_transforms: Dict of available asset transforms

        Returns:
            Resolved object_id

        Raises:
            ValueError: If reference cannot be resolved
        """
        # 1. Try exact match first (for exact IDs without extensions)
        if reference in asset_transforms:
            return reference

        # 2. Strip extension if present
        if reference.endswith(".xml") or reference.endswith(".ply"):
            reference_stem = reference.rsplit(".", 1)[0]
        else:
            reference_stem = reference

        # 3. Try exact match with stem
        if reference_stem in asset_transforms:
            return reference_stem

        # 4. If original had .xml extension, find first asset with prefix
        if reference.endswith(".xml"):
            prefix = reference_stem + "_"
            matches = [oid for oid in asset_transforms if oid.startswith(prefix)]
            if matches:
                matches.sort()
                logger.info(
                    f"Resolved '{reference}' to '{matches[0]}' (first of {len(matches)} matches)"
                )
                return matches[0]

        # 5. Not found - provide error
        available = sorted(asset_transforms.keys())
        if reference.endswith(".xml"):
            prefix = reference_stem + "_"
            xml_assets = [a for a in available if a.startswith(prefix)]
            if xml_assets:
                raise ValueError(
                    f"Asset reference '{reference}' matches {len(xml_assets)} assets with prefix '{prefix}'. "
                    f"Matches: {xml_assets[:5]}... Use full object_id for specific asset."
                )
            else:
                raise ValueError(
                    f"No assets found for XML '{reference}'. "
                    f"Available assets: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
        else:
            raise ValueError(
                f"Asset '{reference}' not found. "
                f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
            )

    def translate_sensors(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        include_irradiance_measures: bool = True,
        sensor_ids: Optional[Set[str]] = None,
        irradiance_disk_coords: Optional[Dict[str, tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """Translate generic sensors and measurements to Eradiate measures.

        Args:
            scene_description: Scene description with metadata
            scene_dir: Scene directory path
            include_irradiance_measures: Whether to include irradiance measurements
            sensor_ids: Optional set of sensor IDs to include. If None, all sensors
                are included. Use this to filter sensors for specific workflows
                (e.g., only BRF sensors for BRF workflow).
            irradiance_disk_coords: Optional mapping of IrradianceConfig ID to
                (x, y, z) disk coordinates. Required when include_irradiance_measures
                is True and IrradianceConfig measurements are present.

        Returns:
            List of Eradiate measure dictionaries

        Note:
            Sensors with for_reference_only=True are skipped during translation
            but remain accessible for geometry queries (e.g., pixel coordinate mapping).
        """
        self._current_scene_description = scene_description
        self._current_scene_dir = scene_dir

        asset_transforms = self._extract_asset_transforms(scene_description)

        measures = []

        for sensor in self.simulation_config.sensors:
            if sensor_ids is not None and sensor.id not in sensor_ids:
                continue

            if getattr(sensor, "for_reference_only", False):
                logger.debug(
                    f"Skipping reference-only sensor '{sensor.id}' "
                    f"(used for geometry specification only for pixel measurements, not simulated)"
                )
                continue

            if isinstance(sensor, SatelliteSensor):
                measures.append(
                    self.translate_satellite_sensor(sensor, scene_description)
                )
            elif isinstance(sensor, UAVSensor):
                measures.append(
                    self.translate_uav_sensor(
                        sensor, scene_description, scene_dir, asset_transforms
                    )
                )
            elif isinstance(sensor, GroundSensor):
                measures.append(
                    self.translate_ground_sensor(
                        sensor, scene_description, scene_dir, asset_transforms
                    )
                )
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor)}")

        if include_irradiance_measures:
            coords_map = irradiance_disk_coords or {}
            for measurement in self.simulation_config.measurements:
                if isinstance(measurement, IrradianceConfig):
                    disk_coords = coords_map.get(measurement.id)
                    if disk_coords is None:
                        # Disk coords not yet available (e.g. called before irradiance
                        # simulation has run, or from a workflow that doesn't use irradiance).
                        logger.debug(
                            f"Skipping irradiance measure '{measurement.id}': "
                            f"no disk coordinates provided."
                        )
                        continue
                    measures.append(
                        self.create_irradiance_measure(measurement, disk_coords)
                    )
                elif isinstance(
                    measurement, (HDRFConfig, HCRFConfig, BRFConfig, BHRConfig)
                ):
                    continue  # Handled in derived measurements
                else:
                    raise ValueError(
                        f"Unsupported measurement type: {type(measurement)}"
                    )

        return measures

    def translate_satellite_sensor(
        self, sensor: SatelliteSensor, scene_description: SceneDescription
    ) -> Dict[str, Any]:
        """Translate satellite sensor to Eradiate mpdistant measure.

        Args:
            sensor: Satellite sensor configuration
            scene_description: Scene description with location

        Returns:
            Eradiate measure dictionary
        """
        scene_location = scene_description.location
        scene_center_lat = scene_location.get("center_lat")
        scene_center_lon = scene_location.get("center_lon")

        if scene_center_lat is None or scene_center_lon is None:
            raise ValueError(
                "Scene description missing center_lat or center_lon in location"
            )

        coords = CoordinateSystem(scene_center_lat, scene_center_lon)

        if isinstance(sensor.target_size_km, (int, float)):
            width_km = height_km = sensor.target_size_km
        else:
            width_km, height_km = sensor.target_size_km

        target_bounds = coords.create_rectangle(
            sensor.target_center_lat, sensor.target_center_lon, width_km, height_km
        )

        measure_config = {
            "type": "mpdistant",
            "construct": "from_angles",
            "angles": [sensor.viewing.zenith, sensor.viewing.azimuth],
            "id": sanitize_sensor_id(sensor.id),
            "film_resolution": sensor.film_resolution,
            "target": {
                "type": "rectangle",
                "xmin": target_bounds["xmin"],
                "xmax": target_bounds["xmax"],
                "ymin": target_bounds["ymin"],
                "ymax": target_bounds["ymax"],
            },
            "srf": self.translate_srf(sensor.srf),
            "spp": sensor.samples_per_pixel,
        }

        return measure_config

    def translate_uav_sensor(
        self,
        sensor: UAVSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
        asset_transforms: Dict[str, _AssetTransform],
    ) -> Dict[str, Any]:
        """Translate UAV sensor to Eradiate measure.

        Args:
            sensor: UAV sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path
            asset_transforms: Asset transforms for relative positioning (backend-internal)

        Returns:
            Eradiate measure dictionary
        """
        view = sensor.viewing

        # Handle asset-relative positioning
        if hasattr(view, "relative_to_asset") and view.relative_to_asset is not None:
            asset_name = self._resolve_asset_reference(
                view.relative_to_asset, asset_transforms
            )
            asset_transform = asset_transforms[asset_name]

            view.origin = apply_asset_relative_transform(view.origin, asset_transform)

            if isinstance(view, LookAtViewing):
                view.target = apply_asset_relative_transform(
                    view.target, asset_transform
                )

            # Avoid double application of terrain adjustment
            if view.terrain_relative_height:
                view.terrain_relative_height = False

        origin, target = self.geometry_utils.adjust_origin_target_for_terrain(
            view, scene_description, scene_dir
        )

        base_config = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": self.translate_srf(sensor.srf),
            "origin": origin,
        }

        if sensor.instrument == UAVInstrumentType.PERSPECTIVE_CAMERA:
            base_config["type"] = "perspective"
            base_config["fov"] = sensor.fov or 70.0
            base_config["film_resolution"] = sensor.resolution or [1024, 1024]

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
                base_config["up"] = view.up or [0, 0, 1]
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self.geometry_utils.calculate_target_from_angles(view)
                base_config["target"] = target
                base_config["up"] = view.up or [0, 0, 1]

        elif sensor.instrument == UAVInstrumentType.RADIANCEMETER:
            base_config["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_config["target"] = target if target is not None else view.target
            elif isinstance(view, AngularFromOriginViewing):
                target, _ = self.geometry_utils.calculate_target_from_angles(view)
                base_config["target"] = target

        else:
            raise ValueError(f"Unsupported UAV instrument type: {sensor.instrument}")

        return base_config

    def translate_ground_sensor(
        self,
        sensor: GroundSensor,
        scene_description: SceneDescription,
        scene_dir: UPath,
        asset_transforms: Dict[str, _AssetTransform],
    ) -> Dict[str, Any]:
        """Translate ground sensor to Eradiate measure.

        Args:
            sensor: Ground sensor configuration
            scene_description: Scene description
            scene_dir: Scene directory path
            asset_transforms: Asset transforms for relative positioning (backend-internal)

        Returns:
            Eradiate measure dictionary
        """
        view = sensor.viewing

        if isinstance(view, DistantViewing):
            return self.translate_distant_viewing_to_mdistant(
                viewing=view,
                sensor_id=sensor.id,
                srf=sensor.srf,
                spp=sensor.samples_per_pixel,
                scene_description=scene_description,
                scene_dir=scene_dir,
            )

        simulation_srf = self.translate_srf(sensor.srf)

        base_measure = {
            "id": sanitize_sensor_id(sensor.id),
            "spp": sensor.samples_per_pixel,
            "srf": simulation_srf,
        }

        if hasattr(view, "relative_to_asset") and view.relative_to_asset is not None:
            asset_name = self._resolve_asset_reference(
                view.relative_to_asset, asset_transforms
            )
            asset_transform = asset_transforms[asset_name]

            if hasattr(view, "origin") and view.origin is not None:
                view.origin = apply_asset_relative_transform(
                    view.origin, asset_transform
                )

            if isinstance(view, LookAtViewing):
                view.target = apply_asset_relative_transform(
                    view.target, asset_transform
                )

            # Avoid double application of terrain adjustment
            if (
                hasattr(view, "terrain_relative_height")
                and view.terrain_relative_height
            ):
                view.terrain_relative_height = False

        if isinstance(view, HemisphericalViewing):
            base_measure["type"] = "hdistant"
            base_measure["direction"] = [0, 0, 1] if view.upward_looking else [0, 0, -1]

            if view.origin is not None:
                origin, _ = self.geometry_utils.adjust_origin_target_for_terrain(
                    view, scene_description, scene_dir
                )
                base_measure["target"] = origin
                logger.debug(
                    f"HemisphericalViewing sensor '{sensor.id}' target set to {origin}"
                )
            else:
                logger.warning(
                    f"HemisphericalViewing sensor '{sensor.id}' has no origin. "
                    f"Using scene center at ground level [0, 0, 0] as target. "
                    f"This may produce incorrect results."
                )
                base_measure["target"] = [0.0, 0.0, 0.0]

        elif isinstance(view, (LookAtViewing, AngularFromOriginViewing)):
            origin, target = self.geometry_utils.adjust_origin_target_for_terrain(
                view, scene_description, scene_dir
            )

            base_measure["origin"] = origin

            if sensor.instrument in [
                GroundInstrumentType.PERSPECTIVE_CAMERA,
                GroundInstrumentType.DHP_CAMERA,
            ]:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [1024, 1024]
                base_measure["fov"] = sensor.fov or (
                    180.0
                    if sensor.instrument == GroundInstrumentType.DHP_CAMERA
                    else 70.0
                )
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.HYPSTAR:
                base_measure["type"] = "perspective"
                base_measure["film_resolution"] = sensor.resolution or [5, 5]
                base_measure["fov"] = sensor.fov or 5.0
                base_measure["up"] = view.up or [0, 0, 1]

            elif sensor.instrument == GroundInstrumentType.RADIANCEMETER:
                base_measure["type"] = "radiancemeter"

            else:
                base_measure["type"] = "radiancemeter"

            if isinstance(view, LookAtViewing):
                base_measure["target"] = target if target is not None else view.target

            elif isinstance(view, AngularFromOriginViewing):
                target, direction = self.geometry_utils.calculate_target_from_angles(
                    view
                )
                base_measure["target"] = target

        else:
            raise ValueError(
                f"Unsupported viewing type for ground sensor: {type(view)}"
            )

        return base_measure

    def translate_distant_viewing_to_mdistant(
        self,
        viewing: DistantViewing,
        sensor_id: str,
        srf: Any,
        spp: int,
        scene_description: SceneDescription,
        scene_dir: UPath,
    ) -> Dict[str, Any]:
        """Translate DistantViewing to Eradiate mdistant measure.

        Uses Eradiate's MultiDistantMeasure which measures radiance from
        a distant sensor looking at a target area. Used for pixel-level
        measurements with rectangular targets.

        Args:
            viewing: DistantViewing configuration
            sensor_id: Unique sensor identifier
            srf: Spectral response function
            spp: Samples per pixel
            scene_description: Scene description for terrain lookup
            scene_dir: Scene directory for terrain data

        Returns:
            Eradiate mdistant measure dictionary
        """
        measure = {
            "type": "mdistant",
            "id": sanitize_sensor_id(sensor_id),
            "srf": self.translate_srf(srf),
            "spp": spp,
        }

        # Convert direction vector to angles
        # Direction [0, 0, 1] means looking down (nadir) -> zenith=0
        dx, dy, dz = viewing.direction or [0, 0, 1]
        horiz = np.sqrt(dx**2 + dy**2)
        zenith = np.degrees(np.arctan2(horiz, dz))
        azimuth = np.degrees(np.arctan2(dy, dx)) % 360 if horiz > 1e-10 else 0.0

        measure["construct"] = "from_angles"
        measure["angles"] = [zenith, azimuth]

        # Handle target specification
        target = viewing.target
        if target is None:
            # Default to point target at scene center
            measure["target"] = [0.0, 0.0, 0.0]
        elif isinstance(target, RectangleTarget):
            # Rectangle target for pixel BRF measurements
            measure["target"] = {
                "type": "rectangle",
                "xmin": target.xmin,
                "xmax": target.xmax,
                "ymin": target.ymin,
                "ymax": target.ymax,
                "z": target.z,
            }
            logger.debug(
                f"DistantViewing sensor '{sensor_id}' using rectangle target: "
                f"x=[{target.xmin:.1f}, {target.xmax:.1f}], "
                f"y=[{target.ymin:.1f}, {target.ymax:.1f}], z={target.z:.1f}"
            )
        elif isinstance(target, list) and len(target) == 3:
            # Point target - adjust for terrain if needed
            x, y, z = target
            if viewing.terrain_relative_height:
                terrain_z = self.geometry_utils.query_terrain_elevation(
                    scene_description, scene_dir, x, y
                )
                z = terrain_z + z
            measure["target"] = [x, y, z]
            logger.debug(
                f"DistantViewing sensor '{sensor_id}' using point target: [{x:.1f}, {y:.1f}, {z:.1f}]"
            )
        else:
            raise ValueError(
                f"Invalid target specification for DistantViewing: {target}. "
                f"Expected None, [x, y, z], or RectangleTarget."
            )

        # Optional ray offset
        if viewing.ray_offset is not None:
            measure["ray_offset"] = viewing.ray_offset

        return measure

    def create_irradiance_measure(
        self, irradiance_config, disk_coords: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Create hdistant measure for BOA irradiance measurement.

        The target parameter is set to the white reference disk location.

        Args:
            irradiance_config: IrradianceConfig with measurement parameters
            disk_coords: Pre-computed disk position (x, y, z) in scene coordinates

        Returns:
            Eradiate hdistant measure dictionary with target set to disk location
        """
        if irradiance_config.location and irradiance_config.location.srf:
            srf = self.translate_srf(irradiance_config.location.srf)
            spp = irradiance_config.location.samples_per_pixel
        else:
            srf = {
                "type": "uniform",
                "wmin": 380.0,
                "wmax": 1680.0,
            }
            spp = irradiance_config.samples_per_pixel or 512

        x, y, z = disk_coords

        logger.debug(
            f"Creating hdistant measure '{irradiance_config.id}' with target at "
            f"({x:.2f}, {y:.2f}, {z:.2f}) m, spp={spp}, ray_offset=0.1m"
        )

        return {
            "type": "hdistant",
            "id": sanitize_sensor_id(irradiance_config.id),
            "target": [x, y, z],  # Point at disk location
            "direction": [0, 0, 1],  # Upward-looking hemisphere
            "ray_offset": 0.1,  # Small offset to prevent self-intersection
            "srf": srf,
            "spp": spp,
        }

    def create_distant_flux_measure(
        self,
        bhr_config: BHRConfig,
        target_coords: Tuple[float, float, float],
        is_reference: bool = False,
    ) -> Dict[str, Any]:
        """Create distant_flux measure for BHR measurement.

        The distant_flux measure computes radiosity (power per unit area)
        by integrating over all directions in a hemisphere. This is used
        for BHR computation which requires radiosity ratios.

        Args:
            bhr_config: BHR configuration
            target_coords: Target coordinates (x, y, z) in scene coordinate system
            is_reference: If True, this is for the white reference simulation

        Returns:
            Eradiate distant_flux measure dictionary
        """
        # Create unique measure ID based on whether this is surface or reference
        if is_reference:
            measure_id = f"bhr_reference_{bhr_config.id}"
        else:
            measure_id = f"bhr_surface_{bhr_config.id}"

        x, y, z = target_coords

        srf = self.translate_srf(bhr_config.srf)
        spp = bhr_config.samples_per_pixel

        logger.debug(
            f"Creating distant_flux measure '{measure_id}' at target "
            f"({x:.2f}, {y:.2f}, {z:.2f}) m, spp={spp}"
        )

        if (
            not is_reference
            and bhr_config.viewing is not None
            and isinstance(bhr_config.viewing.target, RectangleTarget)
        ):
            rect = bhr_config.viewing.target
            target = {
                "type": "rectangle",
                "xmin": rect.xmin,
                "xmax": rect.xmax,
                "ymin": rect.ymin,
                "ymax": rect.ymax,
                "z": rect.z,
            }
            logger.debug(
                f"distant_flux '{measure_id}' using rectangle target: "
                f"x=[{rect.xmin:.1f}, {rect.xmax:.1f}], "
                f"y=[{rect.ymin:.1f}, {rect.ymax:.1f}], z={rect.z:.1f}"
            )
        else:
            target = [x, y, z]

        return {
            "type": "distant_flux",
            "id": sanitize_sensor_id(measure_id),
            "target": target,
            "direction": [
                0,
                0,
                1,
            ],  # Upward-looking (measures downward flux onto surface)
            "ray_offset": 1.0,
            "srf": srf,
            "spp": spp,
        }

    def translate_srf(self, srf) -> Union[Dict[str, Any], str]:
        """Translate generic SRF to Eradiate format.

        For gaussian SRF, returns a uniform SRF covering the full wavelength range.
        The actual Gaussian convolution is applied during post-processing, which is
        more efficient than running individual per-wavelength simulations.

        Args:
            srf: SRF configuration object (SpectralResponse or str)

        Returns:
            Eradiate SRF dictionary or identifier string
        """
        if srf is None:
            return None

        # Handle string SRF (dataset identifier)
        if isinstance(srf, str):
            return srf

        if srf.type == "delta":
            return {
                "type": "multi_delta",
                "wavelengths": srf.wavelengths,
            }
        elif srf.type == "uniform":
            return {
                "type": "uniform",
                "wmin": srf.wmin,
                "wmax": srf.wmax,
            }
        elif srf.type == "dataset":
            return srf.dataset_id
        elif srf.type == "gaussian":
            # Gaussian SRF: simulate uniformly over the spectral range,
            # then apply Gaussian SRF convolution in post-processing.
            if srf.output_grid is not None:
                grid = srf.output_grid
                if grid.mode == "regular":
                    wmin = grid.wmin_nm
                    wmax = grid.wmax_nm
                elif grid.mode == "explicit":
                    wmin = min(grid.wavelengths_nm)
                    wmax = max(grid.wavelengths_nm)
                else:
                    # from_file: wavelengths only known at runtime; derive
                    # simulation range from spectral_regions if present.
                    if srf.spectral_regions:
                        wmin = min(r.wmin_nm for r in srf.spectral_regions)
                        wmax = max(r.wmax_nm for r in srf.spectral_regions)
                    else:
                        wmin = 400.0
                        wmax = 2500.0
                    logger.debug(
                        f"Gaussian SRF with from_file output grid: using "
                        f"{wmin}–{wmax} nm for simulation (from spectral_regions)."
                    )
            elif srf.spectral_regions:
                # Infer simulation range from spectral region bounds
                wmin = min(r.wmin_nm for r in srf.spectral_regions)
                wmax = max(r.wmax_nm for r in srf.spectral_regions)
            else:
                wmin, wmax = 400.0, 2500.0  # broad fallback
            return {
                "type": "uniform",
                "wmin": wmin,
                "wmax": wmax,
            }
        else:
            raise ValueError(f"Unsupported SRF type: {srf.type}")

    def generate_radiance_sensor_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> GroundSensor:
        """Generate radiance sensor for HDRF measurement.

        Args:
            hdrf_config: HDRF configuration

        Returns:
            Generated GroundSensor
        """
        sensor_id = f"hdrf_radiance_{hdrf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=hdrf_config.viewing,
            srf=hdrf_config.srf,
            samples_per_pixel=hdrf_config.samples_per_pixel or 1000,
        )

    def generate_irradiance_measurement_for_hdrf(
        self, hdrf_config: HDRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HDRF.

        Args:
            hdrf_config: HDRF configuration

        Returns:
            Generated IrradianceConfig
        """
        irradiance_id = f"hdrf_irradiance_{hdrf_config.id}"

        # Update location with srf from hdrf_config
        location_with_srf = hdrf_config.location.model_copy(
            update={"srf": hdrf_config.srf}
        )

        return IrradianceConfig(
            id=irradiance_id,
            location=location_with_srf,
        )

    def generate_camera_sensor_for_hcrf(self, hcrf_config: HCRFConfig) -> GroundSensor:
        """Generate camera sensor for HCRF measurement.

        Args:
            hcrf_config: HCRF configuration

        Returns:
            Generated GroundSensor
        """
        sensor_id = f"hcrf_camera_{hcrf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.PERSPECTIVE_CAMERA,
            viewing=hcrf_config.viewing,
            fov=hcrf_config.fov,
            resolution=hcrf_config.film_resolution,
            srf=hcrf_config.srf,
            samples_per_pixel=hcrf_config.samples_per_pixel or 1000,
            post_processing=PostProcessingOptions(spatial_averaging=True),
        )

    def generate_irradiance_measurement_for_hcrf(
        self, hcrf_config: HCRFConfig
    ) -> IrradianceConfig:
        """Generate irradiance measurement for HCRF.

        Args:
            hcrf_config: HCRF configuration

        Returns:
            Generated IrradianceConfig
        """
        irradiance_id = f"hcrf_irradiance_{hcrf_config.id}"

        # Update location with srf from hcrf_config
        location_with_srf = hcrf_config.location.model_copy(
            update={"srf": hcrf_config.srf}
        )

        return IrradianceConfig(
            id=irradiance_id,
            location=location_with_srf,
        )

    def generate_radiance_sensor_for_brf(self, brf_config: BRFConfig) -> GroundSensor:
        """Generate radiance sensor for BRF measurement (no atmosphere).

        Args:
            brf_config: BRF configuration

        Returns:
            Generated GroundSensor for radiance measurement
        """
        sensor_id = f"brf_radiance_{brf_config.id}"

        return GroundSensor(
            id=sensor_id,
            instrument=GroundInstrumentType.RADIANCEMETER,
            viewing=brf_config.viewing,
            srf=brf_config.srf,
            samples_per_pixel=brf_config.samples_per_pixel or 512,
        )

    def get_sensor_by_id(self, sensor_id: str):
        """Find sensor config by ID.

        Args:
            sensor_id: Sensor ID to find

        Returns:
            Sensor configuration or None
        """
        for sensor in self.simulation_config.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def apply_post_processing_to_sensors(
        self, sensor_results: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Apply post-processing to sensor results.

        Args:
            sensor_results: Dictionary of sensor results

        Returns:
            Dictionary of post-processed results
        """
        from ...processors.sensor_processor import SensorProcessor

        sensor_processor = SensorProcessor(self.simulation_config)
        processed_results = {}

        for sensor_id, dataset in sensor_results.items():
            logger.debug(f"Processing sensor: {sensor_id}")
            sensor = self.get_sensor_by_id(sensor_id)
            if sensor:
                logger.debug(f"Found sensor config: {sensor_id}")
                post_processed_dataset = sensor_processor.process_sensor_result(
                    dataset, sensor
                )
                # Only save raw version if post-processing changed the data
                if post_processed_dataset is not dataset:
                    logger.debug(f"Post-processing applied: {sensor_id}")
                    processed_results[f"{sensor_id}_raw_eradiate"] = dataset
                processed_results[sensor_id] = post_processed_dataset
            else:
                logger.debug(f"No sensor config found: {sensor_id}, passing through")
                processed_results[sensor_id] = dataset
        return processed_results
