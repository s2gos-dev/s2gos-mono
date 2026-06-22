from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from s2gos_utils import validate_config_version
from s2gos_utils.io.paths import PathRef, open_file
from s2gos_utils.io.resolver import resolver
from s2gos_utils.setting.paths import to_pathref

from .assets import HamsterConfig, MaterialRegion, UserAssets, XmlSceneConfig
from .atmosphere import (
    AtmosphereConfig,
    ThermophysicalConfig,
    _default_atmosphere_config,
)
from .buildings import BuildingsConfig
from .material_match import SpectralMatchingConfig
from .mesh_refinement import MeshRefinementConfig
from .roads import RoadsConfig
from .vegetation import VegetationExclusionZone, VegetationPlacementConfig
from ..._version import get_version
from ...dataset import IndexedGeoTiff, Zarr, dataset_factory


class Month(str, Enum):
    """Month selection for seasonal adjustments."""

    JUNE = "june"
    DECEMBER = "december"


class SnowConfig(BaseModel):
    """Seasonal snow configuration. Presence (non-None) enables snow."""

    season_month: Month = Field(
        ..., description="Month for seasonal snow calculation (JUNE or DECEMBER)"
    )
    material_index: int = Field(
        6, description="Material index to use for snow coverage"
    )
    thermoprops: Optional[ThermophysicalConfig] = Field(
        None,
        description="Optional CAMS thermoprops for snow temperature calculation. "
        "If None, uses synthetic temperature model.",
    )
    random_seed: Optional[int] = Field(
        None,
        ge=0,
        description="Random seed for reproducible snow mask generation. If None, uses system entropy.",
    )


class BufferConfig(BaseModel):
    """Buffer area configuration. Presence (non-None) enables buffer processing."""

    size_km: float = Field(60.0, gt=0.0, description="Buffer size in kilometers")
    resolution_m: float = Field(
        100.0, gt=0.0, description="Buffer resolution in meters"
    )


class BackgroundConfig(BaseModel):
    """Background area configuration. Presence (non-None) enables background processing."""

    size_km: float = Field(
        200.0, gt=0.0, description="Background area size in kilometers"
    )
    resolution_m: float = Field(
        200.0, gt=0.0, description="Background resolution in meters"
    )
    elevation: float = Field(0.0, description="Background elevation in meters")


class SceneLocation(BaseModel):
    """Geographic location configuration."""

    center_lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Center latitude in degrees"
    )
    center_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Center longitude in degrees"
    )
    aoi_size_km: float = Field(
        ..., gt=0.0, description="Area of interest size in kilometers"
    )


def _load_settings_data_sources_config() -> Dict[str, Any]:
    """Load paths from s2gos_settings.toml file and augment global resolver.

    This function serves dual purpose:
    1. Load data source paths (DEM, landcover, materials) from the settings file.
    2. Augment the global resolver with custom asset search paths
    """
    from ...setting import settings

    dem_settings = settings.generator.dataset.dem
    landcover_settings = settings.generator.dataset.landcover
    material_config = settings.generator.files.material_config

    return {
        "dem": dataset_factory(dem_settings, dem_settings.get("name", "DEM")),
        "landcover": dataset_factory(
            landcover_settings, landcover_settings.get("name", "Landcover")
        ),
        "material_config_path": to_pathref(material_config),
    }


class DataSources(BaseModel):
    """Data source configuration using FileResolver."""

    dem: IndexedGeoTiff | Zarr = Field(..., description="DEM Dataset")
    landcover: IndexedGeoTiff | Zarr = Field(..., description="Landcover Dataset")
    material_config_path: PathRef = Field(
        ..., description="Path to custom material configuration JSON"
    )

    @model_validator(mode="before")
    @classmethod
    def _load_defaults_and_merge_overrides(cls, data: Any) -> Any:
        """
        Load defaults from YAML and merge them with user-provided data.
        This allows a base configuration to be set while still allowing
        users to specify their own paths. The user's data takes precedence.
        """
        if not isinstance(data, dict):
            # Let Pydantic handle validation for non-dictionary inputs.
            return data

        default_config = _load_settings_data_sources_config()
        default_config.update(data)
        return default_config

    @field_validator(
        "material_config_path",
    )
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = resolver.resolve(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


class ProcessingOptions(BaseModel):
    """Processing options for scene generation."""

    generate_texture_preview: bool = Field(
        True, description="Generate texture preview images"
    )
    handle_dem_nans: bool = Field(True, description="Handle NaN values in DEM data")
    dem_fillna_value: float = Field(0.0, description="Fill value for DEM NaN values")
    flatten_dem: bool = Field(
        False, description="Flatten DEM to zero elevation for testing"
    )


class SceneGenConfig(BaseModel):
    """Comprehensive scene generation configuration.

    Provides a validated configuration system for all scene generation parameters.
    Pass an instance to
    [SceneGenerationPipeline][s2gos_generator.core.pipeline.SceneGenerationPipeline]
    and call ``run()`` to produce a
    [SceneDescription][s2gos_utils.scene.description.SceneDescription].

    Optional spatial zones are controlled by
    [BufferConfig][s2gos_generator.core.config.scene.BufferConfig] and
    [BackgroundConfig][s2gos_generator.core.config.scene.BackgroundConfig].
    Vegetation placement is configured via
    [VegetationPlacementConfig][s2gos_generator.core.config.vegetation.VegetationPlacementConfig].
    """

    config_version: str = Field(
        default_factory=get_version, description="Configuration schema version"
    )
    scene_name: str = Field(
        ..., min_length=1, description="Scene name (used for output files)"
    )
    description: Optional[str] = Field(
        None, description="Metadata, only used in case of serialization"
    )

    location: SceneLocation = Field(..., description="Geographic location")
    data_sources: DataSources = Field(..., description="Data source configuration")
    output_dir: PathRef = Field(..., description="Output directory for generated scene")

    dem_resolution_m: float = Field(
        30.0,
        gt=0.0,
        description="DEM resolution in meters (controls terrain mesh detail)",
    )
    landcover_resolution_m: float = Field(
        30.0,
        gt=0.0,
        description="Landcover resolution in meters",
    )
    texture_resolution_m: Optional[float] = Field(
        None,
        gt=0.0,
        description=(
            "Texture resolution in meters per pixel. "
            "When set finer than landcover_resolution_m, roads and landcover are rasterized "
            "at higher pixel density, reducing road blockiness. "
            "Defaults to native landcover resolution when None."
        ),
    )

    snow: Optional[SnowConfig] = Field(
        None,
        description="Seasonal snow configuration (None disables snow). See [SnowConfig][s2gos_generator.core.config.scene.SnowConfig].",
    )

    processing: ProcessingOptions = Field(
        default_factory=ProcessingOptions, description="Processing options"
    )
    atmosphere: AtmosphereConfig = Field(
        default_factory=_default_atmosphere_config,
        description="Atmosphere configuration",
    )
    buffer: Optional[BufferConfig] = Field(
        None,
        description="Buffer area configuration (None disables buffer). See [BufferConfig][s2gos_generator.core.config.scene.BufferConfig].",
    )
    background: Optional[BackgroundConfig] = Field(
        None,
        description="Background area configuration (None disables background). See [BackgroundConfig][s2gos_generator.core.config.scene.BackgroundConfig].",
    )
    hamster: Optional[HamsterConfig] = Field(
        None,
        description="HAMSTER albedo data configuration for baresoil. See [HamsterConfig][s2gos_generator.core.config.assets.HamsterConfig].",
    )
    roads: Optional[RoadsConfig] = Field(
        None,
        description="Road infrastructure configuration (None disables roads). See [RoadsConfig][s2gos_generator.core.config.roads.RoadsConfig].",
    )
    spectral_matching: Optional[SpectralMatchingConfig] = Field(
        None,
        description=(
            "Spectral matching configuration (None disables). See "
            "[SpectralMatchingConfig][s2gos_generator.core.config.material_matching.SpectralMatchingConfig]."
        ),
    )
    buildings: Optional[BuildingsConfig] = Field(
        None,
        description="Building configuration (None disables buildings). See [BuildingsConfig][s2gos_generator.core.config.buildings.BuildingsConfig].",
    )
    mesh_refinement: Optional[MeshRefinementConfig] = Field(
        None,
        description="Adaptive mesh refinement configuration (None disables refinement). See [MeshRefinementConfig][s2gos_generator.core.config.mesh_refinement.MeshRefinementConfig].",
    )
    user_assets: list[UserAssets] = Field(
        [],
        description="User assets to be placed in generated scene. See [UserAssets][s2gos_generator.core.config.assets.UserAssets].",
    )
    xml_scenes: list[XmlSceneConfig] = Field(
        [],
        description="XML scene files to import for additional assets and materials. See [XmlSceneConfig][s2gos_generator.core.config.assets.XmlSceneConfig].",
    )
    material_regions: list[MaterialRegion] = Field(
        [],
        description="Material regions for spatially-selective material overrides. See [MaterialRegion][s2gos_generator.core.config.assets.MaterialRegion].",
    )
    region_material_defs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Material definitions for region materials"
    )
    vegetation_placement: Optional[VegetationPlacementConfig] = Field(
        None,
        description="Vegetation placement configuration (None disables vegetation). See [VegetationPlacementConfig][s2gos_generator.core.config.vegetation.VegetationPlacementConfig].",
    )
    vegetation_exclusion_zones: List[VegetationExclusionZone] = Field(
        default_factory=list,
        description="Standalone vegetation exclusion zones",
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Configuration creation time"
    )

    model_config = {
        "validate_assignment": True,
        # "extra": "forbid",
        "arbitrary_types_allowed": True,
    }

    @field_serializer("created_at", when_used="json")
    def serialize_datetime_iso(self, created_at: datetime):
        return created_at.isoformat()

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v):
        """Validate and create output directory if needed."""
        from s2gos_utils.io.paths import mkdir

        mkdir(v)
        return v

    @model_validator(mode="after")
    def validate_scene_config(self):
        """Validate complete scene configuration."""
        if self.buffer is not None:
            if self.buffer.size_km <= self.location.aoi_size_km:
                raise ValueError("Buffer size must be larger than AOI size")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self, path: Optional[PathRef] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            with open_file(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, path: PathRef) -> SceneGenConfig:
        """Load from JSON file."""
        with open_file(path, "r") as f:
            data = json.load(f)

        validate_config_version(
            "scene_config", data, get_version(), "scene generation configuration"
        )

        return cls(**data)

    def enable_hamster_albedo(
        self,
        data_path: PathRef,
        variable_name: str = "albedo",
        fallback_on_error: bool = True,
    ):
        """Enable HAMSTER albedo data for baresoil material replacement.

        Args:
            data_path: Path to HAMSTER NetCDF data file
            variable_name: Variable name in NetCDF file (default: "albedo")
            fallback_on_error: Fall back to standard baresoil material on errors
        """
        self.hamster = HamsterConfig(
            enabled=True,
            data_path=data_path,
            variable_name=variable_name,
            fallback_on_error=fallback_on_error,
        )

    def disable_hamster_albedo(self):
        """Disable HAMSTER albedo system."""
        self.hamster = None

    def set_atmosphere_homogeneous(
        self,
        aerosol_dataset,
        optical_thickness: float = 0.1,
        scale_height: float = 1000.0,
    ):
        """Set atmosphere using homogeneous configuration."""
        from .atmosphere import HomogeneousAtmosphereConfig

        homogeneous_config = HomogeneousAtmosphereConfig(
            aerosol_dataset=aerosol_dataset,
            optical_thickness=optical_thickness,
            scale_height=scale_height,
        )
        self.atmosphere = AtmosphereConfig(
            details=homogeneous_config,
        )

    def set_atmosphere_molecular(self, molecular_config):
        """Set atmosphere using molecular configuration."""
        self.atmosphere = AtmosphereConfig(
            details=molecular_config,
        )

    def set_atmosphere_heterogeneous(
        self,
        molecular_config=None,
        particle_layers=None,
    ):
        """Set atmosphere using heterogeneous configuration with molecular and particle layers."""
        from .atmosphere import HeterogeneousAtmosphereConfig

        heterogeneous_config = HeterogeneousAtmosphereConfig(
            molecular=molecular_config, particle_layers=particle_layers
        )
        self.atmosphere = AtmosphereConfig(
            details=heterogeneous_config,
        )

    @property
    def scene_output_dir(self) -> PathRef:
        """Get the specific output directory for this scene."""
        return self.output_dir / self.scene_name

    @property
    def meshes_dir(self) -> PathRef:
        """Get the meshes output directory."""
        return self.scene_output_dir / "meshes"

    @property
    def textures_dir(self) -> PathRef:
        """Get the textures output directory."""
        return self.scene_output_dir / "textures"

    @property
    def data_dir(self) -> PathRef:
        """Get the data output directory."""
        return self.scene_output_dir / "data"


def create_scene_config(
    scene_name: str,
    center_lat: float,
    center_lon: float,
    aoi_size_km: float,
    output_dir: PathRef,
    dem_resolution_m: float = 30.0,
    landcover_resolution_m: float = 10.0,
    description: Optional[str] = None,
    data_overrides: Optional[dict] = None,
    atmosphere: Optional[AtmosphereConfig] = None,
    **kwargs,
) -> SceneGenConfig:
    """Scene generation configuration using PathResolver.

    Args:
        scene_name: Scene name (used for output files)
        center_lat: Center latitude in degrees
        center_lon: Center longitude in degrees
        aoi_size_km: Area of interest size in kilometers
        output_dir: Output directory for generated scene
        dem_resolution_m: DEM resolution in meters (default: 30.0)
        landcover_resolution_m: Landcover resolution in meters (default: 30.0)
        description: Optional scene description
        data_overrides: Optional dict with user data overrides:
            - dem_index: Custom DEM index file
            - landcover_index: Custom landcover index file
            - materials_config: Custom materials config file
        atmosphere: Optional atmosphere configuration
        **kwargs: Additional configuration options
    """
    data_sources = DataSources(**(data_overrides or {}))

    return SceneGenConfig(
        scene_name=scene_name,
        description=description,
        location=SceneLocation(
            center_lat=center_lat, center_lon=center_lon, aoi_size_km=aoi_size_km
        ),
        data_sources=data_sources,
        output_dir=output_dir,
        dem_resolution_m=dem_resolution_m,
        landcover_resolution_m=landcover_resolution_m,
        atmosphere=atmosphere or _default_atmosphere_config(),
        **kwargs,
    )
