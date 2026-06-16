import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from upath import UPath

from .materials import Material
from .._version import get_version
from ..io.paths import PathLike, open_file, to_upath
from ..versioning import validate_config_version


@dataclass
class SceneDescription:
    """Complete scene description."""

    name: str
    location: Dict[str, float]
    resolution_m: float

    schema_version: str = field(default_factory=get_version)
    materials: Dict[str, Material] = field(default_factory=dict)

    atmosphere: Optional[Dict[str, Any]] = None
    target: Optional[Dict[str, Any]] = None
    buffer: Optional[Dict[str, Any]] = None
    background: Optional[Dict[str, Any]] = None
    objects: List[Dict[str, Any]] = field(default_factory=list)
    material_indices: Dict[int, str] = field(default_factory=dict)
    material_regions: List[Dict[str, Any]] = field(default_factory=list)
    include_files: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_material(self, name: str, material_type: str, **properties) -> None:
        """Add a material definition."""
        material_dict = {"type": material_type, **properties}
        self.materials[name] = Material.from_dict(material_dict, id=name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "schema_version": self.schema_version,
            "name": self.name,
            "location": self.location,
            "resolution_m": self.resolution_m,
        }

        if self.materials:
            result["materials"] = {
                name: mat.to_dict() for name, mat in self.materials.items()
            }

        # Add core scene components
        if self.atmosphere:
            result["atmosphere"] = self.atmosphere
        if self.target:
            result["target"] = self.target
        if self.buffer:
            result["buffer"] = self.buffer
        if self.background:
            result["background"] = self.background
        if self.objects:
            result["objects"] = self.objects
        if self.material_indices:
            result["material_indices"] = self.material_indices
        if self.material_regions:
            result["material_regions"] = self.material_regions
        if self.include_files:
            result["include_files"] = list(self.include_files)

        # Only include metadata if it has useful data
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def save_yaml(self, output_path: PathLike) -> None:
        """Save scene description as YAML file."""
        with open_file(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def save_json(self, output_path: PathLike) -> None:
        """Save scene description as JSON file."""
        with open_file(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_yaml(cls, file_path: PathLike) -> "SceneDescription":
        """Load scene description from YAML file with version validation."""
        file_path = to_upath(file_path)
        with open_file(file_path, "r") as f:
            data = yaml.safe_load(f)

        validated_data = validate_config_version(
            "scene_description", data, get_version(), "scene description"
        )

        return cls.from_dict(validated_data, base_dir=file_path.parent)

    @classmethod
    def load_json(cls, file_path: PathLike) -> "SceneDescription":
        """Load scene description from JSON file with version validation."""
        file_path = to_upath(file_path)
        with open_file(file_path, "r") as f:
            data = json.load(f)

        validated_data = validate_config_version(
            "scene_description", data, get_version(), "scene description"
        )

        return cls.from_dict(validated_data, base_dir=file_path.parent)

    @staticmethod
    def _deep_merge(target: dict, source: dict) -> None:
        """Merge *source* into *target* recursively.  *target* wins on conflicts.

        - Dicts: merge recursively, only adding keys missing from *target*
        - Lists: extend *target* with *source* items
        - Scalars: *target* wins (never overwritten)
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(target[key], dict) and isinstance(value, dict):
                SceneDescription._deep_merge(target[key], value)
            elif isinstance(target[key], list) and isinstance(value, list):
                target[key].extend(value)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_dir: Optional[UPath] = None
    ) -> "SceneDescription":
        """Create scene description from dictionary.

        Args:
            data: Dictionary with scene description fields.
            base_dir: Directory used to resolve ``include_files`` paths.
                When *None*, includes are stored but not resolved.
        """
        include_files = data.pop("include_files", [])

        if base_dir:
            for rel_path in include_files:
                ext_path = base_dir / rel_path
                if not ext_path.exists():
                    raise FileNotFoundError(
                        f"Include file not found: {ext_path} "
                        f"(referenced from scene description)"
                    )
                cls._deep_merge(data, cls._load_include_file(ext_path))

        materials: Dict[str, Material] = {}
        if "materials" in data:
            for name, mat_data in data["materials"].items():
                materials[name] = Material.from_dict(mat_data, id=name)

        return cls(
            name=data["name"],
            location=data["location"],
            resolution_m=data["resolution_m"],
            schema_version=data.get("schema_version", get_version()),
            materials=materials,
            atmosphere=data.get("atmosphere"),
            target=data.get("target"),
            buffer=data.get("buffer"),
            background=data.get("background"),
            objects=data.get("objects", []),
            material_indices=data.get("material_indices", {}),
            material_regions=data.get("material_regions", []),
            metadata=data.get("metadata", {}),
            include_files=include_files,
        )

    def resolve_includes(self, base_dir: UPath) -> "SceneDescription":
        """Return a new SceneDescription with include files merged.

        Round-trips through dict form so that ``from_dict`` handles includes
        uniformly at the data level.  Returns *self* unchanged when there are
        no include files.
        """
        if not self.include_files:
            return self
        return self.from_dict(self.to_dict(), base_dir=base_dir)

    @staticmethod
    def _load_include_file(path: UPath) -> Dict[str, Any]:
        """Load a YAML or JSON include file."""
        suffix = str(path).rsplit(".", 1)[-1].lower()
        with open_file(path, "r") as f:
            if suffix == "json":
                return json.load(f)
            else:
                return yaml.safe_load(f) or {}
