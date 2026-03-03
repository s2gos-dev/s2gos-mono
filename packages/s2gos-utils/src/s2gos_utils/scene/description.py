import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from upath import UPath

from .materials import Material
from .._version import get_version
from ..io.paths import open_file
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

    _LIST_FIELDS = ("objects", "material_regions")
    _DICT_FIELDS = ("materials", "material_indices")

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

    def save_yaml(self, output_path: UPath) -> None:
        """Save scene description as YAML file."""
        with open_file(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def save_json(self, output_path: UPath) -> None:
        """Save scene description as JSON file."""
        with open_file(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_yaml(cls, file_path: UPath) -> "SceneDescription":
        """Load scene description from YAML file with version validation."""
        with open_file(file_path, "r") as f:
            data = yaml.safe_load(f)

        validated_data = validate_config_version(
            "scene_description", data, get_version(), "scene description"
        )

        return cls.from_dict(validated_data, base_dir=UPath(file_path).parent)

    @classmethod
    def load_json(cls, file_path: UPath) -> "SceneDescription":
        """Load scene description from JSON file with version validation."""
        with open_file(file_path, "r") as f:
            data = json.load(f)

        validated_data = validate_config_version(
            "scene_description", data, get_version(), "scene description"
        )

        return cls.from_dict(validated_data, base_dir=UPath(file_path).parent)

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
        # Handle backward compatibility: move extent_km to target.size_km if needed
        target = data.get("target", {})
        if "extent_km" in data and "size_km" not in target:
            target = dict(target)  # Make a copy
            target["size_km"] = data["extent_km"]

        include_files = data.pop("include_files", [])

        scene = cls(
            name=data["name"],
            location=data["location"],
            resolution_m=data["resolution_m"],
            schema_version=data.get("schema_version", get_version()),
            atmosphere=data.get("atmosphere"),
            target=target,
            buffer=data.get("buffer"),
            background=data.get("background"),
            objects=data.get("objects", []),
            material_indices=data.get("material_indices", {}),
            metadata=data.get("metadata", {}),
            include_files=include_files,
        )

        if "materials" in data:
            for name, mat_data in data["materials"].items():
                mat_type = mat_data.pop("type")
                scene.add_material(name, mat_type, **mat_data)

        if base_dir:
            for rel_path in include_files:
                ext_path = base_dir / rel_path
                if not ext_path.exists():
                    raise FileNotFoundError(
                        f"Include file not found: {ext_path} "
                        f"(referenced from scene description)"
                    )
                cls._merge_include(scene, cls._load_include_file(ext_path))

        return scene

    @staticmethod
    def _load_include_file(path: UPath) -> Dict[str, Any]:
        """Load a YAML or JSON include file."""
        suffix = str(path).rsplit(".", 1)[-1].lower()
        with open_file(path, "r") as f:
            if suffix == "json":
                return json.load(f)
            else:
                return yaml.safe_load(f) or {}

    @classmethod
    def _merge_include(cls, scene: "SceneDescription", data: Dict[str, Any]) -> None:
        """Merge data from an include file into *scene*.

        Rules:
        - List fields (objects, material_regions): extend
        - Dict fields (materials, material_indices): update, main wins on conflict
        - All other fields: ignored
        """
        for field_name in cls._LIST_FIELDS:
            if field_name in data:
                getattr(scene, field_name).extend(data[field_name])

        for field_name in cls._DICT_FIELDS:
            if field_name not in data:
                continue
            existing = getattr(scene, field_name)
            incoming = data[field_name]
            if field_name == "materials":
                # Materials need Material.from_dict conversion
                for name, mat_data in incoming.items():
                    if name not in existing:
                        mat_type = mat_data.pop("type")
                        scene.add_material(name, mat_type, **mat_data)
            else:
                # For dicts like material_indices, merge with main winning
                for k, v in incoming.items():
                    if k not in existing:
                        existing[k] = v
