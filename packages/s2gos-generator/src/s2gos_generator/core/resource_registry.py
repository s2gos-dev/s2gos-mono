import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .context import SceneResourceContext


class Resource:
    """Represents a single resource with dependencies."""

    def __init__(
        self,
        id: str,
        dependencies: List[str],
        func: Callable,
        optional: bool = False,
        optional_deps: Optional[List[str]] = None,
    ):
        self.id = id
        self.dependencies = list(dependencies or [])
        self.optional_deps = list(optional_deps or [])
        self.func = func
        self.optional = optional

    def __call__(self, ctx: SceneResourceContext) -> Optional[Path]:
        """Execute the resource function."""
        return self.func(ctx)


class ResourceRegistry:
    """Registry for managing resource definitions."""

    def __init__(self):
        self.resources: Dict[str, Resource] = {}

    def register(
        self,
        id: str,
        dependencies: List[str],
        func: Callable,
        *,
        optional: bool = False,
        optional_deps: Optional[List[str]] = None,
    ):
        """Register a resource explicitly.

        Args:
            id: Unique resource identifier.
            dependencies: Required dependencies that must run before this resource.
            func: Resource function to execute.
            optional: If True, resource failures are skipped with a warning.
            optional_deps: Optional dependencies — added to the required list only if
                they are already registered when ``update_scene_dependencies`` runs.
        """
        self.resources[id] = Resource(id, dependencies, func, optional=optional, optional_deps=optional_deps)

    def get_resource(self, id: str) -> Resource:
        """Get a resource by ID."""
        if id not in self.resources:
            raise ValueError(f"Resource '{id}' not found")
        return self.resources[id]

    def get_resource_list(self) -> List[Resource]:
        """Get all registered resources."""
        return list(self.resources.values())

    def get_execution_order(self) -> List[str]:
        """Get resources in dependency order using topological sort with cycle detection."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(resource_id: str):
            if resource_id in temp_visited:
                raise ValueError(
                    f"Circular dependency detected involving: {resource_id}"
                )
            if resource_id in visited:
                return

            temp_visited.add(resource_id)

            if resource_id not in self.resources:
                raise ValueError(f"Missing dependency: {resource_id}")

            resource = self.resources[resource_id]

            for dep in resource.dependencies:
                if dep not in self.resources:
                    raise ValueError(f"Missing dependency: {dep} for {resource_id}")
                visit(dep)

            temp_visited.remove(resource_id)
            visited.add(resource_id)
            result.append(resource_id)

        for resource_id in self.resources:
            if resource_id not in visited:
                visit(resource_id)

        return result

    def _categorize_resource(self, resource_id: str) -> str:
        """Categorize a resource by its ID."""
        if resource_id in {
            "scene_description",
            "target_dem",
            "target_landcover",
            "target_mesh",
            "target_texture",
        }:
            return "core"
        elif resource_id.startswith("buffer_"):
            return "buffer"
        elif resource_id.startswith("background_"):
            return "background"
        else:
            return "optional"

    def update_scene_dependencies(self):
        """Resolve optional deps and update scene_description dependencies."""
        # Wire optional deps
        for resource in self.resources.values():
            for opt in resource.optional_deps:
                if opt in self.resources and opt not in resource.dependencies:
                    resource.dependencies.append(opt)

        if "scene_description" not in self.resources:
            return

        # Base dependencies that are always required
        base_dependencies = ["target_mesh", "target_texture"]

        # Dynamically find optional dependencies that contribute to scene
        optional_dependencies = []
        for resource_id in self.resources:
            if resource_id == "scene_description":
                continue

            category = self._categorize_resource(resource_id)
            # Include buffer/background meshes and textures, plus optional resources
            if (
                category in ["buffer", "background"]
                and resource_id.endswith(("_mesh", "_texture"))
            ) or category == "optional":
                optional_dependencies.append(resource_id)

        # Update scene_description dependencies
        scene_resource = self.resources["scene_description"]
        scene_resource.dependencies = base_dependencies + optional_dependencies


class DAGExecutor:
    """Executes resources in dependency order."""

    def __init__(self, registry: ResourceRegistry):
        self.registry = registry

    def execute(self, context: SceneResourceContext) -> Dict[str, Any]:
        """Execute all resources in dependency order."""
        execution_order = self.registry.get_execution_order()
        results = {}

        for resource_id in execution_order:
            try:
                resource = self.registry.get_resource(resource_id)
                context.dependency_outputs = results
                result = resource(context)
                results[resource_id] = result
            except Exception as e:
                if resource.optional:
                    logging.warning(
                        "Optional resource '%s' failed (skipping): %s", resource_id, e
                    )
                    results[resource_id] = None
                else:
                    raise RuntimeError(f"Resource '{resource_id}' failed: {e}") from e

        return results
