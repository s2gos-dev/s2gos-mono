"""Scene generation resources for DAG execution."""

# Import all resource modules to register them with the ResourceRegistry
from . import assets, dem, hamster, landcover, mesh, scene, texture

__all__ = [
    "dem",
    "landcover",
    "mesh",
    "texture",
    "assets",
    "hamster",
    "scene",
]
