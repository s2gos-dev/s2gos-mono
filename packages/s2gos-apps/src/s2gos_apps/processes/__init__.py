from s2gos_apps.free import free_tier
from s2gos_apps.free import registry as free_registry
from s2gos_apps.registry import registry

from .common import generation, simulation
from .frascati import generation_configs as frascati_generation
from .frascati import simulation_configs as frascati_simulation
from .gobabeb import generation_configs as gobabeb_generation
from .gobabeb import simulation_configs as gobabeb_simulation
from .kairouan import generation_configs as kairouan_generation
from .kairouan import simulation_configs as kairouan_simulation
from .pisa import generation_configs as pisa_generation
from .pisa import simulation_configs as pisa_simulation
from .pnp import generation_configs as pnp_generation
from .pnp import simulation_configs as pnp_simulation
from .upscaling import upscaling

# The free tier owns its registry; serve its processes from the main one as well.
registry.workflows().update(free_registry.workflows())

__all__ = [
    "registry",
    "generation",
    "simulation",
    "frascati_generation",
    "frascati_simulation",
    "free_tier",
    "gobabeb_generation",
    "gobabeb_simulation",
    "kairouan_generation",
    "kairouan_simulation",
    "pisa_generation",
    "pisa_simulation",
    "pnp_generation",
    "pnp_simulation",
    "upscaling",
]
