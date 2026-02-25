from s2gos_apps.registry import registry

# from . import frascati, gobabeb, kairouan, pisa, pnp
# from .common import generation, simulation
# from .upscaling import upscaling
from .mtr_demo import mtr_demo_generation, mtr_demo_simulation

__all__ = [
    "registry",
    # "frascati",
    # "gobabeb",
    # "kairouan",
    # "pisa",
    # "pnp",
    # "upscaling",
    # "generation",
    # "simulation",
    "mtr_demo_generation",
    "mtr_demo_simulation",
]
