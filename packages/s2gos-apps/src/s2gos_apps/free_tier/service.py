from wraptile.services.local import LocalService

from .registry import registry

service = LocalService(
    title="MIRROR Free Tier Service",
    description="S2GOS free-tier process server: browse precalculated results",
    process_registry=registry,
)
