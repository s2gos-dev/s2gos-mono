from wraptile.services.local import LocalService

from .processes.mtr_demo import registry

service = LocalService(
    title="S2GOS Demo-Server",
    description="Local DTE-S2GOS process server for demonstration",
    process_registry=registry,
)
