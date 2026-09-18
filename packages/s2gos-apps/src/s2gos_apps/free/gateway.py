"""The deployed S2GOS service: the free tier in-process, everything else on Airflow.

Run as ``wraptile run -- s2gos_apps.free.gateway:service --airflow-base-url=...
--airflow-username=... --airflow-password=... --max-workers=3``.
"""

from wraptile.services.airflow import AirflowService
from wraptile.services.local import LocalService

from .composite import CompositeService
from .registry import registry

service = CompositeService(
    title="S2GOS Processing Service",
    description=(
        "OGC API - Processes gateway for S2GOS: free-tier processes run in-process, "
        "all others on Airflow."
    ),
    free=LocalService(title="S2GOS Free Tier", process_registry=registry),
    main=AirflowService(title="Airflow Service"),
)
