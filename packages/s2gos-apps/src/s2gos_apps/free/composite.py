"""One wraptile service that serves the free tier next to another backend.

wraptile binds exactly one service per deployment, and the OGC job routes
(``/jobs/{jobId}``) carry no process id, so the free tier cannot be split off at the
gateway. This service holds both backends and dispatches each request itself:

- processes by id: free-tier ids go to ``free``, everything else to ``main``;
- jobs by id: free-tier job ids are published with the ``free.`` prefix, which is
  stripped again before the free backend sees them.
"""

from typing import Optional

from gavicore.models import (
    JobInfo,
    JobList,
    JobResults,
    ProcessDescription,
    ProcessList,
    ProcessRequest,
)
from wraptile.exceptions import ServiceConfigException
from wraptile.services.base import ServiceBase
from wraptile.services.local import LocalService

#: Prefix of every free-tier job id. Jobs of ``main`` must never start with it; for
#: Airflow they are ``{dag_id}__{timestamp}_{n}``, so no DAG id may start with "free.".
FREE_JOB_PREFIX = "free."


class CompositeService(ServiceBase):
    def __init__(
        self,
        title: str,
        description: Optional[str] = None,
        *,
        free: LocalService,
        main: ServiceBase,
    ):
        super().__init__(title=title, description=description)
        self.free = free
        self.main = main

    def configure(
        self,
        processes: Optional[bool] = None,
        max_workers: Optional[int] = None,
        **main_kwargs,
    ):
        """Split the service options: ``--processes`` / ``--max-workers`` configure the
        free backend, everything else (e.g. ``--airflow-*``) goes to ``main``."""
        if processes:
            # LocalService's process pool re-imports the service by reference and
            # expects a LocalService there, not this composite.
            raise ServiceConfigException(
                "The free tier runs on threads only; drop --processes."
            )
        self.free.configure(processes=False, max_workers=max_workers)
        self.main.configure(**main_kwargs)

    # --- processes ---

    def _is_free(self, process_id: str) -> bool:
        return process_id in self.free.process_registry

    async def get_processes(self, *args, **kwargs) -> ProcessList:
        free = await self.free.get_processes(*args, **kwargs)
        main = await self.main.get_processes(*args, **kwargs)
        # The free tier wins a clash, e.g. a free-tier DAG generated from the main
        # registry, so each id is listed, and served, once.
        return ProcessList(
            processes=free.processes
            + [p for p in main.processes if not self._is_free(p.id)],
            links=free.links,
        )

    async def get_process(self, process_id: str, *args, **kwargs) -> ProcessDescription:
        backend = self.free if self._is_free(process_id) else self.main
        return await backend.get_process(process_id, *args, **kwargs)

    async def execute_process(
        self, process_id: str, process_request: ProcessRequest, *args, **kwargs
    ) -> JobInfo:
        if self._is_free(process_id):
            job_info = await self.free.execute_process(
                process_id, process_request, *args, **kwargs
            )
            return _publish(job_info)
        return await self.main.execute_process(
            process_id, process_request, *args, **kwargs
        )

    # --- jobs ---

    def _route(self, job_id: str) -> tuple[ServiceBase, str]:
        if job_id.startswith(FREE_JOB_PREFIX):
            return self.free, job_id[len(FREE_JOB_PREFIX) :]
        return self.main, job_id

    async def get_jobs(self, *args, **kwargs) -> JobList:
        free = await self.free.get_jobs(*args, **kwargs)
        main = await self.main.get_jobs(*args, **kwargs)
        return JobList(
            jobs=[_publish(j) for j in free.jobs] + main.jobs,
            links=free.links,
        )

    async def get_job(self, job_id: str, *args, **kwargs) -> JobInfo:
        backend, backend_id = self._route(job_id)
        job_info = await backend.get_job(backend_id, *args, **kwargs)
        return _publish(job_info) if backend is self.free else job_info

    async def dismiss_job(self, job_id: str, *args, **kwargs) -> JobInfo:
        backend, backend_id = self._route(job_id)
        job_info = await backend.dismiss_job(backend_id, *args, **kwargs)
        return _publish(job_info) if backend is self.free else job_info

    async def get_job_results(self, job_id: str, *args, **kwargs) -> JobResults:
        backend, backend_id = self._route(job_id)
        return await backend.get_job_results(backend_id, *args, **kwargs)


def _publish(job_info: JobInfo) -> JobInfo:
    """A free-tier job info under its public id. A copy: the free backend keeps using
    the object it owns."""
    return job_info.model_copy(update={"jobID": FREE_JOB_PREFIX + job_info.jobID})
