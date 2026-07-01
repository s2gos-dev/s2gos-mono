"""Standalone per-batch worker for the batched BOA-irradiance grid.

Invoked as a fresh subprocess::

    python -m s2gos_simulator._grid_worker <task.pkl> <result.pkl>

It runs exactly one batch of grid patches and exits, so the several GB of native
Mitsuba/Dr.Jit memory that Eradiate's expert mesh/bitmap interface allocates (and never
returns in-process) is reclaimed by the OS on exit. Running as its own module means the
child's ``__main__`` is *this* file — the caller's script is never re-imported.

Task pickle: ``(simulation_config, scene_description, scene_dir_str, grid_config,
chunk_centers, disk_ids)``. Result pickle: ``(E_chunk, chunk_w)`` (plain numpy; the
per-cell horizontal BOA irradiance ``E = π·L`` and the wavelength axis in nm or ``None``).
"""

import pickle
import sys

from upath import UPath


def main(task_path: str, result_path: str) -> None:
    with open(task_path, "rb") as fh:
        (
            simulation_config,
            scene_description,
            scene_dir_str,
            grid_config,
            chunk_centers,
            disk_ids,
        ) = pickle.load(fh)

    # Import inside the child so the parent stays lean and free of Eradiate state.
    from .backends.eradiate.backend import EradiateBackend
    from .irradiance_grid_processor import run_single_batch

    # A fresh backend sets the Eradiate mode from simulation_config.backend_hints.
    backend = EradiateBackend(simulation_config)
    E_chunk, chunk_w = run_single_batch(
        backend,
        scene_description,
        UPath(scene_dir_str),
        grid_config,
        chunk_centers,
        disk_ids,
    )

    with open(result_path, "wb") as fh:
        pickle.dump((E_chunk, chunk_w), fh)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m s2gos_simulator._grid_worker <task.pkl> <result.pkl>"
        )
    main(sys.argv[1], sys.argv[2])
