"""Standalone surface-elevation worker for the building-aware BOA-irradiance grid.

Invoked as a fresh subprocess::

    python -m s2gos_simulator._surface_worker <task.pkl> <result.pkl>

It builds a geometry-only Eradiate experiment (terrain + buildings/objects, no atmosphere)
and casts a downward ray at each grid ``(x, y)`` to find the **true top surface** z (ground
where open, roof where built). Running as its own module means the child's ``__main__`` is
this file (the caller's script is never re-imported); the several GB of native Mitsuba/Dr.Jit
memory the mesh scene allocates is reclaimed by the OS on exit (same rationale as
:mod:`._grid_worker`).

Task pickle: ``(simulation_config, scene_description, scene_dir_str, xs, ys, z_top)``.
Result pickle: ``surface_z`` — a numpy array (one z per (x, y)), ``NaN`` where the downward
ray hit no geometry.
"""

import pickle
import sys

import numpy as np
from upath import UPath

# z (scene meters) to start the downward probe ray from — safely above any terrain/building.
DEFAULT_Z_TOP = 10000.0


def raycast_surface_z(mi_scene, xs, ys, z_top: float = DEFAULT_Z_TOP) -> np.ndarray:
    """Downward-ray the surface z at each ``(x, y)`` against a Mitsuba scene.

    Casts a ray from ``(x, y, z_top)`` straight down and returns the first-hit z, or ``NaN``
    where the ray misses all geometry. Eradiate's ``mono``/``ckd`` variants are scalar, so
    this loops per point (measured ~365k rays/s — 250k cells in well under a second).
    """
    import mitsuba as mi

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.full(xs.shape, np.nan, dtype=float)
    down = mi.Vector3f(0.0, 0.0, -1.0)
    for i in range(xs.size):
        si = mi_scene.ray_intersect(
            mi.Ray3f(mi.Point3f(float(xs[i]), float(ys[i]), z_top), down)
        )
        if si.is_valid():
            out[i] = float(si.p.z)
    return out


def _build_geometry_scene(simulation_config, scene_description, scene_dir):
    """Build a geometry-only Mitsuba scene (terrain + objects, no atmosphere) and return it."""
    # Geometry is mode-independent; force mono — the simplest, validated path.
    mono_config = simulation_config.model_copy(
        update={"backend_hints": {"eradiate": {"mode": "mono"}}}
    )
    from .backends.eradiate.backend import EradiateBackend

    backend = EradiateBackend(mono_config)
    # A dummy 1-ray measure keeps the experiment valid; we ignore it and raycast ourselves.
    dummy_measure = {
        "type": "mradiancemeter",
        "id": "surface_probe",
        "origins": [[0.0, 0.0, DEFAULT_Z_TOP]],
        "directions": [[0.0, 0.0, -1.0]],
        "srf": {"type": "uniform", "wmin": 540.0, "wmax": 560.0},
        "spp": 1,
    }
    experiment = backend.create_experiment(
        scene_description, scene_dir, atmosphere=None, measures=[dummy_measure]
    )
    experiment.init()
    return experiment.mi_scene.obj


def main(task_path: str, result_path: str) -> None:
    with open(task_path, "rb") as fh:
        (
            simulation_config,
            scene_description,
            scene_dir_str,
            xs,
            ys,
            z_top,
        ) = pickle.load(fh)

    mi_scene = _build_geometry_scene(
        simulation_config, scene_description, UPath(scene_dir_str)
    )
    surface_z = raycast_surface_z(mi_scene, xs, ys, z_top)

    with open(result_path, "wb") as fh:
        pickle.dump(surface_z, fh)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m s2gos_simulator._surface_worker <task.pkl> <result.pkl>"
        )
    main(sys.argv[1], sys.argv[2])
