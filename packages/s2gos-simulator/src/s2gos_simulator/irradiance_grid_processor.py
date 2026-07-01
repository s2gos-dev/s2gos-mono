"""Batched BOA irradiance over a grid (white-patch + MultiRadiancemeter).

Computes horizontal downwelling bottom-of-atmosphere irradiance on a regular grid:

1. Place a small white Lambertian patch (ρ=1) at every grid cell, ``height_offset_m``
   above the DEM.
2. Sample patches with one ``mradiancemeter`` measure — each downward ray reads its
   patch radiance ``L``.
3. Convert per-cell: ``E = π·L`` (Lambertian ρ=1) → horizontal BOA irradiance.

The grid is covered in ``ceil(N / max_patches_per_batch)`` experiments. Eradiate's
expert mesh/bitmap (``kdict``/``kpmap``) interface leaks several GB of *native*
(Mitsuba/Dr.Jit) memory per experiment that no in-process cleanup (``del`` + ``gc`` +
``drjit.flush_malloc_cache``/``flush_kernel_cache``) reclaims — only a process exit
frees it. So each batch runs in a **fresh subprocess** (:mod:`._grid_worker`); when the
child exits the OS reclaims all its memory and peak stays at ~one batch. Set
``IrradianceGridConfig.isolate_batches=False`` to run in-process (leaks; debugging only).

This is the scalable counterpart of the single-point :mod:`irradiance_processor`.
"""

import logging
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from s2gos_utils.scene import SceneDescription
from upath import UPath

from .irradiance_processor import insert_reference_disks

logger = logging.getLogger(__name__)


def run_single_batch(
    backend,
    scene_description: SceneDescription,
    scene_dir: UPath,
    config,
    chunk_centers: np.ndarray,
    disk_ids,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute one batch of grid patches: white patches + one ``mradiancemeter`` run.

    Builds this chunk's white patches on the scene, samples them with a downward
    ``mradiancemeter``, and returns ``(E_chunk, chunk_w)`` where ``E = π·L`` (detached
    numpy) and ``chunk_w`` is the wavelength axis (nm) or ``None``.

    Shared by the in-process debug path and the per-batch subprocess worker
    (:mod:`._grid_worker`); it holds no cross-batch state, so calling it in a fresh
    process fully bounds memory.
    """
    import eradiate

    chunk_centers = np.asarray(chunk_centers, dtype=float)
    n_chunk = len(chunk_centers)
    disk_scene = insert_reference_disks(
        scene_description, chunk_centers, disk_ids, radius=config.white_patch_radius_m
    )
    origins = chunk_centers + np.array([0.0, 0.0, config.ray_offset_m])
    directions = np.tile([0.0, 0.0, -1.0], (n_chunk, 1))
    measure = backend.eradiate_translator.create_multi_radiancemeter_measure(
        config, origins.tolist(), directions.tolist()
    )
    experiment = backend.create_experiment(disk_scene, scene_dir, measures=[measure])
    measure_id = getattr(experiment.measures[0], "id", config.id)
    eradiate.run(experiment, measures=0)

    result = experiment.results[measure_id]
    if "radiance" not in result:
        raise RuntimeError(
            f"No 'radiance' in results. Available: {list(result.data_vars)}"
        )
    return IrradianceGridProcessor._extract_ray_radiance(result["radiance"], n_chunk)


class IrradianceGridProcessor:
    """Processor for batched grid BOA irradiance using the white-patch technique."""

    def __init__(self, backend):
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def requires_irradiance_grid(self) -> bool:
        """Check if any irradiance-grid measurements are configured."""
        from .config import IrradianceGridConfig

        return any(
            isinstance(m, IrradianceGridConfig)
            for m in self.simulation_config.measurements
        )

    def _patch_elevations(
        self, config, scene_description: SceneDescription, scene_dir: UPath, xs, ys
    ) -> np.ndarray:
        """Absolute patch z for every grid cell (surface + offset, or absolute).

        With ``account_for_buildings`` the surface is the true top of the rendered geometry
        (terrain *or* building roof) from a downward ray-cast, so patches never land inside
        buildings. Otherwise the surface is the bare DEM.
        """
        if not config.terrain_relative_height:
            return np.full_like(xs, float(config.height_offset_m))

        from .terrain_query import TerrainQuery

        terrain = np.asarray(
            TerrainQuery(
                scene_description, scene_dir
            ).query_elevation_at_scene_coords_batch(xs, ys, raise_on_error=False)
        )

        if not config.account_for_buildings:
            return terrain + float(config.height_offset_m)

        # Surface from the rendered geometry (terrain + buildings/objects).
        surface = self._query_surface_subprocess(scene_description, scene_dir, xs, ys)
        # Rays that hit no geometry fall back to the bare DEM.
        missed = ~np.isfinite(surface)
        if missed.any():
            surface = surface.copy()
            surface[missed] = terrain[missed]

        n_roof = int(np.count_nonzero(surface > terrain + 0.5))
        logger.info(
            "[%s] building-aware surface: %d/%d cells lifted to a roof/object "
            "(max +%.1f m above DEM)",
            config.id,
            n_roof,
            surface.size,
            float(np.nanmax(surface - terrain)) if surface.size else 0.0,
        )
        return surface + float(config.height_offset_m)

    def _query_surface_subprocess(
        self, scene_description, scene_dir, xs, ys
    ) -> np.ndarray:
        """Surface z at each ``(x, y)`` via a fresh ``_surface_worker`` subprocess.

        Casts downward rays against the scene geometry in an isolated process (so the mesh
        scene's native memory is reclaimed on exit, and the parent never imports Eradiate).
        Returns one z per cell, ``NaN`` where a ray missed all geometry.
        """
        from ._surface_worker import DEFAULT_Z_TOP

        with tempfile.TemporaryDirectory(prefix="s2gos_surface_") as tmp:
            task_path = Path(tmp) / "task.pkl"
            result_path = Path(tmp) / "result.pkl"
            task = (
                self.simulation_config,
                scene_description,
                str(scene_dir),
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                float(DEFAULT_Z_TOP),
            )
            with open(task_path, "wb") as fh:
                pickle.dump(task, fh)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "s2gos_simulator._surface_worker",
                    str(task_path),
                    str(result_path),
                ],
                check=True,
            )

            with open(result_path, "rb") as fh:
                return pickle.load(fh)

    def execute_irradiance_grid_measurements(
        self,
        scene_description: SceneDescription,
        scene_dir: UPath,
        output_dir: UPath,
    ) -> Dict[str, xr.Dataset]:
        """Execute all configured irradiance-grid measurements.

        The grid is covered in ``config.n_batches`` experiments of at most
        ``max_patches_per_batch`` white patches each (one experiment can only hold so
        many patches); per-cell results are stitched back into the full map.

        Returns a mapping of measurement id → xarray Dataset with a 2-D
        ``boa_irradiance`` map (dims ``(y, x[, w])``) and, for a spectral run, a
        broadband-integrated ``boa_broadband`` map.
        """
        from s2gos_utils.io.paths import expand_mapper, mkdir

        from .config import IrradianceGridConfig

        logger.info("=" * 60)
        logger.info("BOA Irradiance Grid Measurements")
        logger.info("=" * 60)
        mkdir(output_dir)

        grid_configs = [
            m
            for m in self.simulation_config.measurements
            if isinstance(m, IrradianceGridConfig)
        ]

        results: Dict[str, xr.Dataset] = {}
        for config in grid_configs:
            result_ds = self._run_grid(config, scene_description, scene_dir)
            output_file = output_dir / f"{config.id}.zarr"
            result_ds.to_zarr(expand_mapper(output_file), mode="w")
            logger.info("  ✓ Saved %s", output_file.name)
            results[config.id] = result_ds

        logger.info("Complete: %d grid measurement(s)", len(results))
        return results

    def _run_grid(
        self, config, scene_description: SceneDescription, scene_dir: UPath
    ) -> xr.Dataset:
        """Run one irradiance grid across ``config.n_batches`` chunked experiments.

        Each batch is executed by :func:`run_single_batch` (chunk's white patches + one
        ``mradiancemeter`` run). When ``config.isolate_batches`` is True (default) each
        batch runs in a fresh subprocess (:mod:`._grid_worker`) so the several GB of
        native Mitsuba/Dr.Jit memory it allocates is reclaimed by the OS on process exit
        and peak memory stays at ~one batch; otherwise batches run in-process (leaks).

        This method itself stays lean (no ``import eradiate``): it only builds geometry,
        dispatches batches, and stitches per-cell results into the full map.
        """
        ny, nx = config.grid_shape
        x_axis, y_axis = config.grid_axes()
        xs, ys = config.grid_points()  # row-major (y outer, x inner), length N
        n = xs.size
        zs = self._patch_elevations(config, scene_description, scene_dir, xs, ys)
        centers = np.column_stack([xs, ys, zs])

        batch = config.max_patches_per_batch
        n_batches = config.n_batches
        logger.info(
            "[%s] %dx%d = %d cells, %.1fm spacing, %.2fm above DEM, patch r=%.3fm; "
            "%d batch(es) of <=%d patches (%s)",
            config.id,
            nx,
            ny,
            n,
            config.resolution_m,
            config.height_offset_m,
            config.white_patch_radius_m,
            n_batches,
            batch,
            "subprocess-isolated" if config.isolate_batches else "in-process",
        )

        E_flat = None  # allocated once K (spectral bins) is known
        wavelengths = None
        # In-process backend is only built (and reused) for the non-isolated debug path.
        inline_backend = None if config.isolate_batches else self.backend
        for b, start in enumerate(range(0, n, batch)):
            stop = min(start + batch, n)
            n_chunk = stop - start
            chunk_centers = centers[start:stop]
            disk_ids = [f"{config.id}_p{i}" for i in range(start, stop)]
            logger.info(
                "  batch %d/%d: cells %d..%d (%d rays)",
                b + 1,
                n_batches,
                start,
                stop - 1,
                n_chunk,
            )

            if config.isolate_batches:
                E_chunk, chunk_w = self._run_batch_subprocess(
                    scene_description, scene_dir, config, chunk_centers, disk_ids
                )
            else:
                E_chunk, chunk_w = run_single_batch(
                    inline_backend,
                    scene_description,
                    scene_dir,
                    config,
                    chunk_centers,
                    disk_ids,
                )

            if E_flat is None:
                wavelengths = chunk_w
                shape = (n,) if chunk_w is None else (n, chunk_w.size)
                E_flat = np.empty(shape, dtype=float)
            E_flat[start:stop] = E_chunk

        return self._assemble_dataset(
            E_flat, config, x_axis, y_axis, wavelengths, zs.reshape(ny, nx)
        )

    def _run_batch_subprocess(
        self, scene_description, scene_dir, config, chunk_centers, disk_ids
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run one batch in a fresh ``python -m s2gos_simulator._grid_worker`` subprocess.

        The task/result are exchanged as temp-file pickles. Running the worker as its own
        module means the child's ``__main__`` is the worker (never the caller's script),
        so top-level user code is not re-executed; and the child's native memory is fully
        reclaimed by the OS when it exits.
        """
        with tempfile.TemporaryDirectory(prefix="s2gos_grid_") as tmp:
            task_path = Path(tmp) / "task.pkl"
            result_path = Path(tmp) / "result.pkl"
            task = (
                self.simulation_config,
                scene_description,
                str(scene_dir),
                config,
                np.asarray(chunk_centers, dtype=float),
                list(disk_ids),
            )
            with open(task_path, "wb") as fh:
                pickle.dump(task, fh)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "s2gos_simulator._grid_worker",
                    str(task_path),
                    str(result_path),
                ],
                check=True,
            )

            with open(result_path, "rb") as fh:
                return pickle.load(fh)

    @staticmethod
    def _extract_ray_radiance(radiance: xr.DataArray, n_chunk: int):
        """Per-ray radiance from an ``mradiancemeter`` result, as ``E`` (E = π·L).

        The film is ``(n_chunk, 1)`` so every non-``w`` dim (``y_index``, ``x_index``,
        ``saa``, ``sza`` …) multiplies to ``n_chunk``. Collapsing them in their native
        order preserves the origins order and is robust for ``n_chunk == 1``.

        Returns ``(E, wavelengths)`` where ``E`` has shape ``(n_chunk,)`` or
        ``(n_chunk, K)`` and ``wavelengths`` is the ``w`` coordinate (nm) or None.
        """
        has_w = "w" in radiance.dims
        non_w = [d for d in radiance.dims if d != "w"]
        n_total = int(np.prod([radiance.sizes[d] for d in non_w])) if non_w else 1
        if n_total != n_chunk:
            raise RuntimeError(
                f"Expected {n_chunk} rays but radiance has {dict(radiance.sizes)}"
            )

        ordered = radiance.transpose(*non_w, *(["w"] if has_w else []))
        E = np.pi * np.asarray(ordered.values)
        if has_w:
            return E.reshape(n_chunk, radiance.sizes["w"]), np.asarray(
                radiance["w"].values
            )
        return E.reshape(n_chunk), None

    def _assemble_dataset(
        self, E_flat, config, x_axis, y_axis, wavelengths, patch_z
    ) -> xr.Dataset:
        """Build the irradiance map(s) from the stitched per-cell ``E`` array."""
        ny, nx = config.grid_shape
        coords = {"x": ("x", np.asarray(x_axis)), "y": ("y", np.asarray(y_axis))}

        has_w = wavelengths is not None
        if has_w:
            grid = E_flat.reshape(ny, nx, wavelengths.size)
            dims = ("y", "x", "w")
            coords["w"] = ("w", np.asarray(wavelengths))
        else:
            grid = E_flat.reshape(ny, nx)
            dims = ("y", "x")

        boa = xr.DataArray(grid, dims=dims, coords=coords)
        boa.attrs.update(
            {
                "quantity": "boa_irradiance",
                "units": "W m^-2 nm^-1",
                "conversion": "E = π·L (horizontal white Lambertian patch, ρ=1)",
                "height_offset_m": config.height_offset_m,
                "terrain_relative_height": int(config.terrain_relative_height),
                "white_patch_radius_m": config.white_patch_radius_m,
                "method": "white-patch + mradiancemeter (batched grid)",
            }
        )
        data_vars = {"boa_irradiance": boa}

        # Broadband (band-integrated) irradiance, W/m², when spectrally resolved.
        if has_w and wavelengths.size >= 2:
            broadband = np.trapz(grid, np.asarray(wavelengths), axis=-1)
            bb = xr.DataArray(
                broadband, dims=("y", "x"), coords={"x": coords["x"], "y": coords["y"]}
            )
            bb.attrs.update(
                {
                    "quantity": "boa_broadband_irradiance",
                    "units": "W m^-2",
                    "integration": f"trapz over {wavelengths.min():.0f}-{wavelengths.max():.0f} nm",
                }
            )
            data_vars["boa_broadband"] = bb

        patch_elev = xr.DataArray(
            np.asarray(patch_z),
            dims=("y", "x"),
            coords={"x": coords["x"], "y": coords["y"]},
        )
        data_vars["patch_elevation_m"] = patch_elev
        return xr.Dataset(data_vars)
