import logging

import numpy as np
import trimesh
import xarray as xr
from s2gos_utils.io.paths import expand_mapper
from trimesh.intersections import slice_mesh_plane
from upath import UPath

from ...core.grid import aoi_to_uv


class MeshGenerator:
    """Converts DEM data to 3D meshes"""

    def __init__(self):
        """Initialize the mesh generator."""

    def dem_to_mesh(
        self,
        dem_data: xr.DataArray,
        handle_nans: bool = True,
    ) -> trimesh.Trimesh:
        """Build a terrain mesh with one vertex per DEM sample.

        Args:
            dem_data: DEM elevation DataArray, on the DEM raster.
            handle_nans: Whether to drop faces with NaN elevations.
        """
        from .builder import extract_dem

        x_coords, y_coords, elevation = extract_dem(dem_data)

        nx, ny = len(x_coords), len(y_coords)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        vertices = np.column_stack(
            [x_grid.ravel(), y_grid.ravel(), np.asarray(elevation, float).ravel()]
        )

        faces = self._create_grid_faces(nx, ny)

        if handle_nans:
            valid_vertex_mask = ~np.isnan(vertices[:, 2])
            valid_face_mask = np.all(valid_vertex_mask[faces], axis=1)
            faces = faces[valid_face_mask]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh.remove_unreferenced_vertices()

        return mesh

    def _create_grid_faces(self, nx: int, ny: int) -> np.ndarray:
        """
        Creates triangular faces for a regular grid.

        Args:
            nx: Number of points in x direction.
            ny: Number of points in y direction.

        Returns:
            Array of face indices with shape (num_faces, 3).
        """
        i = np.arange(nx * ny).reshape(ny, nx)

        quad_indices = i[:-1, :-1].ravel()

        faces1 = np.vstack([quad_indices, quad_indices + 1, quad_indices + nx + 1]).T
        faces2 = np.vstack([quad_indices, quad_indices + nx + 1, quad_indices + nx]).T

        return np.vstack([faces1, faces2])

    def _clip_to_aoi(self, mesh: trimesh.Trimesh, aoi_size_m: float) -> trimesh.Trimesh:
        """Slice the mesh against the four sides of the AOI square.

        Leaves vertices exactly on the boundary with elevations interpolated along the
        cut. A mesh that already ends on the AOI is returned untouched.
        """
        half = aoi_size_m / 2.0
        sides = (
            ((1.0, 0.0, 0.0), (-half, 0.0, 0.0)),
            ((-1.0, 0.0, 0.0), (half, 0.0, 0.0)),
            ((0.0, 1.0, 0.0), (0.0, -half, 0.0)),
            ((0.0, -1.0, 0.0), (0.0, half, 0.0)),
        )
        for normal, origin in sides:
            mesh = slice_mesh_plane(
                mesh,
                plane_normal=np.array(normal, dtype=float),
                plane_origin=np.array(origin, dtype=float),
                cap=False,
            )
        return mesh

    def fit_to_aoi(self, mesh: trimesh.Trimesh, aoi_size_m: float) -> trimesh.Trimesh:
        """Trim the mesh to the AOI and map it onto ``[0, 1]`` UV.

        The mesh is built over the DEM raster, which reaches past the AOI whenever the
        resolution does not divide it, so it is trimmed before the UVs are taken.

        Args:
            mesh: Terrain mesh in scene coordinates (metres).
            aoi_size_m: Side length of the area the mesh must end up spanning.

        Returns:
            The clipped mesh, carrying UV coordinates.
        """
        mesh = self._clip_to_aoi(mesh, aoi_size_m)

        uv = aoi_to_uv(mesh.vertices[:, :2], aoi_size_m)
        mesh.visual.uv = np.clip(uv, 0.0, 1.0)
        return mesh

    def save_mesh(
        self, mesh: trimesh.Trimesh, output_path: UPath, format: str = "ply"
    ) -> None:
        """
        Saves a mesh to file.

        Args:
            mesh: The mesh to save.
            output_path: UPath where the mesh will be saved.
            format: File format (e.g., 'ply', 'obj', 'stl').
        """

        from s2gos_utils.io.paths import mkdir

        mkdir(output_path.parent)

        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{format}")

        file_type = output_path.suffix.lstrip(".")
        data = mesh.export(file_type=file_type)
        with output_path.open("wb") as f:
            f.write(data)
        logging.info(f"Mesh saved to {output_path}")

    def adaptive_dem_to_mesh(
        self,
        dem_data: xr.DataArray,
        operations,
        refinement_config,
        handle_nans: bool = True,
    ) -> trimesh.Trimesh:
        """Build an adaptive quadtree mesh with terraforming operations.

        Args:
            dem_data: DEM elevation DataArray, on the DEM raster.
            operations: ``list[TerraformOperation]`` — one per way segment,
                or ``None`` for a uniform mesh.
            refinement_config: MeshRefinementConfig instance.
            handle_nans: Whether to remove NaN-containing faces.

        Returns:
            Adaptive Trimesh object.
        """
        from .builder import build_refined_mesh

        return build_refined_mesh(dem_data, operations, refinement_config, handle_nans)

    def generate_mesh_from_dem_file(
        self,
        dem_file_path: UPath,
        output_path: UPath,
        aoi_size_m: float,
        handle_nans: bool = True,
    ) -> trimesh.Trimesh:
        """Load a DEM, build a mesh over the AOI, and save it.

        Args:
            dem_file_path: UPath to the DEM zarr.
            output_path: UPath where the mesh will be saved.
            aoi_size_m: Side length of the area the mesh must span.
            handle_nans: Whether to drop faces with NaN elevations.

        Returns:
            The generated mesh.
        """
        dem_dataset = xr.open_zarr(expand_mapper(dem_file_path))
        dem_data = dem_dataset["elevation"]

        mesh = self.fit_to_aoi(
            self.dem_to_mesh(dem_data, handle_nans=handle_nans), aoi_size_m
        )
        self.save_mesh(mesh, output_path)

        return mesh

    def get_mesh_info(self, mesh: trimesh.Trimesh) -> dict:
        """
        Returns summary information about a mesh.

        Args:
            mesh: The mesh to analyze.

        Returns:
            Dictionary containing mesh statistics.
        """
        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "extents": mesh.extents.tolist(),
            "center": mesh.center_mass.tolist(),
            "volume": mesh.volume,
            "surface_area": mesh.area,
            "is_watertight": mesh.is_watertight,
            "has_uvs": hasattr(mesh.visual, "uv") and mesh.visual.uv is not None,
        }
