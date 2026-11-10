import logging

import numpy as np
import trimesh
import xarray as xr
from upath import UPath


class MeshGenerator:
    """Converts DEM data to 3D meshes"""

    def __init__(self):
        """Initialize the mesh generator."""

    def dem_to_mesh(
        self, dem_data: xr.DataArray, handle_nans: bool = True
    ) -> trimesh.Trimesh:
        """Convert a DEM DataArray to a Trimesh object."""
        from .builder import extract_dem, grid_faces

        x_coords, y_coords, elevation = extract_dem(dem_data)
        nx, ny = len(x_coords), len(y_coords)

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        vertices = np.vstack([x_grid.ravel(), y_grid.ravel(), elevation.ravel()]).T

        faces = grid_faces(nx, ny)

        if handle_nans:
            valid_vertex_mask = ~np.isnan(vertices[:, 2])
            valid_face_mask = np.all(valid_vertex_mask[faces], axis=1)
            faces = faces[valid_face_mask]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh.remove_unreferenced_vertices()

        return mesh

    def add_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Adds planar UV coordinates to a mesh based on its bounding box.

        Args:
            mesh: The input mesh.

        Returns:
            The mesh with UV coordinates added.
        """

        bounds = mesh.bounds
        extent = mesh.extents.copy()

        if extent[0] == 0 or extent[1] == 0:
            raise ValueError(
                "Cannot calculate UV coordinates: Mesh extent on X or Y axis is zero. Check your input DEM data."
            )

        uv_coords = (mesh.vertices[:, :2] - bounds[0, :2]) / extent[:2]

        uv_coords = np.clip(uv_coords, 0.0, 1.0)

        mesh.visual.uv = uv_coords

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
            dem_data: DEM elevation DataArray.
            operations: ``list[TerraformOperation]`` — one per way segment,
                or ``None`` for a uniform mesh.
            refinement_config: MeshRefinementConfig instance.
            handle_nans: Whether to remove NaN-containing faces.

        Returns:
            Adaptive Trimesh object.
        """
        from .builder import build_refined_mesh

        return build_refined_mesh(dem_data, operations, refinement_config, handle_nans)

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
