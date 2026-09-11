import numpy as np
import pytest
import trimesh
import xarray as xr

from s2gos_generator.core.grid import SceneGrid, aoi_to_uv
from s2gos_generator.processors.terrain_mesh import MeshGenerator

RESOLUTION_M = 30.0
DIVIDES = 900.0  # 30 cells exactly, so the raster is the AOI
OVERSHOOTS = 905.0  # 31 cells of 30 m, so the raster reaches 930 m


@pytest.fixture
def generator():
    return MeshGenerator()


@pytest.fixture
def clean_3x3_dem():
    elevation = np.ones((3, 3), dtype=float)
    return xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
    )


def _dem_covering(aoi_size_m):
    """A tilted, bumpy DEM on the raster covering *aoi_size_m*."""
    axis = SceneGrid.covering(aoi_size_m, RESOLUTION_M).nodes()
    x, y = np.meshgrid(axis, axis)
    values = 0.05 * x + 30.0 * np.sin(y / 120.0)
    return xr.DataArray(values, dims=["y", "x"], coords={"x": axis, "y": axis})


class TestCreateGridFaces:
    def test_output_shape_formula(self, generator):
        nx, ny = 3, 4
        faces = generator._create_grid_faces(nx, ny)
        expected_count = 2 * (nx - 1) * (ny - 1)
        assert faces.shape == (expected_count, 3)

    def test_all_indices_in_range(self, generator):
        nx, ny = 4, 5
        faces = generator._create_grid_faces(nx, ny)
        assert faces.max() < nx * ny

    def test_minimal_2x2_exact_faces(self, generator):
        faces = generator._create_grid_faces(2, 2)
        expected = np.array([[0, 1, 3], [0, 3, 2]])
        np.testing.assert_array_equal(faces, expected)

    def test_no_degenerate_faces(self, generator):
        faces = generator._create_grid_faces(5, 5)
        for row in faces:
            assert len(set(row)) == 3


class TestDemToMesh:
    def test_clean_3x3_dem_vertex_and_face_count(self, generator, clean_3x3_dem):
        mesh = generator.dem_to_mesh(clean_3x3_dem)
        assert len(mesh.vertices) == 9
        assert len(mesh.faces) == 8

    def test_nan_removes_affected_faces_and_vertices(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        elevation[0, 0] = np.nan
        da = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        )
        mesh = generator.dem_to_mesh(da)
        assert len(mesh.faces) == 6
        assert len(mesh.vertices) == 8

    def test_raises_on_bad_dims(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        da = xr.DataArray(elevation, dims=["row", "col"])
        with pytest.raises(ValueError):
            generator.dem_to_mesh(da)

    def test_lon_lat_dims_accepted(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        da = xr.DataArray(
            elevation,
            dims=["lat", "lon"],
            coords={"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0]},
        )
        mesh = generator.dem_to_mesh(da)
        assert isinstance(mesh, trimesh.Trimesh)


class TestFitToAoi:
    """The mesh is built over the DEM raster, which reaches past the AOI whenever the
    resolution does not divide it. ``fit_to_aoi`` trims it back and takes the UVs from
    the AOI, in that order."""

    @pytest.mark.parametrize("aoi_size_m", [DIVIDES, OVERSHOOTS])
    def test_spans_the_aoi_exactly(self, generator, aoi_size_m):
        mesh = generator.fit_to_aoi(
            generator.dem_to_mesh(_dem_covering(aoi_size_m)), aoi_size_m
        )

        half = aoi_size_m / 2.0
        for column in (0, 1):
            assert mesh.vertices[:, column].min() == pytest.approx(-half)
            assert mesh.vertices[:, column].max() == pytest.approx(half)

    @pytest.mark.parametrize("aoi_size_m", [DIVIDES, OVERSHOOTS])
    def test_every_uv_is_its_vertex_with_nothing_clamped(self, generator, aoi_size_m):
        """Mapping an untrimmed mesh would push the outer vertices past ``[0, 1]``, and
        clamping them would smear the border texel around the whole edge. Comparing
        against the unclamped mapping proves the trim ran first."""
        mesh = generator.fit_to_aoi(
            generator.dem_to_mesh(_dem_covering(aoi_size_m)), aoi_size_m
        )

        np.testing.assert_allclose(
            mesh.visual.uv, aoi_to_uv(mesh.vertices[:, :2], aoi_size_m)
        )
