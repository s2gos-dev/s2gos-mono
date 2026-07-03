"""Tests for the adaptive-quadtree terrain decimation (processors/terrain_mesh/adaptive_grid.py)
and the plane-residual error pyramid that drives it (processors/terrain_mesh/error_pyramid.py)."""

from collections import defaultdict

import numpy as np
import pytest

from s2gos_generator.processors.terrain_mesh import AdaptiveGrid, DemErrorPyramid


def _grid(nx, ny, max_depth):
    return AdaptiveGrid(np.arange(nx + 1.0), np.arange(ny + 1.0), max_depth)


def _const(val):
    """A ``refine`` predicate that returns the same boolean for every cell.

    Matches the predicate signature ``(xmin, ymin, xmax, ymax) -> bool[N]`` that
    ``AdaptiveGrid.refine`` calls per level; ``_const(True)`` refines everywhere.
    """
    return lambda xmin, ymin, xmax, ymax: np.full(len(xmin), val, dtype=bool)


def _flat_z(xy):
    return np.zeros(len(xy))


def _leaf_levels(grid):
    """The set of quadtree depth levels present among the grid's leaves."""
    return {int(leaf) >> 60 for leaf in grid._leaves}


def _tri_areas(verts, faces):
    """Per-triangle area in the XY plane: half the cross product of two edges."""
    v = verts[:, :2]
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return 0.5 * np.abs(np.cross(b - a, c - a))


def _is_crack_free(verts, faces, xmin, xmax, ymin, ymax):
    """True if the mesh has no interior T-junction cracks.

    In a conforming triangle mesh every *interior* edge is shared by exactly two
    triangles, so an edge used by only one triangle must sit on the domain
    perimeter. A single-use edge anywhere inside the domain is a crack."""
    # Tally triangles per undirected edge (sorted vertex-id pair as the key).
    counts = defaultdict(int)
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            counts[tuple(sorted((int(a), int(b))))] += 1
    for (i, j), c in counts.items():
        if c != 1:
            continue  # interior shared edge -> fine
        # A single-use edge is only allowed if both endpoints lie on the same
        # border line (both at xmin, or both at xmax, or both at ymin/ymax).
        pa, pb = verts[i], verts[j]
        on_perimeter = (
            (np.isclose(pa[0], xmin) and np.isclose(pb[0], xmin))
            or (np.isclose(pa[0], xmax) and np.isclose(pb[0], xmax))
            or (np.isclose(pa[1], ymin) and np.isclose(pb[1], ymin))
            or (np.isclose(pa[1], ymax) and np.isclose(pb[1], ymax))
        )
        if not on_perimeter:
            return False
    return True


class TestAdaptiveGrid:
    def test_construction_rejects_degenerate_dim(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            AdaptiveGrid(np.array([0.0]), np.arange(3.0), max_depth=2)

    def test_refine_everywhere_quadruples_per_level(self):
        g1 = _grid(2, 2, max_depth=2)
        g1.refine(_const(True), max_level=1)
        assert len(g1._leaves) == 2 * 2 * 4  # one level -> x4

        g2 = _grid(2, 2, max_depth=2)
        g2.refine(_const(True), max_level=2)
        assert len(g2._leaves) == 2 * 2 * 16  # two levels -> x16

    def test_refine_only_subdivides_predicate_region(self):
        grid = _grid(4, 4, max_depth=2)
        # True only for the leftmost column of base cells (xmin < 1).
        grid.refine(lambda xmn, ymn, xmx, ymx: xmn < 1.0, max_level=1)
        # 4 left cells -> 16 level-1 leaves; the other 12 base cells stay level 0.
        assert len(grid._leaves) == 28
        assert _leaf_levels(grid) == {0, 1}

    def test_unrefined_mesh_counts_and_area(self):
        grid = _grid(2, 2, max_depth=2)
        verts, faces = grid.to_mesh(_flat_z)

        assert len(verts) == 9  # (nx+1)*(ny+1) deduped corners
        assert len(faces) == 8  # 2 triangles per base cell
        assert faces.max() < len(verts)
        areas = _tri_areas(verts, faces)
        assert (areas > 1e-12).all()  # no degenerate triangles
        assert np.isclose(areas.sum(), 4.0)  # domain area = 2 * 2
        assert _is_crack_free(verts, faces, 0.0, 2.0, 0.0, 2.0)

    def test_balanced_refined_mesh_is_crack_free(self):
        grid = _grid(4, 4, max_depth=2)
        grid.refine(lambda xmn, ymn, xmx, ymx: xmx <= 1.0 + 1e-9, max_level=2)
        grid.balance()
        verts, faces = grid.to_mesh(_flat_z)

        areas = _tri_areas(verts, faces)
        assert (areas > 1e-12).all()
        assert np.isclose(areas.sum(), 16.0)  # domain area = 4 * 4, conserved
        assert _is_crack_free(verts, faces, 0.0, 4.0, 0.0, 4.0)


def _pyramid(elev, decimation_depth=3):
    """A DemErrorPyramid on a unit grid (origin 0, 1 m pixels).

    ``query`` infers the pyramid level from the cell width: with ``D=3`` an 8x8
    DEM has 1x1 cells at the finest level (residual 0) up to one 8x8 cell at the
    coarsest, so a 2 m-wide query cell hits the level whose blocks are 2x2 pixels.
    """
    return DemErrorPyramid(
        elev, x0=0.0, y0=0.0, dx=1.0, dy=1.0, decimation_depth=decimation_depth
    )


def _cell(xmin, xmax, ymin, ymax):
    """Wrap a single query rectangle as the length-1 arrays ``query`` expects."""
    return (
        np.array([float(xmin)]),
        np.array([float(ymin)]),
        np.array([float(xmax)]),
        np.array([float(ymax)]),
    )


class TestDemErrorPyramid:
    def test_flat_dem_never_exceeds_tolerance(self):
        pyr = _pyramid(np.full((8, 8), 5.0))
        xmn, ymn, xmx, ymx = _cell(2, 4, 2, 4)
        assert not pyr.query(xmn, ymn, xmx, ymx, tolerance_m=0.001).any()

    def test_tilted_plane_has_no_residual(self):
        # A perfect plane fits its own least-squares plane exactly -> 0 residual,
        # so a steep-but-flat surface is never refined.
        i, j = np.meshgrid(np.arange(8), np.arange(8), indexing="xy")
        elev = (2.0 * i + 3.0 * j).astype(float)
        pyr = _pyramid(elev)
        for xmin in (0, 2, 4, 6):
            xmn, ymn, xmx, ymx = _cell(xmin, xmin + 2, 0, 2)
            assert not pyr.query(xmn, ymn, xmx, ymx, tolerance_m=0.01).any()

    def test_localized_bump_refines_locally_and_respects_threshold(self):
        elev = np.zeros((8, 8))
        elev[2, 2] = 10.0  # a single spike
        pyr = _pyramid(elev)

        over_bump = _cell(2, 4, 2, 4)  # the 2x2 cell containing the spike
        assert pyr.query(*over_bump, tolerance_m=1.0).all()  # residual exceeds 1 m
        assert not pyr.query(*over_bump, tolerance_m=20.0).any()  # but not 20 m

        away = _cell(4, 6, 4, 6)  # a cell with no spike
        assert not pyr.query(*away, tolerance_m=1.0).any()

    def test_coarse_query_saturates_from_fine_bump(self):
        elev = np.zeros((8, 8))
        elev[2, 2] = 10.0
        pyr = _pyramid(elev)
        # A coarse cell covering the whole DEM still flags True

        whole = _cell(0, 8, 0, 8)
        assert pyr.query(*whole, tolerance_m=1.0).all()
