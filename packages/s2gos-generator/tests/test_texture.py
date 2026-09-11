"""Test way painting onto the selection texture (processors/texture.py)."""

import numpy as np
import pytest
from shapely.geometry import box

from s2gos_generator.core.grid import SceneGrid
from s2gos_generator.processors.terrain_texture import apply_ways

# 4 cells of 10 m centred on the origin, so cell centres sit at -15, -5, 5, 15.
GRID = SceneGrid(40.0, 4)

# A way crossing the southernmost row of cells, which spans y in -20..-10.
SOUTHERN_WAY = box(-25.0, -24.0, 25.0, -16.0)


def test_apply_ways_paints_material_at_south_row_zero():
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(
        texture, GRID, {"asphalt": SOUTHERN_WAY}, {"asphalt": 7}
    )

    assert out.shape == (4, 4)
    assert (out[0, :] == 7).all()  # southern way -> row 0
    assert (out[1:, :] == 0).all()  # rest untouched
    assert union_mask is not None
    assert np.array_equal(union_mask, out == 7)


@pytest.mark.parametrize(
    "way_polys,index_map",
    [
        ({}, {"asphalt": 7}),  # no way geometry at all
        ({"asphalt": SOUTHERN_WAY}, {"concrete": 3}),  # material absent from index map
    ],
    ids=["no-way-geoms", "material-not-in-index"],
)
def test_apply_ways_paints_nothing(way_polys, index_map):
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(texture, GRID, way_polys, index_map)
    assert union_mask is None
    assert (out == 0).all()


def test_matched_materials_sidecar_roundtrip():
    from s2gos_generator.processors.spectral.diversify import (
        matched_materials_from_sidecar,
        matched_materials_to_sidecar,
    )

    defs = {"soil": {"type": "diffuse"}}
    indices = {"soil": 12}
    sidecar = matched_materials_to_sidecar(defs, indices, [30, 60])
    assert sidecar["source_landcover_classes"] == [30, 60]
    restored = matched_materials_from_sidecar(sidecar)
    assert restored == {"materials": defs, "material_indices": indices}
    assert matched_materials_from_sidecar({"version": 2}) == {}
