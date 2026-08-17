"""Test road painting onto the selection texture (processors/texture.py)."""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from s2gos_generator.processors.terrain_texture import apply_ways


def _write_landcover(path):
    """A 4x4 landcover zarr on a 10 m grid spanning 0..30 m (pixel centres)."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.int32),
        dims=("y", "x"),
        coords={"y": [0.0, 10.0, 20.0, 30.0], "x": [0.0, 10.0, 20.0, 30.0]},
    )
    da.to_dataset(name="landcover").to_zarr(str(path))
    return path


def test_apply_roads_paints_material_at_south_row_zero(tmp_path):
    lc_path = _write_landcover(tmp_path / "lc.zarr")

    road_polygons = {"asphalt": box(-5.0, -4.0, 35.0, 4.0)}
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(texture, lc_path, road_polygons, {"asphalt": 7})

    assert out.shape == (4, 4)
    assert (out[0, :] == 7).all()  # southern road -> row 0
    assert (out[1:, :] == 0).all()  # rest untouched
    assert union_mask is not None
    assert np.array_equal(union_mask, out == 7)


@pytest.mark.parametrize(
    "road_polys,index_map",
    [
        ({}, {"asphalt": 7}),  # no road geometry at all
        (
            {"asphalt": box(-5.0, -4.0, 35.0, 4.0)},
            {"concrete": 3},
        ),  # material absent from index map
    ],
    ids=["no-road-geoms", "material-not-in-index"],
)
def test_apply_roads_paints_nothing(tmp_path, road_polys, index_map):
    lc_path = _write_landcover(tmp_path / "lc.zarr")
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(texture, lc_path, road_polys, index_map)
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
