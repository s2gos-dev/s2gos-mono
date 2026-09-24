"""Test way painting onto the selection texture (processors/texture.py)."""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from s2gos_generator.processors.terrain_texture import (
    apply_ways,
    strip_steep_water_pixels,
)


def _write_landcover(path):
    """A 4x4 landcover zarr on a 10 m grid spanning 0..30 m (pixel centres)."""
    da = xr.DataArray(
        np.zeros((4, 4), dtype=np.int32),
        dims=("y", "x"),
        coords={"y": [0.0, 10.0, 20.0, 30.0], "x": [0.0, 10.0, 20.0, 30.0]},
    )
    da.to_dataset(name="landcover").to_zarr(str(path))
    return path


def _write_dem(path, elev):
    """A 4x4 DEM zarr on the same grid as _write_landcover()."""
    da = xr.DataArray(
        np.asarray(elev, dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [0.0, 10.0, 20.0, 30.0], "x": [0.0, 10.0, 20.0, 30.0]},
    )
    da.to_dataset(name="elevation").to_zarr(str(path))
    return path


def test_apply_ways_paints_material_at_south_row_zero(tmp_path):
    lc_path = _write_landcover(tmp_path / "lc.zarr")

    way_polygons = {"asphalt": box(-5.0, -4.0, 35.0, 4.0)}
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(texture, lc_path, way_polygons, {"asphalt": 7})

    assert out.shape == (4, 4)
    assert (out[0, :] == 7).all()  # southern way -> row 0
    assert (out[1:, :] == 0).all()  # rest untouched
    assert union_mask is not None
    assert np.array_equal(union_mask, out == 7)


@pytest.mark.parametrize(
    "way_polys,index_map",
    [
        ({}, {"asphalt": 7}),  # no way geometry at all
        (
            {"asphalt": box(-5.0, -4.0, 35.0, 4.0)},
            {"concrete": 3},
        ),  # material absent from index map
    ],
    ids=["no-way-geoms", "material-not-in-index"],
)
def test_apply_ways_paints_nothing(tmp_path, way_polys, index_map):
    lc_path = _write_landcover(tmp_path / "lc.zarr")
    texture = np.zeros((4, 4), dtype=np.uint8)

    out, union_mask = apply_ways(texture, lc_path, way_polys, index_map)
    assert union_mask is None
    assert (out == 0).all()


def test_apply_roads_upsampled_extent_aligns_with_native_pixels(tmp_path):
    # Regression test: the rasterization bounds margin must be half a
    # *native* landcover pixel, not half the (possibly finer) upsampled
    # raster pixel -- else the polygon-painted layer desyncs from the
    # NEAREST-resized base layer.
    #
    # Native grid: 4x4 @ 10 m (pixel centres 0/10/20/30 -> true edge-to-edge
    # extent -5..35). Upsampled to texture_resolution_m=5.0 (scale 2x) ->
    # 8x8 @ true pixel size 5.0 m, extent still -5..35, column j spans
    # [-5 + 5j, -5 + 5(j+1)).
    #
    # A polygon covering x in [-100, 24.9] (i.e. up to just inside native
    # pixel column 2's span [15, 25)) should, under a correctly-aligned
    # transform, paint exactly upsampled columns 0-5 (x < 25) and leave
    # columns 6-7 (the last native pixel's width, 10 m = 2 upsampled
    # columns) unpainted. Under the pre-fix bug (margin computed from the
    # upsampled resolution instead of native), the assumed extent shrinks
    # and every column boundary shifts, leaking paint into column 6 too.
    lc_path = _write_landcover(tmp_path / "lc.zarr")
    texture = np.zeros((8, 8), dtype=np.uint8)

    way_polygons = {"asphalt": box(-100.0, -100.0, 24.9, 100.0)}
    out, union_mask = apply_ways(
        texture, lc_path, way_polygons, {"asphalt": 7}, texture_resolution_m=5.0
    )

    assert out.shape == (8, 8)
    assert (out[:, :6] == 7).all()
    assert (out[:, 6:] == 0).all()
    assert union_mask is not None


def test_way_pixels_restored_after_water_paint_at_crossing(tmp_path):
    """Regression: apply_ways() rasterizes each polygon independently with
    all_touched=True, which inflates each polygon's raster footprint by up to
    ~1px along its boundary -- independently of any other polygon. So even
    though ctx.water_polygons_by_material vector-subtracts the way footprint
    from water (see SceneResourceContext.water_polygons_by_material), a pixel
    straddling the way/water boundary can genuinely overlap *both* polygons'
    areas and get touched by both rasterizations. Water paints after ways in
    _generate_texture(), so without restoring way pixels afterward, water
    reclaims a rim of the way at every crossing.
    """
    lc_path = _write_landcover(tmp_path / "lc.zarr")
    texture = np.zeros((4, 4), dtype=np.uint8)
    index_map = {"asphalt": 7, "water": 9}

    # Way covers x < 13.7 (native columns span 0/10/20/30 -> edges -5..35,
    # so column index 1 spans x in [5, 15) and straddles the boundary).
    way_poly = box(-100.0, -100.0, 13.7, 100.0)
    # water_polygons_by_material already vector-subtracts the way union --
    # mirror that here, rather than passing an unclipped water polygon.
    water_before = box(-100.0, -100.0, 100.0, 100.0)
    water_after_fix = water_before.difference(way_poly)

    out, way_mask = apply_ways(texture, lc_path, {"asphalt": way_poly}, index_map)
    assert way_mask is not None
    way_snapshot = out.copy()

    out, water_mask = apply_ways(out, lc_path, {"water": water_after_fix}, index_map)
    assert water_mask is not None

    # Without the fix: water's own all_touched rasterization reclaims part of
    # the boundary column back from asphalt, even though the vector water
    # polygon never overlaps the vector way polygon.
    assert not (out[way_mask] == 7).all(), (
        "test setup didn't reproduce the all_touched boundary overlap -- "
        "adjust way_poly's boundary so it falls strictly inside a pixel"
    )

    # With the fix applied (mirrors resources/texture.py::_generate_texture):
    # every pixel ways touched is restored to what ways painted there.
    out[way_mask] = way_snapshot[way_mask]
    assert (out[way_mask] == 7).all()


class TestStripSteepWaterPixels:
    """Regression tests: a vetted water body's texture footprint can still land on ground that isn't flat, and must be repainted with the nearest land material."""

    GRASS = 5
    WATER = 9

    def _texture_all_water_one_grass_corner(self):
        # (0,0) is grass; everything else starts out painted as water.
        texture = np.full((4, 4), self.WATER, dtype=np.uint8)
        texture[0, 0] = self.GRASS
        water_mask = texture == self.WATER
        return texture, water_mask

    def test_steep_water_pixel_repainted_from_nearest_land(self, tmp_path):
        lc_path = _write_landcover(tmp_path / "lc.zarr")
        # Flat everywhere except a cliff at the far corner (row 3, col 3).
        elev = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1000],
        ]
        dem_path = _write_dem(tmp_path / "dem.zarr", elev)
        texture, water_mask = self._texture_all_water_one_grass_corner()

        out, changed = strip_steep_water_pixels(
            texture, dem_path, lc_path, water_mask, self.WATER, max_slope=1.0
        )

        assert changed is True
        # Picked up the only land material present (grass), not a generic fallback.
        assert out[3, 3] == self.GRASS
        assert out[0, 1] == self.WATER  # flat cell stays water

    def test_gentle_water_is_untouched(self, tmp_path):
        lc_path = _write_landcover(tmp_path / "lc.zarr")
        dem_path = _write_dem(tmp_path / "dem.zarr", np.zeros((4, 4)))  # dead flat
        texture, water_mask = self._texture_all_water_one_grass_corner()

        out, changed = strip_steep_water_pixels(
            texture, dem_path, lc_path, water_mask, self.WATER, max_slope=1.0
        )

        assert changed is False
        assert np.array_equal(out, texture)

    def test_no_water_mask_is_a_noop(self, tmp_path):
        lc_path = _write_landcover(tmp_path / "lc.zarr")
        dem_path = _write_dem(tmp_path / "dem.zarr", np.zeros((4, 4)))
        texture = np.full((4, 4), self.WATER, dtype=np.uint8)

        out, changed = strip_steep_water_pixels(
            texture,
            dem_path,
            lc_path,
            np.zeros((4, 4), dtype=bool),
            self.WATER,
            max_slope=1.0,
        )

        assert changed is False
        assert np.array_equal(out, texture)

    def test_never_copies_material_from_another_water_pixel(self, tmp_path):
        # Two steep spots sharing only a grass corner as non-water neighbor -- must never copy each other's index.
        lc_path = _write_landcover(tmp_path / "lc.zarr")
        elev = [
            [0, 0, 0, 500],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [500, 0, 0, 0],
        ]
        dem_path = _write_dem(tmp_path / "dem.zarr", elev)
        texture, water_mask = self._texture_all_water_one_grass_corner()

        out, changed = strip_steep_water_pixels(
            texture, dem_path, lc_path, water_mask, self.WATER, max_slope=1.0
        )

        assert changed is True
        steep_cells = out[water_mask.astype(bool) & (out != self.WATER)]
        assert (steep_cells == self.GRASS).all()


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
