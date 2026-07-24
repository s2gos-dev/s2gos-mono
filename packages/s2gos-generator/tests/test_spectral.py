"""Spectral material-matching test suite."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from s2gos_generator.core.config.material_match import SpectralMatchingConfig
from s2gos_generator.core.materials import MAX_MATERIAL_INDEX, allocate_matched_indices
from s2gos_generator.processors.spectral.diversify import diversify_selection_texture
from s2gos_generator.processors.spectral.library import (
    CandidateSpectrum,
    load_candidate_library,
)
from s2gos_generator.processors.spectral.sam import (
    NO_CLUSTER,
    cluster_class_reflectance,
    match_clusters_to_library,
    spectral_angle,
)
from s2gos_generator.processors.spectral.sentinel2 import (
    _accumulate_until_covered,
    _band_radiometry,
    _scene_geobox,
    _scl_valid_mask,
    _to_band_array,
)

BANDS = ["B02", "B03", "B04", "B08"]

# A real materials.json so the material_library existence validator passes
MATERIALS_JSON = str(
    Path(__file__).resolve().parents[1] / "resources" / "data" / "materials.json"
)


def _valid(**overrides):
    """A validated SpectralMatchingConfig pointing at the shipped materials.json."""
    kwargs = dict(
        landcover_classes=[30, 60],
        material_library=MATERIALS_JSON,
        acquisition_date="2021-07-15",
    )
    kwargs.update(overrides)
    return SpectralMatchingConfig(**kwargs)


def _diversify_config(**overrides):
    """A validation-bypassing config for exercising the diversify algorithm."""
    kwargs = dict(
        landcover_classes=[30],
        clusters_per_class=2,
        random_seed=0,
        bands=BANDS,
        max_sam_angle_deg=None,
    )
    kwargs.update(overrides)
    return SpectralMatchingConfig.model_construct(**kwargs)


class TestSpectralAngle:
    """``spectral_angle``."""

    @pytest.mark.parametrize(
        "v1,v2,expected_rad",
        [
            # Scale-invariant: a vector and its multiple are identical in angle.
            (np.array([0.1, 0.3, 0.5, 0.4]), np.array([0.2, 0.6, 1.0, 0.8]), 0.0),
            # Orthogonal unit axes -> 90 degrees.
            (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.pi / 2),
            # Zero-norm vector is defined as maximally dissimilar.
            (np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.pi / 2),
        ],
        ids=["identical-scaled", "orthogonal", "zero-norm"],
    )
    def test_spectral_angle(self, v1, v2, expected_rad):
        assert spectral_angle(v1, v2) == pytest.approx(expected_rad, abs=1e-9)


def _two_cluster_scene():
    nb, ny, nx = 4, 4, 4
    refl = np.zeros((nb, ny, nx), "float32")
    refl[:, :2, :] = np.array([0.1, 0.1, 0.1, 0.5])[:, None, None]
    refl[:, 2:, :] = np.array([0.4, 0.4, 0.4, 0.45])[:, None, None]
    return refl


class TestClustering:
    """``cluster_class_reflectance`` and ``match_clusters_to_library``."""

    def test_assigns_two_clusters_with_fixed_seed(self):
        refl = _two_cluster_scene()
        mask = np.ones((4, 4), bool)
        label_map, palette = cluster_class_reflectance(refl, mask, 2, random_seed=0)

        assert label_map.shape == (4, 4)
        assert set(np.unique(label_map)) <= {0, 1}
        assert label_map[0, 0] != label_map[3, 0]
        assert np.isfinite(palette[:, 0]).sum() == 2

    def test_excludes_unmasked_and_nonfinite_pixels(self):
        refl = _two_cluster_scene()
        refl[:, 0, 0] = np.nan  # a NaN pixel inside the mask
        mask = np.ones((4, 4), bool)
        mask[3, :] = False  # an out-of-mask row
        label_map, _ = cluster_class_reflectance(refl, mask, 2, random_seed=0)

        assert label_map[0, 0] == NO_CLUSTER  # NaN inside the mask excluded
        assert (label_map[3, :] == NO_CLUSTER).all()  # out-of-mask excluded
        # Every remaining selected, finite pixel is clustered.
        selected = mask & np.isfinite(refl[0])
        assert (label_map[selected] != NO_CLUSTER).all()

    def test_match_picks_nearest_and_respects_angle_gate(self):
        library = [
            CandidateSpectrum(
                "dark", {"type": "diffuse"}, np.array([0.1, 0.1, 0.1, 0.5])
            ),
            CandidateSpectrum(
                "bright", {"type": "diffuse"}, np.array([0.4, 0.4, 0.4, 0.45])
            ),
        ]
        palette = np.array(
            [
                [0.11, 0.10, 0.09, 0.52],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        matches = match_clusters_to_library(palette, library, max_sam_angle_deg=None)
        assert matches[0].material_id == "dark"
        assert matches[1] is None

        # A strict angle gate rejects even the nearest match.
        gated = match_clusters_to_library(palette, library, max_sam_angle_deg=0.001)
        assert gated[0] is None

    def test_all_zero_centroid_matches_nothing(self):
        # A zero centroid is equidistant (pi/2) from every candidate, so any match
        # would be arbitrary iteration order; fall back to the base material instead.
        library = [
            CandidateSpectrum("dark", {"type": "diffuse"}, np.array([0.1, 0.5])),
            CandidateSpectrum("bright", {"type": "diffuse"}, np.array([0.4, 0.45])),
        ]
        palette = np.array([[0.0, 0.0]])
        assert match_clusters_to_library(palette, library, None) == [None]


def _write_spectrum(path, w_nm, reflectance):
    """Write a NetCDF reflectance spectrum with a 'w' wavelength coordinate."""
    xr.Dataset(
        {"reflectance": ("w", np.asarray(reflectance, "float64"))},
        coords={"w": np.asarray(w_nm, "float64")},
    ).to_netcdf(path)


@pytest.fixture
def library_dir(tmp_path):
    """A materials.json-style library with diffuse, out-of-range, and non-diffuse entries."""
    spectra = tmp_path / "spectra"
    spectra.mkdir()

    # In-range diffuse (nm), spans the S2 bands.
    _write_spectrum(spectra / "soil.nc", [400, 900, 2500], [0.1, 0.3, 0.4])
    # In-range diffuse expressed in micrometres -> must be auto-converted.
    _write_spectrum(spectra / "veg_um.nc", [0.4, 0.9, 2.5], [0.05, 0.5, 0.2])
    # Out-of-range (does not reach NIR 842 nm) -> skipped.
    _write_spectrum(spectra / "short.nc", [400, 500, 600], [0.1, 0.2, 0.3])

    config = {
        "version": "0.0.1",
        "materials": {
            "soil": {
                "type": "diffuse",
                "reflectance": {"path": "spectra/soil.nc", "variable": "reflectance"},
            },
            "veg": {
                "type": "diffuse",
                "reflectance": {"path": "spectra/veg_um.nc", "variable": "reflectance"},
            },
            "tooshort": {
                "type": "diffuse",
                "reflectance": {"path": "spectra/short.nc", "variable": "reflectance"},
            },
            # Uniform scalar diffuse -> a flat reflectance vector across every band.
            "uniform_gray": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": 0.2},
            },
            # Uniform RGB triplet carries no NIR/SWIR info -> skipped.
            "rgb_paint": {
                "type": "diffuse",
                "reflectance": {"type": "uniform", "value": [0.2, 0.3, 0.4]},
            },
            "water": {
                "type": "rpv",
                "rho_0": {"path": "spectra/soil.nc", "variable": "reflectance"},
                "k": {"type": "uniform", "value": 0.5},
                "Theta": {"type": "uniform", "value": 0.0},
                "rho_c": {"type": "uniform", "value": 0.1},
            },
        },
        "landcover_mapping": {"cropland": "soil"},
    }
    config_path = tmp_path / "materials.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestLibrary:
    """``load_candidate_library`` filtering, unit conversion, and error handling."""

    def test_loads_only_in_range_diffuse_materials(self, library_dir):
        library = load_candidate_library(library_dir, BANDS)
        ids = {c.material_id for c in library}
        # rpv, out-of-range spectrum, and RGB-uniform (no NIR/SWIR info) all dropped.
        assert ids == {"soil", "veg", "uniform_gray"}
        # Every loaded candidate has one finite reflectance per band.
        for candidate in library:
            assert candidate.s2_vector.shape == (len(BANDS),)
            assert np.isfinite(candidate.s2_vector).all()
            assert candidate.material_def["type"] == "diffuse"

    def test_micrometre_spectrum_interpolates_like_nanometres(self, library_dir):
        # veg_um.nc (µm) and an identical nm spectrum yield the same B03 (560 nm) value.
        library = {c.material_id: c for c in load_candidate_library(library_dir, BANDS)}
        expected_b03 = np.interp(560.0, [400, 900, 2500], [0.05, 0.5, 0.2])
        assert library["veg"].s2_vector[1] == pytest.approx(expected_b03, rel=1e-6)

    def test_uniform_scalar_is_flat_and_rgb_triplet_is_dropped(self, library_dir):
        # A uniform scalar diffuse becomes the same reflectance in every band;
        # a uniform RGB triplet has no NIR/SWIR meaning and must not be a candidate.
        library = {c.material_id: c for c in load_candidate_library(library_dir, BANDS)}
        assert library["uniform_gray"].s2_vector == pytest.approx([0.2] * len(BANDS))
        assert "rgb_paint" not in library

    def test_raises_when_no_usable_diffuse_materials(self, tmp_path):
        # A library with only a non-diffuse material has nothing to match against.
        config = {
            "version": "0.0.1",
            "materials": {
                "water": {
                    "type": "rpv",
                    "rho_0": {"type": "uniform", "value": 0.1},
                    "k": {"type": "uniform", "value": 0.5},
                    "Theta": {"type": "uniform", "value": 0.0},
                    "rho_c": {"type": "uniform", "value": 0.1},
                }
            },
            "landcover_mapping": {},
        }
        path = tmp_path / "materials.json"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="No usable diffuse candidate"):
            load_candidate_library(path, BANDS)


NY = NX = 8


def _layer(value, region):
    """A (band, y, x) layer with ``value`` over ``region`` ('west'/'east'/'all'), NaN else."""
    arr = np.full((2, NY, NX), np.nan, dtype="float32")
    if region == "west":
        arr[:, :, : NX // 2] = value
    elif region == "east":
        arr[:, :, NX // 2 :] = value
    else:  # all
        arr[:] = value
    return xr.DataArray(arr, dims=("band", "y", "x"), coords={"band": ["B02", "B04"]})


def _counting_loader(layers_by_date):
    loaded = []

    def load_fn(d):
        loaded.append(d)
        return layers_by_date[d]

    return load_fn, loaded


def _mask(region):
    """A (y, x) bool mask, True over ``region`` ('west'/'east'/'all'/'none')."""
    arr = np.zeros((NY, NX), dtype=bool)
    if region == "west":
        arr[:, : NX // 2] = True
    elif region == "east":
        arr[:, NX // 2 :] = True
    elif region == "all":
        arr[:] = True
    return xr.DataArray(arr, dims=("y", "x"))


class TestSentinel2Composite:
    """Composite coverage logic (``_accumulate_until_covered`` and helpers)."""

    def test_fills_gap_from_later_date_when_closest_is_single_tile(self):
        # Closest date covers only the west tile; the next date covers only the east.
        layers = {
            0: _layer(1.0, "west"),
            1: _layer(2.0, "east"),
            2: _layer(9.0, "all"),
        }
        load_fn, loaded = _counting_loader(layers)

        composite, coverage = _accumulate_until_covered(
            order=[0, 1, 2], load_fn=load_fn, min_coverage=0.99
        )

        assert coverage == 1.0
        assert not np.isnan(composite.values).any()  # the eastern hole is filled
        assert loaded == [0, 1]  # stopped once covered; the third date was not needed
        assert np.all(composite.values[:, :, : NX // 2] == 1.0)
        assert np.all(composite.values[:, :, NX // 2 :] == 2.0)

    def test_degraded_returns_best_effort_without_raising(self):
        # Every candidate only ever covers the west half -> AOI can't be fully filled.
        layers = {d: _layer(float(d), "west") for d in range(5)}
        load_fn, loaded = _counting_loader(layers)

        composite, coverage = _accumulate_until_covered(
            order=list(range(5)), load_fn=load_fn, min_coverage=0.99
        )

        assert abs(coverage - 0.5) < 1e-6
        # Uncoverable pixels stay NaN -> downstream they keep their base material.
        assert np.isnan(composite.values[:, :, NX // 2 :]).all()
        assert len(loaded) == 5  # exhausts the candidates

    def test_skips_failed_reads_and_recovers(self):
        layers = {1: _layer(2.0, "east"), 2: _layer(3.0, "west")}

        def load_fn(d):
            if d == 0:
                raise RuntimeError("simulated transient read failure")
            return layers[d]

        composite, coverage = _accumulate_until_covered(
            order=[0, 1, 2], load_fn=load_fn, min_coverage=0.99
        )

        assert coverage == 1.0
        assert not np.isnan(composite.values).any()

    def test_raises_when_every_date_fails_to_load(self):
        def load_fn(d):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="every candidate date failed"):
            _accumulate_until_covered(
                order=[0, 1, 2], load_fn=load_fn, min_coverage=0.99
            )

    def test_screened_out_date_is_skipped_without_loading(self):
        # Date 0 has no valid pixels (e.g. fully cloudy per SCL): its
        # reflectance bands must never be fetched.
        layers = {0: _layer(1.0, "all"), 1: _layer(2.0, "all")}
        masks = {0: _mask("none"), 1: _mask("all")}
        load_fn, loaded = _counting_loader(layers)

        composite, coverage = _accumulate_until_covered(
            order=[0, 1],
            load_fn=load_fn,
            min_coverage=0.99,
            valid_mask_fn=lambda d: masks[d],
        )

        assert loaded == [1]
        assert coverage == 1.0
        assert np.all(composite.values == 2.0)

    def test_date_valid_only_over_covered_area_is_skipped(self):
        # Date 1 is valid only where date 0 already covered -> nothing to add.
        layers = {0: _layer(1.0, "west"), 1: _layer(9.0, "all"), 2: _layer(2.0, "all")}
        masks = {0: _mask("west"), 1: _mask("west"), 2: _mask("all")}
        load_fn, loaded = _counting_loader(layers)

        composite, coverage = _accumulate_until_covered(
            order=[0, 1, 2],
            load_fn=load_fn,
            min_coverage=0.99,
            valid_mask_fn=lambda d: masks[d],
        )

        assert loaded == [0, 2]
        assert coverage == 1.0
        assert np.all(composite.values[:, :, : NX // 2] == 1.0)
        assert np.all(composite.values[:, :, NX // 2 :] == 2.0)

    def test_loaded_layer_is_masked_to_valid_pixels(self):
        # The layer carries data everywhere, but only the west half is valid
        # (e.g. clouds over the east): invalid pixels must stay NaN.
        load_fn, _ = _counting_loader({0: _layer(1.0, "all")})

        composite, coverage = _accumulate_until_covered(
            order=[0],
            load_fn=load_fn,
            min_coverage=0.99,
            valid_mask_fn=lambda d: _mask("west"),
        )

        assert abs(coverage - 0.5) < 1e-6
        assert np.all(composite.values[:, :, : NX // 2] == 1.0)
        assert np.isnan(composite.values[:, :, NX // 2 :]).all()

    def test_raises_when_every_date_is_screened_out(self):
        load_fn, loaded = _counting_loader({})

        with pytest.raises(RuntimeError, match="screened out"):
            _accumulate_until_covered(
                order=[0, 1],
                load_fn=load_fn,
                min_coverage=0.99,
                valid_mask_fn=lambda d: _mask("none"),
            )
        assert loaded == []

    def test_mask_fetch_failure_is_skipped_like_a_load_failure(self):
        load_fn, loaded = _counting_loader({1: _layer(1.0, "all")})

        def mask_fn(d):
            if d == 0:
                raise RuntimeError("simulated SCL read failure")
            return _mask("all")

        composite, coverage = _accumulate_until_covered(
            order=[0, 1], load_fn=load_fn, min_coverage=0.99, valid_mask_fn=mask_fn
        )

        assert loaded == [1]
        assert coverage == 1.0


class TestSCLValidMask:
    """``_scl_valid_mask`` — SCL classes to a usable-pixel mask."""

    def test_excludes_configured_classes_and_nodata(self):
        scl = xr.DataArray(
            np.array([[0, 4, 8], [9, 5, 3]], dtype="uint16"), dims=("y", "x")
        )
        valid = _scl_valid_mask(scl, exclude=[3, 8, 9])
        assert valid.values.tolist() == [[False, True, False], [False, True, False]]

    def test_empty_exclude_masks_only_nodata(self):
        scl = xr.DataArray(np.array([[0, 8, 4]], dtype="uint16"), dims=("y", "x"))
        valid = _scl_valid_mask(scl, exclude=[])
        assert valid.values.tolist() == [[False, True, True]]


def _centre_grid(n, res):
    """Ascending pixel-centre coords centred on 0, mirroring the landcover grid."""
    half = n * res / 2
    return np.linspace(-half + res / 2, half - res / 2, n)


class TestSceneGeobox:
    """``_scene_geobox`` - scene-grid GeoBox construction for odc-stac."""

    def test_pixel_centres_round_trip_through_the_affine(self):
        pytest.importorskip("odc.geo")
        res = 10.0
        grid_y, grid_x = _centre_grid(6, res), _centre_grid(4, res)

        gbox = _scene_geobox(grid_y, grid_x, "EPSG:32633")

        # Shape is (ny, nx) — a non-square grid catches swapped axes.
        assert gbox.shape == (6, 4)
        # Centre of the top-left pixel is (west-most x, north-most y)...
        assert gbox.transform * (0.5, 0.5) == pytest.approx((grid_x[0], grid_y[-1]))
        # ...and of the bottom-right pixel (east-most x, south-most y).
        assert gbox.transform * (3.5, 5.5) == pytest.approx((grid_x[-1], grid_y[0]))

    def test_irregular_spacing_raises(self):
        pytest.importorskip("odc.geo")
        grid_y = _centre_grid(4, 10.0)
        grid_x = _centre_grid(4, 10.0).copy()
        grid_x[2] += 1.0
        with pytest.raises(ValueError, match="regularly spaced"):
            _scene_geobox(grid_y, grid_x, "EPSG:32633")


class TestBandArray:
    """``_to_band_array`` — metadata DN->reflectance conversion and non-positive masking."""

    # Per-band radiometry as published by the STAC assets (ESA baseline >= 04.00 values).
    RADIOMETRY = {"B02": (1e-4, -0.1), "B04": (1e-4, -0.1)}

    def test_converts_orders_and_masks_nonpositive_reflectance(self):
        dn = {
            # (time, y, x); with scale 1e-4 / offset -0.1, DN <= 1000 encodes reflectance
            # <= 0: the nodata fill (0) and the degraded swath-edge rim that would otherwise
            # clip to black seams.
            "B04_10m": np.array([[[1000, 0], [2000, 3000]]], dtype="uint16"),
            "B02_10m": np.array([[[1001, 1500], [0, 700]]], dtype="uint16"),
        }
        ds = xr.Dataset(
            {k: (("time", "y", "x"), v) for k, v in dn.items()},
            coords={"time": [np.datetime64("2021-07-15", "ns")]},
        )

        out = _to_band_array(ds, ["B02", "B04"], self.RADIOMETRY)

        # Requested order wins over Dataset variable order.
        assert list(out.band.values) == ["B02", "B04"]
        assert out.dtype == np.float32
        b04 = out.sel(band="B04").values[0]
        assert np.isnan(b04[0, 0])  # DN 1000 -> reflectance exactly 0 -> masked
        assert np.isnan(b04[0, 1])  # DN 0 nodata fill -> reflectance -0.1 -> masked
        assert b04[1, 0] == pytest.approx(0.1)  # DN 2000 -> reflectance 0.1
        assert b04[1, 1] == pytest.approx(0.2)  # DN 3000 -> reflectance 0.2
        b02 = out.sel(band="B02").values[0]
        assert b02[0, 0] == pytest.approx(
            1e-4
        )  # smallest positive-reflectance DN survives
        assert b02[0, 1] == pytest.approx(0.05)  # DN 1500 -> reflectance 0.05
        assert np.isnan(b02[1, 0])  # nodata fill
        assert np.isnan(b02[1, 1])  # degraded swath-edge sample (DN 700 -> -0.03)


def _stub_collection(item_assets: dict):
    """A minimal stand-in for a pystac Collection: ``.item_assets[key].to_dict()``.

    ``item_assets`` maps an asset key to its item-asset property dict. Mirrors the real CDSE
    collection, where radiometry lives on ``collection.item_assets['B02_10m']`` as
    ``raster:scale``/``raster:offset``.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        item_assets={
            k: SimpleNamespace(to_dict=lambda d=v: d) for k, v in item_assets.items()
        }
    )


class TestBandRadiometry:
    """``_band_radiometry`` — per-band scale/offset read from STAC collection/item metadata."""

    def test_reads_from_collection_item_assets(self):
        # Canonical CDSE form: raster:scale/raster:offset on the collection item_assets.
        collection = _stub_collection(
            {"B04_10m": {"raster:scale": 1e-4, "raster:offset": -0.1}}
        )
        assert _band_radiometry(collection, ["B04"], "stac://x") == {
            "B04": (1e-4, -0.1)
        }

    @pytest.mark.parametrize(
        "item_assets",
        [
            {},  # band absent from item_assets entirely
            {"B04_10m": {}},  # asset present but no radiometry keys
            {"B04_10m": {"raster:scale": 1e-4}},  # only scale, offset missing
        ],
        ids=["band-absent", "no-radiometry", "half-declared-pair"],
    )
    def test_raises_when_radiometry_incomplete(self, item_assets):
        # All three drive the same guard: a missing scale/offset must raise rather
        # than silently defaulting the offset to 0, which would skew every reflectance.
        with pytest.raises(RuntimeError, match="B04_10m"):
            _band_radiometry(_stub_collection(item_assets), ["B04"], "stac://catalog")


def _scene_with_one_class():
    # 4x4 grid; rows 0-1 are class 30 (split into two reflectance clusters),
    # rows 2-3 are class 10 (untouched, base index 0).
    ny, nx = 4, 4
    landcover = np.full((ny, nx), 10, dtype=np.int32)
    landcover[:2, :] = 30
    texture = np.zeros((ny, nx), dtype=np.uint8)  # all base index 0

    refl = np.zeros((4, ny, nx), "float32")
    refl[:, 0, :] = np.array([0.1, 0.1, 0.1, 0.5])[:, None]
    refl[:, 1, :] = np.array([0.4, 0.4, 0.4, 0.45])[:, None]
    refl[:, 2:, :] = np.nan  # non-class pixels need no reflectance
    return texture, landcover, refl


class TestDiversify:
    """Index allocation (``core/materials.py``) and selection-texture diversification."""

    def test_allocate_reuses_existing_and_appends_new_contiguously(self):
        base = {"treecover": 0, "grassland": 2, "water": 7, "asphalt": 11}
        got = allocate_matched_indices(base, ["lichen", "lichen", "asphalt", "sand"])
        assert got["asphalt"] == 11  # reused
        # new ids get contiguous indices above the current max.
        assert sorted([got["lichen"], got["sand"]]) == [12, 13]

    def test_allocate_raises_above_byte_ceiling(self):
        base = {"x": MAX_MATERIAL_INDEX}
        with pytest.raises(ValueError, match="exceeds the 8-bit"):
            allocate_matched_indices(base, ["new_one"])

    def test_paints_only_target_class_and_dedups(self):
        texture, landcover, refl = _scene_with_one_class()
        base = {"treecover": 0, "grassland": 1}
        library = [
            CandidateSpectrum(
                "dark",
                {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.1}},
                np.array([0.1, 0.1, 0.1, 0.5]),
            ),
            CandidateSpectrum(
                "bright",
                {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.4}},
                np.array([0.4, 0.4, 0.4, 0.45]),
            ),
        ]

        out, defs, indices = diversify_selection_texture(
            texture, landcover, refl, _diversify_config(), base, library
        )

        # Non-class rows untouched.
        assert (out[2:, :] == 0).all()
        # Class rows now carry the two new spectral indices (2 and 3, above base max 1).
        assert set(np.unique(out[:2, :])) == set(indices.values())
        assert set(indices.values()) == {2, 3}
        # Both candidate ids are new -> defs registered, painted values match the map.
        assert set(defs) == {"dark", "bright"}
        for mid, idx in indices.items():
            assert (out == idx).any()

    def test_uncovered_class_pixels_keep_their_base_material(self, caplog):
        texture, landcover, refl = _scene_with_one_class()
        texture[:, :] = 1  # base landcover index for the whole grid
        refl[:, 0, :] = np.nan  # row 0 of class 30 has no Sentinel-2 coverage
        library = [
            CandidateSpectrum(
                "bright",
                {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.4}},
                np.array([0.4, 0.4, 0.4, 0.45]),
            ),
        ]

        with caplog.at_level("WARNING"):
            out, _, indices = diversify_selection_texture(
                texture,
                landcover,
                refl,
                _diversify_config(clusters_per_class=1),
                {"grassland": 1},
                library,
            )

        # Uncovered class pixels are never painted; the covered row is.
        assert (out[0, :] == 1).all()
        assert (out[1, :] == indices["bright"]).all()
        # A warning fires when class pixels lack Sentinel-2 coverage.
        assert any(
            r.levelname == "WARNING" and "no reflectance" in r.message
            for r in caplog.records
        )

    def test_duplicate_match_collapses_to_one_index(self):
        texture, landcover, refl = _scene_with_one_class()
        base = {"grassland": 1}
        # Both clusters match the SAME material -> one shared index, no paint conflict.
        only = CandidateSpectrum(
            "uniform_mat", {"type": "diffuse"}, np.array([0.25, 0.25, 0.25, 0.48])
        )
        out, defs, indices = diversify_selection_texture(
            texture, landcover, refl, _diversify_config(), base, [only]
        )
        assert indices == {"uniform_mat": 2}
        assert set(defs) == {"uniform_mat"}
        assert (out[:2, :] == 2).all()


class TestConfig:
    """``SpectralMatchingConfig`` validation and attachment to the scene config."""

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (dict(acquisition_date="15-07-2021"), "YYYY-MM-DD"),
            (dict(bands=["B02", "B12"]), "Unsupported Sentinel-2 band"),
            (dict(bands=["B02", "B02"]), "Duplicate bands"),
            (dict(scl_exclude=[12]), "scl_exclude"),
            (dict(scl_exclude=[-1]), "scl_exclude"),
        ],
        ids=[
            "bad-date",
            "unknown-band",
            "duplicate-bands",
            "scl-class-above-11",
            "scl-class-negative",
        ],
    )
    def test_rejects_invalid_config(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _valid(**overrides)

    def test_scl_exclude_accepts_none_and_normalizes_duplicates(self):
        assert _valid(scl_exclude=None).scl_exclude is None
        assert _valid(scl_exclude=[9, 3, 9]).scl_exclude == [3, 9]

    def test_attaches_to_scene_config(self, make_minimal_config):
        cfg = _valid()
        scene = make_minimal_config(spectral_matching=cfg)
        assert scene.spectral_matching is not None
        assert scene.spectral_matching.landcover_classes == [30, 60]


class TestPipelineWiring:
    """Conditional resource wiring and the matched-materials sidecar context property."""

    def test_sentinel2_resource_registered_only_when_configured(
        self, make_minimal_config
    ):
        from s2gos_generator.core.pipeline import SceneGenerationPipeline

        without = SceneGenerationPipeline(make_minimal_config())
        deps_without = without.get_resource_dependencies()
        assert "target_sentinel2" not in deps_without
        assert "target_sentinel2" not in deps_without["target_texture"]

        with_spec = SceneGenerationPipeline(
            make_minimal_config(spectral_matching=_valid(landcover_classes=[30]))
        )
        deps_with = with_spec.get_resource_dependencies()
        assert deps_with["target_sentinel2"] == ["target_landcover"]
        assert "target_sentinel2" in deps_with["target_texture"]

    def test_matched_materials_property_reads_sidecar(
        self, make_minimal_config, tmp_path
    ):
        from s2gos_generator.core.context import SceneResourceContext

        ctx = SceneResourceContext(
            make_minimal_config(spectral_matching=_valid(landcover_classes=[30]))
        )

        # No sidecar -> empty.
        assert ctx.matched_materials == {}

        sidecar = tmp_path / "matched_materials.json"
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "materials": {"sand": {"type": "diffuse"}},
                    "material_indices": {"sand": 12},
                    "source_landcover_classes": [30],
                }
            )
        )
        ctx.assets.matched_materials_file = sidecar
        ctx._matched_materials = None  # reset lazy cache

        loaded = ctx.matched_materials
        assert loaded["materials"] == {"sand": {"type": "diffuse"}}
        assert loaded["material_indices"] == {"sand": 12}

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": 99, "materials": {}}))
        ctx.assets.matched_materials_file = bad
        ctx._matched_materials = None  # reset lazy cache
        assert ctx.matched_materials == {}
