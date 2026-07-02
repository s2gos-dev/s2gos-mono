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
    _utm_epsg,
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
        ],
        ids=["identical-scaled", "orthogonal"],
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
        assert ids == {"soil", "veg"}  # rpv + out-of-range dropped
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
        # Closest date wins on its own region; later date only fills the gap.
        assert np.all(composite.values[:, :, : NX // 2] == 1.0)
        assert np.all(composite.values[:, :, NX // 2 :] == 2.0)

    def test_degraded_returns_best_effort_without_raising(self):
        # Every candidate only ever covers the west half -> AOI can't be fully filled.
        layers = {d: _layer(float(d), "west") for d in range(5)}
        load_fn, loaded = _counting_loader(layers)

        composite, coverage = _accumulate_until_covered(
            order=list(range(5)), load_fn=load_fn, min_coverage=0.99, max_dates=3
        )

        assert abs(coverage - 0.5) < 1e-6
        assert np.isnan(composite.values[:, :, NX // 2 :]).all()
        assert len(loaded) <= 3

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

    def test_utm_epsg_picks_zone_and_hemisphere(self):
        # Greenwich-ish, northern hemisphere -> zone 31N (EPSG:32631).
        assert _utm_epsg(45.0, 3.0) == 32631
        # Same longitude band, southern hemisphere -> 32731.
        assert _utm_epsg(-45.0, 3.0) == 32731
        # Far-west longitude -> low zone number.
        assert _utm_epsg(0.0, -177.0) == 32601


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
        ],
        ids=["bad-date", "unknown-band", "duplicate-bands"],
    )
    def test_rejects_invalid_config(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _valid(**overrides)

    def test_attaches_to_scene_config(self, make_minimal_config):
        cfg = _valid()
        assert cfg.bands == BANDS
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
