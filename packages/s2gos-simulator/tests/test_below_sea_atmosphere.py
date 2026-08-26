"""Tests for below-sea-level atmosphere handling.

Covers the ground_altitude resolution and thermoprops extension added to fix
Invalid-sample warnings on sub-MSL scenes (e.g. Dead Sea). The key invariant
under test is that ABOVE-sea-level scenes are unaffected (bit-identical), while
below-sea scenes get a correctly-floored profile that passes eradiate's
geometry/atmosphere compatibility check.
"""

from unittest.mock import patch

import numpy as np
import pytest
from s2gos_utils.scene.description import SceneDescription

from s2gos_simulator.backends.eradiate.atmosphere_builder import (
    AtmosphereBuilder,
    _build_thermoprops,
    _resolve_ground_altitude,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _scene(toa=75000.0):
    """Minimal in-memory SceneDescription with an atmosphere dict."""
    return SceneDescription(
        name="test_scene",
        resolution_m=30.0,
        location={"center_lat": 0.0, "center_lon": 0.0},
        atmosphere={"toa": toa, "type": "molecular"},
    )


PATCH_TARGET = "s2gos_simulator.terrain_query.TerrainQuery"


def _patch_min_elevation(value):
    """Patch TerrainQuery so .min_elevation() returns `value` without a DEM."""
    return patch(PATCH_TARGET, **{"return_value.min_elevation.return_value": value})


# --------------------------------------------------------------------------- #
# _resolve_ground_altitude
# --------------------------------------------------------------------------- #


def test_above_sea_clamps_to_zero():
    """Positive terrain -> ground_altitude is exactly 0.0 (bit-identical path)."""
    with _patch_min_elevation(350.0):
        z0 = _resolve_ground_altitude(_scene(), "unused", 75000.0)
    assert z0 == 0.0


def test_at_sea_level_stays_zero():
    """z_min == 0 -> 0.0 (no negative padding introduced)."""
    with _patch_min_elevation(0.0):
        z0 = _resolve_ground_altitude(_scene(), "unused", 75000.0)
    assert z0 == 0.0


def test_below_sea_applies_ten_percent_pad():
    """Dead Sea case: z_min=-427 -> z0 = -427 - 0.10*427 = -469.7."""
    with _patch_min_elevation(-427.0):
        z0 = _resolve_ground_altitude(_scene(), "unused", 75000.0)
    assert z0 == pytest.approx(-469.7)


def test_floor_check_raises_when_below_cuboid_floor():
    """z0 at/below -0.01*toa must raise (would leave the medium shape)."""
    # toa=75000 -> floor=-750. z_min=-800 -> z0 = -800 - 80 = -880 <= -750 -> raise.
    with _patch_min_elevation(-800.0):
        with pytest.raises(ValueError, match="cuboid floor|atmosphere_shape"):
            _resolve_ground_altitude(_scene(), "unused", 75000.0)


def test_floor_check_scales_with_toa():
    """Same deep terrain PASSES against the larger mono toa (120 km -> floor -1200)."""
    with _patch_min_elevation(-800.0):
        z0 = _resolve_ground_altitude(_scene(toa=120000.0), "unused", 120000.0)
    assert z0 == pytest.approx(-880.0)  # -800 - 80, clears -1200 floor


def test_no_dem_falls_back_to_zero():
    """min_elevation() == None -> 0.0 (safe default, warns, no crash)."""
    with _patch_min_elevation(None):
        z0 = _resolve_ground_altitude(_scene(), "unused", 75000.0)
    assert z0 == 0.0


# --------------------------------------------------------------------------- #
# _build_thermoprops
# --------------------------------------------------------------------------- #


def test_thermoprops_above_sea_returns_legacy_dict():
    """ground_altitude None -> legacy {identifier, z} dict, floored at 0."""
    tp = _build_thermoprops(
        identifier="afgl_1986-us_standard",
        altitude_max=120000.0,
        altitude_step=1000.0,
        ground_altitude=None,
    )
    assert isinstance(tp, dict)
    assert tp["identifier"] == "afgl_1986-us_standard"
    z = tp["z"].magnitude if hasattr(tp["z"], "magnitude") else np.asarray(tp["z"])
    assert float(np.min(z)) == 0.0


def test_thermoprops_zero_ground_returns_legacy_dict():
    """ground_altitude == 0.0 also takes the legacy dict path (not a Dataset)."""
    tp = _build_thermoprops(
        identifier="afgl_1986-us_standard",
        altitude_max=120000.0,
        altitude_step=1000.0,
        ground_altitude=0.0,
    )
    assert isinstance(tp, dict)


def test_thermoprops_below_sea_returns_extended_dataset():
    """ground_altitude < 0 -> xr.Dataset pinned at z0, top >= altitude_max."""
    import xarray as xr

    z0 = -500.0
    tp = _build_thermoprops(
        identifier="afgl_1986-us_standard",
        altitude_max=120000.0,
        altitude_step=1000.0,
        ground_altitude=z0,
    )
    assert isinstance(tp, xr.Dataset)
    # z coord is in km per joseki convention; convert to m for the check.
    z_units = tp["z"].attrs.get("units", "m")
    scale = 1000.0 if z_units == "km" else 1.0
    z_m = tp["z"].values * scale
    assert float(np.min(z_m)) == pytest.approx(z0, abs=1.0)
    assert float(np.max(z_m)) >= 120000.0
    # No NaN anywhere — the grid is capped at the native profile top.
    p = tp["p"].values
    assert bool(np.all(np.isfinite(p))), "extended profile must contain no NaN"
    # pressure strictly decreasing with altitude across the whole profile
    assert bool(np.all(np.diff(p) < 0))
    # top lands on the native max (120 km), not above it
    assert float(np.max(z_m)) == pytest.approx(120000.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Stash isolation across builder reuse
# --------------------------------------------------------------------------- #


def test_ground_altitude_stash_no_leak_across_scenes():
    """A second config call without ground_altitude must not read a stale stash."""
    ab = AtmosphereBuilder()
    # First scene: a below-sea value stashed. Use a homogeneous atmosphere so the
    # call returns without needing a real molecular profile build.
    sd_first = SceneDescription(
        name="s1",
        resolution_m=30.0,
        location={"center_lat": 0.0, "center_lon": 0.0},
        atmosphere={"toa": 75000.0, "type": "homogeneous", "boa": 0.0},
    )
    try:
        ab.create_atmosphere_from_config(sd_first, ground_altitude=-469.7)
    except Exception:
        pass  # we only care that the stash was set before any downstream work
    assert ab._ground_altitude == -469.7

    # Second scene: no ground_altitude passed -> stash must reset to None.
    sd_second = SceneDescription(
        name="s2",
        resolution_m=30.0,
        location={"center_lat": 0.0, "center_lon": 0.0},
        atmosphere={"toa": 75000.0, "type": "homogeneous", "boa": 0.0},
    )
    try:
        ab.create_atmosphere_from_config(sd_second)
    except Exception:
        pass
    assert ab._ground_altitude is None


# --------------------------------------------------------------------------- #
# End-to-end geometry/atmosphere compatibility (the real gate)
# --------------------------------------------------------------------------- #


def test_below_sea_passes_check_geometry_atmosphere():
    """The extended profile must satisfy eradiate's compatibility gate at z0."""
    eradiate = pytest.importorskip("eradiate")
    eradiate.set_mode("ckd")
    from eradiate.experiments._helpers import check_geometry_atmosphere
    from eradiate.scenes.atmosphere import MolecularAtmosphere
    from eradiate.scenes.geometry import PlaneParallelGeometry
    from eradiate.units import unit_registry as ureg

    z0 = -500.0
    toa = 120000.0
    tp = _build_thermoprops(
        identifier="afgl_1986-us_standard",
        altitude_max=toa,
        altitude_step=1000.0,
        ground_altitude=z0,
    )
    atm = MolecularAtmosphere(
        thermoprops=tp,
        absorption_data="monotropa",
        has_absorption=True,
        has_scattering=True,
    )
    geom = PlaneParallelGeometry(
        toa_altitude=toa * ureg.m,
        ground_altitude=z0 * ureg.m,
    )
    # Raises on incompatibility; passing == no exception.
    check_geometry_atmosphere(geom, atm)
