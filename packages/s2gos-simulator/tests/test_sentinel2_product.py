from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from upath import UPath

from s2gos_simulator.config import (
    DirectionalIllumination,
    SatelliteSensor,
)
from s2gos_simulator.config.viewing import AngularViewing
from s2gos_simulator.products.sentinel2 import write_msi_l1c

LAT, LON = -23.6, 15.12


def _band_result(sensor, band, value=None):
    """Fake mpdistant result whose value is the (south-row-0) y_index, or ``value``."""
    nx, ny = sensor.band_film_resolution(band)
    rows = np.broadcast_to(np.arange(ny, dtype=float), (nx, ny))
    if value is not None:
        rows = np.full((nx, ny), value)

    irradiance = [[1.5 * np.cos(np.radians(30.0))]]
    return xr.Dataset(
        {
            "brf_srf": (("x_index", "y_index"), rows),
            "irradiance_srf": (("sza", "saa"), irradiance),
        }
    )


def _write(tmp_path, bands, value=None, **kwargs):
    sensor = SatelliteSensor(
        id="s2",
        instrument="sentinel_2a-msi",
        bands=bands,
        viewing=AngularViewing(zenith=3.0, azimuth=0.0),
        target_center_lat=LAT,
        target_center_lon=LON,
        target_size_km=1.2,
        **kwargs,
    )
    path = UPath(tmp_path) / "sim_s2.zarr"
    write_msi_l1c(
        sensor,
        {band: _band_result(sensor, band, value) for band in sensor.bands},
        SimpleNamespace(location={"center_lat": LAT, "center_lon": LON}),
        DirectionalIllumination(zenith=30.0, azimuth=90.0),
        {"simulation_name": "sim"},
        path,
    )
    return xr.open_datatree(str(path), engine="zarr")


@pytest.fixture
def product(tmp_path):
    return _write(
        tmp_path,
        ["4", "8a", "1"],
        band_viewing={"8a": AngularViewing(zenith=5.0, azimuth=180.0)},
    )


def test_bands_grouped_by_native_resolution(product):
    reflectance = product["measurements/reflectance"]

    assert set(reflectance.children) == {"r10m", "r20m", "r60m"}
    assert "quality" not in product.children
    assert list(reflectance["r10m"].data_vars) == ["b04"]
    assert list(reflectance["r20m"].data_vars) == ["b8a"]
    assert reflectance["r60m"]["b01"].shape == (20, 20)
    assert reflectance["r10m"]["b04"].dtype == np.float32


def test_rows_are_north_up(product):
    b04 = product["measurements/reflectance/r10m"]["b04"]

    assert b04.y[0] > b04.y[-1]
    assert b04.isel(y=0).values == pytest.approx(119.0)


def test_bands_carry_their_grid(product):
    reflectance = product["measurements/reflectance"]
    b04, b01 = reflectance["r10m"]["b04"], reflectance["r60m"]["b01"]
    sx, _, x0, _, sy, y0, *_ = b01.attrs["proj:transform"]

    assert b04.attrs["proj:shape"] == [120, 120]
    assert b01.attrs["proj:shape"] == [20, 20]
    assert (sx, sy) == (60.0, -60.0)
    assert b01.x[0] == x0 + sx / 2 and b01.y[0] == y0 + sy / 2
    assert b01.attrs["proj:transform"][2::3] == b04.attrs["proj:transform"][2::3]
    assert b01.attrs["proj:bbox"] == b04.attrs["proj:bbox"]
    assert (
        b04.attrs["proj:wkt2"]
        == product.attrs["stac_discovery"]["properties"]["proj:wkt2"]
    )


def test_angles_in_sentinel2_convention(product):
    geometry = product["conditions/geometry"]
    viewing = geometry["mean_viewing_incidence_angles"]

    assert geometry["mean_sun_angles"].values.tolist() == [30.0, 0.0]
    assert viewing.sel(band="b04").values.tolist() == [3.0, 90.0]
    assert viewing.sel(band="b8a").values.tolist() == [5.0, 270.0]


def test_quicklook_from_rgb_bands(tmp_path):
    tci = _write(tmp_path, ["2", "3", "4"], value=0.1)["quality/l1c_quicklook/r10m"][
        "tci"
    ]

    assert tci.dtype == np.uint8
    assert tci.dims == ("band", "y", "x")
    assert tci.shape == (3, 120, 120)
    assert (tci.values == 71).all()


def test_stac_discovery(product):
    stac = product.attrs["stac_discovery"]

    props = stac["properties"]
    b04 = props["bands"][0]

    assert props["platform"] == "sentinel-2a"
    assert props["product:type"] == "S02MSIL1C"
    assert [b["name"] for b in props["bands"]] == ["b04", "b8a", "b01"]
    assert (b04["common_name"], b04["full_width_half_max"]) == ("red", 0.030)
    assert b04["solar_illumination"] == pytest.approx(1500.0)
    assert "reflectance_correction_factor" not in str(product.attrs["other_metadata"])
    west, south, east, north = stac["bbox"]
    assert west < LON < east and south < LAT < north
