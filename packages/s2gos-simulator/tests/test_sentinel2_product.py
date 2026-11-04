from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from s2gos_utils.coordinates import CoordinateSystem
from upath import UPath

from s2gos_simulator.config import (
    DirectionalIllumination,
    ProjectedGrid,
    SatelliteSensor,
)
from s2gos_simulator.config.viewing import AngularViewing
from s2gos_simulator.products.sentinel2 import (
    MAX_REFLECTANCE,
    SATURATED_PIXEL,
    write_msi_l1c,
)

LAT, LON = -23.6, 15.12
GRID = ProjectedGrid(crs="EPSG:32733", origin=(499980.0, 7400020.0))


def _band_result(sensor, band, value=None):
    """Fake mpdistant result whose value is 0.01 × the (south-row-0) y_index, or ``value``."""
    nx, ny = sensor.band_film_resolution(band)
    rows = np.broadcast_to(0.01 * np.arange(ny, dtype=float), (nx, ny))
    if value is not None:
        rows = np.full((nx, ny), value)

    irradiance = [[1.5 * np.cos(np.radians(30.0))]]
    return xr.Dataset(
        {
            "brf_srf": (("x_index", "y_index"), rows),
            "irradiance_srf": (("sza", "saa"), irradiance),
        }
    )


def _write(tmp_path, bands, value=None, illumination=None, **kwargs):
    sensor = SatelliteSensor(
        id="s2",
        instrument="sentinel_2a-msi",
        bands=bands,
        viewing=AngularViewing(zenith=3.0, azimuth=0.0),
        target_center_lat=LAT,
        target_center_lon=LON,
        target_size_km=1.2,
        output_format="eopf",
        **kwargs,
    )
    write_msi_l1c(
        sensor,
        {band: _band_result(sensor, band, value) for band in sensor.bands},
        illumination or DirectionalIllumination(zenith=30.0, azimuth=90.0),
        {"simulation_name": "sim"},
        UPath(tmp_path),
        "sim",
    )
    (path,) = UPath(tmp_path).glob("*.zarr")
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
    assert "l1c_quicklook" not in product["quality"].children
    assert list(reflectance["r10m"].data_vars) == ["b04"]
    assert list(reflectance["r20m"].data_vars) == ["b8a"]
    assert reflectance["r60m"]["b01"].shape == (20, 20)
    assert reflectance["r10m"]["b04"].encoding["dtype"] == np.uint16


def test_rows_are_north_up(product):
    b04 = product["measurements/reflectance/r10m"]["b04"]

    assert b04.y[0] > b04.y[-1]
    assert b04.isel(y=0).values == pytest.approx(1.19, abs=1e-4)


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

    grid = geometry["viewing_incidence_angles"]
    assert grid.dims == ("band", "detector", "angle", "y", "x")
    assert grid.shape[:3] == (3, 1, 2)
    assert (grid.sel(band="b8a", angle="azimuth") == 270.0).all()
    assert (geometry["sun_angles"].sel(angle="zenith") == 30.0).all()


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
    assert (b04["eo:common_name"], b04["eo:full_width_half_max"]) == ("red", 0.030)
    assert b04["eo:solar_illumination"] == pytest.approx(1500.0)
    assert props["bands"][2]["raster:spatial_resolution"] == "60"
    assert "reflectance_correction_factor" not in str(product.attrs["other_metadata"])
    west, south, east, north = stac["bbox"]
    assert west < LON < east and south < LAT < north


def test_grid_product_on_tile_lattices(tmp_path):
    product = _write(tmp_path, ["4", "1"], grid=GRID)
    b04 = product["measurements/reflectance/r10m"]["b04"]
    b01 = product["measurements/reflectance/r60m"]["b01"]
    geometry = product["conditions/geometry"]
    x0, y0 = GRID.origin
    xmin, ymax = float(b01.x[0]) - 30.0, float(b01.y[0]) + 30.0

    assert ((xmin - x0) % 60.0, (y0 - ymax) % 60.0) == (0.0, 0.0)
    assert (float(b04.x[0]) - 5.0, float(b04.y[0]) + 5.0) == (xmin, ymax)
    assert ((geometry.x - x0) % 5000.0 == 0).all()
    assert ((y0 - geometry.y) % 5000.0 == 0).all()
    assert geometry.x[0] <= xmin and geometry.y[0] >= ymax
    assert product.attrs["stac_discovery"]["properties"]["proj:code"] == "EPSG:32733"


def test_sensor_without_grid_written_in_its_utm_zone(product):
    b04 = product["measurements/reflectance/r10m"]["b04"]

    assert product.attrs["stac_discovery"]["properties"]["proj:code"] == "EPSG:32733"
    assert product.attrs["other_metadata"]["horizontal_crs_code"] == "EPSG:32733"
    assert (float(b04.x[0]) - 5.0) % 60.0 == 0.0
    assert (float(b04.y[0]) + 5.0) % 60.0 == 0.0


def test_reflectance_stored_as_l1c_counts(tmp_path):
    _write(tmp_path, ["4"], value=0.2534)
    (path,) = UPath(tmp_path).glob("*.zarr")
    raw = xr.open_datatree(str(path), engine="zarr", mask_and_scale=False)

    counts = raw["measurements/reflectance/r10m"]["b04"]
    assert counts.dtype == np.uint16
    assert (counts == 3534).all()


def test_clipped_pixels_flagged_saturated(tmp_path):
    product = _write(tmp_path, ["4", "1"], value=7.0)
    b04 = product["measurements/reflectance/r10m"]["b04"]

    assert b04.values == pytest.approx(MAX_REFLECTANCE, abs=1e-4)
    assert (product["quality/mask/r10m"]["b04"] == SATURATED_PIXEL).all()
    assert (product["quality/mask/r60m"]["b01"] == SATURATED_PIXEL).all()


def test_classification_mask_on_tile_60m_lattice_without_60m_band(tmp_path):
    product = _write(tmp_path, ["4"], grid=GRID)
    b04 = product["measurements/reflectance/r10m"]["b04"]
    classification = product["conditions/mask/l1c_classification/r60m"]["b00"]
    x0, y0 = GRID.origin
    xmin, ymax = float(classification.x[0]) - 30.0, float(classification.y[0]) + 30.0

    assert ((xmin - x0) % 60.0, (y0 - ymax) % 60.0) == (0.0, 0.0)
    assert xmin <= float(b04.x[0]) - 5.0 and ymax >= float(b04.y[0]) + 5.0
    assert float(classification.x[-1]) + 30.0 >= float(b04.x[-1]) + 5.0
    assert float(classification.y[-1]) - 30.0 <= float(b04.y[-1]) - 5.0


def test_grid_requires_native_pixels():
    with pytest.raises(ValueError, match="native GSD"):
        SatelliteSensor(
            instrument="sentinel_2a-msi",
            bands=["4"],
            viewing=AngularViewing(zenith=0.0, azimuth=0.0),
            film_resolution=(100, 100),
            target_center_lat=LAT,
            target_center_lon=LON,
            target_size_km=1.2,
            grid=GRID,
            output_format="eopf",
        )


def test_grid_pixel_centres_follow_tile_pixels():
    """At 50° N, 2.9° east of the central meridian, grid north is 2.2° east of north."""
    lat, lon = 50.0, 5.9
    grid = ProjectedGrid(crs="EPSG:32631", origin=(600000.0, 5600040.0))
    sensor = SatelliteSensor(
        instrument="sentinel_2a-msi",
        bands=["4"],
        viewing=AngularViewing(zenith=0.0, azimuth=0.0),
        target_center_lat=lat,
        target_center_lon=lon,
        target_size_km=10.0,
        grid=grid,
        output_format="eopf",
    )
    coords = CoordinateSystem(lat, lon)
    footprint = sensor.target_rectangle(coords)
    tile = sensor.grid_bounds()
    to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    nx, ny = sensor.band_film_resolution("4")

    assert footprint.rotation == pytest.approx(-2.2, abs=0.05)
    for row, col in [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)]:
        pixel_lon, pixel_lat = to_lonlat.transform(
            tile["xmin"] + (col + 0.5) * 10.0, tile["ymax"] - (row + 0.5) * 10.0
        )
        assert footprint.pixel_center(row, col, (nx, ny)) == pytest.approx(
            coords.latlon_to_scene(pixel_lat, pixel_lon), abs=0.5
        )


def test_masks_describe_a_clear_single_detector_scene(product):
    footprint = product["conditions/mask/detector_footprint/r20m"]["b8a"]
    quality = product["quality/mask/r10m"]["b04"]
    classification = product["conditions/mask/l1c_classification/r60m"]["b00"]

    assert footprint.dtype == np.uint8 and (footprint == 1).all()
    assert footprint.attrs["flag_meanings"][1] == "d01"
    assert quality.shape == (120, 120) and (quality == 0).all()
    assert (quality.x == product["measurements/reflectance/r10m"].x).all()
    assert classification.shape == (20, 20) and (classification == 0).all()


def test_store_named_like_an_l1c_product(tmp_path):
    sensed = datetime(2024, 6, 22, 9, 17, 15, tzinfo=timezone.utc)
    illumination = DirectionalIllumination(
        zenith=30.0, azimuth=90.0, irradiance_datetime=sensed
    )
    product = _write(tmp_path, ["4"], illumination=illumination)

    assert (UPath(tmp_path) / "S2A_MSIL1C_20240622T091715_sim_s2.zarr").exists()
    assert product.attrs["stac_discovery"]["id"] == "S2A_MSIL1C_20240622T091715_sim_s2"


def _route_bands(tmp_path, monkeypatch, writer):
    """Run the backend's measure loop on fake results of an 'eopf' sensor's two bands,
    with ``writer`` as the 'eopf' product writer."""
    from types import SimpleNamespace

    from s2gos_simulator.backends.eradiate import backend as backend_module
    from s2gos_simulator.config import SimulationConfig

    sensor = SatelliteSensor(
        id="s2",
        instrument="sentinel_2a-msi",
        bands=["2", "3"],
        viewing=AngularViewing(zenith=3.0, azimuth=0.0),
        target_center_lat=LAT,
        target_center_lon=LON,
        target_size_km=0.06,
        output_format="eopf",
    )
    results = {}
    for band in sensor.bands:
        result = _band_result(sensor, band)
        results[sensor.band_measure_id(band)] = result.assign(radiance=result.brf_srf)
    experiment = SimpleNamespace(
        measures=[SimpleNamespace(id=measure_id) for measure_id in results],
        results=results,
    )
    backend = backend_module.EradiateBackend(
        SimulationConfig(
            name="sim",
            illumination=DirectionalIllumination(zenith=30.0, azimuth=90.0),
            sensors=[sensor],
        )
    )
    monkeypatch.setattr(backend, "create_experiment", lambda *a, **k: experiment)
    monkeypatch.setattr(
        backend_module, "eradiate", SimpleNamespace(run=lambda *a, **k: None)
    )
    monkeypatch.setitem(backend_module.PRODUCT_WRITERS, "eopf", writer)
    return backend._run_standard_simulation(None, None, UPath(tmp_path))


def test_backend_hands_every_band_to_the_product_writer(tmp_path, monkeypatch):
    received = {}

    def writer(sensor, band_results, *args):
        received.update(band_results)
        return "product"

    saved = _route_bands(tmp_path, monkeypatch, writer)

    assert sorted(received) == ["2", "3"]
    assert saved == {"s2": "product"}
    assert not list(UPath(tmp_path).glob("sim_s2_[23].zarr"))


def test_backend_saves_the_bands_separately_when_the_writer_fails(
    tmp_path, monkeypatch
):
    def writer(*args):
        raise RuntimeError("broken writer")

    saved = _route_bands(tmp_path, monkeypatch, writer)

    assert sorted(saved) == ["s2_2", "s2_3"]
    assert (UPath(tmp_path) / "sim_s2_2.zarr").exists()
    assert (UPath(tmp_path) / "sim_s2_3.zarr").exists()
