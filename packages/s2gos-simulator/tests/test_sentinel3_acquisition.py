from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from s2gos_simulator.acquisitions.sentinel3 import (
    create_acquisition,
    fetch_product_geometry,
    read_product_geometry,
)

_OLCI = "S3B_OL_1_EFR____20240611T083306_20240611T083606_20240921T163355_0179_094_078_3420_MAR_R_NT_004.SEN3"
_SLSTR = "S3B_SL_1_RBT____20240611T083306_20240611T083606_20240611T225222_0179_094_078_3420_PS2_O_NT_004.SEN3"
_ROWS = _COLS = 4
_TIMES = np.array(
    [
        np.datetime64("2024-06-11T08:33:30") + np.timedelta64(r, "s")
        for r in range(_ROWS)
    ]
).astype("datetime64[ns]")


def _latlon_at(row: float, col: float) -> tuple[float, float]:
    """A descending swath on SLSTR's 1 km by 16 km tie grid, north-up."""
    return 10.0 - 0.009 * row, 20.0 + 0.1459 * col


_MANIFEST = "".join(
    f'<slstr:{view}ImageSize grid="{grid}"><sentinel3:startOffset>{start}'
    f"</sentinel3:startOffset><sentinel3:trackOffset>{track}"
    "</sentinel3:trackOffset></slstr:{view}ImageSize>"
    for view in ("nadir", "oblique")
    for grid, start, track in (("Tie Points", 100, 64), ("0.5 km stripe A", 199, 2100))
)


def _grid(values) -> xr.DataArray:
    values = np.broadcast_to(np.asarray(values, dtype=float), (_ROWS, _COLS))
    return xr.DataArray(values, dims=("rows", "columns"))


def _write(product_dir: Path, file: str, **variables) -> None:
    xr.Dataset(variables).to_netcdf(product_dir / f"{file}.nc")


def _write_product(root: Path, name: str, views: dict, coordinates, time) -> Path:
    """views: file -> {variable: grid values}."""
    product_dir = root / name
    product_dir.mkdir(parents=True)
    rows, cols = np.mgrid[0:_ROWS, 0:_COLS]
    lat, lon = _latlon_at(rows, cols)
    _write(
        product_dir,
        coordinates[0],
        **{coordinates[1]: _grid(lat), coordinates[2]: _grid(lon)},
    )
    _write(product_dir, time[0], **{time[1]: xr.DataArray(_TIMES, dims="rows")})
    for file, variables in views.items():
        _write(product_dir, file, **{k: _grid(v) for k, v in variables.items()})
    (product_dir / "xfdumanifest.xml").write_text(f"<manifest>{_MANIFEST}</manifest>")
    return product_dir


def _olci(root: Path, oaa=100.0) -> Path:
    return _write_product(
        root,
        _OLCI,
        {"tie_geometries": {"OZA": 20.0, "OAA": oaa, "SZA": 30.0, "SAA": 45.0}},
        ("tie_geo_coordinates", "latitude", "longitude"),
        ("time_coordinates", "time_stamp"),
    )


def _slstr(root: Path) -> Path:
    oblique = np.where(np.arange(_COLS) < 2, 55.0, np.nan)
    return _write_product(
        root,
        _SLSTR,
        {
            f"geometry_t{v}": {
                f"sat_zenith_t{v}": zenith,
                f"sat_azimuth_t{v}": 180.0,
                f"solar_zenith_t{v}": 30.0,
                f"solar_azimuth_t{v}": 45.0,
            }
            for v, zenith in (("n", 20.0), ("o", oblique))
        },
        ("geodetic_tx", "latitude_tx", "longitude_tx"),
        ("time_in", "time_stamp_i"),
    )


class TestReadProductGeometry:
    def test_interpolates_at_point_in_eradiate_convention(self, tmp_path):
        geometry = read_product_geometry(_olci(tmp_path), *_latlon_at(1.2, 1.5))

        assert geometry.time == datetime(2024, 6, 11, 8, 33, 31, tzinfo=timezone.utc)
        assert geometry.instrument == "sentinel_3b-olci"
        assert geometry.viewing.zenith == pytest.approx(20.0)
        assert geometry.viewing.azimuth == pytest.approx(350.0)
        assert (geometry.sun_zenith, geometry.sun_azimuth) == pytest.approx(
            (30.0, 45.0)
        )

    def test_azimuth_wraps_through_north(self, tmp_path):
        oaa = np.where(np.arange(_COLS) < 2, 358.0, 2.0)

        geometry = read_product_geometry(_olci(tmp_path, oaa), *_latlon_at(1.0, 1.5))

        assert geometry.viewing.azimuth == pytest.approx(90.0)
        expected = np.degrees(
            np.arctan(np.tan(np.radians(20.0)) * np.cos(np.radians(2.0)))
        )
        assert geometry.viewing.zenith == pytest.approx(expected)

    def test_point_outside_oblique_swath_raises(self, tmp_path):
        product_dir = _slstr(tmp_path)

        assert read_product_geometry(
            product_dir, *_latlon_at(1.0, 2.5)
        ).viewing.zenith == pytest.approx(20.0)
        with pytest.raises(ValueError, match="outside the oblique swath"):
            read_product_geometry(product_dir, *_latlon_at(1.0, 2.5), view="oblique")


class TestCreateAcquisition:
    def test_sun_and_view_come_from_the_product(self, tmp_path, caplog):
        with caplog.at_level("INFO"):
            illumination, sensor = create_acquisition(
                _slstr(tmp_path),
                *_latlon_at(1.1, 0.55),
                ["1", "5"],
                1.0,
                view="oblique",
            )

        assert (illumination.zenith, illumination.azimuth) == pytest.approx(
            (30.0, 45.0)
        )
        assert illumination.irradiance_datetime == datetime(
            2024, 6, 11, 8, 33, 31, tzinfo=timezone.utc
        )
        assert [sensor.band_srf(band) for band in sensor.bands] == [
            "sentinel_3b-slstr-1",
            "sentinel_3b-slstr-5",
        ]
        assert sensor.film_resolution == (2, 2)
        assert sensor.orthorectified
        assert sensor.grid_rotation == pytest.approx(0.0, abs=0.1)
        assert "product rows 3:5 and columns 69:71" in caplog.text
        assert sensor.viewing.zenith == pytest.approx(55.0)
        assert sensor.viewing.azimuth == pytest.approx(270.0)


class TestFetchProductGeometry:
    def test_cache_hit_warns_when_sensed_on_another_day(self, tmp_path, caplog):
        lat, lon = _latlon_at(1.0, 1.0)
        cached = _olci(tmp_path / f"S3_olci_site_{lat:.4f}_{lon:.4f}_20240613")

        with caplog.at_level("WARNING"):
            path = fetch_product_geometry(
                "olci", "2024-06-13", lat, lon, tmp_path, "site"
            )

        assert path == cached
        assert "using the 2024-06-11 pass" in caplog.text
