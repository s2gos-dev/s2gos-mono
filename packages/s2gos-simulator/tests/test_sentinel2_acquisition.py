from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from s2gos_simulator.acquisitions.sentinel2 import (
    create_acquisition,
    fetch_tile_metadata,
    read_tile_metadata,
)
from s2gos_simulator.config.illumination import SolarPosition

_ULX, _ULY, _STEP = 499980.0, 7400020.0, 5000.0
_UTM_TO_LATLON = Transformer.from_crs("EPSG:32733", "EPSG:4326", always_xy=True)


def _latlon_at(row: float, col: float) -> tuple[float, float]:
    lon, lat = _UTM_TO_LATLON.transform(_ULX + col * _STEP, _ULY - row * _STEP)
    return lat, lon


def _grid_xml(values) -> str:
    rows = "".join(f"<VALUES>{' '.join(map(str, row))}</VALUES>" for row in values)
    return (
        f'<COL_STEP unit="m">{_STEP:.0f}</COL_STEP>'
        f'<ROW_STEP unit="m">{_STEP:.0f}</ROW_STEP>'
        f"<Values_List>{rows}</Values_List>"
    )


def _write_mtd_tl(path: Path, angle_grids, band_id: int = 1) -> Path:
    """angle_grids: (detector_id, zenith_grid, azimuth_grid) tuples for band_id."""
    grids = "".join(
        f'<Viewing_Incidence_Angles_Grids bandId="{band_id}" detectorId="{det}">'
        f"<Zenith>{_grid_xml(zenith)}</Zenith><Azimuth>{_grid_xml(azimuth)}</Azimuth>"
        "</Viewing_Incidence_Angles_Grids>"
        for det, zenith, azimuth in angle_grids
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_Tile_ID xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd">
  <n1:General_Info>
    <TILE_ID>S2A_OPER_MSI_L1C_TL_2APS_20220504T120001_A035839_T33KWP_N05.10</TILE_ID>
    <SENSING_TIME>2022-05-04T09:17:15.123Z</SENSING_TIME>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_CODE>EPSG:32733</HORIZONTAL_CS_CODE>
      <Geoposition resolution="10"><ULX>{_ULX:.0f}</ULX><ULY>{_ULY:.0f}</ULY></Geoposition>
    </Tile_Geocoding>
    <Tile_Angles>{grids}</Tile_Angles>
  </n1:Geometric_Info>
</n1:Level-1C_Tile_ID>
"""
    )
    return path


class TestReadTileMetadata:
    def test_interpolates_at_point_in_eradiate_convention(self, tmp_path):
        zenith = [[2 + c + 0.5 * r for c in range(3)] for r in range(3)]
        azimuth = [[100.0] * 3 for _ in range(3)]
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml", [("1", zenith, azimuth)], band_id=8
        )

        time, instrument, views = read_tile_metadata(
            path, *_latlon_at(0.5, 1.25), ["8a"]
        )
        view = views["8a"]

        assert time == datetime(2022, 5, 4, 9, 17, 15, 123000, tzinfo=timezone.utc)
        assert instrument == "sentinel_2a-msi"
        assert view.zenith == pytest.approx(3.5, abs=1e-3)
        assert view.azimuth == pytest.approx(350.0, abs=1e-6)

    def test_averages_overlapping_detectors(self, tmp_path):
        zenith = [[5.0] * 3 for _ in range(3)]
        left = [[90.0, 90.0, np.nan] for _ in range(3)]
        right = [[np.nan, 110.0, 110.0] for _ in range(3)]
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml", [("1", zenith, left), ("2", zenith, right)]
        )

        view = read_tile_metadata(path, *_latlon_at(1.0, 1.5), ["2"])[2]["2"]

        def horizontal(azimuth):
            return np.array([np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))])

        expected = 0.5 * (horizontal(90.0) + horizontal(110.0)) / 2 + 0.5 * horizontal(
            110.0
        )
        expected_zenith = np.degrees(
            np.arctan(np.tan(np.radians(5.0)) * np.linalg.norm(expected))
        )
        expected_azimuth = np.degrees(np.arctan2(*expected)) % 360.0
        assert view.zenith == pytest.approx(expected_zenith, abs=1e-6)
        assert (90.0 - view.azimuth) % 360.0 == pytest.approx(
            expected_azimuth, abs=1e-6
        )

    def test_point_outside_swath_raises(self, tmp_path):
        zenith = [[5.0, np.nan, np.nan] for _ in range(3)]
        azimuth = [[100.0, np.nan, np.nan] for _ in range(3)]
        path = _write_mtd_tl(tmp_path / "MTD_TL.xml", [("1", zenith, azimuth)])

        with pytest.raises(ValueError, match="outside the imaged swath"):
            read_tile_metadata(path, *_latlon_at(1.0, 1.5), ["2"])


class TestCreateAcquisition:
    def test_sun_and_view_come_from_the_tile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "s2gos_simulator.config.illumination.solar_position",
            lambda time, lat, lon: SolarPosition(30.0, 100.0, 1.496e8, time),
        )
        zenith = [[2 + c + 0.5 * r for c in range(3)] for r in range(3)]
        azimuth = [[100.0] * 3 for _ in range(3)]
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml", [("1", zenith, azimuth)], band_id=8
        )

        illumination, sensor = create_acquisition(
            path, *_latlon_at(0.5, 1.25), ["8a"], target_size_km=1.0
        )

        assert illumination.irradiance_datetime == datetime(
            2022, 5, 4, 9, 17, 15, 123000, tzinfo=timezone.utc
        )
        assert sensor.spec.platform == "sentinel-2a"
        assert sensor.band_viewing_for("8a").zenith == pytest.approx(3.5, abs=1e-3)
        assert sensor.band_viewing_for("8a").azimuth == pytest.approx(350.0, abs=1e-6)


class TestFetchTileMetadata:
    def test_cache_hit_warns_when_sensed_on_another_day(self, tmp_path, caplog):
        lat, lon = _latlon_at(1.0, 1.0)
        cached = _write_mtd_tl(
            tmp_path / f"MTD_TL_site_{lat:.4f}_{lon:.4f}_20220506.xml", []
        )

        with caplog.at_level("WARNING"):
            path = fetch_tile_metadata("2022-05-06", lat, lon, tmp_path, "site")

        assert path == cached
        assert "using the 2022-05-04 pass" in caplog.text
