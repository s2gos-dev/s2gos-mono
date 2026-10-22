from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from s2gos_simulator.config.geometry import view_geometry_from_tile_metadata

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


class TestViewGeometryFromTileMetadata:
    def test_interpolates_at_point_in_eradiate_convention(self, tmp_path):
        zenith = [[2 + c + 0.5 * r for c in range(3)] for r in range(3)]
        azimuth = [[100.0] * 3 for _ in range(3)]
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml", [("1", zenith, azimuth)], band_id=8
        )

        view = view_geometry_from_tile_metadata(path, "8a", *_latlon_at(0.5, 1.25))

        assert view.zenith == pytest.approx(3.5, abs=1e-3)
        assert view.azimuth == pytest.approx(350.0, abs=1e-6)

    def test_merges_detectors_across_a_seam(self, tmp_path):
        zenith = [[5.0] * 3 for _ in range(3)]
        left = [[90.0, 90.0, np.nan] for _ in range(3)]
        right = [[np.nan, np.nan, 110.0] for _ in range(3)]
        path = _write_mtd_tl(
            tmp_path / "MTD_TL.xml", [("1", zenith, left), ("2", zenith, right)]
        )

        view = view_geometry_from_tile_metadata(path, "2", *_latlon_at(1.0, 1.5))

        mean_direction_zenith = np.degrees(
            np.arctan(np.tan(np.radians(5.0)) * np.cos(np.radians(10.0)))
        )
        assert view.zenith == pytest.approx(mean_direction_zenith, abs=1e-6)
        assert (90.0 - view.azimuth) % 360.0 == pytest.approx(100.0, abs=1e-6)

    def test_point_outside_swath_raises(self, tmp_path):
        zenith = [[5.0, np.nan, np.nan] for _ in range(3)]
        azimuth = [[100.0, np.nan, np.nan] for _ in range(3)]
        path = _write_mtd_tl(tmp_path / "MTD_TL.xml", [("1", zenith, azimuth)])

        with pytest.raises(ValueError, match="outside the imaged swath"):
            view_geometry_from_tile_metadata(path, "2", *_latlon_at(1.0, 1.5))
