# Acquisitions

Readers that turn a real mission product into simulation configuration, one module per mission.

Each `create_acquisition` returns a sensor that images the product's own pixels, with the sun and view
angles at the footprint's center. Raw sensor output stores the first row last, so reverse the rows to compare
it with the product. A Sentinel-2 footprint must be a whole number of pixels. A Sentinel-3 footprint is
rounded to whole pixels, because its spacing is measured.

A footprint must lie on real pixels. `fetch_tile_metadata` and `fetch_product_geometry` take its size and
return the closest product that holds it, by closest day, then non-time-critical over near-real-time, then
earliest start, trying cached products first. They warn when that means another day or a skipped closer
product, because the sun and view angles change. `read_tile_footprint` and `read_product_footprint` do the
check, and `create_acquisition` raises `FootprintOutsideProduct` if it fails. Products aren't stitched, so a
footprint across two files moves to another pass.

## Sentinel-2

::: s2gos_simulator.acquisitions.sentinel2.fetch_tile_metadata
::: s2gos_simulator.acquisitions.sentinel2.create_acquisition
::: s2gos_simulator.acquisitions.sentinel2.read_tile_metadata
::: s2gos_simulator.acquisitions.sentinel2.TileGeometry
::: s2gos_simulator.acquisitions.sentinel2.read_tile_footprint
::: s2gos_simulator.acquisitions.sentinel2.TileFootprint

## Sentinel-3

::: s2gos_simulator.acquisitions.sentinel3.fetch_product_geometry
::: s2gos_simulator.acquisitions.sentinel3.create_acquisition
::: s2gos_simulator.acquisitions.sentinel3.read_product_geometry
::: s2gos_simulator.acquisitions.sentinel3.ProductGeometry
::: s2gos_simulator.acquisitions.sentinel3.read_product_footprint
::: s2gos_simulator.acquisitions._grid.Grid

## Errors

::: s2gos_simulator.acquisitions._fetch.FootprintOutsideProduct
