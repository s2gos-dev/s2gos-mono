# Acquisitions

Readers that turn a real mission product into simulation configuration, one module per mission.

Each `create_acquisition` returns a sensor that images the product's own pixel grid: its rotation, its
ground spacing and its pixel positions, at the product's height (the mean terrain height when
orthorectified, otherwise sea level standing in for the ellipsoid). A Sentinel-2 sensor carries the L1C
tile's projected `grid`. A Sentinel-3 sensor carries the swath-aligned grid fitted on the product's tie
points as `grid_rotation`, and logs the product pixel range it covers. The raw sensor output stores the
first row last, so once its rows are reversed its pixels are the product's.

## Sentinel-2

::: s2gos_simulator.acquisitions.sentinel2.fetch_tile_metadata
::: s2gos_simulator.acquisitions.sentinel2.create_acquisition
::: s2gos_simulator.acquisitions.sentinel2.read_tile_metadata

## Sentinel-3

::: s2gos_simulator.acquisitions.sentinel3.fetch_product_geometry
::: s2gos_simulator.acquisitions.sentinel3.create_acquisition
::: s2gos_simulator.acquisitions.sentinel3.read_product_geometry
::: s2gos_simulator.acquisitions.sentinel3.ProductGeometry
