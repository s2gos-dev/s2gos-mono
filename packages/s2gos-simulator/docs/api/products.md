# Products

Writers that combine the per-band results of a satellite sensor into one store laid out like
the mission's real product, one module per mission. A sensor asks for one with its
`output_format` (e.g. `output_format="eopf"` for a Sentinel-2 MSI sensor), which picks the writer
from `s2gos_simulator.products.PRODUCT_WRITERS`. Sensors without it keep one dataset per band.

## Sentinel-2

::: s2gos_simulator.products.sentinel2
