"""Instrument-specific output products.

A product writer combines the per-band results of one satellite sensor into a single
store laid out like the mission's real product, named like it in the output directory.
Writers are picked by the sensor's `output_format`.
Sensors without one keep one dataset per band.
"""

from .sentinel2 import write_msi_l1c

PRODUCT_WRITERS = {"eopf": write_msi_l1c}
"""Product writer by sensor `output_format`."""

__all__ = ["PRODUCT_WRITERS", "write_msi_l1c"]
