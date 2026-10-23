"""Instrument-specific output products.

A product writer combines the per-band results of one satellite sensor into a single
store laid out like the mission's real product. Writers are picked by instrument name.
Instruments without a writer keep one dataset per band.
"""

from .sentinel2 import write_msi_l1c

PRODUCT_WRITERS = {"msi": write_msi_l1c}
"""Product writer by instrument name."""

__all__ = ["PRODUCT_WRITERS", "write_msi_l1c"]
