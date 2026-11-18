"""Search for the closest real product that holds a footprint."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterator, Mapping

logger = logging.getLogger(__name__)


class FootprintOutsideProduct(ValueError):
    """The footprint is not entirely covered by the product."""


def _sensing_start(name: str) -> str:
    return re.search(r"_(\d{8}T\d{6})_", name).group(1)


def _sensing_day(name: str) -> date:
    return datetime.strptime(_sensing_start(name), "%Y%m%dT%H%M%S").date()


def closest_fitting(
    cache: Path,
    anchor: date,
    check: Callable[[Path], object],
    search: Callable[[], Mapping[str, Callable[[], Path]]],
) -> Path:
    """Return the closest product that `check` accepts, cached products first.

    Products rank by closest day to `anchor`, then non-time-critical over
    near-real-time, then earliest start, all read from the product name. Rejected
    products stay in `cache`, so a rerun downloads nothing new and the order still holds
    for any footprint.

    Log a warning when the product is from another day or a closer one was rejected,
    because the sun and view angles then differ.

    Args:
        cache: Directory holding the downloaded ``.xml`` or ``.SEN3`` products.
        anchor: Requested day.
        check: Raises `FootprintOutsideProduct` when a product doesn't hold the
            footprint. Any other error stops the search.
        search: Asks the catalog, returning for each product name a function that
            downloads it into `cache`. Only called when no cached product fits.

    Returns:
        Local path to the product.

    Raises:
        RuntimeError: If no product holds the footprint.
    """

    def rank(name: str) -> tuple:
        days = abs((_sensing_day(name) - anchor).days)
        return days, "_NT_" not in name, _sensing_start(name), name

    def candidates() -> Iterator[Path]:
        cached = [path for path in cache.glob("*") if path.suffix in (".xml", ".SEN3")]
        yield from sorted(cached, key=lambda path: rank(path.name))
        downloads = search()
        for name in sorted(set(downloads) - {path.name for path in cached}, key=rank):
            yield downloads[name]()

    rejected = []
    for path in candidates():
        try:
            check(path)
        except FootprintOutsideProduct as error:
            logger.info("%s: %s", path.name, error)
            rejected.append(path.name)
            continue
        day = _sensing_day(path.name)
        if day != anchor or rejected:
            logger.warning(
                "Closest product to %s that holds the footprint: using the %s pass "
                "from %s. Closer products that don't hold it: %s",
                anchor,
                day,
                path,
                rejected or "none",
            )
        return path
    raise RuntimeError(f"No product near {anchor} holds the footprint: {rejected}")
