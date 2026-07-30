"""The S2GOS free demo: browse precalculated results.

Pick a site, a season and an instrument; get back the result dataset, its image and its
metadata. **Nothing is simulated here.** Every result was rendered offline with Eradiate
and is looked up in the :data:`RESULTS` table below, which is the one place to edit when
a run lands or the results move (including to remote object storage).

:func:`plot_bands` and :func:`true_color` draw a banded result in a notebook.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from s2gos_utils.io import PathRef, exists, open_dataset, to_upath

from s2gos_apps.registry import registry

logger = logging.getLogger(__name__)

_DEFAULT_BASE = (
    Path(__file__).resolve().parents[3] / "example" / "precalculated_results"
)

RESULTS_BASE: str = os.environ.get("S2GOS_DEMO_RESULTS") or str(_DEFAULT_BASE)
RESULTS_CID: str | None = os.environ.get("S2GOS_DEMO_RESULTS_CID") or None

_Site = Literal["PNP", "Gobabeb", "Frascati", "Pisa", "Jaén"]
_Season = Literal["December", "June"]
_Instrument = Literal["msi", "hypstar", "RGB camera"]

RESULTS: dict[tuple[str, str, str], dict] = {
    # examples/gobabeb_rgb_demo.py:184
    ("gobabeb", "june", "rgb_camera"): {
        "dataset": "gobabeb/rgb_camera/gobabeb_sim_camera.zarr",
        "image": "gobabeb/rgb_camera/camera_rgb.png",
        "title": "Gobabeb - top-down RGB camera",
        "acquired": "2024-12-21T09:00",
    },
    ("gobabeb", "december", "msi"): {
        "bands": {
            "B2": "gobabeb/january/sim_output/gobabeb_sentinel2_s2-2.zarr",
            "B3": "gobabeb/january/sim_output/gobabeb_sentinel2_s2-3.zarr",
            "B4": "gobabeb/january/sim_output/gobabeb_sentinel2_s2-4.zarr",
            "B8": "gobabeb/january/sim_output/gobabeb_sentinel2_s2-8.zarr",
        },
        "title": "Gobabeb - Sentinel-2 MSI",
        "acquired": "2024-12-21T13:00",
    },
    ("gobabeb", "december", "hypstar"): {
        "dataset": "gobabeb/january/sim_output/hypstar.zarr",
        "title": "Gobabeb - Sentinel-2 MSI",
        "acquired": "2024-12-21T13:00",
    },
    ("pnp", "december", "msi"): {
        "bands": {
            "B2": "pnp/sim_output/pnp_sentinel2_s2-2.zarr",
            "B3": "pnp/sim_output/pnp_sentinel2_s2-3.zarr",
            "B4": "pnp/sim_output/pnp_sentinel2_s2-4.zarr",
            "B8": "pnp/sim_output/pnp_sentinel2_s2-8.zarr",
        },
        "title": "Patagonia NP - Sentinel-2 MSI",
        "acquired": "2024-12-21T14:00",
    },
    ("pnp", "june", "rgb_camera"): {
        "dataset": "pnp/rgb_camera/pnp_demo_june_camera.zarr",
        "image": "pnp/rgb_camera/camera_rgb.png",
        "title": "Patagonia NP - top-down RGB camera",
        "acquired": "2024-06-21T17:00",
    },
}


def resolve(relative: str) -> PathRef:
    return PathRef(str(to_upath(RESULTS_BASE) / relative), cid=RESULTS_CID)


def _key(value: str) -> str:
    """Fold one site/season/instrument to its lookup form.

    The form submits the display labels ("Gobabeb", "RGB camera") while the table is
    keyed however it was typed, so both sides go through this: a key differing only in
    case would otherwise miss at request time with nothing to hint at why.
    """
    return str(value).strip().casefold().replace(" ", "_").replace("-", "_")


def _build_index(table: dict | None = None) -> dict[tuple[str, str, str], dict]:
    """A results table re-keyed for lookup, rejecting rows that collapse together."""
    index: dict[tuple[str, str, str], dict] = {}
    for key, entry in (RESULTS if table is None else table).items():
        folded = tuple(_key(part) for part in key)
        if folded in index:
            # At import, so a broken table stops the server rather than one request.
            raise ValueError(
                f"RESULTS has two rows for {'/'.join(folded)}; keys must differ by more "
                "than case or spacing."
            )
        index[folded] = entry
    return index


_INDEX = _build_index()


def lookup(site: str, season: str, instrument: str) -> dict:
    """The :data:`RESULTS` entry for one combination, matched case-insensitively.

    Raises:
        ValueError: if the combination is not wired, listing those that are.
    """
    entry = _INDEX.get((_key(site), _key(season), _key(instrument)))
    if entry is None:
        wired = ", ".join(f"{s}/{e}/{i}" for s, e, i in sorted(_INDEX))
        raise ValueError(
            f"No precalculated result for {site}/{season}/{instrument}. "
            f"Available: {wired}."
        )
    return entry


def availability():
    """The :data:`RESULTS` table as a ``pandas.DataFrame``, for display in a notebook."""
    import pandas as pd

    return pd.DataFrame(
        [
            dict(zip(("site", "season", "instrument"), key))
            | {
                "acquired": entry["acquired"],
                "title": entry["title"],
                "has_image": bool(entry.get("image")),
            }
            for key, entry in sorted(RESULTS.items())
        ]
    )


@registry.process(
    id="free-demo",
    title="S2GOS Free Demo",
    outputs={
        "dataset": Field(description="Result dataset (Zarr), openable with xarray."),
        "image": Field(description="Rendered image, or null if the run has none."),
        "metadata": Field(description="What was requested, and what was served."),
    },
)
def free_demo(
    site: Annotated[
        _Site,
        Field(
            title="Area of interest",
            description="Site to show results for.",
            json_schema_extra={"x-ui-order": 10},
        ),
    ] = "Gobabeb",
    season: Annotated[
        _Season,
        Field(
            title="Season",
            description="Local season at the site.",
            json_schema_extra={"x-ui-order": 20},
        ),
    ] = "December",
    instrument: Annotated[
        _Instrument,
        Field(
            title="Instrument",
            description="Simulated instrument.",
            json_schema_extra={"x-ui-order": 30},
        ),
    ] = "msi",
) -> tuple[PathRef | dict, PathRef | None, dict]:
    """Serve one precalculated result.

    Returns:
        dataset: `PathRef` to the Zarr store, or `{band: PathRef}` for a banded run
            such as MSI -- one store per band, since the bands do not share a
            wavelength grid. Use `plot_bands`/`true_color` on those.
        image: `PathRef` to the rendered image, or None when the run has none.
        metadata: What was requested, what was served, and the store's own facts.

    Raises:
        ValueError: if the combination is not wired; the message lists those that are.
        FileNotFoundError: if a wired result is not present at `RESULTS_BASE`.
    """
    entry = lookup(site, season, instrument)

    bands = entry.get("bands")
    refs = (
        {band: _checked_ref(entry, path) for band, path in bands.items()}
        if bands
        else _checked_ref(entry, entry["dataset"])
    )

    image_ref: PathRef | None = resolve(entry["image"]) if entry.get("image") else None
    if image_ref is not None and _is_local(image_ref) and not exists(image_ref):
        logger.warning("Image missing, continuing without it: %s", image_ref.href)
        image_ref = None

    reference = (
        refs
        if isinstance(refs, PathRef)
        else refs.get("B4") or next(iter(refs.values()))
    )

    logger.info("Reading %s", reference.href)
    ds = open_dataset(reference)
    try:
        metadata = {
            "site": str(site),
            "season": str(season),
            "instrument": str(instrument),
            "title": entry["title"],
            "acquired": entry["acquired"],
            **_dataset_facts(ds),
            "dataset_href": reference.href,
            "image_href": image_ref.href if image_ref else None,
        }
    finally:
        ds.close()

    if bands:
        metadata["bands"] = {band: _band_centre(ref) for band, ref in refs.items()}

    return refs, image_ref, metadata


def _checked_ref(entry: dict, relative: str) -> PathRef:
    """Resolve one table path, insisting a local store actually exists."""
    ref = resolve(relative)
    if _is_local(ref) and not exists(ref):
        raise FileNotFoundError(
            f"'{entry['title']}' points at {ref.href!r}, which does not exist. "
            f"Check RESULTS_BASE (currently {RESULTS_BASE!r}), or set S2GOS_DEMO_RESULTS."
        )
    return ref


def _band_centre(ref) -> float:
    """A band's SRF-weighted centre wavelength in nm, from the store's ``w_srf``."""
    ds = open_dataset(ref)
    try:
        return float(np.asarray(ds["w_srf"]).ravel()[0])
    finally:
        ds.close()


def _is_local(ref) -> bool:
    """Whether ``exists()`` means anything here -- https answers False regardless (see
    the note in ``s2gos_utils.io.resolver``), so checking would spuriously fail."""
    return to_upath(ref).protocol not in ("http", "https")


def _dataset_facts(ds) -> dict:
    """Summarise an opened result store, for the ``metadata`` output."""
    wavelengths = np.asarray(ds["w"]).ravel() if "w" in ds else np.array([])
    return {
        "sensor_id": ds.attrs.get("sensor_id"),
        "created_at": ds.attrs.get("created_at"),
        "sza": _first(ds, "sza"),
        "saa": _first(ds, "saa"),
        "n_wavelengths": int(wavelengths.size),
        "wavelength_min_nm": float(wavelengths.min()) if wavelengths.size else None,
        "wavelength_max_nm": float(wavelengths.max()) if wavelengths.size else None,
        "variables": sorted(str(name) for name in ds.data_vars),
        "dimensions": {str(k): int(v) for k, v in ds.sizes.items()},
    }


def _first(ds, name: str) -> float | None:
    """First element of a coordinate, as a plain float (or None if absent)."""
    values = np.asarray(ds[name]).ravel() if name in ds else np.array([])
    return float(values[0]) if values.size else None


def _as_ref(value):
    """Accept a ``PathRef``, or an OGC Link (object or dict) as returned over HTTP.

    A job's results arrive as `Link` objects, which carry the path but are not path-like,
    so opening one directly raises `TypeError` deep inside fsspec.
    """
    if value is None or isinstance(value, PathRef):
        return value
    if isinstance(value, dict):
        return PathRef.model_validate(value)
    href = getattr(value, "href", None)
    return PathRef(str(href)) if href is not None else value


def _caption(metadata: dict) -> str:
    """One line of acquisition facts, for the foot of a figure.

    The bands come from ``metadata["bands"]`` rather than from the store's own spectral
    facts: those describe the reference band alone and would read as if the whole run
    were ten wavelengths of B4.
    """
    sza, saa = metadata.get("sza"), metadata.get("saa")
    bands = metadata.get("bands") or {}
    bits = [
        f"SZA {sza:.1f}°" if sza is not None else None,
        f"SAA {saa:.1f}°" if saa is not None else None,
        f"bands {', '.join(sorted(bands))} "
        f"({min(bands.values()):.0f}-{max(bands.values()):.0f} nm)"
        if bands
        else None,
        f"run {metadata['created_at'][:19]}" if metadata.get("created_at") else None,
    ]
    return "   |   ".join(bit for bit in bits if bit)


def _annotate(fig, metadata: dict, subtitle: str = ""):
    """Put the run's title above a figure and its acquisition facts below."""
    title = metadata.get("title", "Precalculated result")
    if metadata.get("acquired"):
        title += f" -- {metadata['acquired'].replace('T', ' ')} UTC"
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=12)
    caption = _caption(metadata)
    if caption:
        fig.supxlabel(caption, fontsize=8, color="#444444")


_BAND_COLOURS = {"B2": "#1f77b4", "B3": "#2ca02c", "B4": "#d62728", "B8": "#7f2704"}


def _band_refs(outputs: dict) -> dict:
    """The ``{band: ref}`` mapping, from a whole result or from the mapping itself.

    Both are natural things to reach for -- `plot_bands(outputs)` and
    `plot_bands(results.root["dataset"])` -- so accept either rather than making the
    caller remember which.
    """
    dataset = outputs["dataset"] if "dataset" in outputs else outputs
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError(
            "plot_bands() and true_color() need a banded result -- a {band: reference} "
            "mapping, as an msi run returns, or a whole result containing one. This one "
            "wires a single store, so there are no bands to plot."
        )
    return {band: _as_ref(ref) for band, ref in dataset.items()}


def _band_metadata(outputs: dict) -> dict:
    """The metadata that goes with a result, or an empty dict for a bare mapping."""
    return (outputs.get("metadata") or {}) if "dataset" in outputs else {}


def _band_maps(outputs: dict) -> list[tuple[str, float, "np.ndarray"]]:
    """``(band, centre_nm, brf_srf map)`` per band, in wavelength order."""
    centres = _band_metadata(outputs).get("bands") or {}
    maps = []
    for band, ref in _band_refs(outputs).items():
        ds = open_dataset(ref)
        try:
            centre = centres.get(band) or float(np.asarray(ds["w_srf"]).ravel()[0])
            values = np.asarray(ds["brf_srf"].squeeze(drop=True), dtype=float)
            maps.append((band, centre, _north_up(values)))
        finally:
            ds.close()
    return sorted(maps, key=lambda row: row[1])


def plot_bands(outputs: dict, figsize=None, columns: int = 2):
    """Draw the ``brf_srf`` map of every band, one panel per band.

    Args:
        outputs: A whole ``free-demo`` result, or just its ``{band: ref}`` mapping.
        figsize: Figure size in inches. Defaults to 5 inches per panel.
        columns: Panels per row.
    """
    import matplotlib.pyplot as plt

    metadata = _band_metadata(outputs)
    maps = _band_maps(outputs)

    ncols = min(columns, len(maps))
    nrows = -(-len(maps) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (5 * ncols, 5 * nrows),
        layout="constrained",
        squeeze=False,
    )

    low, high = np.nanpercentile(np.stack([values for _, _, values in maps]), [2, 98])

    for axis, (band, centre, values) in zip(axes.ravel(), maps):
        shown = axis.imshow(values, cmap="gray", vmin=low, vmax=high)
        axis.set_title(
            f"{band} ({centre:.0f} nm)   mean {np.nanmean(values):.3f}",
            fontsize=11,
            color=_BAND_COLOURS.get(band, "#333333"),
        )
    for axis in axes.ravel():
        axis.axis("off")

    fig.colorbar(shown, ax=axes, shrink=0.7, label="BRF (band-integrated)")
    _annotate(fig, metadata)


def _north_up(array):
    """Flip a ``y_index``-indexed map for display.

    Row 0 of a result grid is the *southern* edge while ``imshow`` draws row 0 at the
    top, so a map handed over as-is comes out upside down. Checked against the rendered
    PNGs: flipped correlates at r=0.92 (Gobabeb) and r=0.99 (PNP), unflipped at ~0.
    """
    return array[::-1]


# Sentinel Hub's "L1C True Color Optimized" evalscript, transcribed from
# https://custom-scripts.sentinel-hub.com/sentinel-2/l1c_optimized/
_MAX_R = 3.0  # max reflectance
_MID_R = 0.13
_SAT = 1.3
_GAMMA = 2.3
_G_OFF = 0.01
#: Minimum Rayleigh scattering removed before the tone curve, per band.
_RAYLEIGH = {"B4": 0.013, "B3": 0.024, "B2": 0.041}


def _adj(a, tx: float = _MID_R, ty: float = 1.0, max_c: float = _MAX_R):
    """Contrast enhancement with highlight compression (the script's ``adj``)."""
    ar = np.clip(a / max_c, 0.0, 1.0)
    return (
        ar
        * (ar * (tx / max_c + ty - 1.0) - ty)
        / (ar * (2.0 * tx / max_c - 1.0) - tx / max_c)
    )


def _adj_gamma(b):
    """The script's ``adjGamma``: gamma with a small offset, renormalised to [0, 1]."""
    off_pow = _G_OFF**_GAMMA
    off_range = (1.0 + _G_OFF) ** _GAMMA - off_pow
    return (np.power(b + _G_OFF, _GAMMA) - off_pow) / off_range


def _srgb(c):
    """Linear light to sRGB (the script's ``sRGB``)."""
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.clip(c, 0.0, None), 0.41666666666) - 0.055,
    )


def _composite(refs: dict):
    """Stack B4/B3/B2 ``brf_srf`` into a ``(y, x, 3)`` sRGB array in [0, 1].

    Applies Sentinel Hub's L1C-optimized rendering: Rayleigh removal, highlight-
    compressing contrast enhancement, offset gamma, saturation boost, sRGB transfer.

    Raises:
        ValueError: if any of the three visible bands is missing.
    """
    missing = [band for band in ("B4", "B3", "B2") if band not in refs]
    if missing:
        raise ValueError(
            f"A true-colour composite needs B4, B3 and B2; this result has "
            f"{sorted(refs)} and is missing {missing}."
        )

    channels = []
    for band in ("B4", "B3", "B2"):
        ds = open_dataset(refs[band])
        try:
            reflectance = np.asarray(ds["brf_srf"].squeeze(drop=True), dtype=float)
            channels.append(reflectance - _RAYLEIGH[band])
        finally:
            ds.close()

    rgb = _adj_gamma(_adj(_north_up(np.stack(channels, axis=-1))))
    # satEnh: pull each channel away from the pixel's own mean.
    average = rgb.mean(axis=-1, keepdims=True) * (1.0 - _SAT)
    rgb = np.clip(average + rgb * _SAT, 0.0, 1.0)
    return np.clip(_srgb(rgb), 0.0, 1.0)


def true_color(outputs: dict, figsize=(8, 8)):
    """Draw a true-colour composite of a banded result: B4/B3/B2 as red/green/blue,
    rendered with Sentinel Hub's L1C-optimized tone mapping.

    Draws into a new figure and returns nothing, for the reason given in `plot_bands`.

    Args:
        outputs: A whole ``free-demo`` result, or just its ``{band: ref}`` mapping.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt

    metadata = _band_metadata(outputs)
    rgb = _composite(_band_refs(outputs))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(rgb)
    ax.axis("off")
    _annotate(fig, metadata, "true colour (B4/B3/B2)")


__all__ = [
    "RESULTS",
    "availability",
    "free_demo",
    "lookup",
    "plot_bands",
    "resolve",
    "true_color",
]
