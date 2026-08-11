"""The S2GOS free demo: browse precalculated results."""

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field
from s2gos_utils.coordinates import CoordinateSystem
from s2gos_utils.io import PathRef, exists, open_dataset, to_upath
from shapely import to_wkt

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


def _msi(folder: str, prefix: str) -> dict:
    """The four MSI band stores of one run."""
    return {f"B{b}": f"{folder}/MSI/{prefix}_s2-{b}.zarr" for b in (2, 3, 4, 8)}


def _camera(folder: str) -> dict:
    """The store and rendered image of one camera run."""
    return {
        "dataset": f"{folder}/camera/aerial_camera.zarr",
        "image": f"{folder}/camera/aerial_camera_rgb.png",
    }


#: Every wired result, keyed by the exact (site, season, instrument) the form submits.
#: Paths are relative to RESULTS_BASE, laid out as ``<site>/<season>/<instrument>/``.
#: Acquisition times are the observation dates in ``examples/run.sh``.
RESULTS: dict[tuple[str, str, str], dict] = {
    ("Gobabeb", "December", "msi"): {
        "bands": _msi("gobabeb/december", "gobabeb_sentinel2"),
        "title": "Gobabeb - Sentinel-2 MSI",
        "acquired": "2024-12-21T13:00",
    },
    ("Gobabeb", "December", "hypstar"): {
        "dataset": "gobabeb/december/hypstar/hypstar_series_01_hcrf_series_01.zarr",
        "title": "Gobabeb - HYPSTAR HCRF",
        "acquired": "2024-12-21T13:00",
    },
    ("Gobabeb", "December", "RGB camera"): {
        **_camera("gobabeb/december"),
        "title": "Gobabeb - oblique aerial RGB camera",
        "acquired": "2024-12-21T13:00",
    },
    ("Jaén", "December", "msi"): {
        "bands": _msi("jaen/december", "jaen_sentinel2"),
        "title": "Jaén - Sentinel-2 MSI",
        "acquired": "2024-12-21T13:00",
    },
    ("Jaén", "December", "RGB camera"): {
        **_camera("jaen/december"),
        "title": "Jaén - oblique aerial RGB camera",
        "acquired": "2024-12-21T13:00",
    },
    ("PNP", "December", "msi"): {
        "bands": _msi("pnp/december", "pnp_sentinel2"),
        "title": "Patagonia NP - Sentinel-2 MSI",
        "acquired": "2024-12-21T15:00",
    },
    ("PNP", "December", "RGB camera"): {
        **_camera("pnp/december"),
        "title": "Patagonia NP - oblique aerial RGB camera",
        "acquired": "2024-12-21T15:00",
    },
    ("PNP", "June", "msi"): {
        "bands": _msi("pnp/june", "pnp_sentinel2"),
        "title": "Patagonia NP - Sentinel-2 MSI",
        "acquired": "2024-12-21T15:00",
    },
    ("PNP", "June", "RGB camera"): {
        **_camera("pnp/june"),
        "title": "Patagonia NP - oblique aerial RGB camera",
        "acquired": "2024-06-21T15:00",
    },
    ("Pisa", "June", "RGB camera"): {
        **_camera("pisa/june"),
        "title": "Pisa - oblique aerial RGB camera",
        "acquired": "2024-06-21T09:00",
    },
}


# Each site's area of interest, as ``(centre latitude, centre longitude, size in km)``.
SITE_AOI: dict[str, tuple[float, float, float]] = {
    "PNP": (-46.9097, -72.4500, 10.0),
    "Gobabeb": (-23.6015417, 15.1258696, 10.0),
    "Frascati": (41.8210, 12.5700, 20.0),
    "Pisa": (43.7320, 10.3500, 15.0),
    "Jaén": (37.78802, -3.778123, 10.0),
}


# The polygon the GUI's map shows for each site: the *same* square the scene was generated
# over, from the same function the generator derives it with.
AOI_WKT: dict[str, str] = {
    site: to_wkt(
        CoordinateSystem(lat, lon).create_scene_polygon(size_km),
        rounding_precision=6,
    )
    for site, (lat, lon, size_km) in SITE_AOI.items()
}


def resolve(relative: str, media_type: str | None = None) -> PathRef:
    """A table path against RESULTS_BASE, optionally tagged with its media type."""
    return PathRef(
        str(to_upath(RESULTS_BASE) / relative), cid=RESULTS_CID, type=media_type
    )


def lookup(site: str, season: str, instrument: str) -> dict:
    """The RESULTS entry for one combination.

    Raises:
        ValueError: if the combination is not wired, listing those that are.
    """
    entry = RESULTS.get((site, season, instrument))
    if entry is None:
        wired = ", ".join(f"{s}/{e}/{i}" for s, e, i in sorted(RESULTS))
        raise ValueError(
            f"No precalculated result for {site}/{season}/{instrument}. "
            f"Available: {wired}."
        )
    return entry


def _aoi_field(site: str) -> Any:
    """A read-only map showing one site's AOI, shown only while that site is picked."""
    return Field(
        title="Area of interest",
        description=f"The {site} AOI. Fixed in the free demo.",
        json_schema_extra={
            "format": "wkt",
            "x-ui-widget": "map",
            "x-ui-order": 15,  # between site (10) and season (20)
            "x-ui-disabled": True,
            "x-ui-visible": f"site === '{site}'",
        },
    )


@registry.process(
    id="free-demo",
    title="S2GOS Free Demo",
    outputs={
        "dataset": Field(description="Result dataset (Zarr), openable with xarray."),
        "image": Field(
            description="Rendered image. Only the RGB camera produces one; null otherwise."
        ),
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
    # One map per site, exactly one of them visible. They exist to give the form spatial
    # context; the body ignores what is submitted and answers from AOI_WKT instead.
    aoi_pnp: Annotated[str, _aoi_field("PNP")] = AOI_WKT["PNP"],
    aoi_gobabeb: Annotated[str, _aoi_field("Gobabeb")] = AOI_WKT["Gobabeb"],
    aoi_frascati: Annotated[str, _aoi_field("Frascati")] = AOI_WKT["Frascati"],
    aoi_pisa: Annotated[str, _aoi_field("Pisa")] = AOI_WKT["Pisa"],
    aoi_jaen: Annotated[str, _aoi_field("Jaén")] = AOI_WKT["Jaén"],
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
    """Serve precalculated results for varios AOIs, instruments and seasons."""
    entry = lookup(site, season, instrument)

    bands = entry.get("bands")
    refs = (
        {band: _checked_ref(entry, path) for band, path in bands.items()}
        if bands
        else _checked_ref(entry, entry["dataset"])
    )

    image_ref: PathRef | None = (
        resolve(entry["image"], "image/png") if entry.get("image") else None
    )
    if image_ref is not None and _is_local(image_ref) and not exists(image_ref):
        logger.warning("Image missing, continuing without it: %s", image_ref.href)
        image_ref = None

    # Facts common to every band are read from one store; B4 by preference, only so that
    # the same one is reported run to run.
    reference = (
        refs
        if isinstance(refs, PathRef)
        else refs.get("B4") or next(iter(refs.values()))
    )

    logger.info("Reading %s", reference.href)
    ds = open_dataset(reference)
    try:
        metadata = {
            "site": site,
            "season": season,
            "instrument": instrument,
            "aoi_wkt": AOI_WKT.get(site),
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


def _read(ref, *names) -> tuple:
    """Named variables from one store, as plain arrays, leaving nothing open."""
    ds = open_dataset(_as_ref(ref))
    try:
        return tuple(np.asarray(ds[name], dtype=float) for name in names)
    finally:
        ds.close()


def _band_centre(ref) -> float:
    """A band's SRF-weighted centre wavelength in nm, from the store's ``w_srf``."""
    return float(_read(ref, "w_srf")[0].ravel()[0])


def _is_local(ref) -> bool:
    """Whether ``exists()`` is meaningful for this ref: http(s) always answers False."""
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
    """Coerce to a ``PathRef``. A job's results arrive as OGC ``Link``s, object or dict,
    which carry the path but are not path-like."""
    if value is None or isinstance(value, PathRef):
        return value
    if isinstance(value, dict):
        return PathRef.model_validate(value)
    href = getattr(value, "href", None)
    return PathRef(str(href)) if href is not None else value


def _caption(metadata: dict) -> str:
    """One line of acquisition facts, for the foot of a figure."""
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


def _unwrap(outputs, name: str) -> tuple:
    """The named output and its metadata, given any of: the client's ``JobResults``,
    its ``.root`` dict, a plain ``free_demo()`` result dict, or just the one output
    named here.

    Every plotter takes any of these, so callers never need to remember which form a
    particular plotter wants.
    """
    root = getattr(outputs, "root", None)
    if isinstance(root, dict):
        outputs = root
    if isinstance(outputs, dict) and "metadata" in outputs:
        return outputs.get(name), outputs["metadata"]
    return outputs, {}


def _band_refs(dataset) -> dict:
    """Coerce a ``{band: ref}`` mapping, rejecting anything that is not banded."""
    if not isinstance(dataset, dict) or not dataset or "href" in dataset:
        raise ValueError(
            "plot_bands() and true_color() need a banded result: a {band: reference} "
            "mapping, as an msi run returns. This one wires a single store."
        )
    return {band: _as_ref(ref) for band, ref in dataset.items()}


def _band_maps(outputs) -> list[tuple[str, float, "np.ndarray"]]:
    """``(band, centre_nm, brf_srf map)`` per band, in wavelength order."""
    dataset, metadata = _unwrap(outputs, "dataset")
    centres = metadata.get("bands") or {}
    maps = []
    for band, ref in _band_refs(dataset).items():
        w_srf, brf = _read(ref, "w_srf", "brf_srf")
        centre = centres.get(band) or float(w_srf.ravel()[0])
        maps.append((band, centre, _north_up(brf.squeeze())))
    return sorted(maps, key=lambda row: row[1])


def plot_bands(outputs, figsize=None, columns: int = 2):
    """Draw the ``brf_srf`` map of every band, one panel per band.

    Args:
        outputs: A free-demo result -- JobResults, its .root dict, or just the dataset mapping.
        figsize: Figure size in inches. Defaults to 5 inches per panel.
        columns: Panels per row.
    """
    import matplotlib.pyplot as plt

    _, metadata = _unwrap(outputs, "dataset")
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
    """Flip a ``y_index``-indexed map for display: row 0 is the southern edge of the
    grid, while ``imshow`` draws row 0 at the top."""
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

    channels = [
        _read(refs[band], "brf_srf")[0].squeeze() - _RAYLEIGH[band]
        for band in ("B4", "B3", "B2")
    ]

    rgb = _adj(_north_up(np.stack(channels, axis=-1)))
    # adjGamma: gamma with a small offset, renormalised to [0, 1].
    off_pow = _G_OFF**_GAMMA
    rgb = (np.power(rgb + _G_OFF, _GAMMA) - off_pow) / (
        (1.0 + _G_OFF) ** _GAMMA - off_pow
    )
    # satEnh: pull each channel away from the pixel's own mean.
    rgb = np.clip(
        rgb.mean(axis=-1, keepdims=True) * (1.0 - _SAT) + rgb * _SAT, 0.0, 1.0
    )
    # sRGB: linear light to display. The clip inside the original transfer function is
    # dropped -- the line above already left every value in [0, 1].
    display = np.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 0.41666666666) - 0.055
    )
    return np.clip(display, 0.0, 1.0)


def true_color(outputs, figsize=(8, 8)):
    """Draw a banded result as a true-colour composite: B4/B3/B2 as red/green/blue.

    Args:
        outputs: A free-demo result -- JobResults, its .root dict, or just the dataset mapping.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt

    dataset, metadata = _unwrap(outputs, "dataset")
    rgb = _composite(_band_refs(dataset))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(rgb)
    ax.axis("off")
    _annotate(fig, metadata, "true colour (B4/B3/B2)")


def _valid_hcrf(hcrf):
    """Blank samples above 1.0. In a saturated water-vapour band the surface irradiance
    goes to zero and ``HCRF = pi * L / E_boa`` runs away, reaching 222 at Gobabeb; NaN
    breaks the drawn line there rather than flattening the rest of the spectrum."""
    return np.where(hcrf > 1.0, np.nan, hcrf)


def plot_hypstar(outputs, figsize=(9, 4.5)):
    """Draw a HYPSTAR result: hemispherical-conical reflectance against wavelength.

    Args:
        outputs: A free-demo result -- JobResults, its .root dict, or just the dataset reference.
        figsize: Figure size in inches.
    """
    import matplotlib.pyplot as plt

    dataset, metadata = _unwrap(outputs, "dataset")
    if isinstance(dataset, dict) and "href" not in dataset:
        raise ValueError(
            f"plot_hypstar() needs a single-store result; this one is banded "
            f"({sorted(dataset)}). Use plot_bands() or true_color() instead."
        )
    w, hcrf = _read(_as_ref(dataset), "w", "hcrf")

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(w.ravel(), _valid_hcrf(hcrf.squeeze()), color="#1f77b4", lw=1.2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("HCRF")
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim((0.0, 0.8))
    ax.grid(alpha=0.3, linestyle="--")
    _annotate(fig, metadata, "HCRF spectrum")


def show_rgb_image(outputs, figsize=(12, 6.75)):
    """Draw the rendered image of an RGB camera result.

    Args:
        outputs: A free-demo result -- JobResults, its .root dict, or just the image reference.
        figsize: Figure size in inches, 16:9 to match the camera film.
    """
    import matplotlib.pyplot as plt

    image, metadata = _unwrap(outputs, "image")
    ref = _as_ref(image)
    if ref is None:
        raise ValueError(
            "show_rgb_image() needs a whole free-demo result or its image reference, "
            "and only the RGB camera runs render an image."
        )

    with to_upath(ref).open("rb") as stream:
        picture = plt.imread(stream, format="png")

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(picture)
    ax.axis("off")
    _annotate(fig, metadata, "rendered image")


__all__ = [
    "AOI_WKT",
    "RESULTS",
    "SITE_AOI",
    "free_demo",
    "lookup",
    "plot_bands",
    "plot_hypstar",
    "resolve",
    "show_rgb_image",
    "true_color",
]
