"""Reflectance computation for BRF, HDRF, HCRF, and BHR."""

from typing import Literal, Optional

import numpy as np
import xarray as xr

ReflectanceType = Literal["hdrf", "hcrf", "brf", "bhr"]


def align_and_broadcast_datasets(
    radiance: xr.DataArray,
    reference: xr.DataArray,
    wavelength_dim: str = "w",
    fill_value: float | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align wavelength dimensions and broadcast reference to match radiance shape.

    This handles the common case where radiance and reference datasets may have:
    - Different wavelength grids (interpolates reference to match radiance)
    - Different dimensions (broadcasts reference to match radiance)

    Args:
        radiance: Radiance DataArray (L)
        reference: Reference DataArray (E for irradiance, or similar)
        wavelength_dim: Name of the wavelength dimension (default: "w")
        fill_value: Value to use for out-of-range wavelengths. If None (default),
            raises ValueError when radiance extends beyond the reference range.

    Returns:
        Tuple of (radiance, aligned_reference) DataArrays
    """
    if wavelength_dim in radiance.dims and wavelength_dim in reference.dims:
        if not np.array_equal(
            radiance[wavelength_dim].values, reference[wavelength_dim].values
        ):
            rad_min, rad_max = (
                float(radiance[wavelength_dim].min()),
                float(radiance[wavelength_dim].max()),
            )
            ref_min, ref_max = (
                float(reference[wavelength_dim].min()),
                float(reference[wavelength_dim].max()),
            )

            out_of_range = rad_min < ref_min or rad_max > ref_max

            if out_of_range:
                if fill_value is None:
                    raise ValueError(
                        f"Radiance wavelength range [{rad_min}, {rad_max}] nm extends beyond "
                        f"reference range [{ref_min}, {ref_max}] nm. "
                        f"Provide a fill_value for out-of-range wavelengths, or ensure the "
                        f"reference covers the full radiance range."
                    )
                interp_kwargs = {"fill_value": fill_value, "bounds_error": False}
            else:
                interp_kwargs = {}

            reference = reference.interp(
                {wavelength_dim: radiance[wavelength_dim]},
                method="linear",
                kwargs=interp_kwargs,
            )

    if set(radiance.dims) - set(reference.dims):
        reference = reference.broadcast_like(radiance)

    return radiance, reference


def compute_brf(
    radiance: xr.DataArray,
    toa_irradiance: xr.DataArray,
    cos_sza: float,
    measurement_id: str,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Compute BRF: BRF = (π × L) / (E_toa × cos(SZA)).
    No atmosphere — uses TOA irradiance directly.
    """
    L, E = align_and_broadcast_datasets(radiance, toa_irradiance)
    result = (np.pi * L) / (E * cos_sza)
    attrs = {
        "measurement_type": "brf",
        "measurement_id": measurement_id,
        "units": "dimensionless",
        "formula": "BRF = (pi * L) / (E_toa * cos(SZA))",
        "cos_sza": float(cos_sza),
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.Dataset({"brf": result}, attrs=attrs)


def _compute_h_reflectance(
    reflectance_type: Literal["hdrf", "hcrf"],
    radiance: xr.DataArray,
    boa_irradiance: xr.DataArray,
    measurement_id: str,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Shared implementation for HDRF and HCRF (same formula, different geometry)."""
    L, E = align_and_broadcast_datasets(radiance, boa_irradiance)
    result = (np.pi * L) / E
    attrs = {
        "measurement_type": reflectance_type,
        "measurement_id": measurement_id,
        "units": "dimensionless",
        "formula": f"{reflectance_type.upper()} = (pi * L) / E_boa",
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.Dataset({reflectance_type: result}, attrs=attrs)


def compute_hdrf(
    radiance: xr.DataArray,
    boa_irradiance: xr.DataArray,
    measurement_id: str,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Compute HDRF: HDRF = (π × L) / E_boa.
    Hemispherical-Directional — single viewing direction, atmosphere present.
    """
    return _compute_h_reflectance(
        "hdrf", radiance, boa_irradiance, measurement_id, extra_attrs
    )


def compute_hcrf(
    radiance: xr.DataArray,
    boa_irradiance: xr.DataArray,
    measurement_id: str,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Compute HCRF: HCRF = (π × L) / E_boa.
    Hemispherical-Conical — conical FOV averaged, atmosphere present.
    Same formula as HDRF; distinction is in measurement geometry.
    """
    return _compute_h_reflectance(
        "hcrf", radiance, boa_irradiance, measurement_id, extra_attrs
    )


def compute_bhr(
    surface_radiosity: xr.DataArray,
    reference_radiosity: xr.DataArray,
    measurement_id: str,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Compute BHR: BHR = J_surface / J_reference.
    Both quantities are radiosity [W/m²]; no π factor needed.
    Reference is a white Lambertian disk (ρ=1.0).
    """
    J, J_ref = align_and_broadcast_datasets(surface_radiosity, reference_radiosity)
    result = J / J_ref
    attrs = {
        "measurement_type": "bhr",
        "measurement_id": measurement_id,
        "units": "dimensionless",
        "formula": "BHR = J_surface / J_reference",
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return xr.Dataset({"bhr": result}, attrs=attrs)
