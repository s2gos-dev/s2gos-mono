"""Irradiancemeter: irradiance on a flat collector via the reference disk technique.

A white Lambertian disk (ρ=1) lying in the collector plane reflects the radiance
L = E/π in every direction, E being the irradiance on that plane. A radiancemeter
looking at the disk along its normal therefore measures E = π·L, with the exact
cosine response of an ideal pyranometer, for any orientation of the collector.
"""

from dataclasses import replace
from typing import Dict, List

import numpy as np
import xarray as xr
from s2gos_utils.scene import SceneDescription

from .config import SpectralResponse


def insert_reference_disks(
    scene_description: SceneDescription, disks: List[Dict]
) -> SceneDescription:
    """Copy of a scene with white Lambertian disk objects added.

    Args:
        scene_description: Scene to copy
        disks: Disk objects (``type="disk"`` with ``center``, ``radius`` and an
            optional ``normal``)
    """
    return replace(
        scene_description, objects=[*disks, *(scene_description.objects or [])]
    )


def to_irradiance(dataset: xr.Dataset, sensor) -> xr.Dataset:
    """Irradiancemeter result from the radiance its radiancemeter saw on the disk.

    Returns a dataset with:

    - ``irradiance``: spectral irradiance on the collector, E = π·L (W m-2 nm-1)
    - ``band_irradiance``: irradiance integrated over the sensor's band SRF (W m-2),
      when Eradiate weighted the radiance by it (ckd mode with a band SRF)
    - ``toa_irradiance``: the solar irradiance at the top of the atmosphere, on a
      horizontal plane (Eradiate's ``irradiance``)
    - ``irradiance_var``: Monte Carlo variance of ``irradiance``, with the moment
      integrator
    """

    def point(name: str) -> xr.DataArray:
        data = dataset[name]
        return data.mean([d for d in data.dims if d != "w"], keep_attrs=True)

    result = {
        "irradiance": (np.pi * point("radiance")).assign_attrs(
            long_name="spectral irradiance on the collector",
            units="W m^-2 nm^-1",
        )
    }
    if "radiance_var" in dataset:
        result["irradiance_var"] = (np.pi**2 * point("radiance_var")).assign_attrs(
            units="W^2 m^-4 nm^-2"
        )
    srf = sensor.srf
    if (
        "radiance_srf" in dataset
        and isinstance(srf, SpectralResponse)
        and srf.type == "band"
    ):
        result["band_irradiance"] = (
            np.pi * point("radiance_srf") * srf.integral()
        ).assign_attrs(long_name="band irradiance on the collector", units="W m^-2")
    toa = "toa_irradiance" if "toa_irradiance" in dataset else "irradiance"
    if toa in dataset:
        result["toa_irradiance"] = point(toa).assign_attrs(
            long_name="top-of-atmosphere solar irradiance on a horizontal plane",
            units="W m^-2 nm^-1",
        )

    return xr.Dataset(result, attrs={**dataset.attrs, "instrument": "irradiancemeter"})
