"""BRF Processor for atmosphere-less BRF measurements.

BRF (Bidirectional Reflectance Factor) is computed from radiance measurements
without atmospheric interference, using TOA irradiance directly from simulation results.

This follows the same pattern as HDRFProcessor for consistency.
"""

import logging
from typing import Dict, List, Optional, Set

import numpy as np
import xarray as xr
from upath import UPath

from .config import BRFConfig, DirectionalIllumination

logger = logging.getLogger(__name__)


class BRFProcessor:
    """Processor for BRF measurements without atmosphere.

    BRF = (pi * L) / (E * cos(SZA))
    """

    def __init__(self, backend):
        """Initialize BRF processor.

        Args:
            backend: EradiateBackend instance
        """
        self.backend = backend
        self.simulation_config = backend.simulation_config

    def get_brf_configs(self) -> List[BRFConfig]:
        """Get all BRF measurement configurations.

        Returns:
            List of BRFConfig instances
        """
        return [
            m for m in self.simulation_config.measurements if isinstance(m, BRFConfig)
        ]

    def get_brf_sensor_ids(self) -> Set[str]:
        """Get sensor IDs required for BRF workflow.

        Returns:
            Set of sensor IDs used by BRF measurements
        """
        return {
            config.radiance_sensor_id
            for config in self.get_brf_configs()
            if config.radiance_sensor_id
        }

    def compute_brf(
        self,
        radiance_dataset: xr.Dataset,
        brf_config: BRFConfig,
    ) -> xr.Dataset:
        """Compute BRF from radiance dataset.

        BRF = (pi * L) / (E * cos(SZA))

        Args:
            radiance_dataset: Dataset with 'radiance' and 'irradiance' variables
            brf_config: BRF configuration

        Returns:
            Dataset containing BRF values

        Raises:
            ValueError: If illumination is not directional or required data missing
        """
        from .backends.eradiate.reflectance_computation import compute_brf

        # Get radiance and irradiance from dataset
        L = radiance_dataset["radiance"]
        E_toa = radiance_dataset["irradiance"]

        # Get cos(SZA) from illumination config
        illumination = self.simulation_config.illumination
        if not isinstance(illumination, DirectionalIllumination):
            raise ValueError(
                f"BRF requires DirectionalIllumination, got {type(illumination)}"
            )

        sza_rad = np.deg2rad(illumination.zenith)
        cos_sza = np.cos(sza_rad)

        # Compute BRF
        brf_ds = compute_brf(
            radiance=L,
            toa_irradiance=E_toa,
            cos_sza=cos_sza,
            measurement_id=brf_config.id,
            extra_attrs={
                "solar_zenith_deg": illumination.zenith,
                "atmosphere": "none",
            },
        )

        return brf_ds

    def compute_all_brf_measurements(
        self,
        sensor_results: Dict[str, xr.Dataset],
        output_dir: Optional[UPath] = None,
    ) -> Dict[str, xr.Dataset]:
        """Compute BRF for all configured BRF measurements.

        Args:
            sensor_results: Dictionary of sensor results keyed by sensor ID
            output_dir: Optional directory to save results

        Returns:
            Dictionary of BRF datasets keyed by measurement ID
        """
        from s2gos_utils.io.paths import mkdir

        brf_results = {}
        brf_configs = self.get_brf_configs()

        if not brf_configs:
            logger.debug("No BRF measurements configured")
            return brf_results

        if output_dir:
            mkdir(output_dir)

        for config in brf_configs:
            radiance_ds = sensor_results.get(config.radiance_sensor_id)

            if radiance_ds is None:
                logger.warning(
                    f"Radiance dataset not found for BRF '{config.id}' "
                    f"(sensor_id: {config.radiance_sensor_id})"
                )
                continue

            try:
                brf_ds = self.compute_brf(radiance_ds, config)
                brf_results[config.id] = brf_ds
                logger.info(f"Computed BRF for '{config.id}'")

                # Save if output directory provided
                if output_dir:
                    output_path = (
                        output_dir / f"{self.simulation_config.name}_{config.id}.zarr"
                    )
                    brf_ds.attrs["id"] = config.id
                    brf_ds.to_zarr(output_path, mode="w")
                    logger.info(f"Saved BRF '{config.id}' to {output_path}")

            except Exception as e:
                logger.error(f"Failed to compute BRF for '{config.id}': {e}")

        logger.info(f"BRF computation complete: {len(brf_results)} measurements")
        return brf_results
