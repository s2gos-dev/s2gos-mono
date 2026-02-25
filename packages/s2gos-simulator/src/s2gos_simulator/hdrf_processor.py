import logging
from typing import Dict, List

import xarray as xr
from upath import UPath

from .backends.eradiate.reflectance_computation import compute_hcrf, compute_hdrf
from .config import HCRFConfig, HDRFConfig
from .irradiance_processor import IrradianceProcessor

logger = logging.getLogger(__name__)


class HDRFProcessor:
    """Processor for HDRF and HCRF measurements"""

    def __init__(self, backend):
        self.backend = backend
        self.simulation_config = backend.simulation_config
        self.irradiance_processor = IrradianceProcessor(backend)

    def requires_hdrf(self) -> bool:
        """Check if any measurements require HDRF computation.

        Returns:
            True if HDRF computation is needed
        """
        from .config import HDRFConfig

        # Check if any sensor produces HDRF
        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and "hdrf" in sensor.produces:
                return True

        # Check if any measurement is an HDRF config
        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                return True

        return False

    def get_hdrf_measure_ids(self) -> List[str]:
        """Get list of measure IDs that produce HDRF.

        Returns:
            List of measure IDs that require HDRF computation
        """
        from .config import HDRFConfig

        hdrf_ids = []

        # Check sensors that produce HDRF
        for sensor in self.simulation_config.sensors:
            if hasattr(sensor, "produces") and "hdrf" in sensor.produces:
                hdrf_ids.append(sensor.id)

        # Check HDRF measurement configs
        for measurement in self.simulation_config.measurements:
            if isinstance(measurement, HDRFConfig):
                hdrf_id = self._get_hdrf_id(measurement)
                hdrf_ids.append(hdrf_id)

        return [x.replace(".", "_") for x in hdrf_ids]

    def _get_hdrf_id(self, measurement) -> str:
        """Get ID for an HDRF measurement.

        Args:
            measurement: HDRFConfig instance

        Returns:
            ID string for the measurement
        """
        return measurement.id

    def compute_h_reflectances(
        self,
        raw_results: Dict[str, xr.Dataset],
        output_dir: UPath,
    ) -> Dict[str, xr.Dataset]:
        """Compute derived measurements (HDRF, HCRF) using shared computation logic.

        Args:
            raw_results: Dictionary containing all raw sensor and irradiance results.
            output_dir: Directory for results (used by caller for saving).

        Returns:
            Dictionary of derived datasets keyed by the original measurement ID.
        """
        from s2gos_utils.io.paths import mkdir

        mkdir(output_dir)

        derived_results = {}

        # Identify all measurements that are derived types (HDRF or HCRF)
        derived_configs = [
            m
            for m in self.simulation_config.measurements
            if isinstance(m, (HDRFConfig, HCRFConfig))
        ]

        logger.info(
            f"Computing derived measurements for {len(derived_configs)} items..."
        )

        for config in derived_configs:
            meas_id = config.id
            rad_id = getattr(config, "radiance_sensor_id", None)
            irr_id = getattr(config, "irradiance_measurement_id", None)

            if isinstance(config, HDRFConfig):
                ref_type = "hdrf"
            elif isinstance(config, HCRFConfig):
                ref_type = "hcrf"
            else:
                continue

            if not rad_id or not irr_id:
                logger.warning(f"Skipping '{meas_id}': Missing linked sensor IDs.")
                continue

            if rad_id not in raw_results:
                logger.warning(
                    f"Skipping '{meas_id}': Missing radiance results '{rad_id}'."
                )
                continue
            if irr_id not in raw_results:
                logger.warning(
                    f"Skipping '{meas_id}': Missing irradiance results '{irr_id}'."
                )
                continue

            L_ds = raw_results[rad_id]
            E_ds = raw_results[irr_id]

            L_data = L_ds.get("radiance")
            E_data = E_ds.get("boa_irradiance")

            if L_data is None or E_data is None:
                logger.error(f"Missing data variables in datasets for {meas_id}")
                continue

            try:
                extra_attrs = {
                    "linked_radiance_id": rad_id,
                    "linked_irradiance_id": irr_id,
                }
                if ref_type == "hdrf":
                    result_ds = compute_hdrf(
                        radiance=L_data,
                        boa_irradiance=E_data,
                        measurement_id=meas_id,
                        extra_attrs=extra_attrs,
                    )
                else:
                    result_ds = compute_hcrf(
                        radiance=L_data,
                        boa_irradiance=E_data,
                        measurement_id=meas_id,
                        extra_attrs=extra_attrs,
                    )

                derived_results[meas_id] = result_ds

                logger.info(f"Computed {ref_type.upper()} for '{meas_id}'")

            except Exception as e:
                logger.error(f"Failed to compute {ref_type} for '{meas_id}': {e}")
                continue

        return derived_results
