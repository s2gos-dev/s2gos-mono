"""Result processing and visualization for Eradiate backend."""

import logging
from typing import Any, Dict

import xarray as xr
from s2gos_utils.io.paths import mkdir
from upath import UPath

logger = logging.getLogger(__name__)


class ResultProcessor:
    """Processor for simulation results, outputs, and visualizations."""

    def __init__(self, simulation_config):
        """Initialize result processor.

        Args:
            simulation_config: SimulationConfig object
        """
        self.simulation_config = simulation_config

    def save_result(
        self,
        sensor_id: str,
        dataset: xr.Dataset,
        output_dir: UPath,
        result_type: str = "sensor",
    ) -> bool:
        """Save a sensor or measurement result.

        Args:
            sensor_id: Unique identifier for this result
            dataset: xarray Dataset to save
            output_dir: Output directory
            result_type: Type of result ("sensor", "irradiance", "derived")

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            output_dir = UPath(output_dir)
            mkdir(output_dir)
            sensor_output = (
                output_dir / f"{self.simulation_config.name}_{sensor_id}.zarr"
            )

            metadata = self.create_output_metadata(output_dir)
            dataset.attrs.update(metadata)
            dataset.attrs["sensor_id"] = sensor_id
            dataset.attrs["result_type"] = result_type

            dataset.to_zarr(sensor_output, mode="w")
            logger.info(f"Saved '{sensor_id}' → {sensor_output.name}")

            return True

        except Exception as e:
            logger.warning(f"Failed to save '{sensor_id}': {e}")
            logger.error(
                f"Failed to save sensor result '{sensor_id}': {e}", exc_info=True
            )
            return False

    def create_output_metadata(self, output_dir: UPath) -> Dict[str, Any]:
        """Create standardized metadata for output files.

        Args:
            output_dir: Output directory path

        Returns:
            Dictionary with metadata fields
        """
        return {
            "simulation_name": self.simulation_config.name,
            "description": self.simulation_config.description,
            "created_at": self.simulation_config.created_at.isoformat(),
            "backend": "eradiate",
            "output_dir": str(output_dir),
            "num_sensors": len(self.simulation_config.sensors),
            "num_measurements": len(self.simulation_config.measurements),
            "sensor_types": [
                s.platform_type.value for s in self.simulation_config.sensors
            ],
            "measurement_types": [m.type for m in self.simulation_config.measurements],
            "illumination_type": self.simulation_config.illumination.type,
        }

    def create_hdrf_visualizations(
        self, hdrf_results: Dict[str, xr.Dataset], output_dir: UPath
    ) -> None:
        """Create visualizations for HDRF results.

        Args:
            hdrf_results: Dictionary of HDRF datasets
            output_dir: Output directory for visualizations
        """
        import matplotlib.pyplot as plt

        vis_dir = output_dir / "hdrf_visualizations"
        mkdir(vis_dir)

        for measure_id, dataset in hdrf_results.items():
            try:
                hdrf_data = dataset["hdrf"]

                if "x_index" in hdrf_data.dims and "y_index" in hdrf_data.dims:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    if "w" in hdrf_data.dims and len(hdrf_data.w) > 1:
                        img_data = hdrf_data.isel(w=0).values
                    else:
                        img_data = hdrf_data.squeeze().values

                    im = axes[0].imshow(img_data, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[0].set_title(f"HDRF - {measure_id}")
                    axes[0].set_xlabel("X index")
                    axes[0].set_ylabel("Y index")
                    plt.colorbar(im, ax=axes[0], label="HDRF")

                    axes[1].hist(img_data.flatten(), bins=50, edgecolor="black")
                    axes[1].set_xlabel("HDRF")
                    axes[1].set_ylabel("Frequency")
                    axes[1].set_title("HDRF Distribution")
                    axes[1].axvline(
                        x=img_data.mean(), color="r", linestyle="--", label="Mean"
                    )
                    axes[1].legend()

                    plt.tight_layout()
                    output_file = (
                        vis_dir
                        / f"{self.simulation_config.name}_{measure_id}_hdrf_visualization.png"
                    )
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    logger.info(f"  Saved HDRF visualization: {output_file.name}")

                elif "w" in hdrf_data.dims:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    wavelengths = hdrf_data.w.values
                    hdrf_values = hdrf_data.values

                    ax.plot(wavelengths, hdrf_values, "b-", linewidth=2)
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("HDRF")
                    ax.set_title(f"Spectral HDRF - {measure_id}")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 1])

                    plt.tight_layout()
                    output_file = (
                        vis_dir
                        / f"{self.simulation_config.name}_{measure_id}_hdrf_spectrum.png"
                    )
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    logger.info(f"  Saved HDRF spectrum: {output_file.name}")

            except Exception as e:
                logger.warning(
                    f"  Warning: Could not create visualization for {measure_id}: {e}"
                )
