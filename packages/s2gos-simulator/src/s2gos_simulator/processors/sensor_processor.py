import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class SensorProcessor:
    """Unified sensor processor for all sensor types.

    Pipeline is driven entirely by config data (sensor.srf and
    sensor.post_processing). No instrument-specific branches.
    """

    def __init__(self, simulation_config):
        """Initialize sensor processor.

        Args:
            simulation_config: SimulationConfig containing sensor definitions
        """
        self.simulation_config = simulation_config

    def process_sensor_result(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply sensor-specific post-processing to a single dataset.

        This is the main entry point for all sensor post-processing.

        Args:
            dataset: Raw simulation result dataset
            sensor: Sensor configuration object

        Returns:
            Post-processed dataset with SRF convolution, averaging, etc.
        """
        return self._apply_post_processing(dataset, sensor)

    def _apply_post_processing(self, dataset: xr.Dataset, sensor) -> xr.Dataset:
        """Apply post-processing

        Args:
            dataset: Raw simulation result
            sensor: Sensor that produced the result

        Returns:
            Post-processed dataset
        """
        from ..config import SpectralResponse

        radiance = dataset["radiance"]
        pp = getattr(sensor, "post_processing", None)

        if pp is not None:
            if getattr(pp, "apply_circular_mask", False):
                fov = getattr(sensor, "fov", None)
                if fov is not None:
                    radiance = self._apply_circular_fov_mask(radiance, fov)
                else:
                    logger.warning(
                        f"Circular mask enabled but sensor '{sensor.id}' has no fov. Skipping."
                    )

            if getattr(pp, "generate_rgb_image", False):
                from upath import UPath

                output_dir = UPath(dataset.attrs.get("output_dir", "./output"))
                self._generate_rgb_image(radiance, sensor.id, output_dir, pp)

            if getattr(pp, "spatial_averaging", True):
                radiance = self._apply_spatial_averaging(radiance)

        srf = getattr(sensor, "srf", None)
        if isinstance(srf, SpectralResponse) and srf.type == "gaussian":
            if pp is None or getattr(pp, "apply_srf", True):
                logger.info(
                    f"Applying Gaussian SRF post-processing to sensor '{sensor.id}'"
                )
                target_wavelengths = self._resolve_target_wavelengths(srf, radiance)
                radiance = self._apply_gaussian_srf_from_config(
                    radiance, srf, target_wavelengths
                )

        return self._build_result(dataset, radiance)

    def _resolve_target_wavelengths(self, srf_config, radiance: xr.DataArray):
        """Determine target wavelengths for SRF convolution.

        Priority order:
        1. output_grid from SRF config
        2. Simulation wavelengths from input radiance (default fallback)
        """
        if srf_config.output_grid is not None:
            return srf_config.output_grid.generate_wavelengths()

        return radiance.w.values

    def _build_result(
        self, original_dataset: xr.Dataset, processed_radiance: xr.DataArray
    ) -> xr.Dataset:
        """Assemble output dataset, propagating available variables at new wavelength grid."""
        result = {"radiance": processed_radiance}
        for key in ["irradiance", "toa_irradiance"]:
            if key in original_dataset:
                result[key] = original_dataset[key].interp(w=processed_radiance.w)
        return xr.Dataset(result, attrs=original_dataset.attrs)

    def _apply_gaussian_srf_from_config(
        self,
        radiance: xr.DataArray,
        srf_config,
        target_wavelengths: np.ndarray,
    ) -> xr.DataArray:
        """Apply Gaussian SRF convolution at each target wavelength.

        Preserves spatial dimensions (y_index, x_index) when present in input.
        Works for both 2D satellite data and 1D point spectra.

        Args:
            radiance: Radiance DataArray with wavelength coordinate and bin edges
                     (requires 'bin_wmin' and 'bin_wmax' coordinates)
            srf_config: SpectralResponse with type="gaussian"
            target_wavelengths: Target wavelength grid (nm)

        Returns:
            Radiance DataArray with Gaussian SRF convolution applied.
        """
        import eradiate
        from eradiate.pipelines.logic import apply_spectral_response
        from eradiate.spectral.response import make_gaussian

        logger.info(
            f"Applying Gaussian SRF to {len(target_wavelengths)} wavelengths "
            f"({target_wavelengths.min():.1f}-{target_wavelengths.max():.1f} nm)"
        )

        # Check required bin edge coordinates for SRF application
        required_coords = {"bin_wmin", "bin_wmax"}
        if not required_coords.issubset(set(radiance.coords.keys())):
            raise ValueError(
                f"Radiance missing required SRF coordinates: {required_coords}. "
                f"Available: {set(radiance.coords.keys())}. "
                f"Eradiate simulations should include bin edge coordinates."
            )

        # Detect if this is 2D satellite data (has spatial dimensions)
        spatial_dims = [d for d in ["x_index", "y_index"] if d in radiance.dims]
        is_2d_data = len(spatial_dims) == 2

        if is_2d_data:
            logger.info(
                f"Processing 2D satellite data with spatial dims: {spatial_dims}"
            )

        # Apply Gaussian SRF at each target wavelength
        processed_bands = []
        for wl in target_wavelengths:
            fwhm = srf_config.get_fwhm_for_wavelength(wl)
            gaussian_srf = make_gaussian(wl, fwhm, pad=True)
            band_srf = eradiate.spectral.BandSRF.from_dataarray(gaussian_srf.srf)
            l_srf = apply_spectral_response(radiance, band_srf)
            processed_bands.append(l_srf)

        # Build result DataArray - preserve spatial dimensions for 2D data
        if is_2d_data:
            # Stack along wavelength dimension, preserving spatial dims
            result_radiance = xr.concat(processed_bands, dim="w")
            result_radiance = result_radiance.assign_coords(w=target_wavelengths)
            # Reorder dims to put 'w' first, then spatial dims, then any others
            other_dims = [
                d for d in result_radiance.dims if d not in ["w", "y_index", "x_index"]
            ]
            result_radiance = result_radiance.transpose(
                "w", "y_index", "x_index", *other_dims
            )
        else:
            # 1D case (point measurement or already spatially averaged)
            data_array = np.array(
                [float(band.values.squeeze()) for band in processed_bands],
                dtype=np.float64,
            )
            result_radiance = xr.DataArray(
                data=data_array,
                dims=["w"],
                coords={"w": target_wavelengths},
            )

        result_radiance = result_radiance.sortby("w")
        result_radiance.attrs = {
            **radiance.attrs,
            "gaussian_srf_applied": True,
            "srf_type": "gaussian",
        }

        return result_radiance

    def _apply_spatial_averaging(self, radiance: xr.DataArray) -> xr.DataArray:
        """Average over spatial dimensions (FOV pixels).

        Uses nanmean to properly handle NaN values from circular masking.
        Pixels outside the circular FOV are excluded from averaging.

        Args:
            radiance: Radiance DataArray with spatial dimensions

        Returns:
            Radiance averaged over spatial dimensions
        """
        spatial_dims = [d for d in ["x_index", "y_index"] if d in radiance.dims]
        if spatial_dims:
            return radiance.reduce(np.nanmean, dim=spatial_dims)
        return radiance

    def _apply_circular_fov_mask(
        self,
        radiance: xr.DataArray,
        fov_degrees: float,
    ) -> xr.DataArray:
        """Apply circular FOV mask to camera data.

        Sets pixels outside the circular field of view to NaN.
        Uses simple geometric approach: circle radius = min(width, height) / 2.

        Args:
            radiance: Radiance DataArray with spatial dimensions (x_index, y_index)
            fov_degrees: Field of view in degrees (for metadata only)

        Returns:
            Radiance with circular mask applied (values outside circle are NaN)
        """
        if "x_index" not in radiance.dims or "y_index" not in radiance.dims:
            logger.warning(
                "Circular mask requires x_index and y_index dimensions. Skipping."
            )
            return radiance

        ny, nx = len(radiance.y_index), len(radiance.x_index)

        center_x = (nx - 1) / 2.0
        center_y = (ny - 1) / 2.0
        radius_pixels = min(nx, ny) / 2.0

        y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        circular_mask_array = distances <= radius_pixels

        circular_mask = xr.DataArray(
            circular_mask_array,
            dims=["y_index", "x_index"],
            coords={
                "y_index": radiance.coords["y_index"],
                "x_index": radiance.coords["x_index"],
            },
        )

        masked_radiance = radiance.where(circular_mask, other=np.nan)

        masked_radiance.attrs["circular_mask_applied"] = True
        masked_radiance.attrs["fov_degrees"] = fov_degrees
        masked_radiance.attrs["mask_radius_pixels"] = float(radius_pixels)

        logger.info(
            f"Applied circular FOV mask: radius={radius_pixels:.1f} pixels, "
            f"FOV={fov_degrees}°"
        )

        return masked_radiance

    def _generate_rgb_image(
        self,
        radiance: xr.DataArray,
        sensor_id: str,
        output_dir,
        config,
    ) -> None:
        """Generate RGB visualization from spatial radiance data.

        Creates RGB image by selecting wavelengths closest to red, green, blue
        target wavelengths and saving as PNG.

        Args:
            radiance: Radiance DataArray with spatial and spectral dimensions
            sensor_id: Sensor identifier for output filename
            output_dir: Directory to save RGB image
            config: Post-processing configuration with RGB settings
        """
        import numpy as np
        from PIL import Image
        from s2gos_utils.io.paths import mkdir, open_file

        if "x_index" not in radiance.dims or "y_index" not in radiance.dims:
            logger.warning("RGB generation requires spatial dimensions. Skipping.")
            return

        if "w" not in radiance.dims:
            logger.warning("RGB generation requires wavelength dimension. Skipping.")
            return

        wavelengths = radiance.coords["w"].values
        if len(wavelengths) < 3:
            logger.warning(
                f"RGB generation requires at least 3 wavelengths "
                f"(found {len(wavelengths)}). Skipping."
            )
            return

        try:
            from eradiate.xarray.interp import dataarray_to_rgb
        except ImportError:
            logger.warning(
                "eradiate.xarray.interp.dataarray_to_rgb not available. Skipping RGB."
            )
            return

        target_r, target_g, target_b = config.rgb_wavelengths
        actual_wavelengths = [
            radiance.sel(w=target_r, method="nearest").w.item(),
            radiance.sel(w=target_g, method="nearest").w.item(),
            radiance.sel(w=target_b, method="nearest").w.item(),
        ]

        logger.info(
            f"RGB generation for {sensor_id}: "
            f"R={actual_wavelengths[0]:.1f}nm, "
            f"G={actual_wavelengths[1]:.1f}nm, "
            f"B={actual_wavelengths[2]:.1f}nm"
        )

        channels = [("w", w) for w in actual_wavelengths]
        img = dataarray_to_rgb(radiance, channels=channels, normalize=False)

        img = img * config.rgb_brightness_factor
        img = np.clip(img, 0, 1)

        selected_wavelength = radiance.sel(w=actual_wavelengths[0], method="nearest")
        squeezed = selected_wavelength.squeeze()
        nan_mask = np.isnan(squeezed.values)

        if nan_mask.ndim != 2:
            logger.warning(
                f"Expected 2D nan_mask for RGB, got {nan_mask.ndim}D with shape {nan_mask.shape}. "
                f"Dimensions: {squeezed.dims}"
            )
            nan_mask = None

        img = (img * 255).astype(np.uint8)

        if nan_mask is not None and nan_mask.any():
            if img.shape[:2] == nan_mask.shape:
                img[nan_mask, 0] = 255
                img[nan_mask, 1] = 255
                img[nan_mask, 2] = 255
                logger.info(
                    f"Set {nan_mask.sum()} pixels outside circular FOV to white background"
                )
            else:
                logger.warning(
                    f"Cannot apply white background: img shape {img.shape[:2]} "
                    f"doesn't match nan_mask shape {nan_mask.shape}"
                )

        rgb_path = output_dir / f"{sensor_id}_rgb.png"

        mkdir(output_dir)
        rgb_image = Image.fromarray(img)

        with open_file(rgb_path, "wb") as f:
            rgb_image.save(f, format="PNG")

        logger.info(f"RGB image saved: {rgb_path}")
