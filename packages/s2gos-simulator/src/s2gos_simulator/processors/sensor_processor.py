import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def image_circle_area_px(attrs: dict) -> float | None:
    """Area in pixels of a fisheye result's image circle, None for other results.

    Eradiate leaves the pixels outside the image circle at 0 and the pixels on its
    rim partly lit, so the sum over all pixels divided by this area is the exact mean
    over the image circle. For the equisolid projection every pixel spans the same
    solid angle, which makes it the radiance averaged over the FOV cone.

    Args:
        attrs: Attributes of the simulation result, as published by Eradiate's
            fisheye measure

    Returns:
        The image circle area in pixels, or None when the result is not a fisheye

    Raises:
        ValueError: if the image circle extends beyond the film
    """
    if "image_circle_radius" not in attrs:
        return None

    width, height = attrs["film_width"], attrs["film_height"]
    radius = attrs["image_circle_radius"] * width * min(1.0, attrs["rho_max"])
    center_x, center_y = attrs["center_x"] * width, attrs["center_y"] * height
    if min(center_x, width - center_x, center_y, height - center_y) < radius:
        raise ValueError(
            "Spatial averaging needs the image circle inside the film, got radius "
            f"{radius:g} px around ({center_x:g}, {center_y:g}) on a "
            f"{width}x{height} film"
        )
    return np.pi * radius**2


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

    def process_sensor_result(
        self, dataset: xr.Dataset, sensor, output_dir=None
    ) -> xr.Dataset:
        """Apply sensor-specific post-processing to a single dataset.

        This is the main entry point for all sensor post-processing.

        Args:
            dataset: Raw simulation result dataset
            sensor: Sensor configuration object
            output_dir: Output directory (UPath with credentials). When provided,
                used directly for RGB image generation instead of reading from
                dataset attributes (which strips credentials).

        Returns:
            Post-processed dataset with SRF convolution, averaging, etc. A
            irradiancemeter's is its irradiance (see `irradiancemeter.to_irradiance`).
        """
        from ..config import GroundInstrumentType
        from ..irradiancemeter import to_irradiance

        processed = self._apply_post_processing(dataset, sensor, output_dir=output_dir)
        if getattr(sensor, "instrument", None) == GroundInstrumentType.IRRADIANCEMETER:
            return to_irradiance(processed, sensor)
        return processed

    def _apply_post_processing(
        self, dataset: xr.Dataset, sensor, output_dir=None
    ) -> xr.Dataset:
        """Apply post-processing

        Args:
            dataset: Raw simulation result
            sensor: Sensor that produced the result
            output_dir: Output directory with credentials (avoids re-constructing from attrs)

        Returns:
            Post-processed dataset. The input dataset itself is returned unchanged
            when no post-processing step applies, so callers can detect a no-op with
            an identity check.
        """
        from ..config import SpectralResponse

        original_radiance = dataset["radiance"]
        radiance = original_radiance
        pp = getattr(sensor, "post_processing", None)

        if pp is not None and getattr(pp, "generate_rgb_image", False):
            from upath import UPath

            if output_dir is None:
                output_dir = UPath(dataset.attrs.get("output_dir", "./output"))
            self._generate_rgb_image(radiance, sensor.id, output_dir, pp)

        radiance = self._apply_averaging_step(radiance, pp, dataset.attrs)
        spatial_changed = radiance is not original_radiance

        spectral_changed = False
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
                spectral_changed = True

        if not (spatial_changed or spectral_changed):
            logger.debug(
                f"No post-processing applied to sensor '{sensor.id}', "
                "returning simulation result unchanged"
            )
            return dataset

        return self._build_result(
            dataset, radiance, pp, spectral_changed=spectral_changed
        )

    def _apply_averaging_step(
        self, data: xr.DataArray, pp, attrs: dict
    ) -> xr.DataArray:
        """Apply spatial averaging if the sensor is configured for it.

        Returns the input object unchanged when the step does not apply, so callers
        can detect a no-op with an identity check.
        """
        if pp is None or not getattr(pp, "spatial_averaging", True):
            return data

        return self._apply_spatial_averaging(data, image_circle_area_px(attrs))

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
        self,
        original_dataset: xr.Dataset,
        processed_radiance: xr.DataArray,
        pp=None,
        spectral_changed: bool = False,
    ) -> xr.Dataset:
        """Assemble output dataset, propagating available variables at new wavelength grid.

        Args:
            original_dataset: Raw simulation result
            processed_radiance: Post-processed radiance
            pp: PostProcessingOptions of the sensor (may be None), replayed on the
                extra pixel variables except Eradiate's ``valid`` mask
            spectral_changed: Whether the wavelength grid changed

        Returns:
            Post-processed dataset
        """
        result = {"radiance": processed_radiance}

        for key in ["irradiance", "toa_irradiance"]:
            if key in original_dataset:
                result[key] = original_dataset[key].interp(w=processed_radiance.w)

        extras = [name for name in original_dataset.data_vars if name not in result]

        if spectral_changed:
            if extras:
                logger.debug(
                    f"Dropping variables not integrated onto the target wavelength "
                    f"grid: {extras}"
                )
        else:
            for name in extras:
                variable = original_dataset[name]
                if name != "valid" and any(
                    d in variable.dims for d in ("x_index", "y_index")
                ):
                    variable = self._apply_averaging_step(
                        variable, pp, original_dataset.attrs
                    )
                result[name] = variable

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

    def _apply_spatial_averaging(
        self, radiance: xr.DataArray, circle_area_px: float | None = None
    ) -> xr.DataArray:
        """Average over spatial dimensions (FOV pixels).

        Args:
            radiance: Radiance DataArray with spatial dimensions
            circle_area_px: Image circle area of a fisheye result (see
                `image_circle_area_px`), whose sum over all pixels is then divided
                by it. Other results take the mean, skipping NaN pixels.

        Returns:
            Radiance averaged over spatial dimensions
        """
        spatial_dims = [d for d in ["x_index", "y_index"] if d in radiance.dims]
        if not spatial_dims:
            return radiance
        if circle_area_px is not None:
            return radiance.sum(dim=spatial_dims) / circle_area_px
        return radiance.reduce(np.nanmean, dim=spatial_dims)

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
