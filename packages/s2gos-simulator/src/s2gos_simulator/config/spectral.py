from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils.io.resolver import resolver


class SpectralRegion(BaseModel):
    """Define a spectral region with specific FWHM.

    Used for wavelength-dependent Gaussian SRF where different
    spectral regions have different spectral resolution.

    Example:
        SpectralRegion(name="VNIR", wmin_nm=400, wmax_nm=1000, fwhm_nm=8.5)
    """

    name: str = Field(..., description="Region identifier (e.g., 'VNIR', 'SWIR1')")
    wmin_nm: float = Field(..., ge=0, description="Minimum wavelength (nm)")
    wmax_nm: float = Field(..., description="Maximum wavelength (nm)")
    fwhm_nm: float = Field(..., gt=0, description="Full Width at Half Maximum (nm)")

    @model_validator(mode="after")
    def validate_range(self):
        if self.wmax_nm <= self.wmin_nm:
            raise ValueError(
                f"wmax_nm ({self.wmax_nm}) must be > wmin_nm ({self.wmin_nm})"
            )
        return self


class WavelengthGrid(BaseModel):
    """Configuration for output wavelength grid.

    Supports three modes:

    1. Regular grid: wmin_nm/wmax_nm/step_nm (Spectral Sampling Interval)
    2. Explicit wavelengths: list of specific wavelengths
    3. From file: extract wavelengths from external NetCDF/CSV file

    Example:
        # CHIME-like regular grid (400-2500nm @ 8.4nm SSI)
        WavelengthGrid(mode="regular", wmin_nm=400, wmax_nm=2500, step_nm=8.4)

        # Explicit wavelengths
        WavelengthGrid(mode="explicit", wavelengths_nm=[450, 550, 650, 850])

        # From reference file
        WavelengthGrid(mode="from_file", file_path="reference.nc", wavelength_variable="wavelength")
    """

    mode: Literal["regular", "explicit", "from_file"] = Field(
        "regular", description="Grid mode: 'regular', 'explicit', or 'from_file'"
    )

    # Regular grid mode
    wmin_nm: Optional[float] = Field(None, ge=0, description="Minimum wavelength (nm)")
    wmax_nm: Optional[float] = Field(None, description="Maximum wavelength (nm)")
    step_nm: Optional[float] = Field(
        None, gt=0, description="Wavelength step / Spectral Sampling Interval (nm)"
    )

    # Explicit mode
    wavelengths_nm: Optional[List[float]] = Field(
        None, description="Explicit list of wavelengths (nm)"
    )

    # From file mode
    file_path: Optional[str] = Field(
        None, description="Path to wavelength reference file (NetCDF or CSV)"
    )
    wavelength_variable: str = Field(
        "wavelength", description="Variable/column name for wavelengths in file"
    )

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == "regular":
            if self.wmin_nm is None or self.wmax_nm is None:
                raise ValueError("Regular mode requires wmin_nm and wmax_nm")
            if self.step_nm is None:
                raise ValueError("Regular mode requires step_nm (SSI)")
            if self.wmax_nm <= self.wmin_nm:
                raise ValueError(
                    f"wmax_nm ({self.wmax_nm}) must be > wmin_nm ({self.wmin_nm})"
                )
        elif self.mode == "explicit":
            if not self.wavelengths_nm or len(self.wavelengths_nm) == 0:
                raise ValueError("Explicit mode requires wavelengths_nm list")
        elif self.mode == "from_file":
            if not self.file_path:
                raise ValueError("from_file mode requires file_path")
        return self

    def generate_wavelengths(self) -> "np.ndarray":
        """Generate wavelength array based on configuration.

        Returns:
            numpy array of wavelengths in nm
        """
        import numpy as np

        if self.mode == "regular":
            # Include endpoint by adding half step to avoid floating point issues
            return np.arange(
                self.wmin_nm, self.wmax_nm + self.step_nm / 2, self.step_nm
            )
        elif self.mode == "explicit":
            return np.array(sorted(self.wavelengths_nm), dtype=np.float64)
        elif self.mode == "from_file":
            import xarray as xr

            resolved_path = resolver.resolve(self.file_path, strict=True)
            ds = xr.open_dataset(resolved_path)
            return ds[self.wavelength_variable].values
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class SpectralResponse(BaseModel):
    """Spectral response function configuration.

    Supports multiple SRF types:

    - **delta**: Discrete wavelength points → Eradiate's multi_delta
    - **uniform**: Wavelength range (wmin, wmax) → Eradiate's uniform
    - **dataset**: String ID for predefined SRFs → Eradiate's internal database
    - **gaussian**: Gaussian SRF with configurable FWHM (for hyperspectral instruments)

    Gaussian SRF Example (CHIME-like):
        SpectralResponse(
            type="gaussian",
            spectral_regions=[
                SpectralRegion(name="VNIR", wmin_nm=400, wmax_nm=1000, fwhm_nm=8.5),
                SpectralRegion(name="SWIR1", wmin_nm=1000, wmax_nm=1800, fwhm_nm=10.0),
                SpectralRegion(name="SWIR2", wmin_nm=1800, wmax_nm=2500, fwhm_nm=11.0),
            ],
            output_grid=WavelengthGrid(mode="regular", wmin_nm=400, wmax_nm=2500, step_nm=8.4),
        )
    """

    type: Literal["delta", "uniform", "dataset", "gaussian"] = Field(
        "delta", description="SRF type: 'delta', 'uniform', 'dataset', or 'gaussian'"
    )

    # Delta SRF fields
    wavelengths: Optional[List[float]] = Field(
        None,
        description="Wavelengths in nm for delta SRF",
    )

    # Uniform SRF fields
    wmin: Optional[float] = Field(
        None, description="Minimum wavelength for uniform SRF"
    )
    wmax: Optional[float] = Field(
        None, description="Maximum wavelength for uniform SRF"
    )

    # Dataset SRF fields
    dataset_id: Optional[str] = Field(None, description="Dataset ID for predefined SRF")

    # Gaussian SRF fields
    fwhm_nm: Optional[float] = Field(
        None,
        gt=0,
        description="Single FWHM for all wavelengths (nm). Use this OR spectral_regions.",
    )
    spectral_regions: Optional[List[SpectralRegion]] = Field(
        None,
        description="Wavelength-dependent FWHM via spectral regions. Use this OR fwhm_nm.",
    )
    output_grid: Optional[WavelengthGrid] = Field(
        None,
        description="Output wavelength grid configuration. Required for gaussian type.",
    )

    @field_validator("wavelengths")
    @classmethod
    def validate_wavelengths(cls, v):
        if v is not None:
            return [float(w) for w in v]
        return v

    @model_validator(mode="after")
    def validate_srf_config(self):
        if self.type == "delta" and not self.wavelengths:
            raise ValueError("Delta SRF requires wavelengths")
        elif self.type == "uniform" and (not self.wmin or not self.wmax):
            raise ValueError("Uniform SRF requires wmin and wmax")
        elif self.type == "dataset" and not self.dataset_id:
            raise ValueError("Dataset SRF requires dataset_id")
        elif self.type == "gaussian":
            has_single_fwhm = self.fwhm_nm is not None
            has_regions = (
                self.spectral_regions is not None and len(self.spectral_regions) > 0
            )
            if not (has_single_fwhm or has_regions):
                raise ValueError("Gaussian SRF requires fwhm_nm OR spectral_regions")
            if has_single_fwhm and has_regions:
                raise ValueError("Specify fwhm_nm OR spectral_regions, not both")
        return self

    def get_fwhm_for_wavelength(self, wavelength_nm: float) -> float:
        """Get FWHM for a given wavelength.

        For single fwhm_nm: returns that value for all wavelengths.
        For spectral_regions: returns FWHM of the region containing the wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            FWHM in nanometers for that wavelength
        """
        if self.fwhm_nm is not None:
            return self.fwhm_nm

        if self.spectral_regions:
            for region in self.spectral_regions:
                if region.wmin_nm <= wavelength_nm < region.wmax_nm:
                    return region.fwhm_nm
            # Fallback: use nearest region if wavelength is outside all regions
            if wavelength_nm < self.spectral_regions[0].wmin_nm:
                return self.spectral_regions[0].fwhm_nm
            return self.spectral_regions[-1].fwhm_nm

        raise ValueError("No FWHM configuration available")


# TODO: Perhaps it can be less eradiate specific?
SRFType = Union[SpectralResponse, str]
