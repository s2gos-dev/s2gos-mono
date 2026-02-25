"""Spectrum specification models for material definitions."""

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class FileSpectrum(BaseModel):
    """Spectrum loaded from a NetCDF file.

    The file should contain spectral data as a function of wavelength.
    Paths can be absolute or relative to the XML directory (when used
    in XML scene loading).

    Attributes:
        type: Literal "file" identifier
        path: Path to NetCDF file containing spectral data
        variable: Name of the variable in the NetCDF file (default: "reflectance")

    Examples:
        >>> spectrum = FileSpectrum(
        ...     path="/data/spectra/grass_reflectance.nc",
        ...     variable="reflectance"
        ... )
    """

    type: Literal["file"] = "file"
    path: str = Field(
        ..., description="Path to NetCDF file (absolute or relative to XML directory)"
    )
    variable: str = Field(
        default="reflectance", description="Variable name in NetCDF file"
    )

    @field_validator("path")
    @classmethod
    def validate_path_format(cls, v: str) -> str:
        """Validate path format."""
        if not v:
            raise ValueError(
                "Spectrum file path cannot be empty.\n"
                "Example: 'spectra/grass_reflectance.nc'"
            )

        # Check for common path issues
        if "\\" in v:
            raise ValueError(
                f"Use forward slashes in paths, not backslashes: {v}\n"
                f"Use: {v.replace(chr(92), '/')}"
            )

        return v

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, v: str) -> str:
        """Validate variable name."""
        if not v:
            raise ValueError(
                "Variable name cannot be empty.\n"
                "Common values: 'reflectance', 'transmittance', 'albedo'"
            )
        return v


class InterpolatedSpectrum(BaseModel):
    """Spectrum defined by wavelength-value pairs with linear interpolation.

    Values are linearly interpolated between the specified wavelength points.
    Wavelengths must be strictly increasing and positive.

    Attributes:
        type: Literal "interpolated" identifier
        wavelengths: Wavelength values in nanometers (must be sorted, positive)
        values: Corresponding spectral values (same length as wavelengths)

    Examples:
        >>> spectrum = InterpolatedSpectrum(
        ...     wavelengths=[400, 550, 700, 850],
        ...     values=[0.1, 0.3, 0.5, 0.4]
        ... )
    """

    type: Literal["interpolated"] = "interpolated"
    wavelengths: list[float] = Field(
        ...,
        description="Wavelengths in nanometers (must be sorted, positive)",
        min_length=2,
    )
    values: list[float] = Field(
        ..., description="Spectral values at each wavelength", min_length=2
    )

    @model_validator(mode="after")
    def validate_wavelengths_and_values(self):
        """Validate wavelengths and values are consistent."""
        # Check lengths match
        if len(self.wavelengths) != len(self.values):
            raise ValueError(
                f"Wavelengths and values must have same length.\n"
                f"Got {len(self.wavelengths)} wavelengths but {len(self.values)} values.\n"
                f"Wavelengths: {self.wavelengths}\n"
                f"Values: {self.values}"
            )

        # Check wavelengths are positive
        if any(w <= 0 for w in self.wavelengths):
            negative_wl = [w for w in self.wavelengths if w <= 0]
            raise ValueError(
                f"All wavelengths must be positive.\n"
                f"Found non-positive values: {negative_wl}\n"
                f"Wavelengths should be in nanometers (e.g., 400-800 for visible)"
            )

        # Check wavelengths are sorted
        if self.wavelengths != sorted(self.wavelengths):
            raise ValueError(
                f"Wavelengths must be in ascending order.\n"
                f"Got: {self.wavelengths}\n"
                f"Expected: {sorted(self.wavelengths)}"
            )

        # Check for duplicates
        if len(self.wavelengths) != len(set(self.wavelengths)):
            duplicates = [
                w for w in set(self.wavelengths) if self.wavelengths.count(w) > 1
            ]
            raise ValueError(
                f"Wavelengths must be unique.\nFound duplicates: {duplicates}"
            )

        return self


class UniformSpectrum(BaseModel):
    """Spectrum with a constant value across all wavelengths.

    Can be a scalar (applied to all channels) or RGB triplet.
    RGB values should typically be in [0, 1] range for reflectance.

    Attributes:
        type: Literal "uniform" identifier
        value: Scalar value or RGB triplet [R, G, B]

    Examples:
        Scalar (grayscale):
        >>> spectrum = UniformSpectrum(value=0.5)

        RGB:
        >>> spectrum = UniformSpectrum(value=[0.8, 0.6, 0.4])
    """

    type: Literal["uniform"] = "uniform"
    value: Union[float, list[float]] = Field(
        ..., description="Scalar value or RGB triplet [R, G, B]"
    )

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Union[float, list[float]]) -> Union[float, list[float]]:
        """Validate uniform value."""
        if isinstance(v, list):
            if len(v) != 3:
                raise ValueError(
                    f"RGB value must have exactly 3 components [R, G, B].\n"
                    f"Got {len(v)} components: {v}\n"
                    f"Example: [0.8, 0.6, 0.4]"
                )

            # Check range (warn if outside [0,1] but don't enforce)
            out_of_range = [x for x in v if x < 0 or x > 1]
            if out_of_range:
                # Note: We don't raise here because some use cases might need >1
                # But we could add a warning in the future
                pass

        return v


# Union type for all spectrum specifications
SpectrumSpec = Union[FileSpectrum, InterpolatedSpectrum, UniformSpectrum]


__all__ = [
    "FileSpectrum",
    "InterpolatedSpectrum",
    "UniformSpectrum",
    "SpectrumSpec",
]
