"""Seasonal snow model: estimate where snow lies from an approximate temperature.

Snow cover is derived in two steps. First a surface temperature is estimated for
every pixel — either from a simple synthetic climatology (warmer near the equator,
colder at altitude, swinging between summer and winter) or by interpolating a CAMS
atmospheric profile onto the terrain. Snow probability then follows a smooth,
S-shaped transition around freezing: near-certain below freezing and tapering to
none just above 0 °C.

Two seasons are supported — June and December (the solstices) — for either
hemisphere. The exact formulas are kept as comments on the functions below.
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from ...core.config import Month

# === Temperature Model Constants ===
T_R = 26.0  # Reference temperature at equator, sea level (°C)
BETA = 0.62  # Latitudinal gradient (°C/degree)
LAPSE_RATE = 0.0095  # Temperature lapse rate: 9.5°C/km -> 0.0095 °C/m

# === Seasonal Amplitude Interpolation ===
A_HIGH_LAT = 18.0  # Amplitude at high latitude (°C)
LAT_HIGH = 60.0  # Reference high latitude (degrees)
A_LOW_LAT = 3.0  # Amplitude at low latitude (°C)
LAT_LOW = 10.0  # Reference low latitude (degrees)

# === Phase Constants (NH/SH) ===
D_MAX_NH = 172  # Day the seasonal cycle peaks, Northern Hemisphere (~June 21 solstice)
D_MAX_SH = 355  # Day the seasonal cycle peaks, Southern Hemisphere (~Dec 21 solstice)

# === Snow Probability Parameters ===
T_C = 0.0  # 50% rain-snow transition temperature (°C)
SIGMA = 0.15  # Logistic function width (°C)
HARD_FREEZE_LIMIT = 0.3  # Temperature above which snow is impossible (°C)


def get_day_of_year(month: Month) -> int:
    """Map Month enum to approximate day of year."""
    if month == Month.DECEMBER:
        return 355
    elif month == Month.JUNE:
        return 172
    else:
        return 1


def apply_spatial_smoothing(data: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Apply Gaussian spatial smoothing."""
    if sigma <= 0:
        return data
    return gaussian_filter(data, sigma=sigma, mode="nearest")


def calculate_seasonal_amplitude(abs_lat: np.ndarray) -> np.ndarray:
    """Seasonal temperature swing as a function of latitude.

    Small near the equator, large near the poles (linear interpolation between the
    two reference latitudes, clamped to that range).
    """
    slope = (A_HIGH_LAT - A_LOW_LAT) / (LAT_HIGH - LAT_LOW)
    amplitude = A_LOW_LAT + slope * (abs_lat - LAT_LOW)
    return np.clip(amplitude, A_LOW_LAT, A_HIGH_LAT)


def interpolate_cams_temperature(
    elevations: np.ndarray,
    thermoprops: xr.Dataset,
) -> np.ndarray:
    """Interpolate temperature from CAMS atmospheric profile.

    Temperature is extracted from CAMS vertical profile and interpolated
    to terrain elevations.

    Args:
        elevations: Terrain elevation in meters, shape (height, width)
        thermoprops: CAMS dataset after squeeze(drop=True), containing:
            - t: air temperature [z] in Kelvin
            - z: height [z] in kilometers (0-120km typical)

    Returns:
        Temperature in Celsius, same shape as elevations
    """
    if "t" not in thermoprops.data_vars:
        raise ValueError(
            f"CAMS thermoprops missing 't' (temperature). Found: {set(thermoprops.data_vars)}"
        )
    if "z" not in thermoprops.coords and "z" not in thermoprops.data_vars:
        raise ValueError(
            f"CAMS thermoprops missing 'z' (height). Found coords: {set(thermoprops.coords)}, data_vars: {set(thermoprops.data_vars)}"
        )
    if thermoprops["t"].ndim != 1 or thermoprops["z"].ndim != 1:
        raise ValueError(
            f"CAMS variables must be 1D after squeeze(drop=True). Got t.shape={thermoprops['t'].shape}, z.shape={thermoprops['z'].shape}"
        )

    z_m = thermoprops["z"].values * 1000.0
    t_c = thermoprops["t"].values - 273.15

    interpolator = interp1d(
        z_m,
        t_c,
        kind="linear",
        bounds_error=False,
        fill_value=(t_c[0], t_c[-1]),
    )

    return interpolator(elevations)


def calculate_temperature_field(
    latitudes: np.ndarray,
    elevations: np.ndarray,
    day_of_year: int,
) -> np.ndarray:
    """Estimate surface air temperature from latitude, elevation, and day of year.

    Synthetic climatology: start from a warm equatorial reference, cool toward the
    poles and with altitude, and add a summer/winter swing whose size grows with
    latitude and whose timing depends on the hemisphere.
    """
    # T(phi, z, d) = T_R - BETA*|phi| + A(|phi|)*cos(2*pi/365*(d - d_max)) - LAPSE_RATE*z
    abs_lat = np.abs(latitudes)
    lat_term = -BETA * abs_lat
    amplitude = calculate_seasonal_amplitude(abs_lat)
    d_max_map = np.where(latitudes >= 0, D_MAX_NH, D_MAX_SH)
    phase = (2.0 * np.pi / 365.0) * (day_of_year - d_max_map)
    seasonal_term = amplitude * np.cos(phase)
    elevation_term = -LAPSE_RATE * elevations
    return T_R + lat_term + seasonal_term + elevation_term


def calculate_snow_probability_map(
    latitudes: np.ndarray,
    elevations: np.ndarray,
    day_of_year: int,
    smooth_sigma: float = 0.0,
    thermoprops: Optional[xr.Dataset] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the per-pixel probability of snow from temperature.

    Temperature comes from a CAMS profile when ``thermoprops`` is given, otherwise
    from the synthetic model. Probability follows a smooth transition around freezing
    — close to 1 well below 0 °C, falling to 0 just above it — and any pixel warmer
    than the hard freeze limit (0.3 °C) is left snow-free.

    Args:
        latitudes: Latitude in degrees, shape (height, width)
        elevations: Elevation in meters, shape (height, width)
        day_of_year: Day of year (1-365)
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
        thermoprops: Optional CAMS dataset (after squeeze). If provided,
                     uses CAMS temperature; otherwise uses synthetic model.

    Returns:
        (probabilities, temperatures) - both same shape as inputs
    """
    if thermoprops is not None:
        temperatures = interpolate_cams_temperature(elevations, thermoprops)
    else:
        temperatures = calculate_temperature_field(latitudes, elevations, day_of_year)
    probabilities = np.zeros_like(temperatures, dtype=np.float32)

    # P = 1 / (1 + exp((T - T_C) / SIGMA)) for T <= HARD_FREEZE_LIMIT, else 0
    cold_mask = temperatures <= HARD_FREEZE_LIMIT
    if np.any(cold_mask):
        cold_temps = temperatures[cold_mask]
        exponent = (cold_temps - T_C) / SIGMA
        probabilities[cold_mask] = 1.0 / (1.0 + np.exp(exponent))

    if smooth_sigma > 0:
        probabilities = apply_spatial_smoothing(probabilities, sigma=smooth_sigma)

    return probabilities, temperatures
