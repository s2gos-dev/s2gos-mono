from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from s2gos_utils.io.resolver import resolver
from skyfield.api import load, wgs84


class Illumination(BaseModel):
    """Base illumination configuration."""

    type: str = Field(..., description="Illumination type")
    id: str = Field("illumination", description="Unique identifier")


class DirectionalIllumination(Illumination):
    """Directional illumination."""

    type: Literal["directional"] = Field(
        "directional", description="Illumination type (always 'directional')"
    )
    zenith: float = Field(
        30.0, ge=0.0, le=90.0, description="Solar zenith angle in degrees"
    )
    azimuth: float = Field(
        180.0, ge=0.0, lt=360.0, description="Solar azimuth angle in degrees"
    )
    irradiance_dataset: str = Field(
        "thuillier_2003", description="Solar irradiance dataset"
    )

    @classmethod
    def from_date_and_location(
        cls,
        time: datetime,
        latitude: float,
        longitude: float,
        irradiance_dataset: str = "thuillier_2003",
    ) -> "DirectionalIllumination":
        """
        Creates a DirectionalIllumination instance by calculating solar angles
        for a given time and location, and converting them to Eradiate conventions.

        Args:
            time: The date and time of the observation.
            latitude: Observer's latitude in degrees.
            longitude: Observer's longitude in degrees.
            irradiance_dataset: Name of the solar irradiance dataset to use.

        Returns:
            A new DirectionalIllumination instance with corrected zenith and azimuth.

        Raises:
            ValueError: Likely the sun is below the horizon at the specified time.
        """
        # Load timescale and ephemeris
        ts = load.timescale()
        ephemeris_path = str(resolver.resolve("de421.bsp", strict=False))
        planets = load(ephemeris_path)

        # Convert datetime to skyfield time
        skyfield_time = ts.utc(
            time.year, time.month, time.day, time.hour, time.minute, time.second
        )

        # Define observer location
        earth = planets["earth"]
        location = earth + wgs84.latlon(latitude, longitude)

        # Calculate sun position
        sun = planets["sun"]
        astrometric = location.at(skyfield_time).observe(sun)
        apparent = astrometric.apparent()
        alt, az, _ = apparent.altaz()

        # Check if sun is above horizon
        if alt.degrees < 0:
            raise ValueError(
                f"Sun is below horizon (altitude: {alt.degrees:.2f}°) at the specified time"
            )

        # Convert to zenith angle (degrees)
        zenith_angle = 90.0 - alt.degrees

        # Convert azimuth to Eradiate convention
        # Skyfield: 0°=North, 90°=East, 180°=South, 270°=West
        # Eradiate: 0°=East, 90°=North, 180°=West, 270°=South
        eradiate_az = (90.0 - az.degrees) % 360.0

        return cls(
            zenith=zenith_angle,
            azimuth=eradiate_az,
            irradiance_dataset=irradiance_dataset,
        )


class ConstantIllumination(Illumination):
    """Constant uniform illumination."""

    type: Literal["constant"] = Field(
        "constant", description="Illumination type (always 'constant')"
    )
    radiance: float = Field(1.0, gt=0.0, description="Constant radiance value")
