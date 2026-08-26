from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Literal, NamedTuple, Optional

from pydantic import BaseModel, Field, field_validator
from s2gos_utils.io.resolver import resolver
from skyfield.api import load, utc, wgs84

# Solar radius in km used to derive the Sun's
# apparent angular diameter from the Earth-Sun distance.
SOLAR_RADIUS_KM = 695_700.0


class SolarPosition(NamedTuple):
    """Result of a solar position computation."""

    zenith: float
    """Solar zenith angle in degrees."""

    azimuth: float
    """Solar azimuth angle in degrees, Eradiate convention (0=East, CCW)."""

    distance_km: float
    """Apparent Earth-Sun distance in km."""

    utc_time: datetime
    """The timezone-aware UTC instant the position was computed for."""


def solar_position(
    time: datetime,
    latitude: float,
    longitude: float,
    ephemeris: str = "de421.bsp",
) -> SolarPosition:
    """Compute the solar position for a given time and location.

    Args:
        time: The date and time of the observation. Naive datetimes are
            interpreted as UTC; aware ones are converted to UTC.
        latitude: Observer's latitude in degrees.
        longitude: Observer's longitude in degrees.
        ephemeris: Name of the ephemeris file to resolve.

    Returns:
        A :class:`SolarPosition`.

    Raises:
        ValueError: The sun is below the horizon at the specified time.
    """
    # Load timescale and ephemeris
    ts = load.timescale()
    ephemeris_path = str(resolver.resolve(ephemeris, strict=False))
    planets = load(ephemeris_path)

    utc_time = time.replace(tzinfo=utc) if time.tzinfo is None else time.astimezone(utc)
    skyfield_time = ts.from_datetime(utc_time)

    earth = planets["earth"]
    location = earth + wgs84.latlon(latitude, longitude)

    sun = planets["sun"]
    astrometric = location.at(skyfield_time).observe(sun)
    apparent = astrometric.apparent()
    alt, az, distance = apparent.altaz()

    if alt.degrees < 0:
        raise ValueError(
            f"Sun is below horizon (altitude: {alt.degrees:.2f}°) at the specified time"
        )

    zenith_angle = 90.0 - alt.degrees
    eradiate_az = (90.0 - az.degrees) % 360.0

    return SolarPosition(zenith_angle, eradiate_az, distance.km, utc_time)


def solar_angular_diameter(distance_km: float) -> float:
    """Apparent solar diameter in degrees at a given Earth-Sun distance."""
    return 2.0 * math.degrees(math.asin(SOLAR_RADIUS_KM / distance_km))


class Illumination(BaseModel):
    """Base illumination configuration."""

    type: str = Field(..., description="Illumination type")
    id: str = Field("illumination", description="Unique identifier")


class AbstractDirectionalIllumination(Illumination):
    """Base for illuminations defined by a solar direction and irradiance."""

    zenith: float = Field(
        30.0, ge=0.0, le=90.0, description="Solar zenith angle in degrees"
    )
    azimuth: float = Field(
        180.0, ge=0.0, lt=360.0, description="Solar azimuth angle in degrees"
    )
    irradiance_dataset: str = Field(
        "thuillier_2003", description="Solar irradiance dataset"
    )
    irradiance_datetime: Optional[datetime] = Field(
        None,
        description=(
            "Observation datetime used to scale the solar irradiance by the "
            "Earth-Sun distance. None disables the correction. "
            "Note that enabling it makes Eradiate load its own "
            "ephemeris, which is downloaded on first use."
        ),
    )

    @classmethod
    def from_date_and_location(
        cls,
        time: datetime,
        latitude: float,
        longitude: float,
        irradiance_dataset: str = "thuillier_2003",
    ):
        """Build an illumination from an observation time and location.

        Solar angles are computed with Skyfield and converted to Eradiate
        conventions. The observation time is also recorded so the irradiance is
        scaled by the Earth-Sun distance.

        Args:
            time: The date and time of the observation (UTC).
            latitude: Observer's latitude in degrees.
            longitude: Observer's longitude in degrees.
            irradiance_dataset: Name of the solar irradiance dataset to use.

        Raises:
            ValueError: The sun is below the horizon at the specified time.
        """
        position = solar_position(time, latitude, longitude)

        return cls(
            zenith=position.zenith,
            azimuth=position.azimuth,
            irradiance_dataset=irradiance_dataset,
            irradiance_datetime=position.utc_time,
        )


class DirectionalIllumination(AbstractDirectionalIllumination):
    """Directional illumination.

    Models the sun as a delta emitter: all rays are exactly parallel, so
    shadows have hard edges. This is the recommended type for most work.
    """

    type: Literal["directional"] = Field(
        "directional", description="Illumination type (always 'directional')"
    )


class AstroObjectIllumination(AbstractDirectionalIllumination):
    """Illumination by an astronomical object of finite angular size.
    See Eradiate documentation for more details.

    Warning:
        Eradiate marks the underlying ``AstroObjectIllumination`` as an
        experimental feature and recommends :class:`DirectionalIllumination`
        for production use.
    """

    type: Literal["astro_object"] = Field(
        "astro_object", description="Illumination type (always 'astro_object')"
    )
    angular_diameter: float = Field(
        0.5358,
        gt=0.0,
        lt=180.0,
        description=(
            "Apparent diameter of the celestial body in degrees. The default is "
            "the mean apparent diameter of the Sun seen from Earth. The Mitsuba "
            "plugin requires 0 < angular_diameter < 180."
        ),
    )

    @field_validator("angular_diameter")
    @classmethod
    def _warn_if_far_from_solar_disc(cls, value: float) -> float:
        """Warn about values that are valid but a poor idea."""
        if not 0.1 <= value <= 5.0:
            warnings.warn(
                f"angular_diameter={value}° is far from the Sun's apparent "
                "diameter (~0.5358°); results may be noisy or unphysical. "
                "Below ~0.1°, prefer DirectionalIllumination.",
                UserWarning,
                stacklevel=2,
            )
        return value

    @classmethod
    def from_date_and_location(
        cls,
        time: datetime,
        latitude: float,
        longitude: float,
        irradiance_dataset: str = "thuillier_2003",
        angular_diameter: Optional[float] = None,
    ) -> "AstroObjectIllumination":
        """Build an astro object illumination from a time and location.

        Args:
            time: The date and time of the observation (UTC).
            latitude: Observer's latitude in degrees.
            longitude: Observer's longitude in degrees.
            irradiance_dataset: Name of the solar irradiance dataset to use.
            angular_diameter: Apparent solar diameter in degrees. When omitted,
                it is derived from the Earth-Sun distance at ``time``.

        Raises:
            ValueError: The sun is below the horizon at the specified time.
        """
        position = solar_position(time, latitude, longitude)

        if angular_diameter is None:
            angular_diameter = solar_angular_diameter(position.distance_km)

        return cls(
            zenith=position.zenith,
            azimuth=position.azimuth,
            irradiance_dataset=irradiance_dataset,
            irradiance_datetime=position.utc_time,
            angular_diameter=angular_diameter,
        )


class ConstantIllumination(Illumination):
    """Constant uniform illumination."""

    type: Literal["constant"] = Field(
        "constant", description="Illumination type (always 'constant')"
    )
    radiance: float = Field(1.0, gt=0.0, description="Constant radiance value")
