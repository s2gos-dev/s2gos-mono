from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class BaseViewing(BaseModel):
    """Base class for viewing configurations."""

    type: str


class AngularViewing(BaseViewing):
    """Viewing defined by zenith and azimuth angles, typically relative to a target.

    This is ideal for distant sensors (satellites).
    """

    type: Literal["angular"] = Field(
        "angular", description="Viewing type (always 'angular')"
    )
    zenith: float = Field(
        0.0,
        ge=0.0,
        le=180.0,
        description="Viewing zenith angle (0=nadir, 90=horizon, 180=upward)",
    )
    azimuth: float = Field(0.0, ge=0.0, lt=360.0, description="Viewing azimuth angle")
    target: Optional[List[float]] = Field(
        None,
        description="Center of the observed area [x, y, z]. If None, defaults to scene center [0,0,0].",
    )


class AngularFromOriginViewing(BaseViewing):
    """Viewing defined by an origin position and zenith/azimuth angles from that origin.

    This is ideal for radiancemeters and cameras where you know the sensor position
    and want to specify the pointing direction with angles.
    """

    type: Literal["angular_from_origin"] = Field(
        "angular_from_origin", description="Viewing type (always 'angular_from_origin')"
    )
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters"
    )
    zenith: float = Field(
        0.0,
        ge=0.0,
        le=180.0,
        description="Pointing zenith angle (0=up, 90=horizon, 180=nadir)",
    )
    azimuth: float = Field(
        0.0, ge=0.0, lt=360.0, description="Pointing azimuth angle (0=East, 90=North)"
    )
    up: Optional[List[float]] = Field(
        [0, 0, 1], description="Up direction for cameras [x, y, z] (default Z-up)"
    )
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinate in origin is offset from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinate is absolute elevation in scene coordinate system."
        ),
    )
    relative_to_asset: Optional[str] = Field(
        default=None,
        description=(
            "Asset ID to use as reference frame. When set, origin is "
            "interpreted as an offset in the asset's local coordinate system "
            "and transformed by the asset's position, rotation, and scale."
        ),
    )


class LookAtViewing(BaseViewing):
    """Viewing defined by an origin and a target point.

    This is ideal for UAVs or perspective cameras where the exact position
    of the sensor and its target are known.
    """

    type: Literal["directional"] = Field(
        "directional", description="Viewing type (always 'directional')"
    )
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters"
    )
    target: List[float] = Field(
        [0.0, 0.0, 0.0], description="3D target position [x, y, z] in meters"
    )
    up: Optional[List[float]] = Field(
        [0, 0, 1], description="Up direction for cameras [x, y, z] (default Z-up)"
    )
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinates in origin and target are offsets from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinates are absolute elevations in scene coordinate system."
        ),
    )
    relative_to_asset: Optional[str] = Field(
        default=None,
        description=(
            "Asset ID to use as reference frame. When set, origin and target "
            "are interpreted as offsets in the asset's local coordinate system "
            "and transformed by the asset's position, rotation, and scale."
        ),
    )


class RectangleTarget(BaseModel):
    """Rectangle target specification for distant measurements.

    Defines an axis-aligned rectangle at a specific altitude for mdistant targets.
    Used for pixel-level BRF measurements where we want to measure the reflectance
    leaving a specific rectangular area.
    """

    type: Literal["rectangle"] = Field(
        "rectangle", description="Target type (always 'rectangle')"
    )
    xmin: float = Field(..., description="Minimum X coordinate in meters")
    xmax: float = Field(..., description="Maximum X coordinate in meters")
    ymin: float = Field(..., description="Minimum Y coordinate in meters")
    ymax: float = Field(..., description="Maximum Y coordinate in meters")
    z: float = Field(default=0.0, description="Altitude (z-coordinate) in meters")

    @model_validator(mode="after")
    def validate_bounds(self):
        """Ensure bounds are valid (min < max)."""
        if self.xmin >= self.xmax:
            raise ValueError(f"xmin ({self.xmin}) must be less than xmax ({self.xmax})")
        if self.ymin >= self.ymax:
            raise ValueError(f"ymin ({self.ymin}) must be less than ymax ({self.ymax})")
        return self

    @classmethod
    def from_center_and_size(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: Optional[float] = None,
        z: float = 0.0,
    ) -> "RectangleTarget":
        """Create rectangle from center point, dimensions, and altitude.

        Args:
            cx: Center X coordinate in meters
            cy: Center Y coordinate in meters
            width: Width in meters
            height: Height in meters (defaults to width for square)
            z: Altitude in meters (default 0.0)

        Returns:
            RectangleTarget instance
        """
        if height is None:
            height = width
        hw, hh = width / 2, height / 2
        return cls(xmin=cx - hw, xmax=cx + hw, ymin=cy - hh, ymax=cy + hh, z=z)


class DistantViewing(BaseViewing):
    """Distant viewing for BRF measurements using mdistant measure.

    Models an infinitely distant sensor looking at a target area.
    Uses Eradiate's MultiDistantMeasure with rectangular targets for
    pixel-level BRF measurements.

    The sensor looks in the direction specified by ``direction`` (default: [0, 0, 1]
    meaning looking down from above / nadir view).
    """

    type: Literal["distant"] = Field(
        "distant", description="Viewing type (always 'distant')"
    )

    target: Optional[Union[List[float], RectangleTarget]] = Field(
        default=None,
        description=(
            "Target specification. Either:\n"
            "- [x, y, z] point for a point target\n"
            "- RectangleTarget for a rectangular area (pixel BRF)"
        ),
    )

    direction: List[float] = Field(
        default=[0, 0, 1],
        description=(
            "Viewing direction vector. Default [0, 0, 1] means looking "
            "straight down (nadir view). For oblique views, adjust accordingly."
        ),
    )

    ray_offset: Optional[float] = Field(
        default=None,
        description=(
            "Distance between target and ray origins in meters. "
            "If unset, ray origins are positioned automatically outside the scene."
        ),
    )

    terrain_relative_height: bool = Field(
        default=True,
        description=(
            "For point targets [x, y, z]: if True, z is offset from terrain. "
            "For RectangleTarget: this field is ignored (rectangles use absolute z)."
        ),
    )


class HemisphericalViewing(BaseViewing):
    """Viewing that covers the entire upper or lower hemisphere."""

    type: Literal["hemispherical"] = Field(
        "hemispherical", description="Viewing type (always 'hemispherical')"
    )
    origin: List[float] = Field(
        ..., description="3D sensor position [x, y, z] in meters."
    )
    upward_looking: bool = Field(
        True,
        description="True for upward-looking (sky), False for downward-looking (ground)",
    )
    terrain_relative_height: bool = Field(
        default=False,
        description=(
            "If True, z-coordinate in origin is offset from terrain surface. "
            "Backend will query DEM elevation at (x, y) and add z offset. "
            "If False (default), z-coordinate is absolute elevation in scene coordinate system."
        ),
    )


# LocationBased class removed - was unused duplicate of HemisphericalMeasurementLocation
# Use HemisphericalMeasurementLocation for location-based measurements instead

ViewingType = Union[
    AngularViewing,
    AngularFromOriginViewing,
    LookAtViewing,
    HemisphericalViewing,
    DistantViewing,
]
