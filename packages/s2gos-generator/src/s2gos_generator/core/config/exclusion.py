from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

ExclusionTarget = Literal["vegetation", "buildings"]
ALL_EXCLUSION_TARGETS: Tuple[ExclusionTarget, ...] = ("vegetation", "buildings")


def _validate_lonlat(lon: float, lat: float, prefix: str = "") -> None:
    if not (-180 <= lon <= 180):
        raise ValueError(f"{prefix}Longitude {lon} out of valid range [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"{prefix}Latitude {lat} out of valid range [-90, 90]")


class CircleGeometry(BaseModel):
    """Circular exclusion geometry.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): center=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): center=(x, y) with coord_type="scene"
    """

    type: Literal["circle"] = "circle"

    center: Tuple[float, float] = Field(
        ..., description="Center: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    radius: float = Field(..., gt=0, description="Radius in meters")

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        if self.coord_type == "geographic":
            _validate_lonlat(*self.center)
        return self


class BoxGeometry(BaseModel):
    """Rectangular box exclusion geometry.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): center=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): center=(x, y) with coord_type="scene"
    """

    type: Literal["box"] = "box"

    center: Tuple[float, float] = Field(
        ..., description="Center: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    width: float = Field(..., gt=0, description="Width in meters (east-west)")
    height: float = Field(..., gt=0, description="Height in meters (north-south)")

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        if self.coord_type == "geographic":
            _validate_lonlat(*self.center)
        return self


class PolygonGeometry(BaseModel):
    """Polygon exclusion geometry.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): coordinates=[(lon, lat), ...] with coord_type="geographic"
    - Scene coordinates (meters from scene center): coordinates=[(x, y), ...] with coord_type="scene"
    """

    type: Literal["polygon"] = "polygon"

    coordinates: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="Vertices: [(lon, lat), ...] if geographic, [(x, y), ...] if scene. Min 3 vertices.",
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        if self.coord_type == "geographic":
            for i, (lon, lat) in enumerate(self.coordinates):
                _validate_lonlat(lon, lat, prefix=f"Vertex {i}: ")
        return self


def _validate_excludes(v):
    if not v:
        raise ValueError("excludes must name at least one target")
    return list(dict.fromkeys(v))


class ExclusionZone(BaseModel):
    """Standalone area kept clear of procedural content.

    Excludes both vegetation and buildings unless ``excludes`` narrows it.
    Geometry can be a circle, box, or arbitrary polygon.
    """

    zone_id: str = Field(..., description="Unique identifier for this exclusion zone")
    geometry: Union[CircleGeometry, BoxGeometry, PolygonGeometry] = Field(
        ..., discriminator="type", description="Zone geometry"
    )
    excludes: List[ExclusionTarget] = Field(
        default_factory=lambda: list(ALL_EXCLUSION_TARGETS),
        description="Which procedural content is kept out (default: all)",
    )

    @field_validator("excludes")
    @classmethod
    def _check_excludes(cls, v):
        return _validate_excludes(v)


class ObjectExclusionZone(BaseModel):
    """Exclusion zone centred on a placed object (user asset or XML scene).

    Accepts shorthand: a bare number is a circle ``radius``, a ``(width, height)``
    pair is a box ``size``. Use the full form to narrow ``excludes``, e.g.
    ``{"radius": 15, "excludes": ["vegetation"]}``.
    """

    radius: Optional[float] = Field(None, gt=0, description="Circle radius in meters")
    size: Optional[Tuple[float, float]] = Field(
        None, description="Box (width, height) in meters"
    )
    excludes: List[ExclusionTarget] = Field(
        default_factory=lambda: list(ALL_EXCLUSION_TARGETS),
        description="Which procedural content is kept out (default: all)",
    )

    @field_validator("excludes")
    @classmethod
    def _check_excludes(cls, v):
        return _validate_excludes(v)

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, v):
        if isinstance(v, bool):
            raise ValueError(
                "exclusion_zone must be a radius, (width, height) or mapping"
            )
        if isinstance(v, (int, float)):
            return {"radius": v}
        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError("exclusion_zone box shorthand must be (width, height)")
            return {"size": tuple(v)}
        return v

    @model_validator(mode="after")
    def _check_shape(self):
        if (self.radius is None) == (self.size is None):
            raise ValueError("exclusion_zone needs exactly one of radius or size")
        if self.size is not None and any(s <= 0 for s in self.size):
            raise ValueError("exclusion_zone size must be positive")
        return self
