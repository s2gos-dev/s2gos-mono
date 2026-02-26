#!/usr/bin/env python3
import enum
from typing import Annotated

from pydantic import BaseModel, Field
from s2gos_utils.io import PathRef

from s2gos_apps.registry import registry

target_data = {
    "pnp": {
        "target_lat": -46.917,
        "target_lon": -72.450,
        "target_size": 10,
        # "gmt_hour" : 14,
    },
    "pisa": {
        "target_lat": 43.732,
        "target_lon": 10.350,
        "target_size": 15,
        # "gmt_hour" : 12,
    },
}


class Locations(enum.StrEnum):
    PNP = "pnp"
    PISA = "pisa"


class Month(enum.StrEnum):
    JUNE = "June"


class SatelliteInstrument(enum.StrEnum):
    CHIME = "CHIME"
    MSI = "MSI"


class SatelliteObservation(BaseModel):
    satellite_instrument: Annotated[
        SatelliteInstrument,
        Field(default=SatelliteInstrument.CHIME, description="Satellite Instrument"),
    ]
    spp: Annotated[int, Field(default=8, description="Sample Per Pixel")]
    # Not usable yet
    orthorectified: Annotated[
        bool,
        Field(
            default=True,
            description="Specifies whether the simulation is done in sensor space or target space.",
        ),
    ]
    psf: Annotated[bool, Field(default=False, description="Point spread function.")]
    srf: Annotated[bool, Field(default=True, description="Spectral response function.")]
    radiometric_noise: Annotated[float, Field(default=0.0, description="")]


class GroundObservationType(enum.StrEnum):
    HYPSTAR_HCRF = "Hypstar HCRF"
    CAMERA = "camera"


class GroundObservation(BaseModel):
    observation_type: Annotated[
        GroundObservationType,
        Field(
            default=GroundObservationType.HYPSTAR_HCRF,
            description="Ground observation type",
        ),
    ]
    spp: Annotated[int, Field(default=8, description="Sample Per Pixel")]


class SurfaceL2Type(enum.StrEnum):
    HDRF = "HDRF"


class SurfaceL2(BaseModel):
    L2_product: Annotated[
        SurfaceL2Type, Field(default=SurfaceL2Type.HDRF, description="L2 Product")
    ]

    # pixel footprint, those two should be mutually exclusive.
    footprint: Annotated[
        float, Field(default=30.0, description="Pixel footprint resolution in meters")
    ]
    satellite: Annotated[
        SatelliteInstrument | None,
        Field(
            default=None,
            description="If specified, informs the pixel footprint of the L2 product. Takes precedence over `footprint`.",
        ),
    ]
    spp: Annotated[int, Field(default=8, description="Sample Per Pixel")]


# This is what we should use before it is possible to use nested objects..
class SimulationType(enum.StrEnum):
    CHIME = "CHIME"
    MSI = "MSI"
    HYPSTAR_HCRF = "Hypstar HCRF"
    CAMERA = "camera"
    HDRF = "HDRF"


@registry.process(id="upscaling-demo", title="Upscaling Demo")
def upscaling(
    # Common
    scene_name: Annotated[
        Locations, Field(default=Locations.PNP, description="Scene name.")
    ],
    month: Annotated[Month, Field(default="June", description="Month.")],
    day: Annotated[int, Field(default=1, ge=1, le=30, description="Day of the month.")],
    hour: Annotated[
        float,
        Field(
            ...,
            description="Time of day. For satelites, the closest overpass time will be used",
        ),
    ],
    include_TLS: Annotated[
        bool, Field(default=False, description="Include Telestrial Laser Scanned data.")
    ],
    # Observation
    observation: Annotated[
        SatelliteObservation | GroundObservation | SurfaceL2,
        Field(..., description="Observations."),
    ],
    # Output Config
    config_output_dir: Annotated[
        PathRef | None,
        Field(..., description="Generation configuration output directory."),
    ] = None,
    scene_output_dir: Annotated[
        PathRef | None,
        Field(..., description="Scene description output directiory."),
    ] = None,
) -> PathRef | None:
    """
    Create the scene confifuration corresponding the PNP scene.
    """
    raise NotImplementedError("Upscaling process not yet implemented")
