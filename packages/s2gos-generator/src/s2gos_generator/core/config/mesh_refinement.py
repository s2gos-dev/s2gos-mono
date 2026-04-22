"""Adaptive mesh refinement configuration."""

from pydantic import BaseModel, Field


class MeshRefinementConfig(BaseModel):
    """Adaptive quadtree mesh refinement configuration.

    When enabled, the terrain mesh is refined under feature polygons (e.g. roads)
    and a surrounding transition buffer.  Feature vertices are optionally flattened
    after refinement.

    Presence (non-None) in SceneGenConfig does not enable refinement by
    itself — ``enabled`` must also be True.
    """

    enabled: bool = Field(True, description="Enable adaptive mesh refinement")
    max_depth: int = Field(
        2,
        ge=1,
        le=4,
        description=(
            "Maximum quadtree refinement depth (each level doubles resolution). "
            "Ceiling of 4 is enforced by the uint64 cell encoding: each cell packs "
            "depth (4 bits), column i (30 bits), and row j (30 bits). At depth 4 the "
            "effective grid is nx<<4 × ny<<4; a DEM larger than ~67 k cells per axis "
            "would overflow the 30-bit field. Typical scenes (≤1000 cells/axis) are "
            "safe well beyond depth 4, but the cap prevents accidental memory blowup."
        ),
    )
    flatten: bool = Field(
        True,
        description="Flatten mesh vertices perpendicular to feature centerlines after refinement.",
    )
    transition_buffer_m: float = Field(
        20.0,
        ge=0.0,
        description="Transition zone width outside feature edge (metres).",
    )
    blend_width_factor: float = Field(
        1.2,
        ge=1.0,
        description=(
            "Width multiplier for the blend zone at feature polygon edges. "
            "Vertices beyond factor × half_feature_width blend back to original elevation."
        ),
    )
