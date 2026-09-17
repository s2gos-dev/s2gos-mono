"""
Material system for S2GOS scenes

Contains material definitions, loaders, and related utilities.
"""

from .brdf import BRDFModels
from .definitions import (
    BilambertianMaterial,
    ConductorMaterial,
    DielectricMaterial,
    DiffuseMaterial,
    Material,
    MeasuredMaterial,
    OceanGraspMaterial,
    OceanLegacyMaterial,
    PlasticMaterial,
    PrincipledMaterial,
    RoughConductorMaterial,
    RPVMaterial,
)
from .enums import BackgroundMaterial, MaterialType
from .loader import MaterialConfigLoader, get_landcover_mapping, load_materials

__all__ = [
    "Material",
    "DiffuseMaterial",
    "BilambertianMaterial",
    "RPVMaterial",
    "OceanGraspMaterial",
    "OceanLegacyMaterial",
    "DielectricMaterial",
    "ConductorMaterial",
    "RoughConductorMaterial",
    "PlasticMaterial",
    "PrincipledMaterial",
    "MeasuredMaterial",
    "MaterialConfigLoader",
    "load_materials",
    "get_landcover_mapping",
    "MaterialType",
    "BackgroundMaterial",
    "BRDFModels",
]
