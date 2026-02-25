from enum import Enum


class MaterialType(str, Enum):
    """Material types for scene surfaces."""

    DIFFUSE = "diffuse"
    RPV = "rpv"
    BILAMBERTIAN = "bilambertian"
    OCEAN_LEGACY = "ocean_legacy"


class BackgroundMaterial(str, Enum):
    """Available background materials."""

    WATER = "water"
    BARESOIL = "baresoil"
    CONCRETE = "concrete"
    SNOW = "snow"
    MOSS = "moss"
    TREECOVER = "treecover"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    MANGROVES = "mangroves"
    WETLAND = "wetland"
