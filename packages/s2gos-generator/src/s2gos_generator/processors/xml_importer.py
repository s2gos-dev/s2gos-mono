import fnmatch
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from s2gos_utils.io.paths import exists, open_file
from s2gos_utils.scene.materials.spectrum import (
    FileSpectrum,
    InterpolatedSpectrum,
    SpectrumSpec,
    UniformSpectrum,
)
from upath import UPath


def _sanitize_material_id(material_id: str) -> str:
    """Sanitize material ID to ensure valid S2GOS identifier.

    Replaces characters that may cause issues in material references:
    - Dots (.) → underscores (_)
    - Hyphens (-) → underscores (_)

    Args:
        material_id: Original material ID from XML

    Returns:
        Sanitized material ID safe for use in S2GOS
    """
    return material_id.replace(".", "_").replace("-", "_")


def _parse_bsdf_element(bsdf_element) -> Dict[str, Any]:
    """Parse BSDF element to extract type and properties."""
    mat_data = {"type": bsdf_element.get("type", "diffuse"), "properties": {}}

    if mat_data["type"] == "twosided":
        nested_bsdf = bsdf_element.find("./bsdf")
        if nested_bsdf is not None:
            nested_data = _parse_bsdf_element(nested_bsdf)
            mat_data["nested_type"] = nested_data["type"]

            overlapping_props = set(mat_data["properties"].keys()) & set(
                nested_data["properties"].keys()
            )
            if overlapping_props:
                logging.warning(
                    f"Twosided material property collision: {overlapping_props} - nested properties will override parent"
                )

            mat_data["properties"].update(nested_data["properties"])

    for child in bsdf_element:
        if child.tag in ["rgb", "spectrum", "float", "string", "integer", "boolean"]:
            name = child.get("name")
            if name:
                mat_data["properties"][name] = _parse_property(child)

    return mat_data


def _parse_property(element) -> Any:
    """Parse individual property element following Mitsuba 3 specification.

    For spectrum elements, supports:
    - Inline wavelength:value pairs: "400:0.56, 500:0.18, 600:0.58"
    - Uniform spectrum: "0.5"
    - File-based: filename="spectrum.spd"

    Returns:
        Parsed property value in S2GOS-compatible format
    """
    tag = element.tag
    value = element.get("value", "")

    if tag == "rgb":
        try:
            rgb_values = [float(x) for x in value.split()]
            if len(rgb_values) == 3:
                return rgb_values
            elif len(rgb_values) == 1:
                return [rgb_values[0]] * 3
            else:
                logging.warning(
                    f"RGB value '{value}' has {len(rgb_values)} components, expected 3. Using default."
                )
                return [0.5, 0.5, 0.5]
        except (ValueError, IndexError):
            logging.warning(
                f"Failed to parse RGB value '{value}'. Using default [0.5, 0.5, 0.5]."
            )
            return [0.5, 0.5, 0.5]

    elif tag == "float":
        try:
            return float(value)
        except ValueError:
            logging.warning(
                f"Failed to parse float value '{value}'. Using default 0.0."
            )
            return 0.0

    elif tag == "integer":
        try:
            return int(value)
        except ValueError:
            logging.warning(
                f"Failed to parse integer value '{value}'. Using default 0."
            )
            return 0

    elif tag == "boolean":
        return value.lower() == "true"

    elif tag == "string":
        return value

    elif tag == "spectrum":
        return _parse_spectrum_element(element)

    return value


def _parse_inline_spectrum(value: str) -> Optional[Dict[str, Any]]:
    """Parse an inline ``"wl:val, wl:val, ..."`` spectrum string.

    Returns the interpolated-spectrum dict, or ``None`` when the string cannot be
    parsed — the caller decides the error message and fallback.
    """
    try:
        pairs = [p.strip().split(":") for p in value.split(",")]
        wavelengths = [float(p[0].strip()) for p in pairs]
        values_list = [float(p[1].strip()) for p in pairs]
    except (ValueError, IndexError) as e:
        logging.error(
            f"Failed to parse inline spectrum value '{value}': {e}. "
            "Expected format: 'wavelength:value, wavelength:value, ...'."
        )
        return None

    if wavelengths != sorted(wavelengths):
        logging.warning(
            f"Spectrum wavelengths are not in ascending order: {wavelengths}. "
            "This may cause interpolation issues."
        )
    for wl in wavelengths:
        if not (200 <= wl <= 4000):
            logging.warning(
                f"Wavelength {wl} nm is outside typical range [200, 4000] nm"
            )
    for val in values_list:
        if val < 0 or val > 1:
            logging.warning(
                f"Spectrum value {val} is outside typical reflectance range [0, 1]"
            )

    logging.debug(
        f"Parsed inline spectrum: {len(wavelengths)} wavelength points "
        f"from {wavelengths[0]:.1f} to {wavelengths[-1]:.1f} nm"
    )
    return {
        "type": "interpolated",
        "wavelengths": wavelengths,
        "values": values_list,
    }


def _parse_spectrum_element(element) -> Any:
    """Parse spectrum element following Mitsuba 3 specification.

    Args:
        element: XML element for spectrum

    Returns:
        dict: {"file": path} for file-based spectra
        dict: {"type": "interpolated", "wavelengths": [...], "values": [...]} for inline spectra
        float: scalar value for uniform spectra
    """
    spectrum_type = element.get("type")
    filename = element.get("filename")
    value = element.get("value", "")

    # Handle explicit type attribute from XML
    if spectrum_type == "uniform":
        # Uniform spectrum: single scalar value
        if not value:
            logging.warning("Uniform spectrum element has no value. Using default 0.5.")
            return 0.5

        try:
            return float(value)
        except ValueError:
            logging.error(
                f"Failed to parse uniform spectrum value '{value}' as float. "
                "Using default 0.5."
            )
            return 0.5

    elif spectrum_type == "interpolated":
        # Interpolated spectrum: can be file-based OR inline wavelength:value pairs
        if filename:
            # File-based interpolated spectrum
            return {"file": filename}

        if not value:
            logging.error(
                "Interpolated spectrum element has no value or filename. Using default 0.5."
            )
            return 0.5

        if ":" not in value:
            logging.error(
                f"Interpolated spectrum has value '{value}' but no ':' delimiter. "
                "Expected format: 'wavelength:value, wavelength:value, ...'. "
                "Using default 0.5."
            )
            return 0.5

        spectrum = _parse_inline_spectrum(value)
        return spectrum if spectrum is not None else 0.5

    else:
        if filename:
            return {"file": filename}

        if not value:
            logging.warning(
                "Spectrum element has no value or filename. Using default 0.5."
            )
            return 0.5

        try:
            return float(value)
        except ValueError:
            pass

        if ":" in value:
            spectrum = _parse_inline_spectrum(value)
            if spectrum is not None:
                return spectrum
            return 0.5

        logging.warning(
            f"Could not parse spectrum value '{value}'. "
            "Expected numeric value or 'wavelength:value, ...' format. Using default 0.5."
        )
        return 0.5


def _convert_materials(
    mitsuba_materials: Dict[str, Dict], xml_path: str
) -> Dict[str, Dict]:
    """Convert materials to S2GOS format using converter registry.

    Args:
        mitsuba_materials: Dictionary of material definitions from XML
        xml_path: UPath to source XML file (for resolving relative spectral paths)

    Returns:
        Dictionary of S2GOS material definitions
    """

    s2gos_materials = {}
    xml_dir = UPath(xml_path).parent

    for mat_id, mat_data in mitsuba_materials.items():
        sanitized_mat_id = _sanitize_material_id(mat_id)

        try:
            mat_type = mat_data.get("type", "diffuse")
            nested_type = mat_data.get("nested_type")

            if mat_type == "twosided" and nested_type:
                mat_type = nested_type

            logging.debug(
                f"Converting material '{mat_id}' → '{sanitized_mat_id}' of type '{mat_type}'"
            )
            converter = MATERIAL_CONVERTERS.get(mat_type, convert_diffuse)
            s2gos_materials[sanitized_mat_id] = converter(
                mat_data["properties"], xml_dir
            )

        except Exception as e:
            logging.warning(
                f"Failed to convert material '{mat_id}' → '{sanitized_mat_id}': {e}. "
                "Using diffuse fallback."
            )
            s2gos_materials[sanitized_mat_id] = convert_diffuse({}, xml_dir)

    return s2gos_materials


def convert_diffuse(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert diffuse material following Mitsuba 3 specification.

    Uses Pydantic spectrum models for validation and clear error messages.

    Args:
        props: Material properties from XML (already parsed by _parse_property)
        xml_dir: Directory containing source XML file (for resolving relative paths)

    Returns:
        S2GOS material definition with validated spectrum specification

    Raises:
        FileNotFoundError: If spectral file does not exist
        ValueError: If spectrum specification is invalid
    """
    reflectance = props.get("reflectance", [0.5, 0.5, 0.5])

    spectrum_spec: SpectrumSpec

    try:
        if isinstance(reflectance, dict) and "file" in reflectance:
            file_path = UPath(reflectance["file"])
            if not file_path.is_absolute():
                file_path = (xml_dir / file_path).resolve()

            if not exists(file_path):
                raise FileNotFoundError(
                    f"Spectral data file not found: {file_path}\n"
                    f"Original path: {reflectance['file']}\n"
                    f"XML directory: {xml_dir}"
                )

            logging.debug(f"Using file-based spectrum: {file_path}")
            spectrum_spec = FileSpectrum(path=str(file_path), variable="reflectance")

        elif (
            isinstance(reflectance, dict) and reflectance.get("type") == "interpolated"
        ):
            logging.debug(
                f"Using interpolated spectrum: {len(reflectance['wavelengths'])} wavelength points"
            )
            spectrum_spec = InterpolatedSpectrum(
                wavelengths=reflectance["wavelengths"], values=reflectance["values"]
            )

        elif isinstance(reflectance, (list, tuple, int, float)):
            value = (
                list(reflectance)
                if isinstance(reflectance, (list, tuple))
                else float(reflectance)
            )
            spectrum_spec = UniformSpectrum(value=value)

        else:
            logging.warning(
                f"Unexpected reflectance type {type(reflectance)} with value {reflectance}. "
                "Using default uniform [0.5, 0.5, 0.5]."
            )
            spectrum_spec = UniformSpectrum(value=[0.5, 0.5, 0.5])

    except Exception as e:
        logging.error(f"Spectrum validation failed: {e}")
        logging.warning("Using fallback uniform spectrum [0.5, 0.5, 0.5]")
        spectrum_spec = UniformSpectrum(value=[0.5, 0.5, 0.5])

    return {"type": "diffuse", "reflectance": spectrum_spec.model_dump()}


def convert_conductor(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert conductor material."""
    result = {"type": "conductor"}

    material_preset = props.get("material")
    if material_preset:
        result["material"] = material_preset
    else:
        result["material"] = "Cu"  # Default copper

    spec_refl = props.get("specular_reflectance")
    if spec_refl is not None:
        if isinstance(spec_refl, (list, tuple)):
            result["specular_reflectance"] = {
                "type": "uniform",
                "value": list(spec_refl),
            }
        elif isinstance(spec_refl, (int, float)):
            result["specular_reflectance"] = {
                "type": "uniform",
                "value": float(spec_refl),
            }

    return result


def convert_roughconductor(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert rough conductor material."""
    result = convert_conductor(props)
    result["type"] = "rough_conductor"
    result["distribution"] = props.get("distribution", "ggx")

    alpha_u = props.get("alpha_u")
    alpha_v = props.get("alpha_v")
    if alpha_u is not None or alpha_v is not None:
        if alpha_u is not None:
            result["alpha_u"] = float(alpha_u)
        if alpha_v is not None:
            result["alpha_v"] = float(alpha_v)
    else:
        alpha = props.get("alpha", props.get("roughness", 0.1))
        result["roughness"] = float(alpha)

    return result


def convert_dielectric(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert dielectric material."""
    result = {
        "type": "dielectric",
        "int_ior": float(props.get("int_ior", 1.5)),
        "ext_ior": float(props.get("ext_ior", 1.0)),
    }

    for prop_name in ["specular_reflectance", "specular_transmittance"]:
        prop_val = props.get(prop_name)
        if prop_val is not None:
            if isinstance(prop_val, (list, tuple)):
                result[prop_name] = {"type": "uniform", "value": list(prop_val)}
            elif isinstance(prop_val, (int, float)):
                result[prop_name] = {"type": "uniform", "value": float(prop_val)}

    return result


def convert_plastic(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert plastic material."""
    diffuse_refl = props.get("diffuse_reflectance", [0.5, 0.5, 0.5])

    if isinstance(diffuse_refl, (int, float)):
        diffuse_spec = {"type": "uniform", "value": float(diffuse_refl)}
    elif isinstance(diffuse_refl, (list, tuple)):
        diffuse_spec = {"type": "uniform", "value": list(diffuse_refl)}
    else:
        diffuse_spec = {"type": "uniform", "value": [0.5, 0.5, 0.5]}

    return {
        "type": "plastic",
        "diffuse_reflectance": diffuse_spec,
        "int_ior": float(props.get("int_ior", 1.49)),
        "ext_ior": float(props.get("ext_ior", 1.0)),
        "roughness": float(props.get("alpha", 0.01)),
        "nonlinear": bool(props.get("nonlinear", False)),
    }


def convert_bilambertian(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert bi-lambertian (two-sided diffuse) material.

    Args:
        props: Material properties from XML
        xml_dir: Directory containing source XML file (for resolving relative paths)

    Returns:
        S2GOS material definition with absolute paths

    Raises:
        FileNotFoundError: If spectral file does not exist
    """

    reflectance = props.get("reflectance", [0.5, 0.5, 0.5])
    transmittance = props.get("transmittance", [0.0, 0.0, 0.0])

    if isinstance(reflectance, dict) and "file" in reflectance:
        file_path = UPath(reflectance["file"])
        if not file_path.is_absolute():
            file_path = (xml_dir / file_path).resolve()

        if not exists(file_path):
            raise FileNotFoundError(
                f"Reflectance spectral data file not found: {file_path}\n"
                f"Original path: {reflectance['file']}\n"
                f"XML directory: {xml_dir}"
            )

        reflectance_spec = {
            "path": str(file_path),
            "variable": "reflectance",
        }
    elif isinstance(reflectance, (list, tuple)):
        reflectance_spec = {"type": "uniform", "value": list(reflectance)}
    elif isinstance(reflectance, (int, float)):
        reflectance_spec = {"type": "uniform", "value": float(reflectance)}
    else:
        reflectance_spec = {"type": "uniform", "value": [0.5, 0.5, 0.5]}

    if isinstance(transmittance, dict) and "file" in transmittance:
        file_path = UPath(transmittance["file"])
        if not file_path.is_absolute():
            file_path = (xml_dir / file_path).resolve()

        if not exists(file_path):
            raise FileNotFoundError(
                f"Transmittance spectral data file not found: {file_path}\n"
                f"Original path: {transmittance['file']}\n"
                f"XML directory: {xml_dir}"
            )

        transmittance_spec = {
            "path": str(file_path),
            "variable": "transmittance",
        }
    elif isinstance(transmittance, (list, tuple)):
        transmittance_spec = {"type": "uniform", "value": list(transmittance)}
    elif isinstance(transmittance, (int, float)):
        transmittance_spec = {"type": "uniform", "value": float(transmittance)}
    else:
        transmittance_spec = {"type": "uniform", "value": [0.0, 0.0, 0.0]}

    return {
        "type": "bilambertian",
        "reflectance": reflectance_spec,
        "transmittance": transmittance_spec,
    }


def convert_rpv(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Rahman Pinty Verstraete reflection model."""
    return {
        "type": "rpv",
        "rho_0": float(props.get("rho_0", 0.1)),
        "k": float(props.get("k", 0.5)),
        "g": float(props.get("g", -0.1)),
        "rho_c": float(props.get("rho_c", 0.0)),
    }


def convert_rtls(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Ross-Thick Li-Sparse reflection model."""
    return {
        "type": "rtls",
        "f_iso": float(props.get("f_iso", 1.0)),
        "f_geo": float(props.get("f_geo", 0.0)),
        "f_vol": float(props.get("f_vol", 0.0)),
    }


def convert_hapke(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Hapke surface model."""
    return {
        "type": "hapke",
        "w": float(props.get("w", 0.5)),
        "b": float(props.get("b", 0.0)),
        "c": float(props.get("c", 0.0)),
        "theta": float(props.get("theta", 0.0)),
        "B_0": float(props.get("B_0", 1.0)),
        "h": float(props.get("h", 0.06)),
    }


def convert_oceanic_grasp(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert GRASP oceanic model."""
    return {
        "type": "oceanic_grasp",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_oceanic_6s(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert legacy 6S oceanic model."""
    return {
        "type": "oceanic_6s",
        "wavelength": float(props.get("wavelength", 550.0)),
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "chlorinity": float(props.get("chlorinity", 0.0)),
        "pigmentation": float(props.get("pigmentation", 0.0)),
    }


def convert_oceanic_mishchenko(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert Mishchenko oceanic model."""
    return {
        "type": "oceanic_mishchenko",
        "wind_speed": float(props.get("wind_speed", 5.0)),
        "water_ior": float(props.get("water_ior", 1.33)),
    }


def convert_selectbsdf(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert selector BSDF (texture-based material selection)."""
    return {
        "type": "selectbsdf",
        "texture": props.get("texture", "uniform"),
        "materials": props.get("materials", {}),
    }


def convert_measured(props: Dict[str, Any], xml_dir=None) -> Dict[str, Any]:
    """Convert measured quasi-diffuse material."""
    return {
        "type": "measured",
        "data": props.get("data", ""),
        "scale": float(props.get("scale", 1.0)),
    }


MATERIAL_CONVERTERS = {
    "diffuse": convert_diffuse,
    "conductor": convert_conductor,
    "roughconductor": convert_roughconductor,
    "dielectric": convert_dielectric,
    "plastic": convert_plastic,
    "bilambertian": convert_bilambertian,
    "rpv": convert_rpv,
    "rtls": convert_rtls,
    "hapke": convert_hapke,
    "oceanic_grasp": convert_oceanic_grasp,
    "oceanic_6s": convert_oceanic_6s,
    "oceanic_mishchenko": convert_oceanic_mishchenko,
    "selectbsdf": convert_selectbsdf,
    "measured": convert_measured,
}


def _match_filename(filename: str, pattern: str, mode: str = "glob") -> bool:
    """Check if filename matches pattern using specified matching strategy.

    Args:
        filename: PLY filename (stem, no extension)
        pattern: Pattern to match against
        mode: Matching mode - "glob" for shell-style wildcards or "regex" for regular expressions

    Returns:
        True if filename matches pattern
    """
    if mode == "glob":
        return fnmatch.fnmatch(filename, pattern)
    elif mode == "regex":
        try:
            return bool(re.match(pattern, filename))
        except re.error as e:
            logging.error(
                f"Invalid regex pattern '{pattern}': {e}. Pattern will not match anything."
            )
            return False
    else:
        raise ValueError(
            f"Unknown matching mode: {mode}. Use 'glob' or 'regex'.\n"
            f"Examples:\n"
            f"  - glob: 'tree_*', '?_ground', '*vegetation*'\n"
            f"  - regex: r'tree_\\d+', r'(oak|pine)_.*'"
        )


_INSTANCE_MARKER = '<shape type="instance"'


def _read_scene_header(xml_path: str) -> Tuple[str, bool]:
    """Return the XML text up to the first ``<instance>`` shape, and whether one exists.

    Instanced scenes list their (potentially hundreds of thousands of) ``<instance>``
    placements *after* a small header of BSDFs, top-level shapes, and shapegroups.
    Reading only that header keeps parsing cheap and never builds the full DOM. For an
    ordinary asset XML with no instances, the whole (small) file is the header.
    """
    buffer = ""
    with open_file(xml_path, "r") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                return buffer, False
            buffer += chunk
            idx = buffer.find(_INSTANCE_MARKER)
            if idx != -1:
                return buffer[:idx] + "</scene>\n", True


# Placeholder material id for a shape that declares no ``<ref name="bsdf">``.
NO_BSDF = "default-bsdf"


def _parse_scene_shape(shape, xml_dir: "UPath") -> Optional[Dict[str, Any]]:
    """Parse a ``ply`` or ``ellipsoidsmesh`` shape into a component descriptor.

    Returns ``{"type", "file" (absolute source path), "material", ...}`` or ``None``
    for an unsupported/filename-less shape. A shape without a bsdf reference gets the
    :data:`NO_BSDF` placeholder material id.
    """
    stype = shape.get("type")
    if stype not in ("ply", "ellipsoidsmesh"):
        return None
    filename_elem = shape.find('./string[@name="filename"]')
    if filename_elem is None or not filename_elem.get("value"):
        logging.warning("Skipping %s shape without a filename", stype)
        return None

    src = UPath(filename_elem.get("value"))
    if not src.is_absolute():
        src = (xml_dir / src).resolve()

    ref = shape.find('./ref[@name="bsdf"]')
    material = (
        _sanitize_material_id(ref.get("id"))
        if ref is not None and ref.get("id")
        else NO_BSDF
    )
    component: Dict[str, Any] = {"type": stype, "file": str(src), "material": material}

    if stype == "ply":
        fn = shape.find('./boolean[@name="face_normals"]')
        if fn is not None:
            component["face_normals"] = fn.get("value", "false").lower() == "true"
    else:  # ellipsoidsmesh
        extent = shape.find('./float[@name="extent"]')
        component["extent"] = (
            float(extent.get("value"))
            if extent is not None and extent.get("value")
            else 1.0
        )
    return component


@dataclass(frozen=True)
class ParsedMitsubaScene:
    """A Mitsuba scene reduced to what the generator can place.

    Attributes:
        materials: ``{sanitized_bsdf_id: material_def}`` converted to s2gos form via
            the material converter registry.
        shapes: Top-level placeable ``ply`` components, each
            ``{"type", "file", "material", ...}`` with an absolute source path.
        shapegroups: ``[{"id", "components"}]`` where components are ``ply`` or
            ``ellipsoidsmesh`` descriptors.
        instanced: Whether the scene lists ``<instance>`` placements (read lazily via
            :func:`read_instance_transforms`).
    """

    materials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shapes: List[Dict[str, Any]] = field(default_factory=list)
    shapegroups: List[Dict[str, Any]] = field(default_factory=list)
    instanced: bool = False


def parse_mitsuba_scene(xml_path: str) -> ParsedMitsubaScene:
    """Parse the placeable content of a Mitsuba XML scene (flat or instanced).

    Reads only the pre-``<instance>`` header (see :func:`_read_scene_header`), so it is
    cheap even for multi-hundred-MB instanced scenes; a flat asset XML is parsed whole.
    ``<instance>`` placements are never loaded here — the caller streams them at
    generation time with :func:`read_instance_transforms`.
    """
    header_xml, instanced = _read_scene_header(xml_path)
    try:
        root = ET.fromstring(header_xml)
    except ET.ParseError as e:
        raise ET.ParseError(
            f"Failed to parse Mitsuba XML '{xml_path}': {e}. "
            "Please check that the file is valid Mitsuba XML."
        ) from e

    xml_dir = UPath(xml_path).parent

    raw_materials: Dict[str, Any] = {}
    for bsdf in root.findall(".//bsdf"):
        material_id = bsdf.get("id")
        if material_id:
            raw_materials[material_id] = _parse_bsdf_element(bsdf)
    materials = _convert_materials(raw_materials, xml_path)

    shapegroups: List[Dict[str, Any]] = []
    for sg in root.findall('./shape[@type="shapegroup"]'):
        components = [
            c
            for c in (
                _parse_scene_shape(child, xml_dir) for child in sg.findall("./shape")
            )
            if c is not None
        ]
        shapegroups.append({"id": sg.get("id"), "components": components})

    shapes: List[Dict[str, Any]] = []
    for shp in root.findall("./shape"):
        if shp.get("type") in ("shapegroup", "instance"):
            continue
        component = _parse_scene_shape(shp, xml_dir)
        if component is None:
            continue
        if component["type"] == "ellipsoidsmesh":
            logging.warning(
                "Top-level ellipsoidsmesh in %s is not placeable; wrap it in a "
                "shapegroup. Skipping.",
                xml_path,
            )
            continue
        shapes.append(component)

    logging.info(
        "Parsed Mitsuba scene %s: %d material(s), %d top-level shape(s), "
        "%d shapegroup(s)%s",
        xml_path,
        len(materials),
        len(shapes),
        len(shapegroups),
        " + instances" if instanced else "",
    )
    return ParsedMitsubaScene(materials, shapes, shapegroups, instanced)


def _parse_vec(element, default: float) -> np.ndarray:
    """Read a Mitsuba ``x``/``y``/``z`` or ``value`` vector as ``(3,)`` floats.

    ``value`` may be comma- or space-separated. Missing per-axis attributes fall back
    to ``default`` (0 for translate/rotate axes, 1 for scale).
    """
    value = element.get("value")
    if value is not None:
        parts = [float(p) for p in value.replace(",", " ").split()]
        if len(parts) == 1:  # a single value applies to every axis (e.g. uniform scale)
            return np.array(parts * 3, dtype="float64")
        if len(parts) == 3:
            return np.array(parts, dtype="float64")
        raise ValueError(f"Expected 1 or 3 components in '{value}', got {len(parts)}")
    return np.array(
        [float(element.get(a, default)) for a in ("x", "y", "z")], dtype="float64"
    )


def _op_to_matrix(child) -> np.ndarray:
    """Convert a single ``<transform>`` child element to a ``(4, 4)`` matrix."""
    from scipy.spatial.transform import Rotation

    tag = child.tag
    m = np.eye(4)
    if tag == "translate":
        m[:3, 3] = _parse_vec(child, 0.0)
    elif tag == "scale":
        m[:3, :3] = np.diag(_parse_vec(child, 1.0))
    elif tag == "rotate":
        axis = _parse_vec(child, 0.0)
        norm = float(np.linalg.norm(axis))
        angle = float(child.get("angle", 0.0))
        if norm == 0.0:
            if angle != 0.0:
                raise ValueError("<rotate> has a zero axis but a nonzero angle")
            return m
        m[:3, :3] = Rotation.from_rotvec(axis / norm * np.radians(angle)).as_matrix()
    elif tag == "matrix":
        values = [float(p) for p in child.get("value", "").replace(",", " ").split()]
        if len(values) == 16:
            m = np.array(values, dtype="float64").reshape(4, 4)
        elif len(values) == 9:
            m[:3, :3] = np.array(values, dtype="float64").reshape(3, 3)
        else:
            raise ValueError(f"<matrix> expects 9 or 16 values, got {len(values)}")
    else:
        raise NotImplementedError(
            f"Unsupported <instance> transform operation '<{tag}>' "
            "(supported: translate, rotate, scale, matrix)."
        )
    return m


def _transform_to_matrix(transform) -> np.ndarray:
    """Compose a ``<transform>`` element into a ``(4, 4)`` object-to-world matrix."""
    m = np.eye(4)
    for child in transform:
        m = _op_to_matrix(child) @ m
    return m


def read_instance_transforms(xml_path: str) -> Dict[str, np.ndarray]:
    """Stream ``<shape type="instance">`` placements into per-shapegroup transforms.

    Uses :func:`xml.etree.ElementTree.iterparse` and clears each element, so the
    (potentially hundreds of thousands of) instances never live in memory at once.

    Each instance's ``<transform name="to_world">`` is composed into a ``(4, 4)``
    object-to-world matrix (``translate``/``rotate``/``scale``/``matrix`` supported.
    Instances without a shapegroup ``<ref>`` are skipped with a warning.

    Returns ``{shapegroup_id: (N, 4, 4) float64}`` of instance transforms.
    """
    transforms: Dict[str, List[np.ndarray]] = {}
    skipped_no_ref = 0
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)  # the root element, so completed children can be dropped
    for event, elem in context:
        if event != "end" or elem.tag != "shape" or elem.get("type") != "instance":
            continue
        ref = elem.find("./ref")
        shapegroup_id = ref.get("id") if ref is not None else None
        if shapegroup_id is None:
            skipped_no_ref += 1
        else:
            transform = elem.find('./transform[@name="to_world"]')
            matrix = np.eye(4) if transform is None else _transform_to_matrix(transform)
            transforms.setdefault(shapegroup_id, []).append(matrix)
        root.clear()

    if skipped_no_ref:
        logging.warning(
            "Skipped %d <instance> shape(s) without a shapegroup <ref> in %s",
            skipped_no_ref,
            xml_path,
        )
    return {k: np.asarray(v, dtype=np.float64) for k, v in transforms.items()}


def create_tree_shapegroup(
    tree_xml_path: str, output_dir: Optional["UPath"] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create Mitsuba shapegroup from tree XML file.

    Args:
        tree_xml_path: UPath to tree XML file
        output_dir: Scene output directory where mesh files will be copied

    Returns:
        Tuple of (shapegroup definition, converted material definitions)
    """
    scene = parse_mitsuba_scene(tree_xml_path)
    materials = scene.materials

    shapegroup = {"type": "shapegroup", "id": "tree_group"}

    if output_dir:
        tree_meshes_dir = UPath(output_dir) / "meshes" / "tree"
        tree_meshes_dir.mkdir(parents=True, exist_ok=True)

    for i, shape in enumerate(scene.shapes):
        shape_name = f"tree_component_{i}"

        # Material ids from parse_mitsuba_scene are already sanitized.
        material_id = shape["material"]

        source_file_path = UPath(shape["file"])

        if output_dir and source_file_path.exists():
            from s2gos_utils.io.paths import copy

            dest_filename = source_file_path.name
            dest_path = tree_meshes_dir / dest_filename

            if not dest_path.exists():
                copy(source_file_path, dest_path)
                logging.info(f"Copied tree mesh: {dest_filename}")

            mesh_filename = f"meshes/tree/{dest_filename}"
        else:
            mesh_filename = str(source_file_path)
            if output_dir and not source_file_path.exists():
                logging.warning(f"Tree mesh file not found: {source_file_path}")

        shapegroup[shape_name] = {
            "type": "ply",
            "filename": mesh_filename,
            "face_normals": True,
            "bsdf": {"type": "ref", "id": f"_mat_{material_id}"},
        }

    return shapegroup, materials
