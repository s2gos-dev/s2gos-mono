import logging
from typing import Dict, Optional, Tuple


def get_package_version(package_name: str) -> str:
    """Get actual installed package version.

    Args:
        package_name: Name of the package to get version for

    Returns:
        Version string from installed package, or "0.0.1" as fallback
    """
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        # Fallback version if package not found or metadata unavailable
        return "0.0.1"


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch) tuple.

    Args:
        version_str: Version string in format "major.minor.patch"

    Returns:
        Tuple of (major, minor, patch) integers

    Raises:
        ValueError: If version format is invalid
    """
    try:
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError("Version must have exactly 3 parts (major.minor.patch)")
        return tuple(int(part) for part in parts)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid version format '{version_str}': {e}")


def check_version_compatibility(
    config_version: str, current_version: str, component_name: str = "configuration"
) -> None:
    """Check if config version is compatible with current code version.

    Args:
        config_version: Version from loaded config
        current_version: Current code version
        component_name: Name of component for error messages

    Raises:
        ValueError: If major version mismatch (incompatible)

    Logs:
        Warning: If minor/patch version differences exist
    """
    try:
        config_major, config_minor, config_patch = parse_version(config_version)
        current_major, current_minor, current_patch = parse_version(current_version)
    except ValueError as e:
        logging.warning(f"Could not parse {component_name} version numbers: {e}")
        return

    # Major version mismatch is incompatible
    if config_major != current_major:
        raise ValueError(
            f"Incompatible {component_name} version: config uses v{config_version} "
            f"but current code expects v{current_major}.x.x. "
            f"Please update your {component_name} or use compatible code version."
        )

    # Minor/patch differences get warnings
    if config_minor != current_minor or config_patch != current_patch:
        if config_minor > current_minor or (
            config_minor == current_minor and config_patch > current_patch
        ):
            logging.warning(
                f"{component_name.title()} version v{config_version} is newer than current code v{current_version}. "
                f"Some features may not work as expected."
            )
        else:
            logging.warning(
                f"{component_name.title()} version v{config_version} is older than current code v{current_version}. "
                f"Consider updating your {component_name}."
            )


def validate_config_version(
    component_type: str,
    config_data: Dict,
    current_version: str,
    component_name: Optional[str] = None,
) -> Dict:
    """Validate config version - simple version checking only.

    Args:
        component_type: Type of component being validated
        config_data: Configuration data dictionary
        current_version: Current code's expected version
        component_name: Human-readable component name for messages

    Returns:
        Configuration dictionary (unchanged)

    Raises:
        ValueError: If version is incompatible
    """
    version_field = _get_version_field_name(component_type)
    config_version = config_data.get(version_field)
    component_name = component_name or component_type.replace("_", " ")

    if config_version is None:
        logging.warning(
            f"No version field found in {component_name} - proceeding without validation"
        )
        return config_data

    # Simple version compatibility check
    check_version_compatibility(config_version, current_version, component_name)
    return config_data


def _get_version_field_name(component_type: str) -> str:
    """Get the appropriate version field name for a component type."""
    if component_type == "scene_description":
        return "schema_version"
    elif component_type == "material_config":
        return "version"
    else:
        return "config_version"


def get_version_info() -> Dict[str, str]:
    """Get version information for common S2GOS components.

    Note: This is a legacy function. Each package should now manage
    its own version information using get_package_version().

    Returns:
        Dictionary mapping component names to their current versions
    """
    return {
        "s2gos-utils": get_package_version("s2gos-utils"),
    }


def version_stamp() -> Dict[str, str]:
    """Create a version stamp for inclusion in output metadata.

    Returns:
        Dictionary with version information suitable for file metadata
    """
    return {
        "s2gos_version_info": get_version_info(),
        "versioning_system": "semantic",
    }
