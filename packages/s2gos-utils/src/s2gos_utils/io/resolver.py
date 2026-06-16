from __future__ import annotations

import os

import attrs
from upath import UPath

from .paths import PathLike, to_upath


def _validator_path_exists(instance, attribute, value: UPath):
    """Validator that checks if a UPath exists (works for local and remote)."""
    if not value.exists():
        raise FileNotFoundError(f"Path does not exist: {value}")


def _to_search_path(path: PathLike) -> UPath:
    """Normalize a search-path entry to a ``UPath``, resolving local paths.

    Routes through ``to_upath`` so ``PathRef`` credentials are applied, then
    resolves only local (``file``) paths — remote paths have no meaningful
    ``resolve()``.
    """
    upath = to_upath(path)
    if upath.protocol == "file":
        upath = upath.resolve()
    return upath


@attrs.define
class FileResolver:
    """File resolver with UPath support for local and remote paths."""

    paths: list[UPath] = attrs.field(
        factory=list,
        converter=lambda value: [_to_search_path(x) for x in value],
        validator=attrs.validators.deep_iterable(_validator_path_exists),
    )

    def append(self, path: PathLike, avoid_duplicates: bool = True) -> None:
        """Append an entry to the end of the list of search paths.

        Args:
            path: Path to add to search paths (local or remote)
            avoid_duplicates: If True, avoid adding duplicate paths
        """
        upath = _to_search_path(path)

        if not upath.exists():
            raise FileNotFoundError(f"Path does not exist: {upath}")

        if avoid_duplicates:
            if upath not in self.paths:
                self.paths.append(upath)
        else:
            self.paths.append(upath)

    def prepend(self, path: PathLike, avoid_duplicates: bool = True) -> None:
        """Prepend an entry at the beginning of the list of search paths.

        Args:
            path: Path to add to search paths (local or remote)
            avoid_duplicates: If True, avoid adding duplicate paths
        """
        upath = _to_search_path(path)

        if not upath.exists():
            raise FileNotFoundError(f"Path does not exist: {upath}")

        if avoid_duplicates:
            if upath not in self.paths:
                self.paths.insert(0, upath)
        else:
            self.paths.insert(0, upath)

    def clear(self) -> None:
        """Clear the list of search paths."""
        self.paths.clear()

    def resolve(self, path: PathLike, strict: bool = True) -> UPath:
        """Resolve a path by searching registered locations in order.

        Args:
            path: Path to be resolved
            strict: If True (default), resolution failure will raise FileNotFoundError.
                   Set to False to return unresolved path (not recommended).

        Returns:
            Resolved UPath object

        Raises:
            FileNotFoundError: If strict=True and path not found in any search location
        """
        upath = to_upath(path)

        # NOTE: The "https" protocol returns False on calls to `exists`. This is
        # a hack to go around the issue. We need to understand which protocols
        # are unreliable and either find a reliable check or bypass the list
        # of unreliable protocols.
        if upath.is_absolute() and upath.protocol == "https":
            return upath

        # If already absolute or remote and exists, return as-is
        if (upath.is_absolute() or upath.protocol != "file") and upath.exists():
            return upath

        if not upath.is_absolute():
            for base in self.paths:
                candidate = base / upath
                if candidate.exists():
                    return candidate

        if strict:
            search_paths_str = "\n  - ".join([str(p) for p in self.paths])
            raise FileNotFoundError(
                f"Could not resolve '{upath}' in any search path.\n"
                f"Searched in:\n  - {search_paths_str}\n\n"
                f"To fix this:\n"
                f"  1. Check that the file exists in one of the above locations\n"
                f"  2. Add additional search paths via S2GOS_SEARCH_PATHS environment variable\n"
                f"  3. Provide an absolute path instead of a relative filename"
            )

        return upath

    def get_search_info(self) -> dict:
        """Get information about current search configuration."""
        return {
            "search_paths": [str(p) for p in self.paths],
            "path_count": len(self.paths),
        }

    @classmethod
    def from_environment(cls, prefix: str = "S2GOS") -> FileResolver:
        """Create resolver from environment variables.

        Looks for variables like:
        - S2GOS_SEARCH_PATHS="./data,./config,s3://defaults/"

        Args:
            prefix: Environment variable prefix

        Returns:
            Configured FileResolver instance
        """
        resolver = cls()

        # Load search paths from environment
        search_paths_env = os.getenv(f"{prefix}_SEARCH_PATHS")
        if search_paths_env:
            paths = [p.strip() for p in search_paths_env.split(",")]
            for path in paths:
                try:
                    resolver.append(path)
                except FileNotFoundError:
                    # Skip non-existent paths from environment
                    pass

        return resolver


def create_default_resolver() -> FileResolver:
    """Create a resolver with smart defaults for s2gos."""
    search_paths = []

    resolver = FileResolver()
    for path in search_paths:
        try:
            resolver.append(path)
        except FileNotFoundError:
            # Skip non-existent paths gracefully
            pass

    # Override with environment if available
    env_resolver = FileResolver.from_environment()
    if env_resolver.paths:
        # Prepend environment paths (highest priority)
        for path in reversed(env_resolver.paths):
            resolver.prepend(str(path))

    return resolver


# Global elegant resolver instance - the heart of the new system
resolver = create_default_resolver()
