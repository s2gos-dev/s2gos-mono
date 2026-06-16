from __future__ import annotations

import json
import os
from typing import Annotated, Any, BinaryIO, Dict, Optional, TextIO, Union

import geopandas as gpd
import pandas as pd
import xarray as xr
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    PrivateAttr,
    WithJsonSchema,
    model_serializer,
    model_validator,
)
from pydantic_core import CoreSchema
from upath import UPath


class PathRef(BaseModel):
    """
    Path configuration that preserves credential reference through serialization.

    This model allows paths to reference credentials by ID rather than embedding
    actual credentials. When serialized to JSON it is structurally an OGC ``Link``:
    ``{"href": <path/URI>, "x-cid": <credential id>}`` (only the path and the
    credential reference are stored, never actual credentials). When deserialized,
    credentials are resolved from the credential provider (environment variables or
    .secrets.yaml).

    Attributes:
        href: The actual path/URI (the OGC Link ``href``).
        rel: Optional Link relation type.
        type: Optional Link media (MIME) type.
        hreflang: Optional language of the linked resource.
        title: Optional human-readable title.
        options: Optional extra client options (serialized as ``x-options``).
        cid: Optional reference to a Credential ID in the credential provider.
            Serialized as the ``x-cid`` Link extension.

    Example:
        # With credentials
        path = PathRef(
            "https://data.earthdatahub.destine.eu/data.zarr",
            cid="earthdatahub"
        )

        # Without credentials (public or local path)
        path = PathRef("/local/path/data.zarr")

        # Access the authenticated UPath
        upath = path.upath
    """

    href: str = Field(description="Full path URI")
    rel: str | None = Field(default=None, description="Relation type")
    type: str | None = Field(default=None, description="Media (MIME) type")
    hreflang: str | None = Field(
        default=None, description="Language of the linked resource"
    )
    title: str | None = Field(default=None, description="Human-readable title")
    options: Annotated[dict[str, Any] | None, WithJsonSchema({"type": "object"})] = (
        Field(
            default=None,
            alias="x-options",
            description="Extra client options for accessing href in its storage",
        )
    )
    cid: str | None = Field(default=None, alias="x-cid", description="Credential ID")
    _upath: UPath | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        frozen=True, populate_by_name=True, serialize_by_alias=True
    )

    def __init__(self, href, cid=None, **kwargs):
        super().__init__(href=href, cid=cid, **kwargs)

    @model_serializer(mode="wrap")
    def _omit_unset(self, handler) -> dict[str, Any]:
        """Serialize like an OGC Link: drop optional fields that are unset."""
        return {k: v for k, v in handler(self).items() if v is not None}

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        """Normalize str / PathLike / PathRef / dict input, preserving Link fields."""
        if isinstance(data, PathRef):
            return data.model_dump()
        if isinstance(data, dict):
            d = dict(data)
            href = d.get("href")
            if isinstance(href, PathRef):
                base = href.model_dump()
                base.update(
                    {k: v for k, v in d.items() if k != "href" and v is not None}
                )
                return base
            if href is not None:
                d["href"] = str(href)
            return d
        return {"href": str(data)}

    @property
    def upath(self) -> UPath:
        """
        Get the authenticated UPath by resolving credentials.

        If `cid` is set, retrieves the credential from the credential
        provider and constructs an authenticated UPath. Otherwise returns a
        simple UPath without authentication.

        Returns:
            UPath object with authentication if `cid` is set

        Raises:
            CredentialNotFoundError: If `cid` is set but the credential is not
                found in the active provider.
        """
        if self._upath is not None:
            return self._upath

        if self.cid:
            from s2gos_utils.setting.credentials import get_credential

            cred = get_credential(self.cid)
            kwargs = cred.upath_kwargs
            self._upath = UPath(self.href, **kwargs)
        else:
            # No credentials needed (local path or public URL)
            self._upath = UPath(self.href)

        return self._upath

    def to_dict(self) -> dict[str, Any]:
        """Alias to `model_dump`."""
        return self.model_dump()

    @property
    def _lexical(self) -> UPath:
        """A credential-free ``UPath`` for pure path manipulation.

        Used by the lexical operations (``/``, ``parent``, ``name``, ...) so they
        never resolve credentials — path math should not require secrets. Use
        ``.upath`` instead whenever you actually touch storage.
        """
        return UPath(self.href)

    def __truediv__(self, other) -> PathRef:
        """Returns the joined path, preserving the credential id."""

        if isinstance(other, PathRef):
            if other.cid != self.cid:
                raise ValueError(
                    f"Joining paths with different credential ids! "
                    f"Left: {self.cid}, Right: {other.cid}."
                )
            other = other.href

        return PathRef(self._lexical / other, self.cid)

    @property
    def parent(self) -> PathRef:
        return PathRef(self._lexical.parent, self.cid)

    @property
    def name(self) -> str:
        return self._lexical.name

    @property
    def stem(self) -> str:
        return self._lexical.stem

    @property
    def suffix(self) -> str:
        return self._lexical.suffix

    def __str__(self) -> str:
        """Return the path value as a string."""
        return self.href

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        # Strip title and description so that when PathRef is inlined into a
        # field's anyOf schema, the field-level title and description (from
        # Field() annotations) are not overwritten by PathRef's class name and
        # docstring.
        schema = handler(core_schema)
        schema.pop("title", None)
        schema.pop("description", None)
        return schema


PathLike = PathRef | str | os.PathLike


# PATH UTILITY FUNCTIONS


def to_upath(path: PathLike) -> UPath:
    """Resolve a path to an authenticated ``UPath`` (the fsspec handle)."""
    if isinstance(path, PathRef):
        return path.upath
    if isinstance(path, UPath):
        return path
    return UPath(path)


def open_file(path: PathLike, mode: str = "r", **kwargs) -> Union[TextIO, BinaryIO]:
    """Open a file using UPath for unified access across storage backends.

    Args:
        path: Path to file (local or remote).
        mode: File mode ('r', 'rb', 'w', 'wb', etc.).
        **kwargs: Additional arguments for UPath.open().

    Returns:
        A file object.
    """
    return to_upath(path).open(mode=mode, **kwargs)


def read_feather(path: PathLike, **kwargs) -> pd.DataFrame:
    """Read a feather file from any backend supported by fsspec.

    Args:
        path: Path to the feather file.
        **kwargs: Additional arguments for pd.read_feather().

    Returns:
        A pandas DataFrame.
    """
    with open_file(path, "rb") as f:
        return pd.read_feather(f, **kwargs)


def read_geofeather(path: PathLike, **kwargs) -> gpd.GeoDataFrame:
    """Read a GeoFeather file as a GeoDataFrame from any backend.

    Args:
        path: Path to the feather file.
        **kwargs: Additional arguments for gpd.read_feather().

    Returns:
        A GeoDataFrame.
    """
    with open_file(path, "rb") as f:
        return gpd.read_feather(f, **kwargs)


def read_json(path: PathLike, **kwargs) -> Dict[str, Any]:
    """Read a JSON file from any backend supported by fsspec.

    Args:
        path: Path to the JSON file.
        **kwargs: Additional arguments for json.load().

    Returns:
        A dictionary with the JSON content.
    """
    with open_file(path, "r") as f:
        return json.load(f, **kwargs)


def read_yaml(path: PathLike, **kwargs) -> Dict[str, Any]:
    """Read a YAML file from any backend supported by fsspec.

    Args:
        path: Path to the YAML file.
        **kwargs: Additional arguments for yaml.safe_load().

    Returns:
        A dictionary with the YAML content.
    """
    with open_file(path, "r") as f:
        return yaml.safe_load(f, **kwargs)


def open_dataarray(
    path: PathLike,
    engine: str | None = None,
    fsspec_caching: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    """Open an xarray DataArray, letting xarray handle the fsspec backend.

    Args:
        path: Path to the data file.
        **kwargs: Additional arguments for xr.open_dataarray().

    Returns:
        An xarray DataArray.
    """
    path = to_upath(path)

    if engine is None:
        engine = xr.backends.plugins.guess_engine(path.path)

    # Check for storage options for authentication
    if len(path.storage_options) > 0 and "storage_options" not in kwargs:
        kwargs["storage_options"] = path.storage_options

    if fsspec_caching is not None:
        fs = path.fs
        if engine == "netcdf4":
            import fsspec

            return xr.open_dataarray(
                fsspec.open_local(
                    f"simplecache::{str(path)}", simplecache=fsspec_caching
                ),
                engine=engine,
                **kwargs,
            )
        else:
            return xr.open_dataarray(
                fs.open(str(path), **fsspec_caching), engine=engine, **kwargs
            )

    return xr.open_dataarray(str(path), engine=engine, **kwargs)


def open_dataset(
    path: PathLike,
    engine: str | None = None,
    fsspec_caching: dict | None = None,
    **kwargs,
):
    """
    Open an xarray Dataset.
    Uses `universal_pathlib` (`UPath`) to handle remote location access by
    passing the `storage_options` when relevant and uses `fsspec` to handle by
    opening using the `FileSystem.open` method directly.
    Note that the `netcdf4` engine can only use a local caching strategy. In such
    cases, `fsspec_caching` is passed to the `simplecache` argument of
    `fsspec.open_local`.

    Args:
        path: Path to the data file.
        engine: The backend engine used by xarray.
        fsspec_caching:
            Kwargs arguments for fsspec caching. for engine="netcdf4",
            this is passed to `simplecache`.
        **kwargs: Additional arguments for xr.open_dataset().

    Returns:
        An xarray Dataset.
    """
    path = to_upath(path)

    # Will trigger an helpful assert if it cannot find the proper engine, which
    # gets obfuscated by the storage option exception otherwise.
    if engine is None:
        engine = xr.backends.plugins.guess_engine(path.path)

    # Check for storage options for authentication
    if len(path.storage_options) > 0 and "storage_options" not in kwargs:
        kwargs["storage_options"] = path.storage_options

    if fsspec_caching is not None:
        fs = path.fs
        if engine == "netcdf4":
            import fsspec

            return xr.open_dataset(
                fsspec.open_local(
                    f"simplecache::{str(path)}", simplecache=fsspec_caching
                ),
                engine=engine,
                **kwargs,
            )
        else:
            return xr.open_dataset(
                fs.open(str(path), **fsspec_caching), engine=engine, **kwargs
            )

    return xr.open_dataset(str(path), engine=engine, **kwargs)


def is_remote_path(path: PathLike) -> bool:
    """Check if a path is a remote URL using UPath protocol detection."""
    return to_upath(path).protocol != "file"


def is_absolute_path(path: PathLike) -> bool:
    """Check if a path is absolute (works for both local and remote paths)."""
    upath = to_upath(path)
    return upath.protocol != "file" or upath.is_absolute()


def exists(path: PathLike) -> bool:
    """Check if path exists (local or remote) using UPath."""
    return to_upath(path).exists()


def mkdir(
    path: PathLike, parents: bool = True, exist_ok: bool = True, **kwargs
) -> None:
    """Create directory using UPath (supports local and some remote protocols)."""
    p = to_upath(path)
    protocol = getattr(p, "protocol", None)
    if protocol in ("s3", "s3a"):
        return  # S3 has no real directories; objects are created on write
    p.mkdir(parents=parents, exist_ok=exist_ok, **kwargs)


def copy(src: PathLike, dst: PathLike, **kwargs) -> None:
    """Create directory using UPath (supports local and some remote protocols)."""
    to_upath(src).copy(to_upath(dst), **kwargs)


def write_image(image, path: PathLike, format: str = "PNG") -> None:
    """Write a PIL Image to any backend supported by fsspec.

    Uses an intermediate BytesIO buffer because PIL cannot write directly to
    fsspec/UPath file objects.
    """
    import io

    buf = io.BytesIO()
    image.save(buf, format=format)
    with open_file(path, "wb") as f:
        f.write(buf.getvalue())


def optional_str(path: Optional[PathLike]) -> Optional[str]:
    """Convert a path-like object to a string, handling None elegantly."""
    return str(path) if path is not None else None


def normalize_path(path: PathLike) -> str:
    """Normalize a path using UPath for consistent handling.

    Always returns a string for configuration compatibility.
    """
    return str(to_upath(path))


def expand_mapper(path: PathLike):
    """Expands a path to a FSMapper."""
    upath = to_upath(path)
    return upath.fs.get_mapper(upath.path)
