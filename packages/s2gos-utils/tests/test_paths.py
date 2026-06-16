from pathlib import Path

import pytest
from upath import UPath

from s2gos_utils.io.paths import (
    PathRef,
    exists,
    is_absolute_path,
    is_remote_path,
    mkdir,
    normalize_path,
    optional_str,
    to_upath,
)
from s2gos_utils.setting.credentials.credential import S3Credential
from s2gos_utils.setting.credentials.exceptions import CredentialNotFoundError
from s2gos_utils.setting.credentials.provider import (
    DictCredentialProvider,
    set_credential_provider,
)


@pytest.fixture
def mock_s3_credential():
    """Create a mock S3 credential for testing."""
    cred = S3Credential(
        id="test_s3",
        key="test_key",
        secret="test_secret",
        endpoint_url="https://s3.example.com",
    )
    provider = DictCredentialProvider({"test_s3": cred})
    set_credential_provider(provider)
    yield cred
    # Reset provider after test
    set_credential_provider(None)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    return test_dir


class TestPathRefCreation:
    """Test PathRef creation with various input types."""

    def test_pathref_from_string(self):
        """Test creating PathRef from string, with and without cid."""
        # Without cid
        pr = PathRef("/tmp/test")
        assert pr.href == "/tmp/test"
        assert pr.cid is None

        # With cid
        pr = PathRef("s3://bucket/path", cid="test_cred")
        assert pr.href == "s3://bucket/path"
        assert pr.cid == "test_cred"

    def test_pathref_from_path(self):
        """Test creating PathRef from UPath."""
        upath = Path("/tmp/test")
        pr = PathRef(upath)
        assert pr.href == "/tmp/test"
        assert pr.cid is None

    def test_pathref_from_upath(self):
        """Test creating PathRef from UPath."""
        upath = UPath("/tmp/test")
        pr = PathRef(upath)
        assert pr.href == "/tmp/test"
        assert pr.cid is None

    def test_pathref_from_pathref(self):
        """Test creating PathRef from another PathRef."""
        pr1 = PathRef("s3://bucket/path", cid="cred1")
        pr2 = PathRef(pr1)
        assert pr2.href == pr1.href
        assert pr2.cid == pr1.cid

    def test_pathref_from_dict(self):
        """Test creating PathRef from a dict via model_validate (config path)."""
        pr = PathRef.model_validate({"href": "/tmp/test", "x-cid": "test"})
        assert pr.href == "/tmp/test"
        assert pr.cid == "test"

    def test_pathref_from_dict_without_cid(self):
        """A dict with only 'href' must default cid to None (no KeyError)."""
        pr = PathRef.model_validate({"href": "/tmp/test"})
        assert pr.href == "/tmp/test"
        assert pr.cid is None


class TestPathRefUPath:
    """Test PathRef.upath property with and without credentials."""

    def test_upath_without_cid(self):
        """Test upath generation for paths without credentials."""
        pr = PathRef("/tmp/test")
        upath = pr.upath
        assert isinstance(upath, UPath)
        assert str(upath) == "/tmp/test"

    def test_upath_with_cid(self, mock_s3_credential):
        """Test upath generation with credential ID using mock provider."""
        pr = PathRef("s3://bucket/data.zarr", cid="test_s3")
        upath = pr.upath

        assert isinstance(upath, UPath)
        assert str(upath) == "s3://bucket/data.zarr"
        # Verify storage options contain credential info
        assert "key" in upath.storage_options
        assert upath.storage_options["key"] == "test_key"
        assert upath.storage_options["secret"] == "test_secret"

    def test_upath_with_invalid_cid(self):
        """Test that invalid credential ID raises error."""
        pr = PathRef("s3://bucket/path", cid="nonexistent")
        with pytest.raises(CredentialNotFoundError):
            _ = pr.upath

    def test_upath_caching(self):
        """Test that upath is cached after first access."""
        pr = PathRef("/tmp/test")
        upath1 = pr.upath
        upath2 = pr.upath
        assert upath1 is upath2


class TestPathRefTruediv:
    """Test PathRef.__truediv__ (/) operator with various types."""

    def test_truediv_with_string(self):
        """Test joining PathRef with string."""
        pr = PathRef("/tmp/base")
        result = pr / "subdir"
        assert isinstance(result, PathRef)
        assert result.href == "/tmp/base/subdir"

    def test_truediv_with_pathref_same_cid(self, mock_s3_credential):
        """Test joining two PathRefs with same credential ID."""
        pr1 = PathRef("s3://bucket", cid="test_s3")
        pr2 = PathRef("data", cid="test_s3")
        result = pr1 / pr2
        assert isinstance(result, PathRef)
        assert result.cid == "test_s3"

    def test_truediv_with_pathref_different_cid(self):
        """Test joining PathRefs with different credential IDs raises error."""
        pr1 = PathRef("s3://bucket", cid="cred1")
        pr2 = PathRef("data", cid="cred2")
        with pytest.raises(ValueError, match="different credential ids"):
            _ = pr1 / pr2

    def test_truediv_with_path_object(self):
        """Test joining PathRef with Path object."""
        pr = PathRef("/tmp/base")
        result = pr / Path("subdir")
        assert isinstance(result, PathRef)

    def test_truediv_preserves_cid(self, mock_s3_credential):
        """Test that truediv preserves credential ID."""
        pr = PathRef("s3://bucket", cid="test_s3")
        result = pr / "data" / "file.zarr"
        assert result.cid == "test_s3"


class TestPathRefMethods:
    """Test other PathRef methods."""

    def test_to_dict(self):
        """Test PathRef.to_dict() method."""
        pr = PathRef("s3://bucket/path", cid="test")
        result = pr.to_dict()
        assert isinstance(result, dict)
        assert result["href"] == "s3://bucket/path"
        assert result["x-cid"] == "test"

    def test_serializes_as_ogc_link(self):
        """PathRef must serialize to an OGC Link shape: {href, x-cid}."""
        pr = PathRef("s3://bucket/data.zarr", cid="edh")
        assert pr.model_dump() == {"href": "s3://bucket/data.zarr", "x-cid": "edh"}
        # round-trips from the Link-shaped dict
        assert PathRef.model_validate(pr.model_dump()).cid == "edh"

    def test_unset_link_fields_are_omitted(self):
        """A bare PathRef serializes to just {href} — no null optional fields."""
        assert PathRef("/p").model_dump() == {"href": "/p"}

    def test_full_link_fields_round_trip(self):
        """All OGC Link fields are accepted, serialized, and round-trip."""
        pr = PathRef(
            "s3://bucket/data.zarr",
            cid="edh",
            type="application/x-zarr",
            title="DEM",
            rel="data",
            hreflang="en",
            options={"anon": True},
        )
        dumped = pr.model_dump()
        assert dumped == {
            "href": "s3://bucket/data.zarr",
            "rel": "data",
            "type": "application/x-zarr",
            "hreflang": "en",
            "title": "DEM",
            "x-options": {"anon": True},
            "x-cid": "edh",
        }
        rt = PathRef.model_validate(dumped)
        assert rt.type == "application/x-zarr"
        assert rt.title == "DEM"
        assert rt.rel == "data"
        assert rt.hreflang == "en"
        assert rt.options == {"anon": True}
        assert rt.cid == "edh"

    def test_str_method(self):
        """Test PathRef.__str__() returns value."""
        pr = PathRef("/tmp/test", cid="test")
        assert str(pr) == "/tmp/test"

    def test_pathref_is_frozen(self):
        """Test that PathRef is immutable (frozen)."""
        pr = PathRef("/tmp/test")
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            pr.href = "/new/path"


class TestToUPath:
    """Test to_upath conversion function."""

    def test_to_upath_from_string(self):
        """Test converting string to UPath."""
        result = to_upath("/tmp/test")
        assert isinstance(result, UPath)
        assert str(result) == "/tmp/test"

    def test_to_upath_from_pathref(self):
        """Test converting PathRef to UPath."""
        pr = PathRef("/tmp/test")
        result = to_upath(pr)
        assert isinstance(result, UPath)
        assert str(result) == "/tmp/test"

    def test_to_upath_from_upath(self):
        """Test converting UPath to UPath (passthrough)."""
        upath = UPath("/tmp/test")
        result = to_upath(upath)
        assert isinstance(result, UPath)

    def test_to_upath_from_path(self):
        """Test converting pathlib.Path to UPath."""
        path = Path("/tmp/test")
        result = to_upath(path)
        assert isinstance(result, UPath)


class TestPathUtilities:
    """Test path utility functions."""

    def test_is_remote_path(self):
        """Test remote path detection for explicit remote protocols."""
        # These should correctly identify as remote
        assert is_remote_path("s3://bucket/path")
        assert is_remote_path("https://example.com/data.zarr")
        assert is_remote_path("gs://bucket/path")

        # Note: file:// protocol is explicitly recognized
        assert not is_remote_path("file:///tmp/test")

    def test_is_absolute_path(self):
        """Test absolute path detection."""
        # Absolute local paths
        assert is_absolute_path("/absolute/path")

        # Remote paths are considered absolute
        assert is_absolute_path("s3://bucket/path")
        assert is_absolute_path("https://example.com/data")

    def test_exists(self, temp_dir):
        """Test exists function with local paths."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        assert exists(test_file)
        assert exists(temp_dir)
        assert not exists(temp_dir / "nonexistent.txt")

    def test_mkdir(self, tmp_path):
        """Test mkdir function."""
        new_dir = tmp_path / "new_dir" / "nested"
        mkdir(new_dir)
        assert new_dir.exists()

        # Test idempotency
        mkdir(new_dir)  # Should not raise
        assert new_dir.exists()

    def test_optional_str(self):
        """Test optional_str converter."""
        assert optional_str("/tmp/test") == "/tmp/test"
        assert optional_str(Path("/tmp/test")) == "/tmp/test"
        assert optional_str(None) is None

    def test_normalize_path(self):
        """Test normalize_path function."""
        result = normalize_path("/tmp/../tmp/test")
        assert isinstance(result, str)
        # Should normalize the path
        assert "test" in result

        result = normalize_path("s3://bucket/path")
        assert result == "s3://bucket/path"


class TestPathRefWithEnvCredentials:
    """Test PathRef with environment-based credentials."""

    def test_pathref_with_env_s3_credentials(self, monkeypatch):
        """Test PathRef resolving S3 credentials from environment."""
        # Set up environment variables for S3 credential
        monkeypatch.setenv("S2GOS_CREDENTIALS__envtest__TYPE", "s3")
        monkeypatch.setenv("S2GOS_CREDENTIALS__envtest__KEY", "env_key")
        monkeypatch.setenv("S2GOS_CREDENTIALS__envtest__SECRET", "env_secret")
        monkeypatch.setenv(
            "S2GOS_CREDENTIALS__envtest__ENDPOINT_URL", "https://s3.env.com"
        )

        # Import after setting env vars to get fresh provider
        from s2gos_utils.setting.credentials.provider import (
            EnvCredentialProvider,
            set_credential_provider,
        )

        provider = EnvCredentialProvider()
        set_credential_provider(provider)

        try:
            pr = PathRef("s3://bucket/data.zarr", cid="envtest")
            upath = pr.upath

            assert isinstance(upath, UPath)
            assert "key" in upath.storage_options
            assert upath.storage_options["key"] == "env_key"
            assert upath.storage_options["secret"] == "env_secret"
        finally:
            set_credential_provider(None)
