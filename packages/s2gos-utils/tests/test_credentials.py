"""
Tests for the credential management system.
"""

import pytest

from s2gos_utils.io import PathRef
from s2gos_utils.setting.credentials.credential import (
    BasicAuthCredential,
    S3Credential,
)
from s2gos_utils.setting.credentials.exceptions import CredentialNotFoundError
from s2gos_utils.setting.credentials.provider import (
    DictCredentialProvider,
    EnvCredentialProvider,
    get_credential,
    set_credential_provider,
)


class TestCredentialModels:
    """Test credential model creation and validation"""

    def test_basic_auth_credential(self):
        """Test BasicAuthCredential creation"""
        cred = BasicAuthCredential(
            id="test", username="user", password="pass", protocol="https"
        )
        assert cred.id == "test"
        assert cred.username == "user"
        assert cred.password == "pass"
        assert cred.protocol == "https"
        assert cred.type == "basic_auth"

    def test_s3_credential(self):
        """Test S3Credential creation"""
        cred = S3Credential(
            id="test",
            key="mykey",
            secret="mysecret",
            endpoint_url="https://s3.example.com",
        )
        assert cred.id == "test"
        assert cred.key == "mykey"
        assert cred.secret == "mysecret"
        assert cred.endpoint_url == "https://s3.example.com"
        assert cred.type == "s3"


class TestEnvCredentialProvider:
    """Test environment variable credential provider"""

    def test_basic_auth_from_env(self, monkeypatch):
        """Test loading basic auth credentials from environment"""
        monkeypatch.setenv("S2GOS_CREDENTIALS__test__TYPE", "basic_auth")
        monkeypatch.setenv("S2GOS_CREDENTIALS__test__USERNAME", "testuser")
        monkeypatch.setenv("S2GOS_CREDENTIALS__test__PASSWORD", "testpass")

        provider = EnvCredentialProvider()
        cred = provider.get_credential("test")

        assert cred is not None
        assert isinstance(cred, BasicAuthCredential)
        assert cred.username == "testuser"
        assert cred.password == "testpass"

    def test_s3_from_env(self, monkeypatch):
        """Test loading S3 credentials from environment"""
        monkeypatch.setenv("S2GOS_CREDENTIALS__s3test__TYPE", "s3")
        monkeypatch.setenv("S2GOS_CREDENTIALS__s3test__KEY", "mykey")
        monkeypatch.setenv("S2GOS_CREDENTIALS__s3test__SECRET", "mysecret")
        monkeypatch.setenv(
            "S2GOS_CREDENTIALS__s3test__ENDPOINT_URL", "https://s3.example.com"
        )

        provider = EnvCredentialProvider()
        cred = provider.get_credential("s3test")

        assert cred is not None
        assert isinstance(cred, S3Credential)
        assert cred.key == "mykey"
        assert cred.secret == "mysecret"
        assert cred.endpoint_url == "https://s3.example.com"

    def test_missing_credential(self):
        """Test that missing credentials return None"""
        provider = EnvCredentialProvider()
        with pytest.raises(CredentialNotFoundError):
            _ = provider.get_credential("nonexistent")

    def test_missing_required_fields(self, monkeypatch):
        """Test that missing required fields raise ValueError"""
        monkeypatch.setenv("S2GOS_CREDENTIALS_incomplete_TYPE", "basic_auth")
        monkeypatch.setenv("S2GOS_CREDENTIALS_incomplete_USERNAME", "user")
        # Missing PASSWORD

        provider = EnvCredentialProvider()
        with pytest.raises(CredentialNotFoundError):
            provider.get_credential("incomplete")


class TestDictCredentialProvider:
    """Test in-memory dictionary credential provider"""

    def test_get_credential(self):
        """Test retrieving credential from dict provider"""
        creds = {
            "test": BasicAuthCredential(id="test", username="user", password="pass")
        }
        provider = DictCredentialProvider(creds)

        cred = provider.get_credential("test")
        assert cred is not None
        assert cred.username == "user"

    def test_list_credentials(self):
        """Test listing available credential IDs"""
        creds = {
            "cred1": BasicAuthCredential(id="cred1", username="u1", password="p1"),
            "cred2": S3Credential(id="cred2", key="k", secret="s"),
        }
        provider = DictCredentialProvider(creds)

        cred_ids = provider.list_credentials()
        assert set(cred_ids) == {"cred1", "cred2"}


class TestCredentialToUPathKwargs:
    """Test conversion of credentials to UPath kwargs"""

    def test_basic_auth_to_kwargs(self):
        """Test converting BasicAuthCredential to UPath kwargs"""
        cred = BasicAuthCredential(id="test", username="user", password="pass")
        kwargs = cred.upath_kwargs

        assert "client_kwargs" in kwargs
        assert "auth" in kwargs["client_kwargs"]

    def test_s3_to_kwargs(self):
        """Test converting S3Credential to UPath kwargs"""
        cred = S3Credential(
            id="test",
            key="mykey",
            secret="mysecret",
            endpoint_url="https://s3.example.com",
        )
        kwargs = cred.upath_kwargs

        assert kwargs["key"] == "mykey"
        assert kwargs["secret"] == "mysecret"
        assert kwargs["endpoint_url"] == "https://s3.example.com"

    def test_s3_without_endpoint(self):
        """Test S3 credential without endpoint URL"""
        cred = S3Credential(id="test", key="mykey", secret="mysecret")
        kwargs = cred.upath_kwargs

        assert kwargs["key"] == "mykey"
        assert kwargs["secret"] == "mysecret"
        assert "endpoint_url" not in kwargs or kwargs["endpoint_url"] is None


class TestPathRef:
    """Test PathRef model"""

    def test_create_with_credential_id(self):
        """Test creating PathRef with credential_id"""
        path = PathRef(href="https://example.com/data.zarr", cid="test")
        assert path.href == "https://example.com/data.zarr"
        assert path.cid == "test"

    def test_create_without_credential(self):
        """Test creating PathRef without credentials"""
        path = PathRef(href="/local/path/data.zarr")
        assert path.href == "/local/path/data.zarr"
        assert path.cid is None

    def test_str_representation(self):
        """Test string representation returns the path value"""
        path = PathRef(href="https://example.com/data.zarr", cid="test")
        assert str(path) == "https://example.com/data.zarr"

    def test_upath_without_credentials(self):
        """Test getting UPath without credentials"""
        path = PathRef(href="/tmp/test.txt")
        upath = path.upath
        assert str(upath) == "/tmp/test.txt"

    def test_upath_with_credentials(self):
        """Test getting UPath with credentials"""
        # Set up credential provider
        cred = BasicAuthCredential(id="test", username="user", password="pass")
        provider = DictCredentialProvider({"test": cred})
        set_credential_provider(provider)

        path = PathRef(href="https://example.com/data.zarr", cid="test")
        upath = path.upath

        # UPath should be created (exact auth details may vary by implementation)
        assert upath is not None

    def test_missing_credential_raises_error(self):
        """Test that missing credential raises ValueError"""
        # Set up empty provider
        provider = DictCredentialProvider({})
        set_credential_provider(provider)

        path = PathRef(href="https://example.com/data.zarr", cid="nonexistent")

        with pytest.raises(CredentialNotFoundError):
            _ = path.upath

    def test_serialization(self):
        """Test that PathRef serializes without exposing credentials"""
        path = PathRef(href="https://example.com/data.zarr", cid="secret_cred")

        data = path.model_dump()
        assert data["href"] == "https://example.com/data.zarr"
        assert data["x-cid"] == "secret_cred"
        # Should not contain actual credentials
        assert "password" not in str(data)
        assert "username" not in str(data)

    def test_deserialization(self):
        """Test deserializing PathRef"""
        data = {
            "href": "https://example.com/data.zarr",
            "x-cid": "test",
        }
        path = PathRef.model_validate(data)

        assert path.href == "https://example.com/data.zarr"
        assert path.cid == "test"


class TestGlobalCredentialProvider:
    """Test global credential provider management"""

    def test_set_and_get_credential(self):
        """Test setting global provider and retrieving credentials"""
        cred = BasicAuthCredential(id="global_test", username="user", password="pass")
        provider = DictCredentialProvider({"global_test": cred})
        set_credential_provider(provider)

        retrieved = get_credential("global_test")
        assert retrieved is not None
        assert retrieved.username == "user"
