"""
Credential management system for s2gos.

This module provides a secure way to manage credentials for remote data access.
Credentials are stored separately from configuration files and can be provided via:
1. Environment variables (S2GOS_CRED_<id>_*)
2. Dynaconf .secrets.yaml file
"""

from typing import Literal

from pydantic import BaseModel, field_validator


class Credential(BaseModel):
    """Base credential model"""

    id: str
    type: str

    @property
    def upath_kwargs(self):
        """Convert a credential object to UPath constructor kwargs."""
        pass


class BasicAuthCredential(Credential):
    """HTTP Basic Authentication credentials"""

    type: Literal["basic_auth"] = "basic_auth"
    username: str
    password: str
    protocol: str = "https"

    @property
    def upath_kwargs(self):
        import aiohttp

        auth = aiohttp.BasicAuth(self.username, self.password)
        kwargs = {"client_kwargs": {"auth": auth}}
        return kwargs


class S3Credential(Credential):
    """AWS S3 / S3-compatible storage credentials"""

    type: Literal["s3"] = "s3"
    key: str
    secret: str
    endpoint_url: str | None = None

    @field_validator("endpoint_url")
    @classmethod
    def _default_https(cls, value: str | None) -> str | None:
        """Bare hosts get ``https://``; boto3/s3fs reject an endpoint without a scheme."""
        if value and "://" not in value:
            return f"https://{value}"
        return value

    @property
    def upath_kwargs(self):
        kwargs = {"key": self.key, "secret": self.secret}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return kwargs


# Union of all credential types
CredentialType = BasicAuthCredential | S3Credential
