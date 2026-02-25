import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .credential import BasicAuthCredential, CredentialType, S3Credential
from .exceptions import CredentialNotFoundError


class CredentialProvider(ABC):
    """Abstract base class for credential providers"""

    @abstractmethod
    def get_credential(self, credential_id: str) -> CredentialType | None:
        """
        Retrieve credential by ID.

        Args:
            credential_id: The identifier for the credential

        Returns:
            The credential object or None if not found
        """
        pass

    def list_credentials(self) -> list[str]:
        """
        List available credential IDs.

        Returns:
            List of credential IDs (optional, may return empty list)
        """
        return []


class EnvCredentialProvider(CredentialProvider):
    """
    Load credentials from environment variables.

    Environment variable naming convention:
        S2GOS_CREDENTIALS__<credential_id>__TYPE=basic_auth|s3
        S2GOS_CREDENTIALS__<credential_id>__USERNAME=...
        S2GOS_CREDENTIALS__<credential_id>__PASSWORD=...
        S2GOS_CREDENTIALS__<credential_id>__KEY=...
        S2GOS_CREDENTIALS__<credential_id>__SECRET=...
        S2GOS_CREDENTIALS__<credential_id>__ENDPOINT_URL=...

    Example:
        export S2GOS_CREDENTIALS__earthdatahub__TYPE=basic_auth
        export S2GOS_CREDENTIALS__earthdatahub__USERNAME=edh
        export S2GOS_CREDENTIALS__earthdatahub__PASSWORD=edh_pat_xyz...
    """

    def get_credential(self, credential_id: str) -> CredentialType | None:
        prefix = f"S2GOS_CREDENTIALS__{credential_id}__"
        cred_type = os.getenv(f"{prefix}TYPE")

        if not cred_type:
            raise CredentialNotFoundError(credential_id)

        if cred_type == "basic_auth":
            username = os.getenv(f"{prefix}USERNAME")
            password = os.getenv(f"{prefix}PASSWORD")
            if not username or not password:
                raise ValueError(
                    f"Credential '{credential_id}': missing USERNAME or PASSWORD environment variable"
                )
            return BasicAuthCredential(
                id=credential_id,
                username=username,
                password=password,
                protocol=os.getenv(f"{prefix}PROTOCOL", "https"),
            )
        elif cred_type == "s3":
            key = os.getenv(f"{prefix}KEY")
            secret = os.getenv(f"{prefix}SECRET")
            if not key or not secret:
                raise ValueError(
                    f"Credential '{credential_id}': missing KEY or SECRET environment variable"
                )
            return S3Credential(
                id=credential_id,
                key=key,
                secret=secret,
                endpoint_url=os.getenv(f"{prefix}ENDPOINT_URL"),
            )
        else:
            raise ValueError(f"Unknown credential type: {cred_type}")


class DynaconfCredentialProvider(CredentialProvider):
    """
    Load credentials from Dynaconf settings.

    Dynaconf automatically loads .secrets.yaml, so credentials defined there
    will be available through this provider.

    Expected .secrets.yaml format:
        credentials:
          earthdatahub:
            type: basic_auth
            username: edh
            password: edh_pat_xyz...

          s3ovh:
            type: s3
            key: ba4efc6b...
            secret: 11a04ac9...
            endpoint_url: https://s3.de.io.cloud.ovh.net
    """

    def __init__(self, settings):
        self.settings = settings

    def get_credential(self, credential_id: str) -> CredentialType | None:
        # Check if credentials section exists in settings
        if not hasattr(self.settings, "credentials"):
            raise CredentialNotFoundError(credential_id)

        creds_dict = self.settings.credentials.get(credential_id)
        if not creds_dict:
            raise CredentialNotFoundError(credential_id)

        # Convert to regular dict if it's a DynaBox
        if hasattr(creds_dict, "to_dict"):
            creds_dict = creds_dict.to_dict()

        cred_type = creds_dict.get("type")

        if cred_type == "basic_auth":
            return BasicAuthCredential(
                id=credential_id,
                username=creds_dict["username"],
                password=creds_dict["password"],
                protocol=creds_dict.get("protocol", "https"),
            )
        elif cred_type == "s3":
            return S3Credential(
                id=credential_id,
                key=creds_dict["key"],
                secret=creds_dict["secret"],
                endpoint_url=creds_dict.get("endpoint_url"),
            )
        else:
            raise ValueError(f"Unknown credential type: {cred_type}")

    def list_credentials(self) -> list[str]:
        """List all credential IDs defined in settings"""
        if not hasattr(self.settings, "credentials"):
            return []
        return list(self.settings.credentials.keys())


class DictCredentialProvider(CredentialProvider):
    """
    In-memory credential provider (for testing and programmatic use).

    Example:
        creds = {
            "test": BasicAuthCredential(
                id="test", username="user", password="pass"
            )
        }
        provider = DictCredentialProvider(creds)
    """

    def __init__(self, credentials: Dict[str, CredentialType]):
        self.credentials = credentials

    def get_credential(self, credential_id: str) -> Optional[CredentialType]:
        creds = self.credentials.get(credential_id)
        if creds is None:
            raise CredentialNotFoundError(credential_id)
        return self.credentials.get(credential_id)

    def list_credentials(self) -> list[str]:
        return list(self.credentials.keys())


# Global credential provider instance
_credential_provider: Optional[CredentialProvider] = None


def set_credential_provider(provider: CredentialProvider) -> None:
    """
    Set the global credential provider.

    Args:
        provider: The credential provider to use globally
    """
    global _credential_provider
    _credential_provider = provider


def get_credential_provider() -> CredentialProvider:
    """
    Get the global credential provider (lazy initialization).

    If no provider has been set, creates a default DynaconfCredentialProvider
    that checks environment variables first, then Dynaconf settings.

    Returns:
        The global credential provider
    """
    from .. import settings

    global _credential_provider

    if _credential_provider is None:
        # Fetch provider from settings.
        provider_name = settings.common.credential_provider

        if provider_name == "environment":
            _credential_provider = EnvCredentialProvider()
        elif provider_name == "dynaconf":
            from s2gos_utils.setting._settings import settings

            _credential_provider = DynaconfCredentialProvider(settings)
        else:
            raise NotImplementedError(
                f"{provider_name} is not implemented as a Credential Provider."
            )

    return _credential_provider


def get_credential(credential_id: str) -> Optional[CredentialType]:
    """
    Convenience function to get a credential by ID.

    Args:
        credential_id: The identifier for the credential

    Returns:
        The credential object or None if not found
    """
    return get_credential_provider().get_credential(credential_id)
