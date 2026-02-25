class CredentialNotFoundError(Exception):
    """
    Raised when a credential ID cannot be found in any provider.
    Indicates the credential is not configured at all.
    """

    def __init__(self, credential_id: str):
        self.credential_id = credential_id
        message = (
            f"Credential '{credential_id}' not found in provider. "
            f"To configure:\n"
            f"  - Environment: S2GOS_CRED_{credential_id}_TYPE, "
            f"S2GOS_CRED_{credential_id}_USERNAME, etc.\n"
            f"  - Or add to .secrets.yaml under 'credentials.{credential_id}'"
        )
        super().__init__(message)
