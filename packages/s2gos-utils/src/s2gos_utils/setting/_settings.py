from dynaconf import Dynaconf, Validator

from .paths import to_pathref
from ..io.resolver import resolver

# SETTING DEFAULTS


def _search_paths(settings=None, validator=None) -> list:
    return []


def _local_fsspec_cache(settings=None, validator=None) -> str:
    return "./tmp/fsspec_cache"


def _credential_provider(settings=None, validator=None) -> str:
    return "dynaconf"


settings = Dynaconf(
    settings_files=["s2gos_settings.yaml", "s2gos_settings.toml"],
    secrets=".secrets.yaml",
    envvar_prefix="S2GOS",
    validators=[
        Validator("SEARCH_PATHS", cast=list, default=_search_paths),
        Validator("LOCAL_FSSPEC_CACHE", cast=str, default=_local_fsspec_cache),
        Validator("credential_provider", cast=str, default=_credential_provider),
    ],
    validate_only="common",
)


def load_config():
    """
    Initialize the resolver with the search paths.
    """
    user_paths = settings.common.search_paths
    for path in user_paths:
        upath = to_pathref(path).upath
        if upath.exists():
            resolver.append(path)
