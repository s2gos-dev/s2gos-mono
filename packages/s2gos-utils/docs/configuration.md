# Configuration

The S2GOS generator is configured through a `s2gos_settings.yaml` file. The package automatically searches for this file by climbing up the directory tree from your script's location.

## Installation Modes

The configuration structure is modulable and will depends:

- `common` section: required when using both `s2gos-generator` and `s2gos-simulator`.
- `generator` and `simulator` sections required when using `s2gos-generator` and `s2gos-simulator` respectively.
- **Monorepo**: Includes `common`, `generator`, and `simulator` sections.

## Configuration Example

```yaml
# s2gos_settings.yaml
common:
    search_paths:
        - "./resources/data"
        - "./data"
        - "s3://my-bucket/shared-data"

    local_fsspec_cache: "./tmp/fsspec_cache"  # Optional
    credential_provider: "dynaconf"           # Optional: dynaconf | environment

# generator:
#   See s2gos-generator documentation
# simulator:
#   See s2gos-simulator documentation
```

## Configuration Parameters

### `common` - Shared Settings

Settings used by both generator and simulator packages.

#### `search_paths`, **List of paths**, *optional (default: empty list)*

Prioritized list of directories for resolving relative file paths. The file resolver searches these paths in order and returns the first match. Useful for locating resource files (materials.json, ephemeris data) and index files across different environments.

Paths can be absolute or relative. Local paths are automatically resolved, and remote paths (s3://, etc.) are supported. Environment override available via `S2GOS_SEARCH_PATHS`.

#### `local_fsspec_cache`, **string**, *optional (default: "./tmp/fsspec_cache")*

Cache directory used by xarray when loading a netcdf stored in a remote location. This caching behaviour happens only when the xarray backend engine is set to `netcdf4` and the netcdf is stored remotely.

#### `credential_provider`, **string**, *optional (default: dynaconf)*

Name of the credential provider used to resolve credentials from a credential ID. See [Credentials](credentials.md) for details.