# Configuration

The S2GOS generator reads its settings from `s2gos_settings.yaml`, which is located by
walking up the directory tree from your script. Shared settings (`common:` section) are
documented in the [s2gos-utils Configuration](../s2gos-utils/configuration.md) page.

## Configuration Example

```yaml
# s2gos_settings.yaml — generator-specific section only
# For the common: section see s2gos-utils configuration.
generator:
    dataset:
        dem:
            name: "Copernicus-DEM-30"
            crs: "EPSG:4326"
            type: indexed-geotiff
            root_directory: "/path/to/dem/tiles"
            index_path: "/path/to/dem_index.feather"
            path_column: "path_dem"
            variable_name: elevation

        landcover:
            name: "ESA Worldcover 2021"
            crs: "EPSG:4326"
            type: zarr
            path:
                value: "s3://bucket/worldcover.zarr"
                cid: "my_s3_creds"
            variable_name: landcover

    files:
        material_config: "./materials.json"
```

## Configuration Sections

### `generator.dataset` - Dataset Sources

#### `dem`, **Dataset**, *required*

Digital Elevation Model dataset. Tested with Copernicus DEM 30m.

#### `landcover`, **Dataset**, *required*

Land cover classification dataset. Tested with ESA WorldCover 2021.

### Dataset Types

Datasets are specified as subobjects with a `type` field. Paths use **[PathRef](../s2gos-utils/api/paths.md)** format:
- `value`: URI string (local or remote)
- `cid`: Credential ID (optional) - see [s2gos-utils credentials documentation](../s2gos-utils/credentials.md)

#### Common Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | No | Dataset name (defaults to key name) |
| `crs` | string | No | CRS (default: `EPSG:4326`) |
| `type` | string | Yes | Dataset type |

#### [IndexedGeoTiff](api/datasets.md) (`type: indexed-geotiff`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `root_directory` | [PathRef](../s2gos-utils/api/paths.md) | Yes | Directory containing GeoTIFF tiles |
| `index_path` | [PathRef](../s2gos-utils/api/paths.md) | Yes | Feather index file with tile paths |
| `variable_name` | string | No | Data variable name |
| `path_column` | string | No | Column with file paths (auto-detected) |

#### [Zarr](api/datasets.md) (`type: zarr`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | [PathRef](../s2gos-utils/api/paths.md) | Yes | Path to Zarr archive |
| `variable_name` | string | No | Data variable name |

### `generator.files` - Additional Files

#### `material_config`, **[PathRef](../s2gos-utils/api/paths.md)**, *optional*

Path to materials JSON defining optical properties for land cover classes. Default: `materials.json` (resolved via `search_paths`).