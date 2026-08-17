# Configuration

Two layers configure a generation run:

1. **`s2gos_settings.yaml`** — the ambient environment you set up **once**: where the
   input data lives (DEM, landcover, building tiles), the material library, shared
   `search_paths`, and credentials for remote/authenticated data. **This page covers
   that layer.**
2. **The scene config** ([`SceneGenConfig`][s2gos_generator.core.config.scene.SceneGenConfig],
   serialised to the generation JSON) — *what* to generate for a given run: location,
   size, and the optional features (buildings, roads, spectral matching, mesh
   refinement). Its full field reference lives on the
   [Scene Config API page](api/scene_config.md); the [Concepts](concepts.md) page
   explains what each feature does.

`s2gos_settings.yaml` is located by walking up the directory tree from your script.
The shared `common:` section (including `search_paths`) is documented in the
[s2gos-utils Configuration](../s2gos-utils/configuration.md) page.

## Example

```yaml
# s2gos_settings.yaml
common:
    search_paths:
        - "./packages/s2gos-generator/resources/data"   # ships the base materials.json
        - "./data"

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
                href: "s3://bucket/worldcover.zarr"
                cid: "my_s3_creds"
            variable_name: landcover

    files:
        material_config: "materials.json"    # resolved via search_paths
        building_tiles: "openbuildingmap"    # tile dir for buildings (optional)
```

## `generator.dataset` — input datasets

#### `dem`, **Dataset**, *required*

Digital Elevation Model dataset. Tested with the
[Copernicus DEM GLO-30](https://doi.org/10.5069/G9028PQB) (30 m).

#### `landcover`, **Dataset**, *required*

Land cover classification dataset. Tested with
[ESA WorldCover 2021 (v200)](https://doi.org/10.5281/zenodo.7254221).

### Dataset Types

Datasets are specified as subobjects with a `type` field. Paths use **[PathRef](../s2gos-utils/api/paths.md)** format (an OGC API Link):

- `href`: URI string (local or remote)
- `cid`: Credential ID (optional, serialized as `x-cid`) — see [Credentials](#credentials-for-remote-data-s3)

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

## `generator.files` — resource files

Both entries are resolved through `common.search_paths`, so a bare filename or a
relative path is found wherever you keep your resources (local or remote).

#### `material_config`, **[PathRef](../s2gos-utils/api/paths.md)**, *optional*

Path to the materials JSON defining optical properties for land cover classes.
Default: `materials.json`.

#### `building_tiles`, **[PathRef](../s2gos-utils/api/paths.md)**, *optional*

Directory of quadkey-indexed [OpenBuildingMap](https://www.openbuildingmap.org/) GPKG
tiles (`building.<quadkey>.gpkg`) plus the index CSV, used when
[buildings](#buildings) are enabled. Leave unset if you do not use buildings.

## Credentials for remote data (S3)

Any `href` on `s3://` (a remote DEM/landcover dataset, or the Sentinel-2 `eodata`
bucket used by [spectral matching](#spectral-material-matching)) needs credentials.
Secrets are never written into settings or serialized configs — you reference a
**credential id** and keep the secret in `.secrets.yaml` (or environment variables).
Full details are on the [s2gos-utils Credentials](../s2gos-utils/credentials.md) page;
in short:

```yaml
credentials:
  my_s3_creds:
    type: s3
    key: your_access_key
    secret: your_secret_key
    endpoint_url: https://s3.example.com
```

Reference the id from a dataset path (`cid: my_s3_creds`) or from
`SpectralMatchingConfig.credential_id`.

## Optional features

Each optional feature is enabled per run on
[`SceneGenConfig`][s2gos_generator.core.config.scene.SceneGenConfig] and tuned via its
own config (full fields on the [Scene Config API page](api/scene_config.md)). This
section covers only the **data and credentials** each one needs from your environment.

### Buildings

1. Point `generator.files.building_tiles` at your OpenBuildingMap tile directory (see
   [above](#building_tiles-pathref-optional)).
2. Enable buildings per run with a
   [`BuildingsConfig`][s2gos_generator.core.config.buildings.BuildingsConfig] on the
   scene config; use it to control materials and roofs. No other setup is required.

See [Buildings](concepts.md#buildings) for what the feature does.

### Ways

Enable per run with a [`WaysConfig`][s2gos_generator.core.config.ways.WaysConfig] on the
scene config. Way centrelines come from one of two sources:

- `source="overpass"` — fetched live from OpenStreetMap over the public
  [Overpass API](https://overpass-api.de/). No credentials or local data required.
- `source="file"` — read from a local JSON file; point the config at your path.

Width and surface material are resolved from OpenStreetMap tags where present, falling back to
a built-in per-highway-type table and configurable defaults.

!!! note
    Any published output derived from Overpass/OpenStreetMap data must attribute
    © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.

Enabling roads also engages the [adaptive terrain mesh](#adaptive-terrain-mesh) so roads sit on
well-resolved, flattened geometry. See [Ways](concepts.md#ways) for what the feature does.

### Adaptive terrain mesh

Tuned per run with a
[`MeshRefinementConfig`][s2gos_generator.core.config.mesh_refinement.MeshRefinementConfig] on the
scene config. It is pure geometry processing — it needs nothing from your environment (no extra
data or credentials) — and is also engaged automatically when roads are enabled. See
[Adaptive terrain mesh](concepts.md#adaptive-terrain-mesh) for what the feature does.

### Spectral material matching

Needs two things beyond the base setup:

1. **A material library of diffuse spectra.** Matching draws candidates from the
   **`diffuse`** entries of a `materials.json`-style file (non-diffuse materials are
   ignored). The generator ships a ready-to-use one at
   `packages/s2gos-generator/resources/data/materials.json` (on the default
   `search_paths`), so you can point `material_library` straight at `materials.json`, or
   copy it as a template and extend it. Each `reflectance` may be a NetCDF file
   (`{"path": ..., "variable": ...}`), an `interpolated` spectrum (`wavelengths` +
   `values`), or a `uniform` value.

    !!! warning
        Each candidate spectrum **must span all requested S2 band centres** (490 / 560 /
        665 / 842 nm for the default bands) or it is **dropped** from the
        candidate set. After editing the library, confirm your materials still 
        load (the candidate count is logged during a run).

2. **Copernicus S3 credentials.** Sentinel-2 is read from the
   [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) `eodata` bucket.
   Set `SpectralMatchingConfig.credential_id` to an `s3` credential (see
   [Credentials](#credentials-for-remote-data-s3)), or rely on ambient `AWS_*`.

Configure matching per run with
[`SpectralMatchingConfig`][s2gos_generator.core.config.material_match.SpectralMatchingConfig].
See [Spectral material matching](concepts.md#spectral-material-matching) for the method.

## Data sources & attribution

The generator relies on several external datasets and methods (Sentinel-2, Copernicus
DEM, ESA WorldCover, OpenBuildingMap, OpenStreetMap, the material matching approach) See the
[References](concepts.md#references) on the Concepts page for citations, DOIs,
and the attribution requested by each provider.
