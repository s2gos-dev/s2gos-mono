# S2GOS-APPS

**s2gos-apps** is the integration layer of the S2GOS framework. It connects scene generation (`s2gos-generator`) and observation simulation (`s2gos-simulator`) into ready-to-run **processes** that can be executed through a GUI, a CLI, or directly from Python.

Each process is a Python function decorated with `@registry.process()`. This single definition is enough to expose it as an OGC API endpoint, a GUI form in Jupyter, and a CLI command — powered by the [s2gos-controller](https://s2gos-dev.github.io/s2gos-controller/) stack (procodile, wraptile, s2gos-client).

## Key components

| Module | Role |
|---|---|
| `registry.py` | `ProcessRegistry` singleton — all processes register here on import |
| `service.py` | `LocalService` — wraps the registry as a FastAPI / OGC API server |
| `cli.py` | Generates a CLI (`s2gos_apps`) from the registry using `procodile` |
| `processes/` | Process definitions grouped by site (Gobabeb, Frascati, …) and common operations |
| `gen_util.py` | Shared generation logic — runs the [`SceneGenerationPipeline`](../s2gos-generator/api/pipeline.md) DAG |
| `sim_util.py` | Shared simulation logic — builds [`SimulationConfig`](../s2gos-simulator/api/simulation.md), runs [`EradiateBackend`](../s2gos-simulator/api/backend.md) |

## Available processes

### Site-specific configuration processes

Each site provides a pair of processes that expand a few high-level parameters (location, size, observation time) into full configuration files.
This is still very early but will be developed into the use cases for teh S2GOS Project

| Process ID | Title | Description |
|---|---|---|
| `gobabeb-generation-config` | Gobabeb Generation Config | Scene generation config for the Gobabeb PICS site (Namibia) |
| `gobabeb-simulation-config` | Gobabeb Simulation Config | Simulation config for Gobabeb with configurable GMT hour and samples |
| `frascati-generation-config` | Frascati Generation Config | Scene generation config for Frascati (Italy) with vegetation placement |
| `frascati-simulation-config` | Frascati Simulation Config | Simulation config for the Frascati scene |
| `kairouan-generation-config` | Kairouan Generation Config | Scene generation config for Kairouan (Tunisia) with vegetation |
| `kairouan-simulation-config` | Kairouan Simulation Config | Simulation config for the Kairouan scene |
| `pisa-generation-config` | Pisa Generation Config | Scene generation config for San Rossore / Pisa (Italy) with vegetation |
| `pisa-simulation-config` | Pisa Simulation Config | Simulation config for the Pisa scene |
| `pnp-generation-config` | PNP Generation Config | Scene generation config for Patagonia National Park (Chile) with heterogeneous atmosphere |
| `pnp-simulation-config` | PNP Simulation Config | Simulation config for the PNP scene |

### Common processes

These processes accept a configuration file path and run the full pipeline.

| Process ID | Title | Description |
|---|---|---|
| `common-generation` | Scene Generation | Runs the generation pipeline from a config JSON — produces meshes, textures, and a scene description YAML |
| `common-simulation` | Scene Simulation | Runs the Eradiate radiative transfer simulation from a scene description and simulation config |

### Demo processes

| Process ID | Title | Description |
|---|---|---|
| `pnp_gui/generation` | Scene Generation Demo | PNP GUI demo with seasonal variations — runs generation pipeline directly |
| `pnp_gui/simulation` | Simulation Demo | PNP GUI demo with multiple observation types — runs simulation pipeline directly |
| `upscaling-demo` | Upscaling Demo | Experimental upscaling workflow (not yet implemented) |

## Installation

You can install the default and development version using:

```bash
# default
pixi install
# dev
pixi install -e dev
```
You can then install the data needed by Eradiate by running.

```bash
pixi run eradiate-init
```

See the [Eradiate documentation](https://eradiate.readthedocs.io/en/stable/user_guide/config.html) on a guide on how to Configure the data install location.


## Configuration

* The S2GOS package suite can be configured by having a settings file in your current working directory or any parents to it.
* The file should be named "s2gos_settings.yaml"
* Here is a template example:

```yaml
# s2gos_settings.yaml
common:
    search_paths:
        - "./packages/s2gos-generator/resources/data"   # This is a real path you should include, specifies basic materials
        - "./EXAMPLE_EXTRA_DATA_PATH"

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
        material_config: "./materials.json"
```

* See the [common](../s2gos-utils/configuration.md) and [generator](../s2gos-generator/configuration.md) configuration pages for additional information.
* Secrets can be provided either through environment variables or a `.secrets.yaml` file. See [the credentials page](../s2gos-utils/credentials.md) for details on how to set this up.
