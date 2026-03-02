# Quickstart: Generate a Gobabeb Scene

This example walks through generating a 3D scene of the
Gobabeb site in the Namib Desert, Namibia. You will create a scene configuration, run the
generation pipeline, and produce the meshes, textures, and scene description YAML
that feed into the simulator.

By the end you will have a `gobabeb.yml` scene description file ready to use for simulations.

---

## Prerequisites

- **Pixi** installed and the `s2gos-generator` environment activated.
- A **`s2gos_settings.yaml`** file in your working directory with DEM and
  landcover data paths configured. See the
  [Configuration](../configuration.md) page for details.

---

## Step 1 — Create a scene configuration

Use `create_scene_config()` to define the area of interest around the Gobabeb
site. Then attach a buffer zone, a background zone, and a molecular atmosphere.

```python
from s2gos_generator import create_scene_config
from s2gos_generator.core.config import (
    BufferConfig,
    BackgroundConfig,
    MolecularAtmosphereConfig,
    ThermophysicalConfig,
    AbsorptionDatabase,
)

# Target area: 10 km around Gobabeb at 10 m resolution
config = create_scene_config(
    scene_name="gobabeb",
    center_lat=-23.6015,
    center_lon=15.1259,
    aoi_size_km=10,
    output_dir="./gen_output",
    target_resolution_m=10.0,
    description="Gobabeb PICS site in the Namib Desert.",
)

# Buffer zone — surrounds the target for realistic boundary conditions
config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)

# Background zone — coarse terrain extending to the horizon
config.background = BackgroundConfig(
    size_km=150.0, resolution_m=200.0, elevation=0.0
)

# Molecular atmosphere (US Standard, GECKO absorption database)
config.set_atmosphere_molecular(
    MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
        has_scattering=True,
    )
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `aoi_size_km` | 10 | 10 km &times; 10 km target area |
| `target_resolution_m` | 10.0 | One mesh cell per 10 m |
| `buffer.size_km` | 60.0 | 60 km buffer at 60 m resolution |
| `background.size_km` | 150.0 | 150 km background at 200 m resolution |

---

## Step 2 — Save the configuration

Persist the configuration to JSON so you can reproduce or share the run later.

```python
from pathlib import Path

config_dir = Path("./gen_config")
config_dir.mkdir(exist_ok=True)

config.to_json(config_dir / "gobabeb_gen_config.json")
```

---

## Step 3 — Run the generation pipeline

Create a `SceneGenerationPipeline` from the configuration and execute it.
The pipeline fetches DEM and landcover data, builds terrain meshes and textures,
and writes the final scene description.

```python
from s2gos_generator import SceneGenerationPipeline

pipeline = SceneGenerationPipeline(config)
pipeline.run()
```

Depending on the size of the area and your data source, this step may take
several minutes.

---

## Step 4 — Inspect the output

After the pipeline completes, the output directory contains:

```
gen_output/gobabeb/
├── gobabeb.yml            # Scene description (input for the simulator)
├── meshes/                # PLY terrain meshes (target, buffer, background)
├── textures/              # PNG landcover textures
└── data/                  # Intermediate data like crops of the DEM
```

The key output is **`gobabeb.yml`** — a YAML file describing every mesh,
material, and texture in the scene. This is the file you pass to the simulator.

---

## Next step

To simulate an RGB observation over this scene, continue with
[Quickstart: Simulate an RGB Observation](../../s2gos-simulator/examples/gobabeb_quickstart.md).
