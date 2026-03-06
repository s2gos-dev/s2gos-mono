# Concepts

This page explains the key ideas behind scene generation in S2GOS: how scenes are spatially organised, how the generation pipeline works, and how atmosphere is configured.

## Generation Pipeline

Scene generation follows a DAG (directed acyclic graph) pipeline with automatic dependency resolution. Resources are executed in the correct order automatically — you only need to provide a [`SceneGenConfig`][s2gos_generator.core.config.scene.SceneGenConfig] and call `run()`.

The broad stages are:

1. **Data extraction** — Clips source data (DEM, landcover) to the target and optionally buffer/background extents.
2. **Mesh generation** — Converts elevation data into 3D triangle meshes (PLY format), one per zone.
3. **Texture generation** — Maps landcover classes to material definitions, producing selection textures for each mesh.
4. **Scene description output** — Assembles all resources into a [`SceneDescription`][s2gos_utils.scene.description.SceneDescription] YAML that ties meshes, textures, materials, and atmosphere together for use by the simulator.

Additional optional resources — vegetation placement, user assets, HAMSTER albedo data, and XML scene imports — are included in the DAG when enabled and run at the appropriate point in the dependency graph.

!!! note 
    The pipeline is under active development. The set of available resources and their configuration options continues to grow.

The pipeline caches completed resources to disk. On subsequent runs, any resource whose configuration is unchanged and whose output files are still present is skipped automatically. Pass `use_cache=False` to [`run()`][s2gos_generator.core.pipeline.SceneGenerationPipeline.run] to force a full rebuild.

The figure below shows a typical pipeline DAG for a scene with some optional resources enabled:

![Pipeline DAG example](figures/pipeline_dag.png)

## Scene Zones

A generated scene is composed of up to three concentric zones, each at a different spatial resolution. Only the **target** zone is required; the buffer and background zones are optional and extend the scene to reduce edge effects.

![](figures/nice_rgb_camera_rgb.png)

### Target

The **target zone** is the core area of interest (AOI). It is generated at the highest resolution using full DEM elevation data and landcover-derived material textures and potentially 3D object such as vegetation. All measurements of interest fall within this zone.

The target is defined by a centre coordinate and an AOI size in kilometres ([`SceneLocation`][s2gos_generator.core.config.scene.SceneLocation]).

### Buffer

The **buffer zone** (optional) surrounds the target at a coarser resolution. Its purpose is to reduce adjacency and edge-of-scene artifacts.

Configure via [`BufferConfig`][s2gos_generator.core.config.scene.BufferConfig]:

```python
buffer = BufferConfig(size_km=60.0, resolution_m=100.0)
```

### Background

The **background zone** (optional) is the outermost ring, extending the scene to the horizon. It uses a flat surface at a fixed elevation — no DEM data is used.

Configure via [`BackgroundConfig`][s2gos_generator.core.config.scene.BackgroundConfig]:

```python
background = BackgroundConfig(size_km=200.0, resolution_m=200.0, elevation=0.0)
```

## Atmosphere

The atmosphere is defined at generation time and stored in the scene description YAML. It controls how the simulator models scattering and absorption during radiative transfer.
See the [Atmosphere API reference](api/atmosphere.md) for the full configuration schema and helper functions as well as [Eradiate](https://eradiate.readthedocs.io/en/stable/) for more information.

## Output directory structure

After generation, the output directory looks like:

```
<output_dir>/<scene_name>/
├── meshes/          # PLY mesh files (target, buffer, background)
├── textures/        # Material texture images
├── data/            # Intermediate data (clipped DEM, landcover)
└── scene.yaml       # SceneDescription file for the simulator
```
