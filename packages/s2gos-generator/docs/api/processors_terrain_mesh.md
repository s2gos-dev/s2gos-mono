# Terrain Mesh Processors

Turn DEM elevation grids into 3D triangle meshes via an adaptive quadtree, orchestrated by
`build_refined_mesh`. See [Adaptive terrain mesh](../concepts.md#adaptive-terrain-mesh).

## Orchestration

::: s2gos_generator.processors.terrain_mesh.builder.build_refined_mesh
::: s2gos_generator.processors.terrain_mesh.builder.extract_dem

## Decimation & grid

::: s2gos_generator.processors.terrain_mesh.adaptive_grid.AdaptiveGrid
::: s2gos_generator.processors.terrain_mesh.builder.build_decimated_grid
::: s2gos_generator.processors.terrain_mesh.error_pyramid.DemErrorPyramid

## Feature refinement & meshing

::: s2gos_generator.processors.terrain_mesh.builder.refine_grid_for_operations
::: s2gos_generator.processors.terrain_mesh.builder.grid_to_terrain_mesh
::: s2gos_generator.processors.terrain_mesh.mesh_generator.MeshGenerator

## Terrain flattening

::: s2gos_generator.processors.terrain_mesh.terraforming.GradientFilter
::: s2gos_generator.processors.terrain_mesh.terraforming.WayFlattenOperation
::: s2gos_generator.processors.terrain_mesh.terraforming.apply_way_flatten_batch
::: s2gos_generator.processors.terrain_mesh.terraforming.make_refinement_predicate
::: s2gos_generator.processors.terrain_mesh.terraforming.make_roughness_predicate
::: s2gos_generator.processors.terrain_mesh.terraforming.compute_gradient
::: s2gos_generator.processors.terrain_mesh.terraforming.TerraformOperation
