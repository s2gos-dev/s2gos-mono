# Building Processors

Load OpenBuildingMap footprints, extrude them onto the DEM, and construct meshes with
optional pitched (hip) roofs from the footprint's straight skeleton. See
[Buildings](../concepts.md#buildings).

## Footprints & meshes

::: s2gos_generator.processors.buildings.meshing.quadkeys_for_bbox
::: s2gos_generator.processors.buildings.meshing.select_tile_files
::: s2gos_generator.processors.buildings.meshing.load_building_footprints
::: s2gos_generator.processors.buildings.meshing.make_dem_elevation_sampler
::: s2gos_generator.processors.buildings.meshing.build_meshes
::: s2gos_generator.processors.buildings.meshing.BuildingMeshes
::: s2gos_generator.processors.buildings.meshing.BuildingMeshStats

## Hip roofs

::: s2gos_generator.processors.buildings.roof.build_hip_roof
::: s2gos_generator.processors.buildings.roof.compute_pitched_geometry
::: s2gos_generator.processors.buildings.skeleton.Skeleton
