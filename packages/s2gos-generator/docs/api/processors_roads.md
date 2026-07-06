# Road Processors

Fetch and parse OpenStreetMap road geometry, (de)serialize the roads
sidecar, and build the terrain-flatten operations that sit roads on well-resolved terrain.
For the concept and configuration, see [Roads](../concepts.md#roads) and
[Buildings & Roads config](buildings_roads.md).

## Fetching & parsing

::: s2gos_generator.processors.roads.fetch_osm_data
::: s2gos_generator.processors.roads.parse_roads
::: s2gos_generator.processors.roads.Road
::: s2gos_generator.processors.roads.RoadsFetchError

## Sidecar (de)serialization

::: s2gos_generator.processors.roads.roads_to_sidecar
::: s2gos_generator.processors.roads.roads_from_sidecar

## Terrain flattening

::: s2gos_generator.processors.roads.build_road_terraform_operations
