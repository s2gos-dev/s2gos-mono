# Way Processors

Fetch and parse OpenStreetMap road geometry, (de)serialize the roads
sidecar, and build the terrain-flatten operations that sit roads on well-resolved terrain.
For the concept and configuration, see [Ways](../concepts.md#ways) and
[Buildings & Roads config](buildings_roads.md).

## Fetching & parsing

::: s2gos_generator.processors.ways.fetch_osm_data
::: s2gos_generator.processors.ways.parse_ways
::: s2gos_generator.processors.ways.Way
::: s2gos_generator.processors.ways.WaysFetchError

## Sidecar (de)serialization

::: s2gos_generator.processors.ways.ways_to_sidecar
::: s2gos_generator.processors.ways.ways_from_sidecar

## Terrain flattening

::: s2gos_generator.processors.ways.build_way_terraform_operations
