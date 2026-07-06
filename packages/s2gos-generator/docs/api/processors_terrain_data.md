# Terrain Data Processors

Find, merge, and regrid the source raster tiles (Copernicus GLO-30 DEM, ESA WorldCover land
cover) that clip to a scene's extents, plus the buffer mask derived from them.

## Tile processors

::: s2gos_generator.processors.terrain_data.base_processor.BaseTileProcessor
::: s2gos_generator.processors.terrain_data.dem.DEMProcessor
::: s2gos_generator.processors.terrain_data.landcover.LandCoverProcessor

## Regridding & masks

::: s2gos_generator.processors.terrain_data.datautil.regrid_to_projection
::: s2gos_generator.processors.terrain_data.masks.generate_buffer_mask
