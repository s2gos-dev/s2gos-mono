# Snow Processors

Estimate seasonal snow cover so the texture generator can paint snow onto the selection
texture. The model works in two steps: it estimates a surface temperature for each pixel —
either from a simple built-in climatology (warmer near the equator, colder at altitude, with a
summer/winter swing) or by interpolating a CAMS atmospheric profile onto the terrain — and then
turns that temperature into a snow probability that rises smoothly as it drops below freezing.
Two seasons are supported, June and December. Configured via `SnowConfig`
(see [Scene config](scene_config.md)).

::: s2gos_generator.processors.terrain_texture.snow.get_day_of_year
::: s2gos_generator.processors.terrain_texture.snow.apply_spatial_smoothing
::: s2gos_generator.processors.terrain_texture.snow.calculate_seasonal_amplitude
::: s2gos_generator.processors.terrain_texture.snow.interpolate_cams_temperature
::: s2gos_generator.processors.terrain_texture.snow.calculate_temperature_field
::: s2gos_generator.processors.terrain_texture.snow.calculate_snow_probability_map
