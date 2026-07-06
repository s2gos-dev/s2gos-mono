# Spectral Matching Processors

Functions behind spectral material matching, the algorithm layer called by the
texture pipeline. They fetch Sentinel-2 reflectance, cluster and match it against a
material library with the Spectral Angle Mapper, and (de)serialize the matched-material
result. For more information see
[Spectral material matching](../concepts.md#spectral-material-matching) and
[Spectral Matching config](spectral_matching.md).

## Sentinel-2 acquisition

::: s2gos_generator.processors.spectral.sentinel2.fetch_s2_reflectance

## Spectral Angle Mapper

::: s2gos_generator.processors.spectral.sam.spectral_angle
::: s2gos_generator.processors.spectral.sam.cluster_class_reflectance
::: s2gos_generator.processors.spectral.sam.match_clusters_to_library

## Material library

::: s2gos_generator.processors.spectral.library.load_candidate_library
::: s2gos_generator.processors.spectral.library.CandidateSpectrum

## Diversification & sidecar

::: s2gos_generator.processors.spectral.diversify.diversify_selection_texture
::: s2gos_generator.processors.spectral.diversify.matched_materials_to_sidecar
::: s2gos_generator.processors.spectral.diversify.matched_materials_from_sidecar
