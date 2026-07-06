# Changelog

All notable changes to **s2gos-generator** are documented here.

## [0.2.0] - 2026-07-06

### Added

- **Building generation** — 3D buildings in the target zone from OpenBuildingMap tiles, with
  numeric or taxonomy-based heights, per-building materials, and optional pitched (hip) roofs.
- **Road support** — roads rasterised into the target texture from OpenStreetMap or a local
  JSON file, with widths and surface materials derived from OSM tags.
- **Adaptive terrain mesh** — quadtree refinement of the target mesh under features such as
  roads (with optional flattening), plus decimation of smooth terrain.
- **Spectral material matching** — selected land-cover classes matched to real diffuse
  materials from a Sentinel-2 composite.
- **Resource caching** — completed pipeline resources are cached and reused on re-runs when
  unchanged.

### Changed

- Path handling migrated to `PathRef` / `UPath`, with S3 read/write support (see `s2gos-utils`).
- Reorganized processors into cohesive subpackages; public API unchanged.
