# Changelog

All notable changes to **s2gos-utils** are documented here.

## [0.2.0] - 2026-07-06

### Added

- **`PathRef`** — OGC API Link-style path handling (`href` + optional credential id `cid`) for
  API/config boundaries; credentials are referenced by id and never embedded, so configs
  serialise safely.
- Coordinate-system helper that reprojects a `GeoDataFrame` to scene-local metres.

### Changed

- Path handling refactored around `PathRef` / `UPath`; helpers now accept `PathLike | PathRef`.
- S3/UPath compatibility fixes across generator, simulator, and apps; corrected credential
  environment-variable naming; `print` calls replaced with logging.
