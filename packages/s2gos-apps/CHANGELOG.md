# Changelog

All notable changes to **s2gos-apps** are documented here.

## [0.2.0] - 2026-07-06

### Changed

- Path handling migrated to `PathRef` / `UPath`, with S3 compatibility across the
  site processes (see `s2gos-utils`).
- `simulation_configs()` now returns a `PathRef` consistently (previously returned
  an unwrapped `UPath`).
- `print` calls replaced with logging.
