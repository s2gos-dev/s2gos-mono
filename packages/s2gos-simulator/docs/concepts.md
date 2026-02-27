# Concepts

This page covers the key ideas behind the S2GOS Observation Simulator: spectral modes, how atmosphere interacts with measurements, and the available sensor types.

## Spectral Modes

The simulator uses [Eradiate](https://eradiate.eu) as its radiative transfer backend. Eradiate operates in a **spectral mode** that determines how wavelength integration and gas absorption are handled. The mode is set via `backend_hints` in the simulation config:

```python
backend_hints={"mode": "ckd"}
```

## Sensors

The simulator supports three platform types, each with different instrument options and viewing geometries.

### Satellite

Satellite sensors produce 2D imagery at a specified ground resolution. Two main instrument categories:

- **MSI** (MultiSpectral Instrument) — Few broad bands. Sentinel-2A/2B. Each band is simulated independently with its spectral response function.
- **HSI** (HyperSpectral Instrument) — Hundreds of narrow contiguous channels such as teh guture CHUME mission. Gaussian SRF post-processing simulates the instrument's spectral response.

Built-in satellite platforms: Sentinel-2A/2B, Sentinel-3A/3B, CHIME (approcimation), EnMAP. Custom platforms are supported with user-provided SRFs.

**CKD mode** is recommended for all satellite instrument simulations.

### UAV

UAV sensors are placed at a specified altitude with a configurable viewing geometry:

- **Perspective camera** — 2D imaging with a field of view and film resolution
- **Radiancemeter** — Single-point radiance measurement

### Ground

Ground-based sensors are placed at low heights for in-situ validation measurements:

- **HYPSTAR** — HYPERNETS network sensor. Circular FOV, approximate SRF
- **Perspective camera** — Standard camera with configurable FOV
- **Radiancemeter** — Directional radiance measurement

See the [Sensors API reference](api/sensors.md) for full configuration details and helper functions like `create_chime_sensor()` and `create_hypstar_sensor()`.
