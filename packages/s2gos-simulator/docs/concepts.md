# Concepts

This page covers the key ideas behind the S2GOS Observation Simulator: spectral modes, how atmosphere interacts with measurements, and the available sensor types.

## Spectral Modes

The simulator uses [Eradiate](https://eradiate.eu) as its radiative transfer backend. Eradiate operates in a **spectral mode** that determines how wavelength integration and gas absorption are handled. The mode is set via `backend_hints` in the simulation config:

```python
backend_hints={"mode": "ckd"}
```

## Sensors

The simulator supports the following platform types, each with different instrument options and viewing geometries.

### Satellite

Satellite sensors produce 2D imagery of a target footprint with an instrument, either built in (by key from `INSTRUMENTS`) or defined inline. Two kinds of instrument:

- **Multispectral** — Few broad bands, each simulated at its native resolution.
- **Hyperspectral** — Hundreds of narrow contiguous channels, simulated as one measure. Gaussian SRF post-processing simulates the instrument's spectral response.

Built-in instruments: Sentinel-2A/2B MSI (written as an EOPF-like L1C product), Sentinel-3A/3B OLCI, CHIME (approximation).

**CKD mode** is recommended for all satellite instrument simulations.

### Ground

Ground-based sensors are placed at low heights for in-situ validation measurements:

- **HYPSTAR** — HYPERNETS sensor
- **Perspective camera** — Standard camera with configurable FOV
- **Radiancemeter** — Directional radiance measurement

See the [Sensors API reference](api/sensors.md) for full configuration details and the [`create_hypstar_sensor()`][s2gos_simulator.config.sensors.create_hypstar_sensor] helper.

## Level 2 Products

Every simulation includes top of atmosphere solar irradiance automatically. All other quantities are BOA (bottom-of-atmosphere) measurements and must be explicitly requested by listing them under `measurements` in [`SimulationConfig`][s2gos_simulator.config.simulation.SimulationConfig].

| Quantity | Illumination | Outgoing geometry |
|---|---|---|
| **Irradiance** | — | Downwelling flux at the surface (BOA) |
| **BRF** | Direct beam only | Single direction |
| **HDRF** | Direct + diffuse (atmosphere) | Single direction |
| **HCRF** | Direct + diffuse (atmosphere) | Finite cone (camera FOV) |
| **BHR** | Isotropic | All directions (albedo) |

[`PixelBRFConfig`][s2gos_simulator.config.measurements.PixelBRFConfig], [`PixelHDRFConfig`][s2gos_simulator.config.measurements.PixelHDRFConfig], and [`PixelBHRConfig`][s2gos_simulator.config.measurements.PixelBHRConfig] compute the same quantities at locations derived from a satellite sensor's pixel footprint.

See the [Measurements API reference](api/measurements.md) for full configuration details.
