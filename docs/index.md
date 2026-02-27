# S2GOS

**S2GOS** (Sentinel-2 Ground Observation Simulator) is a modular Python framework for
synthetic scene generation and Earth observation simulation. It enables the creation of
physically realistic synthetic landscapes and simulates satellite, airborne, and ground-based
sensor observations over them.

## Packages

| Package | Description |
|---|---|
| [**s2gos-generator**](s2gos-generator/index.md) | Synthetic scene generation: vegetation placement, terrain, land cover, and material configuration |
| [**s2gos-simulator**](s2gos-simulator/index.md) | Observation simulation: radiative transfer via Eradiate, sensor models, illumination, and spectral configuration |
| [**s2gos-utils**](s2gos-utils/index.md) | Shared utilities: configuration, credentials, data access, and common data structures |
| [**s2gos-apps**](s2gos-apps/index.md) | Application workflows: process definitions for MTR demo, PnP, and multi-temporal radiometric demonstration |


## Quick Start

### Installation

```bash
git clone git@github.com:s2gos-dev/s2gos-mono.git
cd s2gos-mono
pixi install
pixi run apps-init   # initialises Eradiate data (required once)
```

For development (adds testing, linting, docs tools):

```bash
pixi install -e dev
pixi run -e dev test
pixi run -e dev docs
```

## Examples

| Example | Description |
|---|---|
| [**HYPSTAR Gobabeb**](hypstar_simulations.md) | Ground sensor simulations over the Gobabeb validation site driven by a real HYPERNETS L2A dataset |

## Configuration

S2GOS is configured via a `s2gos_settings.yaml` file placed in your working directory or
any parent directory. See the [s2gos-apps documentation](s2gos-apps/index.md) for a
full configuration reference.