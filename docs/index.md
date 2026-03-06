# DTE-S2GOS

**DTE-S2GOS** (Digital Twin Earth Synthetic Scene Generator and Observation Simulator) is a modular Python framework for
synthetic scene generation and Earth observation simulation. It enables the creation of
physically realistic synthetic landscapes and simulates satellite and ground-based
sensor observations over them.

## Packages

| Package | Description |
|---|---|
| [**s2gos-generator**](s2gos-generator/index.md) | Synthetic scene generation: vegetation placement, terrain, land cover, and material configuration |
| [**s2gos-simulator**](s2gos-simulator/index.md) | Observation simulation: sensors, Level 2 products and illumination |
| [**s2gos-utils**](s2gos-utils/index.md) | Shared utilities: scene description, configuration, credentials, data access, and common data structures |
| [**s2gos-apps**](s2gos-apps/index.md) | Process definitions for scene generation and simulation scientific applications |


## Quick Start

### Installation

```bash
git clone git@github.com:s2gos-dev/s2gos-mono.git
cd s2gos-mono
pixi install
pixi run eradiate-init   # initialises Eradiate data (required once)
```

For development (adds testing, linting, docs tools):

```bash
pixi install -e dev
```

## Configuration

S2GOS is configured via a `s2gos_settings.yaml` file placed in your working directory or
any parent directory. See the [s2gos-apps documentation](s2gos-apps/index.md) for a
full configuration reference.

## Examples

| Example | Description |
|---|---|
| [**HYPSTAR Gobabeb**](hypstar_simulations.md) | Ground sensor simulations over the Gobabeb validation site driven by a real HYPERNETS L2A dataset |
