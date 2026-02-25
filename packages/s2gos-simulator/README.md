# S2GOS Observation Simulator

Earth observation simulation engine for the DTE-S2GOS service. This package simulates realistic satellite, UAV, and ground-based observations using physically-based radiative transfer modeling.

## Overview

The S2GOS Observation Simulator provides:

- **Multi-platform simulation**: Satellite, UAV, and ground-based sensor configurations
- **Physics-based modeling**: Accurate radiative transfer simulation via Eradiate
- **Flexible sensor models**: Support for major Earth observation platforms and custom instruments
- **Comprehensive measurements**: Radiance, reflectance, and specialized radiative quantities
- **Atmospheric modeling**: Molecular, aerosol, and heterogeneous atmosphere configurations

## Installation

### Prerequisites

- Python 3.9+
- [pixi](https://pixi.sh/) (recommended) or conda/mamba

### Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd ./s2gos-simulator

# Install development environment with pixi
pixi install -e dev
```