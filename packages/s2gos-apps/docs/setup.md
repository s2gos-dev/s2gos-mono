# Setup

## Installation

S2GOS Apps uses [pixi](https://pixi.sh) for environment management.

```bash
# Install the default environment
pixi install

# Or install the development environment (includes test and docs extras)
pixi install -e dev
```

## Eradiate Data

The simulation workflows depend on [Eradiate](https://eradiate.eu) for radiative transfer. Eradiate requires spectral and atmospheric datasets to be downloaded once before use:

```bash
pixi run eradiate-init
```

See the [Eradiate data guide](https://eradiate.readthedocs.io/en/stable/data/intro.html) for more information on configuring the data install location.
