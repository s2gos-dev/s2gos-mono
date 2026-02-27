# Setup

## Installation

The simulator uses [pixi](https://pixi.sh) for environment management.

```bash
# Install the default environment
pixi install

# Or install the development environment (includes test and docs extras)
pixi install -e dev
```

The main runtime dependency is [Eradiate](https://eradiate.eu),
which provides the underlying radiative transfer engine, we recommend reading its [data guide](https://eradiate.readthedocs.io/en/stable/data/intro.html) before attempting to carry out any simulations.
