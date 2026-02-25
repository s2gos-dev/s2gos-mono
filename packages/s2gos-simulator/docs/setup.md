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

## Quick start

The entry point is `SimulationConfig`. Below is a minimal satellite simulation — a
Sentinel-2 nadir image with the sun at 30 ° zenith:

```python
from s2gos_simulator import (
    SimulationConfig,
    SatelliteSensor, SatellitePlatform, SatelliteInstrument,
    DirectionalIllumination,
    AngularViewing,
)

sensor = SatelliteSensor(
    id="s2_red",
    platform=SatellitePlatform.SENTINEL_2A,
    instrument=SatelliteInstrument.MSI,
    band="4",
    viewing=AngularViewing(zenith=0.0, azimuth=0.0),
    film_resolution=(512, 512),
    target_center_lat=51.5,
    target_center_lon=-1.8,
    target_size_km=2.0,
)

config = SimulationConfig(
    name="quickstart",
    illumination=DirectionalIllumination(zenith=30.0, azimuth=135.0),
    sensors=[sensor],
)
```

Solar angles can also be derived automatically from a date and location:

```python
from datetime import datetime
illumination = DirectionalIllumination.from_date_and_location(
    time=datetime(2024, 6, 21, 10, 30),
    latitude=51.5,
    longitude=-1.8,
)
```

Then to run a simulation you will also need a scene (see S2GOS Scene Generator):

```python
from s2gos_simulator.backends.eradiate.backend import (
    EradiateBackend,
)

simulator = EradiateBackend(config)
simulator.run_simulation(
    scene_description,
    scene_dir=scene_description_path.parent,
    output_dir=simulation_output_dir,
)
```

## Configuration serialisation

Configurations can be saved and reloaded as JSON:

```python
config.to_json("my_config.json")
config2 = SimulationConfig.from_json("my_config.json")
```
