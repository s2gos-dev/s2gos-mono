# Quickstart: Simulate an RGB Observation

This example simulates a nadir **RGB camera** observation from 20 km altitude
over a generated Gobabeb scene. You will define a perspective camera sensor with
three wavelengths (red, green, blue), set the solar illumination from a specific
date and location, run the Eradiate radiative transfer backend, and produce a
synthetic RGB image.

---

## Prerequisites

- **Eradiate** installed and its spectral data downloaded
  (`pixi run apps-init`). See the [Setup](../setup.md) page for details.
- A **generated scene YAML** from the generator quickstart. If you have not
  generated one yet, follow
  [Quickstart: Generate a Gobabeb Scene](../../s2gos-generator/examples/gobabeb_quickstart.md)
  first. You will need the path to the output `gobabeb.yml` file.

---

## Step 1 — Define the sensor

Create a `UAVSensor` with a perspective camera looking straight down from 20 km.
The spectral response uses three delta wavelengths at 440 nm (blue), 550 nm
(green), and 660 nm (red) to produce a synthetic RGB image.

```python
import numpy as np
from s2gos_simulator import (
    UAVSensor,
    UAVInstrumentType,
    LookAtViewing,
    SpectralResponse,
)
from s2gos_simulator.config.sensors import PostProcessingOptions

# Camera altitude and field of view
altitude_m = 20_000        # 20 km
fov_deg = 50               # 50° field of view
target_size_km = 10         # Must match the generator target area

sensor = UAVSensor(
    id="rgb_camera",
    instrument=UAVInstrumentType.PERSPECTIVE_CAMERA,
    viewing=LookAtViewing(
        origin=[0.0, 0.0, altitude_m],
        target=[0.0, 0.0, 0.0],
        up=[0, 1, 0],
    ),
    srf=SpectralResponse(
        type="delta",
        wavelengths=[440.0, 550.0, 660.0],  # Blue, Green, Red
    ),
    fov=fov_deg,
    resolution=[512, 512],
    samples_per_pixel=8,
    post_processing=PostProcessingOptions(generate_rgb_image=True),
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `origin` | `[0, 0, 20000]` | Camera at 20 km altitude, centred on the scene |
| `fov` | 50 | 50° field of view |
| `wavelengths` | 440, 550, 660 nm | Blue, green, red channels for RGB compositing |
| `resolution` | 512 &times; 512 | Pixel grid of the rendered image |
| `samples_per_pixel` | 8 | Monte Carlo samples per pixel (increase for less noise) |

---

## Step 2 — Set the illumination

Use `DirectionalIllumination.from_date_and_location()` to compute the solar
zenith and azimuth angles from a specific date/time and the Gobabeb coordinates.
The method uses Skyfield orbital mechanics under the hood.

```python
from datetime import datetime
from s2gos_simulator import DirectionalIllumination

illumination = DirectionalIllumination.from_date_and_location(
    time=datetime(2024, 7, 15, 9, 0, 0),   # 15 July 2024, 09:00 UTC
    latitude=-23.6015,
    longitude=15.1259,
)
```

!!! note
    The time is in UTC. Make sure the sun is above the horizon at the chosen
    time — the method raises a `ValueError` otherwise.

---

## Step 3 — Build the simulation configuration

Assemble sensor, illumination, and backend hints into a `SimulationConfig`.
The `backend_hints` select Eradiate's monochromatic mode (`mono`), which
evaluates the radiative transfer independently at each of the three wavelengths.

```python
from s2gos_simulator import SimulationConfig

config = SimulationConfig(
    name="gobabeb_rgb",
    description="Nadir RGB camera observation from 20 km over Gobabeb.",
    illumination=illumination,
    sensors=[sensor],
    backend_hints={"eradiate": {"mode": "mono"}},
)
```

---

## Step 4 — Save the configuration

Persist the simulation configuration for reproducibility.

```python
from pathlib import Path

config_dir = Path("./sim_config")
config_dir.mkdir(exist_ok=True)

config.to_json(config_dir / "gobabeb_sim_config.json")
```

---

## Step 5 — Run the simulation

Load the scene description produced by the generator and pass it to the
`EradiateBackend` together with the simulation configuration.

```python
from s2gos_simulator import EradiateBackend
from s2gos_utils import SceneDescription
from upath import UPath

# Path to the generated scene description
scene_path = UPath("./gen_output/gobabeb/gobabeb.yml")
scene_description = SceneDescription.load_yaml(scene_path)

# Run simulation
simulator = EradiateBackend(config)
simulator.run_simulation(
    scene_description,
    scene_dir=scene_path.parent,
    output_dir=UPath("./sim_output/gobabeb"),
)
```

Eradiate performs Monte Carlo ray tracing through the 3D scene once per
wavelength (440, 550, 660 nm). Because `generate_rgb_image=True` is set in the
post-processing options, the backend automatically composites the three channels
into an RGB image after simulation.

---

## Step 6 — Inspect the results

After the simulation completes, the output directory results in zarr format as wel as an rgb plot.

## Previous step

This example continues from
[Quickstart: Generate a Gobabeb Scene](../../s2gos-generator/examples/gobabeb_quickstart.md).