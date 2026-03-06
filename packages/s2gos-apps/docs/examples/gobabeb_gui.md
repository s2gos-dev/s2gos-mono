# Gobabeb GUI Walkthrough

This tutorial walks through a complete scene generation and simulation workflow
for the **Gobabeb site** (Namibia) using the GUI client in Jupyter Lab.
By the end you will have generated a 3D scene from real DEM and landcover data,
simulated an observation with Eradiate, and retrieved a rendered RGB image —
all driven from interactive GUI widgets.

The same steps apply to any site registered with the local service (Frascati,
Kairouan, Pisa, PNP).

---

## How processes work

Workflows are built from **processes** — Python functions decorated with
`@registry.process()`.  The decorator registers the function with
[s2gos-controller](https://s2gos-dev.github.io/s2gos-controller/) (procodile +
wraptile + s2gos-client), which automatically exposes it as an OGC API endpoint,
a GUI form in Jupyter, and a CLI command.  Function parameters annotated with
Pydantic `Field` become input fields, defaults, and descriptions — no extra
wiring needed.

---

## Step 1: Start the local server

Open a terminal and run:

```bash
s2gos-server run -- s2gos_apps.service:service
```

This starts a FastAPI server on `http://127.0.0.1:8008` that exposes every
registered process as an OGC API endpoint. You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8008
```

Leave this terminal open while you work in the notebook.

---

## Step 2: Launch Jupyter Lab

In a second terminal, start Jupyter Lab:

```bash
pixi run lab
```

Navigate to or create a new notebook for this walkthrough.

---

## Step 3: Connect the GUI client

```python
from s2gos_client.gui import Client

client = Client(api_url="http://127.0.0.1:8008")
client.show()
```

`Client` connects to the local server and fetches the list of available
processes.  `client.show()` renders a panel with:

- a **drop-down** listing every registered process,
- **input fields** generated from the selected process's parameter annotations,
- an **Execute** button to submit the job.

---

## Step 4: Create the scene generation configuration

Select **`gobabeb-generation-config`** from the drop-down.

The form shows the following fields (with defaults pre-filled):

| Field | Default | Description |
|---|---|---|
| `scene_name` | `gobabeb` | Identifier used for output directory and file names |
| `target_lat` | `-23.6015417` | Centre latitude of the area of interest |
| `target_lon` | `15.1258696` | Centre longitude of the area of interest |
| `target_size` | `10` | Side length of the target area in km |
| `config_output_dir` | *(empty)* | Where to save the config JSON (defaults to `./gen_config/`) |
| `scene_output_dir` | *(empty)* | Where the generator will write scene assets (defaults to `./gen_output/`) |

Click **Execute** to submit.

**What happens under the hood:** the process creates a [`SceneGenConfig`](../../s2gos-generator/api/scene_config.md) object
with Gobabeb-specific settings — a 10 km target area, a 60 km buffer zone at
60 m resolution, a 150 km background at 200 m resolution, and a US Standard
molecular atmosphere (AFGL 1986, GECKO absorption database).  The config is
saved as a JSON file.

Once the job completes, the GUI stores the result in the notebook variable
`_results`:

```python
_results
# {'return_value': './gen_config/gobabeb_gen_config.json'}

gen_config_path = _results["return_value"]
```

---

## Step 5: Run scene generation

Select **`common-generation`** from the drop-down.

| Field | Value |
|---|---|
| `config_path` | Paste the path from Step 4, e.g. `./gen_config/gobabeb_gen_config.json` |

Click **Execute**.

This process loads the config JSON, builds a [`SceneGenerationPipeline`](../../s2gos-generator/api/pipeline.md) DAG, and
executes it.  The pipeline:

1. defines the area of interest from the target coordinates,
2. fetches DEM tiles and landcover data for target, buffer, and background zones,
3. generates terrain meshes (PLY) and landcover textures (PNG),
4. assembles everything into a **scene description** YAML file.

The output is the path to the scene YAML, e.g.
`./gen_output/gobabeb/gobabeb.yml`.

```python
scene_yaml_path = _results["return_value"]
```

---

## Step 6: Create the simulation configuration

Select **`gobabeb-simulation-config`** from the drop-down.

| Field | Default | Description |
|---|---|---|
| `scene_name` | `gobabeb` | Must match the generation scene name |
| `target_lat` | `-23.6015417` | Same as generation |
| `target_lon` | `15.1258696` | Same as generation |
| `target_size` | `10` | Same as generation |
| `gmt_hour` | `9` | Hour of observation in GMT — used to compute solar angles via Skyfield |
| `spp` | `8` | Monte Carlo samples per pixel — higher values reduce noise |
| `config_output_dir` | *(empty)* | Where to save the config JSON (defaults to `./sim_config/`) |

Click **Execute**.

The process builds a [`SimulationConfig`](../../s2gos-simulator/api/simulation.md) with a top-down perspective camera
(512 x 512 px, 50° FOV) capturing three wavelengths (440, 550, 660 nm for a
synthetic RGB), directional illumination derived from the solar position at the
given hour, and mono-mode Eradiate backend hints. The config is saved as JSON.

```python
sim_config_path = _results["return_value"]
```

---

## Step 7: Run the simulation

Select **`common-simulation`** from the drop-down.

| Field | Value |
|---|---|
| `scene_description_path` | Path from Step 5, e.g. `./gen_output/gobabeb/gobabeb.yml` |
| `config_path` | Path from Step 6, e.g. `./sim_config/gobabeb_sim_config.json` |
| `simulation_output_dir` | *(optional)* e.g. `./sim_output/gobabeb` |

Click **Execute**.

Eradiate loads the 3D scene description and runs Monte Carlo radiative transfer
for each wavelength. This is the most compute-intensive step. When finished the
process writes raw result datasets (Zarr) and a composited RGB image to the
output directory.

```python
sim_output_dir = _results["return_value"]
```

---

## Step 8: Monitor jobs and display results

At any point you can monitor all submitted jobs:

```python
client.show_jobs()
```

This renders a panel listing every job with its current status (`accepted`,
`running`, `successful`, `failed`). The panel auto-refreshes.

To inspect a specific job programmatically:

```python
client.get_job("job_0")
```

Once the simulation job is successful, display the rendered image:

```python
from IPython.display import Image, display

display(Image(filename="./sim_output/gobabeb/rgb_camera_rgb.png"))
```

---

## Alternative: Python API

The same workflow can be run without the GUI by calling the process functions
directly. This is useful for scripting, batch runs, or CI pipelines.

```python
from s2gos_apps.processes.gobabeb import generation_configs, simulation_configs
from s2gos_apps.processes.common import generation, simulation

scene_name = "gobabeb"
target_lat = -23.6015417
target_lon = 15.1258696
target_size = 10
gmt_hour = 9
spp = 8

# Step 1: Create generation config
gen_config_path = generation_configs(
    scene_name=scene_name,
    target_lat=target_lat,
    target_lon=target_lon,
    target_size=target_size,
    config_output_dir="./use_cases/gen_config",
    scene_output_dir="./use_cases/gen_output",
)["return_value"]

# Step 2: Run generation pipeline
scene_yaml_path = generation.generation(
    config_path=gen_config_path,
)["return_value"]

# Step 3: Create simulation config
sim_config_path = simulation_configs(
    scene_name=scene_name,
    target_lat=target_lat,
    target_lon=target_lon,
    target_size=target_size,
    gmt_hour=gmt_hour,
    config_output_dir="./use_cases/sim_config",
)["return_value"]

# Step 4: Run simulation
sim_output_dir = simulation.simulation(
    scene_description_path=scene_yaml_path,
    config_path=sim_config_path,
    simulation_output_dir=f"./use_cases/sim_output/{scene_name}",
)["return_value"]

# Display result
from IPython.display import Image, display
display(Image(filename=f"{sim_output_dir}/rgb_camera_rgb.png"))
```
