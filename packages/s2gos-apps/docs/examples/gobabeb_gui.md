# Gobabeb GUI Walkthrough

This page walks through the GUI client workflow end-to-end using the Gobabeb
PICS site as the reference example.  The same steps apply to any process
registered with the local service.

---

## Prerequisites

1. **Install the environment**

    ```bash
    pixi install
    pixi run apps-init
    ```

2. **Configure data-source credentials** — copy or create `s2gos_settings.yaml`
   in your working directory and fill in the required S3 / Sentinel Hub
   credentials.

---

## Start the local server

Open a terminal and run:

```bash
s2gos-server run -- s2gos_apps.service:service
```

The server starts on `http://127.0.0.1:8008` by default.  You should see a log
line similar to:

```
INFO:     Uvicorn running on http://127.0.0.1:8008
```

Leave this terminal open while you work in the notebook.

---

## Available Gobabeb processes

| Process ID | Description |
|---|---|
| `gobabeb/generation_config` | Creates the scene generation configuration for the Gobabeb PICS site (location, atmosphere, buffer, background). Returns a path to the saved JSON config file. |
| `gobabeb/simulation_config` | Creates the simulation configuration for a given observation time and geometry (GMT hour, samples-per-pixel). Returns a path to the saved JSON config file. |

---

## Connect the client and open the GUI

In a Jupyter notebook (start with `pixi run lab`), run:

```python
from s2gos_client.gui import Client

client = Client(api_url="http://127.0.0.1:8008")
client.show()
```

`client.show()` renders a process-selection panel.  Use the drop-down to pick
`gobabeb/generation_config` or `gobabeb/simulation_config`, fill in the input
fields, then click **Execute**.

---

## Submit a request

After filling in all required inputs and clicking **Get Request**, the GUI
stores the assembled request in the notebook variable `_request`:

```python
# Inspect the request before submitting
_request
# ExecutionRequest(inputs={...}, process_id='gobabeb/generation_config', ...)
```

Click **Execute** to submit the job to the server.

---

## Monitor jobs

```python
client.show_jobs()
```

This renders a jobs panel listing every submitted job with its current status
(`accepted`, `running`, `successful`, `failed`).  The panel auto-refreshes.

To fetch a specific job programmatically:

```python
client.get_job("job_0")
```

---

## Retrieve results

Once a job reaches `successful` status the GUI stores the output in
`_results`:

```python
# Dictionary keyed by output name
_results
# {'return_value': 'path/to/scene_gen_config.json'}

config_path = _results["return_value"]
```

The value is the `PathRef` (local or remote path) returned by the process
function.  You can pass it directly to downstream processing steps.
