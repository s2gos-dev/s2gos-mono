# MTR Demo Processors Documentation

## Overview

The MTR (Multi-Temporal Radiometric) demo processors provide a streamlined interface for generating scenes and running simulations for the Patagonia National Park (PNP) demonstration. The implementation consists of two self-contained processors that combine configuration creation and execution into single steps.

## Files Created

1. **`src/s2gos_apps/processes/mtr_demo.py`** - Main processors implementation
2. **`example/mtr_demo_example.py`** - Example usage script
3. **`docs/mtr_demo_processors.md`** - This documentation

## Processors

### 1. `mtr_demo/generation`

Generates a 3D scene for PNP with seasonal variations.

**Process ID**: `mtr_demo/generation`

**Parameters**:
- `month` (Month, default=DECEMBER): Month for simulation
  - `DECEMBER`: Summer in Patagonia (green vegetation, no snow)
  - `JUNE`: Winter in Patagonia (bare trees, snow cover)
- `hour_utc` (float, required): Hour of observation in UTC (0-23)
- `random_seed` (int, default=42): RNG seed for vegetation placement
- `config_output_dir` (PathLike, optional): Generation config output directory
- `scene_output_dir` (PathLike, optional): Scene description output directory

**Returns**: Path to generated scene description YAML file

**What it does**:
1. Creates seasonal scene configuration (materials, vegetation, atmosphere)
2. Validates configuration
3. Saves generation config JSON
4. Runs scene generation pipeline
5. Returns path to scene YAML

**Example**:
```python
from s2gos_apps.processes.mtr_demo import Month, mtr_demo_generation

scene_path = mtr_demo_generation(
    month=Month.DECEMBER,
    hour_utc=15.0,
    random_seed=42
)
```

---

### 2. `mtr_demo/simulation`

Runs simulation with configurable observation types.

**Process ID**: `mtr_demo/simulation`

**Parameters**:
- `scene_description_path` (PathLike, required): Path to scene YAML from generation
- `month` (Month, default=DECEMBER): Month for simulation (determines observation date)
- `hour_utc` (float, required): Hour of observation in UTC (0-23)
- `observation` (Union of observation configs, required): Observation type configuration
- `spp` (int, default=8): Samples per pixel for Monte Carlo simulation
- `config_output_dir` (PathLike, optional): Simulation config output directory
- `simulation_output_dir` (PathLike, optional): Simulation output directory

**Returns**: Path to simulation output directory

**What it does**:
1. Creates simulation configuration based on observation type
2. Saves simulation config JSON
3. Runs Eradiate simulation
4. Returns path to output directory

---

## Observation Types

The `observation` parameter accepts one of five mutually exclusive configuration types:

### 1. CHIME Sensor

Hyperspectral satellite sensor.

```python
from s2gos_apps.processes.mtr_demo import ChimeSensorConfig

observation = ChimeSensorConfig(
    zenith=3.0  # Viewing zenith angle in degrees (default: 3.0)
)
```

**Output**: Full hyperspectral image from CHIME satellite perspective.

---

### 2. MSI (Sentinel-2)

Multispectral satellite sensor with configurable bands.

```python
from s2gos_apps.processes.mtr_demo import MsiSensorConfig

observation = MsiSensorConfig(
    zenith=3.0,  # Viewing zenith angle (default: 3.0)
    bands=["2", "3", "4", "8", "11", "12"]  # MSI bands (default: visible + NIR + SWIR)
)
```

**Available bands**: 2, 3, 4, 5, 6, 7, 8, 9, 8a, 10, 11, 12

**Output**: Separate images for each selected band from Sentinel-2 MSI perspective.

---

### 3. HYPSTAR

Ground-based hyperspectral sensor with HCRF post-processing.

```python
from s2gos_apps.processes.mtr_demo import HypstarObservation

observation = HypstarObservation(
    fov=5.0,  # Field of view in degrees (default: 5.0)
    resolution=(128, 128)  # Sensor resolution (default: 128x128)
)
```

**Features**:
- Mounted on tower at PNP
- Nadir viewing (zenith=177°, azimuth=180°)
- HCRF (Hemispherical-Conical Reflectance Factor) processing
- Circular FOV mask
- RGB image generation
- Spectral response function application
- Spatial averaging

**Output**:
- Hyperspectral HCRF measurements
- RGB visualization
- Irradiance measurements

---

### 4. RGB Camera

Perspective camera viewing the tower from a configurable position.

```python
from s2gos_apps.processes.mtr_demo import RgbCameraConfig

observation = RgbCameraConfig(
    distance_m=100.0,  # Distance from tower in meters (default: 100.0)
    azimuth=45.0,  # Azimuth angle: 0=North, 90=East, 180=South, 270=West
    elevation_angle=10.0,  # Elevation angle above horizon in degrees
    fov=40.0,  # Field of view in degrees (default: 40.0)
    resolution=(2000, 2000)  # Image resolution (default: 2000x2000)
)
```

**Camera positioning**:
- Camera is positioned at `distance_m` from tower
- Azimuth determines horizontal direction (0°=North, clockwise)
- Elevation determines vertical angle above horizon
- Camera always looks AT the tower center

**Output**: RGB image of the tower from specified viewpoint.

---

### 5. Satellite HDRF ⚠️ PLACEHOLDER

HDRF measurements in satellite projection (3x3 pixels around tower).

```python
from s2gos_apps.processes.mtr_demo import SatelliteHdrfConfig

observation = SatelliteHdrfConfig()
```

**Status**: TO BE IMPLEMENTED

This observation type will print a placeholder message and return `None`. Future implementation will include:
- Pixel coordinate calculation for tower location
- 3x3 grid generation around tower pixel
- Multiple HDRF measurements creation

---

## Seasonal Variations

The `month` parameter controls multiple aspects of the scene:

### December (Summer in Patagonia)

- **Materials**: `materials.json` (green vegetation)
- **Trees**: `tls_tree_*_prospect.xml` (7 variants with leaves)
- **Shrubs**: `tls_tree_336_prospect.xml` (with leaves)
- **Snow**: None
- **Thermoprops date**: 2020-12-21
- **Observation date**: Fixed to December 21, 2024

### June (Winter in Patagonia)

- **Materials**: `materials_winter.json` (bare vegetation)
- **Trees**: `tls_tree_*_winter.xml` (7 variants without leaves)
- **Shrubs**: `tls_tree_336_winter.xml` (without leaves)
- **Snow**: Applied (snow_season_month="june")
- **Thermoprops date**: 2020-06-21
- **Observation date**: Fixed to June 21, 2024

---

## Scene Configuration

All generated scenes include:

- **Location**: PNP, Chile (lat=-46.917, lon=-72.450)
- **Target size**: 10 km × 10 km
- **Resolution**: 10m for target area
- **Vegetation**:
  - Trees (landcover 10): 1067 trees/hectare, 7 species
  - Shrubs (landcover 20): 40 shrubs/hectare, 1 species
  - Scale variation: 0.8-1.15 for trees, 0.4-0.8 for shrubs
- **Atmosphere**:
  - Molecular atmosphere (MONOTROPA database)
  - Aerosol layer (SIXSV_CONTINENTAL, optical_thickness=0.1, 600-1600m altitude)
  - Seasonal thermophysical properties
- **Tower**: Always included at scene coordinates (-220m, 850m)
- **Buffer/Background**: Disabled for demo

---

## Demo Scenarios

The implementation supports the three demo scenarios outlined in the MTR demo plan:

### Demo 1: RGB Image of Tower in December

Generate summer scene and create RGB visualization.

```python
from s2gos_apps.processes.mtr_demo import (
    Month, mtr_demo_generation, mtr_demo_simulation, RgbCameraConfig
)

# Generate summer scene
scene_path = mtr_demo_generation(
    month=Month.DECEMBER,
    hour_utc=15.0,
    random_seed=42
)

# Simulate RGB camera
output = mtr_demo_simulation(
    scene_description_path=scene_path,
    month=Month.DECEMBER,
    hour_utc=15.0,
    observation=RgbCameraConfig(
        distance_m=150.0,
        azimuth=135.0,
        elevation_angle=15.0
    ),
    spp=8
)
```

### Demo 2: RGB Image of Tower in June

Generate winter scene and compare with summer.

```python
# Generate winter scene
scene_path = mtr_demo_generation(
    month=Month.JUNE,
    hour_utc=16.0,
    random_seed=42
)

# Same camera parameters for comparison
output = mtr_demo_simulation(
    scene_description_path=scene_path,
    month=Month.JUNE,
    hour_utc=16.0,
    observation=RgbCameraConfig(
        distance_m=150.0,
        azimuth=135.0,
        elevation_angle=15.0
    ),
    spp=8
)
```

### Demo 3: Multiple Sensors in Parallel

Generate one scene and run multiple simulations.

```python
from s2gos_apps.processes.mtr_demo import (
    ChimeSensorConfig, MsiSensorConfig, HypstarObservation
)

# Generate scene once
scene_path = mtr_demo_generation(
    month=Month.DECEMBER,
    hour_utc=15.0,
    random_seed=42
)

# Run CHIME simulation
chime_output = mtr_demo_simulation(
    scene_description_path=scene_path,
    month=Month.DECEMBER,
    hour_utc=15.0,
    observation=ChimeSensorConfig(zenith=3.0),
    spp=8
)

# Run MSI simulation
msi_output = mtr_demo_simulation(
    scene_description_path=scene_path,
    month=Month.DECEMBER,
    hour_utc=15.0,
    observation=MsiSensorConfig(
        zenith=3.0,
        bands=["2", "3", "4", "8", "11", "12"]
    ),
    spp=8
)

# Run HYPSTAR simulation
hypstar_output = mtr_demo_simulation(
    scene_description_path=scene_path,
    month=Month.DECEMBER,
    hour_utc=15.0,
    observation=HypstarObservation(fov=5.0),
    spp=8
)
```

**Note**: In Airflow, these simulations would run in parallel as independent tasks sharing the same scene.

---

## CLI Usage

Both processors are registered with the OGC-API registry and can be invoked via CLI:

```bash
# Generation
s2gos_apps mtr_demo/generation \
    --month december \
    --hour_utc 15.0 \
    --random_seed 42

# Simulation
s2gos_apps mtr_demo/simulation \
    --scene_description_path ./gen_output/pnp_mtr_demo_december_15utc/scene_description.yml \
    --month december \
    --hour_utc 15.0 \
    --observation '{"type": "rgb_camera", "distance_m": 100, "azimuth": 45, "elevation_angle": 10}' \
    --spp 8
```

**Note**: The `observation` parameter must be a JSON object matching one of the observation config schemas.

---

## Output Structure

### Generation Output

```
gen_config/
  └── pnp_mtr_demo_<month>_<hour>utc_gen_config.json

gen_output/
  └── pnp_mtr_demo_<month>_<hour>utc/
      ├── scene_description.yml
      ├── landcover_*.tif
      ├── dem_*.tif
      └── ... (other scene data)
```

### Simulation Output

```
sim_config/
  └── mtr_demo_<observation_type>_sim_config.json

sim_output/
  └── pnp_mtr_demo_<month>_<hour>utc/
      ├── <sensor_id>_*.nc  (netCDF outputs)
      ├── <sensor_id>_*.png (visualizations)
      └── ... (sensor-specific outputs)
```

---

## Design Notes

### Time Zone Handling

The `hour_utc` parameter is already in UTC (per user clarification). No timezone conversion is performed.

- **December**: Local time is UTC-3 (summer)
  - 12:00 local → 15:00 UTC
- **June**: Local time is UTC-4 (winter)
  - 12:00 local → 16:00 UTC

Users must provide the correct UTC hour for their desired local time.

### Observation Dates

Observation dates are fixed to the 21st of the month (solstice dates):
- December: 2024-12-21 (summer solstice in Southern Hemisphere)
- June: 2024-06-21 (winter solstice in Southern Hemisphere)

The hour is taken from the `hour_utc` parameter.

### Tower Location

The tower is always included in generated scenes at:
- **Scene coordinates**: (-220m, 850m)
- **XML asset**: `only_tower_v0_1.xml`
- **Elevation offset**: 0.1m above terrain

The TLS patch parameter was removed per user feedback - the tower is always present.

### Material Paths

Material configuration files are hardcoded to:
```
/home/gonzalezm/test/test2/s2gos-apps/packages/s2gos-generator/resources/data/
```

Thermophysical properties are hardcoded to:
```
/home/gonzalezm/test/test2/s2gos-apps/example/PNP/
```

These paths may need to be adjusted for different deployment environments.

---

## Integration with Airflow

For Airflow integration, the processors can be wrapped in Airflow tasks:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from s2gos_apps.processes.mtr_demo import (
    Month, mtr_demo_generation, mtr_demo_simulation,
    ChimeSensorConfig, MsiSensorConfig, HypstarObservation
)

# Generation task
generation_task = PythonOperator(
    task_id="generate_scene",
    python_callable=mtr_demo_generation,
    op_kwargs={
        "month": Month.DECEMBER,
        "hour_utc": 15.0,
        "random_seed": 42
    }
)

# Parallel simulation tasks
chime_task = PythonOperator(
    task_id="simulate_chime",
    python_callable=mtr_demo_simulation,
    op_kwargs={
        "scene_description_path": "{{ task_instance.xcom_pull(task_ids='generate_scene') }}",
        "month": Month.DECEMBER,
        "hour_utc": 15.0,
        "observation": ChimeSensorConfig(zenith=3.0),
        "spp": 8
    }
)

msi_task = PythonOperator(...)
hypstar_task = PythonOperator(...)

# Dependencies
generation_task >> [chime_task, msi_task, hypstar_task]
```

---

## Future Enhancements

### Satellite HDRF Implementation

The `SatelliteHdrfConfig` observation type requires implementation of:

1. **Pixel coordinate calculation**: Convert tower scene coordinates to pixel indices
2. **Grid generation**: Create 3x3 grid around tower pixel
3. **Multi-pixel HDRF**: Create HDRF measurements for each pixel or aggregated measurement

Suggested approach:
```python
def _calculate_tower_pixel(film_resolution, target_size_km):
    """Calculate pixel coordinates of tower in satellite image."""
    # Tower at (-220m, 850m) in scene coords
    # Scene center is at (0, 0) → pixel (film_res/2, film_res/2)
    pixel_x = film_resolution[0] / 2 + (TOWER_COORDS[0] / (target_size_km * 1000)) * film_resolution[0]
    pixel_y = film_resolution[1] / 2 + (TOWER_COORDS[1] / (target_size_km * 1000)) * film_resolution[1]
    return (int(pixel_x), int(pixel_y))

def _generate_pixel_grid(center_pixel, grid_size=3):
    """Generate NxN grid around center pixel."""
    pixels = []
    offset = grid_size // 2
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            pixels.append((center_pixel[0] + i, center_pixel[1] + j))
    return pixels
```

### Additional Enhancements

- **Dynamic material paths**: Use environment variables or config file
- **Additional observation types**: BRF, albedo, directional-hemispherical reflectance
- **Custom vegetation densities**: Allow user override of default densities
- **DEM/landcover selection**: Support for different data sources
- **Validation outputs**: Add scene visualization, statistics, validation plots

---

## Troubleshooting

### Common Issues

**Issue**: "Configuration errors: ..."
- Check that material config files exist at the specified paths
- Verify thermophysical property files are available
- Ensure tree XML assets are present in the resources directory

**Issue**: "Eradiate not available"
- Install Eradiate backend: `pip install eradiate`
- Check Eradiate environment setup

**Issue**: "Material validation found issues"
- Scene generation may reference materials not in the material config
- Check material_config.json for missing material definitions
- Verify XML scenes use valid material IDs

**Issue**: Simulation runs but produces no output
- Check `spp` value - very low values may produce noisy results
- Verify observation parameters are physically valid
- Check Eradiate backend mode (ckd vs mono)

---

## References

- **PNP MTR Processors**: `src/s2gos_apps/processes/pnp_mtr.py`
- **Simulation Utilities**: `src/s2gos_apps/sim_util_mtr.py`
- **Generation Utilities**: `src/s2gos_apps/gen_util.py`
- **Upscaling Pattern**: `src/s2gos_apps/processes/upscaling.py`
- **Example Script**: `example/mtr_demo_example.py`
