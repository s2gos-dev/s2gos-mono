
**Script:** [`examples/run_hypstar_simulations.py`](../../../../examples/run_hypstar_simulations.py)

This example runs a series of HYPSTAR ground sensor simulations over the Gobabeb validation
site, driven by a real HYPERNETS L2A NetCDF dataset. Each series in the dataset corresponds
to a different acquisition geometry; the script loops through them and produces a simulated
HCRF output for each, then merges all results into a single HYPERNETS-format NetCDF file for
direct comparison against real measurements.

## What it demonstrates

- **Material regions** — a 1.5 km × 1.5 km centre patch of the target area is overridden
  with a spectrally-rich RPV BRDF fit from field measurements, while the surrounding surface
  retains the default landcover-based materials.
- **HAMSTER albedo map** — measured bare-soil albedo from the HAMSTER network replaces the
  default material for bare-soil pixels across the full target domain.
- **User assets** — two XML scene objects (a mast structure and a fence) are imported and
  positioned at their real-world coordinates within the scene.
- **Heterogeneous atmosphere** — a molecular layer (with MYCENA absorption database) is
  combined with a single aerosol particle layer whose optical depth and height are derived
  from CAMS climatology.
- **Geometry-driven loop** — viewing and solar angles are read directly from the HYPERNETS
  L2A file for each acquisition, converted to the Eradiate convention, and used to configure
  a fresh `SimulationConfig` per series.

## Key configuration points

### Material region (RPV patch)

```python
config.region_material_defs["gobabeb_measured_rpv"] = {
    "type": "rpv",
    "rho_0": {"path": resolved_paths["rpv"], "variable": "rho_0"},
    ...
}
config.material_regions.append(
    MaterialRegion(
        region_id="center_rpv",
        geometry=RectangleGeometry(center_x=0.0, center_y=0.0,
                                   width_m=1500.0, height_m=1500.0).model_dump(),
        material_name="gobabeb_measured_rpv",
        priority=10,
        applies_to=["target"],
    )
)
```

### HYPSTAR sensor (via factory)

```python
create_hypstar_sensor(
    viewing=AngularFromOriginViewing(...),
    fov=5.0,
    resolution=(32, 32),
    reference_file=l2a_path,   # output wavelengths matched to real HYPSTAR grid
    sensor_id=sensor_id,
    samples_per_pixel=SENSOR_SAMPLES,
)
```

`create_hypstar_sensor` pre-configures the Gaussian SRF (VNIR 3 nm / SWIR 10 nm FWHM),
circular-mask post-processing, and spatial averaging. Passing `reference_file` aligns the
output wavelength grid to that of the real L2A dataset for direct comparison.

### Angle conversion (HYPSTAR → Eradiate)

```python
"vza": 180.0 - vza_hyp,
"vaa": (90.0 - vaa_hyp) % 360.0,
```

## Running the script

Edit the `USER CONFIGURATION` block at the top of the script to point to your data,
then run from the repo root:

```bash
pixi run python examples/run_hypstar_simulations.py
```
