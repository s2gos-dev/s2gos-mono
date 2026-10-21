
**Script:** [`examples/run_hypstar_simulations.py`](https://github.com/s2gos-dev/s2gos-mono/blob/main/examples/run_hypstar_simulations.py)

This example simulates a real HYPERNETS sequence over the Gobabeb validation site. It reads
the sequence's L2B reflectance and L1D radiance/irradiance files, simulates every irradiance
scan and the selected radiance series, derives the HCRF the way the HYPERNETS land processor
does, and merges the results into a single HYPERNETS-format NetCDF file for direct comparison
against the real L2B.

## What it demonstrates

- **Material regions** — a 1.5 km × 1.5 km centre patch of the target area is overridden
  with an RPV BRDF fit from field measurements, while the surrounding surface
  retains the default landcover-based materials.
- **HAMSTER albedo map** — measured bare-soil albedo from the HAMSTER network replaces the
  default material for bare-soil pixels across the full target domain.
- **User assets** — two XML scene objects (a mast structure and a fence) are imported and
  positioned at their real-world coordinates within the scene.
- **Heterogeneous atmosphere** — a molecular layer (PANELLUS absorption database) combined
  with a single aerosol layer. AOD550 comes from CAMS or AERONET (`AOD_SOURCE`), the layer
  height from the Kinne climatology, and the aerosol model is Govaerts desert or custom
  dataset (`AEROSOL_MODEL`). With `SCALE_H2O_TO_AERONET`, the H2O profile is scaled to
  the AERONET precipitable water (O3 stays as in EAC4).
- **Geometry-driven loop** — viewing and solar angles are read directly from the HYPERNETS
  files for each scan, converted to the Eradiate convention, and used to configure
  a fresh [`SimulationConfig`][s2gos_simulator.config.simulation.SimulationConfig] per scan.

## Key configuration points

### Material region (RPV patch)

A [`MaterialRegion`][s2gos_generator.core.config.assets.MaterialRegion] overrides the default material for a spatial sub-region of the scene.

```python
config.region_material_defs["gobabeb_measured_rpv"] = {
    "type": "rpv",
    "rho_0": {"path": resolved_paths["rpv"], "variable": "rho_0"},
    ...
}
config.material_regions.append(
    MaterialRegion(
        region_id="center_rpv",
        geometry=RectangleGeometry(center=(0.0, 0.0), coord_type="scene",
                                   width_m=1500.0, height_m=1500.0).model_dump(),
        material_name="gobabeb_measured_rpv",
        priority=10,
        applies_to=["target"],
    )
)
```

### HYPSTAR sensor (via factory)

[`create_hypstar_sensor`][s2gos_simulator.config.sensors.create_hypstar_sensor] accepts an [`AngularFromOriginViewing`][s2gos_simulator.config.viewing.AngularFromOriginViewing] geometry and returns a fully configured ground sensor.

```python
create_hypstar_sensor(
    viewing=AngularFromOriginViewing(...),
    fov=5.0,
    resolution=(128, 128),
    reference_file=ref_path,   # output wavelengths matched to real HYPSTAR grid
    sensor_id=sensor_id,
    samples_per_pixel=SENSOR_SAMPLES,
)
```

[`create_hypstar_sensor`][s2gos_simulator.config.sensors.create_hypstar_sensor] pre-configures the Gaussian SRF (VNIR 3 nm / SWIR 10 nm FWHM),
circular-mask post-processing, and spatial averaging. Passing `reference_file` aligns the
output wavelength grid to that of the real L2B dataset for direct comparison.

### Angle conversion (HYPSTAR → Eradiate)

```python
"vza": 180.0 - vza_hyp,
"vaa": (90.0 - vaa_hyp) % 360.0,
```

## Running the script

Edit the `USER CONFIGURATION` block at the top of the script (sequence, date, AOD source,
aerosol model). All input files are found through the resolver search paths
(e.g. `extra_data/`). Then run from the repo root:

```bash
pixi run python examples/run_hypstar_simulations.py
```
