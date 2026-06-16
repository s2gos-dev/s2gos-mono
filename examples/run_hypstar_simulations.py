"""
Run HYPSTAR simulations from real L2A dataset.

Workflow:
1. Load HYPSTAR L2A NetCDF data.
2. Generate the 3D Scene (Atmosphere + Surface + XML objects).
3. Loop through dataset series: configure geometry, run sim, save output.
"""

import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from s2gos_generator import create_scene_config
from s2gos_generator.core import SceneGenerationPipeline
from s2gos_generator.core.config import (
    AbsorptionDatabase,
    BackgroundConfig,
    BufferConfig,
    ExponentialDistribution,
    MaterialRegion,
    MolecularAtmosphereConfig,
    ParticleLayerConfig,
    ThermophysicalConfig,
    XmlSceneConfig,
)
from s2gos_generator.core.region_geometry import RectangleGeometry
from s2gos_simulator.backends.eradiate.backend import (
    ERADIATE_AVAILABLE,
    EradiateBackend,
)
from s2gos_simulator.config import (
    AngularFromOriginViewing,
    DirectionalIllumination,
    HCRFConfig,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    SimulationConfig,
    SpectralResponse,
    create_hypstar_sensor,
)

# S2GOS / Eradiate Imports
from s2gos_utils.io.paths import to_upath
from s2gos_utils.io.resolver import resolver
from upath import UPath

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

# Input / Output Paths (resolved via s2gos_settings.toml -> ./hypstar_data/)
HYPSTAR_L2A_PATH = "HYPERNETS_L_GHNA_L2A_REF_20220517T0743_20230424T0625_v1.0.nc"
OUTPUT_DIR = Path("./hypstar_simulation_output")
SIM_DIR = OUTPUT_DIR / "simulations"
SCENE_NAME = "hypstar_gobabeb"

# Data Dependencies
PATHS = {
    "hamster": "DOY196_Gobabeb.nc",
    "thermo": "timeseries_ms_2022-05-02_v1.nc",
    "aerosol": "D5_aerosol_model_v5_gobabeb_ert.nc",
    "rpv": "RPV_gobabeb.nc",
    "cams": "Gobabeb.nc",
    "kinne": "altitude_t.nc",
    "mast": "hypernets_mast_better.xml",
    "fence": "gobabeb_fence_custom.xml",
}


# Settings
SERIES_INDICES = [1]  # List of indices or None for all
TARGET_COORDS = (-23.6015417, 15.1258696)  # (Lat, Lon)
SENSOR_ORIGIN = [0.022914, 1.16748, 10.5269]
IRR_HEIGHT = 13.0
SENSOR_SAMPLES = 4
IRR_SAMPLES = 4


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================


def get_geometry_and_time(ds: xr.Dataset, idx: int) -> dict:
    """Extracts timestamp and converts HYPSTAR angles to Eradiate convention.

    Returns both original HYPSTAR angles (for metadata) and Eradiate-converted
    angles (for simulation configuration).
    """
    vza_hyp = float(ds.viewing_zenith_angle.values[idx])
    vaa_hyp = float(ds.viewing_azimuth_angle.values[idx])
    sza_hyp = float(ds.solar_zenith_angle.values[idx])
    saa_hyp = float(ds.solar_azimuth_angle.values[idx])
    ts_unix = int(ds.acquisition_time.values[idx])

    return {
        # Original HYPSTAR angles (for output metadata)
        "vza_hypstar": vza_hyp,
        "vaa_hypstar": vaa_hyp,
        "sza_hypstar": sza_hyp,
        "saa_hypstar": saa_hyp,
        # Eradiate-converted angles (for simulation config)
        "vza": 180.0 - vza_hyp,
        "vaa": (90.0 - vaa_hyp) % 360.0,
        "sza": sza_hyp,
        "saa": (90.0 - saa_hyp) % 360.0,
        "dt": datetime.fromtimestamp(ts_unix, timezone.utc),
    }


def get_atmosphere_params(
    timestamp_dt: datetime, resolved_paths: dict
) -> tuple[float, float]:
    """Retrieves AOD and aerosol height from climatology files."""
    gobabeb_ds = xr.open_dataset(resolved_paths["cams"])
    kinne_ds = xr.open_dataset(resolved_paths["kinne"])

    ts_naive = timestamp_dt.replace(tzinfo=None)
    aod = float(
        gobabeb_ds.aod550.dropna(dim="time").sel(time=ts_naive, method="nearest").values
    )

    month = timestamp_dt.month - 1
    kinne_loc = kinne_ds.sel(
        lat=TARGET_COORDS[0], lon=TARGET_COORDS[1], method="nearest"
    ).isel(time=month)

    aerosol_height = 2000.0  # Fallback
    for i in range(len(kinne_loc.lay)):
        if float(kinne_loc.AODt_frac[: i + 1].sum()) >= 0.92:
            aerosol_height = float(kinne_loc.Zl_top.isel(lay=i).values)
            break

    return aod, aerosol_height


def load_hcrf_zarr(zarr_path: Path, geo: dict, series_id: int) -> xr.Dataset:
    """Load single HCRF Zarr file and add HYPERNETS-compatible coordinates.

    Args:
        zarr_path: Path to the Zarr directory
        idx: Series index from reference dataset
        geo: Geometry dictionary with viewing/solar angles and timestamp
        series_id: Actual series_id from reference dataset

    Returns:
        Dataset with reflectance and coordinate variables using original HYPSTAR conventions
    """
    ds = xr.open_zarr(zarr_path)

    if "x_index" in ds.dims or "y_index" in ds.dims:
        spatial_dims = [d for d in ["x_index", "y_index"] if d in ds.dims]
        ds = ds.mean(dim=spatial_dims)

    ds = ds.rename({"w": "wavelength", "hcrf": "reflectance"})

    # Use original HYPSTAR angles (not Eradiate-converted) for output compatibility
    ds = ds.assign_coords(
        {
            "viewing_azimuth_angle": geo["vaa_hypstar"],
            "viewing_zenith_angle": geo["vza_hypstar"],
            "solar_azimuth_angle": geo["saa_hypstar"],
            "solar_zenith_angle": geo["sza_hypstar"],
            "acquisition_time": geo["dt"].timestamp(),
            "series_id": series_id,
        }
    )

    return ds


def combine_hcrf_results(
    output_dir: Path, indices: list, ds_ref: xr.Dataset
) -> xr.Dataset:
    """Combine all HCRF Zarr files into single HYPERNETS-format NetCDF.

    Args:
        output_dir: Directory containing copied Zarr files
        indices: List of series indices to include
        ds_ref: Reference HYPSTAR dataset (for series_id mapping)

    Returns:
        Combined dataset with all series
    """
    import numpy as np

    datasets = []

    for idx in indices:
        series_id = int(ds_ref.series_id.values[idx])
        geo = get_geometry_and_time(ds_ref, idx)

        zarr_path = output_dir / f"hcrf_series_{idx:02d}.zarr"
        if not zarr_path.exists():
            print(
                f"  Warning: Missing Zarr file for series {idx} (series_id={series_id})"
            )
            continue

        ds = load_hcrf_zarr(zarr_path, geo, series_id)
        datasets.append(ds)

    if not datasets:
        raise ValueError("No HCRF files found to combine")

    combined = xr.concat(datasets, dim="series")

    combined["bandwidth"] = xr.DataArray(
        np.zeros(len(combined.wavelength)),
        dims=["wavelength"],
        coords={"wavelength": combined.wavelength},
    )

    combined.attrs = {
        "title": "S2GOS simulated HYPSTAR observations",
        "simulator": "S2GOS (Synthetic Scene Generation and Observation Simulation)",
        "description": "Simulated HCRF measurements matching HYPERNETS protocol",
        "created": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": Path(HYPSTAR_L2A_PATH).name,
    }

    return combined


# ==============================================================================
# 3. SCENE & SIMULATION SETUP
# ==============================================================================


def build_scene_config(resolved_paths: dict) -> object:
    """Constructs the scene gen configuration."""
    config = create_scene_config(
        scene_name=SCENE_NAME,
        center_lat=TARGET_COORDS[0],
        center_lon=TARGET_COORDS[1],
        aoi_size_km=10.0,
        output_dir=UPath(OUTPUT_DIR) / "scene",
        description="HYPSTAR validation scene",
    )

    config.buffer = BufferConfig(size_km=60.0, resolution_m=60.0)
    config.background = BackgroundConfig(size_km=100.0, resolution_m=200.0)

    config.enable_hamster_albedo(
        resolved_paths["hamster"], "albedo", fallback_on_error=True
    )

    config.region_material_defs["gobabeb_measured_rpv"] = {
        "type": "rpv",
        "rho_0": {"path": resolved_paths["rpv"], "variable": "rho_0"},
        "k": {"path": resolved_paths["rpv"], "variable": "k"},
        "Theta": {"path": resolved_paths["rpv"], "variable": "Theta"},
        "rho_c": {"path": resolved_paths["rpv"], "variable": "rho_c"},
    }
    config.material_regions.append(
        MaterialRegion(
            region_id="center_rpv",
            geometry=RectangleGeometry(
                center=(0.0, 0.0), coord_type="scene", width_m=1500.0, height_m=1500.0
            ).model_dump(),
            material_name="gobabeb_measured_rpv",
            priority=10,
            applies_to=["target"],
        )
    )

    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path=resolved_paths["mast"],
            base_coordinate=(TARGET_COORDS[1], TARGET_COORDS[0]),
            coord_type="geographic",
            elevation_offset=-0.1,
        )
    )
    config.xml_scenes.append(
        XmlSceneConfig(
            xml_path=resolved_paths["fence"],
            base_coordinate=(15.1253501, -23.6011482),
            coord_type="geographic",
        )
    )

    # Atmosphere (Fixed reference time)
    atm_time = datetime(2022, 5, 17, 9, 45, 4, tzinfo=timezone.utc)
    aod, aer_h = get_atmosphere_params(atm_time, resolved_paths)
    print(f"  Atmosphere Params: AOD={aod:.3f}, Height={aer_h:.1f}m")

    config.set_atmosphere_heterogeneous(
        MolecularAtmosphereConfig(
            thermoprops=ThermophysicalConfig(
                identifier=None, thermoprops_file=UPath(resolved_paths["thermo"])
            ),
            absorption_database=AbsorptionDatabase.MYCENA,
            has_absorption=True,
            has_scattering=True,
        ),
        [
            ParticleLayerConfig(
                aerosol_dataset=resolved_paths["aerosol"],
                optical_thickness=aod,
                altitude_bottom=500.0,
                altitude_top=500.0 + aer_h,
                distribution=ExponentialDistribution(rate=5.0),
                has_absorption=True,
            )
        ],
    )

    return config


def build_sim_config(idx: int, geo: dict, l2a_path: str) -> SimulationConfig:
    """Creates the simulation config for a specific time/geometry."""
    sensor_id = f"hypstar_series_{idx:02d}"
    irradiance_id = f"irradiance_series_{idx:02d}"
    hcrf_id = f"hcrf_series_{idx:02d}"

    return SimulationConfig(
        name=f"hypstar_series_{idx:02d}",
        description=f"Series {idx} at {geo['dt']}",
        # Sun Position
        illumination=DirectionalIllumination(
            id=f"sun_series_{idx:02d}",
            zenith=geo["sza"],
            azimuth=geo["saa"],
            irradiance_dataset="coddington_2022-1_nm",
        ),
        # Sensors
        sensors=[
            create_hypstar_sensor(
                viewing=AngularFromOriginViewing(
                    origin=SENSOR_ORIGIN,
                    zenith=geo["vza"],
                    azimuth=geo["vaa"],
                    up=[0, 0, 1] if geo["vza"] != 180 else [0, 1, 0],
                    # terrain_relative_height=True,
                    relative_to_asset="hypernets_mast_better.xml",
                ),
                fov=5.0,
                resolution=(128, 128),
                reference_file=l2a_path,
                sensor_id=sensor_id,
                samples_per_pixel=SENSOR_SAMPLES,
            ),
        ],
        # Measurements (Irradiance + HCRF)
        measurements=[
            IrradianceConfig(
                id=irradiance_id,
                location=HemisphericalMeasurementLocation(
                    target_lat=TARGET_COORDS[0],
                    target_lon=TARGET_COORDS[1],
                    height_offset_m=IRR_HEIGHT,
                    srf=SpectralResponse(type="uniform", wmin=375, wmax=1685),
                    samples_per_pixel=IRR_SAMPLES,
                ),
            ),
            HCRFConfig(
                id=hcrf_id,
                radiance_sensor_id=sensor_id,
                irradiance_measurement_id=irradiance_id,
            ),
        ],
        backend_hints={"eradiate": {"mode": "ckd"}},
    )


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================


def main():
    if not ERADIATE_AVAILABLE:
        raise RuntimeError("Eradiate is missing.")

    # Resolve paths using s2gos_settings.toml search_paths
    l2a_path = str(resolver.resolve(HYPSTAR_L2A_PATH, strict=True))
    resolved_paths = {
        k: str(resolver.resolve(v, strict=True)) for k, v in PATHS.items()
    }

    print(f"{resolved_paths = }")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== HYPSTAR Simulator | Output: {OUTPUT_DIR} ===")

    # 1. Load Data
    ds = xr.load_dataset(l2a_path)
    indices = SERIES_INDICES if SERIES_INDICES else list(range(len(ds.series)))
    print(f"[1/5] Loaded dataset. Processing {len(indices)} series.")

    # 2. Generate Scene
    print("\n[2/5] Generating Scene...")
    scene_config = build_scene_config(resolved_paths)
    pipeline = SceneGenerationPipeline(scene_config)
    scene_desc = pipeline.run()
    scene_config.to_json(OUTPUT_DIR / "scene_config.json")

    # 3. Run Simulations
    print("\n[3/5] Running Simulations...")
    for i, idx in enumerate(indices):
        geo = get_geometry_and_time(ds, idx)
        print(
            f"  > Series {idx} ({i + 1}/{len(indices)}) | {geo['dt']} | SZA:{geo['sza']:.1f} VZA:{geo['vza']:.1f}"
        )

        try:
            sim_config = build_sim_config(idx, geo, l2a_path)
            sim_config.to_json(SIM_DIR / f"config_{idx:02d}.json")

            sim_render_dir = SIM_DIR / f"series_{idx:02d}"
            backend = EradiateBackend(sim_config)
            backend.run_simulation(
                scene_desc,
                to_upath(scene_config.scene_output_dir),
                output_dir=sim_render_dir,
                plot_image=False,
            )

            src = (
                sim_render_dir
                / "derived"
                / f"hypstar_series_{idx:02d}_hcrf_series_{idx:02d}.zarr"
            )
            dst = SIM_DIR / f"hcrf_series_{idx:02d}.zarr"
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"Copied to: {dst.name}")
            else:
                print("Error: Output file not created.")

        except Exception:
            print(f"Failed: {traceback.format_exc().splitlines()[-1]}")

    # 4. Combine Results into Single NetCDF
    print("\n[4/5] Combining results into HYPERNETS format...")
    try:
        combined_ds = combine_hcrf_results(SIM_DIR, indices, ds)
        output_nc = SIM_DIR / f"{SCENE_NAME}_combined_hcrf.nc"
        combined_ds.to_netcdf(output_nc, mode="w")
        print(f"  Saved combined NetCDF: {output_nc.name}")
        print(f"    - Wavelengths: {len(combined_ds.wavelength)}")
        print(f"    - Series: {len(combined_ds.series)}")
    except Exception as e:
        print(f"  Failed to combine results: {e}")

    print("\n[5/5] Finished.")


if __name__ == "__main__":
    main()
