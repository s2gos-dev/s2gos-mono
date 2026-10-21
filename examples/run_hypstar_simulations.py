"""
Simulate a HYPERNETS (HYPSTAR) sequence over Gobabeb and derive its HCRF.

The HCRF reproduces the HYPERNETS land processor. There, the L2B reflectance is
pi L / E, with E the L1D irradiance scans of the sequence (taken at its start and
end) interpolated to each radiance scan. Here every usable irradiance scan is
simulated at its own sun position, through the radiance's Gaussian SRF centred on
the irradiance file's wavelengths, then goes through the same interpolation
(see ``usable_irradiance_scans`` and ``interpolate_irradiance``).

The atmosphere is built once, at the time of the first selected series. AOD550
comes from CAMS or AERONET, and the H2O profile can be scaled to the AERONET
precipitable water.

Workflow:
1. Load the sequence's L2B reflectance and L1D irradiance files.
2. Generate the 3D Scene (Atmosphere + Surface + XML objects).
3. Simulate every irradiance scan of the sequence.
4. Loop through the selected series: simulate radiance, derive HCRF, save.
5. Combine the series into one HYPERNETS-format NetCDF.

All file names are resolved via the s2gos resolver search paths.
"""

import traceback
from datetime import datetime, timezone
from pathlib import Path

# S2GOS / Eradiate Imports
import numpy as np
import pandas as pd
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
    HYPSTAR_SPECTRAL_REGIONS,
    HYPSTAR_SRF,
    AngularFromOriginViewing,
    DirectionalIllumination,
    HemisphericalMeasurementLocation,
    IrradianceConfig,
    SimulationConfig,
    SpectralResponse,
    WavelengthGrid,
    create_hypstar_sensor,
)
from s2gos_utils.io.paths import to_upath
from s2gos_utils.io.resolver import resolver
from scipy.interpolate import interp1d
from upath import UPath

if not hasattr(
    np, "trapezoid"
):  # Eradiate 1.2 uses np.trapezoid; numpy<2 exposes it as trapz
    np.trapezoid = np.trapz

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================

# Sequence date: selects the CAMS and thermo files and names the output dir
DATE = "2025-11-05"
# HYPERNETS sequence folder holding the *_L2B_REF_*, *_L1D_RAD_* and *_L1D_IRR_* files
HYPSTAR_SEQUENCE = "GHNA_L1DL2B_Nov2025/05/SEQ20251105T123143"
# AOD550 source: "cams" (PATHS["cams"]) or "aeronet" (PATHS["aeronet"])
AOD_SOURCE = "aeronet"
# Scale the thermo H2O profile to the AERONET precipitable water
SCALE_H2O_TO_AERONET = True
# Aerosol model: "govaerts" (Eradiate's govaerts_2021-desert-extrapolated) or "d5" (PATHS["aerosol"])
AEROSOL_MODEL = "govaerts"

OUTPUT_DIR = Path(
    f"./hypstar_simulation_output_{DATE.replace('-', '')}_{AOD_SOURCE}_{AEROSOL_MODEL}"
    + ("_h2o_aeronet" if SCALE_H2O_TO_AERONET else "")
)
SIM_DIR = OUTPUT_DIR / "simulations"
SCENE_NAME = "hypstar_gobabeb"

# Data Dependencies
PATHS = {
    "hamster": "DOY196_Gobabeb.nc",
    "thermo": f"timeseries_ms_{DATE}_v1.nc",
    "aerosol": "D5_aerosol_model_v5_gobabeb_ert.nc",
    "rpv": "RPV_gobabeb.nc",
    "cams": f"Gobabeb_cams_{DATE}.nc",
    "aeronet": "20140101_20261231_Gobabeb.lev15",
    "kinne": "altitude_t.nc",
    "mast": "hypernets_mast_better.xml",
    "fence": "gobabeb_fence_custom.xml",
}


# Settings
SERIES_INDICES = [17]  # List of indices or None for all
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


def interpolate_irradiance(
    irradiance: np.ndarray,
    irr_wl: np.ndarray,
    irr_time: np.ndarray,
    irr_sza: np.ndarray,
    rad_wl: np.ndarray,
    rad_time: float,
    rad_sza: float,
) -> np.ndarray:
    """Irradiance (wavelength, scan) at one radiance scan, as in HYPERNETS land L1C.

    Reproduces ``Interpolate.interpolate_irradiance`` of hypernets_processor with
    the land defaults ``InterpolationWavLinear`` and
    ``InterpolationTimeLinearCoscorrected``:

    1. linear in wavelength onto the radiance grid, extrapolating at the edges;
    2. E / cos(SZA) linear in time, clamped to the first/last scan (a single
       scan is used as is);
    3. rescaled by cos(SZA) of the radiance scan.

    The L2B reflectance is pi L / E with this E (``LandNetworkProtocol``), so the
    simulated HCRF uses the same estimator instead of an irradiance simulated at
    the radiance scan's own time.
    """
    E = interp1d(irr_wl, irradiance, axis=0, fill_value="extrapolate")(rad_wl)
    E = E / np.cos(np.radians(irr_sza))
    order = np.argsort(irr_time)
    E_t = np.array([np.interp(rad_time, irr_time[order], e[order]) for e in E])
    return E_t * np.cos(np.radians(rad_sza))


def usable_irradiance_scans(irr: xr.Dataset) -> xr.Dataset:
    """Irradiance scans HYPERNETS interpolates, as in ``Interpolate.interpolate_irradiance``.

    Scans flagged ``vza_irradiance``, ``not_enough_dark_scans`` or
    ``not_enough_irr_scans`` are dropped.
    """
    flags = irr.quality_flag.attrs["flag_meanings"].split()
    bad = sum(
        2 ** flags.index(f)
        for f in ("vza_irradiance", "not_enough_dark_scans", "not_enough_irr_scans")
    )
    return irr.isel(series=(irr.quality_flag.values.astype(int) & bad) == 0)


def read_aeronet(
    path: str, columns: list, timestamp_dt: datetime
) -> tuple[pd.DataFrame, pd.Series]:
    """Records of an AERONET V3 export (.lev15/.lev20) with all ``columns`` valid.

    Returns the records and their time offset (s) from ``timestamp_dt``, and
    asserts that one record lies within 30 min of it.
    """
    df = pd.read_csv(
        path,
        skiprows=6,
        na_values=[-999],
        usecols=["Date(dd:mm:yyyy)", "Time(hh:mm:ss)", *columns],
    ).dropna()
    t = pd.to_datetime(
        df.iloc[:, 0] + " " + df.iloc[:, 1], format="%d:%m:%Y %H:%M:%S", utc=True
    )
    dt_s = (t - pd.Timestamp(timestamp_dt)).dt.total_seconds()
    assert dt_s.abs().min() < 1800, (
        f"no AERONET record within 30 min of {timestamp_dt} "
        f"(nearest {dt_s.abs().min() / 3600:.1f} h away)"
    )
    return df, dt_s


def get_aeronet_aod550(path: str, timestamp_dt: datetime) -> float:
    """AOD at 550 nm from an AERONET export at ``timestamp_dt``.

    Each record is taken to 550 nm with the Angstrom law between 500 and 675 nm,
    then interpolated linearly in time between the records bracketing ``timestamp_dt``.
    """
    df, dt_s = read_aeronet(path, ["AOD_500nm", "AOD_675nm"], timestamp_dt)
    alpha = np.log(df.AOD_675nm / df.AOD_500nm) / np.log(675 / 500)
    aod550 = df.AOD_500nm * (550 / 500) ** alpha
    return float(np.interp(0.0, dt_s, aod550))


def scale_h2o_to_aeronet(
    thermo_path: str, aeronet_path: str, timestamp_dt: datetime, out_dir: Path
) -> str:
    """Write a copy of ``thermo_path`` with x_H2O scaled to the AERONET precipitable water.

    The precipitable water is interpolated linearly in time at ``timestamp_dt``.
    The profile is scaled by one factor, so its shape is kept, and the column is
    matched above the AERONET site elevation. O3 is left as EAC4 gives it, since
    AERONET's ozone is a climatology, not a measurement.
    """
    df, dt_s = read_aeronet(
        aeronet_path, ["Precipitable_Water(cm)", "Site_Elevation(m)"], timestamp_dt
    )
    z0_km = float(df["Site_Elevation(m)"].iloc[0]) / 1000
    target = 10 * np.interp(0.0, dt_s, df["Precipitable_Water(cm)"])  # kg m^-2

    ds = xr.load_dataset(thermo_path)

    def column():
        """H2O column (kg m^-2) above the site."""
        z = ds.z.values
        nx = (ds.n * ds.x_H2O).values.squeeze()
        zz = np.r_[z0_km, z[z > z0_km]]
        return np.trapezoid(np.interp(zz, z, nx), 1000 * zz) * 18.015e-3 / 6.02214e23

    before = column()
    ds.x_H2O.data *= target / before
    after = column()
    assert abs(after / target - 1) < 1e-6, f"H2O column {after} != {target}"
    print(
        f"  H2O column above {1000 * z0_km:.0f} m: "
        f"{before:.3f} -> {after:.3f} kg/m2 (x{target / before:.3f})"
    )

    out_path = (out_dir / f"{Path(thermo_path).stem}_h2o_aeronet.nc").resolve()
    ds.to_netcdf(out_path)
    return str(out_path)


def get_atmosphere_params(
    timestamp_dt: datetime, resolved_paths: dict
) -> tuple[float, float]:
    """Retrieves AOD550 (from ``AOD_SOURCE``) and aerosol height (Kinne climatology)."""
    kinne_ds = xr.open_dataset(resolved_paths["kinne"])

    if AOD_SOURCE == "aeronet":
        aod = get_aeronet_aod550(resolved_paths["aeronet"], timestamp_dt)
    else:
        cams = xr.open_dataset(resolved_paths["cams"]).aod550.dropna(dim="time")
        ts_naive = timestamp_dt.replace(tzinfo=None)
        aod = float(cams.sel(time=ts_naive, method="nearest").values)

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


def hypernets_series(hcrf_ds: xr.Dataset, geo: dict, series_id: int) -> xr.Dataset:
    """One simulated series in the HYPERNETS L2B layout, with the original HYPSTAR angles."""
    return hcrf_ds.rename({"w": "wavelength", "hcrf": "reflectance"}).assign_coords(
        viewing_azimuth_angle=geo["vaa_hypstar"],
        viewing_zenith_angle=geo["vza_hypstar"],
        solar_azimuth_angle=geo["saa_hypstar"],
        solar_zenith_angle=geo["sza_hypstar"],
        acquisition_time=geo["dt"].timestamp(),
        series_id=series_id,
    )


# ==============================================================================
# 3. SCENE & SIMULATION SETUP
# ==============================================================================


def ensure_aer_core_v2(aerosol_path: str, out_dir: Path) -> str:
    """Ensure the aerosol dataset is in Eradiate 1.2's aer_core_v2 format."""
    ds = xr.open_dataset(aerosol_path)
    if "phamat" in ds.dims:  # already aer_core_v2
        return aerosol_path

    from eradiate.data.convert import aer_v1_to_aer_core_v2

    v2 = aer_v1_to_aer_core_v2(ds)  # defaults: normalize=True, check="full"
    out_path = out_dir / f"{Path(aerosol_path).stem}_v2.nc"
    v2.to_netcdf(out_path)
    print(f"  Converted aerosol dataset to aer_core_v2: {out_path.name}")
    return str(out_path)


def build_scene_config(resolved_paths: dict, atm_time: datetime) -> object:
    """Constructs the scene gen configuration, with the atmosphere at ``atm_time``."""
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
        **{
            p: {"path": resolved_paths["rpv"], "variable": p}
            for p in ("rho_0", "k", "Theta", "rho_c")
        },
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

    aod, aer_h = get_atmosphere_params(atm_time, resolved_paths)
    print(f"  Atmosphere Params: AOD ({AOD_SOURCE})={aod:.4f}, Height={aer_h:.1f}m")

    if AEROSOL_MODEL == "d5":
        aerosol_dataset = ensure_aer_core_v2(resolved_paths["aerosol"], OUTPUT_DIR)
    else:
        aerosol_dataset = "govaerts_2021-desert-extrapolated"
    thermo_path = resolved_paths["thermo"]
    if SCALE_H2O_TO_AERONET:
        thermo_path = scale_h2o_to_aeronet(
            thermo_path, resolved_paths["aeronet"], atm_time, OUTPUT_DIR
        )

    config.set_atmosphere_heterogeneous(
        MolecularAtmosphereConfig(
            thermoprops=ThermophysicalConfig(
                identifier=None, thermoprops_file=UPath(thermo_path)
            ),
            absorption_database=AbsorptionDatabase.PANELLUS,
            has_absorption=True,
            has_scattering=True,
        ),
        [
            ParticleLayerConfig(
                aerosol_dataset=aerosol_dataset,
                optical_thickness=aod,
                altitude_bottom=500.0,
                altitude_top=500.0 + aer_h,
                distribution=ExponentialDistribution(scale=aer_h / 5.0),
                has_absorption=True,
            )
        ],
    )

    return config


def build_irradiance_config(scan: int, geo: dict, irr_path: str) -> SimulationConfig:
    """Creates the simulation config for one irradiance scan of the sequence.

    The irradiance gets the radiance's Gaussian SRF, centred on the irradiance
    scan's own wavelengths. The irradiance head is modelled as an ideal cosine
    collector (white Lambertian disk, E = π·mean(L)).
    """
    srf = SpectralResponse(
        type="gaussian",
        spectral_regions=HYPSTAR_SPECTRAL_REGIONS,
        output_grid=WavelengthGrid(mode="from_file", file_path=irr_path),
    )
    return SimulationConfig(
        name=f"irradiance_scan_{scan:02d}",
        description=f"Irradiance scan {scan} at {geo['dt']}",
        illumination=DirectionalIllumination(
            id=f"sun_irradiance_scan_{scan:02d}",
            zenith=geo["sza"],
            azimuth=geo["saa"],
            irradiance_dataset="coddington_2022-1_nm",
            irradiance_datetime=geo["dt"],
        ),
        measurements=[
            IrradianceConfig(
                id=f"irradiance_scan_{scan:02d}",
                location=HemisphericalMeasurementLocation(
                    target_lat=TARGET_COORDS[0],
                    target_lon=TARGET_COORDS[1],
                    height_offset_m=IRR_HEIGHT,
                    srf=srf,
                    samples_per_pixel=IRR_SAMPLES,
                ),
            )
        ],
        backend_hints={"eradiate": {"mode": "ckd"}},
    )


def build_sim_config(idx: int, geo: dict, ref_path: str) -> SimulationConfig:
    """Creates the radiance simulation config for a specific time/geometry."""
    sensor_id = f"hypstar_series_{idx:02d}"

    sensor = create_hypstar_sensor(
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
        reference_file=ref_path,
        sensor_id=sensor_id,
        samples_per_pixel=SENSOR_SAMPLES,
    )

    return SimulationConfig(
        name=f"hypstar_series_{idx:02d}",
        description=f"Series {idx} at {geo['dt']}",
        illumination=DirectionalIllumination(
            id=f"sun_series_{idx:02d}",
            zenith=geo["sza"],
            azimuth=geo["saa"],
            irradiance_dataset="coddington_2022-1_nm",
            irradiance_datetime=geo["dt"],
        ),
        sensors=[sensor],
        backend_hints={"eradiate": {"mode": "ckd"}},
    )


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================


def main():
    if not ERADIATE_AVAILABLE:
        raise RuntimeError("Eradiate is missing.")

    seq_dir = Path(str(resolver.resolve(HYPSTAR_SEQUENCE, strict=True)))
    ref_path = str(next(seq_dir.glob("*_L2B_REF_*.nc")))
    irr_path = str(next(seq_dir.glob("*_L1D_IRR_*.nc")))
    unused = {
        "aeronet": AOD_SOURCE != "aeronet" and not SCALE_H2O_TO_AERONET,
        "cams": AOD_SOURCE != "cams",
        "aerosol": AEROSOL_MODEL != "d5",
    }
    resolved_paths = {
        k: str(resolver.resolve(v, strict=True))
        for k, v in PATHS.items()
        if not unused.get(k)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== HYPSTAR Simulator | Output: {OUTPUT_DIR} ===")

    # 1. Load Data
    ds = xr.load_dataset(ref_path)
    irr_ds = usable_irradiance_scans(xr.load_dataset(irr_path))
    indices = SERIES_INDICES if SERIES_INDICES else list(range(len(ds.series)))
    print(
        f"[1/5] Loaded sequence: {len(irr_ds.series)} usable irradiance scans, "
        f"{len(indices)} series to process."
    )

    # 2. Generate Scene
    print("\n[2/5] Generating Scene...")
    scene_config = build_scene_config(
        resolved_paths, get_geometry_and_time(ds, indices[0])["dt"]
    )
    pipeline = SceneGenerationPipeline(scene_config)
    scene_desc = pipeline.run()
    scene_config.to_json(OUTPUT_DIR / "scene_config.json")
    scene_dir = to_upath(scene_config.scene_output_dir)

    # 3. Simulate every irradiance scan at its own time and wavelengths
    print("\n[3/5] Simulating irradiance scans...")
    irradiances = []
    for scan in range(len(irr_ds.series)):
        geo = get_geometry_and_time(irr_ds, scan)
        print(f"  > Irradiance scan {scan} | {geo['dt']} | SZA:{geo['sza']:.2f}")
        irr_config = build_irradiance_config(scan, geo, irr_path)
        irr_config.to_json(SIM_DIR / f"config_irradiance_scan_{scan:02d}.json")
        results = EradiateBackend(irr_config).run_simulation(
            scene_desc,
            scene_dir,
            output_dir=SIM_DIR / f"irradiance_scan_{scan:02d}",
            plot_image=False,
        )
        irradiances.append(results[irr_config.measurements[0].id]["boa_irradiance"])
    E_scans = xr.concat(irradiances, dim="series").transpose("w", "series")

    # 4. Simulate radiance per series and derive HCRF
    print("\n[4/5] Running radiance simulations...")
    series = []
    for i, idx in enumerate(indices):
        geo = get_geometry_and_time(ds, idx)
        print(
            f"  > Series {idx} ({i + 1}/{len(indices)}) | {geo['dt']} | SZA:{geo['sza']:.1f} VZA:{geo['vza']:.1f}"
        )

        try:
            sim_config = build_sim_config(idx, geo, ref_path)
            sim_config.to_json(SIM_DIR / f"config_{idx:02d}.json")

            results = EradiateBackend(sim_config).run_simulation(
                scene_desc,
                scene_dir,
                output_dir=SIM_DIR / f"series_{idx:02d}",
                plot_image=False,
            )
            L = results[sim_config.sensors[0].id]["radiance"]

            E = interpolate_irradiance(
                E_scans.values,
                E_scans.w.values,
                irr_ds.acquisition_time.values,
                irr_ds.solar_zenith_angle.values,
                L.w.values,
                float(ds.acquisition_time.values[idx]),
                float(ds.solar_zenith_angle.values[idx]),
            )
            E = xr.DataArray(E, dims="w", coords={"w": L.w.values})
            hcrf_ds = xr.Dataset(
                {"hcrf": np.pi * L / E, "radiance": L, "boa_irradiance": E},
                attrs={
                    "formula": "HCRF = pi * L / E",
                    "irradiance_interpolation": "HYPERNETS land: linear in wavelength, "
                    "cos(SZA)-corrected linear in time",
                },
            )
            dst = SIM_DIR / f"hcrf_series_{idx:02d}.zarr"
            hcrf_ds.to_zarr(dst, mode="w")
            print(f"Saved: {dst.name}")
            series.append(hypernets_series(hcrf_ds, geo, int(ds.series_id.values[idx])))

        except Exception:
            print(f"Failed: {traceback.format_exc().splitlines()[-1]}")

    # 5. Combine Results into Single NetCDF
    print("\n[5/5] Combining results into HYPERNETS format...")
    if not series:
        raise RuntimeError("No series simulated successfully")
    combined = xr.concat(series, dim="series")
    combined["bandwidth"] = (
        "wavelength",
        [HYPSTAR_SRF.get_fwhm_for_wavelength(w) for w in combined.wavelength.values],
        {"long_name": "bandwidth FWHM assuming a Gaussian SRF", "units": "nm"},
    )
    combined.attrs = {
        "title": "S2GOS simulated HYPSTAR observations",
        "simulator": "S2GOS (Synthetic Scene Generation and Observation Simulation)",
        "description": "Simulated HCRF measurements matching HYPERNETS protocol",
        "created": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": Path(ref_path).name,
    }
    output_nc = SIM_DIR / f"{SCENE_NAME}_combined_hcrf.nc"
    combined.to_netcdf(output_nc, mode="w")
    print(f"  Saved {output_nc.name}: {len(combined.series)} series")


if __name__ == "__main__":
    main()
