# Setup on a new machine

Everything needed to run `examples/san_rossore_icos.py` from scratch on Linux or macOS.

## 1. Install pixi

[Pixi](https://pixi.sh) manages the Python environment.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new shell, then check `pixi --version` prints a version.

## 2. Clone the repository

```bash
git clone --branch icos_san_rossore git@github.com:s2gos-dev/s2gos-mono.git
cd s2gos-mono
```

The example lives on the `icos_san_rossore` branch, not `main`.

## 3. Build the environment

```bash
pixi install --frozen -e dev
```

`--frozen` installs exactly what `pixi.lock` records.

## 4. Install eradiate's atmosphere data

```bash
pixi run --frozen eradiate-init
```

Downloads the absorption, aerosol and solar-irradiance datasets into
`~/.cache/eradiate` (set `ERADIATE_DATA_PATH` to put them elsewhere). Once per machine.

## 5. Get the scene bundle

Download `san_rossore_icos_bundle.tar.gz` from sharepoint.

It holds the inputs that cannot be fetched automatically: DEM and landcover tiles,
building footprints, the tower and forest meshes, and the material spectra.

## 6. Unpack the bundle

```bash
pixi run --frozen python scripts/setup_scene_bundle.py /path/to/san_rossore_icos_bundle.tar.gz
```

Unpacks the data into `s2gos_data/` and writes `s2gos_settings.yaml` pointing at it. Both
are git-ignored. An existing settings file is kept as `s2gos_settings.yaml.backup`.

## 7. Run the example

```bash
pixi run --frozen -e dev python examples/san_rossore_icos.py
```

Needs network on the first run (roads from Overpass, the ephemeris). Output lands in
`san_rossore_icos/`: the generated scene, `sim_output/visuals/` (DHP and RGB images) and
`sim_output/par/` (PAR flux).

## Troubleshooting

- **`pixi install` complains the lock file is out of date** — keep `--frozen`; pull the
  latest `pixi.lock` rather than re-solving.
- **`... not found in any search path`** — `s2gos_settings.yaml` does not point at the
  bundle. Re-run step 6.
- **eradiate reports a missing resource or absorption database** — re-run step 4.
- **`Eradiate not available — skipping simulation`** — you are not in the `dev`
  environment; use `-e dev` as in step 7.
