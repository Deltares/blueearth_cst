# Environment setup notes

The environment is managed by **pixi** (`pixi.toml` + `pixi.lock`), a single
conda-forge-only env named `default`. Enter it with `pixi shell` or run one-off
commands via `pixi run <cmd>`. Julia is *not* in the pixi env — it is
juliaup-managed and must be on `PATH`; its packages are pinned by the Julia
`Manifest.toml` and installed via `pixi run install-julia`.

This file records local fixes for upstream packaging issues (not project bugs).
They fall in two groups:

- **Live workarounds**, now wired into pixi activation so they apply on every
  `pixi shell` / `pixi run` with no manual steps.
- **Superseded conda-era notes**, kept only as historical diagnosis. The old
  `cst` conda env pulled packages from the anaconda `defaults` channel and was
  repaired by hand-edited `environment.yml` pins. pixi resolves a modern,
  conda-forge-only, locked stack, so those pins no longer apply — do **not**
  carry them forward.

## Live workarounds (automated via pixi activation)

### Graphviz `dot` — missing `ffi.dll` / `cairo.dll` aliases (Windows)

`snakemake --dag | dot -Tpng` printed warnings and rendered fonts as Times
instead of `sans`:

```
Warning: Could not load "...\Library\bin\gvplugin_pango.dll" - ...
Warning: no hard-coded metrics for 'sans'.  Falling back to 'Times' metrics
```

**Root cause.** Two name-mismatch bugs in conda-forge's Windows graphviz/glib
build (diagnosed by walking the PE import table with `pefile`):

- `gobject-2.0-0.dll` imports `ffi.dll`, but conda-forge libffi ships `ffi-8.dll`.
- `gvplugin_pango.dll` imports `cairo.dll`, but conda-forge cairo (historically)
  shipped `cairo-2.dll`.

**Fix.** `dev/scripts/pixi_activate.bat` recreates the missing alias DLLs from
whatever is shipped. It is registered as a Windows activation script in
`pixi.toml`:

```toml
[target.win-64.activation]
scripts = ["dev/scripts/pixi_activate.bat"]
```

pixi runs it on every activation, so — unlike the old conda `activate.d`
approach — it survives env rebuilds with no manual re-creation. The script is
idempotent (guarded by `if not exist`). The current build already ships
`cairo.dll`, so in practice only the `ffi.dll` alias is created today; the cairo
branch is kept as a guard against a future regression.

Verify after `pixi shell`:

```powershell
Test-Path (Join-Path $env:CONDA_PREFIX 'Library\bin\ffi.dll')   # -> True
```

### netCDF4 `RuntimeError: Invalid argument` reading off the Deltares P: share

Reading ERA5 netCDF/HDF5 from `p:\wflow_global\hydromt\meteo\...` failed
mid-read inside `xarray`/`netCDF4`.

**Root cause.** HDF5 takes an `flock`-style lock on the source file by default.
The Deltares `P:` SMB share rejects that lock call, so the read fails.

**Fix.** `HDF5_USE_FILE_LOCKING=FALSE`, set cross-platform via pixi so it applies
to every shell (also generally safe/recommended on network filesystems):

```toml
[activation.env]
HDF5_USE_FILE_LOCKING = "FALSE"
```

Verify: `pixi run pwsh -NoProfile -Command '$env:HDF5_USE_FILE_LOCKING'` → `FALSE`.

The local data-staging path below (mirroring a bbox subset onto local SSD)
avoids the SMB read entirely and is the more robust option for repeated runs.

## Superseded conda-era notes (historical)

These were repairs to the old `cst` conda env (recorded 2026-05-05). They are
retained for context only; pixi resolves the versions shown below from
conda-forge with a lockfile, so the symptoms and their hand-pin fixes no longer
apply.

| Symptom (conda era) | Old fix | Status under pixi |
|---------------------|---------|-------------------|
| `np.unicode_` AttributeError in hydromt — env had `numpy 2.x` beside an ancient `xarray 0.20.1` from anaconda `defaults`. | Pin `numpy<2`, `xarray<=2024.3.0`; add `nodefaults` + strict priority. | Gone. conda-forge-only lock resolves `numpy 2.4.6` + `xarray 2026.4.0` (compatible). Do **not** re-pin `numpy<2`. |
| `to_zarr` codec/chunk errors — `xarray 2024.3.0` predates `zarr 3.0` but env had `zarr 3.1`. | Pin `zarr<3`. | Gone. Modern `xarray 2026.4.0` + `zarr 3.2.1` interoperate. Do **not** re-pin `zarr<3`. |
| Wflow.jl `Invalid value kinematic-wave in the TOML` — `hydromt_wflow 0.5.0` emitted pre-1.0 TOML, local Wflow.jl was 1.0.x. | Pin Wflow.jl to `0.8.1` to match `hydromt_wflow 0.5.0`. | **Obsolete and now wrong.** pixi ships `hydromt_wflow 1.0.2`, which emits Wflow.jl **1.x** TOML. The Julia env (`Manifest.toml`) must track a matching Wflow.jl 1.x — not 0.8.1. |

If the Wflow TOML/version mismatch resurfaces under the 1.x stack, that is a new
issue to diagnose against `hydromt_wflow 1.0.2` / Wflow.jl 1.x — the 0.8.1 note
above will not help.

## Local data staging (P-drive bypass)

To eliminate the SMB read failures entirely (and speed up runs), stage a
bbox-clipped subset of the Deltares data onto the local SSD.

### What gets staged

Each spatial dataset the model_creation workflow reads is clipped to the bbox
and mirrored under a local root. Default bbox is
`[8.5, -0.5, 11.0, 1.5]` (~1.5° pad around the test point at `(9.666, 0.4476)`).
Default destination is `C:\data\wflow_global\hydromt`.

Datasets covered:

| Dataset                         | Method                  |
|---------------------------------|-------------------------|
| `topography/merit_hydro_ihu/30sec/*.tif` | rasterio window read |
| `topography/merit_hydro/basin_index.gpkg` | geopandas bbox filter |
| `hydrography/rivers_lin2019/rivers_ge30m.gpkg` | geopandas bbox filter |
| `landuse/vito/ProbaV_LC100_*.tif` | rasterio window read |
| `landuse/modis/MODIS_MCD15A3H_LAI/*.tif` | rasterio window read |
| `soil/soilgrids_v1.0/*.tif` | rasterio window read |
| `meteo/era5_daily.zarr` | xarray.sel + to_zarr |
| `meteo/era5/meta/era5_orography_2018.nc` | xarray.sel + to_netcdf |

### Run it

The staging script reads its inputs (source root, destination root, bbox,
list of datasets) from `dev/scripts/stage_data.yml`. Edit that file to add or
remove datasets, change paths, or shift the bbox.

```powershell
pixi run python dev/scripts/stage_data.py                    # uses dev/scripts/stage_data.yml
pixi run python dev/scripts/stage_data.py --config <path>    # alternate config
pixi run python dev/scripts/stage_data.py --bbox 8 -1 12 2   # one-off bbox override
```

Supported dataset `type` values:

| type          | clipped via                                   |
|---------------|-----------------------------------------------|
| `raster`      | `rasterio` window read on a single GeoTIFF    |
| `raster_glob` | iterate `pattern` (default `*.tif`) in a dir  |
| `vector`      | `geopandas.read_file(..., bbox=...)`          |
| `zarr`        | `xarray.open_zarr` + `sel` + `to_zarr`        |
| `netcdf`      | `xarray.open_dataset` + `sel` + `to_netcdf`   |

The script is idempotent — it skips files that already exist at the
destination. On failure it removes the partial output so the next run can
retry cleanly, and continues past errors so one bad dataset does not block
the rest.

### Switch the workflow to the local catalog

Use `config/deltares_data_local.yml` (a copy of `deltares_data.yml` with
`root:` pointing at `C:/data/wflow_global/hydromt`) by either:

- pointing the snake config's data catalog key at it, or
- passing `--config data_catalog=config/deltares_data_local.yml` to snakemake
  if your Snakefile reads it from `config`.

### Limits

The script only stages spatial datasets the **model_creation** workflow
touches. CMIP6 data used by `analyze_projections.smk` /
`run_stress_test.smk` lives in `cmip6_data.yml` and is not yet
mirrored — extend the script when you need those flows.

## Recovering from a fresh env build

`pixi install` (native deps) then `pixi run install` (R + Julia layers) rebuilds
everything. The graphviz DLL shim and `HDF5_USE_FILE_LOCKING` are applied
automatically by pixi activation — no manual `activate.d` re-creation is needed
anymore. Sanity-check inside `pixi shell`:

```powershell
python -c "import xarray, numpy; print(xarray.__version__, numpy.__version__)"
Test-Path (Join-Path $env:CONDA_PREFIX 'Library\bin\ffi.dll')
```
