# Environment maintenance

For maintainers diagnosing installations, staging local data or borrowing a runtime. User setup lives in
[Install the toolbox](../../docs/guide/install.md); ordinary recovery is in
[Troubleshooting](../../docs/guide/troubleshooting.md). Session allocation and evidence rules remain in the
development workflow (private-checkout-only citation: `../reference/private/dev-task-and-session-workflow.md`) and
[validation reference](../reference/dev-validation.md). This file owns tool-specific recipes, not a second
session or validation policy.

The environment is managed by **pixi** (`pixi.toml` + `pixi.lock`), a single
conda-forge-only env named `default`. Enter it with `pixi shell` or run one-off
commands via `pixi run <cmd>`. Julia is *not* in the pixi env — it is
juliaup-managed and must be on `PATH`; its packages are pinned by the Julia
`Manifest.toml` and installed via `pixi run install-julia`.

This file describes automated workarounds for upstream packaging issues and local data staging.

## Inspecting running Python processes

The Pixi environment includes `py-spy` for profiling slow Python code and inspecting hung processes.
Run `pixi run py-spy top --pid <process-id>` for live profiling or
`pixi run py-spy dump --pid <process-id>` for current thread stacks. Replace `<process-id>` with
the Python process ID shown in Task Manager. These commands inspect a live process; capture a
Python exception traceback when diagnosing a process that has already exited.

## Live workarounds (automated via pixi activation)

### Graphviz `dot` — missing `ffi.dll` / `cairo.dll` aliases (Windows)

`snakemake --dag | dot -Tpng` printed warnings and rendered fonts as Times
instead of `sans`:

```
Warning: Could not load "...\Library\bin\gvplugin_pango.dll" - ...
Warning: no hard-coded indicators for 'sans'.  Falling back to 'Times' indicators
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

## Local data staging (P-drive bypass)

To eliminate the SMB read failures entirely (and speed up runs), stage a
bbox-clipped subset of the Deltares data onto the local SSD.

### What gets staged

Each spatial dataset the build_model workflow reads is clipped to the bbox
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

The optional `config/catalogs/deltares_data.yml` uses the local
`C:/data/wflow_global/hydromt` root. Select it with `project.catalog` in a
project config, then review `basin.sources`: the local catalog's river and
land-cover names differ from the default P-drive catalog. The supplied seed
configs use `config/catalogs/deltares_data_pdrive.yml`.

### Limits

The script only stages spatial datasets the **build_model** workflow
touches. CMIP6 data used only by `analyze_projections.smk` lives in `cmip6_data.yml` and is not yet
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

### Missing console script with a healthy pixi package record

A package can import and appear installed in `pixi list` while its generated
console executable is absent. A clean `pixi install` can trust the package
metadata and leave the missing file behind; the failure then appears only in
tests or tools that shell out to the command. `check_env.py` detects missing
files by comparing distribution `RECORD` entries with `Scripts/`.

For the recorded Snakemake 9.6.2 incident, pixi repaired the environment after
its exact `snakemake-9.6.2.dist-info` record was removed and `pixi install`
rerun. Inspect the candidate path, confirm it is the expected record, and use
that exact path with `Remove-Item -LiteralPath`:

```powershell
Get-ChildItem .pixi\envs\default\Lib\site-packages -Directory -Filter 'snakemake-9.6.2.dist-info'
Remove-Item -LiteralPath '.pixi\envs\default\Lib\site-packages\snakemake-9.6.2.dist-info' -Recurse -Force
pixi install
```

`pixi clean` + `pixi install` rebuilds the environment. Do not edit
`pixi.lock` or `pixi.toml`. Incident date: 2026-08-12; see `t2608121104` in
`dev/LOG.md`.

### Local import timings under real-time scanning

On one Windows 11 machine, fresh-process `python -X importtime -c "import hydromt"`
measured 13.3–15.6 seconds across three runs (3,173 modules; about 4.79 ms per
module), while normal per-module imports were estimated at 0.3–1 ms. This is
observed slow import timing; a real-time scanner was present, but no exclusion
A/B established it as the cause. Treat scanner attribution and the estimated
suite-time payoff as a hypothesis, not a general pixi property. The owner
could not obtain an exclusion on 2026-09-04. At that time the local full suite
took about 59 minutes versus about 9 minutes on CI; that gap is historical and
not wholly attributed to the scanner. Incident ID: `t2608202331`.


## Session runtimes

Session creation preserves separate Git worktrees, fixture copies and approval-based landing, but
does not provision a runtime automatically. Existing `.pixi` environments are left in place. Start a
task through the configured Git workflow launcher; use `--no-reseed` only when deliberately retaining
the session's existing fixture snapshot. Session allocation and runtime selection are separate choices.

For a clean, detached slot allocated by the user, run from the primary checkout (replace the skill
directory and task name with the resolved local values), then enter the worktree reported by the tool:

```powershell
$gitWorkflow = '<resolved git-workflow skill directory>'
& .pixi/envs/default/python.exe "$gitWorkflow/scripts/worktree-session.py" task-start --task example --session session-2 --base main --no-launch
Set-Location 'C:/workspace/.worktrees/blueearth_cst/session-2'
```

Do not claim an occupied slot. All three kinds of work below start this way; select the runtime only
after entering the session. Artifact work needing no tools stops before installation or borrowing.

| Work | Runtime |
|---|---|
| Text, task records, or artifacts needing no executable checks | No installation |
| Source checks/artifacts; unchanged dependencies | Borrow a stable runtime, or use the existing local one |
| Dependency changes, install tests, incompatible contracts, workflow qualification | Isolated environment |

### Borrow a stable runtime for development checks

The owner of the lending checkout owns all installation state: Pixi packages, the out-of-lock R
`weathergenr` installation, Graphviz aliases and Julia package/artifact installation. Before lending,
the owner finishes `pixi install --locked`, `pixi run install`, activation and
`pixi run python dev/scripts/check_env.py`. Keep that checkout's dependency files and installation
unchanged until all borrowers finish. Coordinate that interval with its owner; never lend a checkout
whose owner is changing dependencies. A verified primary checkout can be the lender between upgrades.

From the **borrowing session's root**, using PowerShell:

```powershell
$runtime = 'C:/workspace/blueearth_cst'
& "$runtime/.pixi/envs/default/python.exe" -B dev/scripts/session_runtime.py --runtime-root $runtime -- python -m pytest tests/test_cli.py
& "$runtime/.pixi/envs/default/python.exe" -B dev/scripts/session_runtime.py --runtime-root $runtime -- ruff check .
```

Use the corresponding `<runtime>/.pixi/envs/default/bin/python` bootstrap on Linux. The bootstrap runs
only standard-library coordination code; the child receives Pixi activation. The helper compares
TOML values in `pixi.toml` and both Julia project files, and newline-normalized text in `pixi.lock`,
the R installer and Windows activation script. Any difference fails before Pixi is invoked. This conservative
check also rejects task-only manifest changes. It checks contract equality, not the health of an
arbitrarily modified installation; the owner's provisioning and verification remain prerequisites.

TOML comments and formatting do not affect this contract comparison.
The helper runs an **executable**, so spell checks out as `python -m pytest ...`, `ruff ...`, or
`Rscript --vanilla ...`; Pixi task names such as `test-fast` are not expanded. It uses
`pixi run --as-is --executable --manifest-path <runtime>/pixi.toml`, preserving the session's working
directory and preventing lock updates and installations. It prints the actual imported Python source
path and rejects imports outside the session. `blueearth_cst` is not an editable installation.
Do not use a lending checkout's named Pixi task directly: named tasks normally execute at that
checkout's manifest root. Pixi documents these flags in its
[run reference](https://pixi.sh/latest/reference/cli/pixi/run/).

`PYTHONPATH` points at the session, user Python packages and bytecode writes are disabled, and the
intended Julia project is the session's `Project.toml`. Use `--project=.` for Julia commands, as the
workflow already does. Julia writes caches into a session-local first depot and reads installed
packages/artifacts from the existing depots. R uses the lending Pixi library; invoke it with
`--vanilla` to avoid personal startup files. The helper clears ambient `R_LIBS`/`R_LIBS_SITE` and disables the
personal R library. Matplotlib, XDG/Cartopy, Numba, PROJ and Ruff caches live in
`.tmp/session-runtime/`; pytest retains
its existing unique, session-local `.tmp/pytest/run-*` directories. Give application outputs explicit
session-local paths. External data mirrors remain external resources and must not be modified by
checks. Existing fixture copies remain independent; do not execute workflows into another session's
`test_local` tree.

Borrowing is cooperative development tooling, not a filesystem sandbox. The helper blocks common
installer commands, but arbitrary Python/R/Julia code can still write to installed packages. Do not
run installers, package-manager APIs, or dependency-mutating checks through it. Do not use borrowed
checks as evidence that a session installs successfully. Full workflow qualification continues to
use an isolated session environment and the validation ladder's required execution coverage.

### Use or return to isolated execution

No `.pixi` link, environment relocation, or persistent shell change is made by borrowing. From the
session root, use the ordinary commands to switch back:

```powershell
pixi install --locked
pixi run install
pixi run python dev/scripts/check_env.py
pixi run pytest tests/test_cli.py
```

For an intentional dependency change, edit the session's manifest, run `pixi install` to solve its
lock, and retain both in the same change. R install changes stay in that session's Pixi library.
Julia's default depot is machine-wide even with an isolated Pixi prefix: when testing Julia
installation itself, set `JULIA_DEPOT_PATH` to a fresh ignored session-local directory **before**
`pixi run install-julia`, and retain it while checking that installation. Never edit shared Julia
packages in place. After dependency work is accepted, only the runtime owner upgrades the lending
environment, with no active borrowers; reverify before lending again.

Keep the configured session capacity; unprovisioned slots are cheap compared with installed runtimes. A
separate lightweight dependency set would add another lock/qualification burden. Pixi's detached environment
setting relocates prefixes; it does not by itself make workspaces share one environment. Conda hard links can
already share package bytes with the machine cache, so logical prefix sizes overstate marginal disk cost,
while every prefix still adds directory entries and private files. Do not copy or junction `.pixi`
directories. See the retained worktree-runtime evidence (private-checkout-only citation:
`../records/validation/board-adoption-2026-10-30/worktree-runtime/evidence.md`) for the measured sizes and
limitations; no existing environment, fixture, output or worktree is removed.

## Julia installation file locks

The following existing remedy concerns Windows installation file-lock failures.
Confirm the symptom and local security permissions before applying it.

The hydrological model (Wflow) runs on Julia. Julia's installer downloads files
to a temporary folder and then moves them into place. Windows Defender's
real-time scanning sometimes locks those files mid-move, which makes the install
fail with errors like *"Access is denied"* or *"directory not empty"*.

To prevent this, tell Defender to leave Julia alone. Open PowerShell **as
Administrator** (right-click PowerShell → *Run as administrator*) and run:
```powershell
Add-MpPreference -ExclusionProcess "julia.exe"
Add-MpPreference -ExclusionProcess "juliaup.exe"
```
This is a one-time step and only affects these two programs.
