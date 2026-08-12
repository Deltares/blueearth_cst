# Installing BlueEarth-CST (Windows)

Step-by-step setup for running the BlueEarth Climate Stress Test toolbox on a
fresh Windows machine. Use **Windows PowerShell** for every command below.

Steps 1–3 are a one-time machine setup. Steps 4–7 install the project itself.

## Step 1 — Install pixi

pixi manages the project's Python and R tools inside a self-contained folder, so
nothing clutters the rest of your system.

Open PowerShell and run:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```
Close and reopen PowerShell afterward so it can find the `pixi` command.

## Step 2 — Let Julia install without interference

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

## Step 3 — Install Julia (version 1.11.7)

Install juliaup (the Julia version manager), then the exact Julia version the
project needs:
```powershell
winget install Julialang.Juliaup
juliaup add 1.11.7
```
Using the full name `Julialang.Juliaup` avoids the "multiple packages found"
prompt — do not pick plain "Julialang.Julia".

The version must be exactly **1.11.7**. The project is pinned to it, and a newer
1.11.x will not work. Confirm it installed:
```powershell
julia +1.11.7 --version
```
You should see `julia version 1.11.7`.

## Step 4 — Download the project

```powershell
git clone https://github.com/tanerumit/blueearth_cst.git
cd blueearth_cst
```
Run all the remaining steps from inside this `blueearth_cst` folder.

## Step 5 — Install the Python and R tools

```powershell
pixi install
```
This downloads everything the project needs into a local `.pixi` folder. The
first run can take several minutes.

## Step 6 — Install the weather generator and hydrological model

```powershell
pixi run install
```
This adds two more pieces on top of step 5:

- **weathergenr** — the R weather generator
- **Wflow.jl** — the Julia hydrological model and its dependencies (~130 packages)

The Julia part downloads a lot, so give it time. If it stops partway with a
*"directory not empty"* error, simply run `pixi run install` again — it resumes
where it left off. (If it keeps failing here, go back and do step 2.)

## Step 7 — Check that it works

Enter the project environment:
```powershell
pixi shell
```
Run the quick check (validates all three workflows without heavy computation):
```powershell
pytest tests/test_cli.py
```
This is fast but only checks that the workflows are wired correctly — it does not
actually run them.

For a full end-to-end check that builds and runs the test model to completion
(needs the data files from step 5 and Julia, and takes a few minutes):
```powershell
pytest tests/test_workflow_model_creation.py --run-integration
```

There is a matching end-to-end check for the climate-projections workflow. It
downloads CMIP6 data from the internet and must run *after* the model-creation
workflow (it reuses that model's basin outline):
```powershell
pytest tests/test_workflow_climate_projections.py --run-integration
```

Or run the small test model directly:
```powershell
snakemake all -c 1 -s Snakefile_model_creation --configfile test_case/snake_config_baseline.yml
```

## Troubleshooting

- **"`1.11.7` is not installed"** when running a workflow → you have a different
  Julia version. Run `juliaup add 1.11.7`.
- **"Access is denied" or "directory not empty"** during a Julia install → do
  step 2 (the Defender exclusion), then retry the command.
- **Re-running is safe.** Both `pixi install` and `pixi run install` skip work
  that is already done, so you can rerun them anytime.
- **weathergenr location.** On Windows the R weather generator installs into your
  personal R library (under `AppData\Local\R`) rather than the project folder.
  This is intentional and avoids a known Windows build crash.
- **Certificate messages from Julia.** If you run Julia by hand and see repeated
  `curl_easy_setopt: 4` messages, they come from a Windows certificate setting.
  The project's install step works around this automatically, so you can ignore
  them.
