---
title: "Troubleshooting"
subtitle: "Find the cause before changing the case or environment"
---

Start with the first specific error in the failed workflow's log, linked from the closing `summary.md`.
If no summary was produced, read the PowerShell output or `console.log`.

Find the matching symptom below. Keep the configuration and logs while diagnosing the problem;
deleting outputs can remove evidence and force unnecessary reruns.

- [Python or shell command is unavailable](#python-or-shell-command-is-unavailable)
- [Julia or weathergenr is unavailable](#julia-or-weathergenr-is-unavailable)
- [The case folder is missing or already exists](#the-case-folder-is-missing-or-already-exists)
- [The delineated basin is unexpected](#the-delineated-basin-is-unexpected)
- [A catalog root is missing](#a-catalog-root-is-missing)
- [CMIP6 data cannot be downloaded](#cmip6-data-cannot-be-downloaded)
- [A project is locked or a run was interrupted](#a-project-is-locked-or-a-run-was-interrupted)
- [An experiment refuses changed inputs](#an-experiment-refuses-changed-inputs)
- [The preview is partial or an indicator value is missing](#the-preview-is-partial-or-an-indicator-value-is-missing)
- [Memory or performance is a concern](#memory-or-performance-is-a-concern)
- [If you need further help](#if-you-need-further-help)

## Python or shell command is unavailable {#python-or-shell-command-is-unavailable}

If PowerShell reports `Python was not found`, run the command with `pixi run` from the `blueearth_cst` folder.
If entering `pixi shell` is blocked by script policy, use `pixi run` directly; it does not require
changing PowerShell execution policy. Confirm installation with the [setup checks](install.md).

## Julia or weathergenr is unavailable {#julia-or-weathergenr-is-unavailable}

**Symptom:** scenario generation stops with `there is no package called 'weathergenr'`, or a model run cannot
find Julia or Wflow.

**Cause:** the weather generator and the Julia packages for Wflow are installed by `pixi run install`, not by
`pixi install`. Julia itself comes from juliaup and must be on `PATH`, outside Pixi.

**Fix:** check `julia +1.11.7 --version` and, if it fails, run `juliaup add 1.11.7` and reopen PowerShell.
Then run `pixi run install` and repeat the failed command. The toolbox uses its own R library, not a personal
one. If the installation itself fails, read its specific error: an Rtools message alone does not mean the
weather generator needs a separate Rtools installation.

### Julia installation file locks {#julia-installation-reports-access-is-denied}

**Symptom:** installing Julia or its packages fails with `Access is denied` or `directory not empty`.

**Cause:** Windows Defender real-time scanning can lock files while Julia moves them into place.

**Fix:** see the diagnosed installation fixes in
[environment maintenance](../../dev/scripts/environment-notes.md). Use those remedies for the diagnosed
symptom, not as an automatic prerequisite for every installation.

## The case folder is missing or already exists {#the-case-folder-is-missing-or-already-exists}

[Case](glossary.md#case) creation stops with `Cannot create trial:` followed by the reason, and writes no
files. Find the reason below and correct the [command](new-case.md#create-the-case-files); append `--dry-run`
to check it without writing.

- **`--cases-root must name an existing absolute directory`.** The cases folder does not exist, or the path is
  relative. Create the folder, then pass its full path, such as `C:/cst-cases`:

  ```powershell
  New-Item -ItemType Directory -Force C:/cst-cases
  ```

- **`Trial already exists`.** The case already has a [trial](glossary.md#trial) with this name. Open its
  configuration to continue that trial, or choose another name with `--trial`. Creation never overwrites a
  trial.
- **`must be lowercase kebab-case and not a reserved filename`.** Use lowercase letters, digits and single
  hyphens, such as `my-basin` or `v2`. Names the toolbox uses for its own folders (`data`, `outputs`, `logs`,
  `cache`, `applications`, `runs`, `trials`, `templates`) and Windows device names such as `con` or `nul` are
  refused.
- **`A new case requires --longitude and --latitude`.** Supply both coordinates when creating the first trial
  of a case. For a later trial, omit them or repeat the case's own values; different coordinates are refused.

## The delineated basin is unexpected {#the-delineated-basin-is-unexpected}

**Symptom:** the basin boundary, river network or [reporting locations](glossary.md#reporting-location) do not
match the catchment you expect.

**Cause:** usually the outlet coordinates (longitude comes before latitude) or the `uparea` threshold in
`basin.region`, which is a minimum upstream area, not the catchment area.

**Fix:** correct them and check the new boundary, especially near a confluence, before continuing. If the
outlet falls on an isolated drainage cell, search for an outlet within an area as described below.

### Search for an outlet within an area {#search-for-an-outlet-within-an-area}

Use a `subbasin` search box instead of a point. HydroMT expects coordinates in west, south, east, north order:

```{.yaml filename="project_config.yml (excerpt)"}
basin:
  region: "{'subbasin': [29.81375, 31.12208, 29.87375, 31.18208], 'uparea': 100}"
  sources:
    hydrography: merit_hydro_ihu
    basin_index: merit_hydro_ihu_index
```

Use a hydrography and index pair available in your catalog. The box searches for outlets: upstream catchments
can extend outside it, and several rivers can select several basins. It is distinct from the `bounds` argument
that limits available hydrography. It does not change your reporting-location file. Inspect the result before
running further workflows, and record any manual coordinate correction in the case notes.

## A catalog root is missing {#a-catalog-root-is-missing}

**Symptom:** `None of the specified roots were found`, or a named dataset cannot be found.

**Cause:** the data folders named by `project.catalog` are not reachable, or a source name or coverage does
not match the catalog. Installing the toolbox does not install datasets.

**Fix:** for the supplied P-drive catalog, check access with `Test-Path 'P:\wflow_global\hydromt'` and mount
the drive, or choose a catalog that points to data you can reach. Then check that the names under
`basin.sources` and `climate.forcings` in the [project file](glossary.md#project-file) match catalog entries
that cover your basin and period; see
[Define the catchment and data sources](new-case.md#define-the-catchment-and-data-sources). Another catalog is
not automatically equivalent scientific input: review the source choice before reusing results.

## CMIP6 data cannot be downloaded {#cmip6-data-cannot-be-downloaded}

**Symptom:** projection analysis fails while fetching CMIP6 data.

**Cause:** no internet access to the data source, or a requested model, experiment, variable or period that
the configured catalog does not provide.

**Fix:** repeat the CMIP6 access check in the [installation guide](install.md); if it fails, check your
connection and network restrictions. If it succeeds, read the projection log for the combination involved and
check it against the catalog, then rerun and check the reported coverage. An excluded model or missing period
is not zero climate change.

## A project is locked or a run was interrupted {#a-project-is-locked-or-a-run-was-interrupted}

1. Confirm that the previous run has stopped and no other process is writing to the same project.
2. Use the same toolbox folder and case configuration. Replace the example path below with your own;
   for incomplete jobs, run:

   ```powershell
   $config = 'C:/cst-cases/my-basin/default/project_config.yml'
   pixi run python scripts/run_workflows.py --config $config --cores 3 -- --rerun-incomplete
   ```

3. If the workflow engine (Snakemake) reports a stale lock after the active process has stopped, unlock
   through the [run command](glossary.md#run-command):

   ```powershell
   pixi run python scripts/run_workflows.py --config $config --cores 3 -- --unlock
   ```

4. Rerun the normal command and inspect the new summary. If the run command reports damaged or partial
   model-response files, keep the logs for diagnosis; [experiment](glossary.md#experiment) number recovery
   does not repair them.

Use [experiment recovery](running.md#continue-after-an-interruption) for an interrupted experiment
number allocation. Do not recycle allocated experiment numbers or delete locks while a run is active.

### Windows stops advancing after successful climate-projection writes

**Symptom:** fetch logs report successful CMIP6 writes, but the workflow engine stops reporting completed
jobs, or the run command reports `Windows socket-pair setup timed out after 3 attempts`.

**Fix:** keep the console log, stop the run, then resume with the incomplete-job command above and the same
configuration. Keep the CMIP6 cache and completed outputs; do not use `--rebuild` for this symptom. For
diagnosis, see the socket-pair notes in
[environment maintenance](../../dev/scripts/environment-notes.md#windows-socket-pair-stalls-after-climate-projection-writes).

## An experiment refuses changed inputs {#an-experiment-refuses-changed-inputs}

**Symptom:** the run command refuses to continue an existing [experiment](glossary.md#experiment).

**Cause:** the model, climate scenarios or scientific simulation settings changed since the experiment
started.

**Fix:** start a new experiment with `--new-experiment`, keeping the previous results for comparison; see
[Changed inputs](running.md#start-a-new-experiment-after-changing-inputs). For different indicators
over completed simulations, use the
[indicators-only procedure](running.md#calculate-different-indicators-from-existing-simulations); it does not
supply missing model outputs. Check [indicator inputs](results.md#check-status-before-value).

## The preview is partial or an indicator value is missing {#the-preview-is-partial-or-an-indicator-value-is-missing}

**Partial preview:** some tasks need files produced by earlier workflows before they can be planned.
On a first run, a preview may report pending inputs rather than show the entire downstream plan.
Check whether the message describes an expected dependency or a configuration error that stops the command.

**Missing indicator value:** read [`status` and `reason`](results.md#check-status-before-value) in the result
tables, then check whether the required response variable, threshold and
[unperturbed reference](glossary.md#unperturbed-reference) run were supplied.
For historical flow-threshold preparation, read the
[crop-water requirements](../../blueearth_cst/indicators/sets/crop_water/README.md). Distinguish `undefined`
from `unavailable` using the result's status/reason and set conventions.

## Memory or performance is a concern {#memory-or-performance-is-a-concern}

Lower `--cores` if several jobs compete for memory. Basin size, saved model variables, simulation period
and scenario count all affect memory and disk needs.

For hydrological simulations, review `compute` settings before increasing concurrent simulation batches.
Use [runtime and resource measurements](../technical-manual/runtime-and-resources.md) to guide adjustments.

## If you need further help {#if-you-need-further-help}

Share the following, with credentials and private paths removed:

- The command, first specific error, [run summary](glossary.md#run-summary) and affected workflow log.
- The toolbox version reported at startup and relevant configuration settings.
- What you expected and whether the same case previously completed.
