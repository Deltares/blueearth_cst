---
title: "Troubleshooting"
subtitle: "Find the cause before changing the case or environment"
---

Start with the first specific error in the failed workflow's log, linked from the closing `summary.md`.
If no summary was produced, read the PowerShell output or `console.log`.

Find the matching symptom below. Keep the configuration and logs while diagnosing the problem;
deleting outputs can remove evidence and force unnecessary reruns.

1. [Python or shell command is unavailable](#python-or-shell-command-is-unavailable)
2. [Julia or weathergenr is unavailable](#julia-or-weathergenr-is-unavailable)
3. [The case folder is missing or already exists](#the-case-folder-is-missing-or-already-exists)
4. [The delineated basin is unexpected](#the-delineated-basin-is-unexpected)
5. [A catalog root is missing](#a-catalog-root-is-missing)
6. [CMIP6 data cannot be downloaded](#cmip6-data-cannot-be-downloaded)
7. [A project is locked or a run was interrupted](#a-project-is-locked-or-a-run-was-interrupted)
8. [An experiment refuses changed inputs](#an-experiment-refuses-changed-inputs)
9. [The preview is partial or an indicator value is missing](#the-preview-is-partial-or-an-indicator-value-is-missing)
10. [Memory or performance is a concern](#memory-or-performance-is-a-concern)
11. [If you need further help](#if-you-need-further-help)

## 1. Python or shell command is unavailable {#python-or-shell-command-is-unavailable}

If PowerShell reports `Python was not found`, run the command with `pixi run` from the `blueearth_cst` folder.
If entering `pixi shell` is blocked by script policy, use `pixi run` directly; it does not require
changing PowerShell execution policy. Confirm installation with the [setup checks](install.md).

## 2. Julia or weathergenr is unavailable {#julia-or-weathergenr-is-unavailable}

1. Check `julia +1.11.7 --version`. Install that version with `juliaup add 1.11.7` if it is missing.
   Julia must be on `PATH`; it is not installed inside Pixi. Reopen PowerShell after installation
   so it picks up the updated environment.
2. Run `pixi run install` to complete the pinned R/Julia layers.
   The project uses its own R library, not a separately installed personal R library.
3. Repeat the failed command after installation. If installation itself fails, read the specific
   error before adding other tools: an Rtools message alone does not mean that the weather-generator
   package requires a separate Rtools installation.

For Julia `Access is denied` or `directory not empty` errors, see the diagnosed installation fixes in
[environment maintenance](../../dev/scripts/environment-notes.md).
Use those remedies for the diagnosed symptom, not as an automatic prerequisite for every installation.

## 3. The case folder is missing or already exists {#the-case-folder-is-missing-or-already-exists}

Case creation stops with `Cannot create trial:` followed by the reason. It writes no files when it
refuses. Find the reason below, then follow [Creating a project](new-case.md#create-the-case-files) for
the complete command. Append `--dry-run` to check a corrected command without writing.

- **`--cases-root must name an existing absolute directory`.** The cases folder does not exist, or the
  path is relative. Create the folder, then pass its full path, such as `C:/cst-cases`:

  ```powershell
  New-Item -ItemType Directory -Force C:/cst-cases
  ```

- **`Trial already exists`.** The case already has a trial with this name. Open its configuration to
  continue that trial, or choose another name with `--trial`. Creation never overwrites a trial.
- **`must be lowercase kebab-case and not a reserved filename`.** Use lowercase letters, digits and
  single hyphens, such as `my-basin` or `v2`. Names the toolbox uses for its own folders (`data`,
  `outputs`, `logs`, `cache`, `applications`, `runs`, `trials`, `templates`) and Windows device names
  such as `con` or `nul` are refused.
- **`Legacy split-layout case exists`** or **`Legacy flat case detected`.** A case with this name
  already exists in an older layout: under `applications/` or `runs/` in the cases folder, or with
  `project_config.yml` directly in the case folder. Choose a new case name, or first copy the old case
  into a named trial, following
  [Copy and verify an existing case](new-case.md#copy-and-verify-an-existing-case).
- **`A new case requires --longitude and --latitude`.** Supply both coordinates when creating the first
  trial of a case. For a later trial, omit them or repeat the case's own values; different coordinates
  are refused.

## 4. The delineated basin is unexpected {#the-delineated-basin-is-unexpected}

Review the outlet coordinates and `basin.region` upstream-area threshold. Coordinates are longitude
followed by latitude; the `uparea` threshold is not a prescribed catchment area. Check the resulting
boundary, river network and reporting locations, especially near a confluence, before continuing.

If the point falls on an isolated drainage cell, see
[Search for an outlet within an area](new-case.md#search-for-an-outlet-within-an-area).

## 5. A catalog root is missing {#a-catalog-root-is-missing}

For `None of the specified roots were found`, inspect the catalogs named by `project.catalog`.
For the supplied P-drive catalog:

```powershell
Test-Path 'P:\wflow_global\hydromt'
```

Mount the drive or choose a catalog that points to available data, then check source names and coverage.
Installing software does not install those datasets. A different catalog is not automatically equivalent
scientific input; review the source choice before reusing results.

If the catalog is accessible but a dataset cannot be found, check that the configured source name
matches an entry in the catalog and that the dataset covers the basin and analysis period. Review
`basin.sources` and `climate.forcings` in the project file; see
[Define the catchment and data sources](new-case.md#define-the-catchment-and-data-sources).

## 6. CMIP6 data cannot be downloaded {#cmip6-data-cannot-be-downloaded}

Repeat the CMIP6 access check in the [installation guide](install.md). If access fails, check your
internet connection and any network restrictions. If access succeeds but the workflow still fails,
inspect its projection log for the model, experiment, variable and period involved.

Check that the requested combination is available in the configured catalog. Do not interpret an
excluded model or missing period as zero climate change. After resolving access or selection issues,
rerun the workflow and check the reported coverage.

## 7. A project is locked or a run was interrupted {#a-project-is-locked-or-a-run-was-interrupted}

1. Confirm that the previous run has stopped and no other process is writing to the same project.
2. Use the same toolbox folder and case configuration. Replace the example path below with your own;
   for incomplete jobs, run:

   ```powershell
   $config = 'C:/cst-cases/my-basin/default/project_config.yml'
   pixi run python scripts/run_workflows.py --config $config --cores 3 -- --rerun-incomplete
   ```

3. If Snakemake reports a stale lock after the active process has stopped, unlock through the runner:

   ```powershell
   pixi run python scripts/run_workflows.py --config $config --cores 3 -- --unlock
   ```

4. Rerun the normal command and inspect the new summary. If the runner reports damaged or partial
   model-response files, retain the logs for diagnosis; experiment reservation recovery does not repair them.

Use [experiment recovery](running.md#select-resume-or-create-an-experiment) for an interrupted reservation.
Do not recycle allocated experiment numbers or delete locks while a run is active.

### Windows stops advancing after successful climate-projection writes

If fetch logs report successful CMIP6 writes but Snakemake stops reporting completed jobs, retain
the console log and process stacks before stopping the run. One observed failure is Python waiting
in `socket._fallback_socketpair` → `accept` while Snakemake holds its scheduler lock. Successful
downloads alone do not establish this diagnosis; other stalls need their own evidence.

The toolbox's owned runners bound Windows fallback socket-pair setup to five seconds per attempt
and retry timeouts up to three attempts. A transient failure prints a warning and retries; persistent
failure reports `Windows socket-pair setup timed out after 3 attempts`. This protection applies to
the Snakemake process launched by the toolbox, including dry runs and verbose mode. Direct `snakemake`
invocations and separate job processes do not receive it. Network download timeouts are unchanged.

After the old process has stopped, use the incomplete-job resume command above with the same
configuration. Preserve the CMIP6 cache and completed outputs; do not use `--rebuild` for this symptom.
If socket-pair timeouts persist, investigate local loopback filtering and socket-resource pressure.
Reducing `--cores` may help diagnosis but is not an established fix. The workaround has been checked
with injected failures and parallel job completion; confirmation on the originally affected machine
is still needed.


## 8. An experiment refuses changed inputs {#an-experiment-refuses-changed-inputs}

Review changes to the model, climate scenarios and simulation settings.
Use `--new-experiment` when scientific inputs changed. Keep previous results for comparison and traceability.
For different indicators over intact completed responses, use the
[indicators-only procedure](running.md#calculate-different-indicators-from-existing-simulations).
It does not supply missing model outputs. Check [indicator inputs](outputs.md#reading-an-indicator-set).

## 9. The preview is partial or an indicator value is missing {#the-preview-is-partial-or-an-indicator-value-is-missing}

**Partial preview:** some tasks need files produced by earlier workflows before they can be planned.
On a first run, a preview may report pending inputs rather than show the entire downstream plan.
Check whether the message describes an expected dependency or a configuration error that stops the command.

**Missing indicator value:** read `status` and `reason` in the result tables,
then check whether the required response variable, threshold and reference run were supplied.
For historical flow-threshold preparation, read the
[crop-water requirements](../../blueearth_cst/indicators/sets/crop_water/README.md).
Distinguish `undefined` from `unavailable` using the result's status/reason and set conventions.

## 10. Memory or performance is a concern {#memory-or-performance-is-a-concern}

Lower `--cores` if several jobs compete for memory. Basin size, saved model variables, simulation period
and scenario count all affect memory and disk needs.

For hydrological simulations, review `compute` settings before increasing concurrent simulation batches.
Use [runtime and resource measurements](../technical-manual/runtime-and-resources.md) to guide adjustments.

## 11. If you need further help {#if-you-need-further-help}

Share the following, with credentials and private paths removed:

- The command, first specific error, run summary and affected workflow log.
- The toolbox version reported at startup and relevant configuration settings.
- What you expected and whether the same case previously completed.
