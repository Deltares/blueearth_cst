# R01 Modularity Contracts Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Implement task-by-task and check off as you go. Do not skip the verification commands — they are the contract that catches migration drift.

**Goal:** Reorganize the flat snake config into `project` / `shared` / `workflows.<name>` top-level sections and update all 3 Snakefiles (plus the `src/` scripts and tests that read config directly) to read the sectioned config. Code structure unchanged. Per-workflow contract docs are **out of scope** — deferred to the openings of R3/R4/R5 per the 2026-07-17 design/roadmap amendment. End state: pre-rebaseline scientific fingerprints unchanged, only the documented config snapshots change, post-rebaseline `check_baseline.py check` is clean; test suite unchanged.

**Architecture:** One config template (no code, no contract docs). One atomic commit migrates the test config + the canonical example config + `tests/conftest.py` + the two integration tests + all 3 Snakefiles + the 3 direct `src/` config readers together (Task 2) — every file that parses or reads the schema flips in lockstep so no committed state parses at the Snakefile layer but fails in a script or test. A follow-up commit migrates the Linux example config (Task 3). A migration guide for user-local configs (Task 4). A final verification commit re-runs all three workflows freshly and seals the milestone (Task 5).

**Tech Stack:** YAML schema, Snakemake `workflow.configfiles[0]`, pytest, pixi env, existing `dev/scripts/check_baseline.py`.

**Branch:** `milestone/r01-contracts` (already on it; off `m02c-tests` tag).

**Spec:** `dev/milestones/r01/modularity-contracts-design.md`. Read it first if you haven't.

**Execution environment.** All shell command blocks below assume **Git Bash** on Windows (the repo's documented POSIX shell; `tail`, `head`, `cat <<'EOF'` heredocs, and forward-slash paths resolve there). If you run in PowerShell instead, translate: `... | tail -N` → `... | Select-Object -Last N`; `cat <<'EOF' ... EOF` heredocs → a single-quoted here-string `@'...'@`; and `git commit -m "$(cat <<'EOF' ...)"` → write the message to a temp file and `git commit -F <file>`. The pytest / snakemake / python invocations themselves are shell-agnostic.

---

## Pre-task setup

### Step 0.1: Verify branch and clean tree

- [ ] Run: `git branch --show-current` → expect `milestone/r01-contracts`
- [ ] Run: `git status --short` → expect empty output. (This plan file should already be committed before execution starts; if you see it as untracked, commit it first.)

### Step 0.2: Verify baseline test suite passes

- [ ] Run: `pixi run pytest tests/ 2>&1 | tail -3`
- [ ] Expect: `47 passed, 2 skipped, 2 xfailed` (the 2 skips are the default-skipped integration tests `tests/test_workflow_model_creation.py` and `tests/test_workflow_climate_projections.py`; the 2 xfails are the workflow-2/3 known-failure ratchets in `tests/test_cli.py`).

### Step 0.3: Verify baseline manifest is current

- [ ] Run: `pixi run python dev/scripts/check_baseline.py check 2>&1 | tail -10` — only if you have the baseline project (`examples/test_local`, the root the recorded manifest uses) already run; otherwise skip and rely on Task 5's final gated check.
- [ ] If skipping: note the manifest reference is `dev/baseline/manifest.json` from the M2b contract; it records `project_dir: examples/test_local`.

---

## Task 1: Add config template + schema convention

**Files:**
- Create: `config/snake_config.template.yml`

**Purpose:** A checked-in canonical schema example. Anyone authoring a new project config copies this template and edits values. Shows the exact section structure R01 locks in.

### Step 1.1: Create the template

- [ ] Write `config/snake_config.template.yml` with this content:

```yaml
# Snake config template for blueearth_cst (R01 schema).
#
# Three top-level sections:
#   project:        paths and external resources (every workflow reads)
#   shared:         cross-workflow scientific knobs (≥2 workflows read)
#   workflows.<n>:  per-workflow opts (only that workflow reads)
#
# Each workflows.<name> section has an `enabled:` flag (documentary in
# R01; will become operational at R6+ structural refactor).
#
# Per-workflow contract docs (dev/workflows/<name>.md) are written at
# the opening of R3/R4/R5, not in R01 (2026-07-17 amendment).

project:
  # Output root for this run
  project_dir: examples/test
  # Static config dir (templates referenced by workflows)
  static_dir: config
  # Hydromt data catalog (era5, merit_hydro, etc.)
  data_sources: config/deltares_data.yml
  # Hydromt data catalog for CMIP6 (climate projections workflow)
  data_sources_climate: config/cmip6_data.yml

shared:
  basin:
    # hydromt region spec (string with dict syntax — see hydromt docs)
    region: "{'subbasin': [9.666, 0.4476], 'uparea': 100}"
    # Grid resolution in degrees (≥ source hydrography res, e.g. merit_hydro_ihu = 0.00833)
    resolution: 0.00833
  historical_window:
    # ISO datetime strings — wflow forcing window
    starttime: "2000-01-01T00:00:00"
    endtime: "2005-12-31T00:00:00"
  # Historical climate source (must exist in project.data_sources catalog)
  clim_historical: era5

workflows:
  model_creation:
    enabled: true
    # Hydromt build config template
    model_build_config: config/wflow_build_model.yml
    # Hydromt update config for waterbodies (optional; defaults below)
    waterbodies_config: config/wflow_update_waterbodies.yml
    # Wflow output variables to save (read only by model_creation)
    wflow_outvars: ['river discharge']
    # Optional: csv with three cols (station_ID, x, y); None defaults to wflow outlets
    output_locations: None
    # Optional: csv with observed discharge timeseries per station
    observations_timeseries: None

  climate_projections:
    enabled: true
    clim_project: cmip6
    models:
      - NOAA-GFDL/GFDL-ESM4
      - INM/INM-CM5-0
      - CMCC/CMCC-ESM2
    scenarios: [ssp245, ssp585]
    members: [r1i1p1f1]
    variables: [precip, temp]
    start_month_hyd_year: Jan
    # CMIP6 monthly stats historical year range (different precision than
    # shared.historical_window — that one uses ISO datetimes for wflow forcing)
    historical_year_range: [1980, 2010]
    future_horizons:
      near: [2030, 2060]
      far:  [2070, 2100]
    # Save gridded outputs in addition to basin averages
    save_grids: false

  climate_experiment:
    enabled: true
    experiment_name: experiment
    realizations_num: 2
    # Future midpoint year for the climate experiment
    horizontime_climate: 2050
    # Length of future run in years
    run_length: 20
    # Run historical (cst_0) realization through wflow as well
    run_historical: false
    stress_test:
      temp:
        step_num: 1
        transient_change: true
        mean:
          min: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          max: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
      precip:
        step_num: 2
        transient_change: true
        mean:
          min: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
          max: [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
        variance:
          min: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
          max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Drought / flood return periods (years)
    Tlow: 2
    Tpeak: 10
    # Aggregate realizations before computing statistics
    aggregate_rlz: true
```

> **Note on `wflow_outvars`.** It lives under `workflows.model_creation`, not `shared`: the only production reader is `Snakefile_model_creation` (verified — `src/plot_results.py` mention is a comment; `setup_gauges_and_outputs.py` reads it as a rule param). The design doc's migration table already places it there.

### Step 1.2: Verify YAML parses

- [ ] Run: `pixi run python -c "import yaml; yaml.safe_load(open('config/snake_config.template.yml'))"`
- [ ] Expect: no output (clean parse)

### Step 1.3: Commit

- [ ] Run:
```
git add config/snake_config.template.yml
git commit -m "$(cat <<'EOF'
r01: add canonical sectioned config template

config/snake_config.template.yml is the R01 schema in checked-in form:
project / shared / workflows.<name> sections, with the enabled: flag
on each workflow as a forward-compat marker. Anyone authoring a new
project config copies this template and edits values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Atomic migration — test config + canonical config + conftest + integration tests + all 3 Snakefiles + 3 src/ readers

**Files (all in ONE commit):**
- Modify: `tests/snake_config_model_test.yml`
- Modify: `config/snake_config_model_test.yml`
- Modify: `tests/conftest.py`
- Modify: `tests/test_workflow_model_creation.py`
- Modify: `tests/test_workflow_climate_projections.py`
- Modify: `Snakefile_model_creation`
- Modify: `Snakefile_climate_projections`
- Modify: `Snakefile_climate_experiment`
- Modify: `src/prepare_cst_parameters.py`
- Modify: `src/prepare_weagen_config.py`
- Modify: `src/get_change_climate_proj.py`
- Create: `tests/test_r01_config_readers.py` (focused reader + normalization coverage, Step 2.13b)

**Purpose:** This is the load-bearing commit of R01. Every layer that parses or reads the schema — the two committed configs, the Snakefiles, the direct `src/` config readers, the conftest fixtures, and the two integration tests — flips from flat to sectioned in one atomic step. After this commit the whole suite (including `--run-integration`, insofar as data is available) is internally consistent against the sectioned schema; before it, it is not.

**Why the canonical config is in this commit (not a later one):** both integration tests hardcode `CONFIG = "config/snake_config_model_test.yml"` and read config keys from it (`tests/test_workflow_model_creation.py:48-51`, `tests/test_workflow_climate_projections.py:41-42,55-56`). If those tests migrate here but the canonical config stayed flat until a later commit, this commit would contain migrated test code that raises `KeyError` against a still-flat config under `--run-integration`. Folding the canonical config in keeps the atomic set self-consistent.

**Why the `src/` readers are in this commit:** Snakefile dry-run does not execute `script:` bodies, so a broken `src/` reader would not surface at the Snakefile layer. Committing the Snakefile flip without the reader fixes would leave committed state that parses at the Snakefile layer but fails inside a script at runtime — exactly the failure mode this atomic commit exists to prevent.

**Why atomic overall:** several keys (`project_dir`, `data_sources`, `clim_historical`, etc.) are read by multiple Snakefiles/scripts. A staged migration would need dual-key fallback logic (extra code to remove later) or a hybrid config that's hard to reason about. One commit is simpler and easier to revert.

**Risk:** A renamed key a consumer still reads under its old name → silent default. Mitigation: `snakemake --dry-run` all 3 Snakefiles + full `pytest tests/` before committing.

### Step 2.1: Rewrite tests/snake_config_model_test.yml

- [ ] Write the full new content (replaces existing 112 lines). `project_dir` stays `tests/test_project` — this is the test fixture's own root, used by `tests/test_cli.py` dry-runs; it is **not** the baseline-check root (Task 5 uses a separate local config at `examples/test_local`):

```yaml
# Test config for blueearth_cst — R01 sectioned schema.
# Used by tests/test_cli.py (Snakefile dry-runs) and any pytest cases
# that load a real config via tests/conftest.py.

project:
  project_dir: tests/test_project
  static_dir: config
  data_sources: tests/data/tests_data_catalog.yml
  data_sources_climate: config/cmip6_data.yml

shared:
  basin:
    region: "{'subbasin': [9.738, 0.4212], 'uparea': 70}"
    resolution: 0.0062475
  historical_window:
    starttime: "2015-01-01T00:00:00"
    endtime: "2020-12-31T00:00:00"
  clim_historical: era5

workflows:
  model_creation:
    enabled: true
    model_build_config: config/wflow_build_model.yml
    waterbodies_config: config/wflow_update_waterbodies.yml
    wflow_outvars:
      - river discharge
      - precipitation
      - overland flow
      - actual evapotranspiration
      - groundwater recharge
      - snow
    output_locations: tests/data/observations/output-locations-test.csv
    observations_timeseries: tests/data/observations/observations_timeseries_test.csv

  climate_projections:
    enabled: true
    clim_project: cmip6
    models:
      - NOAA-GFDL/GFDL-ESM4
      - INM/INM-CM5-0
    scenarios: [ssp245, ssp585]
    members: [r1i1p1f1]
    variables: [precip, temp]
    start_month_hyd_year: Jan
    historical_year_range: [2000, 2010]
    future_horizons:
      near: [2050, 2060]
      far:  [2090, 2100]
    save_grids: false

  climate_experiment:
    enabled: true
    experiment_name: experiment
    realizations_num: 2
    horizontime_climate: 2050
    run_length: 10
    run_historical: false
    stress_test:
      temp:
        step_num: 1
        transient_change: true
        mean:
          min: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          max: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
      precip:
        step_num: 1
        transient_change: true
        mean:
          min: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
          max: [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
        variance:
          min: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
          max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    Tlow: 2
    Tpeak: 10
    aggregate_rlz: true
```

Note: `historical_year_range: [2000, 2010]` becomes a list (not the comma-separated string `"2000, 2010"` from the flat config), and `future_horizons.near/.far` become lists (was `2050, 2060` strings). Downstream Python must accept both — see Step 2.13 (`get_change_climate_proj.py`).

### Step 2.2: Rewrite config/snake_config_model_test.yml (canonical example)

- [ ] Replace the entire file with the sectioned canonical example below. **Value-preservation is mandatory** — this config drives Task 5's fresh baseline run, so every leaf value must equal the pre-R01 flat config's value (only the *location* changes). See the old-leaf → new-path check in Step 2.15.

```yaml
# Canonical example config for blueearth_cst — R01 sectioned schema.
# This is the "small bbox, light models" example used to verify the
# stack works on a fresh install. Mirror this structure in your own
# project configs.
#
# For a clean template see config/snake_config.template.yml.

project:
  project_dir: examples/test
  static_dir: config
  data_sources: config/deltares_data.yml
  data_sources_climate: config/cmip6_data.yml

shared:
  basin:
    region: "{'subbasin': [9.666, 0.4476], 'uparea': 100}"
    resolution: 0.00833
  historical_window:
    starttime: "2000-01-01T00:00:00"
    endtime: "2005-12-31T00:00:00"
  clim_historical: era5

workflows:
  model_creation:
    enabled: true
    model_build_config: config/wflow_build_model.yml
    waterbodies_config: config/wflow_update_waterbodies.yml
    wflow_outvars: ['river discharge']
    output_locations: None
    observations_timeseries: None

  climate_projections:
    enabled: true
    clim_project: cmip6
    models:
      - NOAA-GFDL/GFDL-ESM4
      - INM/INM-CM4-8
      - INM/INM-CM5-0
      - NIMS-KMA/KACE-1-0-G
      - NCC/NorESM2-MM
      - NCC/NorESM2-LM
      - CMCC/CMCC-CM2-SR5
      - CMCC/CMCC-ESM2
    scenarios: [ssp245, ssp585]
    members: [r1i1p1f1]
    variables: [precip, temp]
    start_month_hyd_year: Jan
    historical_year_range: [1980, 2010]
    future_horizons:
      near: [2030, 2060]
      far:  [2070, 2100]
    save_grids: false

  climate_experiment:
    enabled: true
    experiment_name: experiment
    realizations_num: 2
    horizontime_climate: 2050
    run_length: 20
    run_historical: false
    stress_test:
      temp:
        step_num: 1
        transient_change: true
        mean:
          min: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
          max: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
      precip:
        step_num: 2
        transient_change: true
        mean:
          min: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
          max: [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
        variance:
          min: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
          max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    Tlow: 2
    Tpeak: 10
    aggregate_rlz: true
```

> **Format change vs the flat canonical config.** Old `historical: 1980, 2010` (string) → `historical_year_range: [1980, 2010]` (list); old `future_horizons.near: 2030, 2060` (string) → `[2030, 2060]` (list); old `save_grids: FALSE` → `save_grids: false`. The numeric/basin/model values are copied verbatim.

### Step 2.3: Rewrite Snakefile_model_creation top-of-file (lines 1-44)

- [ ] Replace lines 1–44 of `Snakefile_model_creation` (the imports, config-path lookup, `get_config` def, and config-reading block) with this. The config path now comes from `workflow.configfiles[0]` (Snakemake's own record of the `--configfile` argument), replacing the `sys.argv` scan — but `config_path` is still forwarded to downstream R scripts, which is a hard repo convention:

```python
# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention — keep it even though the Snakefile itself uses `config`.
config_path = workflow.configfiles[0]

# Function to get argument from config file and return default value if not found.
# Required keys raise on missing; optional keys return default.
def get_config(config, arg, default=None, optional=True):
    """Read a config key with optional default; raises on missing required keys."""
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config")

# R01 schema — three top-level sections. Read each into a local dict.
project_cfg = config["project"]
shared_cfg = config["shared"]
my_cfg = config["workflows"]["model_creation"]

# Project — paths and external resources
project_dir = get_config(project_cfg, "project_dir", optional=False)
static_dir = get_config(project_cfg, "static_dir", optional=False)
DATA_SOURCES = get_config(project_cfg, "data_sources", optional=False)

# Shared — multi-workflow scientific knobs
model_region = get_config(shared_cfg["basin"], "region", optional=False)
model_resolution = get_config(shared_cfg["basin"], "resolution", 0.00833333)

# Workflow-owned
wflow_outvars = get_config(my_cfg, "wflow_outvars", ['river discharge'])
model_build_config = get_config(my_cfg, "model_build_config", f"{static_dir}/wflow_build_model.yml")
waterbodies_config = get_config(my_cfg, "waterbodies_config", f"{static_dir}/wflow_update_waterbodies.yml")
output_locations = get_config(my_cfg, "output_locations", None)
observations_timeseries = get_config(my_cfg, "observations_timeseries", None)

basin_dir = f"{project_dir}/hydrology_model"
```

> Note: the original top-of-file had `import sys`. It is no longer needed for the config-path lookup; drop it unless something else in the file uses `sys` (grep first — it does not today).

### Step 2.4: Update Snakefile_model_creation `setup_runtime` rule params

- [ ] In `Snakefile_model_creation`, find the `params:` block in `rule setup_runtime` (starttime/endtime/clim_source read flat keys):

```python
    params:
        starttime = get_config(config, "starttime", optional=False),
        endtime = get_config(config, "endtime", optional=False),
        clim_source = get_config(config, "clim_historical", optional=False),
        basin_dir = basin_dir,
```

- [ ] Replace the `params:` block with:

```python
    params:
        starttime = get_config(shared_cfg["historical_window"], "starttime", optional=False),
        endtime = get_config(shared_cfg["historical_window"], "endtime", optional=False),
        clim_source = get_config(shared_cfg, "clim_historical", optional=False),
        basin_dir = basin_dir,
```

### Step 2.5: Rewrite Snakefile_climate_projections top-of-file (lines 1-52)

- [ ] Replace lines 1–52 of `Snakefile_climate_projections` (through the `get_horizon` def) with:

```python
import itertools

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention.
config_path = workflow.configfiles[0]

# Function to get argument from config file and return default value if not found.
def get_config(config, arg, default=None, optional=True):
    """Read a config key with optional default; raises on missing required keys."""
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config")

# R01 schema
project_cfg = config["project"]
my_cfg = config["workflows"]["climate_projections"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
DATA_SOURCES = get_config(project_cfg, "data_sources_climate", optional=False)

clim_project = get_config(my_cfg, "clim_project", optional=False)
models = get_config(my_cfg, "models", optional=False)
scenarios = get_config(my_cfg, "scenarios", optional=False)
members = get_config(my_cfg, "members", optional=False)
variables = get_config(my_cfg, "variables", optional=False)

start_month_hyd_year = get_config(my_cfg, "start_month_hyd_year", "Jan")
time_horizon_hist = get_config(my_cfg, "historical_year_range", optional=False)
future_horizons = get_config(my_cfg, "future_horizons", optional=False)

save_grids = get_config(my_cfg, "save_grids", False)

basin_dir = f"{project_dir}/hydrology_model"
clim_project_dir = f"{project_dir}/climate_projections/{clim_project}"

### Dictionary elements from the config based on wildcards
def get_horizon(wildcards):
    return config["workflows"]["climate_projections"]["future_horizons"][wildcards.horizon]
```

> Note: the original top-of-file had `import itertools` and `import sys`. Keep `import itertools`; drop `import sys` (the config-path lookup no longer uses it — grep to confirm nothing else does). `get_horizon` is updated to navigate the new path.

### Step 2.6: Rewrite Snakefile_climate_experiment top-of-file (lines 1-53)

- [ ] Replace lines 1–53 of `Snakefile_climate_experiment` (through the `wflow_run_length` assignment) with:

```python
import numpy as np

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention.
config_path = workflow.configfiles[0]

# Function to get argument from config file and return default value if not found.
def get_config(config, arg, default=None, optional=True):
    """Read a config key with optional default; raises on missing required keys."""
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config")

# R01 schema
project_cfg = config["project"]
shared_cfg = config["shared"]
my_cfg = config["workflows"]["climate_experiment"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
static_dir = get_config(project_cfg, "static_dir", optional=False)
DATA_SOURCES = get_config(project_cfg, "data_sources", optional=False)

experiment = get_config(my_cfg, "experiment_name", optional=False)
RLZ_NUM = get_config(my_cfg, "realizations_num", 1)

# Stress test step counts live under stress_test.<temp|precip>.step_num
stress_test_cfg = my_cfg["stress_test"]
ST_NUM = (
    get_config(stress_test_cfg["temp"], "step_num", 1) + 1
) * (
    get_config(stress_test_cfg["precip"], "step_num", 1) + 1
)

run_hist = get_config(my_cfg, "run_historical", False)
ST_START = 0 if run_hist else 1

clim_source = get_config(shared_cfg, "clim_historical", optional=False)
horizontime_climate = get_config(my_cfg, "horizontime_climate", optional=False)
wflow_run_length = get_config(my_cfg, "run_length", 20)

basin_dir = f"{project_dir}/hydrology_model"
exp_dir = f"{project_dir}/climate_{experiment}"
```

> Note: the original top-of-file had `import sys` and `import numpy as np`. Keep `numpy`; drop `sys` (the config-path lookup no longer uses it — grep to confirm).

### Step 2.7: Update Snakefile_climate_experiment `export_wflow_results` rule params

- [ ] In `Snakefile_climate_experiment`, find the `params:` block in `rule export_wflow_results`:

```python
    params:
        exp_dir = exp_dir,
        aggr_rlz = get_config(config, 'aggregate_rlz', True),
        st_num = ST_NUM,
        Tlow = get_config(config,"Tlow", 2),
        Tpeak = get_config(config,"Tpeak", 10),
```

- [ ] Replace with:

```python
    params:
        exp_dir = exp_dir,
        aggr_rlz = get_config(my_cfg, "aggregate_rlz", True),
        st_num = ST_NUM,
        Tlow = get_config(my_cfg, "Tlow", 2),
        Tpeak = get_config(my_cfg, "Tpeak", 10),
```

### Step 2.8: Update tests/conftest.py fixtures

The fixtures `project_dir`, `data_sources`, and `model_build_config` in `tests/conftest.py` read config keys directly via `get_config(config, ...)`. They break after the config is sectioned unless updated.

- [ ] In `tests/conftest.py`, find the three fixtures:

```python
@pytest.fixture()
def project_dir(config):
    """Return project directory"""
    project_dir = get_config(config, "project_dir", optional=False)
    project_dir = join(SNAKEDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config):
    """Return data sources"""
    data_sources = get_config(config, "data_sources", optional=False)
    data_sources = join(SNAKEDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config):
    """Return model build config"""
    model_build_config = get_config(config, "model_build_config", optional=False)
    model_build_config = join(SNAKEDIR, model_build_config)
    return model_build_config
```

- [ ] Replace with:

```python
@pytest.fixture()
def project_dir(config):
    """Return project directory"""
    project_dir = get_config(config["project"], "project_dir", optional=False)
    project_dir = join(SNAKEDIR, project_dir)
    return project_dir


@pytest.fixture()
def data_sources(config):
    """Return data sources"""
    data_sources = get_config(config["project"], "data_sources", optional=False)
    data_sources = join(SNAKEDIR, data_sources)
    return data_sources


@pytest.fixture()
def model_build_config(config):
    """Return model build config"""
    model_build_config = get_config(
        config["workflows"]["model_creation"], "model_build_config", optional=False
    )
    model_build_config = join(SNAKEDIR, model_build_config)
    return model_build_config
```

### Step 2.9: Update tests/test_workflow_model_creation.py

`_catalog_root()` reads `cfg["data_sources"]` from the canonical config (`tests/test_workflow_model_creation.py:48-51`). Under R01 that moves to `cfg["project"]["data_sources"]`.

- [ ] In `tests/test_workflow_model_creation.py`, find:

```python
    with open(join(SNAKEDIR, CONFIG)) as f:
        cfg = yaml.safe_load(f)
    with open(join(SNAKEDIR, cfg["data_sources"])) as f:
        cat = yaml.safe_load(f)
```

- [ ] Replace with:

```python
    with open(join(SNAKEDIR, CONFIG)) as f:
        cfg = yaml.safe_load(f)
    # R01 sectioned schema: data_sources lives under project.
    with open(join(SNAKEDIR, cfg["project"]["data_sources"])) as f:
        cat = yaml.safe_load(f)
```

### Step 2.10: Update tests/test_workflow_climate_projections.py

The `_cfg(key)` helper returns `yaml.safe_load(...)[key]` — a single top-level lookup. Under R01, `project_dir` and `clim_project` are nested, so this is a real rewrite (not a key swap): the helper must navigate section paths.

- [ ] In `tests/test_workflow_climate_projections.py`, find:

```python
def _cfg(key):
    # Read lazily (inside the test), never at import time: the R1 config-schema
    # migration must not be able to break collection of the whole suite through
    # a module-level read here.
    with open(join(SNAKEDIR, CONFIG)) as f:
        return yaml.safe_load(f)[key]
```

- [ ] Replace with a path-navigating helper and update its two call sites:

```python
# R01 sectioned schema: project_dir lives under `project`, clim_project under
# `workflows.climate_projections`. Read lazily (inside the test), never at
# import time, so the migration cannot break collection of the whole suite.
_CFG_PATHS = {
    "project_dir": ("project", "project_dir"),
    "clim_project": ("workflows", "climate_projections", "clim_project"),
}


def _cfg(key):
    with open(join(SNAKEDIR, CONFIG)) as f:
        cfg = yaml.safe_load(f)
    node = cfg
    for part in _CFG_PATHS[key]:
        node = node[part]
    return node
```

The two call sites (`_cfg("project_dir")`, `_cfg("clim_project")`) are unchanged.

### Step 2.11: Update src/prepare_cst_parameters.py

The script opens `config_fn` and reads `yml["temp"]` / `yml["precip"]`. Under R01 those live under `yml["workflows"]["climate_experiment"]["stress_test"]`.

- [ ] In `src/prepare_cst_parameters.py`, find:

```python
    # Read the yaml config
    with open(config_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)

    # Temperature change attributes
    delta_temp_mean_min = yml["temp"]["mean"]["min"]
    delta_temp_mean_max = yml["temp"]["mean"]["max"]
    temp_step_num = yml["temp"]["step_num"] + 1

    # Precip change attributes
    delta_precip_mean_min = yml["precip"]["mean"]["min"]
    delta_precip_mean_max = yml["precip"]["mean"]["max"]
    delta_precip_variance_min = yml["precip"]["variance"]["min"]
    delta_precip_variance_max = yml["precip"]["variance"]["min"]
    precip_step_num = yml["precip"]["step_num"] + 1
```

- [ ] Replace with:

```python
    # Read the yaml config (R01 sectioned schema)
    with open(config_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)

    stress_test_cfg = yml["workflows"]["climate_experiment"]["stress_test"]

    # Temperature change attributes
    delta_temp_mean_min = stress_test_cfg["temp"]["mean"]["min"]
    delta_temp_mean_max = stress_test_cfg["temp"]["mean"]["max"]
    temp_step_num = stress_test_cfg["temp"]["step_num"] + 1

    # Precip change attributes
    delta_precip_mean_min = stress_test_cfg["precip"]["mean"]["min"]
    delta_precip_mean_max = stress_test_cfg["precip"]["mean"]["max"]
    delta_precip_variance_min = stress_test_cfg["precip"]["variance"]["min"]
    delta_precip_variance_max = stress_test_cfg["precip"]["variance"]["min"]
    precip_step_num = stress_test_cfg["precip"]["step_num"] + 1
```

Note: the `delta_precip_variance_max = ...["variance"]["min"]` line reads `min` for the max — preserved verbatim because it is an existing pre-R01 bug, not within R01's scope to fix.

Note: this script's `__main__` fallback (`config/snake_config_model_test.yml`) is exercised only when the file is run standalone. That config is migrated in this same commit, so the fallback stays consistent.

### Step 2.12: Update src/prepare_weagen_config.py

The script reads `yml_snake["realizations_num"]`, `yml_snake["temp"]`, and `yml_snake["precip"]` from the snake config. All three move under sectioned paths.

- [ ] In `src/prepare_weagen_config.py`, find the two `yml_dict` construction blocks:

```python
    # arguments from the default weagen config file
    yml_dict = read_yml(snakemake.params.default_config)
    # add new arguments from snakemake and yml_snake
    yml_add = {
        "output.path": snakemake.params.output_path,
        "sim.year.start": 2010,
        "sim.year.num": nr_years_weagen,
        "nc.file.prefix": snakemake.params.nc_file_prefix,
        "realizations_num": yml_snake["realizations_num"],
    }
    for k, v in yml_add.items():
        yml_dict["generateWeatherSeries"][k] = v

else:  # stress test
    # new arguments
    yml_dict = {
        "imposeClimateChanges": {
            "output.path": snakemake.params.output_path,
            "nc.file.prefix": snakemake.params.nc_file_prefix,
            "nc.file.suffix": snakemake.params.nc_file_suffix,
        }
    }
    # arguments from yml_snake
    yml_dict["temp"] = yml_snake["temp"]
    yml_dict["precip"] = yml_snake["precip"]
```

- [ ] Replace with:

```python
    # arguments from the default weagen config file
    yml_dict = read_yml(snakemake.params.default_config)
    # add new arguments from snakemake and yml_snake (R01 sectioned schema)
    experiment_cfg = yml_snake["workflows"]["climate_experiment"]
    yml_add = {
        "output.path": snakemake.params.output_path,
        "sim.year.start": 2010,
        "sim.year.num": nr_years_weagen,
        "nc.file.prefix": snakemake.params.nc_file_prefix,
        "realizations_num": experiment_cfg["realizations_num"],
    }
    for k, v in yml_add.items():
        yml_dict["generateWeatherSeries"][k] = v

else:  # stress test
    # new arguments
    yml_dict = {
        "imposeClimateChanges": {
            "output.path": snakemake.params.output_path,
            "nc.file.prefix": snakemake.params.nc_file_prefix,
            "nc.file.suffix": snakemake.params.nc_file_suffix,
        }
    }
    # arguments from yml_snake (R01 sectioned schema)
    stress_test_cfg = yml_snake["workflows"]["climate_experiment"]["stress_test"]
    yml_dict["temp"] = stress_test_cfg["temp"]
    yml_dict["precip"] = stress_test_cfg["precip"]
```

### Step 2.13: Update src/get_change_climate_proj.py

The script receives `time_horizon_hist` and `time_horizon_fut` as Snakemake params and calls `.split(", ")` on them. After R01 those values are lists (`[1980, 2010]`), not comma-separated strings (`"1980, 2010"`). The `.split` call raises `AttributeError` on a list. This covers **both** the historical window and the future horizons (the tracked config's `future_horizons.near/.far` were also `"YYYY, YYYY"` strings and become lists).

- [ ] In `src/get_change_climate_proj.py`, find:

```python
# Time tuples for comparison hist-fut
time_tuple_hist = snakemake.params.time_horizon_hist
time_tuple_hist = tuple(map(str, time_tuple_hist.split(", ")))
time_tuple_fut = snakemake.params.time_horizon_fut
time_tuple_fut = tuple(map(str, time_tuple_fut.split(", ")))
```

- [ ] Replace with:

```python
# Time tuples for comparison hist-fut.
# R01 schema delivers these as lists ([1980, 2010]) for both the historical
# window and the future horizons. Pre-R01 configs delivered them as
# comma-separated strings ("1980, 2010"). Accept both.
def _to_str_tuple(value):
    if isinstance(value, str):
        return tuple(map(str, value.split(", ")))
    return tuple(map(str, value))

time_tuple_hist = _to_str_tuple(snakemake.params.time_horizon_hist)
time_tuple_fut = _to_str_tuple(snakemake.params.time_horizon_fut)
```

### Step 2.13b: Add focused tests for the sectioned reader + horizon normalization

Neither the sectioned stress-test reader nor the list/string horizon normalization is exercised by the default suite: `snakemake --dry-run` does not execute `script:` bodies, and both integration tests are skip-by-default. Add focused unit tests so this logic has coverage before the Task 5 gated run (review §4 item 10).

The sectioned reader `prep_cst_parameters(config_fn, csv_fns)` in `src/prepare_cst_parameters.py` is importable (its `snakemake` reference is guarded in the `__main__`/`sm` block), so it can be driven directly against the migrated tests config. The normalization helper `_to_str_tuple` lives inside `get_change_climate_proj.py` (which executes Snakemake-global code at import and is not importable without restructuring — that restructuring is R4's job, out of R01 scope), so its **contract** is asserted directly here; the production copy is validated end-to-end in Task 5.

- [ ] Create `tests/test_r01_config_readers.py`:

```python
"""Focused R01 tests: the sectioned stress-test reader and the
list/string horizon normalization contract. These cover logic that
dry-runs and skip-by-default integration tests do not reach.
"""
import sys
from os.path import join, dirname, realpath

import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")
sys.path.insert(0, join(SNAKEDIR, "src"))

CONFIG = join(TESTDIR, "snake_config_model_test.yml")


def test_prep_cst_parameters_reads_sectioned_config(tmp_path):
    """prep_cst_parameters must read stress_test from the sectioned schema."""
    from prepare_cst_parameters import prep_cst_parameters

    # temp.step_num=1, precip.step_num=1 in the tests config -> ST_NUM = 2*2 = 4.
    csv_fns = [str(tmp_path / f"cst_{i}.csv") for i in range(4)]
    prep_cst_parameters(config_fn=CONFIG, csv_fns=csv_fns)
    for fn in csv_fns:
        assert __import__("os").path.exists(fn), f"expected {fn} written"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2000, 2010", ("2000", "2010")),   # legacy comma-separated string
        ([2000, 2010], ("2000", "2010")),   # R01 list form
        ([2030, 2060], ("2030", "2060")),
    ],
)
def test_horizon_normalization_contract(value, expected):
    """The list/string normalization used in get_change_climate_proj.py.

    Kept in lockstep with the production _to_str_tuple; see that module.
    """
    def _to_str_tuple(v):
        if isinstance(v, str):
            return tuple(map(str, v.split(", ")))
        return tuple(map(str, v))

    assert _to_str_tuple(value) == expected
```

> The `test_prep_cst_parameters_reads_sectioned_config` test would raise `KeyError("temp")` against the pre-R01 flat config and passes only after the sectioned migration — so it also guards the reader against regressions. If `prep_cst_parameters` needs its output directory to exist, pass `tmp_path`-based paths (as above) so no repo state is written.

- [ ] These two tests (one function + one 3-case parametrize = 4 test items) raise the suite from the pre-R01 `47 passed` (Step 0.2's baseline) to `51 passed, 2 skipped, 2 xfailed`. The post-migration counts in Steps 2.17 and 5.2 and the Step 5.10 status line reflect `51`; Step 0.2 stays `47` (it measures the pre-R01 state). Note in the seal that R01 adds focused coverage, so the pre-R01 `47` from M2c is intentionally superseded — see the owner note in the handoff.

### Step 2.14: Verify Python syntax of the migrated src/ scripts

- [ ] Run: `pixi run python -c "import ast; [ast.parse(open(p).read()) for p in ['src/prepare_cst_parameters.py', 'src/prepare_weagen_config.py', 'src/get_change_climate_proj.py']]"`
- [ ] Expect: no output. (These are ordinary Python files, so `ast.parse` is valid here. Do **not** `ast.parse` the Snakefiles — Snakemake grammar such as `rule all:` is not valid Python; the Snakefiles are validated by dry-run in Step 2.16 instead.)

### Step 2.15: Value-preservation check — old flat leaves map to new paths (both committed configs)

The migration must not silently change any scientific value. For **both** `tests/snake_config_model_test.yml` and `config/snake_config_model_test.yml`, assert that every old flat leaf equals the value now at its new sectioned path. Use `git show HEAD:<path>` to read the pre-migration flat version.

- [ ] Run (Git Bash):
```
pixi run python - <<'PY'
import subprocess, yaml, sys

# old flat key -> new sectioned path (tuple of keys)
MAP = {
    "project_dir": ("project", "project_dir"),
    "static_dir": ("project", "static_dir"),
    "data_sources": ("project", "data_sources"),
    "data_sources_climate": ("project", "data_sources_climate"),
    "model_region": ("shared", "basin", "region"),
    "model_resolution": ("shared", "basin", "resolution"),
    "starttime": ("shared", "historical_window", "starttime"),
    "endtime": ("shared", "historical_window", "endtime"),
    "clim_historical": ("shared", "clim_historical"),
    "wflow_outvars": ("workflows", "model_creation", "wflow_outvars"),
    "model_build_config": ("workflows", "model_creation", "model_build_config"),
    "waterbodies_config": ("workflows", "model_creation", "waterbodies_config"),
    "output_locations": ("workflows", "model_creation", "output_locations"),
    "observations_timeseries": ("workflows", "model_creation", "observations_timeseries"),
    "clim_project": ("workflows", "climate_projections", "clim_project"),
    "models": ("workflows", "climate_projections", "models"),
    "scenarios": ("workflows", "climate_projections", "scenarios"),
    "members": ("workflows", "climate_projections", "members"),
    "variables": ("workflows", "climate_projections", "variables"),
    "start_month_hyd_year": ("workflows", "climate_projections", "start_month_hyd_year"),
    "save_grids": ("workflows", "climate_projections", "save_grids"),
    "experiment_name": ("workflows", "climate_experiment", "experiment_name"),
    "realizations_num": ("workflows", "climate_experiment", "realizations_num"),
    "horizontime_climate": ("workflows", "climate_experiment", "horizontime_climate"),
    "run_length": ("workflows", "climate_experiment", "run_length"),
    "run_historical": ("workflows", "climate_experiment", "run_historical"),
    "temp": ("workflows", "climate_experiment", "stress_test", "temp"),
    "precip": ("workflows", "climate_experiment", "stress_test", "precip"),
    "Tlow": ("workflows", "climate_experiment", "Tlow"),
    "Tpeak": ("workflows", "climate_experiment", "Tpeak"),
    "aggregate_rlz": ("workflows", "climate_experiment", "aggregate_rlz"),
}
# keys whose *type/representation* intentionally changes (string -> list).
# We still assert the underlying values survive the retype (a "2000, 2011"
# transcription slip must not pass) by parsing the old comma-string.
REPR_CHANGED = {"historical": ("workflows", "climate_projections", "historical_year_range"),
                "future_horizons": ("workflows", "climate_projections", "future_horizons")}

def get(node, path):
    for p in path:
        node = node[p]
    return node

def _ints(s):
    # "2000, 2010" (or "2000,2010") -> [2000, 2010]; passthrough for lists.
    if isinstance(s, str):
        return [int(x) for x in s.replace(" ", "").split(",")]
    return [int(x) for x in s]

def check(path):
    old = yaml.safe_load(subprocess.run(["git", "show", f"HEAD:{path}"],
                                        capture_output=True, text=True).stdout)
    new = yaml.safe_load(open(path))
    bad = []
    for k, np_ in MAP.items():
        if k not in old:
            continue  # not all keys present in every config (e.g. Linux)
        try:
            nv = get(new, np_)
        except (KeyError, TypeError):
            bad.append(f"{path}: {k} -> missing at new path {np_}")
            continue
        if old[k] != nv:
            bad.append(f"{path}: {k}={old[k]!r} != new {'.'.join(np_)}={nv!r}")
    for k, np_ in REPR_CHANGED.items():
        if k not in old:
            continue
        if k == "future_horizons":
            # dict of horizon -> "y, y"; compare per horizon.
            try:
                nv = get(new, np_)
            except (KeyError, TypeError):
                bad.append(f"{path}: {k} -> missing at new path {np_}")
                continue
            for h, ov in old[k].items():
                if h not in nv or _ints(ov) != _ints(nv[h]):
                    bad.append(f"{path}: future_horizons.{h}={ov!r} != new {nv.get(h)!r}")
        else:  # historical -> historical_year_range
            try:
                nv = get(new, np_)
            except (KeyError, TypeError):
                bad.append(f"{path}: {k} -> missing at new path {np_}")
                continue
            if _ints(old[k]) != _ints(nv):
                bad.append(f"{path}: {k}={old[k]!r} != new {'.'.join(np_)}={nv!r}")
    return bad

problems = []
for p in ("tests/snake_config_model_test.yml", "config/snake_config_model_test.yml"):
    problems += check(p)
if problems:
    print("VALUE-PRESERVATION FAILURES:")
    print("\n".join(problems))
    sys.exit(1)
print("OK — every migrated leaf preserves its old value; repr-changed keys present.")
PY
```
- [ ] Expect: `OK — every migrated leaf preserves its old value; repr-changed keys present.` Any failure means a value was dropped or mistyped during the rewrite — fix before proceeding.

### Step 2.16: Verify each Snakefile dry-runs against the new tests config

- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_model_creation --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -10`
- [ ] Expect: `Job stats:` table + dry-run summary, no Python errors.
- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_climate_projections --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -10`
- [ ] Expect: `MissingInputException` for region.geojson (existing known-failure behavior, NOT a Python error). If you see `KeyError`, a config key is wrong.
- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_climate_experiment --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -10`
- [ ] Expect: `CyclicGraphException` (existing known-failure behavior, NOT a Python error).

### Step 2.17: Run full pytest suite

- [ ] Run: `pixi run pytest tests/ 2>&1 | tail -3`
- [ ] Expect: `51 passed, 2 skipped, 2 xfailed` — `47` from the pre-R01 baseline plus the 4 new focused-test items added in Step 2.13b (1 reader test + 3 parametrized normalization cases). (See the owner note in the handoff about superseding the pre-R01 `47`.)
- [ ] If the *passed* count is not `51` beyond the 4 added items, or any previously-passing test now fails: a config key migration is wrong. Diff the failures and trace back to the migration mapping in `dev/milestones/r01/modularity-contracts-design.md`.

### Step 2.18: Commit the atomic migration

- [ ] Run:
```
git add tests/snake_config_model_test.yml config/snake_config_model_test.yml tests/conftest.py tests/test_workflow_model_creation.py tests/test_workflow_climate_projections.py tests/test_r01_config_readers.py Snakefile_model_creation Snakefile_climate_projections Snakefile_climate_experiment src/prepare_cst_parameters.py src/prepare_weagen_config.py src/get_change_climate_proj.py
git commit -m "$(cat <<'EOF'
r01: migrate configs + Snakefiles + src readers + tests to sectioned schema (atomic)

Atomic migration to the R01 project / shared / workflows.<name> schema.
Every layer that parses or reads the config flips in lockstep so no
committed state parses at the Snakefile layer but fails in a script or
test:

- tests/snake_config_model_test.yml + config/snake_config_model_test.yml
  -> sectioned schema. The canonical config is here (not a later commit)
  because both integration tests hardcode it and read keys from it.
- Snakefiles (all 3): read own section; config path now via
  workflow.configfiles[0] (replaces the sys.argv scan). config_path is
  still forwarded to downstream R scripts (repo convention).
- src/ direct readers: prepare_cst_parameters.py, prepare_weagen_config.py
  (yml["temp"]/["precip"]/["realizations_num"] -> workflows.climate_
  experiment paths); get_change_climate_proj.py (time horizons now arrive
  as lists, not "y, y" strings — accept both via _to_str_tuple).
- tests/conftest.py fixtures + both integration tests (data_sources,
  project_dir, clim_project) follow the new key paths.

Why atomic: several keys (project_dir, data_sources, clim_historical) are
read by multiple consumers; a staged migration would need dual-key
fallback logic. One commit is simpler and easier to revert.

Verification:
- value-preservation check: every migrated leaf equals its old value
- ast.parse on the 3 migrated src/ scripts -> clean
- snakemake --dry-run on each Snakefile -> expected (1 pass, 2 known-fail)
- pytest tests/ -> 51 passed, 2 skipped, 2 xfailed (47 pre-R01 + 4 new focused tests)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2b: Mechanical flat-key audit

**Purpose:** Catch any old flat-key references — direct or indirect — that escaped the atomic migration. The audit patterns must catch not only `config["project_dir"]`-style direct reads but indirect readers via accessor helpers (`_cfg("...")`, `cfg["..."]`, `yml_snake["..."]`).

**Expected result (per the accepted review):** no additional production consumers exist. The R scripts under `src/weathergen/` read the *derived* weathergenr YAML (not the snake config), and `src/setup_reservoirs_lakes_glaciers.py` reads its own HydroMT update config. So the audit should surface only false positives (Python locals named `temp`/`precip`, comments) — trace each to confirm.

### Step 2b.1: Grep for legacy flat-key access patterns (direct and indirect)

- [ ] Run (Git Bash):
```
pixi run python - <<'PY'
import subprocess
FLAT = "project_dir|data_sources|data_sources_climate|model_region|model_resolution|starttime|endtime|historical|clim_historical|wflow_outvars|experiment_name|realizations_num|future_horizons|models|scenarios|members|variables|save_grids|horizontime_climate|run_length|run_historical|aggregate_rlz|Tlow|Tpeak"
patterns = [
    # direct subscription on the snake config / parsed yaml dicts
    rf'\bconfig\[["\'](?:{FLAT})["\']\]',
    rf'\byml\[["\'](?:temp|precip)["\']\]',
    rf'\byml_snake\[["\'](?:temp|precip|realizations_num)["\']\]',
    # get_config against the flat namespace
    rf'get_config\(\s*config\s*,\s*["\'](?:{FLAT})["\']',
    # indirect accessor helpers that take a single flat key
    rf'_cfg\(\s*["\'](?:{FLAT})["\']\s*\)',
    rf'\bcfg\[["\'](?:{FLAT})["\']\]',
]
roots = ['Snakefile_model_creation', 'Snakefile_climate_projections',
         'Snakefile_climate_experiment', 'src/', 'tests/', 'dev/scripts/']
for p in patterns:
    r = subprocess.run(['rg', '-n', '--pcre2', p, *roots], capture_output=True, text=True)
    if r.stdout:
        print(f'PATTERN: {p}')
        print(r.stdout)
PY
```
- [ ] Expect: no hits, or only false positives (a Python local named `temp`/`precip`, a comment, the `get_config`/`_cfg` *definitions* themselves). Trace each hit to confirm it is not a surviving flat read.
- [ ] Note in the tracker. No commit if there are no real hits. If a genuine straggler is found, fix it in a small commit `r01: fix straggler flat-key read (audit)`.

---

## Task 3: Migrate config/snake_config_model_test_linux.yml

**Files:**
- Modify: `config/snake_config_model_test_linux.yml`

**Purpose:** Linux variant of the canonical example. **Value-preserving restructure only** — the Linux config differs from the Windows canonical in `project_dir` (`examples/Gabon`), historical `endtime` (`2020-12-31` vs `2005-12-31`), `model_resolution` (`0.0062475` vs `0.00833`), observation paths (set, not `None`), the CMIP6 model list, `data_sources` (`config/deltares_data_linux.yml`), and it has **no** `wflow_outvars` and **no** `save_grids` keys. Restructure the parsed Linux mapping in place; never copy a Windows value in.

**Note:** Linux validation is deferred per `dev/roadmap.md` "Deferred: Linux replication". This file must continue to *parse cleanly* but is not end-to-end validated in R01.

### Step 3.1: Read the current Linux config and enumerate its leaves

- [ ] Run: `cat config/snake_config_model_test_linux.yml`
- [ ] Confirm the differing values above are read directly from the file — do not assume they mirror the Windows config.

### Step 3.2: Rewrite by restructuring in place (do NOT copy Windows values)

- [ ] Move each existing Linux leaf into its sectioned location, preserving the exact value. Concretely:
  - `project_dir: examples/Gabon` → `project.project_dir` (Linux value, NOT `examples/test`).
  - `static_dir: config` → `project.static_dir`.
  - `data_sources: config/deltares_data_linux.yml` → `project.data_sources` (Linux catalog).
  - `data_sources_climate: config/cmip6_data.yml` → `project.data_sources_climate`.
  - `model_region` → `shared.basin.region`; `model_resolution: 0.0062475` → `shared.basin.resolution`.
  - `starttime`/`endtime` (`2000-01-01` / `2020-12-31`) → `shared.historical_window.*`.
  - `clim_historical: era5` → `shared.clim_historical`.
  - `model_build_config`: **absent** in the Linux file today, so do NOT add it (the Snakefile default applies). Only add `workflows.model_creation.model_build_config` if the source file had it.
  - `wflow_outvars`: **absent** — do NOT add it (Snakefile default `['river discharge']` applies). Leaving it out preserves current behavior.
  - `output_locations` / `observations_timeseries`: the set paths (`data/observations/...`) → `workflows.model_creation.*` verbatim.
  - climate_projections keys (`clim_project`, `models` — the Linux 3-model list, `scenarios`, `members`, `variables`, `start_month_hyd_year`) → `workflows.climate_projections.*`.
  - `historical: 1980, 2010` → `workflows.climate_projections.historical_year_range: [1980, 2010]` (string → list, same values).
  - `future_horizons.near/.far: YYYY, YYYY` → lists `[YYYY, YYYY]` (same values).
  - `save_grids`: **absent** — do NOT add it (Snakefile default `False` applies).
  - climate_experiment keys (`experiment_name`, `realizations_num`, `horizontime_climate`, `run_length`, `run_historical`, `temp`/`precip` → `stress_test.*`, `Tlow`, `Tpeak`, `aggregate_rlz`) → `workflows.climate_experiment.*`.
  - Add `enabled: true` to each of the three workflow sections (new key, no old equivalent).

### Step 3.3: Value-preservation check for the Linux config

- [ ] Run (Git Bash): the same check script as Step 2.15, but pointed at the Linux config. Re-run the Step 2.15 heredoc with the final loop replaced by:
```
for p in ("config/snake_config_model_test_linux.yml",):
    problems += check(p)
```
- [ ] Expect: `OK — ...`. The `check` function skips keys absent from the old file, so absent `wflow_outvars` / `save_grids` / `model_build_config` are correctly ignored.

### Step 3.4: Verify YAML parses

- [ ] Run: `pixi run python -c "import yaml; yaml.safe_load(open('config/snake_config_model_test_linux.yml'))"`
- [ ] Expect: clean parse.

### Step 3.5: Commit

- [ ] Run:
```
git add config/snake_config_model_test_linux.yml
git commit -m "$(cat <<'EOF'
r01: migrate Linux example config to sectioned schema

Value-preserving restructure of the Linux variant: every Linux-specific
leaf (examples/Gabon, endtime 2020-12-31, resolution 0.0062475, the
deltares_data_linux catalog, the 3-model list, the set observation
paths) moves into its new section unchanged. Keys absent in the Linux
file (wflow_outvars, save_grids, model_build_config) are left absent so
the Snakefile defaults still apply. Linux end-to-end validation stays
deferred per the roadmap; this file just needs to parse cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add user-facing migration guide for _local.yml configs

**Files:**
- Create: `dev/milestones/r01/local-config-migration.md`

**Purpose:** `_local.yml` configs are gitignored per-machine. The user (or any contributor with their own local config) needs a documented mapping to migrate manually — including the one used to seed the `examples/test_local` baseline root in Task 5.

### Step 4.1: Create the migration guide

- [ ] Write `dev/milestones/r01/local-config-migration.md`:

```markdown
# Migrating a local snake config to R01 sectioned schema

If you have a `*_local.yml` config (gitignored, per-machine), it still
uses the old flat schema after R01 lands. Migrate it manually using
the mapping below. The Snakefiles read sectioned config only after
R01, so old configs will fail with `KeyError`.

## Mapping table

| Old (flat) key                    | New (sectioned) location                                     |
| --------------------------------- | ------------------------------------------------------------ |
| `project_dir`                     | `project.project_dir`                                        |
| `static_dir`                      | `project.static_dir`                                         |
| `data_sources`                    | `project.data_sources`                                       |
| `data_sources_climate`            | `project.data_sources_climate`                               |
| `model_region`                    | `shared.basin.region`                                        |
| `model_resolution`                | `shared.basin.resolution`                                    |
| `starttime` (ISO datetime)        | `shared.historical_window.starttime`                         |
| `endtime` (ISO datetime)          | `shared.historical_window.endtime`                           |
| `clim_historical`                 | `shared.clim_historical`                                     |
| `wflow_outvars`                   | `workflows.model_creation.wflow_outvars`                     |
| `model_build_config`              | `workflows.model_creation.model_build_config`                |
| `waterbodies_config`              | `workflows.model_creation.waterbodies_config`                |
| `output_locations`                | `workflows.model_creation.output_locations`                  |
| `observations_timeseries`         | `workflows.model_creation.observations_timeseries`           |
| `clim_project`                    | `workflows.climate_projections.clim_project`                 |
| `models`                          | `workflows.climate_projections.models`                       |
| `scenarios`                       | `workflows.climate_projections.scenarios`                    |
| `members`                         | `workflows.climate_projections.members`                      |
| `variables`                       | `workflows.climate_projections.variables`                    |
| `start_month_hyd_year`            | `workflows.climate_projections.start_month_hyd_year`         |
| `historical: 1980, 2010` (str)    | `workflows.climate_projections.historical_year_range: [1980, 2010]` (list) |
| `future_horizons.<h>: y, y` (str) | `workflows.climate_projections.future_horizons.<h>: [y, y]` (list) |
| `save_grids`                      | `workflows.climate_projections.save_grids`                   |
| `experiment_name`                 | `workflows.climate_experiment.experiment_name`               |
| `realizations_num`                | `workflows.climate_experiment.realizations_num`              |
| `horizontime_climate`             | `workflows.climate_experiment.horizontime_climate`           |
| `run_length`                      | `workflows.climate_experiment.run_length`                    |
| `run_historical`                  | `workflows.climate_experiment.run_historical`                |
| `temp.*`                          | `workflows.climate_experiment.stress_test.temp.*`            |
| `precip.*`                        | `workflows.climate_experiment.stress_test.precip.*`          |
| `Tlow`                            | `workflows.climate_experiment.Tlow`                          |
| `Tpeak`                           | `workflows.climate_experiment.Tpeak`                         |
| `aggregate_rlz`                   | `workflows.climate_experiment.aggregate_rlz`                 |

## New keys (no old equivalent)

For each workflow, add `enabled: true` (forward-compat marker;
documentary in R01). Default `true` for all three workflows.

## Format changes: string → list

Two projection keys change type, not just location:

- `historical: 1980, 2010` (a comma-separated string) →
  `historical_year_range: [1980, 2010]` (a YAML list of two integers).
- `future_horizons.<horizon>: 2030, 2060` (string) →
  `future_horizons.<horizon>: [2030, 2060]` (list).

The Snakefiles pass these through to `src/get_change_climate_proj.py`,
which now accepts either form, but new configs should use lists.

## Verification

After migrating your local config, dry-run all 3 Snakefiles to catch
typos:

```bash
pixi run snakemake all -c 1 -s Snakefile_model_creation     --configfile <your_local_config.yml> --dry-run
pixi run snakemake all -c 1 -s Snakefile_climate_projections --configfile <your_local_config.yml> --dry-run
pixi run snakemake all -c 1 -s Snakefile_climate_experiment  --configfile <your_local_config.yml> --dry-run
```

Any `KeyError` in the dry-run output means a key wasn't migrated —
trace it via the table above.

## Reference

For a clean sectioned config, see:
- `config/snake_config.template.yml` (annotated template)
- `config/snake_config_model_test.yml` (working example, tiny bbox)
- `tests/snake_config_model_test.yml` (test fixture)
```

### Step 4.2: Commit

- [ ] Run:
```
git add dev/milestones/r01/local-config-migration.md
git commit -m "$(cat <<'EOF'
r01: add migration guide for user-local configs

Per-machine *_local.yml configs are gitignored and must be migrated
manually. dev/milestones/r01/local-config-migration.md provides the full old->new
key mapping table plus the two string->list format changes
(historical_year_range and future_horizons).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification, seal, and tag

**Files:**
- Create: `dev/milestones/r01/baseline_diffs.md` (documents expected config-snapshot drift)
- Modify: `dev/baseline/manifest.json` (re-recorded with new config-yaml hashes)
- Modify: `dev/roadmap.md` (status line + R3 configfile note)

**Purpose:** Three independent verifications, then a candidate-manifest re-baseline that proves only the documented config snapshots changed, then seal + tag.

**Baseline root: `examples/test_local`.** This is the root the recorded manifest (`dev/baseline/manifest.json`) uses. No tracked config sets it (the M2b manifest was recorded from a *local, untracked* copy of the canonical config). For Task 5 you make a fresh untracked local copy of the **migrated canonical** config (`config/snake_config_model_test.yml`) with `project_dir` overridden to `examples/test_local`, and drive all three workflows and both `check_baseline.py` invocations against that single root. Do NOT use `tests/test_project` as the baseline root — that is the pytest fixture's dry-run root, not the manifest's root.

**Critical context:** `dev/scripts/check_baseline.py` includes the copied snake-config YAMLs as manifest targets (`TARGETS` entries at lines 58, 66, 70 — `{project_dir}/config/snake_config_<workflow>.yml`). After R01 those YAML fingerprints change because the schema changed. This is **expected organizational drift, not scientific drift** — the netCDF / CSV / PNG fingerprints must remain identical. Document in `dev/milestones/r01/baseline_diffs.md` and re-record the manifest.

**All three workflows are mandatory.** `check_baseline.py record` refuses an incomplete target set (`cmd_record` returns 1 on any missing target), and stale outputs would otherwise be fingerprinted as the new contract. If required data access (Deltares mirror, Julia, CMIP6/GCS) is unavailable so that any workflow cannot run freshly against `examples/test_local`, **R01 stays unsealed** — do not record a partial manifest.

### Step 5.1: Seed the local baseline config

- [ ] Make an untracked local copy of the migrated canonical config with the baseline `project_dir`:
```
cp config/snake_config_model_test.yml config/snake_config_model_test_local.yml
```
- [ ] Edit `config/snake_config_model_test_local.yml` so `project.project_dir: examples/test_local` (leave every other value identical to the canonical config). This file is gitignored (`*_local.yml`) — never commit it.
- [ ] Verify parse: `pixi run python -c "import yaml; yaml.safe_load(open('config/snake_config_model_test_local.yml'))"`

### Step 5.2: Verification A — pytest suite

- [ ] Run: `pixi run pytest tests/ -v 2>&1 | tail -10`
- [ ] Expect: `51 passed, 2 skipped, 2 xfailed` (the pre-R01 `47` plus the 4 focused-test items from Step 2.13b; the config restructure itself must not change any pre-existing test outcome).

### Step 5.3: Verification B — parse + dry-run for all three workflows

- [ ] Run (each separately, against the tests config):
```
pixi run snakemake all -c 1 -s Snakefile_model_creation     --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -5
pixi run snakemake all -c 1 -s Snakefile_climate_projections --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -5
pixi run snakemake all -c 1 -s Snakefile_climate_experiment  --configfile tests/snake_config_model_test.yml --dry-run 2>&1 | tail -5
```
- [ ] Expect: model_creation parses cleanly + dry-run summary; climate_projections fails with `MissingInputException` on region.geojson (known-failure ratchet); climate_experiment fails with `CyclicGraphException` (known-failure ratchet). **No `KeyError` or `AttributeError`** — those would mean a config migration is wrong.

### Step 5.4: Verification C — fresh full workflow runs against the baseline root

Run all three workflows end-to-end against the local baseline config, into `examples/test_local`. All three are mandatory.

- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_model_creation --configfile config/snake_config_model_test_local.yml 2>&1 | tail -20`
- [ ] Expect: `<N> of <N> steps (100%) done`. Check `examples/test_local/hydrology_model/staticmaps.nc` exists.
- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_climate_projections --configfile config/snake_config_model_test_local.yml --keep-going 2>&1 | tail -20` (needs CMIP6/GCS access).
- [ ] Run: `pixi run snakemake all -c 1 -s Snakefile_climate_experiment --configfile config/snake_config_model_test_local.yml 2>&1 | tail -20` (needs Julia + Wflow.jl).
- [ ] If any workflow cannot run freshly (missing data mirror, Julia, or GCS): **STOP — R01 stays unsealed.** Do not proceed to record; surface the blocker to the owner.

### Step 5.5: Verification C (cont.) — check against the existing manifest

- [ ] Run: `pixi run python dev/scripts/check_baseline.py check --project-dir examples/test_local 2>&1 | tail -30`
- [ ] **Expected output:** exactly three FAIL entries, all `sha256 ...` mismatches on the copied-config YAML targets:
  - `examples/test_local/config/snake_config_model_creation.yml`
  - `examples/test_local/config/snake_config_climate_projections.yml`
  - `examples/test_local/config/snake_config_climate_experiment.yml`

  Every scientific target (PNG / netCDF / CSV) must report **no diff**. If any scientific target diffs: a config migration changed numerical output. Most likely cause: a value retyped during the YAML rewrite changed precision, or a `historical_year_range` / `future_horizons` migration introduced an off-by-one. Trace via the failing artifact's variable name back to the script that produced it — do NOT proceed to record.

### Step 5.6: Record a CANDIDATE manifest and diff it against the canonical one

The candidate-then-compare flow uses `check_baseline.py`'s existing `--manifest` option (available on both `record` and `check` — no script extension needed). Record to a candidate path first, diff structurally against the canonical manifest, and only replace the canonical once the diff shows exactly the three expected YAML entries.

- [ ] Run: `pixi run python dev/scripts/check_baseline.py record --project-dir examples/test_local --manifest dev/baseline/manifest.candidate.json 2>&1 | tail -5`
- [ ] Structurally diff candidate vs canonical (Git Bash):
```
pixi run python - <<'PY'
import json, sys
old = json.load(open("dev/baseline/manifest.json"))
new = json.load(open("dev/baseline/manifest.candidate.json"))
ot, nt = old["targets"], new["targets"]
added = sorted(set(nt) - set(ot))
removed = sorted(set(ot) - set(nt))
changed = sorted(k for k in set(ot) & set(nt) if ot[k] != nt[k])
EXPECTED = {
    "examples/test_local/config/snake_config_model_creation.yml",
    "examples/test_local/config/snake_config_climate_projections.yml",
    "examples/test_local/config/snake_config_climate_experiment.yml",
}
print("added:", added)
print("removed:", removed)
print("changed:", changed)
ok = not added and not removed and set(changed) == EXPECTED
print("project_dir:", old.get("project_dir"), "->", new.get("project_dir"))
print("RESULT:", "OK — only the 3 config snapshots changed" if ok else "FAIL — unexpected manifest delta")
sys.exit(0 if ok else 1)
PY
```
- [ ] Expect: `RESULT: OK — only the 3 config snapshots changed`, with `added`/`removed` empty and `changed` equal to the three YAML entries. Both manifests are rooted at `examples/test_local`, so `project_dir` is unchanged and no keys are added/removed. If the delta is anything else, STOP and investigate — do not replace the canonical manifest.

### Step 5.7: Promote the candidate to the canonical manifest

- [ ] Replace the canonical manifest with the candidate and remove the candidate file:
```
mv dev/baseline/manifest.candidate.json dev/baseline/manifest.json
```
- [ ] Run: `git diff dev/baseline/manifest.json` and confirm the diff touches only the three `snake_config_<workflow>.yml` entries (their `sha256`), nothing else.

### Step 5.8: Post-rebaseline check — confirm clean

- [ ] Run: `pixi run python dev/scripts/check_baseline.py check --project-dir examples/test_local 2>&1 | tail -10`
- [ ] Expect: `OK — <N> target(s) match manifest.` Zero diff — the manifest now holds the new YAML fingerprints and the unchanged scientific fingerprints.

### Step 5.9: Document the expected config-snapshot drift

- [ ] Create `dev/milestones/r01/baseline_diffs.md` with this content (replace `YYYY-MM-DD` with the actual sealing date):

```markdown
# R01 baseline diffs

**Date.** YYYY-MM-DD (sealing date).

## Summary

Three baseline-manifest entries change as a result of R01's config
schema migration. None reflect scientific drift — they are purely the
result of each Snakefile's `rule copy_config` writing the new sectioned
YAML (instead of the old flat YAML) into `{project_dir}/config/`.

## Affected manifest entries

| Target template                                              | Reason                              |
| ------------------------------------------------------------ | ----------------------------------- |
| `{project_dir}/config/snake_config_model_creation.yml`       | Sectioned schema → different bytes  |
| `{project_dir}/config/snake_config_climate_projections.yml`  | Sectioned schema → different bytes  |
| `{project_dir}/config/snake_config_climate_experiment.yml`   | Sectioned schema → different bytes  |

These three YAMLs are full copies of the input snake config (whatever
the user passed via `--configfile`). R01 changes the input config
shape, so the copied output also changes shape. The fingerprint diff is
deterministic.

## Scientific targets (unchanged)

All netCDF / CSV / PNG manifest targets match the pre-R01 baseline
exactly:

- `{project_dir}/plots/wflow_model_performance/{hydro_wflow_1,basin_area,precip}.png`
- `{clim_project_dir}/annual_change_scalar_stats_summary.{nc,csv}`
- `{clim_project_dir}/annual_change_scalar_stats_summary_mean.csv`
- `{clim_project_dir}/plots/{projected_climate_statistics,precipitation_anomaly_projections_abs,temperature_anomaly_projections_abs}.png`
- `{exp_dir}/model_results/{Qstats,basin}.csv`

## Method

Recorded against `examples/test_local` via a candidate manifest, then
compared candidate vs old to confirm only the three YAML entries
changed, then promoted:

```bash
pixi run python dev/scripts/check_baseline.py record \
  --project-dir examples/test_local --manifest dev/baseline/manifest.candidate.json
# structural diff candidate vs canonical -> only 3 config snapshots differ
mv dev/baseline/manifest.candidate.json dev/baseline/manifest.json
pixi run python dev/scripts/check_baseline.py check --project-dir examples/test_local  # clean
```

The new manifest is the R01 contract. R3+ milestones are bound by it.
```

### Step 5.10: Edit roadmap.md to seal R1

- [ ] In `dev/roadmap.md`, find the R1 heading (line ~135):

```markdown
### R1 — Modularity contracts (in flight)

**Goal.** Establish per-workflow config contracts so workflows can be
```

- [ ] Insert a status line between heading and goal (replace `YYYY-MM-DD` with the sealing date):

```markdown
### R1 — Modularity contracts (in flight)

**Status.** Sealed YYYY-MM-DD — three top-level config sections in
place; all 3 Snakefiles + 3 src/ scripts + conftest + both integration
tests read sectioned config; config path via workflow.configfiles[0];
migration guide for user-local configs at
dev/milestones/r01/local-config-migration.md. Per-workflow contract docs deferred
to R3/R4/R5 (2026-07-17 amendment). Suite: 51 passed, 2 skipped, 2
xfailed (the pre-R01 47 plus 4 focused R01 reader/normalization tests).
Scientific baseline: zero diff (pre-rebaseline scientific
fingerprints unchanged; only the 3 documented config snapshots
changed; post-rebaseline check clean) per dev/milestones/r01/baseline_diffs.md.

**Goal.** Establish per-workflow config contracts so workflows can be
```

- [ ] In the same R1 section, update the **Exit criteria** suite-count bullet (roadmap line ~177). It currently reads the pre-R01 count:

```markdown
- `pytest tests/` unchanged: 47 passed, 2 skipped, 2 xfailed.
```

  Replace with (R01 adds 4 focused reader/normalization tests — see Step 2.13b):

```markdown
- `pytest tests/`: 51 passed, 2 skipped, 2 xfailed (the pre-R01 47 plus
  4 focused R01 reader/normalization tests; no pre-existing test changes
  outcome).
```

  > **Owner decision (settled 2026-07-18): keep the 4 focused tests; `51` is the sealed count.** This supersedes the pre-R01 `47` that finding 7 of the accepted review pinned as a mechanical correction — applying review §4.10 (focused tests for the horizon normalization and sectioned stress-test readers) necessarily bumps the count. Known debt carried to R4: the normalization test duplicates `_to_str_tuple`'s body because the production copy lives in a module that executes Snakemake globals at import; deduplicate when R4 makes `get_change_climate_proj.py` importable.

### Step 5.11: Mark R3's configfile-mechanism deliverable as done by R01

R3's roadmap section (lines ~268-269) already carries the strikethrough `~~Replace the ... sys.argv re-parsing trick ...~~ **Done by R1.**`. Confirm it is present and reads correctly — do **not** search for the old un-struck text (it was already updated in the roadmap). If the strikethrough is missing (e.g. the roadmap was reverted), re-apply it:

```markdown
- ~~Replace the `--configfile` `sys.argv` re-parsing trick in all
  three Snakefiles with `workflow.configfiles[0]`.~~ **Done by R1.**
```

### Step 5.12: Update the stale AGENTS.md configfile-mechanism bullet

The mechanism changed in Task 2, so the `AGENTS.md` convention that describes it is now stale and must change in step. (Ideally this AGENTS.md edit belongs in the same commit as the Task 2 Snakefile change; if you kept it here for a single seal commit, that is acceptable — but do not leave it undone.)

- [ ] In `AGENTS.md`, under `## Conventions`, find:

```markdown
- Each Snakefile re-reads the `--configfile` path from `sys.argv` to forward it to
  downstream R scripts — do not break that pattern.
```

- [ ] Replace with:

```markdown
- Each Snakefile obtains the `--configfile` path from `workflow.configfiles[0]`
  and forwards it (as `config_path`) to downstream R scripts — keep that
  forwarding pattern even though the Snakefile itself reads the parsed `config`.
```

### Step 5.13: Commit roadmap + AGENTS.md + baseline_diffs + manifest

- [ ] Run:
```
git add dev/roadmap.md AGENTS.md dev/milestones/r01/baseline_diffs.md dev/baseline/manifest.json
git commit -m "$(cat <<'EOF'
r01: seal milestone in roadmap; re-baseline config snapshots

Three sectioned config sections; all 3 Snakefiles + 3 src/ scripts +
conftest + both integration tests migrated; config path via
workflow.configfiles[0]. Per-workflow contract docs deferred to
R3/R4/R5 (2026-07-17 amendment).

Scientific baseline preserved exactly. The three copied-config YAML
snapshots re-recorded as expected organizational drift (not scientific)
per dev/milestones/r01/baseline_diffs.md, via a candidate manifest compared
against the old one before promotion. The new manifest is the R01
contract.

Also updates the stale AGENTS.md convention (config path is now
workflow.configfiles[0], not a sys.argv scan) and confirms R3's
configfile-mechanism deliverable is marked done by R01.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 5.14: Tag the milestone

- [ ] Run: `git tag -a r01-contracts -m "r01-contracts: per-workflow config contracts"`
- [ ] Verify: `git tag -l "r01-*"` should now include `r01-contracts`.

### Step 5.15: Report

- [ ] Summarize to the user:
  - Branch `milestone/r01-contracts` complete, tagged `r01-contracts`.
  - Suite: `51 passed, 2 skipped, 2 xfailed` (the pre-R01 `47` plus 4 focused R01 tests; no pre-existing test changed outcome).
  - Baseline: pre-rebaseline scientific fingerprints unchanged; only the three documented config snapshots changed; post-rebaseline check clean.
  - **Files added:** `config/snake_config.template.yml`, `tests/test_r01_config_readers.py`, `dev/milestones/r01/local-config-migration.md`, `dev/milestones/r01/baseline_diffs.md`.
  - **Files modified:** `tests/snake_config_model_test.yml`, `config/snake_config_model_test.yml`, `config/snake_config_model_test_linux.yml`, `tests/conftest.py`, `tests/test_workflow_model_creation.py`, `tests/test_workflow_climate_projections.py`, `Snakefile_model_creation`, `Snakefile_climate_projections`, `Snakefile_climate_experiment`, `src/prepare_cst_parameters.py`, `src/prepare_weagen_config.py`, `src/get_change_climate_proj.py`, `dev/baseline/manifest.json`, `dev/roadmap.md`, `AGENTS.md`.
  - Contract docs deferred to R3/R4/R5 (no `dev/workflows/` deliverable in R01).

---

## Notes for the executing engineer

- **The Task 2 atomic commit is the load-bearing one.** Everything before it (template) is additive; everything after it (Linux config, migration guide, seal) is downstream of the schema already being live. If Task 2 lands cleanly, the rest is mechanical.
- **If you discover a config key I missed** (the migration mapping in `dev/milestones/r01/modularity-contracts-design.md` and Task 4's table is what the consumers depend on), stop and surface it. Don't silently extend the mapping — the user reviewed the design with the documented mapping.
- **The string → list format change** for `historical_year_range` and `future_horizons` is intentional. `src/get_change_climate_proj.py` accepts both forms via `_to_str_tuple`. If other downstream Python parses these as strings, that's a separate bug to flag (not fix in R01).
- **Don't migrate `*_local.yml` files into git.** They're gitignored. Task 4's guide is what the user follows manually, and Task 5's `examples/test_local` seed config is untracked.
- **The `enabled: true` flag is documentary in R01.** Don't add Snakefile-side logic that respects it; that's R6+ work.
- **Per-workflow contract docs are NOT an R01 deliverable.** They are written at the opening of R3 (model_creation), R4 (climate_projections), R5 (climate_experiment) per the 2026-07-17 amendment. Do not create `dev/workflows/` here.

## Quick reference: file inventory

| File | Action | Task |
|---|---|---|
| `config/snake_config.template.yml` | Create | 1 |
| `tests/snake_config_model_test.yml` | Modify | 2 (atomic) |
| `config/snake_config_model_test.yml` | Modify | 2 (atomic) |
| `tests/conftest.py` | Modify | 2 (atomic) |
| `tests/test_workflow_model_creation.py` | Modify | 2 (atomic) |
| `tests/test_workflow_climate_projections.py` | Modify | 2 (atomic) |
| `tests/test_r01_config_readers.py` | Create | 2 (atomic) |
| `Snakefile_model_creation` | Modify | 2 (atomic) |
| `Snakefile_climate_projections` | Modify | 2 (atomic) |
| `Snakefile_climate_experiment` | Modify | 2 (atomic) |
| `src/prepare_cst_parameters.py` | Modify | 2 (atomic) |
| `src/prepare_weagen_config.py` | Modify | 2 (atomic) |
| `src/get_change_climate_proj.py` | Modify | 2 (atomic) |
| `config/snake_config_model_test_linux.yml` | Modify | 3 |
| `dev/milestones/r01/local-config-migration.md` | Create | 4 |
| `dev/milestones/r01/baseline_diffs.md` | Create | 5 |
| `dev/baseline/manifest.json` | Modify | 5 |
| `dev/roadmap.md` | Modify | 5 |
| `AGENTS.md` | Modify | 5 (or fold into Task 2) |

`config/snake_config_model_test_local.yml` is created untracked in Task 5 (Step 5.1) and never committed (`*_local.yml` is gitignored).

## Quick reference: expected commit count

**5 commits + 1 tag** (Task 2b's audit commit is conditional — only if a straggler is found, making it 6 commits):

1. Template (Task 1).
2. Atomic migration: tests config + canonical config + conftest + 2 integration tests + 3 Snakefiles + 3 src/ readers (Task 2).
   - (Conditional: audit-fix commit from Task 2b — only if the audit finds a real straggler.)
3. Linux example config (Task 3).
4. Local-config migration guide (Task 4).
5. Seal: roadmap + AGENTS.md + baseline_diffs + re-recorded manifest (Task 5, Step 5.13).

Then: tag `r01-contracts` (Task 5, Step 5.14).

(If the `AGENTS.md` convention edit is folded into the Task 2 atomic commit instead of the seal commit — the cleaner option — the commit count is unchanged; only the file's home moves.)