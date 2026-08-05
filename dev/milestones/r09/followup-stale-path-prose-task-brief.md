Task Brief — R9 follow-up: pre-R9 paths in code prose and in one dev tool

Raised by P5's grep falsifier on 2026-08-05 and ruled the same day: P5 fixes the
documentation classes in its own scope and briefs these two, because one is a
per-file reading job inside files P5 forbids and the other is a **functional**
defect that wants a test, not a doc edit.

### Class A — prose inside `blueearth_cst/**` and the Snakefiles

Module docstrings and rule comments that name pre-R9 paths. They are wrong in
the way that costs most: a contributor opening the module to change it reads the
docstring as the current contract.

| File | What it says |
| --- | --- |
| `climate_analysis/climate_figures.py:9-36` | `climate_historical/<key>/plots/`, `hydrology_model/forcing/plots/`, `hydrology_model/evaluation/plots/` |
| `climate_analysis/plot_climate_source.py:7-176` | same three, plus the store path |
| `climate_analysis/extract_historical_climate.py:4` | `climate_historical/<key>/` |
| `climate_analysis/prepare_climate_data_catalog.py:38` | `experiments/<name>/weather_generator/output/` |
| `experiment/check_project_consistency.py:7` | `hydrology_model/` |
| `experiment/downscale_climate_forcing.py:84` | `experiments/<name>/hydrology_runs/rlz_<r>/` — a comment explaining the R7 depth, in the module R9 P2 flattened |
| `experiment/export_wflow_results.py:25,70` | `hydrology_runs/rlz_<n>/output/cst_<m>.csv`, `indicators/` |
| `model/plot_map_forcing.py:3-60` | `hydrology_model/forcing/…` ×3 |
| `model/plot_results.py:164` | `climate_historical/<key>/extract_historical.nc` |
| `shared/climate_parity.py:53` | same |
| `shared/snake_utils.py:696,923-945,1642` | `climate_historical/`, `hydrology_model/`, `climate_projections/` |
| `weathergen/generate_weather.R:23,89` | `weather_generator/` ×2 |
| `Snakefile_climate_experiment:151-186,617` | a 30-line layout block describing `weather_generator/` and `hydrology_runs/rlz_<r>/` as current, with the R9 correction appended **below** it |

**Do not sweep this with a find-replace.** Two distinct kinds are mixed in, and
they need opposite treatments:

- **Stale descriptions** — prose asserting where a file lives today. Fix.
- **Change narrative** — "R07 B5 moved X to Y; R9 P2 moved it back", which is a
  correct historical record and the reason the current shape is what it is.
  `Snakefile_climate_experiment:151-186` is the hard case: an R7 design block
  with an R9 amendment appended, so the file is not wrong but the first thirty
  lines read as current until the reader reaches line 169. Prefer restructuring
  it (current shape first, history after) over deleting the history.

Deciding which is which needs the module read, not the line matched.

### Class B — `dev/scripts/scaffold_project_tree.py`

Not prose. `_stage_cross_workflow_inputs` (line 99) stages the region at
`hydrology_model/staticgeoms/region.geojson`, a path P2 deleted, so the tool
would fail to produce a WF2/WF3 summary if anyone ran it.
`dev/scripts/scaffold_extras.yml` lists ten overlay paths under
`hydrology_model/` and `experiments/*/weather_generator/`.

**Verified inert, and the verification matters more than the fix.** It is not on
P1's evidence chain: `declared_inventory.txt`'s REGENERATE recipe is three raw
`snakemake --summary` invocations, not this script, and nothing in the
repository references the script. So the stale staging did not contaminate the
declared tier. Confirm that independently before trusting it — the same
inertness claim was made for a *different* stale staging in P4 (`test_cli.py`'s
region, inert because ADR 0003 lets WF3 delineate its own), and inheriting a
verdict across files is how this class survives.

Note the same fixture question bit for real elsewhere: the P4 escape P5's
full gate caught (`test_guard_invalidation.py`'s missing model leaves) was a
staging helper that fell behind a new declared input. This script is the third
copy of that staging logic. Consider whether it should share one.

### Class C — three more `dev/scripts/` files

Smaller, and one of them is R9's own.

| File | Line | Says | Should be |
| --- | --- | --- | --- |
| `prune_climate_store.py` | 5, 9, 69 | `climate_historical/`, `climate_projections/<proj>/scalar/` | `data/climate/historical/`, `data/climate/projections/<proj>/scalar/` |
| `probe_store_read_timing.py` | 42 | `test_case/test_local/climate_historical/era5_…/store_region.geojson` | the `data/climate/historical/` store |
| `inspect_spatial_ref.py` | 13 | `examples/test_local/climate_historical/raw_data/…` | pre-R9 debt — `examples/` was retired at R7 and `raw_data/` at R7 B1 |

`prune_climate_store.py` is the sharpest of the three because **P1 shipped it and
P2 repointed it**: `STORE_ROOT = "data/climate/historical"` is correct in code
while three docstrings around it still name the old root. The constant was
edited; the prose describing the constant was not. That is class A's failure
mode inside R9's own deliverable, which is the argument for doing all three
classes in one task rather than treating prose staleness as other people's debt.

`probe_store_read_timing.py` carries a hardcoded path, not a docstring, so it is
class-B-shaped: verify whether it still runs.

`inspect_spatial_ref.py` predates R9 by two milestones. Fix or delete — a probe
pointing at a tree retired at R7 has no reader.

### Allowed scope

**Permitted** — `blueearth_cst/**` (comments and docstrings only), the three
`Snakefile_*` (comments only), `dev/scripts/scaffold_project_tree.py`,
`dev/scripts/scaffold_extras.yml`, `dev/scripts/prune_climate_store.py`,
`dev/scripts/probe_store_read_timing.py`, `dev/scripts/inspect_spatial_ref.py`,
tests.

**Forbidden** — any executable statement in `blueearth_cst/**` or a Snakefile;
`config/**`; `dev/baseline/**`; sealed records under `dev/milestones/`.

### Validation

**Named scope** — `pixi run test-cli` (a comment edit that breaks a Snakefile
fails here), plus the test module of any file whose docstring is touched. Class
B needs a scaffold invocation, since the defect is that the tool does not run.

**Falsifier.** For class A the check is the same grep P5 used, re-run over
`blueearth_cst/**` and `Snakefile_*` with every surviving hit justified as
change narrative rather than as description. Note that the loose term `RT_`
over-matches (`SHORT_DIGEST_CHARS`, `HISTORICAL_START_YEAR`, `_MAX_VERT_EXAG`);
use `\bRT_[0-9A-Za-z]`.

For class B a unit test asserting the constructed path would pass without ever
proving the tool works. **Run it**: `--print-tree` against a temp project must
produce a WF2 and WF3 summary, which it cannot do while the staged leaf is
missing — and it must be shown to FAIL before the fix.

### Acceptance criteria

- Every prose path in `blueearth_cst/**` and the Snakefiles either describes the
  current tree or is explicitly framed as history.
- `scaffold_project_tree.py` runs end to end and stages every cross-workflow
  leaf the three Snakefiles currently declare — including P4's model files.
- The failing-before-fixing demonstration is recorded, not just asserted.
- No executable line changed.
