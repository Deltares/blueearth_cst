Task Brief — R9 follow-up: pre-R9 paths in code prose and in one dev tool

> **COMPLETE 2026-08-05.** All three classes done. Class B was demonstrated
> FAILING before the fix, as the brief requires, and the failure was exactly
> the P4 model files it predicted. Three defects beyond the brief's text were
> found and fixed in the same files -- a silently-zero log matcher, an overlay
> file whose entries the fixture disproves, and a guard test that had become
> vacuous. Evidence at the foot of this brief; the brief is kept unedited
> above that record.

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

### Class C — four more `dev/scripts/` files

Smaller, and one of them is R9's own.

| File | Line | Says | Should be |
| --- | --- | --- | --- |
| `prune_climate_store.py` | 5, 9, 69 | `climate_historical/`, `climate_projections/<proj>/scalar/` | `data/climate/historical/`, `data/climate/projections/<proj>/scalar/` |
| `probe_store_read_timing.py` | 42 | `test_case/test_local/climate_historical/era5_…/store_region.geojson` | the `data/climate/historical/` store |
| `inspect_spatial_ref.py` | 13 | `examples/test_local/climate_historical/raw_data/…` | pre-R9 debt — `examples/` was retired at R7 and `raw_data/` at R7 B1 |
| `verify_constant_pars.py` | 25, 88 | `examples/test_local/hydrology_model` — a usage string **and an argparse default** | doubly stale: `examples/` retired at R7, `hydrology_model/` at R9 |

**`probe_store_read_timing.py` is now broken rather than merely stale**: the
2026-08-05 orphan sweep deleted the file its hardcoded path points at
([`orphan-sweep-2026-08-05.txt`](orphan-sweep-2026-08-05.txt)).
`verify_constant_pars.py` is the more dangerous of the two, because its stale
path is an **argparse default** — running it with no `--model-dir` silently
targets a tree that has not existed since R7, which reads as a working default
rather than as an error.

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

---

## Completion record — 2026-08-05

### Class A — 13 files, read one at a time

Fixed as stale DESCRIPTION (prose asserting where a file lives today):

| File | Was | Now |
| --- | --- | --- |
| `climate_analysis/climate_figures.py` | `climate_historical/<key>/plots/`, `hydrology_model/forcing/plots/`, `hydrology_model/evaluation/plots/` | the `data/climate/historical/` and `models/hydrology/wflow/` forms (reST table columns re-widened) |
| `climate_analysis/plot_climate_source.py` | the same three, plus the store path in two `Parameters` entries | ditto |
| `climate_analysis/extract_historical_climate.py` | `climate_historical/<key>/`, and `spatial/geoms/region.geojson` (not in the brief's list, same defect) | `data/climate/historical/<key>/`, `data/spatial/geoms/region.geojson` |
| `climate_analysis/prepare_climate_data_catalog.py` | "now `experiments/<name>/weather_generator/output/`, R07 B5" | the `climate/weathergenr/` path; the "moved twice" point is kept, which is what the sentence is *for* |
| `experiment/check_project_consistency.py` | `hydrology_model/` | `models/hydrology/wflow/` |
| `experiment/downscale_climate_forcing.py` | "the run dir moved to `hydrology_runs/rlz_<r>/`, one level deeper" | current dir first, then BOTH moves — the "../ depth is not a literal to maintain by hand" point is strengthened by R9 P2, not weakened |
| `experiment/export_wflow_results.py` | `indicators/` | `results/` |
| `model/plot_map_forcing.py` | `hydrology_model/forcing/…` x3 | `models/hydrology/wflow/forcing/…` |
| `model/plot_results.py`, `shared/climate_parity.py` | `climate_historical/<key>/extract_historical.nc` | `data/climate/historical/…` |
| `shared/snake_utils.py` | the store dir x3 and the banner example | `data/climate/historical/`, `models/hydrology/wflow/`, `data/climate/projections/` |
| `projections/series_identity.py` | `climate_projections/{clim_project}/` (not in the brief's list) | `data/climate/projections/{clim_project}/` |
| `weathergen/generate_weather.R` | `weather_generator/` x2 | `climate/weathergenr/`, with R7-set-the-split / R9-renamed-it as history |
| `Snakefile_climate_experiment:150-186` | 30-line R7 block read as current, R9 amendment appended at line 170 | RESTRUCTURED: `CURRENT SHAPE` first, then `HOW IT GOT HERE` — history kept, not deleted |
| `Snakefile_model_creation:748` | "build with NEITHER hydrology_model/ …" | the R9 model root |

Rule numbers were checked, not assumed: 1.11 / 1.13 / 1.15 are still correct,
so only paths moved.

**Four hits survive the falsifier, each change narrative rather than
description** — the distinction the brief asks the sweep to make:

| Hit | Why it stays |
| --- | --- |
| `export_wflow_results.py:25` | "R07 B5 took it out of the filename … R9 P2 dissolves that level" — correct history, and the current path is stated right after |
| `Snakefile_model_creation:536` | "wf1's **old** `climate_historical/wf1_raw/` store … **retired**" — describes a store that no longer exists at any path |
| `Snakefile_climate_projections:189` | "what **freed it from** wf1's `hydrology_model/staticgeoms/region.geojson`" — the dependency it names is the one that was removed |
| `Snakefile_climate_experiment:172` | inside the new explicit `HOW IT GOT HERE` section |

One further hit, `Snakefile_climate_projections:61`, is CODE
(`config/runs/climate_projections/<digest>`) and correct.

### Class B — the scaffold, demonstrated failing first

**Before** (`--out` to a scratch tree, nothing else changed):

    wf1: 50 declared outputs, 0 logs
    wf2: 37 declared outputs, 0 logs
    !! Snakefile_climate_experiment --summary failed:
    wf3: 0 declared outputs, 0 logs
    scaffolded 95 paths

    MissingInputException in rule write_model_reference … line 390:
        affected files:
            …/models/hydrology/wflow/wflow_sbm.toml
            …/models/hydrology/wflow/.outputs_configured

Exactly the P4 model files the brief predicted. Two things the run showed that
the brief did not:

1. **The tool exits 0 while producing an incomplete tree.** `_summary` writes
   the failure to stderr and returns `[]`, so WF3 contributed zero paths and the
   script still reported success. A caller checking the exit code learns nothing.
2. **The stale region was not the cause, and staging one at the corrected path
   would still have been wrong.** WF2's summary succeeded *with* the region at
   the pre-R9 path, because since ADR 0003 neither downstream workflow consumes
   a wf1 region: each delineates its own `data/spatial/geoms/region.geojson` and
   declares it as an OUTPUT. So the fix REMOVES the region staging rather than
   repointing it, and `_STAGED_LEAVES` is now the three leaves WF3 actually
   declares — the wf1 config snapshot (3.00b) and the two model files (3.01c).

**After:**

    wf1: 50 declared outputs, 15 logs
    wf2: 37 declared outputs, 6 logs
    wf3: 95 declared outputs, 16 logs
    scaffolded 224 paths

The scaffolded tree's top level is now exactly the six ruled roots —
`benchmarks/ config/ data/ experiments/ logs/ models/` — with no pre-R9 path
anywhere in it.

### Beyond the brief, and why

**`_log_paths` was silently returning zero for every workflow** ("0 logs"
above). Rules no longer interpolate `project_dir` directly; they interpolate
`LOG_PARTS_DIR`, itself assigned from `project_dir` (WF1/WF2) or `exp_dir`
(WF3). The matcher resolved only the two root names, so it matched nothing.
Fixed by resolving one level of indirection out of the same file, plus bare
`.log` string constants (`WORKFLOW_LOG_NAME`) that would otherwise reduce to
`logs/1`. This is the brief's own defect class — a helper left behind by a
refactor, reporting success while producing a wrong tree — in the file the
brief assigns, so it is fixed here rather than deferred; flagged because it is
not in the brief's text. Resolving indirection introduced and then fixed a bug
of its own (`exp_dir` is itself assigned from `project_dir`, so resolving it
replaced the seeded value with one still carrying a wildcard, rendering
`experiments/1/`); the guard is `name not in roots`.

**`scaffold_extras.yml` was rederived, not repointed.** All nine overlay entries
named pre-R9 paths. Repointing them would have preserved four that the fixture
disproves: `run_default/outstates.nc` (wrong depth — it is under
`run_default/outstate/`), `run_default/output_scalar.nc` (not written by this
config at all), and two weathergenr filenames that do not exist. Each is
recorded at the foot of that file with its reason rather than silently dropped,
and the surviving entries were checked against `test_case/test_local`.

**A guard test had become vacuous.**
`tests/test_cli.py::test_climate_projections_declares_wf1_region_input`
asserted that `staticgeoms/region.geojson` appears in
`Snakefile_climate_projections`, standing in for a wf1 to wf2 contract. That
contract is gone, and the only remaining occurrence of the string is the comment
recording that WF2 was *freed* from it — so the assertion passed while guarding
nothing, and would have kept passing had the sweep deleted the prose it depends
on. Replaced by `test_climate_projections_owns_its_region`, which pins the
current shape: WF2 resolves its region through `region_spec` and reads nothing
from the model root except in comments. The fixture's own vestigial region
staging is left alone — that is the staging-consolidation item (P5 F3), which
the brief says not to fold in.

### Class C — four `dev/scripts/` files

| File | Fix |
| --- | --- |
| `prune_climate_store.py` | three docstrings brought into line with the `STORE_ROOT` constant P2 already repointed |
| `probe_store_read_timing.py` | was BROKEN, not stale: its default pointed at `store_region.geojson`, which ADR 0003 retired, so the file exists nowhere. Repointed to the fixture's model-free `data/spatial/geoms/region.geojson` |
| `inspect_spatial_ref.py` | two milestones stale. Fixed rather than deleted; the two realization NCs are `temp()`, so "(not present)" between runs is now documented as normal rather than looking like rot |
| `verify_constant_pars.py` | usage string AND the argparse default — the dangerous one, since running it bare silently targeted a tree retired at R7 |

### Validation

- Class A falsifier: the P5 grep re-run over `blueearth_cst/**` and all three
  Snakefiles. Four hits, each justified above. `\bRT_[0-9A-Za-z]` (the tight
  form, avoiding the over-match the brief warns about): no hits.
- Class B falsifier: the scaffold run above, failing before and passing after.
- `ruff check` on every changed file: no new findings. The two `F541` in
  `prune_climate_store.py` are pre-existing at HEAD.
- No executable line changed in `blueearth_cst/**` or any Snakefile.
