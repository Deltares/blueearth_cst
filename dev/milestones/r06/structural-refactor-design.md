# R06 — Structural refactor — design

**Status. ACCEPTED 2026-07-22** via the `r06-structural-refactor` design-review-loop
(G1 framing + G2 approval both granted; converged via arbitration after the 2-round
external cap). Full audit trail — verdict table, internal panel index, both external
rounds verbatim, and the 23-row finding ledger — is in
`structural-refactor-design-review-record.md` beside this file. Implementation is a
separate `task-brief` handoff; this is the accepted plan, not yet built. Provenance of
this (v4) content follows.

Authored as v4 for the `r06-structural-refactor`
design-review-loop run. **v4 is the arbitration revision.** The external review cap
(2 rounds) was reached; the user arbitrated the three surviving round-2 findings on
2026-07-22 as **accepted, fix required**, and v4 is confined to exactly those three
fixes — **ext2-01** (the copied-config snapshots under `{project_dir}/config/` are an
*expected structural difference*, adjudicated by a narrow normalize-then-compare YAML
policy used identically in both baseline gates), **ext2-02** (the full-tree `.nc`
comparison is **element-wise**, not the `fingerprint_nc`/`diff_nc` aggregate stats),
**ext2-03** (the wrapper boolean contract is the *parsed* value: any scalar YAML
resolves to a boolean is accepted, so unquoted `yes` is valid) — plus the
cross-references those edits force. v4 edits are marked inline as
**[v4: <finding-id>]**; the ext2 rows are dispositioned in `ledger.md` (round
`external-r2`, "user arbitration 2026-07-22, fix required"). Everything else carries
forward from v3 verbatim. **v3 answered the external (GPT) review round 1** (4 findings:
3 major, 1 minor; verdict *revise*). Every external finding is dispositioned in
`ledger.md` (rows `ext1-01`…`ext1-04`, round `external-r1`); the substantive edits are
called out inline as **[v3: <finding-id>]**. v2 answered the internal panel (3
blocking, 5 major, 8 minor); all 16 internal rows are unchanged. The two G1-approved
positions (defer the functional decomposition; wrapper for `enabled:`) are **unchanged
and unchallenged** by the external round — v3 hardens the verification machinery and
pins the wrapper contract underneath them, it does not re-argue them. Genre:
**decision-record** (a milestone/layout refactor), mirroring the R3/R4/R5 house pattern
under `dev/r0#/` (Goal · Why now · Approach · What changes · `enabled:`
operationalization · Functional-decomposition fork · Behavior-preservation stance ·
Verification plan · Commit plan · Alternatives · Risks/open questions). Implementation
is a later `task-brief` handoff — this run produces an **accepted design only** (intake
non-goal). Landing location on acceptance: `dev/milestones/r06/structural-refactor-design.md`.

**Scope authority.** `dev/roadmap.md` § "R6 — Structural refactor" (the 7-item lock
list, exit criteria, tag `r06-refactor`) and
`dev/working/design-runs/r06-structural-refactor/intake.md` (problem, constraints,
decision criteria, the two required G1 positions). This doc is self-contained: a
reviewer needs only this file plus the cited paths.

**Two required G1 positions, stated up front (rationale in §7 and §8):**

- **Functional decomposition of climate analysis → DEFER** to a later milestone.
  R6 stays pure-layout; the §1 package tree is chosen to keep the decomposition
  *forward-compatible* (the already-model-independent helpers stay liftable).
- **`enabled:` operationalization → WRAPPER script** (`scripts/run_workflows.py`
  reading `workflows.<name>.enabled`), framed as the evolution of the existing
  `run_snake_test.cmd` / `run_snake_docker.sh` runners — **not** a Snakemake
  `module:` master Snakefile.

**Cross-cutting method correction (carried from v2).** The internal panel's root
theme: v1's "uniform mechanical transform + green `pytest` + clean `--dry-run` =
invariance by construction" **under-counts the reference surface and over-trusts
`--dry-run`.** `--dry-run` is blind to (a) `params:` string paths, (b)
`shell:`/`Rscript` command bodies, (c) R `source()` calls, and (d) the `run_logged.py`
bootstrap — none of which are declared `input:`/`output:`, so the DAG resolver never
touches them. v2 therefore (i) drives the import/path rewrite from a **grep-derived
per-module stage table** (§1a) rather than an a-priori pattern find-replace, and (ii)
adds **execution-level smokes** to the gates that touch runtime-only paths (R, shell,
`run_logged`). The by-construction stance and the doc structure are kept; the machinery
is hardened honestly.

**External-round correction [v3].** The external round accepts the by-construction
stance and the whole layout; it faults the **verification/orchestration machinery** on
four bounded points: (ext1-01) the run-relative baseline gate as written **mutated the
tracked `dev/baseline/manifest.json`** across a `git checkout`; (ext1-02) the new
`scripts/run_workflows.py` **contract was under-specified** (missing/invalid
`enabled:`, exit propagation, arg forwarding); (ext1-03) the `enabled: false` skip test
**did not require a fresh `project_dir`** and asserted output-absence rather than
invocation; (ext1-04) the full-tree diff was **a raw byte diff for NetCDF**. v3 fixes
each **using tooling that already exists** where it does — the `check_baseline.py`
`--manifest` redirect (ext1-01) and its per-type semantic comparators (ext1-04) — and
pins the wrapper contract (ext1-02) and the skip test (ext1-03). These are bounded
hardening fixes, not reframes.

---

## Goal

Reorganize the repository so source code, configuration, data catalogs, generated
outputs, and documentation are cleanly separated and discoverable, and
operationalize the R1 `workflows.<name>.enabled` flag so a workflow can be skipped
from a single config switch rather than by user discipline. R3/R4/R5 cleaned up
*within* each workflow (contract docs, per-rule `log:`/`benchmark:`, the shared
`get_config`/`tee_to_log`/`run_and_tee` helpers, naming). The cross-cutting layout
is still the pre-refactor shape; R6 sets it.

Concretely, R6 delivers the roadmap's 7-item lock list:

1. Split the flat `src/` into a package `blueearth_cst/` with per-stage submodules.
2. Split `config/` into workflow configs vs data catalogs (plus a third bin for
   build templates — see §5); keep `*_local.yml` gitignored.
3. Settle `dev/` (planning + `dev/scripts/`) vs `docs/` (user-facing) boundaries.
4. Settle the data-catalog directory layout under the config split.
5. Leave `project_dir/` output layout alone (no concrete pain point surfaced).
6. Fold the top-level runners into a settled home.
7. Operationalize `workflows.<name>.enabled`.

**R6 is a behavior-preserving refactor.** Per the roadmap cross-cutting principle,
R1/R2/R6 preserve the M1/R5 baseline modulo numerical-noise tolerance; the only
admissible behavioral change is what `enabled:` skip semantics require. Because R6
touches **no computational path** — every change is a file move + a mechanical
reference rewrite, or a new opt-in runner — scientific invariance is established
**by construction**, with `check_baseline` as a regression tripwire. **The
by-construction proof is only as good as its two premises**: (a) the reference
surface is *completely* enumerated (§1a table; §2/§5 inventory; not an a-priori
pattern set), and (b) the gates actually exercise the runtime-only paths `--dry-run`
cannot see (§9 execution smokes). v1 asserted both premises but delivered neither
completely; v2 makes both explicit. This mirrors R1's invariance-by-construction
stance (`dev/milestones/r01/baseline_diffs.md`), not R3/R4/R5's per-workflow-slice re-record.

## Why now

R3/R4/R5 are sealed; the vertical-by-workflow cleanup is done and the cross-cutting
layout is the last Phase-2 milestone. The pain points are concrete and observable
in the current tree:

- **`src/` is flat** — 27 Python modules (verified: 28 `src/*.py` files minus the
  empty `__init__.py`) for all three workflows plus shared helpers in one directory,
  with `src/weathergen/*.R` the only sub-grouping. No package boundary, no per-stage
  grouping; a newcomer must grep to learn which module belongs to which workflow.
  Note `src/__init__.py` is **empty** (1 line) — it is **not** a `sys.path` shim
  (correcting the intake's phrasing). The shim is
  `sys.path.insert(0, str(Path(workflow.basedir)))` at the top of each Snakefile
  (`Snakefile_model_creation` line 7, etc.) plus `run_logged.py`'s own two-level
  `dirname` walk (line 20).
- **`config/` mixes three kinds of file** in one flat directory (verified via
  `config/*.yml`): (a) workflow snake configs (`snake_config_model_test.yml`,
  `snake_config.template.yml`, the `*_projections_*` and `*_gabon.yml` /
  `*_local.yml` variants); (b) hydromt **data catalogs** (`deltares_data*.yml`,
  `cmip6_data.yml`); and (c) hydromt/wflow/weathergen **build templates**
  (`wflow_build_model.yml`, `wflow_update_waterbodies.yml`,
  `weathergen_config.yml`). The intake names only (a) vs (b); (c) is a real third
  bin the split must place (§5, blind-spot 1).
- **Top-level runners** (`run_snake_test.cmd`, `run_snake_docker.sh`) sit at the
  repo root with no settled home, alongside `pixi.toml`, `Project.toml`,
  `Manifest.toml`.
- **`enabled:` is dormant.** `workflows.<name>.enabled` is present in every full
  orchestration config (`snake_config.template.yml` lines 39/55/75;
  `snake_config_model_test.yml` 28/40/56; the `_linux`/`_gabon` variants) but **read
  by nothing** — a grep of the three Snakefiles finds no read. A workflow can only be
  skipped by choosing not to invoke its Snakefile. (The `snake_config_projections_*`
  configs carry **no** `workflows:` section at all — they are single-workflow direct
  inputs, not orchestration configs; this distinction is load-bearing for the wrapper
  contract, §7 / ext1-02.)
- **`dev/` vs `docs/` boundaries** are conventionally understood (AGENTS.md
  describes them) but not codified as a rule a newcomer can point to.

## Approach

The same staged, behavior-preserving discipline R3/R4/R5 used, adapted to a
layout refactor:

1. **Every commit leaves the repo runnable and baseline-checkable.** Each stage is
   an independently verifiable unit: after it, `pytest tests/` is green and all
   three Snakefiles `--dry-run` clean, **plus — for the commits that touch
   runtime-only paths — an execution smoke**, because `--dry-run` and the
   `test_cli.py` dry-run gate exercise neither R, shell bodies, nor the `run_logged`
   bootstrap. The one unavoidable exception — the atomic package move — is handled
   explicitly in the commit plan (§10, blind-spot 2) so it stays reviewable.
2. **Minimal churn for value gained.** Every move is justified by a lock-list pain
   point; roadmap item 5's "leave alone unless a concrete pain point emerges"
   applies repo-wide (so `project_dir/` output layout is untouched, and the config
   split introduces exactly the directories the pain points demand — no more).
3. **Consume upstream conventions verbatim.** Catalogs stay in hydromt's schema;
   build templates stay hydromt/wflow/weathergen files moved as-is. R6 changes
   *where files live and how they are referenced*, never *what a catalog means or
   how a `setup_*` method works* (AGENTS.md Hard Constraints;
   `stay-within-cst-automation-scope` memory).
4. **Every moved file gets an old→new map.** A `MIGRATION.md` (roadmap exit
   criterion) records every path so downstream forks (`upstream`) and user-local
   configs (`*_local.yml`, `*_gabon.yml`) can rebase.
5. **Snakemake stays the only entry point.** The three `Snakefile_*` files remain
   the only Snakemake entry points; `workflow.configfiles[0]` forwarding, the
   `get_config` contract, and `script:`/`shell:` wiring survive the move. The
   `enabled:` wrapper is a *runner over* those Snakefiles, not a replacement entry
   point (§7).
6. **The rewrite is table-driven, not pattern-driven.** The import/path rewrite
   is enumerated from a grep-derived per-module stage table (§1a) covering every
   reference class the panel surfaced — `from src.<module>`, `from src import
   <module>`, `script:` paths, `Rscript`/`shell:` bodies, R `source()`, `run_logged`
   bootstrap, and hardcoded `params:`/literal config paths — not a four-pattern
   find-replace.

## What changes

### 1. Split flat `src/` into the `blueearth_cst/` package (lock item 1)

**Target module tree.** Group the 27 `src/*.py` modules + `src/weathergen/` by
workflow stage, with a `shared/` submodule for the cross-cutting helpers. Every
module keeps its filename (grandfathered per `dev/reference/naming.md`); only the
package prefix changes.

```
blueearth_cst/
  __init__.py                      # empty, as src/__init__.py is today
  shared/
    __init__.py
    snake_utils.py                 # get_config, tee_to_log, run_and_tee, stress_test_grid, ...
    run_logged.py                  # CLI tee wrapper for shell: rules (depth-sensitive — see below)
    func_plot_signature.py         # shared plotting primitives (plot_clim etc.)
    plot_map.py                    # shared map helper
    merge_logs.py                  # shared log/benchmark reducers (used by all workflows)
    merge_benchmarks.py
    metrics_definition.py          # shared metric definitions
    setup_time_horizon.py          # shared time-horizon helper
  model/                           # workflow 1 — model creation
    __init__.py
    prepare_build_config.py
    setup_gauges_and_outputs.py
    setup_reservoirs_lakes_glaciers.py
    write_outlet_index.py
    get_region_preview.py
    plot_results.py
    plot_map_forcing.py
    climate_forcing.py             # ADR 0002 subcatchment-forcing aggregation
    copy_config_files.py           # (shared-shaped; see note)
  projections/                     # workflow 2 — climate projections
    __init__.py
    prepare_climate_data_catalog.py
    get_stats_climate_proj.py
    get_change_climate_proj.py
    get_change_climate_proj_summary.py
    plot_proj_timeseries.py
  experiment/                      # workflow 3 — climate experiment
    __init__.py
    prepare_cst_parameters.py
    prepare_weagen_config.py
    extract_historical_climate.py
    downscale_climate_forcing.py
    export_wflow_results.py
  weathergen/                      # R weather generator (moved verbatim)
    generate_weather.R
    impose_climate_change.R
    global.R
```

Notes on placement (each defensible, none load-bearing on the *mechanism*):
`copy_config_files.py`, `merge_logs.py`, `merge_benchmarks.py` are used across
workflows and could sit in `shared/`; `merge_logs.py`/`merge_benchmarks.py` **are**
placed in `shared/` (their reducers run in all three workflows), and
`copy_config_files.py` is shown under `model/` only because its concrete rule lives
there historically — final bin is a detail the reviewer may push on, but it does not
change the migration mechanism. **`prepare_climate_data_catalog.py`,
`extract_historical_climate.py`, `setup_time_horizon.py`, `metrics_definition.py`
are placed to keep the model-independent climate helpers liftable** (§8,
forward-compatibility): they are not buried in a workflow-specific leaf that a later
decomposition milestone could not reach.

### 1a. The reference rewrite is driven by a per-module stage table

**This table is the single source of truth for the rewrite. §9, the "Tests move"
paragraph, and §10 commit 1 all cite it and state the identical rule — no
paragraph paraphrases a different transform (the v1 inconsistency panel finding
repo-01 flagged).** The rule is: **for every module, the new import prefix is
`blueearth_cst.<stage>.<module>`, where `<stage>` is fixed by this table — it is
*not* a uniform `blueearth_cst.` and *not* a uniform `blueearth_cst.<stage>.`
applied blindly, because the stage segment differs per module and the `shared`
modules are not under any workflow stage.**

| Module | New package path | Stage segment |
| --- | --- | --- |
| `snake_utils.py` | `blueearth_cst.shared.snake_utils` | `shared` |
| `run_logged.py` | `blueearth_cst.shared.run_logged` | `shared` |
| `func_plot_signature.py` | `blueearth_cst.shared.func_plot_signature` | `shared` |
| `plot_map.py` | `blueearth_cst.shared.plot_map` | `shared` |
| `merge_logs.py` | `blueearth_cst.shared.merge_logs` | `shared` |
| `merge_benchmarks.py` | `blueearth_cst.shared.merge_benchmarks` | `shared` |
| `metrics_definition.py` | `blueearth_cst.shared.metrics_definition` | `shared` |
| `setup_time_horizon.py` | `blueearth_cst.shared.setup_time_horizon` | `shared` |
| `prepare_build_config.py` | `blueearth_cst.model.prepare_build_config` | `model` |
| `setup_gauges_and_outputs.py` | `blueearth_cst.model.setup_gauges_and_outputs` | `model` |
| `setup_reservoirs_lakes_glaciers.py` | `blueearth_cst.model.setup_reservoirs_lakes_glaciers` | `model` |
| `write_outlet_index.py` | `blueearth_cst.model.write_outlet_index` | `model` |
| `get_region_preview.py` | `blueearth_cst.model.get_region_preview` | `model` |
| `plot_results.py` | `blueearth_cst.model.plot_results` | `model` |
| `plot_map_forcing.py` | `blueearth_cst.model.plot_map_forcing` | `model` |
| `climate_forcing.py` | `blueearth_cst.model.climate_forcing` | `model` |
| `copy_config_files.py` | `blueearth_cst.model.copy_config_files` | `model` |
| `prepare_climate_data_catalog.py` | `blueearth_cst.projections.prepare_climate_data_catalog` | `projections` |
| `get_stats_climate_proj.py` | `blueearth_cst.projections.get_stats_climate_proj` | `projections` |
| `get_change_climate_proj.py` | `blueearth_cst.projections.get_change_climate_proj` | `projections` |
| `get_change_climate_proj_summary.py` | `blueearth_cst.projections.get_change_climate_proj_summary` | `projections` |
| `plot_proj_timeseries.py` | `blueearth_cst.projections.plot_proj_timeseries` | `projections` |
| `prepare_cst_parameters.py` | `blueearth_cst.experiment.prepare_cst_parameters` | `experiment` |
| `prepare_weagen_config.py` | `blueearth_cst.experiment.prepare_weagen_config` | `experiment` |
| `extract_historical_climate.py` | `blueearth_cst.experiment.extract_historical_climate` | `experiment` |
| `downscale_climate_forcing.py` | `blueearth_cst.experiment.downscale_climate_forcing` | `experiment` |
| `export_wflow_results.py` | `blueearth_cst.experiment.export_wflow_results` | `experiment` |

**Two import *forms* must both be rewritten.** A grep for `from src.` alone (v1's
counting method) **structurally misses** the `from src import <module>` form — the
panel found four sites in that form. The rewrite (and the counting grep) must match
**both**:

- **Form A — `from src.<module> import ...`** → `from blueearth_cst.<stage>.<module>
  import ...` (stage from the table). This is the dominant form. It includes the
  in-function deferred imports (e.g. `from src.snake_utils import tee_to_log` inside
  many modules' `__main__` blocks) — those are rewrite sites too.
- **Form B — `from src import <module>`** → `from blueearth_cst.<stage> import
  <module>` (stage from the table). **Grep-verified sites (all four in tests):**
  `tests/test_extract_historical_climate.py:147`
  (`from src import extract_historical_climate` → `from blueearth_cst.experiment
  import extract_historical_climate`); `tests/test_metrics_definition.py:22`
  (→ `from blueearth_cst.shared import metrics_definition`);
  `tests/test_prepare_climate_data_catalog.py:47`
  (→ `from blueearth_cst.projections import prepare_climate_data_catalog`);
  `tests/test_setup_time_horizon.py:48`
  (→ `from blueearth_cst.shared import setup_time_horizon`).

There is **no** bare `import src` form anywhere (grep-confirmed; the repo-fit lens
concurred).

**Grep-derived inventory (re-derived mechanically, not by pattern memory).** The
rewrite set was re-derived with
`grep -rnE "from src import|from src\." **/*.py`:
- **`src/` package modules:** every module that imports `snake_utils` at top level
  or in a `__main__` deferred import (verified: `copy_config_files`,
  `downscale_climate_forcing`, `export_wflow_results`, `extract_historical_climate`,
  `func_plot_signature`, `get_change_climate_proj`, `get_change_climate_proj_summary`,
  `get_stats_climate_proj`, `merge_benchmarks`, `plot_map`, `plot_map_forcing`,
  `plot_proj_timeseries`, `plot_results`, `prepare_build_config`,
  `prepare_climate_data_catalog`, `prepare_cst_parameters`, `prepare_weagen_config`,
  `run_logged`, `setup_gauges_and_outputs`, `setup_reservoirs_lakes_glaciers`,
  `setup_time_horizon`, `write_outlet_index`).
- **`tests/`:** `conftest.py:11`, plus the test modules importing a target
  (`test_climate_forcing`, `test_extract_historical_climate`,
  `test_get_change_climate_proj`, `test_get_change_climate_proj_summary`,
  `test_get_stats_climate_proj`, `test_merge_benchmarks`, `test_merge_logs`,
  `test_metrics_definition`, `test_prepare_build_config`,
  `test_prepare_climate_data_catalog`, `test_prepare_cst_parameters`,
  `test_prepare_weagen_config`, `test_run_logged` (two imports, lines 12–13),
  `test_setup_gauges_and_outputs` (two imports, lines 9, 27),
  `test_setup_reservoirs_lakes_glaciers`, `test_setup_time_horizon`,
  `test_snake_utils`, `test_stress_test_grid`).
- **`dev/scripts/`:** **only** `probe_attrs_chain.py:101`
  (`from src.get_change_climate_proj_summary import preprocess_coords` →
  `from blueearth_cst.projections.get_change_climate_proj_summary import ...`).
  **`dev/scripts/stage_data.py` is NOT a rewrite site** — its many `src` tokens are a
  local `Path` variable/parameter named `src`, and its cross-module import is
  `from console import ...` (line 76), not `from src.`. v1 wrongly listed it;
  **dropped.**
- **Non-import literal `config/` / `src/` string paths** (the class v1's four
  a-priori patterns omitted): `Snakefile_climate_experiment:131` `default_config`
  param (§2/§5, blind-spot 1 corrected); the two `Rscript` shell strings and the R
  `source()` calls (below); and the **known dead-path literal**
  `src/copy_config_files.py:99` — a `__main__` fallback
  (`config="config/snake_config_model_test.yml"`) that is unreachable under
  `script:` invocation. It is flagged so a reviewer is not surprised by it; it is
  harmless (dead path) but is enumerated rather than silently skipped. The
  `tests/test_workflow_*.py` `CONFIG = "config/snake_config_model_test.yml"`
  literals resolve to `config/` root, which is **not** a moved directory (only the
  three template files move out of it under §2), so those literals are unaffected —
  enumerated and confirmed no-op.

**How `script:` rule paths change.** Each Snakefile has 15 `script: "src/..."`
directives (verified: `grep -c 'script: "src/'` = 15 across the three files). These
become `script: "blueearth_cst/<stage>/<module>.py"` (stage from the §1a table).
Snakemake resolves `script:` paths relative to the Snakefile directory
(`workflow.basedir`), so the rewrite is a pure string substitution per directive.
Example: `script: "src/get_stats_climate_proj.py"` →
`script: "blueearth_cst/projections/get_stats_climate_proj.py"`.

**The `sys.path` shim stays.** The `sys.path.insert(0,
str(Path(workflow.basedir)))` shim at the top of each Snakefile **stays** —
`workflow.basedir` is the repo root, the parent of the `blueearth_cst/` package, so
the package is importable exactly as `src` is today. `src/__init__.py` (empty)
becomes `blueearth_cst/__init__.py` (empty); no shim logic lives in it. There is no
`pyproject.toml`/`setup.py`/editable install to touch (repo-fit lens confirmed:
`pixi.toml` `[pypi-dependencies]` are third-party only), so the package name is not
pinned anywhere but these `sys.path` inserts and `conftest.py`'s `parents[1]`
insert.

**`run_logged.py` is depth-sensitive (blind-spot 3).** Its bootstrap
`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
(line 20) assumes the file sits **one** level under repo root (`src/run_logged.py`
→ two `dirname`s = repo root). Moving it to `blueearth_cst/shared/run_logged.py`
puts it **two** levels down, so that same two-`dirname` walk resolves to
`blueearth_cst/`, not the repo root, and `from src.snake_utils import run_and_tee`
(line 21 → `from blueearth_cst.shared.snake_utils`) fails. **Fix:** add one more
`dirname` level to the bootstrap (or, cleaner, compute the repo root as the package
parent). This is a required, named edit in the package-move commit.
**`--dry-run` cannot verify this fix** — it does not execute the `shell:` body, so
`python -u run_logged.py ...` never runs under dry-run. The real guards are (i)
`tests/test_run_logged.py` (a **required commit-1 rewrite**: it does `from
src.run_logged import main` (line 13) and `from src.snake_utils import run_and_tee`
(line 12) and exercises the bootstrap by importing and calling `main`), and (ii) the
commit-1 execution smoke that actually invokes `run_logged.py` on a trivial command
(§9). v1's claim that "the commit-1 `--dry-run` gate exercises a shell rule's
`run_logged` invocation path" was **false and is removed.**

**`run_logged` reference in the three Snakefiles.** Each Snakefile builds
`run_logged = str(Path(workflow.basedir) / "src" / "run_logged.py")`
(`Snakefile_model_creation` line 21, etc.). This becomes
`... / "blueearth_cst" / "shared" / "run_logged.py"`. Named edit, three sites.

**The R layer is part of the migration surface (blind-spot 4).** `src/weathergen/`
is referenced by relative path in places that all move to
`blueearth_cst/weathergen/`:
- the two `shell:` command strings in `Snakefile_climate_experiment` (lines 174,
  199: `Rscript --vanilla src/weathergen/generate_weather.R ...` and
  `... impose_climate_change.R ...`) — rewritten to `blueearth_cst/weathergen/...`;
- the internal `source("./src/weathergen/global.R")` calls (verified:
  `generate_weather.R:2`, `impose_climate_change.R:6`) — rewritten to
  `source("./blueearth_cst/weathergen/global.R")`. These `source()` paths are
  relative to the **working directory** (repo root at run time), not to the R file,
  so they must move with the shell-command paths or the R scripts break at
  `source()`.
- **The R-layer migration checklist greps ALL `*.R` under the package (and
  `dev/scripts/*.R`) for any `src/weathergen` / `./src` literal — not only `source(`
  in the two shell-invoked entrypoints** — to catch any transitively `source()`-d
  path a partial rewrite would miss. (Current grep shows the only `source()` targets
  are the two `global.R` calls above; the broadened grep is the standing check so a
  future transitive source is not silently orphaned.
  `dev/scripts/install_weathergenr.R` is a dev-process helper, not on the package
  path, and holds no `src/weathergen` literal — confirmed no-op.)

**`workflow.configfiles[0]` forwarding survives** unchanged: it reads from the
Snakemake `workflow` object, independent of where `src/` lives. `config_path =
workflow.configfiles[0]` and its forwarding into R scripts and `copy_config` stay
verbatim.

**Tests move with their targets.** `tests/*.py` rewrite per the §1a table and the
two-form rule above — **both** `from src.<module>` (form A) and `from src import
<module>` (form B, the four sites named in §1a). `tests/conftest.py` line 11
(`from src.snake_utils import get_config` → `from blueearth_cst.shared.snake_utils
import get_config`) and `tests/test_run_logged.py` (lines 12–13) are in the rewrite
set. `tests/test_cli.py` locates Snakefiles by `SNAKEDIR` at repo root — unaffected
by the package move (but see §2 for the config fixture it is *not* immune to). The
one dev script `probe_attrs_chain.py` is rewritten; `stage_data.py` is **not** (see
§1a).

### 2 & 4. Split `config/` into workflows / catalogs / templates (lock items 2, 4)

**Target directory tree** (three bins, resolving blind-spot 1's third-bin
problem):

```
config/
  workflows/                       # snake configs (the --configfile targets)
    snake_config.template.yml
    snake_config_model_test.yml
    snake_config_model_test_linux.yml
    snake_config_projections_cmip5_full.yml
    snake_config_projections_cmip5_full_linux.yml
    snake_config_projections_cmip6_full.yml
    snake_config_projections_isimip3.yml
    snake_config_projections_isimip3_linux.yml
    # *_gabon.yml / *_local.yml land here too (gitignored / untracked variants)
  catalogs/                        # hydromt data catalogs (-d targets)
    deltares_data.yml
    deltares_data_linux.yml
    deltares_data_climate_projections.yml
    deltares_data_climate_projections_linux.yml
    cmip6_data.yml
  templates/                       # hydromt / wflow / weathergen BUILD templates
    wflow_build_model.yml
    wflow_update_waterbodies.yml
    weathergen_config.yml
```

**hydromt `-d` catalog paths.** The catalog path is a **config value**, not
hardcoded in a Snakefile: `DATA_SOURCES = get_config(project_cfg, "data_sources",
...)` and the shell rule reads `-d "{DATA_SOURCES}"` (`Snakefile_model_creation`
lines 31, 108). So moving a catalog is safe **at the Snakefile level** — the fix is
one edit per config, changing e.g. `data_sources: config/deltares_data.yml` →
`data_sources: config/catalogs/deltares_data.yml`. **Catalog-internal paths are
also safe** (verified, blind-spot 5): `deltares_data.yml` resolves its `uri:`
entries against `meta.roots: [C:\data\wflow_global\hydromt]` — an absolute root,
**not** the catalog file's own location — so relocating the catalog file down one
directory does not break any `uri:`. `cmip6_data.yml` uses `gs://` URIs
(location-independent). **The catalog move is therefore a value-only move** (path
strings in configs), no schema change — honoring the CST-scope constraint.

**Build-template paths — a *complete explicit-key inventory*, not a default edit.**
The Snakefile defaults key off `static_dir`:
`model_build_config = get_config(my_cfg, "model_build_config",
f"{static_dir}/wflow_build_model.yml")` and `waterbodies_config` likewise
(`Snakefile_model_creation` lines 39–40), with `static_dir: config` in every config.
Moving templates into `config/templates/` requires **two things done together**:

1. **Update the Snakefile default expressions** to
   `f"{static_dir}/templates/wflow_build_model.yml"` and
   `f"{static_dir}/templates/wflow_update_waterbodies.yml"` (`static_dir` itself
   stays `config`). **This edit is load-bearing, not cosmetic:** grep shows
   `config/snake_config_model_test_linux.yml` sets `static_dir`/`data_sources` but
   **no** `model_build_config`/`waterbodies_config` key — it **relies on the
   default**, so a config *does* exist that the default drives. (arch-3 assumed
   *every* config sets the key explicitly; the Linux test variant does not.)
2. **Rewrite every explicit `model_build_config:` / `waterbodies_config:` key.** The
   correctness hinge is a *complete* inventory — one missed explicit key silently
   keeps `config/wflow_build_model.yml`, which no longer exists, and breaks at build
   with no `--dry-run` signal for the configs that pass their own fixture. **Verified
   inventory (grep `model_build_config:|waterbodies_config:` across
   `config/**/*.yml` + `tests/*.yml`):**

   | File | Lines | Status |
   | --- | --- | --- |
   | `config/snake_config.template.yml` | 41 (`model_build_config`), 43 (`waterbodies_config`) | tracked |
   | `config/snake_config_model_test.yml` | 29, 30 | tracked |
   | `config/snake_config_model_test_gabon.yml` | 29, 30 | **untracked / working-tree-only** (per `baseline-manifest-stale` memory) — a fresh clone will not have it; rewrite it in the working tree if present, note in `MIGRATION.md` |
   | `config/snake_config_model_test_linux.yml` | — (no explicit key) | tracked; **relies on the default** → covered by edit (1) |
   | `tests/snake_config_model_test.yml` | 23 (`model_build_config`), 24 (`waterbodies_config`) | tracked test fixture — **see §"test fixture" below and B3** |

   The projections configs (`snake_config_projections_*`) set neither key (they do
   not build the wflow model), so they are confirmed no-op.

   Keep edit (1) as the default-relying-config fix (required for `model_test_linux`)
   **and** do the complete explicit-key rewrite of edit (2). Neither is "belt and
   suspenders" — both are load-bearing on the current tree.

**`weathergen_config.yml` DOES have a Snakefile consumer.** v1 wrongly said this
file "has no Snakefile-default consumer to worry about." It is consumed via a
**hardcoded `params:` literal**: `Snakefile_climate_experiment:131` sets
`default_config = "config/weathergen_config.yml"` in rule 3.04
`prepare_weagen_config`'s `params:` block;
`src/prepare_weagen_config.py:57` reads it via `read_yml(default_config)` to build
the base weathergen config that drives realization generation — **a direct
computational input.** When the file moves to
`config/templates/weathergen_config.yml` (commit 2), **line 131 must be rewritten**
to `"config/templates/weathergen_config.yml"`, or rule 3.04 raises
`FileNotFoundError` at runtime and WF3 breaks. **This reference is
`--dry-run`-blind**: line 131 is a `params:` string, not a declared `input:`, so the
Snakemake DAG resolver never touches it and the config-split dry-run gate passes
green while WF3 is broken. Line 131 is therefore added to commit-2's named-edit set,
and commit 2's gate gains a **runtime check** that resolves this param (§9): a
`pytest` assertion covering `prepare_weagen_config`'s default-config resolution
and/or a WF3 run to at least rule 3.04. The migration checklist greps `params:`
blocks and raw string literals for `config/`, not only the four a-priori pattern
classes.

**Test fixture `tests/snake_config_model_test.yml` MUST be rewritten.** v1's
verification plan said the reorg "must **not** touch"
`tests/snake_config_model_test.yml` because "tests carry their own fixtures." That
is **false and is deleted.** Verified: `tests/snake_config_model_test.yml:7` sets
`static_dir: config` and **line 23** sets `model_build_config:
config/wflow_build_model.yml` — the fixture resolves its build template from the
**shared `config/` tree**, not from the co-located `tests/wflow_build_model.yml`
(which exists on disk but no config value points at it — `test_model_creation.py:27`
only asserts the *copied* `{project_dir}/config/wflow_build_model.yml` output
exists). So the template move **must** rewrite `tests/snake_config_model_test.yml`
lines 23–24 to `config/templates/...`, or WF1 dry-run
(`test_snakefile_cli_model_creation`) reds with `MissingInputException`
(`model_build_config` is a `copy_config`/`prepare_build_config` input). The fixture
is enumerated in commit-2's edit set (table above). `tests/wflow_build_model.yml` is
**not** load-bearing (no config points at it) and is left in place; if a later
audit finds it dead it can be removed separately. Note `tests/snake_config_model_test.yml`
`data_sources: tests/data/tests_data_catalog.yml` (line 8) points at a test-local
catalog **not** under `config/`, so the catalog move does not touch it — enumerated,
no-op.

**`*_local.yml` gitignore survives.** `.gitignore` line 144 is `*_local.yml`
(unanchored glob), so it already matches `config/workflows/foo_local.yml` — no
`.gitignore` change needed for the pattern. **`*_gabon.yml`**
(`snake_config_model_test_gabon.yml`, an untracked per-machine variant per the
`baseline-manifest-stale` memory) is a user-local config; the design recommends
renaming the pattern to `*_local.yml` when it lands in `config/workflows/` so the
gitignore covers it uniformly — recorded as a migration note, not forced here.

### 3. Settle `dev/` vs `docs/` boundaries (lock item 3)

No files move for this item; it **codifies** the existing convention as an
AGENTS.md rule so it is pointable:

- **`dev/`** = planning, audits, design docs, conventions, roadmap, baseline
  manifest, and **dev-process helpers** under `dev/scripts/` (e.g.
  `check_baseline.py`, `console.py`, probes, `install_weathergenr.R`). Not
  user-facing; not shipped.
- **`docs/`** = user-facing reference (`install.md`, `env_setup_notes.md`, the
  vendored hydromt / hydromt-wflow / wflow user guides, the technical note).

R6 makes it an explicit one-line boundary statement so a newcomer (and future
agents) can cite it. This is a docs-only change (AGENTS.md + README).

### 5. Output layout under `project_dir/` — LEAVE ALONE (lock item 5)

No concrete pain point surfaced. The `logs/`, `benchmarks/`, `config/`,
`hydrology_model/`, `climate_*`, `model_results/` sub-layout under `project_dir/`
is the manifest-anchored contract R3/R4/R5 verified against; touching it would move
`check_baseline` targets for zero discoverability gain. Explicitly out of scope,
per roadmap item 5.

### 6. Fold the top-level runners into `scripts/` (lock item 6)

**Decision: introduce a top-level `scripts/` for production/operational runners**,
rather than folding them into `dev/scripts/`. Rationale: `dev/scripts/` is
dev-process tooling (baseline checks, probes, one-off inspections — not part of a
user's run), whereas `run_snake_test.cmd` / `run_snake_docker.sh` **are** how a user
drives the pipeline. They deserve a user-facing home distinct from dev tooling, and
that home is also where the new `enabled:` wrapper lands (§7), keeping all "how to
run the workflows" entry points in one directory.

```
scripts/
  run_snake_test.cmd               # moved verbatim (Windows dev runner)
  run_snake_docker.sh              # moved verbatim (Linux/Docker runner)
  run_workflows.py                 # NEW — the enabled:-aware wrapper (§7)
```

`run_snake_test.cmd` invokes `snakemake -s Snakefile_model_creation ...` etc. with
paths relative to repo root (`-s Snakefile_...`, `--configfile config/...`), and it
is invoked *from* repo root, so moving the file into `scripts/` does not change
those relative paths — the runner is still run as `scripts/run_snake_test.cmd` from
the repo root. `run_snake_docker.sh` uses `$(pwd)` mounts, likewise repo-root
relative — value-neutral move. Both keep their `config/...` references updated to
the new `config/workflows/` paths (§2). These are Windows-first / Linux-parity
files; the roadmap's "Linux files must continue to parse but are not exercised"
rule holds — `run_snake_docker.sh` is moved and path-updated but not run.

**The runners do NOT invoke the three workflows with identical flags.** Verified:
`run_snake_test.cmd:13` runs `Snakefile_climate_projections` with `--keep-going`;
the other two lines (8, 18) run without it. `--keep-going` lets a workflow-2 run
continue past a partially-failing ensemble member. A flag-uniform wrapper would
silently drop `--keep-going` from projections, changing it from "continue on partial
failure" to "abort on first failure" — a behavioral change to the execution model,
the exact category §7 uses to reject Option A. See §7 for the wrapper's
flag-preservation requirement and its test.

### 7. Operationalize `workflows.<name>.enabled` (lock item 7) — see §"The enabled: operationalization"

Covered in its own section below (the required G1 mechanism decision).

## The `enabled:` operationalization (required position: WRAPPER)

The roadmap offers two mechanisms; the intake requires a recommendation on
**operational simplicity + fidelity to the three-Snakefile entry-point contract,
not novelty**.

**Option A — Snakemake `module:` composition.** A new master Snakefile uses
Snakemake's `module:` directive to `include` each of the three workflows
conditionally on its `enabled:` flag, producing one target rule that spans all
three.

**Option B — Wrapper script (RECOMMENDED).** A small `scripts/run_workflows.py`
reads the `--configfile` YAML, checks each `workflows.<name>.enabled` flag, and
invokes `snakemake -s Snakefile_<name> --configfile <cfg> ...` for exactly the
enabled workflows, in the fixed order (model_creation → climate_projections →
climate_experiment). A disabled workflow is simply not invoked, so none of its
outputs are produced — the clean skip the exit criterion asks for.

**Recommendation: Option B (wrapper).** Three reasons, in decision-criterion order:

1. **Fidelity to the entry-point contract — the two dispositive arguments.**
   (i) **Fourth entry point.** The AGENTS.md contract is "the three `Snakefile_*`
   files are the only entry points; there is no package CLI." Option A introduces a
   **fourth** Snakemake entry point (the master), violating that contract and the
   non-novelty criterion. (ii) **DAG merge is itself the forbidden behavioral
   change.** A master `module:` composition merges the three separately-invoked
   workflows into **one combined DAG** — changing the execution model from "run
   separately, in order" to "one DAG," which R6's non-goal ("no behavioral change
   beyond what `enabled:` requires") forbids outright. *Secondary (downgraded):* a
   combined resolver also **enlarges the cross-workflow rule-ambiguity surface
   generally** (it sees more rules at once — e.g. workflow-1's `region.geojson`
   producer alongside workflow-3's rules). This is *not* the same as reopening the
   specific `CyclicGraphException` R5 resolved: that self-loop
   (`Snakefile_climate_experiment:186` `wildcard_constraints: st_num=[1-9][0-9]*`;
   `test_cli.py:95-111`) is intrinsic to the WF3 DAG and its rule-local constraint
   would carry along into a `module:` include, so a master would not *reopen* it.
   The larger-surface point stands on its own; the "reopens R5's cycle" framing
   (v1) is overstated and removed.
2. **Operational simplicity + non-novelty.** `run_snake_test.cmd` and
   `run_snake_docker.sh` **already are** wrapper scripts that invoke the three
   Snakefiles in sequence. A config-reading wrapper that gates each invocation on
   `enabled:` is their **natural evolution**, not a new mechanism — it lands in the
   same `scripts/` home (§6). It is a *runner*, not a workflow definition, so the
   three Snakefiles remain the only Snakemake entry points and there is still no
   package CLI.
3. **Reversibility.** The wrapper is additive — it adds a file and reads a flag;
   nothing in the Snakefiles changes for it. If it proves unsatisfactory it is
   deleted with no residue. A `module:` master, by contrast, is a structural
   commitment that other rules could start to depend on.

### The wrapper contract — pinned [v3: ext1-02]

`scripts/run_workflows.py` is a NEW component; the external round required its full
contract be specified rather than left implementation-dependent. The contract:

**(a) Accepted config classes.** The wrapper accepts **full orchestration configs
only** — configs that carry a `workflows:` section with **all three** subsections
(`model_creation`, `climate_projections`, `climate_experiment`), each with an
`enabled:` key. On the current tree these are `snake_config.template.yml`,
`snake_config_model_test.yml`, `snake_config_model_test_linux.yml`, and the
untracked `snake_config_model_test_gabon.yml` (grep-verified: each has the three
`enabled:` keys). The **single-workflow projections configs**
(`snake_config_projections_*`) carry **no `workflows:` section at all**
(grep-verified: `snake_config_projections_cmip6_full.yml` has none) — they are
**direct `snakemake -s Snakefile_climate_projections --configfile ...` inputs, not
wrapper inputs**, and the wrapper does not accept them (see (b)).

**(b) Missing `workflows.<name>.enabled` → ERROR, not default-true (justified).**
When the wrapper is pointed at a config, a missing `workflows:` section or a missing
`<name>.enabled` subkey is a **hard error** (nonzero exit, clear message naming the
absent key), **not** a silent default to `true`. Justification, decisive on this
tree: default-true on a projections-only config would make the wrapper attempt
workflows 1 and 3, whose *other* required keys (`model_build_config`,
`stress_test`, `historical_window`, …) are **absent** from that config — producing a
cryptic downstream Snakemake/`get_config` failure deep inside a workflow rather than
a clear "this config is not an orchestration config" message at the wrapper
boundary. Fail-fast at the boundary is the clearer contract and matches the
config-class distinction in (a). (A user who wants to run only projections uses the
direct `snakemake -s` path, unchanged.)

**(c) Boolean values required — a parsed-value contract [v4: ext2-03].** Each
`enabled:` value must **parse** to a real boolean: the wrapper checks
`isinstance(v, bool)` on the post-`yaml.safe_load` value. YAML 1.1 (PyYAML
`safe_load`) resolves unquoted `true`/`false`/`yes`/`no`/`on`/`off` to booleans, so
**every one of those spellings is accepted** — a post-parse check cannot (and does
not try to) distinguish `yes` from `true`. Rejected are exactly the scalars the
parser resolves to a **non-bool**: quoted strings (`"true"`, `"false"`), integers
(`1`, `0`), or any other type — truthiness of a non-empty string would enable a
workflow the author may have meant to disable. Reject rather than coerce. (v3
listed unquoted `yes` among the rejected examples while also noting the parser
resolves it to `True` — a self-contradiction under a parsed-value check; the
contract is the parsed value, so `yes` is valid.)

**(d) Stop on first nonzero Snakemake exit; return that code.** The wrapper invokes
the enabled workflows **in the fixed order** (model_creation → climate_projections →
climate_experiment). If any `snakemake` subprocess returns nonzero, the wrapper
**stops immediately and returns that exit code** — it does **not** continue to the
next workflow. This (i) preserves the "workflow 3 needs workflow 1's artifacts"
guarantee (a failed WF1 must not be followed by a WF3 run against absent/partial
artifacts) and (ii) prevents partial/stale-result runs that look successful. **This
inter-workflow stop is orthogonal to `--keep-going`** (see (f)): `--keep-going` is a
*within-a-single-Snakemake-run* flag that lets one workflow finish past a partially
failing ensemble member; the wrapper's stop-on-nonzero is *between* workflows. They
do not contradict — a projections run may internally keep going yet still return
nonzero if it ultimately failed, at which point the wrapper stops.

**(e) `--cores`/`-c` and extra-arg forwarding.** The wrapper exposes `--cores N`
(default matching the runners today, `-c 3`) and forwards it to **every** `snakemake`
invocation. Any arguments after a `--` sentinel are captured
(`argparse.REMAINDER`) and appended verbatim to **every** invocation, *on top of* the
per-workflow flag map in (f) — so a user can pass e.g. `-- --dry-run` or `--unlock`
to all three without the wrapper enumerating every Snakemake flag. `--configfile` is
supplied by the wrapper from its own `--config` argument; the user does not repeat it.

**(f) Per-workflow flag preservation [v3: ext1-02 + M1 = risk-02].** The wrapper must
**not** emit a flag-uniform `snakemake` call. It preserves each workflow's existing
invocation flags — at minimum `--keep-going` for `climate_projections`
(`run_snake_test.cmd:13`), and **not** on the other two. The per-workflow flag set
comes from a small mapping (a hardcoded default matching the runners today, or a
config-driven `workflows.<name>.snakemake_flags` field). Dropping `--keep-going` on
projections would silently commit the exact behavioral change §7 argument 1 uses to
reject Option A.

**(g) Unit tests for the contract.** The wrapper unit test (commit 4) asserts, by
**monkeypatching `subprocess.run` to capture the argv list** (no real Snakemake run):
(1) all-`true` config → all three Snakefiles invoked in fixed order; (2) projections
invocation carries `--keep-going`, the other two do not (flag parity, M1); (3) a
missing `enabled:` key → wrapper exits nonzero with the naming message (b); (4) a
value the parser resolves to a non-bool (quoted `enabled: "true"`; `enabled: 1`) →
wrapper exits nonzero, **and** an alternative boolean spelling (unquoted
`enabled: yes`) is **accepted** as `True` — the test pins the parsed-value contract
(c) [v4: ext2-03]; (5) a nonzero exit from the
*first* invoked workflow → wrapper stops and does **not** invoke the later ones, and
returns that code (d); (6) `--cores`/`-- <extra>` forwarding reaches every captured
argv (e). The `enabled: false` skip test is separate — see §9 / ext1-03.

**Trade-off acknowledged.** Option A would give a single `snakemake` invocation and
cross-workflow DAG-level parallelism; the wrapper runs the three sequentially (as
today). For CST's rapid first-order basin assessments the workflows are already run
in order (workflow 3 needs workflow 1's artifacts), so sequential invocation is the
operational reality regardless — the wrapper loses no capability the pipeline uses
today, and avoids the combined-DAG behavioral change.

**`enabled:` skip semantics (the one admissible behavior change).** The wrapper's
only new behavior: `enabled: false` for a workflow means the wrapper does not
invoke that Snakefile, so that workflow's outputs are not produced. The Snakefiles
themselves do **not** read `enabled:` (they are still invoked one at a time and
build their own `rule all`); this keeps the flag's semantics entirely in the
runner and avoids threading a skip condition through every rule. A user who invokes
a single Snakefile directly (the current path) is unaffected — `enabled:` governs
the wrapper only. **Disabling a workflow neither deletes that workflow's prior
outputs nor guarantees downstream freshness** — see §9 / ext1-03 for the precise
statement and the downstream-consumption rule. This is documented in the wrapper's
usage and README.

## The functional-decomposition scope fork (required position: DEFER)

**Context.** There is a standing strategic direction
(`modularization-climate-analysis-subworkflow` memory) that climate
analysis/visualization should be a **model-independent subworkflow** — minimal
dependency = region/AOI geometry + data catalog, not the built wflow model —
reusable across workflow-1 QA plots, workflow-2 projections, and workflow-3
experiment. Seeds exist: `extract_climate_grid` (rule 3.02) and the
`monthly_stats_*` rules depend only on `region.geojson`, not the built model.

**The ADR-0002 tension.** ADR 0002
(`dev/decisions/0002-revive-subcatchment-climate-plots.md`, accepted 2026-07-21,
**implemented on this very branch `test/pre06`**) sources the workflow-1
subcatchment climate plots from `mod.forcing.data` — the ERA5-derived forcing
already loaded into the built wflow model (`src/plot_results.py` §4, new
`src/climate_forcing.py`). That **re-couples** the plots to the model build. A fully
modular design would instead source **raw gridded climate** (region + catalog),
decoupled from the build.

**Position: DEFER the functional decomposition to a later milestone.** Rationale,
in order of weight:

1. **Behavior-preservation is dispositive.** Re-sourcing the climate plots from raw
   gridded climate instead of `mod.forcing.data` is a **computational-path change**:
   different input data (raw catalog grid vs the model's regridded forcing),
   different aggregation, different output pixels. R6's own non-goal forbids "any
   behavioral change beyond what `enabled:` requires." Decomposition is therefore
   out of R6's charter on the plain reading — it is not "just layout."
2. **It reworks freshly-accepted code.** ADR 0002's `mod.forcing.data` coupling
   landed on `test/pre06` days ago and was accepted after a visual QA on the real
   gabon model. Pulling decomposition into R6 means unwinding accepted,
   just-landed code inside a milestone whose whole point is *not* to change
   behavior — a poor risk/value trade.
3. **R6 is already a large mechanical migration.** Adding a behavioral
   re-architecture of the climate-analysis data source on top of a ~50-site import
   rewrite dilutes the milestone's single coherent purpose (roadmap principle: one
   coherent purpose per milestone) and muddies the by-construction invariance
   argument (§9) with a genuine value change.

**Forward-compatibility requirement.** Deferring the *behavioral* decomposition does
**not** mean ignoring it in the layout. The §1 package tree is deliberately chosen so
a later decomposition milestone can lift the model-independent helpers into a shared
subpackage with a **mechanical** later move — not "no move at all." v1 over-promised
"no second move" while placing `extract_historical_climate.py` in a workflow leaf,
which is self-undermining. The honest, consistent statement:

- `extract_historical_climate.py` (backs `extract_climate_grid`) is placed in
  `experiment/` today because that is its **sole current consumer**, but it depends
  only on `region.geojson` + catalog. A future `climate_analysis/` subpackage will
  need a **mechanical move** of this one file — cheap because R6 keeps it
  model-independent *in its dependencies*, but it is a move, not zero-move. (Moving
  it to `shared/` now was considered and rejected: G1 **defers** the decomposition,
  so pre-positioning a single file for a deferred, not-yet-designed subpackage would
  invent structure ahead of the decision — the DEFER position argues against
  pre-building.)
- `prepare_climate_data_catalog.py`, `setup_time_horizon.py`, `metrics_definition.py`
  are **not** buried under a workflow-specific leaf (`setup_time_horizon` and
  `metrics_definition` are in `shared/` already; `prepare_climate_data_catalog` sits
  under `projections/`, its sole consumer, and is likewise a mechanical later move).
- `func_plot_signature.py` (carrying `plot_clim`) and `plot_map.py` are already in
  `shared/`, so the plotting primitive a decomposed subworkflow would reuse is
  already shared-scoped — genuinely zero-move.

So R6 stays pure-layout **and** leaves the door open with a *mechanical* (not zero)
follow-on move. The decomposition becomes a clean follow-on milestone (e.g.
`r0X-climate-analysis`) that changes a data source under a locked layout, with its
own baseline diff — which is where a computational-path change belongs.

## Behavior-preservation stance and exact baseline consequence

**Stance (modeled on R1, not R3/R4/R5).** R6 touches **no computational path**:
every change is (a) a file move, (b) a mechanical reference rewrite (import strings
per the §1a table, `script:` paths, R `source()` paths, `run_logged` path,
catalog/template path strings in configs **including the `weathergen_config` param
at `Snakefile_climate_experiment:131`**), (c) a new opt-in runner, or (d) a
docs/AGENTS.md edit. None of these alter any value any rule computes. Scientific
invariance is therefore established **by construction**, resting on **two premises
that v2 makes explicit and gates**:

- **Premise 1 — the reference surface is completely enumerated.** The §1a
  grep-derived stage table (both import forms), the §2/§5 explicit-key inventory,
  the `params:` literal, the R `source()`/`Rscript` strings, and the `run_logged`
  bootstrap are the *whole* surface. A **uniform mechanical transform** *given that
  enumeration* is reviewed as such — but the transform is table-driven, not a
  four-pattern guess (v1's under-count).
- **Premise 2 — the gates exercise the runtime-only paths.** **Green `pytest
  tests/`** after the move (imports the moved modules, dry-runs the Snakefiles) and
  **clean `--dry-run`** on all three Snakefiles prove the DAG resolves with the new
  `script:`/`shell:` paths — **but neither exercises R `source()`, `shell:` bodies,
  the `run_logged` bootstrap, or the `params:` config path.** v2 therefore adds
  **execution-level smokes** to the commits that touch those paths (Verification
  plan): a `run_logged.py` invocation, one `Rscript` `source()`-only run, and a
  runtime resolution of the `weathergen_config` param. `--dry-run` is a *necessary*
  gate, never a *sufficient* one for a runtime-only reference.

with `check_baseline` as a **regression tripwire**, not as proof.

**Flag parity is a preservation requirement.** Behavior preservation is not only
*path* parity but *flag* parity: the wrapper (§7) preserves `--keep-going` on
projections. Dropping it is a behavioral change no less than a DAG merge.

**Exact baseline consequence [v4: ext2-01].** R6 performs **no manifest re-record**,
and no *computed* value changes. But v3's stronger claim — that no manifested value
legitimately changes — was **wrong for one output class: the copied config
snapshots.** `copy_config_files.py` (each workflow's `copy_config` rule) copies the
orchestration config into `{project_dir}/config/snake_config_<workflow>.yml`; those
three snapshots are manifested `yaml` targets (`check_baseline.py` `TARGETS`),
fingerprinted by `fingerprint_yaml` (parse → sorted-JSON → SHA256), so a *value*
change inside them is hash-visible. Commit 2 rewrites path values **inside** the
orchestration config (`data_sources:` → `config/catalogs/...`;
`model_build_config:`/`waterbodies_config:` → `config/templates/...`), so the
post-R6 snapshots **legitimately differ** from a pre-R6 recording even though
computational behavior is preserved. This is an **expected structural difference**,
and it is *not* hand-waved as "classify as expected": it is adjudicated by the
**normalize-then-compare policy** (sub-section below) — only the documented old→new
path mappings are normalized, and **any other difference in a snapshot remains a
real FAIL**. (`copy_config_files.py` also copies the build/waterbodies templates and
the data catalog(s) into `{project_dir}/config/` — un-manifested, but seen by the
full-tree diff; those files move verbatim, so their copies are content-identical and
the policy is a no-op for them.) Honest caveat on the tracked manifest itself (per
`MEMORY.md` / `dev/milestones/r01/baseline_diffs.md` /
`check_baseline.py` docstring): the manifest is **stale and mixed-provenance** —
wf1's discharge slice was restored (t260719a) but wf2/wf3 slices are pre-restoration,
and the R1 audit found it recorded from an untracked 3-model config while the
canonical uses 8. So a full `check_baseline check` may show pre-existing wf2/wf3
drift **unrelated to R6**.

**Run-relative gate — scope, tooling, determinism [v2: M2 = risk-03; v3: ext1-01].**
R6 uses `check_baseline` as a **per-workflow, run-relative** gate rather than a match
against the stale committed manifest. Three properties the gate must state honestly:

- **(i) Coverage boundary — what the gate proves and what it does NOT.** The
  run-relative gate proves invariance **only for the outputs the manifest
  fingerprints**. That slice is thin and mixed-provenance: wf1 discharge
  (`hydrology_model/run_default/output.csv`, a tolerance comparator) + the `rule all`
  PNGs/config snapshot/`outlet_index.csv`; wf2/wf3 are the pre-restoration slice.
  **Explicitly NOT covered:** wf2/wf3 staticmaps, `wflow_sbm.toml`, change-factor
  NetCDFs, and any output outside the manifested `rule all` set. An R6 change to an
  unmanifested output would pass the manifest-slice gate silently. **Mitigation:** at
  the milestone gate, run at least one **full-`project_dir` pre/post semantic diff**
  (a recursive compare of the whole output tree, not just the manifest slice, using
  per-type comparators — see the ext1-04 sub-section below) to cover the uncovered
  classes.
- **(ii) Tooling — the tool has no one-shot two-run compare, but `--manifest`
  redirects both the manifest AND its sidecars [v3: ext1-01].**
  `dev/scripts/check_baseline.py` exposes `record`, `check`, and `compare` (verified
  docstring). `check` compares a fresh run against a manifest; `compare` operates on
  **single discharge CSVs** only (`--ref A/output.csv --cur B/output.csv`). There is
  **no native full-slice two-run compare**. **Crucially, `record` and `check` both
  key their reference off `--manifest` (default `dev/baseline/manifest.json`), and
  the discharge-reference sidecar directory is `ref_dir = args.manifest.parent`
  (`cmd_record`/`cmd_check`, verified).** So pointing `--manifest` at a **scratch
  path** redirects *both* the manifest read/write *and* the `discharge_ref/`
  sidecars away from the tracked artifact. The run-relative gate is therefore a
  **manual three-step procedure using `--manifest <scratch>`** (below) that **never
  records to, overwrites, or restores the tracked `dev/baseline/manifest.json`** —
  the tracked artifact is untouched, so no `git checkout -- dev/baseline/manifest.json`
  restore is needed and no modified-file can block or contaminate a `git checkout`.
  (v2's sequence recorded into the tracked manifest, copied it aside, then relied on
  `git checkout -- dev/baseline/manifest.json` to restore it — the external round
  faulted that as mutating a tracked file across a checkout. v3 removes the mutation
  entirely by using the existing `--manifest` redirect. This is *not* a new flag: it
  is the existing `--manifest` option used with a scratch destination.)
- **(iii) Determinism — weathergenr must be seed-pinned.** "Byte-for-byte on the
  manifest slice" presumes reproducible runs. The stochastic realizations are
  seed-pinned: `config/.../weathergen_config.yml` sets `seed: 123` (verified). The
  gate's precondition is that the **pre-R6 and post-R6 runs use the identical
  `seed:` value** (they do, since the config moves verbatim). If the seed ever
  differed, the gate would misread weathergen non-determinism as R6 drift (or mask
  R6 drift as noise).
- **(iv) Same `--project-dir` on `record` and `check` — the correctness hinge
  [v3: ext1-01].** `check_baseline` is a **temporal** A/B tool, not a spatial one:
  manifest rows are keyed by the **resolved output path** `resolve(template,
  project_dir)` (`compute_manifest`/`record`, verified), and `cmd_check` matches
  recorded-vs-current rows **by exact path key** (`if path not in current: continue`,
  ~L571) with an orphan check `set(current) - set(rec_targets)` (~L579). So `record`
  and `check` **must be given the identical `--project-dir`**. The A/B is achieved in
  *time*: record the reference into `examples/test_local` on the pre-R6 tip, then
  **re-run all three workflows into the same `examples/test_local`** on the post-R6
  tip (overwriting it), then `check` the same dir. Recording with a *different*
  `--project-dir` (e.g. a stashed copy) would key every row under a path the check
  side never produces → every fingerprinted target skipped and every current target
  flagged an orphan → a vacuous FAIL that proves nothing. Recording at the pre-R6 tip
  is safe against comparator-version skew precisely because **R6 does not modify
  `check_baseline.py`** (Hard-Constraints scope; the tool is unchanged across the two
  tips).

**Concrete run-relative command sequence [v3: ext1-01 — no tracked-file mutation,
same `--project-dir` on both calls]** (per workflow slice; `project_dir` =
`examples/test_local` by default):

```
# 1. Pre-R6 tip: run all three workflows INTO examples/test_local, then RECORD to a
#    SCRATCH manifest (never the tracked one). Also stash the tree for the step-3 diff:
git checkout <pre-R6-tip>
# ... run the three workflows (seed 123) via run_snake_test.cmd ...
python dev/scripts/check_baseline.py record \
    --project-dir examples/test_local --manifest /tmp/preR6/manifest.json
cp -r examples/test_local /tmp/projdir-preR6     # stash pre-R6 tree — used ONLY by the step-3 diff

# 2. Post-R6 tip: re-run all three workflows INTO THE SAME examples/test_local
#    (overwrite it; seed 123 unchanged):
git checkout <post-R6-tip>
# ... run the three workflows via scripts/run_snake_test.cmd ...

# 3. CHECK the post-R6 tree against the scratch reference — SAME --project-dir as record:
python dev/scripts/check_baseline.py check \
    --project-dir examples/test_local --manifest /tmp/preR6/manifest.json
#    (add --workflow model_creation etc. to gate a single slice; --tolerance 1e-9
#     for cross-env numeric comparison. dev/baseline/manifest.json is NEVER touched.)
#    EXPECTED FAIL [v4: ext2-01]: exactly the three copied-config snapshot rows
#    ({project_dir}/config/snake_config_*.yml) mismatch by hash, because commit 2
#    rewrote path values inside the orchestration config. Adjudicate exactly those
#    rows with the normalize-then-compare policy (sub-section below), comparing the
#    stashed /tmp/projdir-preR6/config/ copies against the post-R6 files. Any OTHER
#    failing row is a real FAIL.

# 4. Full-tree SEMANTIC diff for the un-manifested slice (at least once, milestone gate):
#    /tmp/projdir-preR6  vs  examples/test_local — see the ext1-04 sub-section
#    (per-type comparators, not a raw byte diff).
```

`record` (pre-R6 tip) and `check` (post-R6 tip) both pass the **identical
`--project-dir examples/test_local`** and both write only to `/tmp/preR6/`, so the
tracked `dev/baseline/manifest.json` is never recorded to, overwritten, or restored —
no `git checkout` crosses a modified tracked file. The stashed `/tmp/projdir-preR6`
tree feeds **only** the step-4 ext1-04 semantic diff, never `check_baseline check`.
(On Windows/PowerShell the copies use `Copy-Item -Recurse`; the `--manifest` redirect
and the same-`--project-dir` rule are the contract, the shell is incidental.)
**The only behavioral change anywhere in R6 is `enabled: false` causing a workflow's
outputs not to be produced — an absence, not a value change to any *computed*
output. The one expected *content* change to a produced output is the copied-config
snapshot path rewrite, which alters no computed value and is gated by the
normalize-then-compare policy below [v4: ext2-01].**

**Copied-config snapshots — the normalize-then-compare policy [v4: ext2-01].** The
policy is a precise rule, not a classification: it normalizes **only** the
documented old→new path mappings and requires everything else equal.

- **Scope.** Every YAML file `copy_config_files.py` lands under
  `{project_dir}/config/`: the three manifested snapshots
  (`snake_config_model_creation.yml`, `snake_config_climate_projections.yml`,
  `snake_config_climate_experiment.yml`) plus the un-manifested copied
  templates/catalogs (for which the normalization is a no-op — they move verbatim,
  so their copies must be structurally identical).
- **The rule.** (1) Parse both sides with `yaml.safe_load`. (2) Apply a **fixed
  mapping table** to the pre-R6 (reference) document: the same old→new path map
  recorded in `MIGRATION.md`, restricted to the path-valued keys commit 2 rewrites
  (`data_sources`, `model_build_config`, `waterbodies_config`). For each mapped key
  present: if its value equals the documented OLD path **exactly**, replace it with
  the documented NEW path; any other value is left untouched (and will fail the
  equality step). No other key, value, or structure is normalized — no key-level
  ignore list, no tolerance. (3) Require **deep structural equality** of the two
  parsed documents after step (2). Any residual difference — an unmapped path
  value, a changed non-path value, a missing or extra key — is a **real FAIL**.
- **Used identically in BOTH gates.** (a) *Run-relative gate:* `check_baseline
  check` against the pre-R6 scratch manifest is expected to FAIL by hash on exactly
  the three snapshot rows; each is then adjudicated by this rule (stashed
  `/tmp/projdir-preR6/config/` copy vs the post-R6 file), and the gate passes iff
  every non-snapshot row passes `check_baseline` AND every snapshot row passes this
  rule. (b) *Full-tree diff:* the walker dispatches `{project_dir}/config/*.yml` to
  this same comparator (dispatch list in the next sub-section).
- **Tooling.** The comparator is part of the same bounded `dev/` tooling addition
  as the walker and `.toml` comparator (see the commit-plan tooling note);
  `check_baseline.py` is unchanged — `fingerprint_yaml` remains the manifest-slice
  comparator, and the adjudication runs beside it.

**Full-`project_dir` pre/post diff — semantic, not raw byte [v3: ext1-04; v4:
ext2-02].** The
milestone full-tree diff (§9 milestone gate; risk-03/M2 mitigation) must **not** be a
raw byte diff, because semantically identical outputs differ bytewise on
serialization metadata (NetCDF `history`/`Conventions`/timestamps, benchmark
wall-times, log timestamps) — a raw diff would be noisy and its failures
unclassifiable. The policy — **reusing `check_baseline.py`'s existing per-type
comparators where they are strict enough** (CSV, PNG, discharge) **and specifying
stricter ones where they are not** (`.nc` element-wise [v4: ext2-02]; `.toml`; the
copied-config YAML policy [v4: ext2-01]):

- **Walk `project_dir` recursively** and dispatch each file to a comparator by
  extension, driven by the directory walk (not the fixed thin `TARGETS` list — the
  whole point of this diff is the *un-manifested* slice: wf2/wf3 staticmaps,
  `wflow_sbm.toml`, change-factor NetCDFs).
- **`.nc` (NetCDF) — ELEMENT-WISE, not the aggregate fingerprint [v4: ext2-02].**
  v3 delegated `.nc` to `fingerprint_nc`/`diff_nc`, whose equality criterion is
  per-variable **aggregate** stats (shape/dtype/count_non_nan/min/max/mean/std).
  That is not a valid equality criterion for this gate: spatial or temporal
  **permutations, and compensating cell-level changes, preserve every aggregate**
  while materially changing the dataset — and this diff is the design's only
  stated coverage for the un-manifested staticmaps and change-factor NetCDFs. The
  full-tree `.nc` comparator is therefore **element-wise**:
  - **Dimensions:** identical names and sizes.
  - **Coordinates:** identical coordinate-variable sets, compared as **labels AND
    order** — element-by-element in stored order, with **no sorting or
    realignment** before comparison (a permuted coordinate must FAIL, never be
    aligned away). Non-numeric coordinates exact; numeric coordinates under the
    same tolerance rule as data values.
  - **Variable values:** identical variable sets; per variable identical shape and
    dtype, then values compared **element-by-element at aligned positions**: the
    **NaN masks must match exactly** (NaN at a position on one side iff NaN at the
    same position on the other), and every finite pair must satisfy the stated
    numerical tolerance — the `_within_tol` relative rule
    (`|c − r| / max(|r|, |c|, 1e-300) ≤ tol`; default `1e-9` for cross-env, `0` =
    exact). Any mask mismatch or out-of-tolerance element FAILs, reporting the
    first offending variable and position.
  - **Attributes:** dataset- and variable-level attrs equal after excluding
    **`VOLATILE_NC_ATTRS`** (reused verbatim from `check_baseline.py:83`:
    `history`, `creation_date`, `Conventions`, `software`, `software_version`,
    `production_date`/`creation_time`, `date_created`/`date_modified`).
  - **Summary fingerprints are NOT an equality criterion anywhere in this gate.**
    `fingerprint_nc`/`diff_nc` stay **as-is for the manifest-slice targets**
    (`check_baseline.py` unchanged, per the Hard-Constraints scope note); the
    full-tree comparator is deliberately **stricter** — the belt-and-suspenders
    element-wise check for the un-manifested NetCDFs. Like the `.toml` comparator,
    it is *specified here and built in the implementation task-brief*: a small
    routine (open both sides via `xarray.open_dataset` and apply the checks above;
    the rules stated here — not a library default — are the contract).
- **`.csv`:** normalized-hash comparator `fingerprint_csv`/`diff_hashed` (CRLF-
  normalized, trailing-whitespace-stripped); the wf1 discharge `output.csv` uses the
  tolerance `compare_discharge` comparator (ADR-0001 step 6) as in the manifest gate.
- **`.yml`/`.yaml` under `{project_dir}/config/` (the copied config snapshots)
  [v4: ext2-01]:** the ext2-01 **normalize-then-compare** policy (previous
  sub-section) — parse, apply only the documented old→new path map, require all
  other keys and values equal.
- **`.toml` (e.g. `wflow_sbm.toml`):** parse-and-normalize compare (load, sort keys,
  compare structurally) rather than byte compare, so key-ordering or comment churn is
  not flagged. (This is the one comparator not already in `check_baseline.py`; it is a
  small addition specified for the task-brief — see the commit plan note.)
- **`.png`:** size-tolerance comparator `diff_png` (`PNG_TOLERANCE_FRAC = 0.10`).
- **Exclusion list (compared as absent, never byte-diffed):** everything under
  `logs/` and `benchmarks/` (wall-time/timestamp non-determinism), plus any
  `.snakemake/` metadata. Files with an unrecognized extension are compared by
  normalized hash and any mismatch reported for manual classification.

This makes the full-tree gate deterministic and its failures classifiable. (The
*policy* is fixed here; the small recursive walker + the three specified
comparators — `.toml` normalized, element-wise `.nc` [v4: ext2-02], copied-config
YAML normalize-then-compare [v4: ext2-01] — are a bounded addition specified for
the implementation task-brief; the CSV/PNG/discharge comparators are reused from
`check_baseline.py`, which is itself unchanged.)

## Verification plan

Per-stage gates (each stage = one commit or a tight commit group; §10). A stage is
not done until its row passes. **Every row that touches a runtime-only path carries
an execution smoke — `--dry-run`/`pytest` alone do not gate R, shell bodies,
`run_logged`, or `params:` config paths.**

| Stage | Proof the repo still runs | Proof of baseline preservation |
| ----- | ------------------------- | ------------------------------ |
| Package move (`src/` → `blueearth_cst/`) | `pytest tests/` green (imports resolve — both `from src.` and `from src import` forms per §1a); all three Snakefiles `--dry-run` clean. **Execution smokes:** (a) `tests/test_run_logged.py` (rewritten) imports `main` and calls `run_and_tee`, proving the `run_logged` bootstrap resolves repo root; (b) invoke `run_logged.py` on a trivial command; (c) one `Rscript --vanilla` `source()`-only smoke proving the R `source()` path rewrite. `--dry-run` does NOT exercise any of these. | No computational path touched → by construction; `check_baseline check --workflow model_creation --manifest /tmp/preR6/manifest.json` on the wf1 slice matches the run-relative reference (scratch manifest; tracked artifact untouched) |
| Config split (workflows/catalogs/templates) | All three Snakefiles `--dry-run` clean on the moved configs (proves `-d` catalog paths + `static_dir` template defaults resolve); `pytest tests/` green. **Note: the reorg DOES touch `tests/snake_config_model_test.yml` (lines 23–24 → `config/templates/...`)** — the v1 "must not touch" claim is deleted; that fixture is in the edit set. **Runtime check:** a `pytest` assertion (or WF3 run to rule 3.04) that `prepare_weagen_config` resolves `default_config` at `config/templates/weathergen_config.yml` — `--dry-run` is blind to this `params:` string. | Catalog move value-only (`meta.roots` absolute; verified); template-key inventory (§2/§5) + `weathergen_config` param edit produce identical build inputs → wf1 slice unchanged — **except the copied-config snapshot rows, EXPECTED to differ by hash and adjudicated by the ext2-01 normalize-then-compare policy [v4: ext2-01]** |
| Runner move (`scripts/`) | `scripts/run_snake_test.cmd` runs the three workflows end-to-end from repo root; `run_snake_docker.sh` parses (not exercised — Linux parked) | End-to-end run + `check_baseline` on all three workflow slices, run-relative (scratch `--manifest`); **full-`project_dir` pre/post SEMANTIC diff at least here [v3: ext1-04; v4: ext2-02 — element-wise `.nc`]** to cover the un-manifested slice |
| `enabled:` wrapper | `pytest tests/` adds the wrapper unit test suite (§7(g)); `scripts/run_workflows.py` on the seed config invokes the three in order. **Contract assertions [v3: ext1-02]:** monkeypatched `subprocess.run` captures argv and asserts (1) fixed-order invocation, (2) `--keep-going` on projections only, (3) missing `enabled:` → nonzero exit, (4) parsed-non-bool `enabled:` (quoted `"true"`, `1`) → nonzero exit while unquoted `yes` is accepted [v4: ext2-03], (5) first-workflow nonzero → later workflows NOT invoked + that code returned, (6) `--cores`/`-- <extra>` reaches every invocation. | — (additive; no output change when all enabled) |
| **`enabled: false` skip behavior [v3: ext1-03]** | A dedicated test in a **FRESH temp `project_dir`** (`tmp_path`): set `workflows.climate_projections.enabled: false` in a scratch config pointing at the fresh dir, run the wrapper with `subprocess.run` monkeypatched to capture argv, and assert **at the subprocess boundary** that (a) the wrapper does **not** invoke `Snakefile_climate_projections` and (b) it **does** invoke the other two — invocation/non-invocation, **not** output presence. Assert the inverse (all `true` → all three invoked). See the ext1-03 note below for why a fresh dir and boundary assertion are required. | Non-invocation is the intended behavior, not a drift; no output is asserted (a reused dir could carry stale outputs — ext1-03) |
| Docs (AGENTS.md/CLAUDE.md/README + MIGRATION.md) | Markdown renders; `MIGRATION.md` old→new map is complete (every moved path present, incl. the untracked `*_gabon.yml` note) | — (docs-only) |

**The `enabled: false` skip test — fresh `project_dir`, boundary assertion,
downstream rule [v3: ext1-03].** The external round faulted the v2 skip test for
asserting "that workflow's `rule all` outputs are absent" against a possibly-reused
`project_dir`. Three corrections, all required:

1. **Fresh temp `project_dir`.** Disabling a workflow does **not** delete its prior
   outputs. Against a reused `project_dir`, a disabled workflow's outputs can still be
   present from an earlier run — so an "outputs absent" assertion **false-fails** even
   when the wrapper correctly skipped invocation. The skip test therefore runs in a
   **fresh temporary `project_dir`** (pytest `tmp_path`), with the scratch config's
   `project_dir` pointed at it.
2. **Assert invocation, not output presence.** The test asserts **at the subprocess
   boundary** (monkeypatched `subprocess.run` capturing the argv list — the same
   mechanism the flag-parity test uses) that the disabled workflow's `Snakefile` is
   **not** invoked and the enabled ones are. This is a direct, deterministic check of
   the wrapper's decision, independent of any filesystem state.
3. **Documented semantics — no deletion, no freshness guarantee, downstream rule.**
   The design states explicitly: **disabling a workflow neither deletes its prior
   outputs nor guarantees downstream freshness.** The downstream question — *may an
   enabled downstream workflow run against a disabled prerequisite's pre-existing
   artifacts?* — is answered **yes, by design**: the wrapper invokes each Snakefile
   **independently** and enforces **no prerequisite-freshness check**. An enabled WF3
   consumes whatever WF1 artifacts already exist on disk, or fails with
   `MissingInputException` if they are absent — **identical to invoking a single
   Snakefile directly today**. The wrapper does **not** build freshness enforcement
   (that is beyond `enabled:`'s charter and would be a new behavior R6's non-goal
   forbids). This is documented in the wrapper usage/README so a user who disables a
   prerequisite understands they are responsible for the staleness of what downstream
   consumes.

**Full milestone gate (before tag `r06-refactor`):** a clean `pytest tests/`
(expected count carried forward from R5's 120 passed / 3 skipped / 7 xfailed, plus
the new wrapper contract suite + skip test), clean `--dry-run` on all three
Snakefiles, the commit-1 execution smokes, a full end-to-end run of all three
workflows via `scripts/run_snake_test.cmd`, `check_baseline` matching the
run-relative reference (scratch `--manifest`, tracked manifest untouched) on every
workflow slice — with exactly the three copied-config snapshot rows expected to
FAIL by hash and adjudicated by the ext2-01 normalize-then-compare policy
[v4: ext2-01] — and a **full-`project_dir` pre/post SEMANTIC diff** once (ext1-04
policy with the ext2-02 element-wise `.nc` comparator) to cover the un-manifested
output classes (§9).

## Commit plan

Prefix `r06:`, tag `r06-refactor` at the end (roadmap commit strategy). One logical
change per commit; each independently runnable/verifiable **except** the one
unavoidably atomic move, handled explicitly below (blind-spot 2).

1. `r06: move src/ to blueearth_cst/ package and rewrite all references`
   — **the unavoidably atomic commit.** A pure `git mv` is **not** independently
   runnable: the instant `src/` moves, the ~50 `from src.`/`from src import` sites
   (§1a table), 15 `script:` paths, the two R `shell:` paths + `source()` calls, the
   three `run_logged=` constructions, and `run_logged.py`'s own bootstrap all break.
   **Chosen handling: one atomic move+rewrite commit**, justified as a single
   **table-driven** mechanical transform (§1a) gated by `pytest tests/` + clean
   `--dry-run` **+ the execution smokes** (run_logged invocation; one R `source()`
   smoke; `test_run_logged.py` rewritten and passing). To keep it reviewable despite
   its size: (i) do the `git mv`s and the reference rewrites as separate *staged
   hunks* the reviewer can read as "moves" vs "rewrites"; (ii) the rewrite is
   enumerated in the commit body **from the §1a table** — both import forms, the
   per-module stage segment, the `script:`/R/`run_logged` path edits — not a
   four-pattern find-replace; (iii) the `run_logged.py` depth fix, the three
   `run_logged=` edits, and the `test_run_logged.py` rewrite are called out by name.
   **Open-question note:** the commit-1 `pytest`+`--dry-run` gate is
   `--dry-run`-blind to at least three runtime-only reference classes — R `source()`,
   the `run_logged` bootstrap, and (in commit 2) the `weathergen_config` param — so
   the v2 execution smokes are what actually close that gap. This is precisely why
   the rejected re-export shim's *per-commit-green* property has more value than v1's
   "more moving parts" framing credited: without the smokes, a run-time-only breakage
   would surface only at the milestone e2e run. The smokes are the mitigation; the
   shim remains the open G1 alternative. **Alternative considered and rejected for
   this repo:** a transitional `src/` → `blueearth_cst/` re-export shim so move and
   rewrite are two runnable commits, deleted in a third — rejected as more moving
   parts for a solo repo where one gated atomic commit **plus execution smokes** is
   cleaner; recorded so the reviewer can overrule if they prefer the shim.
2. `r06: split config/ into workflows/, catalogs/, templates/` — `git mv` the three
   config bins + **the complete explicit-key inventory rewrite** (§2/§5 table:
   template 41/43, model_test 29/30, gabon 29/30 [untracked], **tests fixture
   23/24**) + the Snakefile template-default expressions (`static_dir` stays
   `config`; defaults gain `/templates/`, load-bearing for `model_test_linux`) +
   **`Snakefile_climate_experiment:131` `default_config`** → `config/templates/...`
   + the `data_sources:` catalog-path edits. Runnable: `--dry-run` clean, `pytest`
   green, **+ runtime check that rule 3.04 resolves the moved `weathergen_config`**
   (the `params:` path is `--dry-run`-blind).
3. `r06: move runners to scripts/ and update config paths` — `git mv`
   `run_snake_test.cmd` + `run_snake_docker.sh` into `scripts/`; update their
   `config/...` references to `config/workflows/...`; **preserve the per-workflow
   flags verbatim** (`--keep-going` stays on the projections line). Runnable via the
   moved runner.
4. `r06: add scripts/run_workflows.py enabled-aware wrapper` — the wrapper
   implementing the pinned §7 contract (config-class acceptance; missing/non-bool
   `enabled:` → error; stop-on-first-nonzero-exit; `--cores`/`--`-extra forwarding;
   per-workflow flag map) + its unit-test suite (**§7(g): fixed-order, flag parity,
   missing-key, non-bool, stop-on-nonzero, forwarding**) + the `enabled: false`
   skip-behavior test (**fresh `tmp_path` project_dir, boundary assertion — ext1-03**).
   Additive.
5. `r06: codify dev/ vs docs/ boundary and new layout in AGENTS.md + README` —
   docs-only; the AGENTS.md Repo Map, Conventions, and References updated to the new
   tree; README run instructions point at `scripts/`; the wrapper usage documents the
   config-class contract and the "no deletion / no freshness guarantee / downstream
   consumes pre-existing artifacts" semantics (ext1-02/ext1-03).
6. `r06: add MIGRATION.md mapping every moved path old to new` — the complete
   old→new map (roadmap exit criterion) for downstream forks / user-local configs,
   including the untracked `*_gabon.yml` note.

**Small in-scope tooling note (ext1-01 / ext1-04 / ext2-01 / ext2-02).** Neither
the run-relative gate nor the semantic full-tree diff requires *editing*
`dev/scripts/check_baseline.py`: the run-relative gate uses the **existing
`--manifest` redirect** (ext1-01), and the CSV/PNG/discharge comparators already
exist and are reused (ext1-04). The genuinely new dev tooling is the milestone
full-tree **walker** plus the three small comparators it dispatches to: the
**`.toml` normalized comparator**, the **element-wise `.nc` comparator**
(ext2-02 — the full-tree gate must not use the aggregate `fingerprint_nc`/`diff_nc`
stats as its equality criterion), and the **copied-config YAML normalize-then-compare
comparator** (ext2-01 — also used to adjudicate the three expected snapshot FAILs in
the run-relative gate). All are `dev/` tooling in scope to add — specified as a
bounded addition for the implementation task-brief, not built in this design run.
`check_baseline.py` itself is unchanged (`fingerprint_nc` stays the manifest-slice
comparator).

(Commits 2–6 are each independently runnable/verifiable. Commit 1 is the single
atomic exception, justified above.)

## Alternatives considered

- **Keep `src/` flat (do nothing / cosmetic only).** Rejected: the flat 27-module
  directory is the named lock-item-1 pain point; discoverability (a newcomer
  locating a workflow's code without grep) is a stated decision criterion the flat
  layout fails. A package boundary is the minimal structural change that satisfies
  it.
- **Package split by *kind* (`setup/`, `plot/`, `analysis/`) instead of by
  *workflow stage*.** Rejected: the roadmap names per-**stage** submodules (model,
  projections, experiment, weathergen), and the whole Phase-2 framing is
  vertical-by-workflow. By-kind would scatter one workflow's code across
  subpackages — the opposite of the "locate a workflow's code" criterion.
- **`enabled:` via Snakemake `module:` master Snakefile.** Rejected (§7): adds a
  fourth entry point (violates the three-Snakefile contract) and merges the three
  workflows into one combined DAG (a behavioral change to the execution model, and a
  generally larger cross-workflow rule-ambiguity surface), violating the non-goal and
  the non-novelty criterion.
- **`enabled:` via a rule-level skip inside each Snakefile** (each `rule all` reads
  its own `enabled:` and empties its target list when false). Rejected: threads the
  flag through three Snakefiles, is easy to get subtly wrong (an emptied `rule all`
  vs a `checkpoint`), and still leaves "run them in order" to user discipline — the
  wrapper solves both in one additive file.
- **Wrapper defaults missing `enabled:` to true (instead of erroring).** Rejected
  (§7(b)): default-true on a projections-only config (no `workflows:` section) would
  attempt WF1/WF3 whose other required keys are absent → a cryptic downstream failure
  instead of a clear boundary error. Fail-fast at the wrapper boundary, scoped to
  full-orchestration configs, is the clearer contract.
- **Status-quo user discipline for `enabled:`** (leave the flag documentary).
  Rejected: it is a named roadmap exit criterion to operationalize it; leaving it
  dormant fails the milestone.
- **Big-bang single refactor commit for everything.** Rejected: unreviewable and
  not stage-verifiable; violates the staged/reversible criterion. Only the package
  move is atomic, and only because its references cannot be split from the move.
- **Include the functional decomposition in R6.** Rejected (§8): it is a
  computational-path change forbidden by R6's non-goal, and reworks freshly-accepted
  ADR-0002 code. Deferred to a follow-on milestone; R6's layout keeps it
  forward-compatible with a *mechanical* later move.
- **Two-commit move via a transitional re-export shim** (vs the atomic move+rewrite,
  §10 commit 1). Rejected for this solo repo as more moving parts than a single
  gated atomic commit **plus execution smokes**; recorded so the reviewer can prefer
  it (its per-commit-green property is credited in commit 1's open-question note).
- **Fold runners into `dev/scripts/` instead of a new `scripts/`.** Rejected (§6):
  conflates user-facing runners with dev-process tooling; the `enabled:` wrapper
  wants a user-facing home anyway.
- **Move `extract_historical_climate.py` to `shared/` now** (to make the later
  decomposition zero-move). Rejected (§8): G1 defers the decomposition, so
  pre-positioning a single file for a not-yet-designed subpackage invents structure
  ahead of the decision; the file is kept in `experiment/` (its sole consumer) and
  the forward-compat claim is stated as a *mechanical* later move, not zero-move.
- **Run-relative baseline gate that records into the tracked manifest and restores
  it via `git checkout --`** (v2's sequence). Rejected (§9, ext1-01): mutating a
  tracked file across a `git checkout` can be blocked or contaminate the checkout;
  the existing `--manifest <scratch>` redirect avoids the mutation entirely.
- **Raw byte diff for the full-tree milestone gate.** Rejected (§9, ext1-04):
  semantically identical NetCDF/TOML outputs differ bytewise on serialization
  metadata, making the gate noisy and its failures unclassifiable; the per-type
  comparators (§9 ext1-04/ext2-02 policy) are used instead.
- **Aggregate-stat NetCDF fingerprints (`fingerprint_nc`/`diff_nc`) as the
  full-tree equality criterion** (v3's ext1-04 resolution). Rejected (§9, ext2-02):
  permutations and compensating cell-level changes preserve every aggregate stat
  (shape/dtype/count/min/max/mean/std) while materially changing the dataset, and
  the full-tree gate is the only coverage for the un-manifested NetCDFs. The
  full-tree `.nc` comparator is element-wise (dims, coordinate labels+order, exact
  NaN masks, per-element tolerance, non-volatile attrs); the aggregate fingerprint
  remains the manifest-slice comparator inside unchanged `check_baseline.py`.
- **Blanket-classify the post-R6 copied-config snapshot mismatch as "expected", or
  exclude `{project_dir}/config/` from the gates.** Rejected (§9, ext2-01): an
  unspecified expected-fail invites dismissing genuine drift alongside the
  structural one, and exclusion would blind the gate to a corrupted snapshot. The
  snapshots are instead adjudicated by the narrow normalize-then-compare policy —
  only the documented old→new path map is normalized; everything else must be
  equal.
- **Reject unquoted `yes`/`no`/`on`/`off` for `enabled:` by validating YAML source
  tokens before parsing** (the canonical-spelling resolution of ext2-03). Rejected
  (§7(c)): it requires a second, token-level pass over the YAML source for one key,
  and the post-parse `isinstance(v, bool)` check cannot see spelling anyway. The
  parsed-value contract — accept every scalar the parser resolves to a boolean,
  reject the rest — is simpler, implementable, and testable, and still rejects the
  dangerous case (quoted `"true"` enabling a workflow via string truthiness).

## Risks and open questions

1. **The atomic package-move commit is large.** ~50 import sites + 15 `script:`
   paths + 2 R paths + `run_logged` depth fix in one commit. Mitigation: the rewrite
   is a **table-driven** transform (§1a) enumerated in the commit body, gated by
   `pytest` + `--dry-run` **+ execution smokes**; staged as readable "move" vs
   "rewrite" hunks. **Open question for G1:** accept the atomic commit (with the v2
   execution smokes), or mandate the transitional shim (two runnable commits)? — the
   shim's per-commit-green value is now weighed explicitly (arch-8).
2. **`run_logged.py` `sys.path` depth (blind-spot 3).** Moving it to
   `blueearth_cst/shared/` breaks its two-`dirname` repo-root walk. Named as a
   required edit in commit 1. **The guard is NOT `--dry-run`** (which does not
   execute the `shell:` body): it is the rewritten `tests/test_run_logged.py`
   (imports `main`, exercises the bootstrap) plus the commit-1 `run_logged.py`
   execution smoke plus the milestone end-to-end run.
3. **The R layer's relative `source()` and shell paths (blind-spot 4).**
   `source("./blueearth_cst/weathergen/global.R")` and the two `Rscript
   .../weathergen/*.R` shell strings must move together; a partial rewrite breaks
   weathergenr at `source()` time, invisible to the Python-only `--dry-run`.
   Mitigation: the migration checklist greps **all** `*.R` for `source(` /
   `src/weathergen` / `./src` literals (arch-4), not just the two entrypoints; the
   commit-1 `Rscript` `source()`-only smoke and the milestone end-to-end run
   exercise the R layer.
4. **Build-template paths + `weathergen_config` param (blind-spot 1, B1).** If the
   template move updates some config keys but misses one (or misses the `--dry-run`-
   blind `params:` literal at `Snakefile_climate_experiment:131`), a config silently
   reads a missing/old template and breaks at build. Mitigation: commit 2 does the
   **complete explicit-key inventory** (§2/§5) + the line-131 param edit; a
   **runtime check** (not `--dry-run`) confirms rule 3.04 resolves the moved param;
   wf1 `check_baseline` confirms build inputs identical.
5. **Manifest is stale/mixed-provenance + run-relative gate coverage.**
   `check_baseline` against the committed manifest may show pre-existing wf2/wf3
   drift unrelated to R6. Mitigation (§9): a **run-relative** baseline (pre-R6 tip vs
   post-R6) per workflow, gated via the existing `--manifest <scratch>` redirect so
   the tracked manifest is never touched (ext1-01). **Named coverage boundary:** the
   manifest slice is thin (wf1 discharge + PNGs; wf2/wf3 pre-restoration; no
   staticmaps/TOML/change-factors), so the run-relative gate is supplemented by a
   **full-`project_dir` pre/post SEMANTIC diff** at least once at the milestone gate,
   using the §9 per-type policy (ext1-04; element-wise `.nc` per ext2-02) and the
   `seed: 123` weathergenr precondition. The three copied-config snapshot rows are
   an **expected** hash FAIL adjudicated by the ext2-01 normalize-then-compare
   policy — not drift. **Open question for G1:** is a run-relative gate +
   one full-tree semantic diff acceptable, or does R6 also owe a clean manifest
   re-record (which R1 explicitly deferred to a dedicated task)?
6. **Downstream-fork / user-local rebase pain.** The package move breaks any
   `upstream` fork rebase and any user script importing `from src.`. Mitigation:
   `MIGRATION.md` (commit 6) is the complete old→new map; the roadmap PR path
   (`pr/<NN>-<topic>`) carries it upstream. **Open question:** does upstream need a
   deprecation shim (a stub `src/` re-exporting `blueearth_cst`), or is
   `MIGRATION.md` sufficient for a fork this early?
7. **`*_gabon.yml` / `*_local.yml` handling.** The **untracked** `*_gabon.yml`
   variant (per `baseline-manifest-stale` memory) is a user-local config; it holds
   `model_build_config`/`waterbodies_config` keys (lines 29/30) that must be rewritten
   **in the working tree if present** — but a fresh clone will not have it, so the
   `MIGRATION.md` note is the durable record. Recommending a rename to `*_local.yml`
   on move is a suggestion, not a forced change. **Open question:** rename in R6, or
   leave as a documented migration note?
8. **`config/reticulate_config.R` and `config/archieve/`** are gitignored config-dir
   residents not on the lock list. The split leaves them in `config/` root unless
   they collide with the new tree; flagged so the reviewer can decide whether they
   move.
