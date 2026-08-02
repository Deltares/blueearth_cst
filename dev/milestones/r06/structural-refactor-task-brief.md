# Task Brief — R06 Structural refactor (implementation)

> **Handoff from the ACCEPTED design.** The authoritative, load-bearing spec is
> `dev/milestones/r06/structural-refactor-design.md` (ACCEPTED 2026-07-22). Read it in full
> before touching code — this brief bounds and sequences the work; the design
> owns every path, inventory, and comparator rule. Where the two differ, the
> design wins. Audit trail: `dev/milestones/r06/structural-refactor-design-review-record.md`.

### Context

- **Canonical ruleset:** `AGENTS.md` (repo root). Honor its Hard Constraints —
  especially the **CST-automation-scope** rule: consume hydromt/wflow/weathergen
  conventions verbatim; a catalog/template/config move changes *where files live
  and how they are referenced*, never *what a catalog means or how a `setup_*`
  method works*. `check_baseline.py` and any vendored upstream package stay
  **unchanged**.
- **This is a behavior-preserving, pure-layout refactor.** Every change is a file
  move + mechanical reference rewrite, a new opt-in runner, or a docs edit. No
  computational path is touched. Scientific invariance is established **by
  construction**, gated by execution smokes and a run-relative baseline tripwire —
  *not* by a manifest re-record.
- **The one admissible behavior change** is `enabled: false` causing a workflow's
  outputs not to be produced (an absence). The one expected *content* change to a
  produced output is the copied-config snapshot path rewrite — adjudicated, not
  hand-waved (design §"normalize-then-compare policy").
- **Reference surface is table-driven, not pattern-guessed.** The §1a per-module
  stage table is the single source of truth for the import rewrite (both
  `from src.X` **and** `from src import X` forms). `--dry-run` is blind to R
  `source()`, `shell:` bodies, the `run_logged` bootstrap, and `params:` config
  paths — those need execution smokes, not just a green DAG.
- **Milestone mechanics (per `branch-model`).** Work on a short-lived branch off
  the current `main` tip; commit prefix `r06:`; merge to `main`; then create
  `milestone/r06-refactor` + tag `r06-refactor` at the tip. The current `main`
  tip is the **pre-R6 reference** for the baseline gate — capture it before you
  start.

### Goal

Land the roadmap's R6 7-item lock list: split flat `src/` into the
`blueearth_cst/` package (per-stage submodules + `shared/`); split `config/` into
`workflows/` / `catalogs/` / `templates/`; move the top-level runners into
`scripts/`; operationalize `workflows.<name>.enabled` via a `scripts/run_workflows.py`
wrapper; codify the `dev/` vs `docs/` boundary; and record every moved path in
`MIGRATION.md` — with the M1/R5 baseline preserved modulo numerical-noise tolerance.

### Non-goals

- **No functional decomposition of climate analysis** (design §8 DEFER). Keep the
  package tree forward-compatible for a later mechanical lift, but do not re-source
  any climate plot from raw gridded climate or otherwise change a computed value.
- **No `enabled:` logic inside the Snakefiles.** The flag governs the wrapper only;
  the three Snakefiles remain the only Snakemake entry points and do not read it.
- **No Snakemake `module:` master Snakefile** (rejected §7 — fourth entry point +
  combined-DAG behavior change).
- **No manifest re-record** and **no edit to `check_baseline.py`** (deferred, as R1
  did; scope constraint).
- **No `project_dir/` output-layout changes** (lock item 5 — leave alone).
- **No upstream/vendored package edits.**

### Allowed scope

**Permitted (touch freely):**
- `src/**` → `blueearth_cst/**` (moves + reference rewrites per §1a table).
- The three `Snakefile_*` files (`script:`/`shell:` paths, `run_logged=`
  constructions, `sys.path` shim left intact, template-default expressions,
  `Snakefile_climate_experiment:131` `default_config` param).
- `config/**` (the three-bin split + explicit-key inventory rewrite, §2/§5).
- `tests/**` (import rewrites both forms; `tests/snake_config_model_test.yml`
  lines 23–24 → `config/templates/...`; new wrapper + skip tests).
- `scripts/**` (new — runner moves + `run_workflows.py`).
- `dev/scripts/**` (new full-tree diff tooling only — walker + three comparators).
- `AGENTS.md`, `CLAUDE.md`, `README.rst`/README, `MIGRATION.md` (new).

**Approval-gated (decide before touching — see Human gates):**
- `dev/scripts/check_baseline.py` — **do not edit** (unchanged is the contract).
  Flag if you believe an edit is unavoidable.
- Upstream deprecation shim (a stub `src/` re-exporting `blueearth_cst`) —
  **open decision Q6**; default is *no shim, `MIGRATION.md` suffices*.
- `config/reticulate_config.R`, `config/archieve/` — **open decision Q8**; default
  is *leave in `config/` root, do not move*.

**Forbidden / out of scope:**
- Any vendored upstream package (hydromt / hydromt_wflow / wflow / weathergenr
  internals).
- `dev/baseline/manifest.json` (no re-record).
- Anything under `project_dir/` output layout.
- `pixi.lock`, `Manifest.toml` (never hand-edit).

### Required changes (checklist)

Each is one `r06:` commit unless noted. Commits 2–6 + T are independently
runnable/verifiable; commit 1 is the single unavoidable atomic exception.

1. **`r06: move src/ to blueearth_cst/ package and rewrite all references`**
   *(atomic — design §10.1)*. `git mv` the tree per §1 layout; rewrite **all**
   references from the §1a table: both import forms, 15 `script:` paths/Snakefile,
   the two `Rscript` shell strings + the two R `source("./src/weathergen/...")`
   calls, the three `run_logged = .../src/run_logged.py` constructions, and the
   **`run_logged.py` depth fix** (extra `dirname` level — §1a "depth-sensitive").
   Rewrite `tests/conftest.py:11` and `tests/test_run_logged.py:12–13`. Stage as
   readable "moves" vs "rewrites" hunks; enumerate the rewrite in the commit body
   from the §1a table. `dev/scripts/probe_attrs_chain.py:101` is a rewrite site;
   `dev/scripts/stage_data.py` is **not** (local `src` var).
2. **`r06: split config/ into workflows/, catalogs/, templates/`** *(§2/§4/§5)*.
   `git mv` the three bins; do the **complete explicit-key inventory rewrite**
   (`model_build_config:`/`waterbodies_config:` at template 41/43, model_test
   29/30, tests-fixture 23/24, gabon 29/30 if present in working tree) + the
   Snakefile template-default expressions (`static_dir` stays `config`; defaults
   gain `/templates/`) + `Snakefile_climate_experiment:131` `default_config` →
   `config/templates/weathergen_config.yml` + the `data_sources:` catalog-path
   edits. `.gitignore` `*_local.yml` glob already covers the new paths.
3. **`r06: move runners to scripts/ and update config paths`** *(§6)*. `git mv`
   `run_snake_test.cmd` + `run_snake_docker.sh` → `scripts/`; update their
   `config/...` refs to `config/workflows/...`; **preserve per-workflow flags
   verbatim** (`--keep-going` stays on the projections line only).
4. **`r06: add milestone full-tree semantic-diff tooling`** *(design §9 +
   commit-plan tooling note — NEW `dev/` tooling; land before the milestone
   full-tree diff, i.e. at/after commit 3's verification)*. A recursive
   `project_dir` walker dispatching by extension to: the **`.toml` normalized**
   comparator, the **element-wise `.nc`** comparator (dims; coordinate labels+order
   with no realignment; exact NaN masks; per-element `_within_tol`; non-`VOLATILE_NC_ATTRS`
   attrs — **not** aggregate `fingerprint_nc`), and the **copied-config YAML
   normalize-then-compare** comparator (apply only the documented old→new path map;
   all else must be equal). Reuse `check_baseline.py`'s CSV/PNG/discharge
   comparators; **do not modify `check_baseline.py`**.
5. **`r06: add scripts/run_workflows.py enabled-aware wrapper`** *(§7, additive)*.
   Implement the pinned contract (a)–(g): full-orchestration configs only;
   missing `workflows.<name>.enabled` → hard nonzero error naming the key;
   **parsed-value bool** check (`isinstance(v, bool)` post-`safe_load` — unquoted
   `yes`/`on` accepted, quoted `"true"`/`1` rejected); fixed order
   model→projections→experiment; **stop on first nonzero exit, return that code**;
   `--cores` + `-- <extra>` forwarding to every invocation; **per-workflow flag
   map** preserving `--keep-going` on projections only. Plus the unit-test suite
   §7(g) (monkeypatch `subprocess.run`, capture argv) **and** the `enabled: false`
   skip test in a **fresh `tmp_path` project_dir** asserting invocation/non-invocation
   at the subprocess boundary (design §9 ext1-03).
6. **`r06: codify dev/ vs docs/ boundary and new layout in AGENTS.md + README`**
   *(§3, docs-only)*. Update AGENTS.md Repo Map / Conventions / References to the
   new tree; point README run instructions at `scripts/`; document the wrapper
   config-class contract and the "no deletion / no freshness guarantee / downstream
   consumes pre-existing artifacts" semantics. **Also codify the toolbox-vs-project
   separation** (user expectation, 2026-07-23): production `project_dir` lives
   **outside the repository tree** — one sentence in AGENTS.md/README plus a comment
   on the `project_dir:` key in `config/workflows/snake_config.template.yml`
   (comment-only edit; no value or behavior change). The in-repo untracked
   `examples/test_local` dev/test convention is explicitly exempt — the baseline
   gate (rung 4) depends on it.
7. **`r06: add MIGRATION.md mapping every moved path old to new`** *(roadmap exit
   criterion)*. Complete old→new map for downstream forks / user-local configs,
   including the untracked `*_gabon.yml` note and the `*_local.yml`-rename
   recommendation.

### Validation

Report each rung; a stage is not done until its gate passes. Per-stage gates are
tabulated in design §9 — follow that table. Ladder:

1. **Narrow (per stage):** `pytest tests/test_cli.py` (dry-runs all three
   Snakefiles) after any Snakefile/config/script-path edit; targeted module tests
   for the stage.
2. **Execution smokes (runtime-only paths `--dry-run` cannot see):**
   - Commit 1: rewritten `tests/test_run_logged.py` imports `main` + calls
     `run_and_tee`; invoke `run_logged.py` on a trivial command; one `Rscript
     --vanilla` `source()`-only smoke proving the R path rewrite resolves.
   - Commit 2: a `pytest` assertion (or WF3 run to rule 3.04) that
     `prepare_weagen_config` resolves `default_config` at
     `config/templates/weathergen_config.yml`.
   - Commit 5: the wrapper contract suite + fresh-dir skip test.
3. **Full gate:** `pytest tests/` green (expected ≈ R5's 120 passed / 3 skipped /
   7 xfailed **plus** the new wrapper contract suite + skip test); clean
   `--dry-run` on all three Snakefiles; full end-to-end run of all three workflows
   via `scripts/run_snake_test.cmd`.
4. **Baseline / non-regression (run-relative — never touch the tracked manifest):**
   Follow design §9 exactly. Capture the **pre-R6 tip** (current `main`), run all
   three workflows into `examples/test_local` (seed 123), `check_baseline.py record
   --project-dir examples/test_local --manifest <scratch>/manifest.json`, stash the
   tree. On the post-R6 tip, re-run into the **same** `examples/test_local`, then
   `check_baseline.py check --project-dir examples/test_local --manifest
   <scratch>/manifest.json`. **Expected FAIL:** exactly the three copied-config
   snapshot rows (`{project_dir}/config/snake_config_*.yml`) — adjudicate them with
   the normalize-then-compare policy; **any other failing row is a real FAIL.** Then
   run the **full-`project_dir` semantic diff** (commit-4 tooling) once against the
   stashed pre-R6 tree to cover the un-manifested slice (staticmaps, `wflow_sbm.toml`,
   change-factor NetCDFs). The tracked `dev/baseline/manifest.json` is never
   recorded to, overwritten, or restored.

### Acceptance criteria

- All 7 required changes landed as `r06:` commits; commits 2–7 each independently
  runnable, commit 1 atomic-and-gated.
- Full gate (rung 3) green; all execution smokes pass.
- Run-relative baseline (rung 4) clean except the three expected+adjudicated
  copied-config snapshot rows; full-tree semantic diff clean on the un-manifested
  slice.
- `MIGRATION.md` lists every moved path (a `git mv` audit confirms none missed).
- `milestone/r06-refactor` branch + `r06-refactor` tag created at the merged `main`
  tip.
- **Rollback trigger:** any un-adjudicated baseline FAIL, any out-of-tolerance
  element in the full-tree `.nc` diff, or a broken runtime path that a smoke did
  not catch → stop, do not tag, surface the diff.

### Output requirements

- The commits above on a short-lived branch, merged to `main`, plus the tag/branch.
- `MIGRATION.md` and the updated AGENTS.md/README.
- A short **Results delta**: confirm the only produced-output content change is the
  copied-config snapshot path rewrite (with the adjudication result), and that no
  computed value changed. Note the resolution taken on the two open decisions
  (Q6 shim, Q8 `reticulate_config.R`/`config/archieve/`).

### Task constraints

- Consume upstream conventions verbatim; do not re-engineer hydromt/wflow/weathergen
  behavior. `check_baseline.py` and vendored packages stay unchanged.
- Preserve `workflow.configfiles[0]` forwarding, the `get_config` contract, and the
  `sys.path` shim.
- Linux/Docker files (`*_linux.yml`, `run_snake_docker.sh`) must continue to parse
  but are **not** exercised (Linux replication parked).
- `--dry-run` is necessary but never sufficient for a runtime-only reference — an
  execution smoke gates every such path.

**Human gates** (otherwise drive the commit plan commit-to-commit autonomously,
per the user's standing preference — do not pause between every commit):

- **Gate 1 — after commit 1 (atomic move+rewrite), PAUSE for review.** Its size is
  a stated risk; present the "moves" vs "rewrites" hunks and the smoke results
  before continuing. (If you conclude the transitional re-export shim is preferable,
  raise it here — the design records it as a rejected-but-overridable alternative.)
- **Gate 2 — at the milestone gate, before merging to `main` and tagging, PAUSE.**
  Present the full gate + run-relative baseline + full-tree semantic-diff results,
  the snapshot adjudication, and the two open-decision resolutions (Q6, Q8) for
  sign-off.
- Raise immediately if `check_baseline.py` seems to need editing (it must not), or
  if any baseline FAIL is not the expected copied-config snapshot set.
