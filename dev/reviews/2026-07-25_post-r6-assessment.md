# Post-R6 assessment — observation register

Live register of observations from the owner's own assessment and testing of the
repo **after the R6 structural refactor**. Opened 2026-07-25 against `3c8c2a9`
(`main`). Each row is one observation, from intake through triage to disposition.

**Scope and boundary.** This file owns the owner's post-R6 assessment
observations only. Items that survive triage and belong to a later milestone are
promoted to [`../followups.md`](../followups.md); items needing tracked,
multi-session work get a [`../TODO.md`](../TODO.md) row. Either way the
**Disposition** column keeps the pointer, so nothing lives in two registers. The
pre-existing `## Post-R6` entries already in `followups.md` (surfaced 2026-07-23
during R6 milestone validation) stay where they are — do not re-log them here.

**How to add a row.**

1. Append an index row with the next `O-nn` ID; record `Created`, `Rev` (short
   sha the observation was made against), and `Status: open`.
2. Add a matching detail block below with the exact command, configfile, and
   observed-vs-expected — enough for a future session to confirm the issue still
   applies before acting on it.
3. On any status change, update `Updated` and, once routed, `Disposition`.

**Status vocabulary:** `open` (logged, not yet triaged) · `triaged` (cause
understood, routed) · `fixed` (landed; put the sha in Disposition) ·
`wontfix` (accepted as-is, with a reason) · `not-reproducible` (does not
reproduce under current pins) · `by-design` (expected behaviour, not a defect).

**Kind:** `defect` · `regression` (worked before R6, broken after) ·
`docs` · `usability` · `performance` · `question` (needs a decision, not a fix).

---

> **Routing note (2026-07-26).** The layout-bearing observations — O-01, O-02,
> O-05, O-20, O-21, O-22, O-24, plus the recorded by-design rulings O-03, O-23 and
> O-23a — are consolidated into milestone **R07**:
> `dev/r07/project-layout-design.md`, with the old→new path map at
> `dev/r07/migration_project-layout.md`. That design is the implementation
> authority; this register remains the provenance record of how each item was
> found and triaged. Items **not** in R07: O-06 (Docker, parked), O-18/O-19
> (Linux, parked), O-12/O-13 (open decisions, surfaced in the R07 open questions),
> O-14/O-15/O-16 (tooling contract — unrelated to layout), and O-07 … O-10
> (accepted drive-by fixes, landing as their own commits).

## Index

| ID | Observation | Area | Kind | Severity | Created | Updated | Rev | Status | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| O-01 | Root `data/` holds basin-specific data inside the toolbox source tree | config | usability | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Direction decided 2026-07-26 (delete values, ship templates); implementation pending review closure |
| O-02 | Root `dag/` holds generated run artifacts | dev-tooling | usability | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Direction decided 2026-07-26 (write under `project_dir`); implementation pending review closure |
| O-03 | Rationale for the nested `blueearth_cst/` package layout is recorded nowhere user-facing | docs | question | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Answered 2026-07-26 — keep the layout, record the reasoning in `AGENTS.md` |
| O-04 | `tests/snake_config_model_test.yml` points at a nonexistent `tests/data/observations/` | tests | defect | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Fold into O-01 |
| O-05 | `docs/config/` is a 16-file pre-R6 mirror of `config/` | docs | defect | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Retire; accepted into scope 2026-07-26 |
| O-06 | `Dockerfile` stages the pre-R6 `src/`, so the image ships without the Python package | env | regression | high→n/a | 2026-07-26 | 2026-07-26 | `75eb4d6` | wontfix | **Parked by owner 2026-07-26** — Docker is not in active use; rebuild rather than repair when Linux/Docker work resumes |
| O-07 | `prepare_cst_parameters.py` `sys.path` insert is one level short of the repo root | experiment | defect | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Accepted into scope 2026-07-26 |
| O-08 | `plot_map.py` `is not None` guard never fires for the `"None"` string sentinel | shared | defect | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Accepted into scope 2026-07-26 |
| O-09 | `plot_results.py` docstring states the wrong separator for `observations_fn` | docs | docs | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Accepted into scope 2026-07-26 |
| O-10 | `MIGRATION.md` `__init__.py` list omits `climate_analysis/` | docs | docs | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Accepted into scope 2026-07-26 |
| O-11 | Can the root directory be tidied? — per-entry assessment | dev-tooling | question | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Assessed 2026-07-26: 13 of 17 tracked root files are contract-bound or convention-sanctioned; only O-12/O-13 are actionable |
| O-12 | `MIGRATION.md` sits at the root against the repo's own migration-note convention | docs | usability | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Needs a target decision — `docs/` (audience) vs `dev/r06/` (naming.md §7) |
| O-13 | `blueearth_cst.Rproj` is an unreferenced RStudio project file | dev-tooling | usability | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Delete or scope to the R sources — needs owner input on whether it is used |
| O-14 | The repo has no Python tool-config layer at all (no `pyproject.toml` / `pytest.ini` / `ruff.toml`) | dev-tooling | question | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Assessed 2026-07-26 — three separable decisions; see detail |
| O-15 | The PR template mandates a Black pass, but no formatter or linter is in the environment | ci | defect | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Needs a new-dependency decision (`ruff`) before anything can be enforced |
| O-16 | `flit` is a declared dependency with nothing to build | env | defect | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Drop it, or give it a job — depends on the O-14 build decision |
| O-17 | `README.rst` states the Dockerfile "builds against the pixi env"; it does not build at all | docs | docs | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Correct the claim as part of the O-06 parking |
| O-18 | Should Linux be parked too? — the per-iteration tax is mostly not being paid | config | question | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | Assessed 2026-07-26: park the goal (already parked), **keep the CI leg**, treat the variants as stale |
| O-19 | `deltares_data_linux.yml` is a pre-1.0 hydromt catalog, incompatible with the pinned `hydromt >=1.3` | config | defect | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Rebuild at the Linux milestone, not repair; mark stale now |
| O-20 | `examples/` is misnamed — it holds the local test fixture, not examples | dev-tooling | usability | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | Rename agreed in principle; needs a target name + a naming.md §7 migration note. **Not cosmetic:** 4 of 18 baseline fingerprints go stale. `runs/` withdrawn — use a test-scoped name |
| O-21 | `snake_config.template.yml` ships a repo-relative `project_dir` while its own comment says point outside | config | defect | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | The template teaches the wrong default — origin of the tier confusion |
| O-22 | No mechanical check that a production `project_dir` points outside the repo | shared | usability | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | triaged | **Accepted 2026-07-26** — add a parse-time warning; design sketched below. Land after O-20 |
| O-23 | Why a root `scripts/` when `blueearth_cst/` already holds scripts? | dev-tooling | question | low | 2026-07-26 | 2026-07-26 | `75eb4d6` | by-design | **Keep all three homes unchanged.** The split is by invocation model and was deliberated in R6; the gap is documentary — add one contrastive line to `AGENTS.md` |
| O-24 | Rule 1.13 `plot_forcing` writes three PNGs but declares only one as `output:` | wf1 | defect | medium | 2026-07-26 | 2026-07-26 | `75eb4d6` | open | `temp.png` / `pet.png` are untracked by Snakemake and absent from the baseline. Independent of the output-layout work |

Area labels are free-form; keep to the repo's vocabulary where one fits:
`wf1`/`model`, `wf2`/`projections`, `wf3`/`experiment`, `weathergen`, `shared`,
`config`, `tests`, `ci`, `env`, `docs`, `dev-tooling`.

Severity: `high` (blocks a workflow or produces wrong numbers) · `medium`
(works but wrong/awkward in a way users will hit) · `low` (cosmetic, noise,
wording).

---

## Details

O-01 to O-03 are the owner's layout observations from the 2026-07-26 pass;
O-04 to O-10 were surfaced while investigating them and are logged separately so
each can be routed on its own merits.

### O-01 — Root `data/` holds basin-specific data inside the toolbox source tree

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `data/observations/` holds two git-tracked CSVs of Gabon test-basin
  data — `output-locations-test.csv` (7 gauge rows: `wflow_id,station_name,x,y`)
  and `observations_timeseries_test.csv` (667 KB, daily discharge 2000-01-01 →
  2019-12-31, `;`-separated). These are project inputs (outlet/gauge locations for
  discharge extraction, observed discharge for calibration/validation plots), not
  toolbox source.
- **Expected:** the repo carries only the **schema template**; actual basin values
  live in the project folder, consistent with the AGENTS.md rule that a run writes
  to a `project_dir` outside the repository tree.
- **Consumers (complete):**
  - `Snakefile_model_creation:54-55` reads `output_locations` /
    `observations_timeseries`; passed as params at `:150` (rule 1.05
    `add_gauges_and_outputs`), `:275-276` (1.11 `plot_results`), `:294` (1.12
    `plot_map`).
  - `blueearth_cst/model/setup_gauges_and_outputs.py:55` (gauge creation — a real
    model-build input, not only plotting), `blueearth_cst/model/plot_results.py:127`,
    `blueearth_cst/shared/plot_map.py:28`.
  - Only **one** config supplies real paths:
    `config/workflows/snake_config_model_test_linux.yml:25-26`. Every other config
    ships the unset sentinel. `scripts/run_snake_docker.sh:7` mounts the directory
    (`-v $(pwd)/data:${docker_root}/data`).
  - No Snakefile, CI job, or `Dockerfile` stage references the path literally.
- **Notes:**
  - **Sentinel hazard.** The "unset" value is unquoted YAML `None`, which parses to
    the Python **string** `"None"`, not `null` (`snake_utils.py:146-147` returns
    config values verbatim). `os.path.isfile("None")` / `os.path.exists("None")`
    are False, so the guards work by accident. Replacing it with real YAML `null`
    raises `TypeError` in both consumers — any edit here must keep the sentinel
    byte-identical to `config/workflows/snake_config_model_test.yml:36-37`.
  - **Baseline is independent of `data/`.** The tracked baseline seed already has
    both keys at the sentinel, so wf1 has never added gauges or plotted
    observations under it; `dev/baseline/manifest.json`'s wf1 discharge
    `output.csv` cannot move because of this change.
  - Both live consumers are on the deferred Linux-replication path
    (`roadmap.md`, "Deferred: Linux replication"), so nothing exercised today reads
    these files.
  - **Decision 2026-07-26:** delete the tracked values (recoverable from git
    history); add small schema templates under `config/templates/observations/`
    preserving the separator asymmetry (`,` locations, `;` timeseries); repoint the
    Linux config and `tests/` config to the sentinel; drop the Docker mount. The
    key contract stays a raw path string — **no `project_dir`-relative resolution**,
    which would make one config string mean different things across versions and
    split semantics against the `static_dir`-relative sibling keys
    (`model_build_config`, `waterbodies_config`, `Snakefile_model_creation:53-54`).

### O-02 — Root `dag/` holds generated run artifacts

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `dag/` holds six untracked files (three `.dot` + three `.png`,
  ~0.7 MB), written by `scripts/run_snake_test.cmd:32,39,76,80` and ignored via
  `.gitignore:135-136`. A stale `dag_model.png` also sits at the repo root.
- **Expected:** generated artifacts do not occupy a root-level directory of the
  source tree.
- **Notes:**
  - The runner is not the only source. `README.rst:269,285,298` and six notebook
    cells (`docs/notebooks/Model building.ipynb:176,248`; `Climate
    projections.ipynb:197,269`; `Climate Stress Test.ipynb:301,373`) pipe
    `--dag | dot -Tpng` into the **repo root** — that is where the stray
    `dag_model.png` came from, and the clutter returns the first time a user
    follows the README unless those six commands change too.
  - Only `--dag` is used anywhere in the repo; `--rulegraph` / `--filegraph` appear
    nowhere.
  - The render step is deliberately best-effort (`run_snake_test.cmd:73-75,84`)
    because win-64 graphviz needs the DLL aliases from
    `dev/scripts/pixi_activate.bat`; a failure must never abort a run.
  - **Decision 2026-07-26:** write to `<project_dir>/dag/`. The DAG is a function of
    the config, so it belongs with that config's run artifacts; for the test config
    that is `examples/test_local/dag/`, already covered by `.gitignore:124`
    (`examples/`), letting the `dag/` ignore entry go away. Hardcode `DAGDIR` next
    to the already-hardcoded `CFG` rather than parsing YAML in cmd.exe — that needs
    a `for /f` capture inside the `setlocal`/`endlocal` subroutine, and this file
    already carries scar tissue for that class of fragility (`:41-45`, `:81-83`) for
    a step that is cosmetic. Use backslash paths; `mkdir` rejects forward slashes.

### O-03 — Rationale for the nested `blueearth_cst/` package layout is unrecorded

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `blueearth_cst/<stage>/module.py` under a repo already named
  `blueearth_cst` reads as gratuitous nesting; no user-facing doc explains it.
- **Finding — the layout is correct; only the rationale is missing.**
  - **Outer level:** `blueearth_cst/` *is* the Python package root. The single level
    of nesting is what makes `from blueearth_cst.shared.snake_utils import
    get_config` resolve. Bare modules at the repo root would lose namespacing and
    collide with the Snakefile/config bins; a `src/blueearth_cst/` layout needs an
    installed package, deliberately rejected in
    `dev/r06/structural-refactor-design.md:368-376` (no `pyproject.toml`;
    `pixi.toml` `[pypi-dependencies]` are third-party only). Imports resolve via
    explicit `sys.path` inserts of `workflow.basedir` at four entry-point classes:
    the three Snakefile headers, `tests/conftest.py:10`,
    `blueearth_cst/shared/run_logged.py:23-28`, and two self-guarding `experiment/`
    scripts.
  - **Inner level:** one subpackage per workflow stage (`model/`, `projections/`,
    `experiment/`, `climate_analysis/`) plus `shared/` for cross-cutting helpers.
    This is what makes `script: "blueearth_cst/model/plot_results.py"` state which
    workflow owns a step. `weathergen/` has no `__init__.py` on purpose — it is
    R-only.
  - **Cost of flattening,** measured at `75eb4d6`: 41 tracked package files, 31
    `script:` directives across the three Snakefiles, 17 modules with top-level
    absolute `blueearth_cst.*` imports, four `sys.path` shim sites,
    `tests/conftest.py`, and `run_logged.py`'s three-`dirname` walk.
- **Decision 2026-07-26:** keep the layout; record the two-level reasoning in the
  `AGENTS.md` Repo Map entry so the question does not recur.

### O-04 — `tests/snake_config_model_test.yml` points at a nonexistent path

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `tests/snake_config_model_test.yml:32-33` sets `output_locations` /
  `observations_timeseries` to `tests/data/observations/*.csv`. `tests/data/`
  contains only `tests_data_catalog.yml`; there is no `observations/` subdirectory.
- **Expected:** either a real path or the unset sentinel.
- **Notes:** degrades silently — both consumers guard with `os.path.isfile` /
  `os.path.exists`, so the dry-run gate passes and wf1 simply adds no gauges. The
  identical dangle in `tests/test_project/config/snake_config_model_creation.yml:32-33`
  is gitignored generated state (`.gitignore:139`), not a repo defect. Predates R6.
  Fix alongside O-01.

### O-05 — `docs/config/` is a pre-R6 mirror of `config/`

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** 16 tracked files under `docs/config/` duplicate the `config/` tree
  in its **pre-R6 flat schema**, including catalogs, workflow configs, and
  templates. Two of them (`snake_config_model_test_linux.yml:35,37` and
  `snake_config_model_test.yml:41,43`) still reference the `data/observations/`
  paths from O-01.
- **Expected:** one source of truth for configuration; `docs/` carries reference
  prose, not a second copy of `config/`.
- **Notes:** same class of inconsistency as O-01 — repo files duplicated where their
  audience does not need them. Referenced only by `MIGRATION.md:173` and the
  historical `dev/phase-1/m02b/{audit,plan}.md`. `dev/phase-1/m02b/plan.md:250`
  shows the mirror was once kept byte-identical by hand; the R01 config
  restructure ended that. **Decision 2026-07-26:** retire the directory; update the
  `AGENTS.md` `docs/` description and `MIGRATION.md:173`; leave the `dev/phase-1`
  references as historical record.

### O-06 — `Dockerfile` stages the pre-R6 `src/` — **parked**

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** wontfix (parked)
- **Observed:** `Dockerfile:17` is `ADD src src` in the `local_files` staging
  stage. `src/` was renamed to `blueearth_cst/` in `368b30e` (R6), so the directory
  no longer exists and the built image ships **without** the Python package — only
  the three Snakefiles reach `/root/work/`. An `ADD` of a missing path fails the
  build outright, so the image has not been rebuilt since R6.
- **Second, independent drift found while triaging:** `Dockerfile:10` pins
  `ARG julia_version=1.12.6`, while the project pins **1.11.7** (`pixi.toml:100`
  `install-julia`; `docs/install.md:40`) — and `Project.toml:6` records *why*:
  "Wflow.jl v1.0.x has a JIT compilation hang under Julia 1.12.x". A successful
  build would therefore produce an image on the exact Julia version the project
  deliberately avoids.
- **Decision 2026-07-26 (owner):** Docker is **not in active use** and is parked.
  Not repaired now; **rebuilt rather than patched** when Linux/Docker work resumes,
  since with two independent drifts plus the whole R6 layout change, "repair" is a
  fiction.
- **What parking actually buys** (the reason it is recorded here rather than just
  dropped):
  - **It lifts the only Docker prerequisite on a decision that matters.** O-14
    option 2 (editable self-install) was constrained by the Dockerfile's ordering —
    `pixi install` runs before the source is copied, so a self-referencing editable
    dep cannot resolve at image build. Parked, that constraint leaves the critical
    path and the packaging question can be decided on its own merits.
  - Removing the `data/` mount in `scripts/run_snake_docker.sh:7` (O-01) becomes a
    zero-risk edit.
  - It resolves a live contradiction: `dev/roadmap.md`'s standing rule says the
    Linux/Docker files "must continue to build / parse", which has been false for
    the Dockerfile since R6.
- **What parking does _not_ buy — Linux ≠ Docker:**
  - CI's `ubuntu-latest` leg runs natively through pixi, not through Docker, so the
    Linux half of "Deferred: Linux replication" is untouched and still real.
  - `config/workflows/snake_config_model_test_linux.yml` and
    `config/catalogs/*_linux.yml` are Linux artifacts, not Docker artifacts — O-01
    still has to repoint them.
- **Follow-on decision, still open:** park *in place*, relocate, or delete.
  - *In place* — cheapest; but leaves a root file that `naming.md:170` sanctions as
    a "standard root-level file" while it is neither standard nor working. Needs at
    minimum a header comment marking it unmaintained and a `roadmap.md` amendment
    dropping the "must continue to build" claim for it.
  - *Relocate* to `docker/` (with `run_snake_docker.sh`) — signals unmaintained and
    feeds O-11; `docker build -f docker/Dockerfile .` keeps the root as the build
    context. Cost is near zero precisely because the file is already broken.
  - *Delete* — git history retains it; a rewrite is expected anyway. Strongest if
    the v0.1.0-alpha published image (`README.rst:141`) stays the documented option
    for Docker users in the meantime.

### O-07 — `prepare_cst_parameters.py` `sys.path` insert is one level short

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `blueearth_cst/experiment/prepare_cst_parameters.py:14` inserts
  `.parent.parent` — i.e. `blueearth_cst/` itself — but its import at `:15` is
  `from blueearth_cst.shared.snake_utils import stress_test_grid`, which that path
  cannot satisfy. The sibling `check_project_consistency.py:32` correctly walks
  three levels to the repo root.
- **Expected:** the guard makes the module import-clean standalone, as its `:12-13`
  comment claims.
- **Notes:** works only because the repo root is already on `sys.path` from the
  Snakefile shim / `conftest.py` / CWD, so it never fails in practice. Plausibly a
  survivor of the class fixed in `f4be2f6`.

### O-08 — `plot_map.py` `None` guard never fires for the string sentinel

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `blueearth_cst/shared/plot_map.py:28` tests `if gauges_fn is not
  None`, which is True for the string `"None"` the configs actually ship, producing
  `gauges_name = "gauges_None"` at `:29`.
- **Expected:** the unset sentinel yields `gauges_name = None`.
- **Notes:** currently harmless — the membership guard at `:103`
  (`gauges_name in geoms`) rescues it, so no output changes. Sibling consumers use
  a path-existence check; aligning this one removes the discrepancy without
  changing behaviour on any current path. Related to the O-01 sentinel hazard.

### O-09 — `plot_results.py` docstring states the wrong separator

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `blueearth_cst/model/plot_results.py:83` documents "Separator is ,"
  for `observations_fn`; the reader at `:133` passes `sep=";"`, and the shipped file
  is semicolon-separated. The `gauges_locs` entry at `:88` is correct (`,`, matching
  `open_vector(..., sep=",")` at `:131`).
- **Expected:** the docstring matches the code — it is the only place the
  observations schema is written down, so it becomes the template's source.

### O-10 — `MIGRATION.md` `__init__.py` list omits `climate_analysis/`

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** `MIGRATION.md:96-101` enumerates five `__init__.py` files; the
  package has six. `climate_analysis/` was added later in `6a2c105`.
- **Expected:** the list matches the tree.

### O-11 — Can the root directory be tidied?

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Observed:** the root carries 17 tracked files and 9 tracked directories, plus
  seven untracked/ignored tool directories — enough that it reads as cluttered.
- **Assessment.** Most of it is load-bearing. 13 of the 17 tracked files are fixed
  by an external tool contract or already sanctioned by
  `dev/conventions/naming.md:170` ("Standard root-level files — upstream"). Only
  three are genuinely movable, and two of those are worth acting on (O-12, O-13).

| Entry | Verdict | Reason |
|---|---|---|
| `.gitignore`, `.gitattributes` | fixed | Git reads the top-level file from the root. |
| `README.rst`, `LICENSE` | fixed | Forge landing page and license detection. |
| `CLAUDE.md`, `AGENTS.md` | fixed | Discovered at the repo root by Claude Code / Codex. |
| `pixi.toml`, `pixi.lock` | fixed | The manifest defines the workspace root; `[target.win-64.activation] scripts` and `[tasks]` paths are manifest-relative. Moving it means there is no pixi workspace. |
| `.github/` | fixed | Required path for Actions. |
| `CHANGELOG.md` | stays | Keep-a-Changelog convention, and `dev/README.md` already states it "lives at the project root, not in `dev/`". |
| `Dockerfile` | stays | Listed by `naming.md:170` as a standard root-level file; `docker build .` finds it there. Movable via `-f docker/Dockerfile`, but the convention is already recorded — don't fight it. |
| `Snakefile_model_creation`, `_climate_projections`, `_climate_experiment` | stays | See "Snakefiles" below. |
| `Project.toml`, `Manifest.toml` | stays | See "Julia env" below. |
| `MIGRATION.md` | **move** | O-12. |
| `blueearth_cst.Rproj` | **delete or relocate** | O-13. |
| `dag_model.png` | **delete** | Untracked, ignored, stale — covered by O-02. |
| `blueearth_cst/`, `config/`, `dev/`, `docs/`, `scripts/`, `tests/` | stays | The R6 bins; each has a stated audience in `AGENTS.md`. |
| `data/` | **retire** | O-01. |
| `profiles/` | fixed | `profiles/default` is the path Snakemake auto-detects as the workflow profile, resolved from the working directory. Holds one file (`profiles/default/config.yaml`); its `quiet: reason` setting is asserted against by `tests/test_guard_invalidation.py:48`. |
| `.pixi/`, `.snakemake/`, `.pytest_cache/`, `.ruff_cache/`, `dag/`, `examples/` | ignored | Tool-owned working state, all gitignored. |
| `.agents/`, `.claude/`, `.codex/` | ignored | Agent runtime config, root-discovered, untracked. |

- **Snakefiles — movable in principle, not worth it.** `-s` accepts any path, but
  `script:` directives resolve against `workflow.basedir`, so all 31 would need a
  `../` prefix, and the `sys.path.insert(0, str(Path(workflow.basedir)))` in each
  header would need `.parent`. The caller surface is ~88 references across 28
  non-`dev/` files (`README.rst` ×16, `tests/` ×~28, `scripts/` ×10, `Dockerfile`
  ×3, the three notebooks ×15). Against that churn: `AGENTS.md` calls these "the
  only entry points", and entry points at the front door is the conventional
  reading. **Keep at root.**
- **Julia env — movable, but the cons win.** `Project.toml` / `Manifest.toml` are
  reached by `--project=.` in exactly two live places (`pixi.toml`'s
  `install-julia` task; `Snakefile_climate_experiment:434`), both CWD-relative, so
  a move to e.g. `julia/` is a two-line change plus `README.rst:113` and
  `docs/env_setup_notes.md:6-7`. Against: root `Project.toml` is the Julia
  ecosystem convention (`] activate .`, the VS Code Julia extension, Wflow.jl's own
  docs); `dev/phase-1/m02/decisions.md:23` recorded "Project.toml stays where it
  is" as part of the hybrid pixi+native-Julia decision, and the documented reversal
  path (going full-pixi) assumes root. The usual argument for moving — "group the
  env files together" — fails here because `pixi.toml` cannot follow. **Keep at
  root.**
- **The cache dirs are not worth relocating.** `.pytest_cache` / `.ruff_cache` can
  be redirected by config, but there is no `pyproject.toml`, `pytest.ini`, or
  `ruff.toml` to hold the setting — adding a root file to remove two ignored dirs
  is a net loss. (`dev/phase-1/m02b/package_inventory.md:64` already flags the
  pixi `flit` dep as unused for the same underlying reason: no build system.)
- **Conclusion.** The root is close to as tidy as its tool contracts allow. Acting
  on O-12, O-13, plus the `dag_model.png` and `data/` removals already logged takes
  it from 17 tracked files to 14, with no reference churn.

### O-12 — `MIGRATION.md` sits at the root against the repo's own convention

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** root `MIGRATION.md` is a single-milestone document — "MIGRATION —
  R06 structural refactor", git-ref-anchored to `e33ee45` (`:1-14`). Meanwhile
  `dev/conventions/naming.md:141-147` (§7) requires migration notes at
  `dev/<milestone>/migration_<topic>.md`, and the P3-1 design review (`repo-4`,
  accepted — `dev/p31/experiment-structure-design-review-record.md:870`) explicitly
  declined to promote the root file into a multi-milestone index, routing its own
  note to `dev/p31/migration_experiment-structure.md` instead.
- **Expected:** one home for migration notes.
- **Notes:** the root file is therefore the only one of its kind — an R6 artifact
  that predates the convention it now sits outside of. It is referenced by no
  README, no `AGENTS.md` entry, and no doc index; only `dev/p31/*` mentions it.
  **Open decision, two defensible targets:**
  - `docs/migration-r06.md` — matches the stated audience ("rebase a downstream
    fork, a user-local config, or any script that imported `from src.`"), which is
    a *user*, and `AGENTS.md` defines `docs/` as the user-facing bin. Leaves
    §7-style notes under `dev/` inconsistent with this one.
  - `dev/r06/migration_structural-refactor.md` — matches naming.md §7 literally and
    puts it beside the R6 design docs, but `dev/` is declared "not shipped, not
    user-facing", which contradicts who the document is for.
  A third option is to leave it and add a one-line exemption to §7; that is the
  honest choice if downstream forks are expected to look for it at the root.

### O-13 — `blueearth_cst.Rproj` is an unreferenced RStudio project file

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** a stock RStudio project file at the root with **zero references**
  anywhere in the repo (no doc, config, script, or CI job mentions it). It declares
  `Encoding: ISO8859-1`, contradicting the repo's UTF-8 convention, and
  `NumSpacesForTab: 2`.
- **Expected:** either a used and correctly configured project file, or none.
- **Notes:** the repo's R surface is three scripts under `blueearth_cst/weathergen/`
  invoked as `Rscript --vanilla` from Snakemake `shell:` bodies — never as an
  RStudio project session. Most likely an upstream-deltares leftover. Options:
  delete it, or move it to `blueearth_cst/weathergen/` so opening it scopes the
  RStudio session to the actual R code (RStudio sets the working directory to the
  `.Rproj`'s own folder, which would also fix the mismatch between a root-level
  project and the R sources three levels down). Needs owner input: this is the one
  item in this pass whose answer depends on whether you personally use it.

### O-14 — No Python tool-config layer exists

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** the repo has no `pyproject.toml`, `setup.py`, `setup.cfg`,
  `pytest.ini`, `tox.ini`, or `ruff.toml`. Every Python tool therefore runs on
  defaults, and there is nowhere to declare a setting.
- **What that actually costs today** — measured, not assumed:

| Area | Current state | Consequence |
|---|---|---|
| Import path | **41 `sys.path.insert` sites** outside `dev/`: 29 test files → repo root, 5 test files → `dev/scripts`, three Snakefile headers, `run_logged.py:23-28`, two self-guarding `experiment/` scripts | The shim *is* the packaging contract; O-07 is a direct symptom (one of the 41 walks the wrong depth and nobody noticed) |
| pytest | `integration` marker registered programmatically in `tests/conftest.py:28-32`; `-rs` lives only in the CI `run:` line; `strict=True` set per-marker at `test_prepare_climate_data_catalog.py:292` | Nothing is broken, but skip visibility and xfail strictness are per-invocation conventions rather than repo settings — exactly the drift the CI comment warns about ("a new skip is a signal") |
| Lint / format | See O-15 | A PR-template requirement with no tool behind it |
| Build | See O-16 | A build backend with nothing to build |

- **Assessment — `pyproject.toml` is three separable decisions, not one.** Conflating
  them is why this looks harder than it is.

  **(1) Tool-config host only.** A `pyproject.toml` containing *only*
  `[tool.pytest.ini_options]` (and `[tool.ruff]` if O-15 lands) — **no
  `[build-system]`, no `[project]`, nothing installed.** Both tools read their
  tables from an otherwise-empty pyproject. This buys: `testpaths`, declarative
  `markers`, `addopts = "-ra"` so skip reasons show locally and not only in CI,
  `xfail_strict = true` as a global ratchet, `norecursedirs` for `examples/`, and —
  the useful one — `pythonpath = ["."]`, the supported declarative replacement for
  the 29 repo-root inserts in `tests/`. *Cost:* one root file, no dependency, no
  contract reversed. *Verify first:* that pixi still resolves `pixi.toml` as its
  manifest when a `pyproject.toml` without `[tool.pixi]` is present (expected, but
  it is load-bearing here and CI runs `locked: true`).

  **(2) Real packaging** — add `[build-system]` + `[project]` and an editable
  self-install (`[pypi-dependencies] blueearth-cst = { path = ".", editable = true }`).
  *Pros:* removes all 36 repo-root shim sites by construction, makes O-07
  unrepeatable, and makes `blueearth_cst` importable regardless of CWD — which
  currently only holds because Snakemake is always invoked from the repo root.
  *Cons, and they are real:*
  - Reverses a reviewed decision (`dev/r06/structural-refactor-design.md:368-376`),
    so it needs a superseding record in `dev/decisions/`, not a silent flip.
  - Does **not** simplify the Snakemake side: `script:` directives are path-based
    and resolve against `workflow.basedir` either way. The win is import-side only.
  - Does **not** cover the 5 test inserts pointing at `dev/scripts` — that is not a
    package and is not shipped, so those stay.
  - `pixi.lock` regenerates on both platforms, and CI's `locked: true` turns any
    mistake into a red build.
  - ~~Breaks the Dockerfile ordering~~ — **lifted 2026-07-26** by the O-06 parking
    decision. `pixi install` runs before the source is copied, so a
    self-referencing editable dep could not resolve at image build; with Docker
    parked this is no longer a prerequisite, only a note for whoever rebuilds the
    image later.
  - Adds a stale-install failure mode (new top-level package → reinstall needed).

  **(3) Migrating the pixi manifest into `[tool.pixi]`.** Technically possible;
  **recommend against** — it buys nothing, churns a 100+ line manifest with
  load-bearing comments, and pixi's pyproject mode presumes the workspace *is* a
  Python package, which is only true after (2).

- **Recommendation.** Take (1) now — it is pure gain and reverses nothing. Raise (2)
  as its own decision record; it is the only route that actually retires the shims,
  but it is a genuine architecture change with a Docker prerequisite. Skip (3).
- **Note on root tidiness (O-11):** one `pyproject.toml` is a better answer than
  `pytest.ini` + `ruff.toml`, which would add two root files instead of one.

### O-15 — The PR template mandates Black, but no formatter is installed

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `.github/pull_request_template.md:11` carries the checklist item
  "Black formatting pass locally". `pixi.toml` declares **neither `black` nor
  `ruff`** — no formatter or linter exists in the pinned environment. There is no
  lint job in `.github/workflows/ci.yml`, and no config file for either tool.
- **Expected:** a checklist item that can be satisfied with a documented command.
- **Notes:** a `.ruff_cache/` directory sits at the repo root (created 2026-07-17),
  which means ruff has been run here from *outside* the pinned env — unpinned,
  unreproducible, and invisible to CI. So formatting is simultaneously mandated,
  unavailable, and unenforced.
  **Proposal (needs approval — this adds a dependency):** add `ruff` alone rather
  than `black` + `flake8` + `isort`; `ruff format` is Black-compatible and `ruff
  check` covers the lint/import-order surface, so one dep replaces three. Then:
  config in `pyproject.toml` `[tool.ruff]` (per O-14 option 1), a `pixi run lint` /
  `pixi run fmt` task, a CI job, and a rewritten PR-template line. This also gives
  `dev/roadmap.md:900` ("a future linter ... to enforce naming conventions") a place
  to land.
  **Caveat before enabling any rule set:** the first `ruff check` on 41 modules
  will produce a large finding list. Land the config with a deliberately narrow
  initial rule selection and widen it later; a formatting-only first pass keeps the
  diff reviewable and separable from behaviour changes.

### O-16 — `flit` is declared with nothing to build

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `pixi.toml:16` declares `flit = ">=3.2"`. There is no
  `pyproject.toml` and no `[build-system]`, so nothing invokes it.
  `dev/phase-1/m02b/package_inventory.md:64` already flagged it: "Build tool with
  no consumer."
- **Expected:** every declared dependency has a consumer.
- **Notes:** resolution follows O-14. If (2) is declined, drop the dep. If (2) is
  taken, note that flit's default *dynamic* metadata path reads the version and
  description from the package's module docstring and `__version__` — and all six
  `__init__.py` files are 0 bytes, so adopting flit means either declaring
  `version`/`description` statically under `[project]` or populating
  `blueearth_cst/__init__.py`. `hatchling` avoids that question entirely and is the
  lower-friction default if the choice is open.

### O-17 — `README.rst` overstates the Dockerfile's health

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `README.rst:130-131` — "The ``Dockerfile`` builds against the pixi
  env but is not exercised in CI." It does not build: `ADD src src` (O-06) fails on
  a missing path.
- **Expected:** the warning block states the real status.
- **Notes:** the surrounding warning is otherwise accurate and useful — Docker is
  already declared "Not supported in v0.2.0-alpha", deferred, with the v0.1.0-alpha
  published image (`README.rst:141`) named as the standing alternative. Only the
  "builds" clause is wrong. Fix it as part of recording the O-06 parking, and say
  plainly that the file is unmaintained pending a rebuild.

### O-18 — Should Linux be parked alongside Docker?

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged
- **Question (owner):** park Linux compatibility too, and pick up (1) Linux
  consistency and (2) the Docker image together once a beta settles?
- **Finding: "Linux" is three separable things, and only one carries a real
  recurring cost.**

  **(a) Linux end-to-end validation — already parked.** `dev/roadmap.md`
  ("Deferred: Linux replication") defers exactly this, and states the standing rule:
  Linux-specific files "must continue to build / parse but are not exercised
  end-to-end." Parking it again is a no-op.

  **(b) The `ubuntu-latest` CI leg — do NOT park.** It is not iteration work: it
  runs unattended in ~50 s, and `.github/workflows/ci.yml` has exactly **two
  commits** in its history (the initial add and the baseline record) — no
  maintenance since it landed. Its own comment records the reason to keep it: it is
  "the first place the linux-64 half of `pixi.lock` has ever been resolved at all",
  so it "doubles as the de-risking step for that parked work". Parking it in the
  meaningful sense — dropping `linux-64` from `pixi.toml`'s `platforms` — lets the
  Linux half of the lock rot, converting a phased resume into a big-bang re-resolve.
  It also removes the only check on the platform-divergence class the file calls
  out (Windows owns cp1252/CRLF/file-locking defects). *Honest caveat:* the leg
  landed 2026-07-25, so "zero maintenance" is a one-day record, not a trend.

  **(c) The five `*_linux.yml` variants — the real tax, and smaller and stranger
  than expected.** (Counting `config/` only; `docs/config/` holds five more, which
  O-05 retires.)

| File | Nature | Recurring cost |
|---|---|---|
| `snake_config_projections_cmip5_full_linux.yml` | Differs from its sibling by **one line** (the catalog path) plus trailing blank lines | Pure duplication |
| `snake_config_projections_isimip3_linux.yml` | Differs by **exactly one line** (the catalog path) | Pure duplication |
| `deltares_data_climate_projections_linux.yml` | Genuine 259-line twin; diff is only the root prefix (`p:/…` → `/p/…`) | Real sync tax |
| `snake_config_model_test_linux.yml` | Genuine variant in the current R01 schema; O-01 touches it | Real sync tax |
| `deltares_data_linux.yml` | **Not a variant — see O-19** | None; it is already abandoned |

- **Conclusion.** Parking Linux "for a few weeks" would change very little, because
  the sync tax being imagined is mostly **not being paid** — the variants are stale
  rather than maintained, and O-19 shows the largest of them was never migrated at
  all. Recommendation: keep the goal parked (it already is), **keep the CI leg**,
  and mark the variants explicitly stale rather than freezing them.
- **Deliberately not doing now:** collapsing the two one-line projections duplicates,
  which `dev/roadmap.md` already proposes ("Collapsing the OS-specific data catalog
  split … into a single parameterized catalog or config selection"). It looks free
  but is not: `data_sources` lives under the nested `project:` section, and
  Snakemake's `--config key=value` override does not reach nested keys cleanly, so
  removing the files means changing a user-facing config surface for a payoff of two
  files. Leave it to the Linux milestone.

### O-19 — `deltares_data_linux.yml` is a pre-1.0 hydromt catalog

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** the two `deltares_data` catalogs are **different schema
  generations**, not path-swapped siblings:
  - `config/catalogs/deltares_data.yml` (2596 lines) — hydromt **v1** form:
    `meta.roots` list, `uri:`, `driver: {name: pyogrio, options: …}`, `metadata:`.
  - `config/catalogs/deltares_data_linux.yml` (1894 lines) — hydromt **v0** form:
    `meta.root` scalar, `driver: vector`, `kwargs:`, per-entry `meta:`,
    top-level `crs:`.
- **Expected:** both parse against the pinned stack. `pixi.toml:21-22` pins
  `hydromt >=1.3,<2` and `hydromt_wflow >=1.0,<2`, so the v0 catalog would not load.
- **Notes:** this is not drift from a missed sync — the file was never migrated
  when the stack moved to hydromt v1 (the M2b bump, `dev/phase-1/m02b/audit.md`).
  It is masked because no Linux run has been attempted since. Same verdict as O-06:
  **rebuild at the Linux milestone, do not repair.** The right action now is a
  header comment marking it stale and a `roadmap.md` line adding "re-generate
  `deltares_data_linux.yml` for hydromt v1" to the Linux milestone's scope, so the
  work is not discovered mid-resume.

### O-20 — `examples/` is misnamed; it holds the local test fixture

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `examples/` is **entirely gitignored** (`.gitignore:124`, zero
  tracked files) and holds `test_local` — the `project_dir` of the tracked baseline
  seed (`config/workflows/snake_config_model_test.yml:12`) — plus `Gabon` for the
  Linux config. `AGENTS.md` already describes it as "a dev/test convention only
  (used by the baseline gate)". Nothing in it is an example.
- **Reinforcing point:** the repo's actual user-facing examples live elsewhere —
  `docs/notebooks/*.ipynb`. So the name is doubly misleading: it labels the test
  fixture as examples while the real examples sit under `docs/`.
- **Expected:** the directory name states what it is.
- **Is it cosmetic? No — but it is cheap.** Two concrete costs:
  1. **A `naming.md` §7 rename.** §7 (`dev/conventions/naming.md:141-147`) names
     "test fixture paths read by `tests/conftest.py`, `dev/scripts/check_baseline.py`,
     or other scripts" as requiring a `dev/<milestone>/migration_<topic>.md` note.
     This is exactly that.
  2. **4 of 18 baseline fingerprints go stale.** `dev/baseline/manifest.json` stores
     `"project_dir": "examples/test_local"` plus one key per target. The keys
     themselves are *derived* — `check_baseline.py:122-126` resolves
     `{project_dir}` templates — so rewriting them is mechanical. But four targets
     are **copied config snapshots**
     (`{project_dir}/config/snake_config_{model_creation,climate_projections,climate_experiment}.yml`
     and the wf3 `{exp_dir}/config/` one), fingerprinted as raw `sha256` of file
     bytes. Those snapshots contain `project_dir: examples/test_local` verbatim, so
     renaming changes their content and invalidates their hashes. They can be
     re-derived without a full workflow run (the snapshot is the tracked config with
     one string changed), but that must be done deliberately and verified — not
     assumed.
- **Naming — resolved 2026-07-26 after the "should this hold user runs?" question.**
  An earlier suggestion here (`runs/`) is **withdrawn.** The repo already operates a
  correct two-tier model, and the directory's name must encode which tier it is:

| Tier | Where | Why |
|---|---|---|
| **Production runs** | `project_dir` set to an **absolute path outside the repository tree** | Already the rule (`AGENTS.md:77-81`) and already supported — `project_dir` is consumed as a raw f-string prefix (`Snakefile_model_creation:29,57,69-74`), so an absolute path works with no code change |
| **The one test fixture** | repo-relative, gitignored | Forced, not chosen — see below |

  A general name (`runs/`, `projects/`, `output/`) is actively harmful here: it
  reads as "the place runs go" and invites users to write real basin output inside
  the toolbox checkout — precisely the confusion the rename is meant to end. A name
  that says *test fixture* refuses that use by itself.

  **Decision 2026-07-26 (owner): the target name is `test_case/`.** So
  `examples/test_local` → `test_case/test_local`, and the Gabon tree follows as
  `test_case/gabon` if it is kept at all (O-01 removes its observation data and the
  Linux config that points at it is parked per O-18). `.gitignore:124` `examples/`
  becomes `test_case/`. This name is also what O-22's exemption constant keys on.

- **Why the fixture cannot simply move outside the repo too.** Its path is
  load-bearing inside a **tracked** file. `config/workflows/snake_config_model_test.yml`
  is the tracked baseline seed — deliberately so, per the recorded lesson never to
  point `check_baseline.py` at an untracked `*_local.yml`. A tracked config cannot
  carry a machine-specific absolute path, so the fixture must live at a
  repo-relative location, or the seed needs env-var / machine-local-override
  indirection — which reintroduces exactly the untracked-local-config pattern that
  was resolved against. Secondary: `tests/conftest.py:59-62` resolves
  `join(SNAKEDIR, project_dir)`, and ~27 tests need the tree at that path.
  Accepted costs of keeping it in-repo: checkout disk usage (a full Wflow model plus
  experiment tree), and search/index noise in editors. Both tolerable; the
  tracked-seed constraint is not.
- **Sequencing note:** interacts with O-02, which routes DAG renders to
  `examples/test_local/dag`. Either land O-20 first, or accept that O-02's path is
  rewritten by it.

### O-21 — The config template ships a repo-relative `project_dir`, contradicting its own comment

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `config/workflows/snake_config.template.yml:15-18` — the comment
  reads "For production use, point this OUTSIDE the repository tree so generated
  model + result artifacts stay separate from the toolbox source", and the value
  immediately below it is `project_dir: examples/test`.
- **Expected:** the template — the file a new user copies as their starting point —
  ships the behaviour it documents.
- **Notes:** this is where the tier confusion originates. The template is the one
  config whose default is *meant* to be edited by a user, so its default should be
  an obvious outside-the-tree placeholder (e.g. `../cst_runs/my_basin`, or a clearly
  fake absolute path), not a path into the toolbox checkout. The `*_test*.yml`
  configs are correct as-is — they are tier-2 fixtures and *should* point inside.
- **Hardening: accepted — see O-22.**

### O-22 — Add a parse-time warning when `project_dir` resolves inside the repo

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** triaged (accepted)
- **Gap:** the two-tier rule in `AGENTS.md:77-81` (production `project_dir` outside
  the repository tree; one sanctioned in-repo fixture) is documentation only.
  Nothing detects a violation, and O-21 shows the shipped template itself teaches
  the wrong default.
- **Decision 2026-07-26 (owner):** implement it. **Warn, never raise.**
- **Design sketch** — follows the existing parse-time-validator precedent,
  `validate_experiment_name` (`blueearth_cst/shared/snake_utils.py:193-249`):

| Aspect | Choice |
|---|---|
| Location | `blueearth_cst/shared/snake_utils.py`, beside `validate_experiment_name` |
| Name | `warn_if_project_dir_in_repo(project_dir, repo_root)` — `verb_noun`, snake_case per `naming.md` §2 |
| Call site | Each Snakefile at parse time, immediately after `project_dir = get_config(project_cfg, "project_dir", optional=False)` (`Snakefile_model_creation:29` + the two equivalents). `repo_root` is already in hand as `Path(workflow.basedir)`, used for the `sys.path` insert |
| Detection | `os.path.abspath` both paths (matching `validate_experiment_name:242-243`, which uses `os.path` not pathlib), then `os.path.commonpath` for containment — not `startswith`, which mis-fires on sibling names like `…/repo-backup` |
| Exemption | `<repo_root>/test_case` only, held in a module-level constant so O-20's rename has exactly one place to touch |
| Severity | Warning to stderr, once. Never raises |

- **Why warn and not raise.** The tracked tier-2 configs legitimately point inside
  the tree, so an error would have to special-case them anyway; and a parse-time
  raise would break `--unlock` and `--dry-run` for anyone with an existing local
  setup. This mirrors the reasoning already recorded at
  `validate_experiment_name:198-202` for choosing parse-time *raise* there — a bad
  experiment name makes the whole DAG ill-defined, whereas an in-repo `project_dir`
  merely makes a mess.
- **Verify at implementation:**
  - `tests/test_cli.py` matches on combined stdout+stderr for DAG-build exception
    class names; confirm a new stderr line does not disturb those assertions.
  - The CI baselines must not move (386 / 30 / 1 on `windows-latest`, 385 / 31 / 1
    on `ubuntu-latest` — `.github/workflows/ci.yml`).
  - `profiles/default/config.yaml`'s `quiet: reason` suppresses Snakemake's per-job
    Reason block, not our own stderr write — confirm the line actually appears.
  - `tests/conftest.py` stages `tmp_path` project_dirs, which are outside the repo,
    so no warning and no test churn is expected there.
  - The warning fires once per Snakefile parse, so a `run_workflows.py` run prints
    it three times. Acceptable; note it rather than adding suppression state.
- **Sequencing:** land after O-20 so the exemption constant is `test_case` from the
  start and does not need a second edit.

### O-23 — Three homes for executable files: `scripts/`, `dev/scripts/`, `blueearth_cst/`

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** by-design
- **Question (owner):** why a root `scripts/` when `blueearth_cst/` already holds
  the workflow scripts?
- **Finding: the split is by *invocation model*, not by file type, and it is not
  redundant.**

| Home | Contents | Who runs it | How |
|---|---|---|---|
| `blueearth_cst/**` | 41 tracked modules | **Nobody, directly** | Imported/executed **by Snakemake** via `script:` directives and `Rscript --vanilla` `shell:` bodies. They read the injected `snakemake.input/output/params` global and have no `__main__` entry — running one by hand does nothing. `AGENTS.md:52-56` already says "none is a standalone CLI" |
| `scripts/` | 3 runners | **A human**, to execute the pipeline | `scripts/run_workflows.py --config …`, `scripts\run_snake_test.cmd` |
| `dev/scripts/` | 19 helpers | **A developer**, to inspect or maintain the repo | `dev/README.md` scopes it to "build, lint, profile, and exploratory one-offs" — `check_baseline.py`, `semantic_tree_diff.py`, probes, staging |

- **This was deliberated, not accidental.** `dev/r06/structural-refactor-design.md:588-596`
  records the decision to create a top-level `scripts/` *rather than* folding the
  runners into `dev/scripts/`, with the stated reason: "`dev/scripts/` is
  dev-process tooling … whereas `run_snake_test.cmd` / `run_snake_docker.sh` **are**
  how a user drives the pipeline." `MIGRATION.md:197-206` maps the move.
- **Decision: change nothing structural.** The dividing rule is clean and holds for
  every current file — **`scripts/` executes the pipeline; `dev/scripts/` inspects
  or maintains the repository; `blueearth_cst/` is executed by Snakemake, never by
  you.** Under that rule all three `scripts/` files are correctly placed
  (`run_snake_test.cmd` runs workflows; it is not a build/lint/profile helper), and
  nothing in `dev/scripts/` belongs in `scripts/`.
- **Considered and rejected:**
  - *Rename `scripts/` → `run/` or `bin/`* — ~15 reference updates across
    `AGENTS.md` (×5), `README.rst` (×5), `MIGRATION.md` (×4),
    `tests/test_run_workflows.py`, plus the depth comment at
    `scripts/run_workflows.py:71`. It would not fix the actual source of the
    confusion, which is that `blueearth_cst/*.py` *look* like scripts but are not
    runnable.
  - *Merge `scripts/` into `dev/scripts/`* — explicitly rejected in R6 for a stated
    reason; do not reopen.
  - *Refile `run_snake_test.cmd` into `dev/scripts/`* on the grounds that it
    hardcodes the test config (`:30`) and R6's own tree comment calls it a "Windows
    dev runner". Rejected: `dev/README.md` scopes `dev/scripts/` to build/lint/
    profile/one-offs, and a pipeline runner is none of those. Noted here because
    the R6 design's prose ("production/operational runners") and its tree comment
    disagree — if that folder's charter is ever revisited, this is the loose thread.
- **The one real gap is documentary.** `AGENTS.md` describes these three homes in
  three separate Repo Map bullets, by *audience*, which does not discriminate — all
  three sound like "scripts". Fix: one contrastive line stating the invocation-model
  rule above, so the question does not recur. No file moves.

### O-24 — Rule 1.13 `plot_forcing` has two undeclared outputs

- **Created:** 2026-07-26 · **Rev:** `75eb4d6` · **Status:** open
- **Observed:** `blueearth_cst/model/plot_map_forcing.py:170-179` loops over
  `precip` / `temp` / `pet` and writes one PNG per variable (`:137`). Rule 1.13
  (`Snakefile_model_creation:307`) declares **only** `precip.png` as its
  `output:`. The fixture confirms all three exist on disk
  (`plots/wflow_model_performance/{precip,temp,pet}.png`).
- **Expected:** every file a rule produces is a declared output.
- **Consequences:** `temp.png` and `pet.png` are invisible to Snakemake — not
  removed when the rule re-runs, not cleaned by `--delete-all-output`, not
  protected against a partial write, and absent from `dev/baseline/manifest.json`
  (which fingerprints `precip.png` only). A stale `temp.png` from an earlier
  config can survive a rerun and be mistaken for current output.
- **Notes:** independent of the output-layout restructure — fix regardless. It
  becomes more load-bearing under the 2026-07-26 ruling that forcing plots remain
  a first-class product (`dev/r07/2026-07-26_project-output-layout.md` §15a).
  Check the sibling plot rules (1.11 `plot_results`, 1.12 `plot_map`) for the same
  pattern while fixing: `plot_results` writes `clim_wflow_1_{month,year}.png` and
  `performance_metrics.csv`, of which only `hydro_wflow_1.png` is declared.

#### O-23a — Follow-on: should the three Snakefiles move into `scripts/`?

**No — it is a category error under the rule this observation just established.**

- `scripts/` holds things that **invoke** the pipeline: `run_workflows.py` and
  `run_snake_test.cmd` are wrappers whose job is to call `snakemake`. The
  Snakefiles are **the pipeline definition** — the thing `snakemake -s` consumes.
  They execute nothing on their own. Filing them under `scripts/` would put a
  declarative definition in the imperative-runner bin and break the only rule that
  justifies `scripts/` existing. By kinship they sit closer to `config/`.
- **If a subdirectory were wanted, the sanctioned name is `workflow/`, not
  `scripts/`.** Snakemake's own recommended project layout is
  `workflow/Snakefile` + `workflow/rules/` + `workflow/scripts/` alongside a root
  `config/` — a convention this repo already half-follows (root `config/`). But
  adopting it halfway (Snakefiles move, the package stays at root) buys the churn
  without the convention, and adopting it fully would mean relocating
  `blueearth_cst/` to `workflow/scripts/`, which fights the package decision
  recorded in O-03.
- **Cost, either destination** (from O-11): all 31 `script:` directives need a
  `../` prefix, because Snakemake resolves them against `workflow.basedir` — the
  *Snakefile's* directory, not the CWD; the three
  `sys.path.insert(0, str(Path(workflow.basedir)))` headers become `.parent`; and
  ~88 references across 28 non-`dev/` files change (`README.rst` ×16, `tests/`
  ×~28, `scripts/` ×10, the three notebooks ×15). **Benefit: three fewer root
  files.**
- **R6 already made this call.** The refactor moved the runners into `scripts/`
  (`structural-refactor-design.md:588-612`) and deliberately left the Snakefiles at
  the root — `:432` notes `tests/test_cli.py` "locates Snakefiles by `SNAKEDIR` at
  repo root — unaffected by the package move". `dev/conventions/naming.md:168` then
  codified "Snakemake entry points" as their own sanctioned file class.
- **Verdict: keep them at the root.** `AGENTS.md` calls them "the only entry
  points"; entry points at the front door is the conventional reading, and it is
  the one the repo has already committed to twice.

---

## Closure

When the assessment pass is done: promote surviving items to `followups.md` /
`TODO.md`, fill every `Disposition` cell, and add a short outcome summary at the
top of this section (what was checked, what held, what did not). This file then
stays as the durable record of the pass — it is a `dev/reviews/` artifact, not a
working note, so it is not deleted at closure.
