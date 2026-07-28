# AGENTS.md

> **Canonical.** Single source of truth for every runtime. Codex reads this file
> directly; `CLAUDE.md` is a thin entry point that imports it (`@AGENTS.md`).
> Author repo instructions **here**, never only in `CLAUDE.md`.

## Overview

BlueEarth Climate Stress Test — a multi-language (Python + R + Julia) scientific
workflow toolbox stitched together by **Snakemake**. The three `Snakefile_*` files
at the repo root are the only entry points; there is no package CLI. Full
narrative in `README.rst`.

## Background

Method context that changes how code here should be edited (full rationale:
`docs/cst-toolbox-technical-note-2025.md`, §1):

- CST implements **bottom-up climate stress testing** (decision-scaling / DMDU):
  instead of running selected GCM scenarios through the system ("top-down"), it
  systematically perturbs local climate across a temperature × precipitation grid
  and maps where system performance degrades. Stress-test scenarios come from the
  stochastic weather generator, **not** from climate projections — never couple
  the experiment workflow to CMIP scenarios.
- CMIP6 output (workflow 2) is a **plausibility overlay only**: its change factors
  situate the perturbation grid in projection space; they never drive stress-test
  runs.
- Pipeline ↔ method: workflow 1 builds a distributed Wflow-SBM model from global
  datasets via hydromt and runs it once on historical forcing (rapid deployment,
  no local calibration); workflow 2 computes per-(model, scenario, horizon)
  monthly change factors from CMIP6; workflow 3 is the stress test proper —
  weathergenr generates `RLZ_NUM` stochastic realizations, each perturbed across
  `ST_NUM` temp/precip combinations (`cst_0` = unperturbed baseline), run through
  Wflow, and reduced to hydrological indicators (e.g. mean/max discharge, 7-day
  low flow) that form the response surface.
- This repo is the **workflow engine** of a three-part platform (BlueEarth-CST
  workflows + CST-API backend + CST-frontend GUI). No web/API code belongs here;
  the GUI drives these Snakefiles server-side.
- CST targets rapid, first-order basin assessments on global data — not detailed
  engineering design. Prefer robustness and automation over site-specific
  sophistication.

## Repo Map

- `Snakefile_model_creation` / `Snakefile_climate_projections` /
  `Snakefile_climate_experiment` — the three workflow entry points, run in that
  order.
- `blueearth_cst/` — the analysis/orchestration package, split by workflow stage:
  `model/`, `projections/`, `experiment/` (one submodule per workflow), plus
  `shared/` for cross-cutting helpers (`snake_utils.py`, `run_logged.py`, plotting
  primitives, log/benchmark reducers) and `weathergen/` for the R weather
  generator. Modules are invoked from Snakemake `script:` directives (Python) or
  `Rscript --vanilla` `shell:` bodies (R); none is a standalone CLI. Imports are
  `from blueearth_cst.<stage>.<module> import ...`.
- `config/` — split into three bins: `workflows/` (`snake_config_*.yml` — the
  `--configfile` targets), `catalogs/` (hydromt data catalogs — `deltares_data*.yml`,
  `cmip6_data.yml` — the `-d` targets), and `templates/` (hydromt/wflow/weathergen
  build templates — `wflow_build_model.yml`, `wflow_update_waterbodies.yml`,
  `weathergen_config.yml`, plus the tracked `wflow_sbm.toml`).
- `scripts/` — user-facing runners: `run_snake_test.cmd` (Windows), `run_snake_docker.sh`
  (Linux/Docker), and `run_workflows.py` (the `enabled:`-aware wrapper, §"Key Commands").
- `dev/` — planning, audits, design docs, conventions, roadmap, the baseline
  manifest, and dev-process helpers under `dev/scripts/` (`check_baseline.py`,
  `semantic_tree_diff.py`, probes). Not user-facing; not shipped.
- `docs/` — user-facing reference (`install.md`, `env_setup_notes.md`, the vendored
  hydromt / hydromt-wflow / wflow user guides, the technical note, notebooks).
  **Configs are not mirrored here** — `config/` is the single source; the former
  `docs/config/` copies were kept byte-identical by hand and drifted anyway.
- `tests/` — `test_cli.py` is the cheap dry-run gate; `test_model_creation.py`
  is a heavy full build.
- `.github/workflows/ci.yml` — runs `pytest tests/` on push to `main` and on PRs,
  on `ubuntu-latest` + `windows-latest`, with `locked: true` (so `pixi.lock`
  drift fails CI). **It only covers what a bare checkout can run**: the
  fixture-dependent integration layer and the three `--run-integration` tests
  skip there by design, and `check_baseline.py` / whole-tree `semantic_tree_diff`
  cannot run in CI at all because they need the untracked `examples/test_local`
  tree. Those stay **local** gates — CI being green does not mean the baseline
  was checked.
- Outputs land under `project_dir` (set in the config). **Production `project_dir`
  lives outside the repository tree** — a run writes generated model + result
  artifacts to a location distinct from the toolbox source. The in-repo untracked
  `examples/test_local` dir is a dev/test convention only (used by the baseline
  gate), explicitly exempt from that rule.

## Key Commands

Run everything inside `pixi shell`, or prefix each command with `pixi run`, so
`snakemake`, `python`, and `Rscript` resolve to the pixi env.

```bash
pixi install          # conda-forge + PyPI deps (Python stack, R toolchain, snakemake)
pixi run install      # + weathergenr (R, via remotes) and Julia env (Pkg.instantiate)

# Run the three workflows IN ORDER (climate_experiment needs model_creation artifacts):
snakemake all -c 3 -s Snakefile_model_creation      --configfile config/workflows/snake_config_model_test.yml
snakemake all -c 3 -s Snakefile_climate_projections --configfile config/workflows/snake_config_model_test.yml --keep-going
snakemake all -c 3 -s Snakefile_climate_experiment  --configfile config/workflows/snake_config_model_test.yml

# Or drive all enabled workflows through the wrapper (reads workflows.<name>.enabled):
pixi run python scripts/run_workflows.py --config config/workflows/snake_config_model_test.yml

snakemake ... --dry-run           # inspect the DAG before running or after editing rules
snakemake --unlock -s <Snakefile> --configfile <cfg>   # Snakemake locks the workdir on crash

pytest tests/test_cli.py          # cheapest sanity check: dry-runs all three Snakefiles
pytest tests/                     # full suite (test_model_creation.py is slow)
```

Use `config/workflows/*_linux.yml` + `config/catalogs/*_linux.yml` variants on
Linux — data-catalog paths differ from Windows. `scripts/run_snake_test.cmd`
(Windows) and `scripts/run_snake_docker.sh` (Linux/Docker) wrap the test config.

**`scripts/run_workflows.py` — the `enabled:`-aware wrapper.** Reads
`workflows.<name>.enabled` from a **full-orchestration config** (one carrying a
`workflows:` section with all three subsections, each with an `enabled:` key — the
`snake_config_model_test*.yml` / `snake_config.template.yml` class; the
single-workflow `snake_config_projections_*.yml` configs have no `workflows:`
section and are **not** wrapper inputs) and invokes `snakemake` for exactly the
enabled workflows, in fixed order (model → projections → experiment). Contract:

- A missing `workflows:` section or `<name>.enabled` key is a **hard error** (nonzero
  exit naming the absent key), not a silent default-true.
- `enabled:` must **parse** to a real boolean (`isinstance(v, bool)` after
  `yaml.safe_load`): unquoted `true`/`false`/`yes`/`no`/`on`/`off` are accepted;
  quoted `"true"` / integers `1`/`0` are rejected.
- **Stops on the first nonzero Snakemake exit and returns that code** — a failed
  upstream workflow is not followed by a downstream run.
- `--cores N` and any args after a `-- ` sentinel forward to **every** invocation;
  each workflow keeps its own flags (`--keep-going` on projections only).
- **Skip semantics:** `enabled: false` means the wrapper does **not** invoke that
  Snakefile, so its outputs are not produced. It does **not** delete prior outputs
  and does **not** guarantee downstream freshness — an enabled downstream workflow
  consumes whatever prerequisite artifacts already exist on disk (or fails with
  `MissingInputException`), identical to invoking a single Snakefile directly.

## Conventions

- Name new identifiers and files per `dev/conventions/naming.md` (snake_case,
  lowercase acronyms, `_path`/`_dir` for paths vs `_ds`/`_df`/`_cfg` for objects,
  three-tier domain-identifier exemptions). Existing names are grandfathered;
  rename a contract surface only with a migration note.
- Snakefiles are config-driven: each parses one `--configfile` YAML via a shared
  `get_config(config, key, default, optional)` helper. Adding a config key: mirror
  that helper's contract (raise on missing required, return default for optional).
- Each Snakefile obtains the `--configfile` path from `workflow.configfiles[0]`
  and forwards it (as `config_path`) to downstream R scripts — keep that
  forwarding pattern even though the Snakefile itself reads the parsed `config`.
- The `ruleorder:` directive in `Snakefile_climate_projections` is retained as
  stale insurance, not confirmed load-bearing: a 2026-07 dry-run on the pinned
  Snakemake showed it constrains nothing on the tests fixture and a reduced
  config. Removal is deferred to a task that first encodes any ambiguity-sensitive
  config shapes as regression tests (see dev/r04/climate-projections-design.md §3).
- Register new data sources in a `config/catalogs/*_data*.yml` catalog and pass it to
  hydromt via `-d`. Never hardcode data paths in a Snakefile.
- **`dev/` vs `docs/` boundary:** `dev/` is planning + dev-process tooling (not
  shipped, not user-facing); `docs/` is user-facing reference. Put a new file where
  its audience is — a design note or one-off probe under `dev/`, an install/usage
  doc under `docs/`.
- [Python] Scripts run via `script:` read `snakemake.input/output/params` (a global
  Snakemake injects), not `sys.argv`. They are not runnable standalone.
- [R] Scripts run via `shell: Rscript --vanilla ...` take positional args parsed with
  `commandArgs(trailingOnly=TRUE)`.
- netCDF (`.nc`) is the interchange format across R/Python/Julia. Wrap intermediate
  per-realization netCDFs in `temp(...)` so they are deleted after consumers finish —
  omitting it explodes disk usage on large `RLZ_NUM × ST_NUM` runs.

## Workflow

- Before running or after editing any rule, `--dry-run` first to validate the DAG.
- After editing a Snakefile or a script signature, run `pytest tests/test_cli.py`.
- If a run crashed and re-running reports a locked directory, `--unlock` before
  retry.
- Keep edits surgical; follow the existing script/rule patterns before adding
  new structure.

## Hard Constraints

- **IMPORTANT: Julia is not in the pixi env** — it is juliaup-managed and must
  already be on `PATH` (conda-forge has no win-64 Julia build). Do not try to add it
  via pixi.
- Do not commit run outputs written under `project_dir`, or edit `pixi.lock` /
  `Manifest.toml` by hand.
- **Stay within CST's automation scope.** This repo is the workflow engine only.
  Define config/setup (`config/templates/wflow_build_model.yml`, data catalogs,
  `setup_*` blocks, `wflow_sbm.toml`-affecting steps) using hydromt / hydromt_wflow / Wflow
  conventions verbatim — CSDMS Standard Names (`hydromt_wflow/naming.py`), their
  YAML schema, their catalog format. Do **not** re-engineer how hydromt handles
  data, how `setup_*` methods work internally, or how Wflow parameterizes physics.
  Consume upstream behavior; verification steps may *read* upstream docs to
  validate our config but must not patch upstream. A genuine hydromt/wflow bug is
  flagged upstream or worked around in *our* code (`blueearth_cst/`, Snakefiles,
  `dev/scripts/`), never fixed inside the vendored package.

## References

- `README.rst` — the overall pipeline and how the three workflows fit together;
  start here.
- `docs/cst-toolbox-technical-note-2025.md` — stress-test method and design
  rationale; read before changing *what* a workflow computes.
- `docs/env_setup_notes.md`, `docs/install.md` — read when pixi / R / Julia
  setup or env activation misbehaves.
- `docs/hydromt-user-guide/00-index.md`, `docs/hydromt-architecture.md` — read
  when editing model-build config, data catalogs, or region setup.
- `docs/hydromt-wflow/getting-started.md`, `docs/hydromt-wflow/user-guide.md`,
  `docs/hydromt-wflow/api.md` — read when a build/update/clip step touches the
  hydromt_wflow plugin (`api.md` for exact signatures).
- `docs/wflow-user-guide/00-index.md` — read when editing `wflow_sbm.toml`,
  warm states, or Wflow run config.
