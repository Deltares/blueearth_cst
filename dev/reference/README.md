# dev/reference/

Rules the code must obey, and the durable descriptions of how it is put together. Consulted while working; rewritten rarely and deliberately.

This is the **stays-true** tier. What happened lives in `../milestones/`, `../decisions/`, and `../tasks/`; what is happening lives in `../TODO.md` and `../working/`; snapshots that decay live in `../reviews/`.

| Path | Holds |
|---|---|
| `naming.md` | Prescriptive style guide for identifiers and files, with `MUST` / `SHOULD` / `MAY` normative force |
| `wf0-figure-filename-rule.md` | The WF0 figure filename grammar (`<dataset_scope>_<variable>_<plot_context>_<spatial_scope>`), agreed 2026-08-17. Promoted out of `../working/` on 2026-08-19 — two Snakefiles, a shipped module and a test cite it |
| `indicator-glossary.md` | Every spelling of every WF3 output variable — config label, CSDMS name, csv code, token, table, metric — plus the metric vocabulary. **Derived** from the code's dicts and checked against them by `tests/test_indicator_glossary.py` |
| `agent-activation.md` | How roles and skills become available to Claude Code and Codex here, and why the two runtimes differ |
| `git-conventions.md` | Durable-ref inventory, plus the branching, tagging, and commit-message conventions |
| `repo-layout.md` | The long form of `AGENTS.md` § Repo Map — the `.gitignore` pattern for `test_case/`, basemap provenance, the `console.py` re-vendor rules, and what a bare-checkout CI run cannot cover. Split out 2026-08-20 |
| `task-lanes.md` | **Parked, deliberately unreferenced.** Holds the lane mechanics (slots, scope declaration, lifecycle) that belong in the `git-workflow` skill, plus the CST-specific worktree facts that do not. Kept as raw material for a skill revision; cite nothing from it meanwhile. Split out 2026-08-20, unlinked the same day |
| `validation-ladder.md` | The long form of `AGENTS.md` § Validation ladder — why the fast/full split lands at the `workflow_contract` and `process_isolation` markers, which config to run, and how to read a CI run. Split out 2026-08-20 |
| `contracts/` | The two substitution seams — hydrological model, weather generator — pinned as machine-checked contracts (P3-2b) |
| `workflows/` | Per-workflow contract docs for wf1 / wf2 / wf3, plus the WF2 v2.0 design, the CMIP6 member inventories, and `wf2-cmip6-store-readability.md` — which CMIP6 stores this toolbox can read (grid geometry, published-version multiplicity), promoted from two board items on 2026-08-20 |

## Two things to know before editing

- **These paths are cited from shipped code.** Measured, not assumed — `git grep -n "dev/reference/" -- ':!dev/'` is the check, and it is the only one there is, since a path in a comment resolves nowhere and no test fails when it rots:

  | Path | Cited from |
  |---|---|
  | `naming.md` | `AGENTS.md`, all four `*.smk`, `shared/snake_utils.py`, `scripts/run_workflows.py` |
  | `contracts/` | `shared/interchange_contracts.py`, `indicator_tables.py`, `spatial_geoms_parity.py`, `surface_axes.py`, `snake_utils.py`, `tests/test_interchange_contracts.py`, `docs/notebooks/Climate Stress Test.ipynb` |
  | `wf0-figure-filename-rule.md` | `analyze_climate.smk`, `build_model.smk`, `climate_analysis/figure_naming.py`, `climate_analysis/climate_figures.py`, `tests/test_figure_naming.py` |
  | `indicator-glossary.md` | `experiment/export_wflow_results.py`, `shared/indicator_tables.py`, `tests/test_indicator_glossary.py` |
  | `workflows/` | the three `*.smk` (rule-index), `model/write_outlet_index.py`, `shared/indicator_tables.py`, `config/templates/README.md`, two `docs/migration-*.md`; `wf2-cmip6-store-readability.md` from `projections/fetch_gcm_raw.py`, `projections/series_identity.py`, `tests/test_series_identity.py`, `dev/scripts/probe_cmip6_grids.py`, `dev/scripts/stage_cmip6.py` |
  | `repo-layout.md` | `AGENTS.md` |
  | `validation-ladder.md` | `AGENTS.md` |
  | `sealed-records.yml` | `AGENTS.md`, `pyproject.toml` |
  | `git-conventions.md` | `README.md` |

Renaming a file here means updating those citations in the same commit.
- **`workflows/` is not `.github/workflows/`.** This one holds prose contracts; that one holds CI definitions. A third, `config/workflows/`, was retired on 2026-08-10 (`7f776c4`) — every `--configfile` target now sits beside the project it writes into, under `test_case/`.

Everything here moved from the `dev/` root on 2026-08-02 — a path change only, no file renamed, split, or edited beyond the prefix. `conventions/` was flattened into this folder (it held two files); `contracts/` and `workflows/` kept their folders because each is a coherent multi-file unit.
