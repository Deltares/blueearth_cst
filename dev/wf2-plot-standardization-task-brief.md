# Task Brief — Standardize and reorganize Workflow 2 projection figures

### Context

Canonical rules: `AGENTS.md`. Existing plotting conventions:
`blueearth_cst/shared/plot_style.py` and
`blueearth_cst/climate_analysis/climate_figures.py`.

- WF2 is a CMIP6 plausibility overlay; it must not become an ensemble product or
  drive the WF3 stress-test grid.
- Every resolved `(model, scenario, member)` remains a separate trace or point,
  but only scenario receives a visual identity. Models and members do not appear
  in legends or receive distinct colors, markers, or line styles.
- The reference window is configuration-dependent. Monthly change means future
  calendar month versus the corresponding historical calendar month.
- Current monthly-change figures instead compare months with a historical annual
  mean, and current monthly climatologies average the full future series rather
  than the configured horizon.

### Goal

Replace the nine independently styled WF2 plots with a compact, standardized
set organized into full-period overview figures and horizon-specific monthly
change-factor figures, while correcting the monthly semantics.

### Non-goals

- No cross-model/member statistics, envelopes, medians, or aggregation.
- No model/member visual distinction or legend entries.
- No changes to change-factor tables, provenance, climate extraction, regridding,
  or WF3.
- No stress-test-grid overlay in the WF2 change-factor cloud.

### Allowed scope

**Permitted**

- `blueearth_cst/projections/plot_proj_timeseries.py`
- `blueearth_cst/projections/get_change_climate_proj_summary.py`
- `blueearth_cst/projections/report.py`
- `Snakefile_climate_projections`
- Directly affected tests, figure inventories, baseline figure-path declarations,
  and live WF2 reference documentation
- The active plotting-standardization task note

**Approval-gated**

- Changes to `blueearth_cst/shared/plot_style.py` or shared map/figure helpers;
  reuse the existing contract unless a demonstrated missing primitive requires
  owner approval.
- Changes to any non-figure artifact path or schema.

**Forbidden / generated**

- `config/catalogs/cmip6_data.yml`, `config/catalogs/cmip6_store_index.json`,
  `pixi.lock`, `Manifest.toml`, and run outputs under any `project_dir`.

### Required changes (checklist)

1. Adopt the shared WF1 page, typography, layout, grid, month-label, and export
   conventions. Use scenario colors consistently; historical traces are gray.
2. Produce this durable structure, using sanitized configured horizon names and
   their inclusive years:

   ```text
   plots/
   ├── overview/
   │   ├── annual-precipitation.png
   │   ├── annual-temperature.png
   │   └── change-factor-cloud.png
   └── windows/
       └── <horizon>-<start>-<end>/
           └── monthly-change-factors.png
   ```

3. Each annual figure contains absolute and anomaly panels over the full
   historical/future series. Draw every combination; color future traces by
   scenario only; use one compact legend containing historical and scenarios,
   never models or members. State the anomaly reference window and label the
   historical/future transition.
4. The change-factor cloud contains every combination, removes marginal KDEs,
   uses scenario color only, and adds zero-reference lines. Facet by configured
   horizon with identical axes; use one panel when only one horizon exists.
5. Each horizon-specific monthly figure contains precipitation change (%) and
   temperature change (°C) panels. Calculate changes against the corresponding
   historical calendar month and use only that horizon's years. Draw every
   combination with scenario color only, a zero line, and `Jan`–`Dec` labels.
6. Update all Snakefile outputs/inputs, report listings, inventory declarations,
   live references, and tests atomically. Remove old figure names from the live
   contract; do not edit sealed milestone records.
7. Close figures after saving and retain the existing one-trace/point-per-
   combination count as an asserted property.

### Commit plan

| Subject | Paths | Invariant preserved |
|---|---|---|
| Standardize and reorganize WF2 figures | producers, Snakefile, report, tests, live references | Producers, declared outputs, report links, and path inventories move together; no commit exposes stale or undeclared figure paths. |

### Validation

Testing policy: **release / numerical affected**. The workflow entrypoint and
durable output paths change; monthly plotted values change meaning intentionally.

| Rung | Command/check | Frequency |
|---|---|---|
| Narrow | `pytest tests/test_plot_proj_timeseries.py tests/test_get_change_climate_proj_summary.py tests/test_report.py tests/test_project_tree_inventory.py` | Per relevant edit |
| Workflow contract | `pytest tests/test_cli.py` | Once before commit |
| Python gates | `pixi run lint`; `pixi run format-check` | Once before commit |
| Visual gate | From the primary checkout, render the complete set with `<GABON_CONFIG>`, publish self-contained HTML artifacts, and inspect at final size | Once after implementation |
| Full gate | `pytest tests/` | Once before merge |
| Tree contract | `pixi run tree-check --config <GABON_CONFIG>` against a regenerated project tree | Once before merge |

Do not run or re-record the figure baseline: figures are terminal artifacts and
the baseline excludes them. If a non-figure output changes, stop and diagnose.

**Falsifiers**

- Monthly semantics: a synthetic case with unequal monthly baselines and extreme
  values outside the selected horizon must equal a hand calculation using only
  matching months inside the horizon. Demonstrate that this test fails against
  the current implementation before fixing it.
- Combination preservation: trace/point counts must equal the resolved
  combinations. Fewer marks disproves the no-aggregation/no-dropping claim.
- Scenario-only identity: legend labels must equal historical plus configured
  scenarios; any model/member label or model-specific style fails the contract.
- Multi-window navigation: a two-horizon fixture must produce both window
  directories and matching cloud panels. A missing or mixed window fails.
- Non-figure stability: hash the annual/monthly CSVs and `provenance.json` before
  and after the render; any hash change is a regression outside scope.

### Acceptance criteria

- The declared tree and filenames match the structure above for one and multiple
  horizons.
- All combinations remain visible, with scenario as the only visual grouping.
- Monthly changes agree with the authoritative horizon-specific change-factor
  tables for precipitation and temperature.
- Legends contain no model/member names; cloud marginals are absent.
- Figures visually match WF1 typography, dimensions, grids, labeling, and export
  quality, and remain legible at 180 mm width.
- Report links, workflow declarations, tests, and live documentation contain no
  stale figure paths.
- Roll back if monthly values disagree with the tables, any non-figure artifact
  changes, or the multi-window layout cannot be read without model encoding.

### Output requirements

- One verified commit containing the atomic output-contract migration.
- A self-contained HTML artifact showing every new figure, including a run with
  at least two future horizons.
- A results delta stating which plotted values changed because the monthly
  definition/window was corrected and confirming that tables/provenance did not
  change.
- Exact validation commands, results, failures caught, skipped checks, and
  residual risks.

### Task constraints

- Run workflows only from the primary checkout; worktrees may run pytest, lint,
  formatting, and dry-run checks.
- Do not delete existing project outputs automatically. Regenerate into an
  independent project tree or obtain explicit owner approval for cleanup.
- Preserve public data-table schemas and the one-combination-one-mark rule.
- Use UTF-8, existing scenario identifiers internally, and presentation labels
  such as `SSP2-4.5` in figures.
