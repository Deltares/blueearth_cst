# Task Brief — Prototype the standardized Workflow 2 projection figures

> **Prototype only — reframed 2026-08-11.** This brief previously specified an
> integrated change to the WF2 figure contract. It now specifies a **design
> prototype**: render the proposed figure set outside the workflow, publish it as
> an Artifact, and obtain an owner ruling. **No code integration.** The
> integration in `dc40a22` predates this reframing; its disposition (keep, or
> revert to a prototype) is a separate owner decision and is not in scope here.

### Context

Canonical rules: `AGENTS.md`. Existing plotting conventions:
`blueearth_cst/shared/plot_style.py` and
`blueearth_cst/climate_analysis/climate_figures.py` — read them, reuse them,
change neither.

- WF2 is a CMIP6 plausibility overlay; it must not become an ensemble product or
  drive the WF3 stress-test grid.
- Every resolved `(model, scenario, member)` remains a separate trace or point,
  but only scenario receives a visual identity. Models and members do not appear
  in legends or receive distinct colors, markers, or line styles.
- The reference window is configuration-dependent. Monthly change means future
  calendar month versus the corresponding historical calendar month.
- The shipped monthly-change figures instead compare months with a historical
  annual mean, and the shipped monthly climatologies average the full future
  series rather than the configured horizon. **The prototype demonstrates the
  corrected semantics in its own code; it does not fix the producer.**

### Goal

Show the owner what a compact, standardized WF2 figure set would look like —
full-period overview figures plus horizon-specific monthly change-factor figures,
under corrected monthly semantics — as rendered images in a self-contained HTML
Artifact. The deliverable is a **design decision**, not a merged output contract.

### Non-goals

- **No integration.** Do not modify the producers, the workflow, the report, the
  tests, the figure inventories, or the baseline declarations. Explicitly, these
  files are out of bounds:

  ```text
  Snakefile_climate_projections
  blueearth_cst/projections/plot_proj_timeseries.py
  blueearth_cst/projections/get_change_climate_proj_summary.py
  blueearth_cst/projections/report.py
  blueearth_cst/shared/plot_style.py        (read and reuse; do not edit)
  dev/scripts/check_baseline.py
  dev/reference/workflows/rule-index.md
  dev/reference/workflows/wf2_climate_projections_overview.md
  tests/**
  ```

- No new or renamed durable figure paths. Nothing is written under any
  `project_dir` — not the fixture's, not a project's.
- No cross-model/member statistics, envelopes, medians, or aggregation.
- No model/member visual distinction or legend entries.
- No changes to change-factor tables, provenance, climate extraction, or
  regridding; no WF3 contact.
- No stress-test-grid overlay in the WF2 change-factor cloud.

### Allowed scope

**Permitted**

- One new prototype renderer: `dev/scripts/preview_wf2_projection_plots.py`.
- This brief.

**Approval-gated**

- The active plotting-standardization task note
  (`dev/tasks/t2608091006-…md`). Its half-2 checkboxes for the two WF2 producers
  currently read as swept, which the reframing may falsify — but that follows
  from `dc40a22`'s disposition, which is the owner's call, not this task's.
- Any shared-helper edit. If the shared contract is genuinely missing a
  primitive, demonstrate the gap and ask; do not add it.

**Forbidden / generated**

- `config/catalogs/cmip6_data.yml`, `config/catalogs/cmip6_store_index.json`,
  `pixi.lock`, `Manifest.toml`, and run outputs under any `project_dir`.

### The prototype renderer

Model it on `dev/scripts/preview_plots.py` — same argument style, same
`--list`/`--out-dir`/`--open` ergonomics, same header docstring stating what it
rebuilds and from what.

Inherit two of its rules:

1. Renders land in a **gitignored scratch tree** (`.tmp/` by default) and never
   in any project's `plots/`. A preview must not be able to stand in for a run
   product the baseline fingerprints.
2. It rebuilds its inputs from artifacts a finished run already left behind —
   the annual and monthly change-factor CSVs under
   `data/climate/projections/cmip6/summary/` and the projection series they were
   derived from.

Break one of them deliberately, and say so in the docstring: **do not register
this as a `preview_plots.py` family, and do not call the rule-side plotting
functions.** The point of the prototype is that the rule-side functions do not
implement the proposed design. Reaching for the existing `--list` registry
re-couples the prototype to the code it is meant to bypass.

**Input requirement.** The two-horizon case is the discriminating one, and
`test_case/test_local` cannot supply it — its config declares a single horizon
(`far: [2070, 2090]`). The executor supplies a source with **≥2 configured
future horizons** and ≥2 scenarios, either a real project tree opened read-only or
a synthetic frame built in the script; state which was used in the artifact. For
WF2 series the discriminating properties are horizon count and resolved
combination count — not the layer-richness rule that governs the basin map.

### Required prototype content (checklist)

1. Adopt the shared WF1 page, typography, layout, grid, month-label, and export
   conventions by importing `plot_style`. Scenario colors used consistently;
   historical traces gray.
2. Render this proposed structure into the scratch tree, using sanitized horizon
   names and their inclusive years, so the naming can be judged alongside the
   images:

   ```text
   overview/
   ├── annual-precipitation.png
   ├── annual-temperature.png
   └── change-factor-cloud.png
   windows/
   └── <horizon>-<start>-<end>/
       └── monthly-change-factors.png
   ```

3. Each annual figure carries absolute and anomaly panels over the full
   historical/future series. Draw every combination; color future traces by
   scenario only; one compact legend containing historical plus scenarios, never
   models or members. State the anomaly reference window and label the
   historical/future transition.
4. The change-factor cloud carries every combination, no marginal KDEs, scenario
   color only, and zero-reference lines. Facet by horizon with identical axes;
   one panel when only one horizon exists.
5. Each horizon figure carries precipitation change (%) and temperature change
   (°C) panels, computed against the corresponding **historical calendar month**
   using only that horizon's years, with every combination, scenario color only,
   a zero line, and `Jan`–`Dec` labels.
6. Close figures after saving.

### Validation

Figures are terminal artifacts and nothing here is consumed by a rule, so the
integration ladder does not apply. Three rungs, no more:

| Rung | Command/check | Frequency |
|---|---|---|
| Renders | The script runs end to end and writes every declared image without an exception, for both a one-horizon and a two-horizon input | Per iteration |
| Python gates | `pixi run lint`; `pixi run format-check` — `dev/scripts/` is linted (`extend-exclude` omits it deliberately) | Once before commit |
| Visual gate | Publish a self-contained HTML Artifact with every rendered PNG embedded as a base64 `data:` URI, and inspect at final size | Once, as the deliverable |

Do not run `pytest tests/`, `tests/test_cli.py`, `tree-check`, or the figure
baseline. Nothing is written under a `project_dir`, so the baseline question does
not arise.

**Falsifiers** — demonstrate these *in the artifact*, as numbers and images
beside the figures. "No code integration" removes the pytest home for them; it
does not remove the obligation.

- **Monthly semantics** — a synthetic case with unequal monthly baselines and
  extreme values outside the selected horizon must equal a hand calculation using
  only matching months inside the horizon. Show the hand figure, the prototype's
  figure, and what the shipped definition would have produced, so the correction
  is visible as a number and not only as a claim.
- **Combination preservation** — trace/point counts equal the resolved
  combinations. State both counts. Fewer marks disproves the
  no-aggregation/no-dropping claim.
- **Scenario-only identity** — legend labels equal historical plus configured
  scenarios. Any model/member label or model-specific style fails the contract.
- **Multi-window navigation** — the two-horizon input produces both window
  directories and matching cloud panels. A missing or mixed window fails.
- **Read-only** — if a real project tree is the source, hash it before and after
  the render; any change at all is a defect in the prototype.

### Acceptance criteria

- The Artifact shows every figure in the proposed set, at final size and legible
  at 180 mm width, including the two-horizon case.
- All combinations remain visible, with scenario as the only visual grouping;
  legends contain no model/member names; cloud marginals are absent.
- Monthly changes agree with the authoritative horizon-specific change-factor
  tables for precipitation and temperature, with the agreement shown.
- Figures visually match WF1 typography, dimensions, grids, labeling, and export
  quality.
- The repository is unchanged apart from the prototype script and this brief, and
  no `project_dir` was written to.
- The owner can answer "adopt this design, or not" from the Artifact alone.

### Commit plan

| Subject | Paths | Invariant preserved |
|---|---|---|
| Prototype the standardized WF2 projection figures | `dev/scripts/preview_wf2_projection_plots.py`, `dev/wf2-plot-standardization-task-brief.md` | Pathspec-scoped to the two files; no producer, workflow, test, or output-contract path is touched, so the WF2 figure contract is exactly as it was before the commit. |

### Output requirements

- The prototype script, committed.
- A self-contained HTML Artifact showing every proposed figure, including a
  ≥2-horizon case, with the falsifier evidence beside it.
- A short design note: what changes versus the shipped figures, which plotted
  values move because the monthly definition/window is corrected, and what the
  integration would cost if adopted.
- Exact commands run, results, anything skipped, and residual risks.
- A closing question asking the owner to adopt, amend, or reject the design.

### Task constraints

- Worktrees may run the prototype, pytest, lint, formatting, and dry-runs. Do not
  run a workflow from a worktree.
- Read project trees read-only. Do not write into, regenerate, or delete existing
  project outputs.
- Preserve public data-table schemas and the one-combination-one-mark rule.
- Use UTF-8, existing scenario identifiers internally, and presentation labels
  such as `SSP2-4.5` in figures.
