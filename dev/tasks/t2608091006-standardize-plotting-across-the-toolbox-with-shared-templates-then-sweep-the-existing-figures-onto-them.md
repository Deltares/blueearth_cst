---
title: Standardize plotting across the toolbox with shared templates, then sweep the existing figures onto them
type: todo-item
status: active
branch:
effort: 2
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-12
---

> [!warning] This is a sweep, not a lane
> `feat/plotting-standardization` was retired on 2026-08-12 when the standing
> lanes were redrawn by territory (AGENTS.md, *Standing lanes*). Its landed
> content — the prototype preview script and the brief — is on `main`. The
> remaining work edits call sites in both territories by definition, so it
> resumes on a **transient branch cut from `main`** that lands and is deleted,
> never on a parked branch beside the lanes.

> [!note] Overview
> **What** — Two halves. (1) Introduce shared plotting templates/style primitives so figures across the toolbox agree on typography, palette, figure size, colourbar placement and export settings, instead of each module deciding independently. (2) Sweep the existing figure producers onto them. Current plotting surface -- 8 modules under blueearth_cst/ plus 2 dev scripts. climate_analysis/climate_figures.py, climate_analysis/plot_climate_source.py, model/plot_results.py, projections/get_change_climate_proj_summary.py, projections/plot_proj_timeseries.py, shared/func_plot_signature.py, shared/plot_map.py, shared/snake_utils.py (plotting primitives), dev/scripts/basin_map_example.py, dev/scripts/preview_basin_map.py. No R-side plotting exists.
> **Why** — Owner request 2026-08-09. Figures are the deliverable a reader actually sees, and they are styled per-module today, so one basin assessment ships figures that do not look like each other. Design before implementing -- a shared template module is a contract surface every plot imports, and AGENTS.md is explicit that a shared helper edited in service of a plot is NOT a figure-only change. It takes the full validation ladder, while the figures themselves are verified by rendering and looking at them.
> **Effort** — large

## Progress

**Half 1 is complete; half 2 covers the climate-map family only.** The
climate-map sweep supplied the worked example, and `6d3ec75` then extracted the
shared page and typography contract to `shared/plot_style.py`.

Read the order as accidental rather than planned: the map family got swept first
because that is where the owner was looking, and the template it implies has to
be extracted from what those commits converged on rather than designed ahead of
them. That is a fine way round, but it means half 1 was an EXTRACTION job with a
worked example, not a greenfield design.

**The WF2 projection sweep is now a design question, not a sweep.** It was
implemented and landed on this branch as `dc40a22`, then reverted on 2026-08-11
at the owner's direction: the figure design should be ruled on before any
producer, Snakefile, report, test or output-contract change lands. The work is
respecified as a prototype in `dev/working/wf2-plot-standardization-task-brief.md` —
a renderer under `dev/scripts/` plus an Artifact, no code integration. The two
WF2 boxes below stay open until that ruling comes back, and the reverted commit
is where a future integration starts rather than a blank page.

**UNPAUSED 2026-08-17 — both open questions ruled. Resume at integration.**

> **Ruling 1 (owner, 2026-08-17): ADOPT the set as the WF2 output contract.**
> Integration is `dc40a22` **restored and re-validated, not rewritten** — but its
> figure code predates the four 2026-08-11 rulings, so the prototype's drawing
> functions are the newer source and win on every conflict.
>
> **Ruling 2 (owner, 2026-08-17): KEEP the current cloud orientation.** The axes
> do NOT flip to ΔT on x. Integration ships the orientation as rendered; the two
> cloud views stay in agreement with each other.

**What the ruling was taken on.** The prototype was re-rendered 2026-08-17 from
`test_case/test_local` with two horizons declared, and published as
[Artifact 777a2736](https://claude.ai/code/artifact/777a2736-4985-4fab-9c21-91d18a7993e3)
(supersedes the 2026-08-11 page,
[Artifact b04f34e3](https://claude.ai/code/artifact/b04f34e3-5268-4269-beeb-8a022a73f8f6)).
Command: `pixi run python dev/scripts/preview_wf2_projection_plots.py --horizon
near=2040-2060 --horizon far=2070-2090`.

**The falsifier is why this was a correctness change and not a restyle**, and it
should be quoted in the integration commit rather than re-derived: one
model×scenario×member drawn three ways — proposal, shipped definition, and the
authoritative table. The proposal sits ON the table's markers, max abs difference
**0.000491 %** for precip and **0.000894 °C** for temp over 72 rows each (worst
case `GFDL-ESM4 · ssp585` both times); the shipped definition visibly does not.
So the WF2 figures currently ship a monthly-change definition that the
change-factor tables contradict.

The prototype is `a80e47e`, revised once against the four rulings in `9f18b97`.

**Integrated 2026-08-17**, `feat/wf2-projection-figures`, four commits — and it
is a re-implementation rather than a `git revert` of the revert, because `main`
moved 14 commits on those files since `dc40a22` (two breaking renames, the
console overhaul, the config-snapshot P2/P4 wiring, ruff format + import sort).
`dc40a22` supplied the CONTRACT, the prototype supplied the drawing.

1. `projection_figures.py` — the figure set, defined once. Every downstream
   declaration derives from it. **`dc40a22`'s helper named four paths and was
   already wrong for the adopted set**: it predates the combined cloud, so every
   declaration deriving from it would have inherited the omission.
2. The monthly figures **read** `*_change_factors_monthly.csv` instead of
   recomputing it — see below.
3. `projection_plots.py` — the drawing, in its OWN module because the Snakefile
   imports the contract at parse time and `snake_utils` deliberately keeps
   matplotlib off that path.
4. Snakefile, `report.py`, `check_baseline.py`, the tree inventory and the two
   `dev/reference/workflows/` documents.

**The falsifier's finding became a structural fix, not an arithmetic one.** The
shipped rule recomputed the monthly change with two departures from the table
(historical ANNUAL mean rather than the same calendar month; the full 2015–2100
series rather than the horizon). Rather than correcting the formula, the producer
now renders the table — so there is one definition of a monthly change factor in
this repository and no second computation left to drift.
`test_values_come_through_untouched` passes a table carrying 999.0 and −42.0,
numbers no series produces, and asserts they reach the figure; it fails if anyone
reintroduces a formula.

Gates: 53 new unit tests; `test_cli` 14/14 (the rule-output gate); `pixi run
test-full` **2538 passed / 6 skipped / 1 xfailed** in 8:39; lint and
format-check clean. Figure gate: rendered through the PRODUCTION functions
against `test_case/test_local` and published as
[Artifact 3f2d49aa](https://claude.ai/code/artifact/3f2d49aa-c292-4e95-8483-2485912e0d32).
Counts match the prototype's exactly — 9 traces per overview panel, 6 cloud
points, 6 monthly traces — which is the evidence the port carried the semantics
and not only the shapes.

A full `snakemake -s analyze_projections.smk` was deliberately NOT run: its DAG
plans 26 jobs including nine `fetch_gcm_slice` network fetches and would mutate
the seeded fixture, which is the divergence hazard [[t2608121258]] records.

> [!question] Left open, not covered by the ruling
> `dev/scripts/preview_wf2_projection_plots.py` is now a **duplicate
> implementation** of figures the producers draw. Retargeting it at them would
> destroy the property that made its agreement check worth anything — it
> reimplements the change factors independently, which is why it could falsify
> the producers at all, the same reason `t2608152230`'s V21 notebook imported
> nothing from the module it checked. Three options: keep it as an independent
> falsifier (and say so in its docstring), retarget it, or retire it. Needs an
> owner call.

- **Run it:** `pixi run python dev/scripts/preview_wf2_projection_plots.py
  --horizon near=2040-2060 --horizon far=2070-2090`. Renders to `.tmp/`, reads
  the project tree read-only, and prints the agreement check.
- **What is decided:** no titles anywhere (`a)`/`b)` panel labels — a
  TOOLBOX-WIDE convention, not a WF2 one); the WF1 page spec is
  `_publication_rc()` + `series_figure_size()` + constrained layout +
  `supxlabel(wrap=True)`, not `tight_layout`; model names annotated on cloud
  points; the combined all-horizons scatter kept beside the faceted one. **Plus,
  2026-08-17: the set is ADOPTED, and the cloud orientation STAYS.**
- **Nothing is open.** Both questions were ruled on 2026-08-17; see the callout
  above. Integration is the next artifact, not another prototype revision.
- **Watch:** the two "no titles" and "WF1 page spec" rulings are toolbox-wide.
  The three surfaces still unswept below inherit them, and so does any figure
  work outside this item. The cloud-orientation ruling is narrower — it settles
  the two cloud views, and says nothing about other figure families.

- [x] Frame both climate map families on the basin — `11ffcc4`
- [x] Share colourbar levels across each source/forcing pair — `403b55e`
- [x] Bring the annual and monthly charts into the figure family — `d8f8ca8`
- [x] Colourbar title breaks at its unit; seasonal key dropped — `ab31ecc`
- [x] Retire the subcatchment climate plots (ADR 0006) — `700caa8`
- [x] Draw `basin_area` from the spatial foundation (ADR 0007) — `8cf2901`
- [x] **Half 1: extract the shared template/style module** — `6d3ec75` added
      `shared/plot_style.py` for page size, typography and export settings
- [ ] **Half 2, remainder — the surfaces still styled independently:**
  - [x] `projections/get_change_climate_proj_summary.py` — **integrated
        2026-08-17** on `feat/wf2-projection-figures`, four commits. Draws both
        cloud views from its own stage-B merge
  - [x] `projections/plot_proj_timeseries.py` — same sitting. Two annual
        overviews plus one monthly figure per horizon
  - [ ] `shared/func_plot_signature.py` (`plot_signatures`, `plot_hydro`,
        `plot_basavg` — `plot_clim` is gone, ADR 0006)
  - [ ] `dev/scripts/basin_map_example.py`
  - [ ] `dev/scripts/preview_basin_map.py`

**Two traps this item must respect**, both already paid for once:

1. `shared/cartographic_map.py` and `shared/plot_map.py` are contract surfaces —
   rule 1.12's basin map and rule 1.13's three forcing maps both draw through
   them. AGENTS.md § *Figures are terminal artifacts* is explicit that a shared
   helper edited in service of a plot is NOT a figure-only change and takes the
   full validation ladder. A template module every plot imports is that, maximally.
2. Anything assembled from the tunable block must be derived in a FUNCTION, not
   frozen into a module constant — a constant snapshots its inputs at import, so
   `preview_basin_map.py --set` would silently do nothing.

## Refs

- `dev/decisions/0006-retire-subcatchment-climate-plots.md`,
  `dev/decisions/0007-draw-basin-area-from-the-spatial-foundation.md` — the two
  rulings taken during the map sweep. 0007's cost is tracked as [[t2608091730]].
- `dev/working/wf2-plot-standardization-task-brief.md` — the WF2 projection figures,
  respecified as a prototype-only design question after `dc40a22` was reverted.
- `dev/scripts/preview_plots.py`, `dev/scripts/preview_basin_map.py` — render a
  family without a workflow run. The figure gate is: render it, publish the PNG
  as an Artifact, look at it. Never byte-compare, never run the baseline.
- Closed as superseded during this work: `t2608091028` (see `dev/LOG.md`) — its
  signature-figure redesign had already landed, and ADR 0006 then retired the
  figure. Relevant here because the sweep reaches `func_plot_signature.py`.
