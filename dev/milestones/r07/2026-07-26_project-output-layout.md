# Project output layout — proposal for review

> **SUPERSEDED 2026-07-26 by `dev/milestones/r07/project-layout-design.md`** (with the path
> map at `dev/milestones/r07/migration_project-layout.md`), which consolidates this note and
> the repository-side observation register into one milestone design. This file is
> retained only as the working record of how the sixteen rulings were reached;
> delete it at task closure per `dev/README.md`. **Do not edit it further — edit
> the r07 design.**

> **Status: DRAFT for owner review. Not a design doc, not a decision.**
> Working note from the 2026-07-26 layout brainstorm (register:
> `dev/reviews/2026-07-25_post-r6-assessment.md`). Scope: the tree under
> `project_dir` only — the repository-side pass is closed out separately.
> Supersedes nothing until reviewed; the current accepted tree remains
> `dev/milestones/p31/experiment-structure-design.md:287-313`.

## 1. Principles

Three invariants, from which most of the tree below follows.

- **P1 — Figures attach to their producer.** Every figure lives in a `plots/`
  directory that is an immediate child of the subtree that produced it. There is
  **no project-level `plots/`**. Subfolders *inside* a `plots/` leaf are allowed;
  the rule constrains where the leaf attaches, not its internals. `plots/` holds
  figures only — no CSVs. *(Resolves the current three-way split: wf1 centralized,
  wf2 distributed, wf3 neither.)*
- **P2 — One producer per artifact.** No artifact is computed twice by two
  workflows. Where two workflows need the same input, one produces and the other
  consumes.
- **P3 — Engine-shaped artifacts live inside their engine's subtree.**
  *(Revised 2026-07-26 after owner review — the first draft said "`hydrology_model/`
  is upstream-governed, CST adds nothing to it". The evidence contradicts that: the
  live TOML already reaches outside the model root — `path_forcing =
  "../climate_historical/wflow_data/inmaps_historical.nc"`, `dir_output =
  "run_default"`, `path_static = "staticmaps.nc"`. Wflow does not dictate where
  forcing lives; CST chose that path. The upstream-governed surface is narrower:
  the TOML **schema** (CSDMS names, `[input.static.*]`), `staticmaps.nc` internals,
  and hydromt's build semantics — not the directory layout. P3-1 already proved
  this class of move by relocating run dirs to `experiments/<id>/model_runs/` and
  rewriting the pointers; hydromt re-relativizes on write.)*
  Concretely: anything in a **model-specific format** (wflow forcing netCDFs, run
  dirs, TOMLs) belongs under that engine's subtree. Generic, engine-independent
  data (raw climate extractions, projections) stays outside it. This is the
  future-proofing rule: a second engine gets its own subtree and its own input
  format, without touching `climate_historical/`.

A fourth, inherited from P3-1 and retained: **each `experiments/<id>/` is
self-contained and reproducible from its own directory.**

## 2. Target tree

```
<project_dir>/
  config/                                   # provenance snapshots — split confirmed (Q8)
    runs/       snake_config_model_creation.yml, snake_config_climate_projections.yml
    catalogs/   deltares_data.yml, cmip6_data.yml
    templates/  wflow_build_model.yml, wflow_build_model_run.yml,
                wflow_build_forcing_historical.yml, wflow_update_waterbodies.yml

  climate_historical/                       # GENERIC, engine-independent climate data only
    <key>/                                  # SINGLE store (§3.9). key = <clim_source>_<start>_<end>
      extract_historical.nc                 #   produced by wf1, consumed by wf1 + wf3
      orography.nc                          #   chirps branch sidecar
      .guard_ok
      plots/                                #   THE climate figures (§15) — source grid,
                                            #   produced without any wflow model

  hydrology_model/                          # the wflow ENGINE subtree (P3, revised)
    wflow_sbm.toml  staticmaps.nc  staticgeoms/  hydromt.log  hydromt_data.yml
                                            #   ^ hydromt model root == this dir (option A)
    forcing/inmaps_historical.nc            # MOVED from climate_historical/wflow_data/
    forcing/plots/                          #   OPTIONAL model-input QA only (§15) —
                                            #   climate figures live in climate_historical/
    run_default/                            # historical simulation (unchanged)
    evaluation/                             # was the top-level model_evaluation/
      performance_metrics.csv
      plots/                                #   hydro_wflow_1.png, basin_area.png, clim_wflow_1_*

  climate_projections/<clim_project>/       # §3.10 — no raw/ tier for now (Q5/Q15)
    timeseries/gcm_timeseries.nc
    summary/annual_change_scalar_stats_summary.{nc,csv}, *_mean.csv
    plots/

  logs/  benchmarks/                        # wf1 + wf2 (unchanged)

  experiments/<experiment_id>/              # id auto-suggested <project_name>_<YYYYMMDD> (Q1)
    config/snake_config_climate_experiment.yml
    data_catalog_climate_experiment.yml
    .project_consistency_ok

    weather_generator/                      # ENGINE subtree, mirrors hydrology_runs/ (Q6)
      config/weathergen_config.yml          #   the base config (kept)
      output/                               #   inmaps_rlz_*_cst_*.nc (temp), sim_dates.csv,
                                            #   resampled_dates.csv
      plots/                                #   obs_power_spectra, warm_annual_{precip,stats,wavelet}
      _work/                                #   ALL regenerable intermediates (Q16):
                                            #     cst_*.csv  (was stress_test/)
                                            #     weathergen_config_rlz_*_cst_*.yml
                                            #   `realization_*/` dissolves — configs here,
                                            #   netCDFs to output/

    hydrology_runs/rlz_<r>/                 # was model_runs/ (Q13); config/output split (Q3)
      config/cst_<c>.toml
      output/cst_<c>.csv

    indicators/                             # was model_results/ (Qstats, basin, RT_*)
      plots/                                #   response-surface figures (P1: attach to producer)
    logs/  benchmarks/
```

## 3. Changes, by remark

**9 — Collapse the two climate-historical stores. AGREED.**
`wf1_raw/` and `<key>/` hold the same grid; P3-2a shipped an `allclose` check
between them (`dev/milestones/p32a/climate-analysis-design.md:385,407`, ext1-5) — a
duplication the design noticed but did not resolve. One producer, others consume
(P2). The retired `allclose` check must come back as a unit test.

**Which bbox — REVISED 2026-07-26 after the standalone-climate ruling (§15).**
The first draft said "standardize on the staticmaps bbox". **Withdrawn:** the
staticmaps bbox requires a built model, which contradicts the goal that a full
climate analysis must run without a wflow setup. Use the **region-derived bbox**
instead — `blueearth_cst/model/get_region_preview.py` already delineates a
(sub)basin from a region string + data catalog with **no model**, so the store can
be bounded without wf1. Consequence: the producer is the **climate analysis**
step, and wf1 *and* wf3 are both consumers. The keyed store already supports this
— the key is `<clim_source>_<start>_<end>`, keyed by data, not by workflow.

**10 — Tier `climate_projections/`. AGREED.** `plots/` is already split; the gap
is processed-vs-summary. `timeseries/` + `summary/` + `plots/`. `raw/` stays
**opt-in**: wf2 streams CMIP6 from GCS and never persists slices — a deliberate
disk decision, and production configs use more models than the test config.

**11 — Keep `config/`. AGREED, with a caveat worth recording.** These snapshots
are not only provenance: the wf3 drift guard compares against
`config/snake_config_model_creation.yml`; four of the 18 baseline fingerprints
*are* these files; `file_digest_or_absent()` reads them at parse time. Deleting
them breaks the guard and the baseline. Optional tidy-up (low priority, only worth
doing inside a batch already re-recording the baseline): split into
`config/{runs,catalogs,templates}/`, separating provenance-of-settings from
provenance-of-inputs.

**12 — `experiments/` purpose confirmed. No change.**

**13a — `experiment_id`.** Constraint that decides it: **Snakemake needs a stable,
config-derived path.** A runtime-generated timestamp id makes every invocation
target a fresh directory — nothing is ever up to date, incremental reruns are
impossible, `--dry-run` lies, and the baseline gate loses its fixed path.
→ **Auto-*suggest*, never auto-generate.** A helper (e.g.
`run_workflows.py --new-experiment`) writes `experiment_name: gabon_20260726`
into the config **once**; the run then reads it exactly as today. Optionally
complement with an immutable run manifest (timestamp, git sha, config digest)
inside the experiment dir — run *identity* belongs there, not in a folder name.
No validator change needed: both `gabon260725` and `gabon_20260726` satisfy the
existing `^[a-z0-9][a-z0-9_]*$` grammar. **Open:** there is no `project_name`
config key today — derive from `basename(project_dir)`, or add one.

**13b — `model_results/` → `indicators/`.** Not `outputs/`: `model_runs/` also
holds outputs (the per-simulation CSVs), so `outputs/` blurs the boundary it is
meant to sharpen. The real distinction is per-simulation raw vs aggregated
indicators, and "indicators" is the CST term for exactly these. `summary/` is the
fallback.

**13c — Restructure `model_runs/`.** Today: flat, configs and outputs interleaved
— 12 members → 25 files; at production scale (RLZ 20 × ST 25 = 500) that is ~1000
flat files. → `model_runs/rlz_<r>/{cst_<c>.toml, cst_<c>.csv}`: mirrors the
existing realization grouping and keeps both dimensions legible. **Constraint:**
the tomls carry relative pointers (`input.path_static` →
`../../../hydrology_model/staticmaps.nc`); changing depth changes those strings.
P3-1 hit this exact issue (`experiment-structure-design.md:1145`) — hydromt
re-relativizes on write, but `semantic_tree_diff.py`'s path map must be updated
with the move.

**13d — Split intermediates from provenance.**
- **`stress_test/cst_*.csv` — KEEP where it is.** These define the perturbation
  applied to each member; without them the response surface is uninterpretable
  later. Provenance, not intermediate.
- **Per-member `weathergen_config_rlz_*_cst_*.yml` — demote.** Pure machine
  intermediates; nothing reads them after the run. Move under `_work/` (or wrap in
  `temp()`). The per-realization netCDFs are already `temp()`-wrapped and gone.

**13e — Keep `data_catalog_climate_experiment.yml`.** Clarification: there is one
per *experiment*, not one per stress-test member. It exists to fix a real
collision — wf3's `copy_config` used to overwrite wf1's project-level copy
(`dev/milestones/p31/migration_experiment-structure.md` §2). It is a byte-identical duplicate
today, so the instinct is right; but one small YAML is a fair price for the
self-containment principle the design chose deliberately (`:1726`). Alternative if
duplication is unacceptable: store a reference + content hash instead.

**13f — Route the experiment-root files.** Six files sit at the experiment root
(`obs_power_spectra.png`, `warm_annual_{precip,stats,wavelet}.png`,
`resampled_dates.csv`, `sim_dates.csv`) while `plots/` is reserved and empty — the
P3-1 migration doc notes "there is no wf3 plots producer". There now is one
(weathergenr); it writes to the wrong place. Figures → `plots/weathergen/`.
**Open:** where the two date CSVs belong — `_work/weathergen/` or a diagnostics
home.

**14 — Restructure `hydrology_model/` into an engine subtree. AGREED (owner,
2026-07-26); P3 revised accordingly.** Two shapes:

- **Option A — recommended (minimal).** The hydromt model root stays
  `hydrology_model/` itself, so the `model_root` passed to every hydromt
  build/update call is unchanged. Add `forcing/` and `evaluation/`; `run_default/`
  already is the simulation subfolder. The only pointer edit is `path_forcing`,
  from `../climate_historical/wflow_data/inmaps_historical.nc` to
  `forcing/inmaps_historical.nc`.
- **Option B — nest a `model/` subfolder** for `staticmaps.nc` / `staticgeoms/` /
  the TOML. Conceptually tidier (root holds only subfolders) but it changes
  `model_root`, touching every hydromt call and every derived pointer. Real risk
  for a cosmetic gain. **Not recommended.**

Consequence, and the reason this is worth doing: `climate_historical/` becomes
*purely generic, engine-independent* climate data. That advances the standing
modularization direction — climate analysis/visualization should run
model-independently, decoupled from the hydrology build. A second engine later
gets its own subtree and its own input format without touching
`climate_historical/`.

**Plots rule (P1) — the three-way split resolved.**
- wf2: already compliant, no change.
- wf3: route root PNGs into the reserved `plots/`.
- wf1: `{project_dir}/plots/wflow_model_performance/` conflates **three**
  processes in one folder. Under P1 + the revised P3 they separate as:
  - `hydrology_model/evaluation/plots/` — `hydro_wflow_1.png`, `basin_area.png`
    (+ `performance_metrics.csv` moves out of the plots dir entirely: `plots/`
    holds figures only)
  - `hydrology_model/forcing/plots/` — `precip.png`, `temp.png`, `pet.png`.
    **Correction to the first draft**, which routed these to
    `climate_historical/<key>/plots/`: rule 1.13 `plot_forcing` consumes
    `inmaps_historical.nc`, so these are plots of the **wflow forcing**, not of the
    raw extraction.
  - `clim_wflow_1_{month,year}.png` come from `func_plot_signature` via
    `plot_results` (the model run) → `hydrology_model/evaluation/plots/`.

**15 — Climate plots come from the climate data, never from the wflow forcing.
RULING (owner, 2026-07-26).** Standing principle, beyond this restructure: a full
climate analysis must be possible **without a wflow model setup or run**, and
climate figures are therefore produced from the extracted/processed climate store,
not from `inmaps_historical.nc`. This is the same direction as the standing
modularization goal (climate analysis as its own workflow, decoupled from the
hydrology build).

- **The only real obstacle is PET.** The raw extraction carries `precip`, `temp`,
  `press_msl`, `kin`, `kout`; **PET is derived during forcing generation**. Two
  routes:
  - **Source-grid PET (decoupled) — the one this ruling requires.** Compute
    debruin on the extraction grid using the source orography, which is already
    extracted as the sidecar for the chirps branch. No model, no `staticmaps.nc`.
  - **Parity / model-grid PET.** `blueearth_cst/model/climate_parity.py`
    (`model_parity_climate`, P3-2a §5.2) already reproduces "regrid + corrections
    + PET … exactly as the build does" — but its signature requires `dem_model`
    (`staticmaps.nc["land_elevation"]`), so it **needs a built model**. Correct
    tool for forcing QA; unusable for standalone climate analysis.
- **Two distinct products, both legitimate, different owners** (P1 was already
  saying this — the ruling decides which one owns the word "climate"):

| Product | Grid | Needs a model? | Home |
|---|---|---|---|
| Climate analysis figures | source | no | `climate_historical/<key>/plots/` |
| Forcing / model-input QA figures | model | yes | `hydrology_model/forcing/plots/` — optional, explicitly labelled verification, uses the existing parity function |

- **Concrete change:** rule 1.13 `plot_forcing` currently consumes
  `climate_historical/wflow_data/inmaps_historical.nc`
  (`Snakefile_model_creation:305`) via `blueearth_cst/model/plot_map_forcing.py`.
  Its climate role moves to a **new plot producer under
  `blueearth_cst/climate_analysis/`** consuming `<key>/extract_historical.nc`.
- **Remaining coupling to break:** `blueearth_cst/climate_analysis/subcatchment_climate.py`
  still aggregates the **wflow forcing** ("Aggregate gridded wflow forcing to
  per-subcatchment climate timeseries", citing ADR 0002). It sits in the
  `climate_analysis` package but depends on a model artifact. Repoint it at the
  extraction for the decoupling to be real.
- **Already in place:** the `climate_analysis/` subpackage exists
  (`extract_historical_climate.py`, `prepare_climate_data_catalog.py`,
  `subcatchment_climate.py`) and `get_region_preview.py` provides model-free basin
  delineation. The decomposition has started; what is missing is a climate plot
  producer and the subcatchment repoint.
- **Out of scope here.** Promoting climate analysis to a fourth Snakefile is a
  separate milestone. This note only ensures the *output layout* does not obstruct
  it — hence §9's region-derived bbox and this plots ruling.

### 15a. Forcing plots survive — RULING (owner, 2026-07-26)

Forcing/model-input QA plots are **kept**, not retired. If a downscaling step is
added later, its plots get their own separate home rather than being merged into
either existing set.

Consequences:

- **Cheapest possible change to rule 1.13.** `plot_forcing` keeps its input
  (`inmaps_historical.nc`) and its producer (`plot_map_forcing.py`); only its
  output path moves, to `hydrology_model/forcing/plots/`. The *new* work is a
  second producer under `climate_analysis/` reading `<key>/extract_historical.nc`
  → `climate_historical/<key>/plots/`.
- **Two `precip.png` will exist**, at different paths, showing different things
  (source-grid climate vs model-grid forcing). The directories disambiguate them
  for a human, but two things need a decision at design time:
  - **Baseline manifest:** `plots/wflow_model_performance/precip.png` is a
    tracked target today. After the split, decide whether one or both are baseline
    targets.
  - **A GUI/report collector globbing `**/plots/*.png` will surface both** under
    the same basename. If that matters, disambiguate by filename rather than
    relying on the parent directory.
- **The undeclared-outputs defect gets more serious**, not less: if forcing plots
  are a first-class product, `temp.png` and `pet.png` must become declared `output:`
  entries of rule 1.13 (see the defect note below).

### 15b. Source-grid PET need not match the build's PET — RULING (owner, 2026-07-26)

Climate-analysis figures are **approximate quick assessments**; the build's PET is
the detailed, refined model input. The two are allowed to differ.

Consequences:

- **No shared-code obligation.** The debruin call does *not* need factoring out of
  `climate_parity.py`; the climate step computes PET independently. This removes
  the coupling that would otherwise have re-tied climate analysis to
  model-build code.
- **A labelling obligation replaces it.** Two differing PET values will exist for
  the same basin and period, and someone *will* compare them and report it as a
  bug. The climate figures must say "approximate" on their face (title/caption
  convention), and the distinction belongs in the user docs — not only in this
  note.
- **Option opened, not decided:** since parity is no longer required, the climate
  step is free to use a *simpler* PET method (e.g. Hargreaves, temperature-only)
  instead of debruin. That would make climate analysis robust for datasets lacking
  radiation/pressure variables. It does **not** shrink the extraction store — wf1
  still needs the full variable set — so this is purely about the climate step's
  own input requirement. Worth a look when the climate workflow is specified.

### 16. `weather_generator/` — a second engine subtree (owner, 2026-07-26)

The weathergen artifacts currently scattered across the experiment root
(`weathergen_config.yml`, 4 PNGs, 2 date CSVs) and `realization_*/` become an
engine subtree with the **same internal shape as `hydrology_model/`**:
`config/`, `output/`, `plots/`, `_work/`, plus `stress_test/` for the perturbation
definitions.

Why this is more than tidying: it makes the experiment legible as **a pipeline of
two engines** — weather generator → hydrology model — each with an identical
internal contract. That generalises P3 from "the wflow subtree" to "every engine
gets a subtree", and it is what lets a future engine swap in without inventing a
new layout.

It also resolves open question 9 for free: the per-member forcing netCDFs
(`inmaps_rlz_*_cst_*.nc`) are **weather generator outputs**, so they belong in
`weather_generator/output/` rather than needing a separate forcing home. They stay
`temp()`-wrapped.

**Defect found while tracing the above (log to the register).**
`blueearth_cst/model/plot_map_forcing.py:170-179` loops over `precip`/`temp`/`pet`
and writes three PNGs (`:137`), but rule 1.13 (`Snakefile_model_creation:307`)
declares only `precip.png` as an `output:`. `temp.png` and `pet.png` are
**undeclared outputs** — untracked by Snakemake, not removed on rerun, and absent
from the baseline manifest. Independent of this restructure; fix either way.

## 4. Cost — why this lands as ONE milestone

Nearly every move touches the same two machines:
- `dev/baseline/manifest.json` — 18 targets, of which the wf1 plots (3), the wf2
  summary + plots (6), the wf3 results (2) and the 4 config snapshots are all
  path-affected. Piecemeal, each move costs its own re-record.
- `dev/scripts/semantic_tree_diff.py` — the directory-prefix path map plus the
  path-aware toml comparator both need the new mapping.
- `dev/scripts/check_baseline.py` — the `TARGETS` templates.
- A `dev/<milestone>/migration_<topic>.md` note is **required** by
  `dev/conventions/naming.md` §7 (rule-`all` output filenames + fixture paths read
  by `check_baseline.py`).

Single re-record if batched; N re-records if not. **Recommendation: one milestone.**

Sequencing note: interacts with the repository-side O-20 rename
(`examples/` → `test_case/`), which also invalidates the same four config-snapshot
fingerprints. Land them in the same re-record or accept doing it twice.

## 5. Question log

### Resolved 2026-07-26 (owner)

| # | Question | Ruling |
|---|---|---|
| 1 | `experiment_id` generation | **Auto-suggest at config creation** (never at run time — idempotence). `project_name` is **`basename(project_dir)`**; no new config key. Suggested form `<project_name>_<YYYYMMDD>`. |
| 2 | `indicators/` vs `summary/` | **`indicators/`** *(read from "Yes" against the recommendation — correct me if `summary/` was meant)*. |
| 3 | `model_runs/` shape | **Both**: group by realization *and* split by kind — `model_runs/rlz_<r>/{config,output}/`. |
| 4 | Intermediates dir name | **`_work/`**. |
| 5 | `climate_projections/raw/` | **Keep as a reserved placeholder**, empty by default. |
| 6 | Weathergen date CSVs | **New `weather_generator/` engine subtree**, mirroring `hydrology_model/`'s internal shape. See §16. |
| 7 | `evaluation/` naming | **`evaluation/`**, not `model_evaluation/` — it sits inside `hydrology_model/`, so the prefix is redundant. *(NB: this answers the name, not the option-A/option-B choice — see still-open below.)* |
| 8 | `config/` split | **In** — `config/{runs,catalogs,templates}/`. |
| 11 | Keep forcing/QA plots | **Yes, they stay.** §15a. |
| 12 | Source-grid PET parity | **Not required.** §15b. |

### Resolved as a side effect

| # | Question | How |
|---|---|---|
| 9 | Where wf3's per-member forcing lives | Answered by Q6: `inmaps_rlz_*_cst_*.nc` are **weather generator outputs** → `weather_generator/output/`. Still `temp()`-wrapped. |

### Resolved 2026-07-26, round 2 (owner)

| # | Question | Ruling |
|---|---|---|
| 13 | Experiment hydrology subtree name | **`hydrology_runs/`** — keeps the engine symmetry without colliding with the project-level `hydrology_model/`. |
| 14 | `output/` vs `outputs/` | **`output/`** (singular), everywhere. |
| 15 | Empty `raw/` placeholder | **Do not create it.** Add when a producer exists. |
| 16 | `stress_test/` placement | **Demoted to `weather_generator/_work/`** with the per-member weathergen YAMLs. Both are cheap, regenerable intermediates. `realization_*/` dissolves entirely: its configs go to `_work/`, its netCDFs to `output/`. |

**Q16 — the earlier "keep `stress_test/` as provenance" argument is withdrawn; it
was wrong on two counts.** (a) The perturbation grid is a deterministic function of
the `stress_test` block in the **preserved config snapshot**, so nothing is lost.
(b) More directly, the coordinates are **already denormalised into the indicator
outputs**: `indicators/Qstats.csv` carries `tavg` and `prcp` columns, and
`basin.csv` is exactly those two columns. The response surface is therefore
interpretable from `indicators/` alone.

**One precise caveat to carry into the reshape.** `cst_*.csv` holds **monthly**
structure (12 rows: `month`, `temp_mean`, `precip_mean`, `precip_variance`) while
`Qstats.csv` holds a **scalar** per member. For *uniform* monthly perturbations —
what every current config uses — the scalar is a lossless summary. But the config
schema permits per-month `min`/`max` arrays, and under a seasonally-varying
perturbation the scalar no longer identifies what was applied. If such
perturbations are ever used, the planned merged grid table stops being a
convenience and becomes **load-bearing** — and it should then be a first-class
output, not a `_work/` intermediate.

| 7b | `hydrology_model/` structure | **Option A** (confirmed 2026-07-26) — the hydromt model root stays `hydrology_model/` itself; `staticmaps.nc` / `staticgeoms/` / the TOML remain at that level, and `forcing/`, `run_default/`, `evaluation/` are added beside them. No change to the `model_root` passed to any hydromt build/update call. The only pointer edit is `path_forcing`. |

### Still open

- **10 — engine-named subtrees (PARKED, not blocking).** With `weather_generator/`
  beside `hydrology_runs/` in the experiment, and `hydrology_model/` at project
  level: keep descriptive names, or name for the engine (`models/wflow/`,
  `models/weathergenr/`)? Descriptive names read better; engine names scale better
  if a second hydrology engine appears. Revisit if and when a second engine is
  actually on the table — it does not gate this restructure.

**All other questions are resolved.** Next step is promotion from this working note
to a design document (see §6).

## 6. Promotion checklist

> **Executed.** The milestone was assigned R7 and the rationale below was
> promoted to `dev/milestones/r07/project-layout-design.md`. This note moved from
> `dev/working/` to `dev/milestones/r07/` alongside it — it is the R7 intake record, kept
> because three tracked documents cite it. The checklist stays as written for
> the historical trail.

This file was a `dev/working/` note — the convention deletes those at task closure,
so the rationale above had to move to a durable home before implementation:

1. Promote to `dev/<milestone>/project-output-layout-design.md`; **milestone
   number to be assigned**.
2. Write the `dev/<milestone>/migration_<topic>.md` old→new path map required by
   `dev/conventions/naming.md` §7 — every path in §2 that moves, in the same form
   as `dev/milestones/p31/migration_experiment-structure.md`.
3. Decide whether the repository-side items from
   `dev/reviews/2026-07-25_post-r6-assessment.md` (notably O-20,
   `examples/` → `test_case/`) land in the **same** milestone. They invalidate the
   same four config-snapshot fingerprints — one re-record if batched, two if not
   (§4).
4. Update `dev/scripts/check_baseline.py` TARGETS, `dev/scripts/semantic_tree_diff.py`
   path map, and re-record `dev/baseline/manifest.json` **once**, at the end.
