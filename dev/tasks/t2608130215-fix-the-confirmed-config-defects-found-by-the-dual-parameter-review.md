---
title: Fix the confirmed config defects found by the dual parameter review
type: todo-item
area: config / cross-workflow
origin: dual parameter review (2026-08-12)
created: 2026-08-13
updated: 2026-08-13  # all 8 fixed
---

> [!note] Overview
> **What** — Eight defects where a declared parameter does not do what it says. All verified in-session against the tree, not relayed from a reviewer.
> **Why** — One of them makes a SHIPPED config produce an empty experiment with no error. The rest are silent divergences between what a config declares and what runs.
> **Effort** — S each; A and C change results and need their own gates. Bundled because they share one cause, one verification method, and one review.

## Why these are one item

Every row is the same defect shape: *a parameter is declared and does not
reach, or does not control, what it claims*. They were found by one review,
they are verified the same way (trace the key to its consumer), and none is
cited by ID anywhere, so bundling breaks no reference. **A is separable** and
should be split out if it is picked up alone — it is the only one with a live
reproducer.

Source reviews, kept verbatim:
`dev/working/2026-08-12_config-parameter-review-{gpt,fable}.md`.
Brief: `dev/working/2026-08-12_config-parameter-review.task.md`.

## Confirmed defects

### A — a shipped config produces an empty experiment, silently

`wflow_outvars` carries **two different defaults**:
`Snakefile_model_creation:110` → `['river discharge', 'actual
evapotranspiration']`; `Snakefile_climate_experiment:478` → `[]`, which means
**zero indicator tables and no error**.

`test_case/snake_config_baseline_linux.yml` omits the key. Its only occurrence
there is a comment (line 3) stating the omission is deliberate "so the
Snakefile defaults apply" — the config documents an intent that the disagreeing
defaults betray. Verified by parsing: `'wflow_outvars' in model_creation` is
`False`.

Highest consequence in the set: a run completes, writes nothing, exits 0.

### B — `shared.julia_threads` and `runtime.julia_version` are bypassed where the compute is

`Snakefile_climate_experiment:1015` hardcodes the shell as
`julia +1.11.7 --project=. --threads 4` instead of calling `julia_prefix()`.
Rule 3.15 is the WF3 wflow batch — `RLZ_NUM × ST_NUM` runs, i.e. most of the
toolbox's compute — so the documented per-project thread knob misses it
entirely.

Two consequences, and the second is the worse one: the Julia **version** pin is
also hardcoded there, making a fourth spelling of a value AGENTS.md says three
files must agree on. `tests/test_julia_runtime.py` enforces the three; check
whether it sees this literal.

Fixing it is value-neutral today (`defaults.julia_threads` is 4, matching the
literal), so it can land without moving the performance baseline — but confirm
that before landing, since 4 is one leg of the frozen `(-c 3, --threads 4,
B=1)` triple every recorded measurement used.

### C — `resolution` has two unequal defaults

`blueearth_cst/spatial/config.py:243` defaults to `0.00833333`;
`config/templates/snake_config.template.yml:21` ships `0.00833`. Omitting the
key and copying the template give **different grid geometry**. One of the two
must become canonical.

### D — `stress_test.temp.variance.{min,max}` is accepted and never read

`prepare_cst_parameters` reads only `precip.variance.{min,max}`
(`:99-100`), and `DESIGN_COLUMNS` carries only `precip_variance_change`
(`:43`). The axis guard checks **top-level** axes only, so a sub-key of a valid
axis passes unexamined. A user can configure temperature variance and receive
unchanged results.

### E — `shared.clim_historical` is required by WF2 and unused there

`Snakefile_climate_projections:78` reads it `optional=False`; that is its only
occurrence in the file. WF2 fails without a value it never consumes.

### F — `project.static_dir` is required by WF3 and unused there

`Snakefile_climate_experiment:41` reads it `optional=False` and never uses it.
Separately, it can only ever hold `config`, because the WF1 fallbacks it feeds
resolve to in-repo toolbox files. Full analysis: M1 in
`dev/working/parameter-placement.md`.

### H — P1 spatial sources and the Wflow template disagree about who owns what

Verified 2026-08-13, promoted out of the unverified list. Three pairs, three
**different** failure modes — the reviewer's "duplication" framing understates
it:

- **`setup_laimaps.lai_fn` is inert.** `build_wflow_model.py:234` pops it with
  a `None` default and discards the value; LAI comes from
  `maps["leaf_area_index"]`. The template's `lai_fn: modis_lai` reaches
  nothing. This is the fifth inert parameter.
- **`setup_lulcmaps.lulc_fn` is silently REPURPOSED, and this is the hazard.**
  `:226` pops it, but the value survives as `source_name` and derives
  `lulc_mapping_fn = f"{source_name}_mapping_default"` (`:227`). The raster
  comes from P1's `maps["land_cover"]`; the **mapping table** still comes from
  the template. Set `shared.basin.spatial_sources.lulc: corine` and you get
  CORINE land cover interpreted through `vito_mapping_default`. The
  `maps.attrs.get("lulc_source", …)` fallback that would have caught it is
  never reached, because the template always supplies `lulc_fn`.
  Worse than inert: an inert key does nothing, this one does something wrong
  while looking right.
- **`setup_rivers.river_upa` and `shared.basin.river_uparea_km2` are two live
  knobs for one physical threshold.** `river_upa: 32` is forwarded to hydromt
  intact (only `hydrography_fn` / `river_geom_fn` are popped, `:214-215`);
  `river_uparea_km2` (default `DEFAULT_RIVER_UPAREA_KM2 = 32.0`,
  `spatial/config.py:262-264`) drives P1 delineation. Equal today, uncoupled.
  Change either alone and the P1 river mask disagrees with the wflow river map.

Consequence: the only defect in this bundle that can produce **wrong model
numbers** rather than missing or unchanged ones. Check `soil_fn` the same way
before fixing — it takes the generic `getattr(model, name)(**kwargs)` path
(`:236`), so it is forwarded intact and pairs with `spatial_sources.soil`.

### G — `DEFAULT_ANCHOR` is defined twice

`shared/metrics_definition.py:18` and
`climate_analysis/climate_figures.py:120`, same value. Introduced by this
session's own water-year work.

## Progress

Ordered easiest and safest first; A and C are last because they change results.

- [x] **G** — single-source `DEFAULT_ANCHOR` from
      `water_year_end_anchor(DEFAULT_WATER_YEAR_START)`. Trivial,
      non-breaking, no value change.
- [x] **E, F** — remove the two required-but-unused reads. Non-breaking for
      behaviour; F additionally deletes the key (see M1 for its cost, incl.
      `tests/test_guard_invalidation.py:241`, which uses it as its
      `_WF1_GUARDED` example).
- [x] **D** — make the axis guard check sub-keys, or reject
      `temp.variance` explicitly. Decide first whether temperature variance
      *should* be supported; refusing it is honest, implementing it is a new
      stress dimension and out of scope here.
- [x] **B** — call `julia_prefix()` in rule 3.15. Confirm value-neutrality
      (threads 4 → 4) before landing, and check `test_julia_runtime.py`
      against the hardcoded version literal.
- [x] **C** — pick one canonical `resolution` default. Value-changing for any
      config omitting the key; needs its own gate.
- [x] **H (lulc, lai)** — mapping table follows P1's source; both
      duplicated `*_fn` keys removed from the template (b31b9a3).
- [x] **H (soil, river_upa)** — owner ruling 2026-08-13: the project-level key
      owns both. `river_upa` now comes from `river_mask`'s own
      `upstream_area_threshold_km2` attribute (the threshold P1 actually
      delineated with), `soil_fn` from P1's `soil_source`; both deleted from
      the template so the loser is ABSENT. Landed `1464868`.
      **`soil_fn` was NOT the shape this note predicted** — hydromt reads the
      soil data itself, so nothing is injected. The coupling matters for a
      different reason: `_resample_source` namespaces every variable of
      `spatial_sources.soil` with a `soil` prefix, producing the `soil_*` maps
      **rule 1.12 plots**. A grep for the layer finds no reader and the figures
      depend on it anyway. Left free, the template could name a source the
      basin report never showed. Value-neutral on every shipped config
      (32 == 32, soilgrids == soilgrids).
- [x] **A** — align the two `wflow_outvars` defaults, or make WF3 refuse an
      empty set rather than emit nothing. Value-changing; own gate; add a test
      that a config omitting the key still produces indicator tables.

## Reported but NOT verified — do not action without checking first

Each comes from one reviewer only and was not confirmed in-session:

- `realizations_num`'s documented default `1` unreachable, because rule 3.10
  re-reads the YAML with a bare subscript and raises `KeyError` on absence
  (Fable).
- PET computed twice under two methods (already known as F16; not new).

## The meta-finding

The two reviewers **found different inert parameters** — GPT found `lai_fn`,
Fable found `clim_historical`, `julia_threads` and `realizations_num`, and
neither found the other's. Two careful manual passes over one codebase did not
converge, which is the case for the mechanical reach check both proposed
independently: [[t2608122022-verify-the-newly-honoured-water-year-and-sweep-for-projects-it-moves]]
tracks the related verification gap, and the placement draft's P2 states the
detection problem.

## Refs

- `dev/working/parameter-placement.md` — problem statement; M1–M4.
- Related: [[t2608121742-run-weather-generator-does-not-forward-relax-priority]]
  — a sixth parameter reaching nothing, already tracked separately.
