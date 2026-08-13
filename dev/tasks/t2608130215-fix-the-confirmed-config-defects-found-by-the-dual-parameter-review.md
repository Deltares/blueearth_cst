---
title: Fix the confirmed config defects found by the dual parameter review
type: todo-item
area: config / cross-workflow
origin: dual parameter review (2026-08-12)
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — Seven defects where a declared parameter does not do what it says. All verified in-session against the tree, not relayed from a reviewer.
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

### G — `DEFAULT_ANCHOR` is defined twice

`shared/metrics_definition.py:18` and
`climate_analysis/climate_figures.py:120`, same value. Introduced by this
session's own water-year work.

## Progress

Ordered easiest and safest first; A and C are last because they change results.

- [ ] **G** — single-source `DEFAULT_ANCHOR` from
      `water_year_end_anchor(DEFAULT_WATER_YEAR_START)`. Trivial,
      non-breaking, no value change.
- [ ] **E, F** — remove the two required-but-unused reads. Non-breaking for
      behaviour; F additionally deletes the key (see M1 for its cost, incl.
      `tests/test_guard_invalidation.py:241`, which uses it as its
      `_WF1_GUARDED` example).
- [ ] **D** — make the axis guard check sub-keys, or reject
      `temp.variance` explicitly. Decide first whether temperature variance
      *should* be supported; refusing it is honest, implementing it is a new
      stress dimension and out of scope here.
- [ ] **B** — call `julia_prefix()` in rule 3.15. Confirm value-neutrality
      (threads 4 → 4) before landing, and check `test_julia_runtime.py`
      against the hardcoded version literal.
- [ ] **C** — pick one canonical `resolution` default. Value-changing for any
      config omitting the key; needs its own gate.
- [ ] **A** — align the two `wflow_outvars` defaults, or make WF3 refuse an
      empty set rather than emit nothing. Value-changing; own gate; add a test
      that a config omitting the key still produces indicator tables.

## Reported but NOT verified — do not action without checking first

Each comes from one reviewer only and was not confirmed in-session:

- `setup_laimaps.lai_fn` forwarded then dropped (GPT) — a possible fifth
  inert engine parameter.
- `realizations_num`'s documented default `1` unreachable, because rule 3.10
  re-reads the YAML with a bare subscript and raises `KeyError` on absence
  (Fable).
- **T2 spatial sources duplicated inside the Wflow engine template** (GPT) —
  ranked its #1 by consequence, claiming divergent river masks or soil
  parameters could produce wrong numbers. Unverified and the largest claim in
  either report; verify before believing or dismissing.
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
