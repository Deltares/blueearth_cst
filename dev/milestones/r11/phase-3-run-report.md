# R11 P3 — run report

Live document. Written as P3 executes, not reconstructed afterwards.

Primary checkout, `worktree_policy` temporarily `concurrent` (see the file's own
banner — **restore to `always` at the milestone close**).

---

## Gate 1 — Q2, ruled 2026-08-08

`[R10-13]` lands **separately, after R11**. Reasoning and the weighed
counter-argument: scope §8. Board: `t2608071945` closed; `t2608071202` and
`t2608071219` stay open as one fix.

---

## Preconditions

Both were still live when P3 opened, as the brief predicted. One of the brief's
two claims about them was wrong on a fact.

### 1. The gabon config — NOT a tracked config

The brief says *"a **tracked** config no longer parses"*. It is not tracked:

    .gitignore:147:*_gabon.yml   config/workflows/snake_config_model_test_gabon.yml

Consequences, all measured:

- Every **tracked** config was already clean — `git grep aggregate_rlz -- config/`
  returns nothing. The brief's "grep every tracked config for retired keys" is
  therefore **done and green**, not outstanding.
- Fixing the gabon config produces **no commit**. It is a local file.
- The only on-disk survivals of `aggregate_rlz` are generated snapshots under
  `test_case/` — past-run outputs and frozen reference trees. Nothing live.

This is a fact error in the brief, not a ruled decision, so the "if the spec looks
wrong, PAUSE" constraint does not fire. Recorded and proceeded.

### 2. The frozen experiment — deleted, with its *before* preserved

`test_case/test_local/experiments/experiment` carried `aggregate_rlz` in its
frozen `experiment.yml:99`, so `check_not_frozen` would have refused at run time.
(It does **not** refuse at parse time — the WF3 dry-run planned 38 jobs happily.
The refusal lives in a rule body.)

Scope §9 assigned this deletion to P1; P1 did not do it. Done here as a
deliberate step.

**Copied first**, because deleting it would otherwise have destroyed the *before*
side of the scientific-delta gate — three manifest paths live under that
directory:

    test_case/test_local_pre_r11_experiment   # 64 files, 23 MB, verified count-for-count

That tree is **untracked** and joins the existing `*_pre_*` collection under
`test_case/`. It is retained for gate 2 and may be pruned once R11 is sealed.

---

## What the dry-runs said before anything ran

All three workflows plan a **full rebuild** — not just WF3:

| workflow | jobs | trigger |
| --- | --- | --- |
| WF1 | 16 | `code has changed: extract_historical_climate`, cascading to everything |
| WF2 | 15 | `code + params changed: reduce_gcm_series` |
| WF3 | 38 | code change upstream + outputs to generate |

**This widens the re-record beyond what P2's delta predicted.** P2 stated WF1/WF2
targets were "outside P2's blast radius" — true of P2's own *code changes*, and
not true of the *run*. `run_default/output.csv` and the two CMIP6 change-factor
CSVs regenerate too. Two of them will legitimately move for a reason P2 named
elsewhere: `config/runs/snake_config_{model_creation,climate_projections}.yml`
still snapshot `aggregate_rlz` at line 82, because P1 removed the key from the
tracked seed config after those snapshots were written.

It also makes **`[R10-12]`'s acceptance gate live rather than hypothetical**: WF1
really does rebuild, so `inmaps_historical.nc` really does move.

---

## Pre-registered delta expectations (gate 2 input)

Written **before** the run produced a single number, so the gate is a prediction
being tested rather than an explanation being fitted. Sources are the writer's own
contract (`export_wflow_results.py` docstring, `shared/indicator_tables.py`), not
the output.

The old table is wide and rounded:

    statistic,temp_change,precip_change,Q_101,Q_1050,Q_1040,Q_1020,Q_1030,Q_1010

66 data rows = 11 statistics × 6 grid cells, 6 gauge columns, values at 2 dp.
The fixture is 2 realizations × 6 stress-test members × 6 gauges.

**A direct tool comparison is impossible and must not be attempted.** The header
went 6 → 7 columns and the shape went wide → long, so `compare_indicator_table`
on old-vs-new returns a structural failure and nothing else. Gate 2 is manual
reasoning over rows whose identity survives the reshape.

| class | metrics | expectation for the same (metric, cell, gauge) | why |
| --- | --- | --- | --- |
| **A** — linear in years | `mean`, `max`, `min`, `q95`, `Q7day_max`, `Q7day_min`, `BaseFlowIndex` | **mean over the new `realization_id` 1..2 should reproduce the old value**, to within the rounding P1 removed | realizations are equal-length, so per-realization values average back to the pooled value *exactly* (writer docstring). `aggregate_rlz` chose between grains that were not different |
| **B** — non-linear fit | `return_level_max` (was `returninterval`) | pooled either way; **should be close** | same GEV, same pooled sample |
| **B** — 7-day | `return_level_7day_min` (was `returninternval_min_7day`) | **EXPECTED TO MOVE, and a move is CORRECT** | the pre-R11 code butt-spliced realizations onto a synthetic continuous date range; a `rolling(7)` window then crossed each splice and manufactured 7-day flows *that occurred in no realization*, which could become the year's annual minimum and enter the GEV block sample. P1 extracts each realization's minima within that realization and pools the blocks. This is a defect fix |
| **C** — selects a category | `wetmonth_mean`, `drymonth_mean` | **EXPECTED TO MOVE, and a move is CORRECT** | Q5: the month is now fixed once from the `st_0` baseline and evaluated for every member, instead of each member re-picking its own month. That is the ruled behaviour change |

So the gate's question is **not** "did anything move" — four of eleven metrics are
supposed to. It is: *did anything move that is not on this list, and do the class-A
means reconstruct?*

`basin_indicators.csv` needs no delta reasoning. The pre-R11 file is **87 bytes and
carries no values at all** — a header of `temp_change,precip_change` and six grid
rows, because the seed config declares no basin variables. Removing its manifest
entry loses no coverage that ever existed.

---

## The run

WF1 → WF2 → WF3, `-c 3`, from the primary.

### WF1 — `Snakefile_model_creation`, exit 0

16/16 jobs. Log merged into 14 `== 1.NN` sections in rule-number order; no
surviving `logs/_parts/`.

`check_baseline check --workflow model_creation` → **1 of 5 targets moved.**

**`run_default/output.csv` PASSED the discharge comparator.** This was a declared
stop condition, not something to absorb: `[R10-12]`'s entire argument that
`inmaps_historical.nc` moves on *storage layout* and not on values rests on
`output.csv` being reproducible across a rebuild. It was, again — so this run is
fresh evidence *for* the ruling rather than merely a consumer of it. Had it
failed, the correct response was to stop, not to re-record.

The one move is `config/runs/snake_config_model_creation.yml`, and it moved by
**exactly one key**:

    current                              78e0ca0c...
    current + aggregate_rlz: true        e223eaf7...
    recorded in the manifest             e223eaf7...

Adding the retired key back recovers the recorded hash byte-exactly. That is the
same falsifier R9 used when it proved its movement was header-only, and it is
stronger than reading a diff: it shows nothing *else* moved. Cause is P1 removing
`aggregate_rlz` from the tracked seed config after that snapshot was written.

*(Care needed reading `test_case/test_local_pre_r10-12/` as a reference here — it
also differs by `max_count` → `max_per_basin`, which is a pre-R11 change. That
tree predates more than this milestone; the hash-recovery check above is the one
that isolates R11.)*

### WF2 — `Snakefile_climate_projections`, exit 0

15/15 jobs. Log merged into 6 `== 2.NN` sections; no surviving `logs/_parts/`.

`check_baseline check --workflow climate_projections` → **1 of 3 targets moved**,
and it is the same config snapshot with the same one-key cause (all three
workflows snapshot the same source config, hence the identical hashes).

**The two CMIP6 change-factor CSVs did NOT move** — worth stating, because
`reduce_gcm_series` was one of the rules the dry-run flagged as changed in both
code *and* params. Changed inputs to the rule, identical numbers out.

### WF3 — `Snakefile_climate_experiment`, exit 0

40/40 jobs. Log merged into 15 `== 3.NN` sections in rule-number order; no
surviving `logs/_parts/`.

**The R side executed correctly on the first attempt.** `generate_weather.R`'s new
4-argument arity, `save_plots`, `seed` and `pet_method` had never run in P1 or P2
— the brief budgeted for a fix cycle here and none was needed. Member filenames,
the `cst_` → `st_` rename and the zero-padded indices all resolved: no missing
input at 3.12/3.14/3.15, which was the rename's only real falsifier.

The experiment name is pinned (`experiment_name: experiment`), so the tree
rebuilt at `experiments/experiment/` and the manifest paths held. A minted
date-based name would have silently broken every recorded path.

---

## DEFECT FOUND BY THE RUN — two of eleven metrics silently vanished

**This is what the brief meant by "expect the run to find something." It is not a
dry-run-able defect, and it is a regression against the pre-R11 table.**

### What is on disk

The new table has **576 data rows / 9 metrics / `st_id` 1–6**. The design table
next to it declares **`st_id` 0–6**. Expected, had nothing been dropped:

    observed                            576   (st 1-6, class C absent)
    st 0-6 with class C present         756

180 rows missing. `wettest_month_mean` and `driest_month_mean` are absent
**entirely**, and both existed in the pre-R11 table under the same config.

### Mechanism, traced

`config/workflows/snake_config_model_test.yml:63` — the **seed config the
baseline is recorded from** — sets `run_historical: false`. Then:

    Snakefile_climate_experiment:98    ST_START = 0 if run_hist else 1
    export_wflow_results.py:337        if q_locations and 0 in runs:

With `ST_START = 1` the baseline member never runs, so `0 not in runs`, so the
class-C month is never picked and both month metrics are skipped. No warning, no
error, no log line — the WF3 log contains zero matches for warn/skip/drop/empty.

The writer's own comment at line 335 states the dependency and then assumes it
away: *"Requires the baseline runs to exist, which ST_START = 0 guarantees
whenever `run_historical` is set."* The seed config does not set it.

### Why every gate passed

`validate_hm7` checks **results → design**: each row's `temp_change` /
`precip_change` must match the design table's row for its `st_id`. It does not
check **design → results**: that every declared member produced rows. So a
member that never ran, and a metric family that never emitted, are both invisible
to it. That asymmetry is also why P2's reasoning for *not* baselining
`stress_test_design.csv` — "already covered by `validate_hm7`'s per-row check,
which is stronger than a byte fingerprint" — is true for the rows that exist and
silent about the rows that do not.

### It contradicts two standing rulings

- **Q5** (2026-08-05): *"This makes `cst_0` rows mandatory — the month cannot be
  picked from a record that is not there."*
- **`[R9-5]`** (scope §3, ruled 2026-08-07): *the baseline is a member of the
  surface.*

Both say the baseline member belongs in the table. The implementation makes it
conditional on a config flag, and the seed config takes the branch where it is
absent.

**Held for an owner ruling before the re-record.** Recording as-is would bless a
silent regression into the baseline, and the baseline is exactly the artifact
that would then make it look intentional.

### Ruled 2026-08-08, and a SECOND defect found while fixing it

Ruled: set `run_historical: true` in the seed config, and add the design →
results coverage check. Both landed in `25c174b`.

Adding the check meant passing `design=` to `validate_hm7` in
`test_hm7_integration`, which had always called it bare — so **C28's consistency
check had never run against the fixture at all**. It failed on first contact:

    design  precip_change = -30.000001192092896
    results precip_change = -30.0

`prepare_cst_parameters.py` wrote each member CSV from a float32 frame and then
derived the design row from that same **in-memory** frame, while the weather
generator (3.12) and the results writer (3.16) both read the persisted text back
as float64. `float32(0.7)` is `0.69999998807`, so the design table documented a
−30.000001% perturbation **that was never applied to anything** — the run imposed
the round-tripped −30.0%.

Two things worth carrying forward:

- **P2's argument for not baselining `stress_test_design.csv`** — "already
  covered by `validate_hm7`'s per-row check, which is stronger than a byte
  fingerprint" — rested on a check that was not running. The reasoning was sound
  and the premise was false.
- **The unit test compared the right two things at the wrong precision.**
  `test_design_values_use_the_indicator_tables_own_reduction_AND_units` used
  `pytest.approx` (rel 1e-6) while `_close` used 1e-9 for the same invariant. A
  4e-8 error fits between them. The defect lived in the gap between two
  tolerances for one contract. That comparison is now exact.

Ruled: derive the design row from the persisted file. The two sides stay
independent in code, rule and job but now read the same bytes — which costs the
ability to catch a lossy CSV write, accepted deliberately rather than widening
`_close` and leaving the table wrong.

---

## The re-run, and gate 3

Config change → all three workflows re-ran. WF1 14/14, WF2 6/6, both green.

**WF3 stopped at rule 3.06 on the model-drift guard — `[R10-12]`, live.** The
runbook line added earlier this same phase got its first real use and gave the
right answer.

Evidence brought to the gate, hashed directly rather than trusting the guard's
own message:

| model input | recorded | live | |
| --- | --- | --- | --- |
| `staticmaps.nc` | `12a8c2ea…` | `12a8c2ea…` | identical |
| `wflow_sbm.toml` | `ae7f5142…` | `ae7f5142…` | identical |
| `forcing/inmaps_historical.nc` | `2108ec9f…` | `b447df0f…` | **moved** |

Exactly one input, and it is the forcing NC — the known layout-only case.
`run_default/output.csv` also passed the discharge comparator after **both** WF1
rebuilds; it is a deterministic function of (forcing, model, TOML), so had the
forcing values moved, it would have.

**Operator accepted the rebuilt model** (gate 3, 2026-08-08). The superseded
`model_reference.yml` is preserved at
`scratchpad/model_reference_accepted.yml` so the acceptance is auditable rather
than merely asserted. WF3 then ran 38/38, exit 0.

### Both defects confirmed fixed, at the predicted numbers

    rows      756   (predicted 756; was 576)
    metrics    11   (predicted 11;  was 9)
    st_id     0-6   (predicted 0-6; was 1-6)

---

## Gate 2 — the scientific-delta check

**A probe bug first, recorded because it nearly produced a false finding.** The
first run reported five metrics moving 2–16%, with the note *"mean of 4
realization(s)"* — against a 2-realization fixture. Cause: the probe keyed old
rows to new rows on `(temp_change, precip_change)`, and that pair is **not
unique across members**. The baseline sits at the origin by definition and the
seed grid also contains a `(0, 0)` point, so `st_0` and `st_2` share a cell. The
old wide table carried no `st_id`, and its `(0,0)` row is the GRID POINT — `st_0`
never appeared in it, because `aggregate_rlz` dropped the baseline. Keying on the
pair averaged the unperturbed baseline together with the grid point. Re-keyed via
the design table, excluding the baseline.

Had that gone unnoticed it would have been read as "R11 moved every class-A
metric by up to 16%", and the obvious next move — widen the tolerance until it
passes — would have buried the real signal underneath.

### Result, per metric (old values are 2 dp, so ≤ 0.005 is exact at the precision the old file was written in)

| metric | n | max_abs | max_rel | verdict |
| --- | --- | --- | --- | --- |
| `q_annual_mean` | 36 | 0.004778 | 0.0220 | RECONSTRUCTS |
| `q_baseflow_index` | 36 | 4.807e-05 | 0.348 | RECONSTRUCTS |
| `q_driest_month_mean` | 36 | 0.002329 | 0.232 | RECONSTRUCTS |
| `q_mean_annual_7day_max` | 36 | 0.004958 | 0.00205 | RECONSTRUCTS |
| `q_mean_annual_7day_min` | 36 | 4.782e-05 | 0.465 | RECONSTRUCTS |
| `q_mean_annual_min` | 36 | 4.909e-05 | 0.471 | RECONSTRUCTS |
| `q_return_level_2yr_7day_min` | 36 | 4.764e-05 | 0.476 | RECONSTRUCTS |
| `q_wettest_month_mean` | 36 | 0.9892 | 0.0414 | **moved — pre-registered (Q5)** |
| `q_mean_annual_max` | 36 | 0.07629 | 0.00894 | moved, NOT pre-registered |
| `q_mean_annual_p95` | 36 | 0.03447 | 0.00552 | moved, NOT pre-registered |
| `q_return_level_10yr_max` | 36 | 0.1675 | 0.00990 | moved, NOT pre-registered |

**Eight of eleven reconstruct** — the class-A means average back to the old
aggregated values exactly at 2 dp, which is the pre-registered prediction and the
falsifiable half of the gate. It held.

### The three residuals, and an honest note on them

All three moved **< 1% relative**, all **systematically downward**, and all three
are **upper-tail statistics**: the annual maximum, the 95th percentile, and the
10-year return level (which is fitted to annual maxima, so it inherits whatever
the maxima do).

The explanation that fits is the **same butt-splice defect** P1 fixed. At
`a1d9993` — the commit the manifest was recorded from, i.e. the code that
produced the old numbers — `analyze_wflow_results` does:

    178:  sim_all = pd.concat(csv_rlz)
    179:  sim_all.index = pd.date_range(...)
    185:  df_mean = sim.resample("YE").mean().mean()
    186:  df_max  = sim.resample("YE").max().mean()
    188:  df_q95  = sim.resample("YE").quantile(0.95).mean()

The class-A statistics were computed on the spliced series, whose year boundaries
are a synthetic index rather than each realization's own calendar. A **mean** over
a year is robust to where the cut falls; a **maximum** is not. Which is exactly
the observed split: the mean reconstructs, the maxima and the p95 do not, and they
move down — consistent with removing buckets that straddled two realizations and
so took a maximum over a mixed window.

**Stated plainly: this is a post-hoc explanation, not a pre-registered one.** The
pre-registration named the 7-day return level and the two month metrics, because
the writer's docstring names those. It did not anticipate that the same splice
also reached the upper-tail class-A metrics — that inference came after seeing
which metrics moved. The evidence for it is strong (the old code demonstrably
splices; the affected set is exactly the boundary-sensitive statistics; the
direction is consistent), but it is reasoning toward an explanation rather than a
prediction that survived a test, and it should be read at that weight.

One pre-registration miss in the other direction, worth recording because it is
the kind that flatters the predictor if left unsaid: **`q_driest_month_mean` was
predicted to move and did not** (max_abs 0.0023, inside the 2-dp reconstruction
band). Only one of the Q5 pair actually shifted.

**Gate 2 PASSED** (owner ruling, 2026-08-08).

---

## The re-record — once

`check_baseline.py record`, one invocation, then `check`:

    recorded: 7 target(s) -> dev\baseline\manifest.json (7 total)
    OK - 7 target(s) match manifest.

| target | kind | outcome |
| --- | --- | --- |
| `config/runs/snake_config_model_creation.yml` | yaml | moved — `aggregate_rlz` removed (P1) + `run_historical: true` (P3) |
| `config/runs/snake_config_climate_projections.yml` | yaml | same snapshot, same two keys |
| `experiments/experiment/config/snake_config_climate_experiment.yml` | yaml | same |
| `cmip6_change_factors_annual.csv` | csv | **unchanged** |
| `cmip6_change_factors_monthly.csv` | csv | **unchanged** |
| `models/…/run_default/output.csv` | discharge | **passed the comparator** across both rebuilds |
| `experiments/…/results/q_indicators.csv` | **indicator** (was `csv`) | 756 rows, 66 groups, 7 columns; tolerance comparator per Q8 |
| `experiments/…/results/basin_indicators.csv` | — | **entry removed**; no basin variable in the seed, and the pre-R11 file was 87 bytes with no values |

Two judgement calls taken on documented defaults rather than referred up:

- **The six `png` rows are pruned.** `record` excludes `FIGURE_KINDS` by default
  (`AGENTS.md`) and a full record overwrites, so they are gone. They were already
  unreachable by the default `check`. Restoring figure coverage would need
  `record --include-figures` — i.e. a *second* record, which the brief forbids.
- **`stress_test_design.csv` stays out**, per the P3 ruling. That ruling is
  better supported now than when it was made: P2's justification (`validate_hm7`
  covers it) was false at the time and is true only since `test_hm7_integration`
  started passing `design=`.

### The new comparator, proven on the real artifact rather than assumed

| perturbation of the recorded table | result |
| --- | --- |
| identical | pass |
| one row +50% | **fail**, localised to `q_annual_mean␟101` |
| every row +1e-9 | pass — Q8's entire purpose |
| `st_0` dropped | **structural fail**, 108 rows only-ref |

The last row matters: the baseline itself would now catch the defect this phase
found, independently of `validate_hm7`.

---

## Post-record verification

    pytest tests/ -q -rs   ->  1707 passed, 8 skipped, 1 xfailed  (11:14)
    pixi run tree-check    ->  MAP CLEAN: 221 paths, 0 unmapped

**All five Layer-2 cases un-skipped** — `test_wg2_integration`, `test_hm4_integration`,
`test_hm5_integration`, `test_wg5_catalog_grid_integration`, `test_wg3_integration`
are absent from the skip list, so the fixture genuinely regenerated. Read as the
brief demands: the skip LIST, not the pass count.

### A THIRD defect, found by reading that list

Of the 8 remaining skips, 6 are expected (3 × `temp()` capture, 3 ×
`--run-integration`). One is the known board item `t2608081012`. The eighth was
not known:

    tests/test_store_region_bbox.py:60: needs a completed run under test_case/test_local

It said the artifacts were missing immediately after a completed run that
produced them. `_seed_paths()` pointed at `project_dir / "hydrology_model" /
"staticmaps.nc"` — a **pre-R9 path** — so `exists()` was False forever and the
test **skipped silently from R9 until now, asserting nothing.**

This is the third instance in one phase of the pattern `AGENTS.md` names as the
one that survives every gate a branch can run: a wrong path behind an `exists()`
guard degrades to a silent skip rather than a failure. Fixed to
`models/hydrology/wflow/staticmaps.nc`; the test now runs (20.8 s of real work)
and passes, so the region/grid agreement invariant is restored *and* holds.

Notable that all three of this phase's defects share one shape: **something that
should have been checked was not being checked, and nothing said so.** A metric
family absent from a table, a validator called without its argument, a test
guarded on a stale path. None was a wrong answer; each was a missing question.
