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

### WF3 — `Snakefile_climate_experiment`

*(In progress. The experiment name is pinned — `experiment_name: experiment` —
so the tree rebuilds at `experiments/experiment/` and the manifest paths hold. A
minted date-based name would have silently broken every recorded path.)*
