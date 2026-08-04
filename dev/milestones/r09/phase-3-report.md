# R9 P3 report — result tables, rule 3.11, and the single baseline re-record

Date: 2026-08-04. Branch: `feat/r09-p3-result-tables`, merged to
`milestone/r09-project-tree`; commit 4 landed directly on the milestone branch
from the primary checkout (see *The re-record's route*). Brief:
[`phase-3-result-tables-task-brief.md`](phase-3-result-tables-task-brief.md).

**Status: complete. Master Gate 2 closed 2026-08-04.** `check_baseline` is green
for the first time since P2 commit 1.

---

## Commits

| # | Subject | Note |
| --- | --- | --- |
| 1 | `r09: fix the basin-table header assertion` | first, so the rename cannot carry a fixture-shaped assertion |
| — | `fix(r09): align the script's params key with the renamed rule param` | my error in commit 2, caught before it ran |
| 2 | `r09: rename the result tables and rule 3.11` | names + every consumer + `LOG_RULES` together |
| 3 | `r09: drop the RT_*.csv side tables` | separately revertible from the rename |
| — | `r09: repoint the baseline targets at the migrated tree` | prerequisite for 4, kept out of it |
| 4 | `r09: re-record the baseline manifest` | **Gate 2**; alone, so it is attributable to nothing else |

## Results delta

**No value changed.** Stated explicitly because the brief warns that an absent
section reads as an unrun check.

| Artifact | Evidence |
| --- | --- |
| `q_indicators.csv` | **byte-identical** to the retained `Qstats.csv` — 3838 bytes, md5 `0979bc91…` both sides; 66×9, exact element-wise |
| `basin_indicators.csv` | **byte-identical** to the retained `basin.csv` — 71 bytes, md5 `4102a99e…` both sides; 6×2, exact element-wise |
| discharge series | **byte-identical** across the whole migration — md5 `8c65ad3e…`; git recorded the reference file's move as a pure rename, zero changed lines |

The brief asked for element-wise identity; all three are stronger than that.

**The ordering is the point.** The tables were compared against artifacts
retained *before any P3 edit*, and the comparison ran *before* the re-record. A
green baseline afterwards proves nothing about identity — the re-record is what
makes it green. The retained copies live at `.cst_runs/r09_pre_p3_tables/` with
their provenance.

## The falsifier

```
=== Qstats.csv -> q_indicators.csv ===       BYTE-IDENTICAL: True   ELEMENT-WISE: IDENTICAL (exact)
=== basin.csv  -> basin_indicators.csv ===   BYTE-IDENTICAL: True   ELEMENT-WISE: IDENTICAL (exact)
VALUE IDENTITY: HOLDS
```

## The manifest diff

**Wider than the brief anticipated**, accepted at the gate. The brief expected a
diff "limited to the two renamed keys"; P2 moved every target, so **11 of 14
keys move** and the count stays 14:

| Class | Keys |
| --- | ---: |
| `hydrology_model/**` → `models/hydrology/wflow/**` | 4 |
| `climate_projections/**` → `data/climate/projections/**` | 5 |
| `indicators/{Qstats,basin}` → `results/{q_indicators,basin_indicators}` | 2 |
| name unchanged, value changed | 2 |
| unchanged entirely | 1 |

### Two keys kept their name and changed value — explained, not noted

The config snapshots for `climate_projections` and `climate_experiment`. A value
change in a pinned artifact is exactly what Gate 2 exists to adjudicate, so it
was chased rather than accepted.

All three config snapshots are copies of the same `--configfile` and are now
byte-identical on disk (md5 `a32d5aab…`, sha256 `48242f48…`).
`model_creation` **already carried that hash**, so the other two were **stale** —
pinning content from an earlier era — and this run refreshed them into
agreement. The correct invariant is that all three agree; before, they did not.

### The discharge reference moved, and nearly went missing

`record` wrote `discharge_ref/9baa48f90ceaf138.csv` and orphaned
`98aacecbe4a5f235.csv`: the filename is a hash of the **target path**, which
moved. **Committing only `manifest.json` would have left `check` failing on a
fresh clone** for a missing reference series. Caught by reading `git status`
after the commit rather than trusting it.

The two files are byte-identical, so no content is lost and the rename is itself
evidence that the discharge series survived the migration unchanged.

## `--include-figures` was passed deliberately

The manifest pins 6 png targets. `record` **excludes figures by default**, so
recording without the flag would have silently dropped them — shrinking the
manifest from 14 to 8 while printing success. Checked before running, not after.

## The re-record's route

The manifest keys are resolved paths embedding `test_case/test_local`, with a
matching top-level `project_dir`. Recording from the `.cst_runs` tree that P2 and
P3 were validated against would have written every key under *that* path — a
manifest no future `check` reproduces, and a break with the rule that the
baseline comes from the tracked seed config.

So, on an owner ruling: P3 was merged to the milestone branch, that branch was
checked out in the **primary checkout**, the three workflows were run into
`test_case/test_local` (17/17, 23/23, 47/47), and the record was taken there.
Two temporary state changes — the primary off `main`, and the milestone
worktree detached to free the branch — were both **restored afterwards**.

## Acceptance criteria

| Criterion | Result |
| --- | --- |
| Element-wise identity against the retained pre-P3 tables | **byte-identical**, stronger than required |
| No `RT_*.csv` produced anywhere in a full run | **none produced.** Six exist on disk; `find -newermt` returns zero newer than the run — leftovers Snakemake does not delete. Produced ≠ present |
| `validate_hm7` passes under both `wflow_outvars` shapes | yes — the fixture's and the shipped template default's |
| Baseline re-recorded exactly once, `check` green after | yes; 14 targets match |
| Manifest diff limited to the two renamed keys | **NO** — 11 keys move; deviation accepted at the gate |
| Exactly one rule identifier changed | yes — 3.11 only |

## `LOG_RULES` verified the way the brief demands

Not by counting files, but by reading the merged log after a full WF3 run:

```
4622:== 3.11  derive_wflow_indicators
```

An unlisted label is silent, not an error — the defect P1 found sitting on `main`
for months (phase-1 report F7). All four sites were checked consistent at commit
time: the `LOG_RULES` entry, the `rule_banner`, the `log:` part and the
`benchmark:` part.

## Findings

### F1 — a producer/consumer pair broken by a blanket rename — **my error, third occurrence**

Commit 2 renamed the Snakefile binding `indicators_dir` → `results_dir`, which
also renamed the rule's **params key**, while the script kept reading
`sm.params.indicators_dir`. An `AttributeError` the moment rule 3.11 executes;
`pixi run test-cli` passed on the broken pair because a dry-run never runs it.

Caught while reading the `RT_*.csv` code, before it could run.

**This is the third time in this milestone** a rename touched one side of a
producer/consumer pair and not the other (P2 F1, P2 F2, this). The common cause
is a blanket string replace across *one* file when the contract's other side
lives in another. Named here because the pattern, not the instance, is the
lesson.

### F2 — `returnintervalmulti` vs `returninterval`: nearly deleted the wrong one

`q_indicators.csv` keeps `returninterval` and `returninterval_Q7d` rows, fed by
`md.returninterval()` and `md.returninterval_Q7d()`. `RT_*.csv` was fed by
`md.returnintervalmulti()` — a **different function** with a confusingly close
name. Verified independent before removing; deleting the wrong one would have
silently emptied two rows of the response surface with nothing failing.

## Validation

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | `test_export_wflow_results`, `test_interchange_contracts`, `test_check_baseline*` | 70 passed, 26 skipped |
| 2 Integration | `pixi run test-cli` | 12 passed |
| 3 Phase gate | `pixi run test-fast` | **1221 passed**, 30 skipped, 42 deselected, 1 xfailed (94 s) |
| 4 Non-regression | `check_baseline.py check --include-figures` | **OK — 14 targets match** |

Full runs: WF3 with P3 code into `.cst_runs` (40/40), and the three workflows
into `test_case/test_local` from the primary (17/17, 23/23, 47/47).

## Carried forward

- The P2 concurrency falsifier has still not been shown to **fail** with
  `path_log` unset.
- **F4/F5/F6 from P2**: WF2's nondeterministic fetch provenance, rule 1.04's
  undeclared write to `staticmaps.nc`, and `AGENTS.md`'s incorrect shared-env
  claim.
- **P5's queue**: `AGENTS.md`'s stale DAG-render path, the same shared-env claim,
  and the design tree's silence on `spatial/geoms/region.geojson`.
