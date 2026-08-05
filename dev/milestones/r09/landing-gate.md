# R9 landing gate — presentation

Date: 2026-08-05. Prepared from `docs/r09-p5-conventions-docs` after P5 closed.
Gate defined in [`project-tree-task-brief.md`](project-tree-task-brief.md) §Gates
item 3: *"PAUSE before merging. Present all five phase reports, three workflow
dry-runs, a full three-workflow run, and the falsifier results named below."*

**Recommendation: do not close this gate yet.** Five of nine items are
satisfied. Four are not, and one of them — the full three-workflow run — is
outstanding for a reason that only surfaced while preparing this presentation.

---

## The five phases

| Phase | Delivered | Gate outcome |
| --- | --- | --- |
| **P1** — comparator and tooling ([report](phase-1-report.md)) | 59-rule path map, orphan-store pruner, snapshot wrapper, two inventory tiers | **Gate 1 CLOSED.** Declared tier 176 paths → 3 unmapped → 0 after rulings; observed tier 192 → 0. Eight findings |
| **P2** — tree migration ([report](phase-2-report.md)) | six project roots; `models/hydrology/wflow/`, `data/{spatial,climate}/`, the experiment's `climate/`+`hydrology/` pair, member-path flattening | Whole-tree gate **zero numeric and zero structure failures** over 161 files; every residual adjudicated in a written allowlist |
| **P3** — result tables and rule 3.11 ([report](phase-3-report.md)) | `results/{q_indicators,basin_indicators}.csv`; `RT_*.csv` dropped; rule renamed `derive_wflow_indicators` | **Gate 2 CLOSED.** Both tables **byte-identical** to the retained pre-P3 artifacts *before* the re-record — the only ordering that proves anything. Baseline re-recorded exactly once |
| **P4** — fingerprint and lifecycle ([report](phase-4-report.md)) | pointer-derived model digest, `model_reference.yml`, a drift guard ordered *before* simulation, atomic experiment reservation, `experiment.yml` frozen at first successful run | 13 falsifiers pass, including the fixed-file-list rejection shown blind to the same edit, and reservation atomicity **demonstrated by racing** eight threads |
| **P5** — conventions and docs ([report](phase-5-report.md)) | `naming.md` §4/§6/§7/§8/§9; `AGENTS.md`; `README.rst`; both seam contracts; WF1's contract migrated, WF3's sealed | **`pixi run test-full` green — 1312 passed.** Five findings |

### The three findings that outlive the milestone

- **P1 F5 / P2** — identity- and prefix-mapped directories **hide orphans from
  the comparator**. The map proves every path is *accounted for*; it cannot see a
  file that should no longer exist. mtime is the instrument the comparator is
  not. This is why item 7 below is outstanding.
- **P4 F2 / P1 F7** — an unlisted `LOG_RULES` label drops its log section and
  strands its part **silently**. Found once in WF1, then shown to be a class
  across all three workflows. Now closed mechanically by a test that parses every
  Snakefile — with its scope stated rather than overclaimed.
- **P5 F1** — four times in one milestone, a sweep was declared complete on the
  edits made rather than on a re-run of the instrument that found the work.
  Twice mine in P5's own commits.

---

## Gate checklist

| # | Item | Status |
| --- | --- | --- |
| 1 | Five phase reports | **satisfied** — linked above |
| 2 | Three workflow dry-runs | **satisfied** — `pixi run test-cli`, 12 passed at P5 HEAD, 2026-08-05 |
| 3 | Full suite, once, before merging | **satisfied** — `pixi run test-full`, **1312 passed**, 31 skipped, 1 xfailed, 4m26s |
| 4 | Gate 1 — map reproduces the design tree | **closed at P1**, both tiers zero-unmapped |
| 5 | Gate 2 — scientific delta | **closed at P3**, value identity proven byte-identical before the re-record |
| 6 | **Full three-workflow run on the seed config** | **OUTSTANDING — predates P4** |
| 7 | **`semantic_tree_diff` whole-tree, clean modulo allowlist** | **OUTSTANDING — the fixture tree is mixed-era** |
| 8 | **Falsifier: no member's Wflow log is overwritten** | **HALF DONE** — passes; never shown to fail |
| 9 | **Falsifier: sharing a dataset+window does not re-run shared work** | **NOT RUN — appears in no phase report** |
| — | `check_baseline check` | green at P3; **not re-verified since**, and P4/P5 touch no pinned artifact |

---

## The four outstanding items

### 6. The full three-workflow run predates P4

The seed tree at `test_case/test_local` carries a complete run from
**2026-08-04 23:01–23:13** (`logs/wf1_model_creation.log` 23:01,
`logs/wf2_climate_projections.log` 23:06,
`experiments/experiment/logs/wf3_climate_experiment.log` 23:13).

That run is **P3-era**. Evidence, not inference: `experiments/experiment/config/`
contains `snake_config_climate_experiment.yml`, `runs/` and `catalogs/` — and
**no `model_reference.yml` and no `experiment.yml`**. Those are the outputs of
P4's rules 3.01c and 3.01e. They are absent because those rules did not exist
when the run happened.

So three of P4's deliverables have **never executed in a real run**:

- rule 3.01c `write_model_reference` — the digest over the model's pointer-derived file set;
- rule 3.01d `check_model_reference` — the drift guard, whose whole value is the ordering edge into rule 3.09;
- rule 3.01e `write_experiment_config` — and its freeze at first successful run, whose marker *is* the merged WF3 log.

P4's own report carries this as its first line of *Carried forward* — "the
end-to-end drift falsifier in a real run, not just at unit level" — so it is a
known gap, not a discovery. What is new here is that it also makes the **landing
gate's** run item unsatisfied, because the gate exists precisely to exercise what
`--dry-run` structurally cannot see.

### 7. The tree the comparator would run against is mixed-era

`test_case/test_local` currently holds **both** eras side by side:

| Pre-R9, still present | R9, present |
| --- | --- |
| `hydrology_model/`, `spatial/`, `climate_historical/`, `climate_projections/` | `models/`, `data/` |
| `experiments/experiment/{hydrology_runs,indicators,weather_generator}/`, `…/data_catalog_climate_experiment.yml` | `…/{hydrology,results,climate}/`, `…/config/catalogs/` |

This is expected and is **exactly P1's F5**: Snakemake writes the new path and
never deletes the old one, and the comparator cannot flag what it maps by
identity or prefix. It is not evidence of a migration defect — but a whole-tree
diff against this tree compares an era to itself plus orphans.

Two ways to satisfy the item, and they answer different questions:

- **Fresh `project_dir`** — a clean three-workflow run into an empty directory,
  then diff. Answers *"does R9 produce the design's tree?"* Also satisfies item 6
  in the same run.
- **Sweep, then diff in place** — `dev/scripts/prune_climate_store.py` for the
  store, plus an mtime sweep for the rest, then diff. Answers *"is the existing
  tree clean?"* — a different and weaker claim.

The fresh run is the one the gate asks for.

### 8. The concurrency falsifier passes but has never been falsified

`tests/test_wflow_log_attribution.py`, green against the post-P2 run: **12 member
logs, 12 correctly attributed, 0 stray `log.txt`**, with a third test guarding
against a vacuous pass on an empty log. Counting files is not the test, and the
module says so.

The missing half is the brief's: **show it FAILS with `path_log` unset.** Until
that is demonstrated, the test is consistent with a workflow that would pass it
either way. P1's observed tier is partial evidence — six members sharing one
`log.txt` — but partial is the accurate word.

### 9. The shared-store falsifier was never run

*"Sharing a dataset and window does not re-run shared work per experiment."* The
master brief names it; **no phase report mentions it.** Not deferred, not
triaged — it fell between five phases, each of which had its own named scope, and
none of which owned it.

It needs two experiments sharing `clim_historical` + `historical_window`, with
the assertion that the shared store rule's input set is byte-identical for each
and that the second schedules **zero** store jobs. That is a `--dry-run` job-count
check plus an input comparison, so it is the cheapest of the four items — and it
guards the reason the store key exists at all (design tree §`data/climate/historical/`:
a cache key, not multi-window support).

---

## What closing the gate would take

One run satisfies items 6, 7 and 8, and item 9 is cheap and independent.

1. **From the PRIMARY checkout** (`AGENTS.md`: Snakemake's `.snakemake/` metadata
   and its workdir lock make a task worktree wrong for this) on
   `milestone/r09-project-tree` with P5 merged — it is currently on `main` at
   `c9990a5`.
2. **Into a fresh `project_dir`**, so items 6 and 7 are answered by the same run
   and the mixed-era tree is not in the way. The existing tree stays untouched as
   the comparand.
3. Then: `semantic_tree_diff` whole-tree; `check_baseline check` (expected green
   — nothing since P3 touches a pinned artifact, so a red here is itself a
   finding); the log-attribution falsifier **plus** its failing half with
   `path_log` unset; and the shared-store falsifier as a second experiment on
   the same store.

**Do not re-record the baseline.** Gate 2 closed at P3 and the brief says record
exactly once.

The alternative — closing the gate on P1–P3 evidence and carrying P4's run into a
follow-up — is a real option, but it means merging three rules that have never
executed, one of which is a guard whose only value is that it fires at the right
moment in a real DAG.
