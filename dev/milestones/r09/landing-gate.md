# R9 landing gate — presentation

Date: 2026-08-05. Prepared from `docs/r09-p5-conventions-docs` after P5 closed.
Gate defined in [`project-tree-task-brief.md`](project-tree-task-brief.md) §Gates
item 3: *"PAUSE before merging. Present all five phase reports, three workflow
dry-runs, a full three-workflow run, and the falsifier results named below."*

**Recommendation: do not close this gate yet.** **Six of nine items are
satisfied** — item 9 was run and passed on 2026-08-05, after this presentation
was first written. Three remain, and one of them — the full three-workflow run —
is outstanding for a reason that only surfaced while preparing this document.

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
| 6 | Full three-workflow run on the seed config | **SATISFIED 2026-08-05** — fresh tree, P4-inclusive; two defects found and fixed |
| 7 | `semantic_tree_diff` whole-tree, clean modulo allowlist | **SATISFIED 2026-08-05** — 5 attr-only failures, **0 numeric, 0 structure** |
| 8 | Falsifier: no member's Wflow log is overwritten | passing half re-confirmed; failing half in flight |
| 9 | Falsifier: sharing a dataset+window does not re-run shared work | **SATISFIED 2026-08-05** |
| — | `check_baseline check` | **green 2026-08-05** — 8 targets match, no re-record |

**The run found two real defects, which is the argument for having required it.**
Both were in P4 code that had never executed, and both are fixed:
`from __future__` in all three of P4's `script:` modules (`SyntaxError` at run
time, invisible to import-based tests and to every dry-run), and the drift
guard's verdict persisting between invocations.

---

## Gate closure — items 6 and 7, the fresh run

Run 2026-08-05 from the primary checkout into `test_case/r09_gate/post_p4`,
`-c 3`, three workflows in order. **The two defects it found are the return on
requiring it**, and neither was reachable by anything cheaper.

### Two defects, both in P4 code that had never run

**`from __future__ import annotations` in all three of P4's `script:` modules.**
Snakemake PREPENDS its preamble to a `script:` module, so the future import is no
longer at the top of the file and the job dies with `SyntaxError` before running
a line of our code. WF3 failed on its first job. All three of 3.01c/3.01d/3.01e
carried it; none of the 13 pre-existing `script:` modules does.

Nothing below a real run could have caught it: the unit tests **import** these
modules, where the import is perfectly legal — 28 tests passed against code that
could not execute — and `--dry-run`, `test-cli` and `test-full` never execute a
`script:` body. Fixed in `83e1ac1`, closed mechanically by a test that parses
every Snakefile's `script:` paths plus the variable-resolved `REGION.script`.

**The drift guard detected but never fired.** Forced against a perturbed model
it raises `ModelDriftError` naming the changed file — the logic was right all
along. But its sentinel persisted, so rule 3.09's input edge was satisfied by a
verdict about a *different* model. P4 asserted the ordering structurally and that
test was correct and still passes: **an edge orders A before B; it does not make
A re-evaluate.** Fixed in `8d992ef` by making the sentinel `temp()`, so the
verdict dies with the invocation that produced it.

*Evidence correction (`26109f5`).* The severity was first argued from a job count
taken at `-c 1` against a tree built at `-c 3`. P3-3 derives the batch split from
the core count, so the batch rule instances differ and everything downstream
replans; those jobs were that artifact, not the effect under test. Re-established
on a two-rule probe independent of WF3 scheduling, and confirmed end to end at
matching cores: model perturbed, 3.09 forced, **run stopped at 1 of 34 steps**
with no member simulated.

### P4's artifacts, executing for the first time

`config/model_reference.yml`, `config/experiment.yml` and `.model_reference_ok`
all present. The digest is exactly what the design predicted — 6 inputs, 3
hashed, 3 `<absent>` because `dir_output` is deliberately not applied:

| Path | State |
| --- | --- |
| `wflow_sbm.toml`, `staticmaps.nc`, `forcing/inmaps_historical.nc` | hashed |
| `instate/instates.nc`, `output.csv`, `outstate/outstates.nc` | `<absent>` |

### The tree is the design tree

Six roots and nothing else — `benchmarks/ config/ data/ experiments/ logs/
models/` — and the experiment carries exactly `benchmarks/ climate/ config/
hydrology/ logs/ results/`. **No pre-R9 orphans**, because a fresh
`project_dir` cannot inherit any.

### Whole-tree diff — 0 numeric, 0 structure

`semantic_tree_diff --no-path-map`, fresh run vs `test_case/test_local`:
**156 files compared, 5 failed, 173 missing, 20 extra.**

All five failures are **attribute-only**, in classes declared before the run:

| Failure | Class |
| --- | --- |
| `extract_historical.nc` | one attr differs — `region_source`, which records the region file's path. Two `project_dir`s, two paths. Provenance, not value |
| 3 × `cmip6/raw/*.nc` | **P2's F4** — `variable_id` (`pr` vs `tas`), `tracking_id`, `status`. Two fetches of one slice take global attrs from different members of the merge. Values, coords and dims all match |
| `cmip6/summary/provenance.json` | sha256 — it embeds the F4 attrs above. Same root cause |

**The MISSING set is the orphan inventory, exactly as the plan predicted** — a
nuisance turned into the deliverable. 173 = ~156 pre-R9 orphans in the reference
tree + 17 digest-named config bundles that appear as MISSING/EXTRA pairs because
the snapshot digest includes `project_dir`.

| Orphan subtree | Files |
| --- | ---: |
| `hydrology_model/**` | 43 |
| `climate_projections/cmip6/**` | 32 |
| `experiments/experiment/weather_generator/**` | 25 |
| `experiments/experiment/hydrology_runs/**` | 24 |
| `climate_historical/era5_20000101_20201231/**` | 12 |
| `spatial/**` | 10 |
| `experiments/experiment/indicators/**` | 8 |
| `data_catalog_climate_experiment.yml`, `config/generated/` | 2 |

Every one is a subtree P2 migrated away from, still present in `test_local`
because Snakemake writes the new path and never deletes the old. This is P1's F5
made concrete: no comparator can find these from one tree alone, and two trees of
the same era enumerate them for free.

The 20 EXTRA are the digest-named bundles' other halves plus P4's two new config
files — nothing unaccounted for.

### `check_baseline check` — green

**8 targets match the manifest**, no re-record. Gate 2 stays closed at P3, as the
master brief requires. It ran against `test_case/test_local`, which the fresh run
never touched, so it is a *"nothing drifted on disk"* tripwire rather than a
verdict on the new run — the 156-file semantic diff above is the value gate.

### One correction to the plan itself

The plan told the operator to re-run `--check-map` on the fresh tree and expect
zero unmapped. **That is the wrong direction and it reports 163 unmapped.** The
map translates *old → new*; a tree that is already new has no old paths to
translate. P1's falsifier ran it against the pre-migration inventory, which is
the only direction in which it means anything. Step 0's `git checkout` and step
1's `--configfile` flag were wrong too (see the plan's own corrections).

---

## Gate closure — item 9, the shared-store falsifier

**Run 2026-08-05, primary checkout detached at `2b264d8`, against
`test_case/test_local`. PASSES.** Restored to `main` afterwards; the fixture tree
is unmodified (dry-run only — `experiments/` still holds `experiment` alone).

*"Sharing a dataset and window does not re-run shared work per experiment."*
Two configs identical in every store-determining section — `shared.clim_historical`,
`shared.historical_window`, `shared.basin`, `project.project_dir`,
`project.data_sources` — differing **only** in `experiment_name`
(`experiment` vs `shared_store_probe_b`, the latter never run).

### The three assertions

**1. The second experiment schedules zero store jobs.** Config B plans **51
jobs** — it is a fresh experiment, so every experiment-scoped rule runs — and
`extract_climate_grid` is **not among them**.

**2. The store rule's resolved spec is byte-identical across both configs.**
Compared field by field via `snake_utils.climate_store_spec`:

| Field | Value (identical for A and B) |
| --- | --- |
| `store_dir` | `…/data/climate/historical/era5_20000101_20201231` |
| `outputs.climate_nc` | `…/era5_20000101_20201231/extract_historical.nc` |
| `inputs.region_geojson` | `…/data/spatial/geoms/region.geojson` |
| `inputs.catalog` | `config/catalogs/deltares_data.yml` |
| `params` | region, source, window, hydrography, basin index — all equal |
| `script` | `blueearth_cst/climate_analysis/extract_historical_climate.py` |

**3. The discriminator — the rule is SATISFIED, not ABSENT.** A zero job count
alone is also what you get when the rule never entered the second DAG, which
would be a DAG defect wearing the costume of a pass. Targeting the store file
explicitly under config B:

```
snakemake -s Snakefile_climate_experiment --configfile <exp_b.yml> --dry-run \
    test_case/test_local/data/climate/historical/era5_20000101_20201231/extract_historical.nc

Nothing to be done (all requested files are present and up to date).
```

The rule is in B's DAG, resolves to the same output, and is up to date. That is
reuse.

### Two things the run showed that the falsifier did not ask for

- **The sharing extends past the store to the region.** `delineate_region` is
  also absent from B's DAG. `extract_climate_grid`'s own input is
  `data/spatial/geoms/region.geojson` — the single shared region artifact of ADR
  0003 — so the reuse is two artifacts deep, not one. This is the same file
  whose existence P1's F1a and P5's F5 were both about.
- **51 vs 37 jobs is the design, read correctly.** Config A plans fewer jobs
  than the never-run B because A is a completed experiment being partially
  re-triggered. The interesting number is not the totals but *which* rules B
  skips: only the project-scoped shared ones. Everything experiment-scoped runs,
  which is what makes two experiments independent while their inputs stay shared.

### Why this was run from the primary checkout

The measurement **is** a Snakemake job count, and `AGENTS.md` records that one
`project_dir` driven from two checkouts plans differently — 12 jobs from one, 2
from the other, measured 2026-08-02 — because the "what is up to date" metadata
lives in `.snakemake/` under the *working directory*. Taking this count from a
task worktree would have been convenient and unsound.

The milestone branch is checked out in a worktree, so the primary was detached at
its tip rather than switched to it.

---

## The three outstanding items

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

### 9. The shared-store falsifier — CLOSED 2026-08-05

Results in *Gate closure* above. Kept here because **how it went missing** is
worth more than the fact that it passed.

The master brief named it; **no phase report mentioned it.** Not deferred, not
triaged — it fell between five phases, each of which had its own named scope and
none of which owned it. It was also the cheapest of the four items: a dry-run job
count plus an input comparison, about fifteen minutes end to end.

The lesson generalises past this milestone: **a cross-cutting check listed only
in the master brief belongs to nobody.** Every phase brief carried its own
falsifier and every phase ran it. The two whole-program falsifiers sat one level
up, and one of them was simply never picked up — not by the phase that made the
store shared, nor by any phase after. Item 8's missing half went the same way for
the same reason: P2 ran the half its own brief named and left the half the master
brief named.

Worth fixing structurally in R10: assign each cross-cutting check to a *named
phase*, or the landing gate is the first moment anyone notices.

It guards the reason the store key exists at all — design tree
`data/climate/historical/<key>/` keeps the key as a **cache key**, not as
multi-window support.

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
