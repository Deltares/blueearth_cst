# R9 closing record — the generated project tree

Date: 2026-08-05. Branch: `milestone/r09-project-tree`, all five phases merged.
Master brief: [`project-tree-task-brief.md`](project-tree-task-brief.md).
Gate evidence: [`landing-gate.md`](landing-gate.md).

**Landing gate: nine of nine.** The gate found **four defects**, and the useful
fact about all four is that `pixi run test-full` was green over every one of
them.

---

## What R9 delivered

`project_dir` now has six roots — `config/ data/ models/ experiments/ logs/
benchmarks/` — and every artifact sits at the scope of the producer that wrote
it. Verified on a fresh run: those six and nothing else.

| Phase | Delivered |
| --- | --- |
| **P1** | The comparator: a 59-rule path map, `--check-map`, orphan-store pruning, a snapshot wrapper, and two inventory tiers. Both tiers zero-unmapped (declared 176, observed 192) |
| **P2** | The move: `models/hydrology/wflow/`, `data/{spatial,climate}/`, the experiment's symmetric `climate/` + `hydrology/` pair, and the member-path flattening that put both indices back in the filename |
| **P3** | `results/{q_indicators,basin_indicators}.csv`, `RT_*.csv` dropped, rule 3.11 renamed `derive_wflow_indicators`. Value identity proven **byte-identical before** the single allowed baseline re-record |
| **P4** | A pointer-derived model digest, `model_reference.yml` per experiment, a drift guard ordered before simulation, atomic experiment reservation, and `experiment.yml` frozen at first successful run |
| **P5** | `naming.md` §4/§6/§7/§8/§9, `AGENTS.md`, `README.rst`, both seam contracts; WF1's contract migrated and WF3's sealed |

## The four defects, and why the cheap gates missed them

This is the part worth carrying into R10.

### 1. `from __future__` in all three of P4's `script:` modules

Snakemake **prepends** its preamble to a `script:` module, so the future import
is no longer at the top of the file and the job dies with `SyntaxError` before
running a line of our code. WF3 failed on its first job.

*Why nothing cheaper saw it:* the unit tests **import** these modules, where the
import is perfectly legal — **28 tests passed against code that could not
execute.** `--dry-run`, `test-cli` and `test-full` build the DAG and never
execute a `script:` body.

### 2. The drift guard detected but never fired

Its sentinel persisted, so rule 3.09's input edge was satisfied by a verdict
about a *different* model.

*Why nothing cheaper saw it:* P4 asserted the ordering structurally, by parsing
the Snakefile for 3.09's declared input. That test was correct and still passes.
**An edge orders A before B; it does not make A re-evaluate.** Both P4's tests
and mine checked that the guard runs *before* the work; none checked that it
runs *at all*.

### 3. The log falsifier skipped itself in the condition it exists to catch

Discharging the half owed since P2 — showing it FAILS with `path_log` unset —
produced twelve members, zero per-member logs, and one shared `log.txt`. With no
per-member logs, `_member_logs` hit `if not logs: pytest.skip(...)` and **both
attribution tests skipped.** The only assertion that fired was the one the
module itself calls *"the weakest of the three."*

*Also corrected:* the failure mode is **overwriting, not interleaving**. Each
wflow process opens the default path and truncates it, so eleven members' logs
were destroyed rather than merged. The module had been looking for a file
carrying two names, which never appears.

### 4. The shared-store falsifier was never run at all

Named in the master brief, in no phase report. It cost about fifteen minutes and
it passes.

*Why:* it was a **cross-cutting** check, listed one level above every phase
brief. Each phase ran the falsifiers *its own* brief named. Item 8's missing
half went the same way — P2 ran the half its brief named and left the half the
master brief named.

**For R10: assign every cross-cutting check to a named phase, or the landing
gate is the first moment anyone notices.**

## What each instrument was actually good for

R9 ran the full ladder repeatedly. Measured, not assumed:

| Instrument | Caught |
| --- | --- |
| Unit tests (narrow scope) | test-path expectations. **Nothing structural** |
| `pixi run test-cli` (×5 in P2) | **nothing** — every real defect was invisible to it |
| `pixi run test-fast` | nothing in P2; in P5 it could not even *collect* the module that was broken (`workflow_contract` is excluded by definition) |
| `pixi run test-full` | one P4 escape — a fixture that had fallen behind a new declared input |
| **A real three-workflow run** | **all four defects above, plus P2's three** |
| `semantic_tree_diff` whole-tree | a map gap in P2; the orphan inventory at the gate |
| The grep falsifier (P5) | ten stale seam-contract passages a passing 37-test module could not see |

The pattern: **every defect this milestone introduced or surfaced was a path or
a behaviour resolved at RUN time.** The DAG resolves, the dry-run is clean, the
unit tests pass, and the job dies on execution — or worse, succeeds while a
guard sleeps.

## The recurring authoring failure

Four times in one milestone, a sweep was declared complete on the strength of
the edits made rather than on a re-run of the instrument that found the work
(P5 F1, tabulated there). Twice in P5's own commits. Two further variants
appeared at the gate: a job count taken with a parameter varied that I was not
thinking about (`-c`, which P3-3 uses to derive the batch split), and three
wrong commands in a run plan I wrote and then executed myself.

The mechanical fix is one line of process: **re-run the finder before writing
the claim.**

## Evidence at the gate

- Fresh three-workflow run, `-c 3`, into a clean `project_dir`.
- Whole-tree diff: 156 files compared, **0 numeric, 0 structure**; 5 attr-only
  failures, all in classes declared before the run.
- The MISSING set doubles as the **orphan inventory** for the old fixture tree
  (~156 files across seven pre-R9 subtrees) — P1's F5 made concrete.
- `check_baseline check`: 8 targets match, **no re-record**; Gate 2 stays closed
  at P3.
- `pixi run test-full`: 1312 passed.
- Drift guard shown **firing**: run stopped at 1 of 34 steps, no member
  simulated.
- Log falsifier shown **failing** without `path_log`, and both branches of its
  repaired skip logic demonstrated on synthetic trees.

## Carried forward

Not blockers; each has a home.

| Item | Where |
| --- | --- |
| ~~`provenance/runs/` is a seventh project root no inventory tier can see~~ | **DONE 2026-08-05** — moved to `config/runs/invocations/`; map row F3, design-tree line, and the inventory's blind spot documented. [`followup-provenance-root-task-brief.md`](followup-provenance-root-task-brief.md) |
| ~~Pre-R9 paths in `blueearth_cst/**` prose, a Snakefile comment block, and four `dev/scripts/` files~~ | **DONE 2026-08-05** — all three classes; the scaffold was shown failing before the fix, and three further defects surfaced in the same files. [`followup-stale-path-prose-task-brief.md`](followup-stale-path-prose-task-brief.md) |
| **The batch split is core-derived**, so re-running at a different `-c` re-does all Wflow work and invalidates any job-count comparison | P3-3 design territory; recorded in `26109f5` |
| WF2's nondeterministic fetch provenance (P2 F4) — 4 of the gate's 5 diff failures | P2 report |
| Rule 1.04's undeclared write to `staticmaps.nc` (P2 F5's root cause) | P2 report |
| The seal convention has one banner and no enforcement | P5 F2 |
| Three copies of cross-workflow staging logic, one stale | P5 F3 |
| `docs/notebooks/*.ipynb` carry R7-era `examples/` paths | P5, reported not fixed |

## Recommendation

**Close the gate and merge `milestone/r09-project-tree`.** All nine items are
satisfied on evidence, the tree a fresh run produces is the design's tree, no
scientific value moved, and the four defects the gate existed to find are fixed
with their classes closed mechanically rather than by hand.

## The fixture tree was swept before merging

Owner-ruled 2026-08-05. `test_case/test_local` had been mixed-era since P2 —
Snakemake writes the new path and never deletes the old — and the gate's MISSING
list was its inventory, so the sweep was derived rather than guessed.

**160 files removed**, listed in
[`orphan-sweep-2026-08-05.txt`](orphan-sweep-2026-08-05.txt): the eight pre-R9
subtrees (`hydrology_model/ spatial/ climate_historical/ climate_projections/
config/generated/` and the experiment's `weather_generator/ hydrology_runs/
indicators/`) plus the loose `data_catalog_climate_experiment.yml`.

That is 156 the diff reported **plus 4 the comparator excludes by design**
(`.log` / `log.txt`) which sit inside orphan subtrees and are therefore orphans
too. Two of them — `hydrology_runs/rlz_{1,2}/config/log.txt` — are the pre-R9
shared logs the concurrency falsifier exists to prevent: two files for twelve
members, the very artefact of the defect, still on disk.

**Retained deliberately:** the 17 digest-named `config/runs/**` bundles. They
appear in the MISSING set only because the snapshot digest includes
`project_dir`, so the two trees name the same artifact differently. They are
this tree's own current output, not leftovers.

Checked before deleting, not after: none of the 14 baseline manifest targets
lies under any swept path, and no test reads one. Checked after: `check_baseline
check` still reports **8 targets matching**, and the tree now holds exactly the
six R9 roots with the experiment holding exactly six subdirectories — identical
in shape to the fresh gate run.
