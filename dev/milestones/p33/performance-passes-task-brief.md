# Task Brief — P3-3 Performance passes (implementation)

> **Handoff from the ACCEPTED design.** The authoritative, load-bearing spec
> is `dev/milestones/p33/performance-passes-design.md` (ACCEPTED 2026-07-24, v4,
> arbitration-closed). Read it in full before touching anything — this brief
> bounds and sequences; the design owns every mechanism, formula, criterion,
> and gate definition. Where the two differ, the design wins. Audit trail:
> `dev/milestones/p33/performance-passes-design-review-record.md`; scoping anchors:
> `dev/milestones/p33/performance-passes-intake.md`; probe evidence `dev/milestones/p33/probes/`.

### Context

- **Canonical ruleset:** `AGENTS.md`. CST scope: no Wflow.jl/hydromt/
  weathergenr internals; the levers touch OUR invocation layer only.
- **Value-identity is absolute (C2):** every pipeline output byte-identical;
  entry points / `rule all` targets / wrapper contract unchanged; P3-2b
  validators green; the §8 discriminating rule — any diff correlated with
  batching (present batched, absent in a per-process re-run on identical
  inputs) BLOCKS commit 2; ADR-0001 triage only after a per-process re-run
  reproduces the diff.
- **Key design pins — implement exactly:** rule 3.10 restructure to
  parse-time **loop-generated anonymous rules** (`rule:` + `name:
  f"run_wflow_batch_{b}"`, STATIC per-batch output lists, members via
  `params:`, no input function, no checkpoint — the probe-verified
  construct, `dev/milestones/p33/probes/snakemake-output-expressibility/`); the
  `run_wflow_batch.jl` per-TOML try/catch driver; the §5.6 CPU resource
  contract (`N × t ≤ C_logical`; commit-2 before/after at the IDENTICAL
  `(-c N, --threads t)` triple as the commit-1 baseline — B the only moved
  knob); `dev/scripts/estimate_batch_makespan.py` (LPT + Graham bounds,
  remainder batches); GN-1..4 evaluated on commit-2 evidence (GN-1 ≥15%
  lever threshold — distinct from the floor-free milestone gate); C5 is
  DEGRADED (blast radius B) — documented, not hidden.
- **Milestone mechanics:** task branch `task/p33-performance` off main;
  prefix `p33:`; merge + milestone branch/tag `p33-performance` at close.
- **Machine-quiet requirement:** every measured wall number (commit-1
  baseline, commit-2 before/after) is valid only on an otherwise-idle
  machine; record the §5.6 triple + machine state with each measurement.

### Goal

Land the accepted P3-3 design: a recorded wf3 performance baseline +
decomposition + scaling model (commit 1, user-gated), the batching lever
under strict value-identity with the GN-1..4 adjudication (commit 2), and
the seal (commit 3) — closing the last planned roadmap milestone.

### Non-goals

- No sysimage/PackageCompiler work (doubly-gated conditional follow-up —
  only on a GN failure AND a fresh user approval).
- No file-format redesign; no wf1/wf2 changes; no weathergen R edits; no
  memory-headroom work; no upstream re-engineering.
- No manifest re-record (outputs are byte-identical or commit 2 is blocked).

### Allowed scope

**Permitted:** `Snakefile_climate_experiment` (rule 3.10 restructure + the
batch-size config knob per the design); a new Julia driver script (e.g.
`blueearth_cst/model/run_wflow_batch.jl` — follow the design/naming.md);
`dev/scripts/estimate_batch_makespan.py` (new); `dev/milestones/p33/**` (baseline
note, evidence); `dev/roadmap.md` (status); `config/workflows/*` ONLY if
the design's batch-size knob requires a config key (mirror the get_config
contract; default must reproduce today's behavior B=1... the design owns
the knob's shape — verify before adding).

**Approval-gated:** anything touching `Snakefile_model_creation` /
`Snakefile_climate_projections`; `scripts/run_workflows.py`; any config
schema change beyond the designed knob; PAUSE and raise.

**Forbidden:** `blueearth_cst/weathergen/*.R`; vendored packages;
`pixi.lock`/`Manifest.toml`; `examples/test_local` by hand (runs write to
it; hands don't); `dev/scripts/stage_data.*` and `tests/*stage*` (another
workstream's files); the manifest.

### Required changes (checklist)

Design §9, verbatim — each commit suite-green + three dry-runs clean:

1. `p33: record wf3 performance baseline + cost decomposition` — land
   `dev/scripts/estimate_batch_makespan.py`; run the commit-1 measurements
   (end-to-end wf3 sweep wall at the stated §5.6 triple, B=1 today-path;
   re-measure the §5.2 single-sample numbers per the measurement spec);
   write the durable baseline note under `dev/milestones/p33/` (à la baseline_diffs:
   measured walls, decomposition, per-B LPT table, gap attribution,
   measurement spec, machine state). **GATE 1 (human): user reviews the
   baseline numbers before commit 2 proceeds.**
2. `p33: batching N Wflow runs per Julia session` — the driver + the
   loop-generated batch rules + the knob; the arbitration-mandated dry-run
   demonstration (every per-cst target resolves to its `run_wflow_batch_<b>`
   rule) recorded BEFORE the full sweep; the measured before/after at the
   identical triple; the full §8 value-identity gate (semantic diff exact,
   check_baseline wf3 slice, P3-2b validators, the discriminating
   per-process-vs-batched diff); the GN-3 disk observation and GN-4 failure
   injection + clean re-convergence. Adjudicate GN-1..4 and record the
   outcome. Any GN failure → PAUSE (the sysimage ask is the user's).
3. `p33: docs + roadmap seal` — roadmap P3-3 → sealed with the before/after
   headline; baseline note finalized. **GATE 2 (human, milestone): user
   signs off on the measured before/after + GN outcome + value-identity
   evidence (floor-free) before merge/tag.**

### Validation

Design §10 verbatim. Per commit: full `pixi run pytest tests/` + three
`--dry-run`s. Commit 2 adds: the dry-run demonstration; before/after at the
fixed budget; `dev/scripts/semantic_tree_diff.py --ref <pre-tree copy>
--cur examples/test_local` exact/clean; `check_baseline.py check --workflow
climate_experiment` OK; `pytest -rs tests/test_interchange_contracts.py`
split unchanged; GN-4 injection evidence. All measurements on a quiet
machine, triple recorded.

### Acceptance criteria

- Three commits landed; construct matches the probe shape; per-cst output
  paths byte-identical; GN outcome recorded with evidence; both human gates
  passed; rollback = any GN-2 failure (batching-correlated diff) → commit 2
  blocked, do not merge, surface it (the design's own decision rule).

### Output requirements

- Commits on `task/p33-performance`; merged after Gate 2; milestone branch
  + tag `p33-performance`. **Results delta:** the measured before/after
  wall (the milestone's deliverable number) + "no output value changed"
  evidence.

### Task constraints

- §9 sequencing binding (baseline gate before the lever; demonstration
  before the sweep). Preserve house patterns (tee/log/benchmark,
  `workflow.configfiles[0]` forwarding, get_config contract).
- Another workstream (stage_data) shares this repo — explicit pathspecs on
  every commit; never stage its files.
