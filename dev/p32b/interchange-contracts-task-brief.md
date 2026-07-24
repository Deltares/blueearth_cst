# Task Brief — P3-2b Model-swap interchange contracts (implementation)

> **Handoff from the ACCEPTED design.** The authoritative, load-bearing spec is
> `dev/p32b/interchange-contracts-design.md` (ACCEPTED 2026-07-24, v3,
> external round 2 approve with zero findings). Read it in full before
> writing anything — this brief bounds and sequences the work; the design
> owns every contract row, validator signature, counting-axis number, and
> gate definition. Where the two differ, the design wins. Audit trail:
> `dev/p32b/interchange-contracts-design-review-record.md`; scoping anchors:
> `dev/p32b/climate-interchange-intake.md` (four confirmed decisions, fixed).

### Context

- **Canonical ruleset:** `AGENTS.md`. Governing constraint: CST automation
  scope — contracts pin OUR consumed subset of upstream formats
  (hydromt/wflow/weathergenr); validators never assert upstream internals.
- **Zero behavior change, absolutely:** no Snakefile/pipeline/module-runtime
  edit, no output change, no manifest re-record, no fixture modification.
  `git diff` over the milestone touches ONLY `dev/contracts/**`,
  `blueearth_cst/shared/interchange_contracts.py`,
  `tests/test_interchange_contracts.py` (+ this brief/roadmap/dev notes).
- **Key design pins — implement exactly:** pure `-> list[str]` divergence
  reports over PARSED objects (never paths; `assert`/`AssertionError` banned
  in validator bodies — §6.5); asserted-if-present semantics for HM-2 units;
  two relational validators (`validate_hm_gauge_column_identity` — the
  rule-3.11 `Q_`-prefix/first-file mechanics, C3 boundary: the numeric id's
  derivation stays wflow-owned; `validate_wg5_catalog_grid` — rlz 1..N ×
  cst 0..M incl. cst_0, intent from the experiment's recorded config snapshot
  via `stress_test_grid`); two-layer test model (§5.5): 30 synthetic
  pass/fail tests always-run + 15 integration cases with the
  `_FIXTURE_ABSENT` named-skip constant mirroring
  `tests/test_extract_climate_wf1.py`; temp cases additionally
  skip-until-captured; HM-6a is a doc row with NO validator (risk-1).
- **Counting axis (§5.5, single authoritative statement — reference, never
  re-derive):** 15 validators / 30 synthetic tests; fixture present → 12
  integration green + 3 temp skips; fixtureless → 15 named skips, suite
  still green, synthetic layer still executes.
- **Milestone mechanics:** task branch `task/p32b-interchange-contracts` off
  main; prefix `p32b:`; merge + milestone branch/tag
  `p32b-interchange-contracts` at close per `branch-model`.

### Goal

Land the accepted P3-2b design: two seam contract docs under
`dev/contracts/`, the unwired 15-validator module + two-layer test file, and
the close-out index — pinning both substitution seams as checkable contracts
with zero pipeline behavior change.

### Non-goals

- No PoC swap; no in-pipeline enforcement; no new dependencies.
- No `--notemp` capture run (documented only; it would modify the fixture).
- No edits to Snakefiles, `blueearth_cst/**` runtime modules,
  `blueearth_cst/weathergen/*.R`, `examples/test_local`, or the manifest.
- No re-specification of upstream schemas (staticmaps full schema, wflow
  physics, weathergenr internals).

### Allowed scope

**Permitted:** `dev/contracts/**` (new); `blueearth_cst/shared/interchange_contracts.py`
(new); `tests/test_interchange_contracts.py` (new); `dev/p32b/**`;
`dev/roadmap.md` (status only).

**Approval-gated:** anything else — if a contract cannot be pinned without
touching another file, PAUSE and raise it (the design says this cannot
happen; finding otherwise is a design defect to surface, not work around).

**Forbidden:** all pipeline/runtime files; fixtures; manifest; vendored
packages; `pixi.lock`/`Manifest.toml`.

### Required changes (checklist)

The design §8 commit plan, verbatim — one `p32b:` commit each; every commit
full-suite-green + three dry-runs clean:

1. `dev/contracts/weather-generator-seam.md` — WG-1..WG-6 inventory per the
   §5.4 table schema + §5.6 walkthrough + validator index (incl.
   considered-and-excluded notes: sim_dates/resampled_dates,
   wf1_raw/extract_historical.nc; chirps facts marked not-fixture-verified).
2. `dev/contracts/hydrological-model-seam.md` — HM-1..HM-5, HM-6a/HM-6b
   split, HM-7 + walkthrough + index (HM-6a path derivation; HM-4
   rewrite-field set incl. time.timestepsecs + wf3 "standard"-calendar note;
   the HM-4→HM-5→HM-7 column-identity invariant; heterogeneous unit/units
   attr-key facts).
3. `blueearth_cst/shared/interchange_contracts.py` (10 persisted + 2
   relational validators) + `tests/test_interchange_contracts.py` (24
   synthetic tests at this commit + 12 integration cases with
   `_FIXTURE_ABSENT` skipif). Gate adds: `pytest -rs
   tests/test_interchange_contracts.py` shows the 12-green (+ named skips
   fixtureless) split; dry-run DAG unchanged.
4. WG-4/WG-6/HM-6b temp validators (+ their synthetic pairs → 30 total;
   real-artifact cases doubly skip-guarded) + `--notemp` capture procedure
   documented in both seam docs' validator indexes. Gate: 3 temp cases skip
   with the documented reason; all 30 synthetic pass.
5. `dev/contracts/README.md` + roadmap close-out note, coverage stated by
   REFERENCE to the §5.5 counting axis.

### Validation

Design §9 verbatim. Ladder: (1) per-commit — full `pixi run pytest tests/` +
three `--dry-run`s (all clean; no re-record, no semantic diff needed);
(2) validator suite — `pixi run pytest -rs tests/test_interchange_contracts.py`
with the green/skip/synthetic split read against the counting axis;
(3) relational fail-paths — the synthetic fail cases break one member of
each correlated set (renamed Qstats gauge column; dropped catalog key);
(4) fixtureless proof — run the suite once with the fixture path temporarily
renamed... NO: never touch the fixture; instead prove fixtureless behavior
by running the test module with a monkeypatched/absent path only if cheaply
possible, else verify by code inspection of the skipif guards + reason
constants (state which was done); (5) scope audit (C3) — every validator
asserts only tier-1/2 surfaces; every pinned fact names a consumer.

### Acceptance criteria

- All 5 commits landed; the §9 acceptance list holds: complete inventory
  (incl. exclusion notes), 12 integration green + 3 documented temp skips on
  the fixture, 30 synthetic tests passing, `-O`-safe list[str] idiom (no
  assert in bodies), zero behavior change (diff confined to the three
  declared paths + dev notes), walkthroughs present, scope-clean.
- **User sign-off at the milestone gate** on the two seam docs + the
  `pytest -rs` report before merge/tag.
- **Rollback trigger:** any pipeline/runtime file in the diff, any
  pre-existing test perturbed, or a validator asserting an upstream-owned
  internal → stop, do not merge, surface it.

### Output requirements

- Commits on `task/p32b-interchange-contracts`; merged to `main` after the
  gate; milestone branch + tag `p32b-interchange-contracts`.
- A **Results delta**: none expected (zero behavior change) — report the
  validator-suite split (green/skip/synthetic counts) as the deliverable
  evidence instead.

### Task constraints

- Design §8 sequencing binding (docs → validators → temp layer → index).
- Naming per `dev/conventions/naming.md`; docs kebab-case; validators
  snake_case `validate_<id>` matching the design's names exactly.
- Every contract fact in the docs must trace to a Snakefile/script line or
  an observed fixture artifact (C4) — transcribe from the design, which is
  already grounded; do not re-derive from memory.

**Human gates** (otherwise drive commit-to-commit autonomously per the
standing preference):

- **Gate 1 — milestone gate, before merge/tag, PAUSE:** present the two seam
  docs, the `pytest -rs` split vs the counting axis, and the zero-diff-scope
  proof for user sign-off.
