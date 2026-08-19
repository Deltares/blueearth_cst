# Task Brief — implement the WF3 stress-test lookup and derived surface axes

### Context

Canonical ruleset: `AGENTS.md` (repo root). Read it first; `CLAUDE.md` only imports it.

- **The specification is `dev/milestones/r12/stress-test-lookup-design.md`** (ACCEPTED
  2026-08-15, D1–D35). This brief bounds and sequences the work; it does **not**
  restate the design, and where the two differ the design wins. Per-finding
  argument, the 35-row ledger and all five reviewer verdicts:
  `dev/milestones/r12/stress-test-lookup-review-record.md`.
- **Board item:** `t2608152230` (queue 1). Its "design conversation" sections are
  provenance, **not instruction** — three of their conclusions were superseded
  (the alias ruling was withdrawn on measurement, the schema moved to WG-2, and
  D35 added a config constraint none of them anticipated).
- **Units:** temperature change in °C; precipitation mean and variance change in
  **percent**. The config keeps its multiplier convention; percent is the
  artifact's convention. Two conversion sites and only two — D3.
- **The artifact:** `<exp>/config/stress_test_lookup.csv`, `12 × ST_NUM` rows,
  keyed `(st_id, month)`, **no `st_0` row** (D4). `st_id` is a zero-padded
  **string** — read it with an explicit dtype on both the Python and R sides or
  the join silently misses.
- **This lands before R12** (`t2608082036`), whose member-identity re-derivation
  will key `member_hash` on these monthly rows.

### Goal

Replace `<wg>/_work/st_*.csv` and `<exp>/config/stress_test_design.csv` with one
monthly lookup table, drop the baked axis columns from the indicator tables, and
make the response-surface axis a declared post-processing parameter — so the axis
reports the range actually explored instead of an annual collapse that misreports
any seasonal design.

### Non-goals

- The projection overlay (Q6/OQ-2), `st_0`'s comparability warning (OQ-3, owned by
  `t2608151154`), and a third perturbation parameter (OQ-4, refused by C28).
- R12's execution model — manifest, ledger, `member_hash`, resumable sweeps.
- Giving `shared/surface_axes.py` an in-repo caller. **Deliberate** (owner ruling):
  the compensating requirement is that both contract texts stay complete enough to
  re-implement from.

### Allowed scope

**Permitted** — `blueearth_cst/experiment/prepare_cst_parameters.py`,
`export_wflow_results.py`; `blueearth_cst/shared/surface_axes.py` (new),
`interchange_contracts.py`, `snake_utils.py`; `blueearth_cst/weathergen/impose_climate_change.R`,
`read_member_grid.R` (new); `run_stress_test.smk`; `tests/**`;
`dev/scripts/semantic_tree_diff.py`; `dev/reference/contracts/{weather-generator,hydrological-model}-seam.md`;
`dev/reference/workflows/rule-index.md`; `dev/milestones/r12/migration_stress-test-lookup.md` (new);
`docs/notebooks/Climate Stress Test.ipynb`.

**Approval-gated** — `dev/baseline/manifest.json` and its reference tables: only
via `check_baseline.py record`, at steps 0 and 7, in the **primary checkout** with
no other session live. Released by Gate 1 / Gate 3 below.

**Forbidden** — `dev/TODO.md` (generated: `todoboard render`); `pixi.lock`,
`Manifest.toml`; any vendored package under `.pixi/`; the accepted design and its
review record (records, not drafts); `test_case/test_local/**` except as the
step-0/7 re-record rewrites it.

### Required changes (checklist)

Derived from the design's §8; each item is that step's deliverable, not a restatement.

1. **Step 0 (prerequisite, not a commit)** — re-record the baseline "before".
2. **Step 1** — the artifact: `prepare_cst_parameters.py` writes one lookup, percent,
   `st_0`-less, plus D35's `refuse_out_of_domain_multipliers` guard and its
   parse-time call site.
3. **Step 2** — the consumer: rule 3.12's input, `impose_climate_change.R` arity 4→5,
   and the new `read_member_grid.R` with its post-filter assertion.
4. **Step 3** — the reduction: `export_wflow_results.py` loses `perturbation_axes`;
   indicator tables lose `temp_change` / `precip_change`.
5. **Step 4** — `shared/surface_axes.py` + `tests/test_surface_axes.py`, including
   D28's three ordered partition checks and the D8/D13 axis-distinctness refusal.
6. **Step 5** — **two** contracts, WG-2 and HM-7, plus the tree inventory.
7. **Step 5b** — sweep the suite for the old roots **and the old columns** (the
   column sweep is the one v2 lacked; three lenses found three disjoint misses).
8. **Step 6** — docs, seeds, and the notebook **rewritten as a contract-based
   consumer that does not import `surface_axes`**.
9. **Step 7** — re-record the baseline "after".

### Commit plan

Derived from the design's §8 — do not author a competing sequence; read it for each
step's file list and rationale. The boundaries carry correctness properties:

| Commit | Paths | Invariant it preserves |
|---|---|---|
| 1 — the artifact | `prepare_cst_parameters.py`, `run_stress_test.smk` (3.09) | the writer and the grid enumeration stay one loop (C26) |
| 2 — the R consumer | `impose_climate_change.R`, `read_member_grid.R`, 3.12 | producer and consumer of the new shape land together |
| 3 — the reduction | `export_wflow_results.py`, 3.16 | the columns and their last writer die in one commit |
| 4 — the library | `surface_axes.py`, its tests | new surface, independently runnable |
| 5 — contracts + inventory | both seam docs, `semantic_tree_diff.py` | a contract never describes a tree that no longer exists |
| 5b — test sweep | `tests/**` | no test asserts a dead contract while passing |
| 6 — docs, seeds, notebook | as scoped | every live reference updated in the same commit (C5) |

### Validation

**Falsifiers: the design's §9 carries V1–V23, one per claimed runtime property, each
with the observation that would disprove it and the check that produces it. Use that
table as written — it is the deliverable of two review rounds, and three of its
entries exist because an earlier version's claim was measured and found false.**

Do not weaken a V-claim to make it pass. Four are load-bearing: **V17** (three
negative R fixtures *and* two positives — unordered months must normalise, not
raise), **V18** (partition refuses the *incomplete* case, not only the mis-keyed
one), **V20** (a bound on the reconstructed **multiplier**, and explicitly **not** on
any indicator), **V23** (D35 refuses out-of-domain bounds *before* the DAG is built).

| Rung | What | Frequency |
|---|---|---|
| 1 — narrow | the changed module's own tests | per edit |
| 2 — new behaviour | `tests/test_surface_axes.py`, `test_read_member_grid.py` | per edit to their subjects |
| 3 — integration | `pytest tests/test_cli.py` | **every commit** — a rule's declared input changes, and this is the only place a malformed `config/defaults/*.yml` surfaces |
| 4 — full gate | `pixi run test-full` | at the merge — this touches a Snakefile, a `script:` signature *and* `shared/` |
| 5 — baseline | `check_baseline.py check` | after step 7 only; it fails **structurally** between steps 0 and 7 and that is expected, not a defect |

Also per commit: `pixi run lint` and `pixi run format-check` (both CI gates, both near-instant).

**Report what each rung caught, not only that it passed.**

### Acceptance criteria

- All nine checklist items land; V1–V23 pass as written.
- `check_baseline.py check` is green against the step-7 re-record, and the diff
  between the step-0 and step-7 references is explained per V20 — a *multiplier*
  claim, never an indicator claim.
- `pixi run tree-check` reports the retired `_work/` as **UNMAPPED** and the new
  lookup as **IDENTITY** (P3: the current whole-directory prefix would otherwise
  accept a leftover `_work/` silently).
- No live reference to `stress_test_design.csv`, `_work/`, `perturbation_axes` or
  `_PERTURBATION_AXIS` survives outside `dev/` records.
- **Rollback:** if step 3's column removal cannot be reconciled with a consumer not
  named in the design, stop and revert to the step-2 commit rather than widening —
  an unlisted consumer is a design gap, not an implementation decision.

### Output requirements

A **Results delta**: which indicator values moved between the step-0 and step-7
baselines, and why. The expected answer is **none** — the migration changes the
artifact's shape, not the forcing, except where D25's percent round-trip is
inexact. Any indicator movement beyond that is a defect until explained.

### Task constraints

- Run the pipeline from the **primary checkout**, never a worktree (`.snakemake`
  divergence + concurrent locks). WF1 needs `--notemp` for any run feeding the baseline.
- `pixi run install` before WF3 in a fresh env — `weathergenr` comes from `remotes`.
- Never hand-edit `dev/TODO.md`; run `todoboard render`.
- hydromt / wflow conventions verbatim — do not re-engineer upstream behaviour.

**Human gates.**

1. **Gate 1 — before step 1.** PAUSE after the step-0 re-record. The owner confirms
   the "before" reference is recorded from `snake_config_baseline.yml` in the primary
   checkout. *Nothing landing before this is gateable, ever* — a comparison gate
   cannot be applied retrospectively.
2. **Gate 2 — after step 5b.** PAUSE with the sweep's results. The owner confirms no
   test asserts a dead contract while passing. Three reviewers found three **disjoint**
   sets of stale references here; a clean sweep is a claim, not a default.
3. **Gate 3 — before step 7.** PAUSE before the "after" re-record. Destructive:
   it overwrites the comparison reference.
