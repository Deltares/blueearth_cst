# Task Brief — R10 step 6: settle the rule identifier surface

### Context

Canonical rules: `AGENTS.md`. Design: `dev/milestones/r10/rule-naming-design.md`
(**accepted 2026-08-04, amended three times 2026-08-06**) and
`dev/reference/workflows/rule-index.md`, which publishes the target state.
Landing order: `dev/followups.md`, step 6.

- **The rule set is finally stable.** Step 6 was deliberately scheduled last;
  `[R10-6]` §8–12 has now landed and added three rules (`1.01c` / `2.03c` /
  `3.01f` `delineate_spatial_units`). This is the first moment the numbers can
  be assigned once rather than twice.
- **`rule-index.md`'s renumber map is explicitly NOT final** — it says so in its
  own banner, because it predates those three rules. **Regenerating it is part
  of this task**, not a follow-up.
- **`rule-index.md` is stale in two further ways** (both from 2026-08-06
  rulings, neither yet reflected there): `[R10-2]` is DROPPED, so its
  `plot_results` split and the target rule `1.14 evaluate_wflow_run` must go;
  and `evaluate_` is withdrawn from the verb table. `rule-naming-design.md` is
  already corrected (Amendment 3) — `rule-index.md` is not.
- **`3.01f` exists only because `3.01c`–`3.01e` were taken.** The renumber is
  what dissolves that; expect `delineate_spatial_units` to land adjacent to
  `delineate_region` in all three workflows.
- Every rule number appears in **six places**: `rule_banner(...)`, `log:`,
  `benchmark:`, `LOG_RULES`, the `# W.NN name — …` comment header, and
  `rule-index.md`. `[R10-9]` proposes one label constant per rule, which is free
  only while every call site is already being edited.

### Goal

Rule numbers follow logical order (data → build → run → records, contiguous,
every dependency low→high), the twelve renames land, the shared-rule helpers
take the `_rule` suffix, and the two `LOG_RULES` test defects close — with
`rule-index.md` regenerated as the published truth.

### Non-goals

- **No behaviour change.** No rule gains, loses or moves an output; the baseline
  must pass **unchanged**, with no re-record.
- No `[R10-2]` split, and no `evaluate_` verb — both withdrawn 2026-08-06.
- No new rules, no merges beyond `[R10-1]` (already landed), and none of the two
  merges `[R10-3]` rejected.
- No `dev/milestones/` archive rewriting **except** `rule-naming-design.md`,
  which is a live, not-yet-implemented design (its own status line says so).

### Allowed scope

**Permitted**
- `Snakefile_model_creation`, `Snakefile_climate_projections`,
  `Snakefile_climate_experiment`
- `blueearth_cst/shared/snake_utils.py` — the `_spec` → `_rule` renames
- `tests/` — including `tests/test_region_spec.py` → `tests/test_region_rule.py`,
  `tests/test_model_reference.py`, `tests/test_log_rules_contract.py`
- `dev/reference/workflows/rule-index.md`, `dev/milestones/r10/rule-naming-design.md`,
  `dev/followups.md`
- `benchmarks`/rule-map prose that names a number or label

**Approval-gated** — pause and ask
- Any change that alters a **durable** output path. Rule renames move
  `logs/_parts/<label>.log` and `benchmarks/_parts/<label>.tsv`, which are
  transient and merged-then-deleted each run — that is in scope. A change
  reaching anything under `data/`, `models/` or `results/` is not.
- The `gather_benchmarks` vs `gather_logs` ordering disagreement (below).

**Forbidden**
- `dev/milestones/` other than `rule-naming-design.md`; `pixi.lock`;
  `Manifest.toml`; anything under a `project_dir`; `.pixi/`.

### Required changes (checklist)

1. **Regenerate the renumber map** in `rule-index.md` from the CURRENT rule set,
   including the three `delineate_spatial_units` rules. Remove the "not final"
   banner once it is.
2. **Correct `rule-index.md` for the two 2026-08-06 rulings**: drop `[R10-2]`'s
   split row and the `1.14 evaluate_wflow_run` detail section; remove `evaluate_`
   from the verb table.
3. **Apply the twelve renames** (`rule-index.md` § *Twelve renames*), each
   across all six call sites.
4. **Apply the renumber**, same six call sites.
5. **`[R10-7]`**: `region_spec` → `region_rule`, `climate_store_spec` →
   `climate_store_rule`, dataclasses `RegionSpec`/`ClimateStoreSpec` →
   `RegionRule`/`ClimateStoreRule`, and `tests/test_region_spec.py` →
   `tests/test_region_rule.py`. `spatial_units_rule` is already correct.
6. **`[R10-10]`**: `test_model_reference.py` slices the `LOG_RULES` block to the
   first `]`, so a bracket in a comment blinds it; and its assertion sits inside
   the per-Snakefile loop, so the first failing file hides the rest. Fix both, or
   fold the check into `tests/test_log_rules_contract.py` and delete the
   duplicate — two modules asserting one property by different parsers is how
   they came to disagree.
7. **`[R10-9]`'s deferred ordering assertion**: add it now that number,
   execution and sort order coincide. **This forces the `gather_benchmarks` vs
   `gather_logs` ordering decision** — WF1/WF3 and WF2 currently disagree. Raise
   it, do not pick silently.
8. *(Optional, and only if it stays contained)* one label constant per rule, per
   `[R10-9]`. Drop it the moment it grows the diff.

### Commit plan

Renames and renumbers break every reference the instant they land, so each
commit is an atomic transform. Each must leave the tree runnable.

| # | Subject | Paths | Invariant preserved |
|---|---|---|---|
| 1 | `[R10-10]` + `[R10-9]` ordering assertion | `tests/` | the label contract is verified BEFORE the sweep edits it |
| 2 | regenerate + correct `rule-index.md` | `dev/reference/`, `dev/milestones/r10/` | the published map matches the live rule set; no code moves yet |
| 3 | `[R10-7]` helper rename | `snake_utils.py`, three Snakefiles, `tests/` | Python identifiers only; no rule name or number moves |
| 4 | the twelve renames | three Snakefiles, `tests/`, prose | every rule's six call sites move together |
| 5 | the renumber | three Snakefiles, `tests/`, prose | ditto; `rule-index.md` already says these numbers |

Commit 1 first is deliberate: it is the instrument that catches commits 4–5
going wrong, and `[R10-9]` was written for exactly this sweep.

### Validation

| Rung | Command | Frequency |
|---|---|---|
| 1 Narrow | `pytest tests/test_log_rules_contract.py tests/test_model_reference.py tests/test_region_rule.py tests/test_climate_store_contract.py tests/test_spatial_units_rule.py` | per edit |
| 2 Contract | `pytest tests/test_cli.py` — all three Snakefiles parse and dry-run | per commit |
| 3 Full gate | `pytest tests/` | per commit from 3 onward |
| 4 Baseline | `check_baseline.py check` | ONCE, before merge, primary checkout |
| 5 Tree shape | `pixi run tree-check` | with rung 4 |

**Falsifiers.** Each property this task claims, with the observation that would
disprove it:

- **"Behaviour-preserving."** `check_baseline.py check` must pass **unchanged** —
  a re-record is a defect signal here, not a step. Rule identifiers reach no
  artifact the manifest fingerprints; if one moves, a rename touched something
  durable.
- **"No stale reference survives."** Grep the tree for every OLD rule name and
  every old `W.NN` label. Expected: zero hits outside `dev/milestones/` archives
  and `rule-index.md` § *What changed*, which is the one place that names them
  on purpose. This is the failure mode of a rename sweep, and no test reaches it.
- **"Every label has a producing rule, and every logging rule a label."**
  `test_log_rules_contract.py` — and it must be shown **failing** at least once
  during the sweep (revert one label, watch it fire) or it is untested insurance.
- **"Log parts still merge."** `pixi run tree-check` clean: renamed parts land
  under `logs/_parts/`, which the post-migration inventory covers as a prefix
  (`[R10-11]`). A stranded part from a renamed rule shows up as an unmapped path.

Report what each rung **caught**, not only that it passed.

### Acceptance criteria

- All three Snakefiles parse; `pytest tests/` green.
- `check_baseline.py check` passes **with no re-record**.
- Zero stale rule names or numbers outside the two sanctioned places.
- `rule-index.md` carries no "not final" banner, no `[R10-2]` split, no
  `evaluate_`, and its numbers match the Snakefiles exactly.
- **Rollback trigger:** any baseline diff, or a `LOG_RULES` label without a
  producer.

### Output requirements

- The five commits above, each independently runnable.
- `dev/followups.md`: close `[R10-5]`, `[R10-7]`, `[R10-9]`'s deferral and
  `[R10-10]`; mark landing-order step 6 done.
- **Results delta:** none expected. If any output moves, stop — that contradicts
  the design and is a rollback trigger, not a finding to write up.
- A one-line note on whether `[R10-9]`'s label-constant idea was taken or
  dropped, and why.

### Task constraints

- Run the pipeline from the **primary checkout**, never a worktree
  (`AGENTS.md`). `pytest` and dry-runs are fine in a worktree; the baseline is
  not.
- `AGENTS.md` validation ladder governs: unit tests while iterating, broader
  checks once at the commit.
- Rule numbers are a **reading aid, not execution order** — keep that framing in
  every comment the sweep touches.

**Human gates**

1. **After commit 2, PAUSE.** The regenerated number map is published
   documentation and a reading contract; the owner confirms it before any code
   moves against it.
2. **At checklist item 7, PAUSE.** The `gather_benchmarks` vs `gather_logs`
   ordering is a disagreement between workflows that nobody has ruled on. Raise
   it as a decision; do not pick.
3. **Before the baseline run, PAUSE.** It needs the primary checkout, and it is
   the gate that decides whether the sweep was behaviour-preserving.
