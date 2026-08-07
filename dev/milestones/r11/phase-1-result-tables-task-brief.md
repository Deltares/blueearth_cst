# Task Brief — R11 P1: WF3 result tables, wide → long

### Context

Canonical rules: `AGENTS.md`. Program scope and rulings:
`dev/milestones/r11/wf3-consolidation-scope.md`. **Normative specification:**
`dev/milestones/r09/wf3-change-requests.md` CR-2 (and its `### Decision —` blocks)
plus `wf3-changes-proposal.md`, which carries the stable C/F/O numbers. This brief
does not restate the spec; it bounds the unit and names what the spec leaves to
the executor.

- **The spec is already ruled.** CR-2's shape, `realization_id = 0` means pooled,
  metrics split three ways, `value` is `float32` unrounded, `location` is the bare
  gauge id, `metric` is the composite `<variable>_<statistic>`. Do not re-open
  these; implement them.
- **One table per variable** present in `wflow_outvars`, named from the register's
  token table. `q_indicators.csv` keeps its name.
- **Six columns in P1**, not seven: `metric, temp_change, precip_change,
  realization_id, location, value`. `st_id` (C28) is **deferred to P2** — see
  Non-goals.
- **`[R9-5]` was ruled 2026-08-07**: the unperturbed baseline is emitted in both
  shapes as a `(temp_change=0, precip_change=0)` row. It needs no new rule —
  `realization_id = 0` already means pooled.
- **`aggregate_rlz` is retired** (CR-2 ruling b1). In the long shape "aggregated"
  is no longer a shape choice, so the table always carries the finest grain and
  downstream aggregates as it likes.

### Goal

`derive_wflow_indicators` emits one long-format table per configured output
variable, with a fixed six-column header that does not grow with gauge count, and
the contract validators enforce the new shape.

### Non-goals

- **C28 / `st_id` — moved to P2. RULED 2026-08-07**, so P1's header is **six**
  columns and this is settled, not provisional. C28's consistency check compares
  the results against the design table's row for that `st_id`, and that table is
  unit B's; it carries both of its obligations to P2 with it. Full reasoning,
  including why pulling the design table forward was rejected on a half-migrated
  `st_0`/`cst_0` window: `wf3-consolidation-scope.md` §3.
- Unit B's `cst_`→`st_` rename, the design table, C34, F7 — all P2.
- `[R10-13]`, the `[R10-12]` runbook line, the baseline re-record — all P3. **P1
  leaves the baseline red**; that is expected and is why the re-record is one step
  at the end of the milestone.
- Unit D (config surfaces) and the v2 execution model (R12).

### Allowed scope

**Permitted**
- `blueearth_cst/experiment/export_wflow_results.py` — the writer. *(Module keeps
  its pre-R10 name by deliberate narrowing; the rule is `derive_wflow_indicators`.)*
- `blueearth_cst/shared/interchange_contracts.py` — `validate_hm7`,
  `validate_hm_gauge_column_identity`
- `tests/test_export_wflow_results.py`, `tests/test_interchange_contracts.py`
- `Snakefile_climate_experiment` — the rule's `output:`/`params:`, **`WF3_TARGETS`**,
  and the new read of `workflows.model_creation.wflow_outvars`
- `dev/scripts/check_baseline.py` — its literal `TARGETS` entries for the two
  tables become derived
- `dev/scripts/semantic_tree_diff.py` — the R9 path map needs a **pattern** row
  where it currently has two literal result-table paths
- `config/workflows/snake_config.template.yml:194`,
  `config/workflows/snake_config_model_test.yml:82` — remove `aggregate_rlz`
- `dev/reference/contracts/hydrological-model-seam.md` — HM-7's pinned columns
  **and the variable-token vocabulary**, which CR-2 says belongs in the seam doc
- `dev/milestones/r11/` — this brief's `Progress`

**Scope correction, 2026-08-07.** The first four entries above were missing or
too narrow in the brief as first written. CR-2's "accepted costs" make the output
set config-dependent, so the DAG, the baseline target list and the path map all
stop being literal — and WF3 acquires a **new cross-workflow config read** it has
never had (`wflow_outvars` lives under `workflows.model_creation`). Without them
the writer would emit tables Snakemake does not declare. Found before
implementation, by tracing the rule's `output:` block rather than by hitting it.

**Approval-gated** *(Gate 2)*
- Deleting `test_case/test_local/experiments/experiment/` — required before any
  gate run, because that experiment is frozen and `check_not_frozen` will refuse
  the `aggregate_rlz` removal (`wf3-consolidation-scope.md` §9)

**Forbidden**
- `dev/baseline/manifest.json` — P3 owns the single re-record
- `dev/milestones/r09/wf3-change-requests.md` — the spec is settled; if it is
  wrong, PAUSE rather than edit
- Anything under unit B, C34, F7, or unit D

### Required changes (checklist)

1. Rewrite the writer to emit long format, one file per variable in
   `wflow_outvars`, six columns, per CR-2 and its `### Decision —` blocks.
2. Emit the baseline as a `(0, 0)` row in **both** shapes; **delete** the dead
   `st_nb == "0"` guard rather than repairing it — it is an int-to-string
   comparison that can never be true, so it never expressed a choice.
3. Retire `aggregate_rlz`: remove the read at `Snakefile_climate_experiment:892`
   and the key from both config files.
4. Rework `validate_hm7` for the six-column shape, keeping C2's
   `metric.startswith(variable + "_")` assertion.
5. Rework `validate_hm_gauge_column_identity` check 3 to compare the `location`
   column's **value set** instead of the header's column set — same invariant,
   simpler expression.
6. Derive the output set: `WF3_TARGETS` and the rule's `output:` from
   `wflow_outvars`, read from `workflows.model_creation`. New cross-workflow
   coupling — WF3 discovers variables at runtime today
   (`[x for x in sim.columns if "basavg" in x]`), which Snakemake cannot use
   because it needs paths at DAG-construction time.
7. Derive `check_baseline.py`'s `TARGETS` for the tables, and give
   `semantic_tree_diff.py` a pattern row where it has two literal paths.
8. Update `hydrological-model-seam.md`: the new HM-7 columns **and** the
   variable-token vocabulary, which CR-2 places in the seam doc.
9. Update both test modules; add tests for the baseline row, the `float32`
   unrounded values, and the bare-id `location`.

### Commit plan

HM-7 is a pinned contract, and the output set is declared in three places that
must agree. Three commits, each independently runnable.

| # | Subject · paths | Invariant it preserves |
|---|---|---|
| 1 | `feat(wf3): derive the indicator table set from wflow_outvars` · Snakefile (`WF3_TARGETS`, rule `output:`, the new `wflow_outvars` read) + `check_baseline.py` + `semantic_tree_diff.py` | The DAG, the baseline target list and the path map agree on which tables exist **before** any writer emits them. Landing this second would mean a commit where Snakemake declares outputs nothing produces |
| 2 | `feat(wf3): emit indicator tables in long format` · writer + `interchange_contracts` + both test modules + seam doc | The contract and its producer move together — a validator pinning six columns must never exist against a writer emitting the old shape |
| 3 | `chore(wf3): retire aggregate_rlz` · Snakefile + both configs | Isolated so the experiment-freeze break has its own diff and its own line in the migration note |

**Commit 1 before 2 is the correctness property**, not a preference: the reverse
order leaves a commit in which the writer emits `aet_indicators.csv` while the
rule still declares `basin_indicators.csv`, so `--dry-run` fails and the tree is
not runnable at that point in history.

### Validation

**Falsifiers** — each names the observation that would disprove the property:

| Property | Falsifier | Command |
|---|---|---|
| Header does not grow with gauge count | Add a gauge to the fixture config; a seventh column appears | `pytest tests/test_interchange_contracts.py -k gauge` |
| Baseline present in **both** shapes | Either table lacks a `(0,0)` row | `pytest tests/test_export_wflow_results.py -k baseline` |
| `value` is unrounded `float32` | Any value equals its 3-dp rounding for all rows | new test in `test_export_wflow_results.py` |
| `aggregate_rlz` is genuinely dead | `grep -rn "aggregate_rlz\|aggr_rlz"` returns a live read | `git grep -n "aggregate_rlz\|aggr_rlz" -- ':!dev'` |

**Ladder** — report what each rung *caught*, not only that it passed:

| Rung | Scope | Frequency |
|---|---|---|
| 1 | `pytest tests/test_export_wflow_results.py tests/test_interchange_contracts.py` | per edit |
| 2 | the new behavioural tests above | per edit |
| 3 | `pytest tests/test_cli.py` — the rule's `output:` set changes | once, before commit 1 |
| 4 | `pytest tests/` **from the primary checkout**, fixture layer included — a worktree run cannot reach it | once, at phase end |
| 5 | Baseline: **expected red.** Record which targets moved and why; do **not** re-record | once, at phase end |

### Acceptance criteria

- Checklist 1–7 complete; both commits landed in order
- Rungs 1–4 green; rung 5's red is documented target-by-target
- `git grep aggregate_rlz` clean outside `dev/`
- A WF3 run produces one table per configured variable, each with exactly the six
  columns, and no `Q_<id>.1` style duplicate anywhere
- **Rollback:** if the long shape cannot preserve every metric currently emitted,
  stop and report — a silently narrower table is worse than the wide one

### Output requirements

A **Results delta** naming, per moved target: what changed and why. The
baseline is expected to move; the delta is what makes P3's re-record a decision
rather than a rubber stamp.

### Task constraints

- Do not re-open ruled decisions. If the spec looks wrong, PAUSE.
- `float32` is a ruled decision, not a suggestion — do not round for readability.
- Keep the register and `wf3-changes-proposal.md` in step if either is touched.

**Human gates**

1. ~~Confirm C28's move to P2.~~ **RELEASED 2026-08-07** — ruled, C28 is P2's.
   P1 starts without waiting.
2. **Before any gate run** — deleting the fixture's frozen experiment is
   destructive and needs explicit approval, even though it is a fixture.
3. **At phase end** — report the baseline delta and STOP. P3 owns the re-record;
   P1 must not perform it.
