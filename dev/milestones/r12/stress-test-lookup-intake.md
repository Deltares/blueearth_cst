> **Landed with the accepted design, 2026-08-15.** This is the run's stage-0
> intake — the scope authority every author spawn worked under, and the record of
> what was declared *before* any drafting: the six scope gaps, the eleven settled
> owner constraints, the decision criteria, the non-goals, the evidence register
> (E1–E10, with two carried as explicit hypotheses), the three
> framework-feasibility probes, the gate-materialization check and the
> derived-artifact register.
>
> It is kept unedited. Its value is that it predates the design, so a later reader
> can see which premises were checked up front and which were not — including the
> two stage-0 findings that survived to the approval gate as R1 and R2.
>
> Design: `stress-test-lookup-design.md`. Reviews, ledger and process log:
> `stress-test-lookup-review-record.md`. Per-round scratch (the `design-vN`
> series and the raw reviewer files) lives in git history under
> `dev/working/design-runs/wf3-stress-test-lookup/`.

# Intake — WF3 stress-test lookup and surface axes

Stage 0 of a `design-review-loop` run. Driver-authored; no design content here.
Run state: `status.md`.

## The change request, verbatim

> Continue working on the item from the previous session t2608152230 — the
> lookup-table redesign. […] Design and six[ rulings are in]
> `dev/working/2026-08-15_wf3-scenario-generation-trace/stress-test-design-and-surface-axes.md`.

and, after the alias precondition was run and the R12 boundary ruled:

> start on step 3

where step 3 was named as: *promote the working note to an accepted design,
closing the axis-declaration schema, the two unenforced constraints, the caption
spec, HM-7's replacement text, and the migration note.*

## Problem

WF3 fuses two things that are conceptually distinct: **the experiment** (what
perturbation was imposed, and what the system did) and **the response surface**
(a post-processed view of the result). The fusion has two costs.

1. **A correctness cost.** `export_wflow_results.annual_perturbation` collapses
   each member's twelve monthly values to one annual figure *at reduction time*
   and bakes it into the indicator tables, so no other axis is recoverable from
   the results. For a seasonal design the fixed annual collapse **misreports what
   was explored**: +30% imposed in JJA reads as +7.6% on the axis, and a
   single-month perturbation compresses to roughly a twelfth of its magnitude.
2. **A duplication cost.** The grid is written as two artifacts —
   `<wg>/_work/st_<id>.csv` (monthly, precip as a *multiplier*) and
   `<exp>/config/stress_test_design.csv` (per member, precip as a *percent*, the
   annual collapse of the first). The second is a materialized cache of the
   first, derived independently by the writer, which is why `validate_hm7` exists
   to police the drift between them.

The design conversation that opened these questions is
`dev/working/2026-08-15_wf3-scenario-generation-trace/stress-test-design-and-surface-axes.md`.
It carries **six owner rulings plus two same-day revisions** and states plainly
that it is *"not an accepted design and not a task brief"*. This run closes that
gap: the contested questions are ruled, the mechanism is largely unwritten.

## Scope — what this design must close

Six gaps, derived from reading the note against the tree. None re-opens a ruling.

| # | Gap | Why it blocks implementation |
|---|---|---|
| 1 | **The axis declaration has no config schema.** §3 gives a shape (`{variable, months, statistic}`) and nothing else: where it lives, how many surfaces per experiment, how it validates, how it meets `advanced_settings.yml`'s closed schema | Everything downstream reads it |
| 2 | **The consumer side is unassigned.** Dropping the baked axis means something must join lookup × indicators at plot time; today `export_wflow_results.py` bakes it | A rule? a helper module? — and the R12 boundary (below) constrains the answer |
| 3 | **Two stated constraints have no enforcement designed** — "linear statistics only" (§3) and the §4 "this is checkable" warning about varying months carrying differing `(min, max)` pairs | Both are correctness properties the design asserts and nothing checks |
| 4 | **HM-7's replacement text does not exist** — the seven-column contract loses two columns and the cache-drift check retires with the cache | A contract document is normative here, not descriptive |
| 5 | **The derived caption is a sketch** (§4) — undefined for case 3 (months held at an offset) and for a design with no varying months | It is the note's strongest argument for the merged table |
| 6 | **Migration and tree shape** — `naming.md` §7 requires a migration note; `_work/` disappearing touches the project-tree inventory, the scaffold, the cross-workflow input set and the baseline scope | A tree-shape change the fixture-dependent test layer cannot catch in a worktree |

## Constraints — settled, not open for review

These arrive as **settled framing**. A reviewer may note a consequence; it may not
re-litigate the ruling.

| Constraint | Source |
|---|---|
| **Percent everywhere.** `temp_change` in °C; `precip_change` and `precip_variance_change` in percent. Column names stay unsuffixed | Q1, owner 2026-08-15 |
| **The lookup is the source of truth.** Indicator tables carry `st_id` + `value`; no baked axis; axis values are derived, never stored | Q2, owner 2026-08-15 |
| **The lookup determines the AXIS, not the SCENARIO** | §3 qualifier, owner 2026-08-15 |
| **No external consumer constrains this.** CST-API / frontend out of scope; `csthelpers` is parameterized and its owner updates it | Q3, owner 2026-08-15 |
| **Name: `stress_test_lookup.csv`**, in `<exp>/config/`; `_work/` disappears | Q4, owner 2026-08-15 |
| **`st_0` is not a surface member** — baseline reference only, reported as an annotated value; it stays simulated | §5, owner 2026-08-15, standing with a caveat |
| **The identity member is simulated like any other.** The alias is withdrawn; `st_id` stays dense | §5, owner 2026-08-15 (withdrawal) |
| **The lookup lands before R12's member-identity re-derivation**, which then keys `member_hash` on the monthly rows | §7b, owner 2026-08-15 |
| **Only linear statistics may define an axis**, or HM-7's evenly-spaced guarantee breaks | §3, inherited from HM-7 |
| **The same collapse must apply to the projection overlay** | HM-7; treatment deferred as Q6 |
| Repo-wide: this is the workflow engine only; hydromt / wflow conventions are used verbatim, never re-engineered | `AGENTS.md` § Hard Constraints |

## Decision criteria

1. **Correctness first.** The axis must report the range that was explored. A
   design that keeps the misreport is rejected regardless of its other merits.
2. **Store the finest grain imposed; derive every summary.** The principle that
   killed both the design-table cache and the proposed `axes.csv`.
3. **No new cache of a derivation**, at any layer.
4. **A new perturbation parameter should be a column, not a file shape** — while
   respecting C28's deliberate refusal of a third *axis*.
5. **The migration is a rename plus a shape change**, and must be executable in
   one commit with every live reference updated (`AGENTS.md` § Conventions).
6. **Gate-ability.** Every claimed runtime property must have an observation that
   would falsify it (see the gate check below).

## Success criteria

- All six scope gaps closed in normative text, not sketch.
- Every ruling in the constraints table carried forward unaltered.
- A migration note satisfying `naming.md` §7.
- An HM-7 replacement that can be dropped into
  `dev/reference/contracts/hydrological-model-seam.md`.
- A claim → falsifier table handed to `task-brief` at stage 7.

## Non-goals

- **The projection overlay (Q6).** Deliberately deferred; the constraint is
  pinned so it cannot drift.
- **A third stress-test axis.** C28 refuses one deliberately; removing the
  *shape* barrier does not remove the *contract* barrier.
- **Members varying seasonal pattern independently** — a second design dimension,
  colliding with C28. Named in Q5 and out of scope.
- **R12's execution model** — manifest, ledger, `member_hash`, resumable sweeps,
  epochs, quarantine, atomic publication. Owned by `t2608082036`.
- **Fixing `st_0`'s comparability** — `t2608151154`, `origin: R12`.
- **`precip_variance` in `member_hash`** — G1 retention ruling, followup `R9-F1`.
- Any change to CST-API, CST-frontend, or `csthelpers`.

## Evidence register

Empirical premises the design leans on. Falsity of any row changes a decision.

| # | Premise | Source | Exact observation | Precision | Reproduction | Confidence |
|---|---|---|---|---|---|---|
| E1 | `stress_test_design.csv` is a materialized cache of the member files, derived independently by the writer | `prepare_cst_parameters.py:175-189` | The rule writes `st_<i>.csv`, reads it back off disk, and calls the same `perturbation_axes` the reduction calls | exact (code read) | Read the function | **Verified** 2026-08-15 |
| E2 | The annual collapse misreports a seasonal design | §3 arithmetic | `(92 × 1.30 + 273 × 1.00) / 365 = 1.0757…` → +7.6% for a +30% JJA perturbation | 4 sf | Arithmetic | **Verified** (closed form) |
| E3 | WF2 already emits percent, with an explicit units column | `projections/change_factor_table.py:65,88,154` | `relative_units` = `%` when `change_kind == "relative"`; `PERCENT` constant | exact (code read) | Read the module | **Verified** 2026-08-15 |
| E4 | The per-file split buys no invalidation granularity | `run_stress_test.smk:817-836` | Rule 3.09 declares **all** `st_<m>.csv` plus the design CSV as one job's outputs, so any config change rewrites all of them | exact (rule read) | Read the rule | **Verified** 2026-08-15 |
| E5 | Rule 3.09 is deaf to `stress_test` edits | `run_stress_test.smk:819-821` | `config = ancient(config_path)`, and the rule carries **no `params:`** | exact (rule read) | Read the rule | **Verified** 2026-08-15 |
| E6 | Unit perturbation factors are **not** the identity | `weathergenr` 1.2.0 | `adjust_precipitation_qm` called unconditionally (body line 263, no `mean_factor == 1` short-circuit). Probed: temperature identity exact (max abs Δ = 0); precipitation changes every wet day; all twelve monthly means preserved to **+0.0000%**; single max day −32.9%, max 7-day sum −19.9%, sd −4.9% | monthly means to 4 dp; tail to 1 dp | `scratchpad/identity_probe2.R` (this session) | **Verified** 2026-08-15 — measured |
| E7 | `st_0` and the grid origin differ materially | `test_local` `q_indicators.csv` | Of eleven `q` metrics: one preserved (`q_annual_mean` +0.2%), five ≤20%, five by a factor (`q_mean_annual_min` −69.7%, `q_return_level_2yr_7day_min` +127.9%) | 1 dp on percentages | Compare `st_id == 0` vs `st_id == 2` | **Verified, magnitudes provisional** — the fixture predates the 2026-08-12 weathergenr 1.2.0 rename, so the numbers come from the older `imposeClimateChanges`; direction and forcing-side magnitude re-measured on 1.2.0 (E6) |
| E8 | R12's `member_hash` indexes members by the annual collapse this design deletes | `design-v4.md:987` on `docs/wf3-redesign` | Hash tuple includes `tavg`, `prcp`, `precip_variance`, field-noted as "the annual scalars … derived exactly as the reduction derives them today". Current HEAD equivalent: `perturbation_axes` → `annual_perturbation`, `export_wflow_results.py:300-318` | exact (both read) | `git show docs/wf3-redesign:dev/working/design-runs/wf3-experiment-v2/design-v4.md` | **Verified** 2026-08-15 |
| E9 | `csthelpers::plot_climate_surface` is parameterized, so dropping the axis columns is not a live-integration break | §5b of the note | Takes `x_var` / `y_var` and validates against the columns passed; only the *example* names `precip_change` | — | Read `~/workspace/csthelpers` | **HYPOTHESIS — asserted in the note, not re-verified this run.** Settled by reading the function. Q3 rules it out of scope either way, so falsity changes no decision here — recorded so a reviewer prices it rather than inherits it |
| E10 | The three interpretable designs are already expressible in the current config | §4 | Cases 1–3 need only the existing 12-element `min`/`max` vectors | — | Construct the three configs and dry-run | **HYPOTHESIS** — argued, not executed |

## Framework-feasibility probes

Mechanisms whose feasibility depends on Snakemake execution semantics. Each needs
a probe with a recorded result, **not a paragraph of prose**.

| Probe | Question | Why prose cannot settle it |
|---|---|---|
| P1 | Does collapsing twelve declared outputs into one change rule 3.12's fan-out or its `wildcard_constraints`? | 3.12 is the fan-out point and carries a constraint restricting `st_num ≥ 1` that exists to stop a `CyclicGraphException`; a changed input shape is exactly where that bites |
| P2 | Does the axis-derivation consumer need a rule, and if so does it re-fire correctly on an axis-declaration edit? | The `ancient()`/no-`params:` trap (E5) is live in this workflow and a new rule inherits the same hazard |
| P3 | Does removing `_work/` leave a declared-but-unwritten directory anywhere in the scaffold or inventory? | `tree-check` compares against a coded inventory, not a snapshot; a stale entry reports as undeclared on every future run |

## Gate materialization

Every gate this design's migration and validation plan will cite, and whether it
can execute **today**.

| Gate | Verdict | Detail |
|---|---|---|
| `pytest tests/test_prepare_cst_parameters.py`, `test_interchange_contracts.py`, `test_export_wflow_results.py` | **Runnable** | The narrow tier; these three own the changed surfaces |
| `pytest tests/test_cli.py` | **Runnable** | Required here — a rule's declared input changes, and this is the only place a malformed `config/defaults/*.yml` surfaces |
| `pixi run test-fast` / `test-full` | **Runnable** | `test-full` is required at the merge: this touches a Snakefile, a `script:` signature *and* `shared/` |
| `pixi run tree-check` | **Runnable, needs a code change with the design** | The inventory is code, not a snapshot; `_work/` removal and the new `<exp>/config/stress_test_lookup.csv` must land in `semantic_tree_diff.py` in the same commit |
| `check_baseline.py check` | **NEEDS A PRE-CHANGE ARTIFACT — and the current one is not it** | Dropping `temp_change`/`precip_change` changes the `indicator` target's columns, so the gate *will* fail by design and needs a re-record. Two board items say the existing baseline cannot serve as the "before": `t2608131718` (the baseline's two flat config copies stale since 2026-08-12) and `t2608121258` (the `test_local` fixture predates the weathergenr 1.2.0 rename). **A re-record must happen before the first implementation commit**, from `snake_config_baseline.yml`, in the primary checkout, with `--notemp` on WF1 and no other session live — otherwise every step that lands before it is permanently ungateable |
| `check_baseline.py check` coverage | **Runnable but thin** | `FIGURE_KINDS` excluded by default, and `stress_test_design.csv` is deliberately outside the manifest (R11 ruling) — so the artifact this design replaces is *not* covered by the numerical gate at all. Carry to G2 as a named gap |
| `validate_hm7` | **Changes with the design** | It is part of the deliverable, not an independent check; the cache-drift half retires with the cache |
| The fixture-dependent test layer | **Cannot run in a worktree** | It skips rather than fails, and this is a tree-shape change — exactly the case `AGENTS.md` records as surviving every gate a branch can run. Implementation gating must happen in the primary checkout |

## Derived-artifact register

Artifacts that derive from this design and go stale the moment review changes it.
**Author spawns are barred from touching them**; each is regenerated from the
accepted version after G2.

| Artifact | Regenerated by |
|---|---|
| `dev/tasks/t2608152230-…md` (queue 1) | Rewrite its Overview and Refs from the accepted design; close on landing |
| `dev/tasks/t2608082036-…md` (queue 2) | Update the `member_hash` dependency block against the accepted lookup schema |
| `dev/tasks/t2608151154-…md` | Re-point its §6.6 corroboration if the design changes st_0's treatment |
| `dev/reference/contracts/hydrological-model-seam.md` (HM-7) | Replace from the accepted design's contract section |
| `dev/reference/workflows/rule-index.md` | Regenerate the 3.09 / 3.12 / 3.16 rows |
| `dev/roadmap.md` § R12 | Refresh the gate paragraph if sequencing changes |
| The migration note (`naming.md` §7) | Author from the accepted design |
| The `task-brief` for implementation | Stage 7 handoff, with the claim → falsifier table |
| `docs/notebooks/*.ipynb` (3 files reference `stress_test_design`) | Re-render after implementation, per `t2608132100` |

## Genre mapping

`workflow-spec` — it specifies artifacts, rule boundaries and a data contract
inside one workflow. It carries a **method component** (what statistic may define
an axis, and why only linear ones), recorded here rather than by inventing a new
enum value, per `run-artifacts.md`.

## Seeding

Stage 1 seeds from
`dev/working/2026-08-15_wf3-scenario-generation-trace/stress-test-design-and-surface-axes.md`
rather than re-authoring owner-ruled content. **Structural checks fail** — the note
is a design *conversation record*: it has no `## Alternatives considered` section
and no genre sections. So the seed takes the restructure path (author spawn scoped
to *reshape to the genre contract, preserving all content verbatim*), not the
mechanical-copy path.

The source note stays where it is. It is the provenance record for six rulings and
two same-day revisions, and `dev/README.md`'s promotion rule governs its eventual
disposition.
