# R11 — WF3 artifact and identification consolidation (SCOPE)

Confirmed scope, 2026-08-07. This is a **scoping record**, not a design: it fixes
purpose, boundaries, constraints and success criteria, and names what is still
open. The normative specification already exists — `dev/milestones/r09/wf3-change-requests.md`
(implementation, dense with file/line references) and `wf3-changes-proposal.md`
(the reviewable companion carrying the stable **C / F / O** numbers). R11 does not
restate them; it says which parts land, in what order, and under what gates.

**Next artifact:** a task brief per phase, once the two open questions below are
ruled.

---

## 1. Why this milestone exists, and why now

Two WF3 redesigns exist and they are **not rival versions of the same thing** —
they are different layers, written against different trees:

| | `docs/wf3-redesign` (the v2 design run) | `dev/milestones/r09/wf3-change-requests.md` |
| --- | --- | --- |
| Authored | 2026-08-01 → 08-04 | opened 2026-08-05 |
| Written against | the **pre-R9** tree | "the first full WF3 run of the **migrated R9 tree**" |
| Changes | how WF3 **executes** — member-level incrementality out of Snakemake into our own code: a manifest (the plan), a ledger (append-only record), a `member_hash` fingerprint, resumable sweeps | what WF3 **emits** and what its members are **called** |
| Weight | 14 855 lines, v1→v4, 2 external review rounds, 65 unique findings | 34 changes / 16 findings, four units |
| State | **G2 pending** since 2026-08-04, flagged `rejected-major-part-pending-G2` | CR-1 landed, CR-5/C29 landed, CR-2 specified, unit D deferred |

The register postdates the migration; the design run predates it. WF3 is exactly
the surface R9 rewrote (member filenames, result tables, `model_reference.yml`,
the drift guard) and R10 renamed five rules in. Measured on `design-v4.md`
(3 420 lines) 2026-08-07:

| stale reference | count | why it is stale |
| --- | --- | --- |
| `Qstats` | 25 | R9 renamed it `q_indicators.csv` |
| `RT_` | 15 | R9 **deleted** those tables |
| `export_wflow_results` | 14 | R9 renamed it `derive_wflow_indicators` |
| `prepare_weagen_config` | 12 | R10 renamed it `prepare_weathergen_config` |
| `generate_climate_stress_test` | 3 | R10 renamed it `perturb_climate_realization` |
| `climate_stress_parameters` | 3 | R10 renamed it `prepare_stress_test_grid` |
| `hydrology_model` | 2 | R9 moved it to `models/hydrology/wflow` |
| `model_reference` / drift guard | 3 | barely aware of R9's central WF3 mechanism |

~74 references to artifacts and rules that no longer exist, fifteen of them
reasoning about tables that were deleted. **The findings remain valuable — two
external review rounds do not come cheap — but the mechanics need re-deriving.**

So the two efforts need **sequencing, not merging**. R11 takes the layer that is
already specified against the current tree; the execution model becomes **R12**,
re-derived, with `cst-run-control` governing. The one genuine coupling is narrow
and runs in R11's favour: unit A settles the result-table shapes that v2's gate
arithmetic depends on, so R12 starts from a stable base rather than a moving one.

---

## 2. Purpose

Land the WF3 changes already specified against the post-R9 tree, and close the
WF3-territory followups. R11 changes **what WF3 emits and what its members are
called** — not how it executes.

---

## 3. In scope

| Unit | Changes | Work |
| --- | --- | --- |
| **A — result tables** | C2–C21 | `q_indicators.csv` wide→long; one table per variable present in `wflow_outvars`; **six** fixed columns; rows-not-columns for locations. Rewrite `derive_wflow_indicators`, rework `validate_hm7` and the relational validator, update tests |
| **B — run identification** | C22–C27, **C28** | `cst_`→`st_` rename, the design table, DAG-time enumeration, and `st_id` in the results tables |
| **C — generator plumbing** | C34 only | audit both weathergenr call sites. C29 already landed as CR-5 (rule 3.05 retired, `9260668`) |
| **F7** | one line | declare the weathergen template as an input to rule 3.04 |
| **`[R9-5]`** | folds into A | the unperturbed baseline in both table shapes |
| **`[R10-13]`** | small, cross-cutting | a failing `script:` rule must write its traceback to its own log part |

### C28 moved from unit A to unit B — ruled 2026-08-07 (Gate 1)

The register's batch plan lists C28 under unit A, but its header is
`metric, st_id, …` and its mandatory consistency check compares the results
against *the design table's row for that `st_id`* — and that table is C23–C27, in
unit B. C28 cannot land before it exists, and it takes both of its obligations
with it: the consistency assertion, and the hard stop when a third stress
dimension arrives.

**The alternative was rejected on a half-migrated window.** Pulling the design
table forward into unit A looks tidier — the results header would reach its final
seven columns in one move — but C23 specifies a row for **`st_0`**, the *new*
token, while C22's rename stays in unit B. Unit A would therefore either emit
`st_0` beside `cst_0` filenames for a whole phase, or write `cst_0` and have unit
B rewrite the table it had just built. Moving C28 has no such window: unit A has
no `st_id` at all and filenames stay `cst_`, then unit B lands the rename and the
table together, consistent at both ends.

**Cost accepted:** HM-7's pinned columns and the seam contract change twice, six
then seven. The second change is new behaviour rather than redone work, and the
single P3 re-record absorbs both table moves.

### F7 is here because unit D's deferral orphaned it

The register disposes of F7 (undeclared template input) via C31, inside unit D.
With D deferred, F7 needs its own one-line fix or it stays open, and the register
says so explicitly: *"Do not let it fall between the two."* It is carried here
for that reason and no other.

### `[R9-5]` — ruled 2026-08-07: the baseline is a member of the surface

The unperturbed baseline (`cst_0`, the 0/0 point) is emitted as a row in **both**
table shapes. Today the two code paths silently disagree — present under
`aggregate_rlz: false`, absent under `true` — and the disagreement is hidden by a
dead comparison: `st_nb == "0"` is int-to-string in the aggregated branch and can
never be true, which is harmless only because that loop never reaches the
baseline anyway.

**The ruling needs no new rule**, because CR-2's schema already carries it:
`realization_id = 0` means *pooled over all realizations*, so under aggregation
the baseline is simply another `(temp_change=0, precip_change=0)` row pooled like
every other. Rationale for including it: it is a legitimate point on the response
surface, the runs exist whenever `run_hist` is set, and a surface missing its own
origin forces every downstream consumer to reconstruct it.

### `[R10-13]` is deliberately cross-cutting

The fix lands in `tee_to_log`, which improves every `script:` rule in all three
workflows, so R11 is not purely a WF3 milestone. This is recorded rather than
hidden — see open question **Q2**.

---

## 4. Explicitly out of scope

- **Unit D (config surfaces, C30–C33).** The only unit with a breaking migration
  — the hydrological-year unification touches every project config. Deferred once
  already; its specification stays complete in the register. Not what this thread
  set out to fix.
- **The v2 execution model.** Manifest, ledger, `member_hash`, resumable sweeps.
  Becomes **R12**. Do not merge `docs/wf3-redesign`: it is 289 commits behind with
  an unratified gate, so it is an *input* to R12, not part of it.
- **`[R10-12]` — the drift guard.** Ruled 2026-08-07: **accept and document**, no
  code. `inmaps_historical.nc` is not byte-reproducible (chunk/encoding layout,
  not values), so every WF1 rebuild trips the guard. R11 adds a runbook line
  stating that a post-rebuild re-record is expected — and stating that accepting
  it is an **operator decision** ("this experiment now accepts the rebuilt
  model"), not a chore. *The known cost of this ruling, recorded so it stays
  visible: routine re-records are how a real drift eventually gets waved through.
  Becomes a watch-item carrying that trigger.*
- **`[R10-14]`** — the shared-rule comment-edit cascade. Already a watch-item;
  stays one.

---

## 5. Constraints

- **Exactly one baseline re-record for the whole milestone.** The register's
  collect-then-implement rule exists so the contract, the validators, the
  migration note and the re-record move once rather than per change. Units A and
  B both move recorded artifacts.
- **HM-7 and the interchange contracts are migrated, not broken.** CR-2 changes
  the shape they pin. `validate_hm_gauge_column_identity`'s check 3 compares
  column *sets* today; post-CR-2 it compares the `location` column's value set —
  same invariant, simpler expression.
- **`LOG_RULES` gets its own careful pass**, separate from the table work. The
  register flags this as a shared hazard: a missed entry makes a log section
  silently vanish while its parts stay on disk forever. Two separate passes beat
  one tangled one — the same reason R10 kept its sweep apart.
- **Unit B moves filenames and catalog keys**, so the R9 path map and
  `pixi run tree-check` are updated in the same commits, not afterwards.
- **The register and `wf3-changes-proposal.md` must be kept in step.** A decision
  changed in one without the other leaves the reviewable version lying.

---

## 6. Success criteria

- Units A, B, C34 and F7 landed; `[R9-5]` and `[R10-13]` closed
- `pytest tests/` green **from the primary checkout**, fixture layer included —
  a worktree run cannot exercise it (`AGENTS.md`, worktree section)
- A full three-workflow run: every merged-log section present in rule-number
  order, no surviving `_parts/`
- `check_baseline check` passing after **exactly one** documented re-record
- `pixi run tree-check` clean against an updated path map
- Migration note recorded per `naming.md` §7
- A repository-wide grep for `cst_` returning nothing outside the migration record

---

## 7. Shape

Three phases, following the register's own order (its steps 1–3 are already
complete: the owed WF3 re-run closed with R9-2, C29 was ruled and landed, R10
sealed 2026-08-07).

| Phase | Contents | Risk |
| --- | --- | --- |
| **P1** | Unit A — result tables, six columns. **Deletes the fixture's frozen experiment first** (§9). Brief: [`phase-1-result-tables-task-brief.md`](phase-1-result-tables-task-brief.md) | medium; moves numbers |
| **P2** | Unit B **including C28**, with C34 and F7 alongside — the rename, the design table, and `st_id` land together | medium; moves paths and names |
| **P3** | `[R10-13]`, the `[R10-12]` runbook line, then the single re-record | low |

**One human gate, before the re-record:** the scientific-delta check R9 used.
That gate is what established that R9-2's movement was header-only and nothing
numeric had moved; without it a re-record blesses whatever happens to be on disk.

---

## 8. Open questions

**Q1 — RESOLVED 2026-08-07. The answer inverted the question; see §10.**

**Q2 — Should `[R10-13]` land here or separately?**
It fixes `tee_to_log`, improving every `script:` rule in all three workflows, so
carrying it makes R11 not purely a WF3 milestone. Landing it here is cheap and it
is genuinely a WF3 pain point (`check_model_reference`'s empty log is how it was
found). Splitting it keeps the milestone's boundary clean. No strong recommendation.

---

## 9. The experiment freeze — support decision (2026-08-07)

Investigated because Q1 asked whether unit B's rename reaches the frozen
`experiment.yml`. **It does not** — C22's surfaces are filenames and catalog keys
(`cst_<m>.csv`, `rlz_<n>_cst_<m>.*`, the WG-5 catalog keys, `cst_0`→`st_0`), and
the frozen document holds configuration, not member identifiers.

**The collision is real but it is in unit A.** `experiment.yml` records the
resolved `workflows.climate_experiment` section, and CR-2 retires
`aggregate_rlz`, which is a key in that section. Tested against
`test_case/test_local/experiments/experiment` rather than reasoned about:

```
frozen: True
RESULT: RAISES -> ExperimentConfigFrozenError
  changed: ['aggregate_rlz']
```

Every experiment that has already run would fail its next WF3 invocation.

**It is conditional on a decision the register already parked.** The frozen
document is built from the *resolved* config section, so removing `aggregate_rlz`
from the config files makes the key vanish from the document and the freeze
fires; leaving it in as an unread key keeps the document identical and nothing
breaks. That choice is the register's own **open Q7 (removal policy)**, which
does not mention this consequence.

### Ruling — accept the break

`aggregate_rlz` is removed, and an experiment that has run under the old table
shape **cannot continue under the new one**: it is re-run as a new experiment.

This is the freeze working, not a defect to route around. Retiring the flag
changes the table's grain, so the existing results genuinely mean something
different — which is exactly what `check_not_frozen` exists to refuse. Rejected
alternatives: keeping the key as an accepted no-op through R11 (a whole milestone
in which a user's setting silently does nothing — the hazard the register already
names), and teaching the freeze a retired-keys list (new machinery for a problem
that has occurred once).

Consistent with R7's GA-2 and R9's support decision, both of which ruled
pre-existing state unsupported because no production trees exist. That reasoning
was re-checked here rather than inherited, since R9 is what shipped the freezing
mechanism.

**Consequence for the phase plan, which must not be discovered mid-run:** the
fixture's own experiment at `test_case/test_local/experiments/experiment` is
frozen and will refuse. **P1 deletes it before the gate run** — that is a step,
not an accident, and the deletion belongs in P1's brief.

### Recorded, not solved

Any future milestone that adds or removes a `climate_experiment` key hits this
same wall; R11 is simply the first. If it recurs, the freeze wants a
schema-version concept rather than a per-milestone ruling. Carried as a
watch-item with that trigger.

---

## 10. Provenance

Scope confirmed with the owner 2026-08-07 through `design-scoping`. Rulings taken
in that session: milestone boundary (register now, v2 as R12); `[R9-5]` include
the baseline; `[R10-12]` accept and document. The staleness measurement in §1 was
run against `design-v4.md`, not estimated.
