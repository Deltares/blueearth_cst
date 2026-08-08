# R12 — G2 gate assessment

Prepared 2026-08-08, immediately after R11 sealed. **G2 is an owner gate; this
document prepares it and recommends, it does not decide.**

Source: `docs/wf3-redesign` (branch head `531bcc6`), design run
`dev/working/design-runs/wf3-experiment-v2/`.

---

## What G2 was left holding

From the run's `status.md`, the final stage entry:

    [open] G2 — presenting: approve design-v4; ratify risk-7 part-3
    rejection; sv-minor handling; body budget (1903 -> 3420 lines).

So four items, and the run is otherwise **complete**:

| stage | outcome |
| --- | --- |
| G1 | approved 2026-08-01 |
| external rounds | 2 of 2 (the cap), both clean-room via `codex exec`, read-only intent verified |
| convergence | NOT CONVERGED both rounds → round-cap arbitration |
| arbitration | owner ruled 2026-08-01, all seven ext2 findings ACCEPTED, fix required |
| 6a revision | design-v4 (3420 lines), ledger 65 rows, driver scope check PASSED |
| scoped verification | **PASS** — 0 blocking, 0 major, 7 minor (sv-1..sv-7) |

**The process is not the problem.** Nothing about this run was left half-done:
65 unique findings, 9 re-raised and each chased, two external rounds, an
arbitration with explicit rulings, and a verification pass that came back clean.
Whatever G2 decides, it should not be framed as rescuing an unfinished design.

Two flags are outstanding by construction:

- `rejected-major-part-pending-G2` — risk-7 part 3 (a data-derived wall-clock
  ceiling) was rejected by the author with rationale; parts 1–2 accepted. The
  status file adds the sharp note: **`ext1-2` faults exactly those accepted
  parts.** That is the specific ratification G2 owes.
- `fable-escalation-r2` — a tier trigger, already discharged.

---

## What changed underneath it

The run ran **2026-08-01 → 04**, against the pre-R9 tree. Since then R9 moved the
project tree, R10 renamed rules, and R11 changed what WF3 emits and what its
members are called. `main` is 289 commits ahead of the branch.

The roadmap records "~74 references to artifacts and rules that no longer exist",
**measured 2026-08-07 — before R11 P2 and P3 landed.** Re-measured today against
`design-v4.md`:

| stale identifier | hits | invalidated by |
| --- | --- | --- |
| `cst_` | **98** | R11 P2 — member token is `st_` |
| `Qstats` | 25 | R9 — now `q_indicators.csv` |
| `aggregate_rlz` | **15** | R11 P1 — **retired as a HARD ERROR** |
| `RT_` | 15 | R9 — tables deleted, nothing replaces them |
| `export_wflow_results` | 14 | R9 — rule is `derive_wflow_indicators` |
| `hydrology_runs` | 11 | R9 — tree move |
| `stress_test_design` | **0** | R11 P2 added it; the design has no concept of it |

~178, not ~74. But the count is the least of it.

### `aggregate_rlz` is load-bearing, not cosmetic

This is the finding that decides the gate. The design does not merely *mention*
the retired key — its two most central concepts branch on it:

    :704  | `member_id` | str | `rlz_<n>/cst_<m>`; `"agg/cst_<m>"` when
                                `aggregate_rlz: true` |

    :683  | Cell completeness | Under `aggregate_rlz: true`, a `cst_m` is
             complete iff all `RLZ_NUM` of its members are `succeeded`. Under
             `aggregate_rlz: false`, completeness is per member |

    :664  **`allow_partial` under `aggregate_rlz` — the shape a hole actually
          takes.**

A manifest-and-ledger architecture is, at its core, *an identity scheme for
members plus a predicate for when one is done*. Both of those are defined here in
terms of a config key that R11 made a parse-time error, on the reasoning that in
the long table shape "aggregated" was never a shape choice at all. There is no
mechanical substitution available: the key did not get renamed, the distinction
it expressed was dissolved.

### The member set moved, and a named gate falsifier now expects the wrong result

**Correction to a first reading of this, made while writing the record.** It is
tempting to say the design assumes `K = 12` and knows nothing of the baseline.
That is not true and the record should not say it. `ext2-1` raised the missing
baseline branch, it was accepted, and the 6a revision handled it properly —
`GF-21` is a whole baseline-member lifecycle falsifier, and the scoped
verification confirmed the arithmetic *"`ST_START=0` ⇒ K = 2 × 7 = 14 as
stated"*. The design is not ignorant here.

What is stale is narrower and sharper. `design-v4.md:2183-2186`:

    test_case/test_local, **as tracked**: `K = 12`, no `cst_0` members,
    … `K = 14` arithmetic assumed `run_historical: true`, which no tracked
    config sets,

**R11 P3 made that sentence false.** The tracked seed config now sets
`run_historical: true`, because leaving it false silently cost two of eleven
metrics. So `GF-21`'s elaborate apparatus — copy the config to
`dev/tmp/gf21_config.yml`, flip `run_historical`, use a distinct
`experiment_name` to avoid touching the tracked fixture — is now unnecessary
scaffolding around what the tracked config does by default.

And worse than unnecessary, its expected observation is now **inverted**. GF-21
asserts that the reduce *"under `aggregate_rlz: true`, emits **no `cst_0` row**
(the inherited §4.4 asymmetry, observed as specified)"*. R11 removed that
asymmetry on `[R9-5]`'s ruling that the baseline is a member of the surface:
`st_0` rows **are** emitted, and `aggregate_rlz` no longer exists to condition
anything. A falsifier that passes only when the implementation reproduces
behaviour R11 deliberately removed is worse than a missing falsifier.

`design-v4.md:237` lists as an open question *"whether the `cst_0` asymmetry in
the reduction is desirable"*. **R11 answered it — no.** That is a design question
closed by evidence rather than by argument, which is the good outcome; but it
means the surrounding mechanics were built around the other answer.

---

## Assessment

Separate the two things G2 is being asked to approve, because they have different
shelf lives:

**The findings and the architecture survive.** Manifest, ledger, `member_hash`,
resumable sweeps, epochs, quarantine, checked atomic publication, the
counterbalanced timing protocol from ext2-7 — none of these depend on what a
member file is called or on which config key selects a grain. Two external review
rounds and 65 findings are expensive to reproduce and should not be discarded.

**The mechanics do not survive.** Paths, rule names, artifact names, the member
identity scheme, the completeness predicate, and K. Re-deriving them is not a
find-and-replace: `aggregate_rlz` has no successor, and `st_id` plus
`stress_test_design.csv` are new axes the design never saw.

So the honest position is that **design-v4 is a sound design for a tree that no
longer exists.** Ratifying it as an implementable spec would import a data model
built on a retired key. Discarding it would throw away two external review rounds
over a naming problem.

---

## Recommendation

**Ratify the run, seal the document, re-derive the mechanics in R12.** Concretely:

1. **Approve design-v4 as an architectural input, not an implementable spec** —
   which is what the roadmap already says R12 should treat it as. G2 closes.
2. **Ratify the risk-7 part-3 rejection.** It concerns a data-derived wall-clock
   ceiling and is independent of everything R9/R10/R11 moved; leaving it pending
   keeps a flag open on a run that is otherwise finished. `ext1-2`'s objection to
   the accepted parts 1–2 is on the record and travels with the seal.
3. **Take sv-1..sv-7 as author's discretion** (their stated disposition) —
   re-deriving the mechanics supersedes minor editorial findings anyway.
4. **Write a consolidated review record, do not seal and do not merge.**
   *(Revised from this document's first draft, which proposed a
   `sealed-records.yml` entry. That was wrong on the mechanics and wrong on the
   precedent.)*

   Wrong on mechanics: `tests/test_sealed_records.py` asserts
   `(REPO / record["path"]).is_file()`, so the registry can only hold documents
   **in the tree**. The run lives on `docs/wf3-redesign`; registering it would
   require merging 14,855 lines onto `main` first, which is the opposite of what
   sealing is for.

   Wrong on precedent: **WF2's design run already solved this**, and the answer
   is neither seal nor merge. `dev/reference/workflows/wf2-climate-analysis-v2-design-review-record.md`
   states it outright — *"This record is the durable artifact; the per-round
   scratch (`design-v1.md` … `design-v4.md`, `status.md`, the briefs and
   transcripts) lives in git history"* — with the commits naming each verbatim
   round. The record is a **current** document, so it needs no banner and no seal;
   the scratch stays where scratch belongs.

   Do the same for WF3: `dev/reference/workflows/wf3-experiment-v2-design-review-record.md`.
   The one deliberate difference from WF2 is that WF2's run promoted an ACCEPTED
   design into `dev/reference/workflows/`; WF3's does not, because the mechanics
   do not survive. The record carries the audit trail and the surviving findings;
   R12 produces the design.
5. **R12's first task is the re-derivation**, with a written mapping from each
   surviving finding to its post-R11 expression. That mapping is the deliverable
   that makes the 65 findings portable.

**On the body budget (1903 → 3420 lines):** worth noting rather than accepting
silently. The document nearly doubled across revisions while its subject was
being invalidated underneath it. If R12 re-derives, the re-derivation should be
allowed to be *shorter* than v4, not treated as needing to carry every line
forward.

---

## What this does not decide

- Whether R12 adopts the v2 execution model **at all**. That is R12's scoping
  question, not G2's. G2 only decides whether this design run is closed and what
  status its output carries.
- Whether `docs/wf3-redesign` merges. It should not: sealing preserves it, and
  merging 14,855 lines of superseded design into `main` is the opposite of what
  the seal convention exists for. It stays a branch, cited by path.
