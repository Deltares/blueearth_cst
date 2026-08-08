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

### The member set itself moved, and R11 proved it can move silently

    :62   Therefore **K = 2 × 6 = 12 members, none of them `cst_0`**

Post-R11 P3 the fixture runs **14 members, including `st_0`**. Worse for the
design's assumptions, R11 P3 established that *whether the baseline member exists
at all* is toggled by `run_historical` / `ST_START`, and that it had been
silently absent — taking two of eleven metrics with it.

This bears directly on `ext2-1`, the external finding that there was **no `cst_0`
branch in the fused lifecycle**. That was accepted and fixed in v4 — but it was
fixed as *a special case for an asymmetric baseline*. R11 went the other way and
made `st_0` an ordinary member of the response surface (`[R9-5]`), whose presence
is conditional on config. The v4 fix is not wrong so much as aimed at a shape that
no longer exists.

`design-v4.md:237` even lists as an open question *"whether the `cst_0` asymmetry
in the reduction is desirable"*. **R11 answered it** — the asymmetry is gone.

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
4. **Seal `design-v4.md` and the run directory** under
   `dev/reference/sealed-records.yml`. This is squarely the seal case the registry
   describes: valuable *because* unedited, and actively harmful to freshen —
   migrating 98 `cst_` references would produce a document that looks current
   while its architecture still assumes a retired config key. The R11 close sealed
   two documents for exactly this reason.
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
