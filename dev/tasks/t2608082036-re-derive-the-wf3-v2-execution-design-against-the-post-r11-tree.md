---
title: Re-derive the WF3 v2 execution design against the post-R11 tree
type: todo-item
status: backlog
effort: 2
area: wf3 execution
origin: R12
queue: 1
created: 2026-08-08
updated: 2026-08-16
---

> [!note] Overview
> **What** — Rewrite the ratified wf3-experiment-v2 architecture -- manifest, ledger, member_hash, resumable sweeps, epochs, quarantine, checked publication, the AB/BA timing gate -- against the tree R9, R10 and R11 left behind, and produce a written mapping from each of the run's 65 findings to its post-R11 expression or to a reason it lapsed.
> **Why** — Without the mapping, two external review rounds and an owner arbitration are archived rather than usable, and R12 either re-litigates settled questions or silently inherits a data model built on aggregate_rlz -- a config key that is now a parse-time error.
> **Effort** — large

## Blocked on the lookup redesign — ruled 2026-08-15

> **Ruling (owner).** `t2608152230` lands first; this re-derivation follows and
> defines `member_hash` over the **monthly lookup rows** rather than the annual
> scalars.
>
> **Updated 2026-08-15:** the lookup's design is now **ACCEPTED** —
> `dev/milestones/r12/stress-test-lookup-design.md`. Read **D1–D2** for the
> schema this re-derivation must key on, and **§5.7 (WG-2)** for its normative
> definition — the schema lives on the *weather-generator* seam, not HM-7, by
> owner ruling. Two things there change what this item inherits: **D35** imposes
> a parse-time admissible multiplier domain (`≥ 0.5`), and the lookup carries
> **no `st_0` row** (D4), so a member-identity scheme cannot derive the baseline's
> identity from it.

`design-v4.md` § 5.1 defines

```
member_hash = sha256({member_id, rlz, cst, baseline, seed_r,
                      weagen_template_digest, st_params_digest,
                      tavg, prcp, precip_variance, run_config_digest})
```

and its field note calls `tavg` / `prcp` / `precip_variance` *"the annual scalars
the response surface is indexed by, derived exactly as the reduction derives them
today."* **`t2608152230` abolishes that derivation** — the annual collapse becomes
a declared post-processing parameter, and the indicator tables stop carrying it.
So the member-level freshness boundary is currently defined over an artifact that
is about to be deleted. Verified at current HEAD, not from the doc's pre-R9 line
citation: `perturbation_axes` → `annual_perturbation`,
`blueearth_cst/experiment/export_wflow_results.py:300-318`.

Re-deriving the identity scheme is this item's *declared first task* and the
review record lists it among what does **not** survive — so doing it against the
old artifact spends the work twice. The replacement is also strictly more
faithful: a digest over the member's twelve lookup rows, versus a collapse that
misreports a seasonal design by construction.

**Three findings from the same reading, so they are not re-derived:**

- **`st_params_digest` transfers verbatim.** It keys on the *config section*
  rather than the member files because rule 3.01 runs before those files exist.
  One lookup table instead of twelve member files does not change that ordering.
- **`ancient()` on rule 3.09 is solved in the design this item inherits** —
  `file_digest_or_absent(...)` threaded through `params:` so a content-only
  change re-triggers past `ancient()`. The critique in the lookup design's §5d is
  a *solved-elsewhere* item, not an open one.
- **§6.6's `baseline: true` branch is `t2608151154`'s subject**, encoded
  structurally: `st_csv: null`, no perturb step, downscale straight from
  `baseline_nc`. The v2 design formalized st_0's separate production path without
  recognising it as a comparability problem.

Falsified and recorded so it is not re-run: `st_0` and the grid's identity member
do **not** collide under `member_hash` — `member_id` and `baseline` are both terms
in the tuple.

Adjacent, same terms, not resolved here: `precip_variance` sits in the hash under
the G1 retention ruling with `R9-F1` as its named followup.

## Progress

- [ ] <first step>
