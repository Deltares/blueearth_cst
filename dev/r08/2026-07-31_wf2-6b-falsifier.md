# Falsifier — step 6b, the dry-month rule (A2)

```
Written: 2026-07-31, BEFORE any 6b code
Design:  §5.6 dry-month rule (risk-05, ext1-05), ruling A2 closing OQ-9
Ref:     test_case/ref_wf2_pre_5f_fixed (+ 6a's three additive files)
```

## What 6b claims

A relative change is undefined-in-practice when the reference is near zero — the
delta-method failure the annual product largely avoids and **the monthly product
walks straight into on any basin with a dry season**, which 6a-ii has just made
reachable.

* `relative_change.min_reference`, per variable, in canonical units. Applies to
  `change: relative` variables only. Default **`precip: 0.1 mm/day`** (≈3 mm/month).
* Comparison is **strict**: flagged when `σ(yₘ(R)) < min_reference`; a reference
  exactly at the threshold is **not** flagged.
* Flagged ⇒ `value = NaN`, `status = "reference_below_threshold"`, and the
  **absolute** change in `absolute_value`, so the information is not lost.
* `relative_change.max_flagged_months`, default **3**, strict (`> 3` flags the
  combination, exactly 3 does not).
* A user-configured relative variable beyond the shipped set has **no default** —
  DAG-build validation raises.

## N1 — the threshold comparison must be strict, on both sides

**Falsified if** a reference exactly equal to `min_reference` is flagged, or one
just below is not. The design says §9 tests the boundary on both sides; this is
that test. Off-by-one on a threshold is invisible on any basin that never sits
near it, and the seed is wet year-round.

## N2 — flagged months must LOSE the ratio and KEEP the difference

**Falsified if** a flagged month emits a finite `value`, or leaves
`absolute_value` empty. The whole point is that the ratio is meaningless while the
difference still carries information — dropping both would be a worse answer than
the infinity it replaces.

## N3 — the rule must not touch absolute variables

**Falsified if** `temp` is ever flagged. It is `change: absolute`; a near-zero
reference temperature is a perfectly ordinary 0 °C, and flagging it would be the
name-based inference 5e-iii removed, reintroduced through a threshold.

## N4 — a relative variable with no threshold must RAISE at DAG build

**Falsified if** a config declaring `change: relative` for a variable with no
configured `min_reference` runs. Silently defaulting to `precip`'s 0.1 would apply
a precipitation threshold to an unrelated quantity in unrelated units.

**Also falsified if** it raises at run time — same argument as 5e-i's `save_grids`.

## N5 — `max_flagged_months` is a combination-level signal, also strict

**Falsified if** exactly 3 flagged months flags the combination, or 4 does not.
A basin with a genuine dry season produces about a season of structurally flagged
months as its *normal* state; the signal is for "over a quarter of the year is
undefined", not for "this basin has a dry season".

## N6 — the seed must be unaffected, and that must be shown, not assumed

The seed basin is equatorial and wet year-round. **Falsified if** any seed month
is flagged — that would mean the threshold is mis-scaled or applied to the wrong
quantity.

So 6b is **value-neutral on this fixture** despite being value-changing by
classification, and every behaviour above is unit-test territory. The tree diff
can only confirm the absence of an effect.

**Falsified (opposite direction) if** the `status` column stays literally `"ok"`
because the rule was never invoked — absence of flags must come from the data, not
from the code path being dead. Verified by asserting the threshold reached the
computation, not by observing no flags.

## N7 — `flagged_months` in `provenance.json` must be populated from the same pass

6a-iii left the key present and empty. **Falsified if** it is filled by a second
traversal that could disagree with the table — the fourth instance of that drift
is one too many.

## Order of work

1. `dry_month.py` + N1–N5 unit tests; no fixture.
2. Wire into the monthly computation; carry `absolute_value` and `status`.
3. DAG-build validation for N4.
4. Run; N6 both directions; `flagged_months` from the same pass.

---

## Outcome — 2026-07-31, all seven discharged

| | Result |
|---|---|
| N1 | strict on both sides: 0.099 flags, 0.100 does not, 0.101 does not |
| N2 | flagged month: `value = NaN`, `absolute_value = 0.45`, status set |
| N3 | `temp` never flagged; no companions emitted for absolute variables |
| N4 | raises at **DAG build**, naming the variable and refusing to borrow precip's default |
| N5 | strict: 3 does not flag the combination, 4 does |
| N6 | **both directions** — 432 rows all `ok`, and all 216 precip rows carry `absolute_value`, so the rule ran |
| N7 | `flagged_months` counted from the rows the table wrote |

15 unit tests. None of this is reachable through the fixture: the seed basin is
equatorial and wet year-round.

### N6's second clause is the one that mattered

"No months flagged" is the correct result here and is also exactly what a dead
code path produces. The falsifier required proving the threshold *reached the
computation*, not merely that nothing was flagged — verified by the companion
columns being populated on all 216 precip rows. Without that clause, a rule that
never ran would have passed.

### N7 and the fifth chance to disagree

`flagged_months` is counted from the monthly rows already written, not by a second
traversal of the datasets. A value recorded in two places has disagreed four times
in this milestone; this was the fifth opportunity and it was closed by
construction rather than by checking afterwards.
