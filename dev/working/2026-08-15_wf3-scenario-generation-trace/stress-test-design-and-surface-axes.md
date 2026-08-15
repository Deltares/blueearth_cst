# Separating the experiment from the response surface

Status: DESIGN NOTE, 2026-08-15. **Not an accepted design and not a task brief.**
It records a design conversation and the owner rulings taken during it, so they
survive as something reviewable. Everything under *Open questions* is undecided.

Companions in this folder: `trace.md` (the run, and its measured cost profile) and
`wf3-rule-reference.md` (every rule, its scripts and file shapes).

---

## 1. The separation this rests on

Two things are currently fused in WF3 and are conceptually distinct:

| | what it is | what it owns |
|---|---|---|
| **The experiment** | perturbed climatology → simulated hydrology → simulated indicators | what was actually imposed, and what the system did in response |
| **The response surface** | a post-processed *view* of those indicators | how a member is summarised into an axis value, and how the plot is labelled |

Today the second is baked into the first: `export_wflow_results.annual_perturbation`
collapses each member's twelve monthly values to one annual figure **at reduction
time**, and writes it into the indicator tables as `temp_change` / `precip_change`.
Any other axis is then unrecoverable from the results alone.

The consequence is not just missing flexibility — see §3, it can make the plot
misreport what the experiment explored.

## 2. Merging the two parameter artifacts

Today the grid is written as two shapes:

- `<wg>/_work/st_<id>.csv` — per member, twelve monthly rows, columns
  `month, temp_mean, precip_mean, precip_variance`. Precip is a **multiplier**.
- `<exp>/config/stress_test_design.csv` — one row per member, columns
  `st_id, temp_change, precip_change, precip_variance_change`. Precip is a
  **percent**, and the values are the **annual collapse** of the monthly ones.

**The second is a materialized cache of the first.** `prepare_cst_parameters.py:175`
writes the member CSV, reads it back off disk, and calls the same
`annual_perturbation` the reduction later calls, to build the design row. HM-7 says
so plainly — those axis columns are *"a cached copy, derived independently by the
writer, so they really can drift"* — and `validate_hm7` exists to police exactly
that gap.

**Proposal (owner, 2026-08-15): one long lookup table** at monthly grain, keyed by
member:

```
st_id, month, temp_change, precip_change, precip_variance_change
1,     1,     …
1,     2,     …
…             (twelve rows per member)
2,     1,     …
```

What it buys, in order of importance:

1. **The monthly detail survives to post-processing**, which is what makes §3
   possible at all.
2. **It removes a cache and one of its two derivations**, and with them the
   drift class `validate_hm7` currently guards.
3. **A new perturbation parameter is a column**, not a new file shape. (Caveat:
   this removes the *shape* barrier to a third axis, not the *contract* barrier —
   C28 refuses one deliberately because a new dimension must reach the design
   table and the results columns together.)
4. `_work/` disappears entirely. The merged table belongs in `<exp>/config/`
   beside the config snapshot: it is a record of what ran, not scratch.

### Units — RULED

> **Ruling (owner, 2026-08-15): percent, everywhere.** `temp_change` in °C,
> `precip_change` and `precip_variance_change` in **percent**. Column names stay
> `temp_change` / `precip_change` rather than unit-suffixed variants
> (`precip_change_pct`), by owner preference.

The criterion was consistency across artifacts, not internal convenience, and the
tally settles it — WF2 **already emits percent**, with an explicit `relative_units`
column carrying `%` for precip and `degC` for temp:

| artifact | precip convention |
|---|---|
| WF2 change factors | percent |
| WF3 `stress_test_design.csv` | percent |
| WF3 indicator tables (`precip_change`) | percent |
| WF3 member files `st_*.csv` | **multiplier** — the sole outlier |

Three of four already agree, and the two that matter most cannot diverge: HM-7
requires the stress-test axes to match WF2's definition *because the GCM dots are
overlaid on them*. Percent makes the imposed change, the reported axis and the
projection factor one quantity in one unit.

The multiplier survives only as the generator's operation form —
`impose_climate_change.R` converts `1 + p/100` once, at the point of application.

**An argument considered and withdrawn.** "Store what is applied (factors), because
a rounding error there changes the science" does not survive scrutiny. The incident
that seemed to support it — `float32(0.7)` → `−30.000001%`, which is why
`prepare_cst_parameters.py:174` computes the design row from the persisted CSV —
was a **float32-vs-float64 CSV round-trip** problem, not a unit-choice one. Percent
stored at adequate precision has the same property: every reader converts
deterministically to the same factor.

Consequence worth knowing: no-change becomes `0.0` rather than `1.0`, so a row of
zeros now visibly means "no perturbation" and `st_0`'s row reads as the origin.

**Invalidation is not a reason against it.** Rule 3.12 declares one member file per
job today, but rule 3.09 writes *all* member files in a single job, so any config
change rewrites all of them and re-fires every 3.12 job anyway. The per-file split
buys no invalidation granularity.

## 3. Why the axis has to be a parameter

Stress testing here is **extended sensitivity analysis**: the question is how the
system responds across an explored range. The axis must therefore report *the range
that was explored*.

The fixed annual collapse fails that for any seasonal design. Perturb JJA by +30%
and leave the rest unchanged, and the month-length-weighted annual figure is

```
(92 × 1.30 + 273 × 1.00) / 365 = 1.076  →  +7.6%
```

so the axis reads **+7.6%** for a member that imposed **+30% in the wet season**.
The response came from a concentrated seasonal change; the label describes a mild
uniform one. The more concentrated the perturbation, the worse it gets — perturb a
single month and the entire explored range compresses to roughly a twelfth of its
true magnitude.

So for a seasonal design the axis definition is a **correctness requirement**, not a
presentation preference.

**Shape.** The axis becomes a user-declared triple applied at post-processing —
variable, month set, statistic — e.g. `{variable: precip, months: [6,7,8],
statistic: mean}`.

**Two constraints to carry:**

- **Linear statistics only, or the grid guarantee breaks.** HM-7 lets consumers
  rely on the axis being evenly spaced: "the collapse is affine in the member's
  step index, so the surface is rectilinear." Any mean over any month subset
  preserves that. A non-linear statistic — a max, a quantile — does not, and the
  surface stops being a regular grid.
- **The same collapse must be applied to the projection overlay.** HM-7 already
  records why: the CMIP6 dots are placed on these axes, and "two different
  collapses would compare two different quantities." (Overlay treatment is
  explicitly deferred — see Open questions.)

### The lookup is the source of truth — RULED

> **Ruling (owner, 2026-08-15).** The lookup table is the **source of truth**. From
> the lookup plus the results, any response surface can be generated — within the
> logical bounds of §4. Nothing derived from it is stored.

The lookup holds all twelve months for every member, which makes it a **sufficient
statistic** for any collapse: annual, a season, a single month, even a non-linear
one. Every axis is a projection of it.

Three consequences, and the third corrects an earlier draft of this note:

1. **Indicator tables carry `st_id` and `value`, not a baked axis.** Keeping
   `temp_change` / `precip_change` there would privilege one collapse and
   re-create the drift `validate_hm7` polices. It also spares no one a join —
   consumers need the lookup regardless, so it *is* the second file.
2. **A surface is a declaration plus a figure**, not a directory of data: the
   collapse, the caption and the exclusions (`st_0`) are a *choice* and must be
   recorded; the axis values are a derivation and must not.
3. **An earlier proposal to materialize a per-surface `axes.csv` is withdrawn.**
   It reintroduced, one layer up, exactly the cache-of-a-derivation this design
   removes. Caught by the owner: given the lookup, an axis table stores something
   already fully determined.

The general principle, which is the same one that killed the design-table cache:
**store the finest grain that was actually imposed; derive every summary.**

The one case for materializing, recorded so it stays a decision rather than an
oversight: archiving a *published* figure, where the exact plotted numbers should
sit beside it rather than be recomputed years later from code that has moved. That
is a publication-provenance concern, better served by an export-on-demand than by
writing a file every run.

## 4. The three interpretable designs

Owner's taxonomy, 2026-08-15. All three are **already expressible** in the current
config; no mechanism change is needed on the design side. In config terms precip is
a multiplier, so "no change" is `1.0`:

| case | `min` | `max` | axis caption |
|---|---|---|---|
| 1 — uniform | `[0.7]×12` | `[1.3]×12` | mean change in precipitation |
| 2 — some months vary, rest unchanged | `[0.7,0.7,0.7, 1.0×9]` | `[1.3,1.3,1.3, 1.0×9]` | mean change over JFM; Apr–Dec unchanged |
| 3 — some months vary, rest held at an offset | `[0.7,0.7,0.7, 0.8×9]` | `[1.3,1.3,1.3, 0.8×9]` | mean change over JFM; Apr–Dec held at −20% |

### The criterion underneath the taxonomy

Members are built by `np.linspace(min_vector, max_vector, …)`, so member *j* is
`min + (j/n)(max − min)` month by month — a one-parameter family. A scalar axis
therefore always exists mathematically, even for a design where Jan swings ±30%
while Feb swings 0→+50%.

It is only **interpretable** when every varying month shares the same `min` and
`max`. Then the mean over the varying months *equals the change applied to each of
them*, so the axis reports the imposed value rather than an average of unlike
things. The three cases above are exactly that family.

**This is checkable, and could be a validation:** warn when varying months carry
differing `(min, max)` pairs, because the axis then averages dissimilar
perturbations and no caption can describe it honestly.

### The caption should be derived, not typed

From a per-member × per-month table you can read off the **varying months** (value
differs across `st_id`) and the **held months and their level** (constant across
`st_id`). That yields the captions in the table above mechanically. A typed label
can drift from the design it describes; a derived one cannot — and this is the
strongest argument for the merged table beyond simplification.

### A consequence for `step_num`

A no-change member exists only when the no-change value lands on a level. For
`min 0.7, max 1.3` that requires an **even `step_num`**. The shipped rapid config
uses `step_num: 1`, whose levels are 0.7 and 1.3 with nothing at 1.0 — so an even
`step_num` is a real modelling choice (it puts the origin inside the design), not
an arbitrary number.

## 5. `st_0`, and a duplication

> **Ruling (owner, 2026-08-15).** `st_0` is **not a member of the response
> surface**. It exists to give users information about the baseline — no climate
> change, stochastic realizations — and is reported as an annotated reference
> value beside the surface rather than plotted as a grid node.

Two things follow.

**It stays simulated.** Two of the eleven `q` metrics are derived *from* `st_0`, so
`run_historical: true` is load-bearing; `false` drops them with nothing reporting
it. Excluding it from the surface is a presentation filter (`st_id != 0`) and costs
nothing structurally — the indicator tables already carry `st_id`.

**It amends a recorded rationale.** `prepare_cst_parameters.py:117` justifies the
`st_0` design row the other way round — *"a response surface missing its own origin
forces every downstream consumer to reconstruct it"* — i.e. C23 assumed `st_0` **is**
the surface origin. The row stays; the stated reason for it changes, and the comment
should say so rather than continue asserting the old intent.

### The duplication

When a grid member's perturbation is the identity in **every** month (temp +0,
precip ×1.0), that member *is* `st_0` — the same scenario simulated twice. It
happens in cases 1 and 2 with an even `step_num`, and **never in case 3**, where the
zero-axis member still holds Apr–Dec at −20%.

Scale, stated so it is not over-valued: the duplicate is exactly **one member
regardless of grid size**, so the saving is `1/(ST_NUM+1)` — 10% on a 3×3, 4% on a
5×5, and zero on the shipped rapid config, which has no identity member at all.
This is a design-cleanliness argument more than an efficiency one.

> **Ruling (owner, 2026-08-15): option A — alias the result, keep `st_id` dense.**
> Do not simulate the identity member; reuse `st_0`'s result for it. The rejected
> alternative was letting `st_0` occupy that grid slot, which removes the duplicate
> entirely but leaves a hole in the `st_id` enumeration.

Two obligations this creates:

1. **Verify the premise first — it is a precondition, not a nice-to-have.** Option A
   *copies* `st_0`'s result into another member's slot, so if the two are not truly
   the same scenario, it fabricates a result rather than reusing one. `× 1.0` and
   `+ 0.0` are exact in floating point and a ramped zero perturbation is still zero,
   so the values *should* match — but `impose_climate_change.R` round-trips the
   netCDF through R's writer, so this is a claim to test, not assume. Cheap test:
   an even-`step_num` config, `--notemp`, compare the identity member's forcing and
   output CSV against `st_0`'s. A mismatch is itself a finding.
2. **Mark the alias.** A duplicated result that looks simulated is the kind of thing
   that reads as a defect months later. It needs to be visible — an `alias_of`
   column in the lookup, or a line in the run metadata — rather than two identical
   CSVs with nothing explaining why.

## 6. Open questions

- **Does anything outside this repo read `stress_test_design.csv` in its current
  wide shape?** HM-7 names the CST-API/GUI as the consumer of the *indicator*
  tables; if the GUI also reads the design table to label a surface, wide→long is a
  breaking API change rather than an internal simplification. **Unchecked.**
- **Naming.** `stress_test_design.csv` is a contract-named artifact. Does the merged
  long table keep that name (same identity, different shape) or take a new one with
  a migration note?
- **Multiple surfaces per experiment.** Within one experiment all members lie on the
  line from `min` to `max`, so every linear axis is an affine image of every other:
  two surfaces from one experiment differ in magnitude and label, not in shape or
  member ordering. Genuinely different response shapes would need members that vary
  seasonal *pattern* independently — a second design dimension, which collides with
  C28's deliberate two-axis refusal. Not proposed here.
- **The projection overlay.** Deliberately deferred by the owner until the above is
  settled. The constraint is already recorded (HM-7): whatever collapse the surface
  uses must be applied to the CMIP6 change factors too.

## 7. Where this would land

Not here. R12 owns *how WF3 executes* (`dev/roadmap.md` § Phase 8) and
`t2608082036` is its open design item; the reduction and reporting side sits
adjacent to it. Anything from this note that becomes work should be admitted to the
board on its own terms, with `trace.md` § 3 as the cost baseline.
