# Falsifier — step 6c, remove cross-combination statistics (R3′/R3″)

```
Written: 2026-07-31, BEFORE any 6c code
Design:  §8 row 6c; rulings R3′ / R3″; N10
Ref:     test_case/ref_wf2_pre_5f_fixed (+ 6a/6b additive files)
```

## What 6c claims

Under **R3′/R3″** each `(model, scenario, member)` is **one data point**. Nothing
in v2.0 aggregates across combinations: no member averaging, no model-level
collapse, no percentile envelope, no ±σ, no min–max. The two anomaly figures
become **one trace per combination**, and composition reporting replaces the
ensemble summary.

Today `plot_proj_timeseries.py` computes `.quantile([0.05, 0.5, 0.95], axis=1)`
across the model axis — sixteen times — and draws a 5–95 % band with a line
labelled "multi-model median". That label is the clearest statement of what R3′
removes: a median *across models* is a claim about an ensemble the design does not
make.

§8: **value-changing for the two anomaly figures**; the three `summary/*` targets
are unaffected because they carry no cross-model statistic.

## O1 — no reduction across the combination axis may remain

The structural falsifier, and the one that does not depend on looking at a picture.

**Falsified if** any `quantile(..., axis=1)`, `mean(axis=1)`, `std(axis=1)`,
`min(axis=1)` or `max(axis=1)` over the model/combination axis survives in the
plotting path.

Checked by grep, deliberately: a visual gate cannot distinguish a removed envelope
from one drawn in white.

## O2 — one trace per combination, and the count must be right

**Falsified if** the number of plotted traces is not the number of resolved
combinations. On the seed that is **6** per figure (3 models × 2 scenarios),
against today's 1 median line plus 1 band per scenario group.

## O3 — the "multi-model median" label must be gone

**Falsified if** any legend entry still claims a multi-model quantity. The label
is the design's own example of the claim R3′ withdraws, and leaving it while
changing the data underneath would be worse than leaving both.

## O4 — the two anomaly PNGs must change; the third must not

**Falsified if** `precipitation_anomaly_projections_abs.png` and
`temperature_anomaly_projections_abs.png` are byte-identical after the change —
they are the figures being redrawn.

**Falsified (contamination) if** `projected_climate_statistics.png` changes: the
ΔT/ΔP cloud is already one point per combination and carries no cross-model
statistic, so 6c must not touch it.

**Falsified if** any `summary/*`, `change_factors/*` or `series/*` value moves.
6c is a figure change; it computes no new number.

## O5 — the manifest must move for exactly the two PNGs

The three pinned PNGs are manifest targets (by size). **Falsified if**
`check_baseline` reports anything other than the two anomaly figures differing.

That is a sharper gate than it looks: PNG size is a crude fingerprint, and two
figures changing while a third does not is a stronger signal than any of them
changing alone.

## O6 — the figures must remain legible with 6 traces, and say what they show

Not machine-checkable, and stated anyway so it is a decision rather than an
oversight: with one trace per combination the reader needs to know **which** trace
is which. A legend naming every combination is the honest form; a legend naming
none is how "one trace per combination" becomes "a hairball".

Recorded as a limitation if not achieved, never as a silent outcome.

## Order of work

1. Replace the quantile computations with per-combination traces via one helper,
   so "one trace per combination" is a single testable function rather than eight
   near-identical blocks.
2. O1 by grep; O2/O3 by unit test on the helper.
3. Run; O4/O5 by diff and manifest; re-record the two PNGs.

---

## Outcome — 2026-07-31

| | Result |
|---|---|
| O1 | **0** cross-combination quantiles, **0** `fill_between` envelopes remain |
| O2 | 9 `plot_combination_traces` call sites; one trace per column, labelled |
| O3 | **0** "multi-model median" labels outside the docstring |
| O4 | 6 of 8 figures FAIL the tree diff; `projected_climate_statistics.png` **not regenerated** — contamination clause holds |
| O5 | see below — the gate is weaker than the falsifier assumed |
| O6 | every trace carries its combination label |

### O5 exposed a gate weakness, not a code defect

Two regenerated figures compared as *unchanged*. Checked directly rather than
believed:

```
precipitation_anomaly_projections_abs.png  609649 vs 562849  sha DIFFERS   8.3% drift -> PASSED
temperature_anomaly_projections_abs.png    386751 vs 324309  sha DIFFERS  19.3% drift -> failed
```

Both were redrawn; both differ by sha256. `check_baseline` fingerprints PNGs
**by size with `PNG_TOLERANCE_FRAC = 0.10`**, so a figure whose content changed
completely passes whenever its compressed size lands within 10 %.

That is not a 6c defect — it is the gate's sensitivity, and it was already true
for every PNG this milestone has re-recorded. It matters now because **the seal
will lean on `check_baseline`**, and "3 PNGs pinned" reads as stronger coverage
than "3 PNG file sizes, ±10 %".

Recorded rather than worked around: changing the comparator is a gate change and
belongs to its own decision, not to a figure commit.

**Practical consequence for 6c's own verification:** O4 was discharged by sha256
comparison and by the mtime evidence that `projected_climate_statistics.png` was
never regenerated — not by the tree diff, which shares the same size-based
comparator for PNGs.
