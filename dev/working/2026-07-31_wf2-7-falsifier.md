# Falsifier — step 7, the report stage

```
Written: 2026-07-31, BEFORE any step-7 code
Design:  §5.9 (report artifacts + disclaimer block), §8 row 7
Ref:     test_case/ref_wf2_pre_5f_fixed (+ 6a/6b/6c changes)
```

Three commits:

| | Change |
|---|---|
| i | Declare every figure; fix the `timeseries_csv` mislabel |
| ii | `report.md` with the disclaimer block |
| iii | Declare the optional gridded branch (R2/R5, D11) |

## P1 — every written figure must be declared

Measured: rule 2.06 declares **3** outputs and writes **9** files — eight PNGs and
`gcm_timeseries.nc`.

**Falsified if** any file the job writes is undeclared. An undeclared output is
invisible to Snakemake: it is not cleaned on failure, not re-made when deleted,
and not seen by any consumer that depends on it.

## P2 — `gcm_timeseries.nc` must stop being called `timeseries_csv`

**Falsified if** the label survives. It is a netCDF; the name says CSV. A wrong
name on a declared output is worse than an undeclared one, because it reads as
deliberate.

## P3 — `report.md` must carry the whole disclaimer block

§5.9 requires: requested vs effective reference window and whether clipped; the
alignment result against `shared.historical_window`; the effective window length
and any short-window warning; the spatial weighting scheme **and its approximation
label**; the dry-month rule and threshold; the catalog snapshot date; and the
count of requested-but-unresolved combinations by status.

**Falsified if** any is missing. Every one already exists in a durable record —
`provenance.json`, `composition.csv`, the stage-B log — so the report **reads
them**. A recomputed disclaimer is a sixth chance to disagree, and five have
already been taken this milestone.

## P4 — the report must state what it cannot show

The seed has no clipped window, no short window, no flagged months and no
unresolved combinations. A disclaimer block that renders nothing under those
conditions is indistinguishable from one that is broken.

**Falsified if** the absence of a condition produces silence rather than an
explicit statement of the negative — "no months flagged (threshold 0.1 mm/day)"
rather than an empty section.

This is the same lesson as 6b's N6: on this fixture, "nothing to report" is the
correct output and also what a dead code path emits.

## P5 — the gridded branch must DECLARE, not add jobs

§8: "`save_gridded: true` declares-not-adds-jobs dry-run".

**Falsified if** flipping `save_gridded` changes the job **count**. It changes
which outputs are declared, not how many jobs run — the gridded products are
written by jobs that already exist.

## P6 — no value may move

Step 7 is additive plus a plot-set change.

**Falsified if** any `series/`, `change_factors/`, `summary/` or `provenance.json`
value moves. Declaring an output does not recompute it.

**Note on the PNG gate (from 6c):** `check_baseline` compares PNGs by size with a
10 % tolerance, so it cannot be trusted to confirm a figure changed or did not.
Any figure claim in this step is checked by sha256 and mtime, as 6c's O4 was.

## Order of work

1. 7-i: declare the eight figures; rename the mislabelled output; P1/P2/P6.
2. 7-ii: `report.md` reading the durable records; P3/P4.
3. 7-iii: gridded declaration; P5.
