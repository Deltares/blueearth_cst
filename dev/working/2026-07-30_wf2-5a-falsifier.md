# Falsifier — step 5a, spherical cell-area weighting (D10)

```
Written: 2026-07-30, BEFORE any 5a code exists
Gate:    R8 handoff §4 item 5 — "write each step's falsifier before coding it:
         the observation that would disprove its claimed property, plus the
         command that produces it"
Design:  wf2-climate-analysis-v2-design.md §5.3 (D10), §9, R8
Ref:     test_case/ref_wf2_pre_5a  (taken after 4d; CLEAN against itself, 126 files)
```

The design's §9 lists the *cases to test*. This note states what result would mean
**5a is wrong**, so the tests can fail for the right reason rather than merely run.

---

## What 5a claims

Replace the current unweighted `mean([x_dim, y_dim])` with a weighted mean whose
per-cell weight is the exact spherical area implied by midpoint-derived edges:

```
w(cell) = (sin φ_north − sin φ_south) × Δλ
```

edges from adjacent-center midpoints, the two boundary edges extrapolated
symmetrically (center ± half the adjacent spacing), a length-1 axis taking the
degenerate weight 1. Accept iff lat/lon are 1-D, finite and strictly monotonic;
otherwise raise naming the source.

---

## F1 — "strict generalization" is the load-bearing claim, and the easiest to break

The design argues D10 *replaces* cos-latitude rather than competing with it,
because on a uniformly spaced grid the two coincide exactly:
`sin(φ+d/2) − sin(φ−d/2) = 2·sin(d/2)·cos φ`, the constant cancelling in a
weighted mean.

**Falsified if:** on a uniform 1-D lat grid, normalised D10 weights differ from
normalised `cos(lat)` weights by more than floating-point tolerance.

That result would mean 5a changes numbers on uniform grids for some reason *other
than* area — i.e. an edge or extrapolation bug — and every uniform-grid diff
attributed to "weighting" would be attributing a bug.

```python
lat = np.arange(-60, 61, 2.5)          # uniform
w_d10 = cell_area_weights(lat, lon)     # the new function
np.testing.assert_allclose(w_d10 / w_d10.sum(),
                           np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).sum(),
                           rtol=1e-12)
```

## F2 — if it also matches cos-latitude on a NON-uniform grid, 5a is pointless

ext2-02's whole objection was that cos(lat) is a valid area weight only for
uniformly spaced grids, and that a Gaussian grid passes a "1-D + monotonic" check
while receiving wrong weights.

**Falsified if:** on a Gaussian-like (non-uniformly spaced) latitude axis, D10
weights are proportional to `cos(lat)` within tolerance.

Equality there would mean the per-cell latitude widths never entered the weights,
so the rejected scheme was reimplemented under a new name.

## F3 — the weights must be an area, checkable without reference to any implementation

**Falsified if:** over a grid partitioning the whole sphere, `Σ w` differs from the
sphere's area (`4π` steradians, or `2·Δλ_total` in the `sinφ` formulation) beyond
tolerance.

This is the one check that does not trust the formula — it tests the *edges and the
symmetric boundary extrapolation*, which is exactly where a midpoint scheme goes
wrong. A partition that does not sum to the sphere means boundary cells are
mis-sized.

## F4 — the check must refuse exactly the unrepresentable class, no more

R8 narrowed the refusal class in revision 4. Two ways to be wrong, in opposite
directions:

**Falsified (unsafe) if:** 2-D/curvilinear coordinates, or a non-monotonic axis
(including a dateline-wrapped subset), are accepted and silently weighted.

**Falsified (over-refusing) if:** a non-uniformly spaced 1-D monotonic axis —
Gaussian latitudes above all — **raises**. That would contradict the
"strict generalization" framing by narrowing coverage relative to today, and R8's
claim that the refusal class is "genuinely unrepresentable" would be false.

The second is the one worth watching: it is the failure that looks like caution.

## F5 — degenerate axis

**Falsified if:** an axis of length 1 yields `0`, `NaN`, or raises, rather than the
degenerate weight 1. A single-cell basin is the *common* case at `Amon` resolution
on a small catchment — this is not an edge case here, it is the small-basin path.

## F6 — REWRITTEN 2026-07-30, before wiring: the fixture cannot gate this step

**As first written, F6 said "falsified (no-op) if the diff reports CLEAN". That is
wrong for this fixture, and acting on it would have condemned a correct 5a.**

Measured on `test_case/test_local`'s actual grid before writing any wiring:

```
lat = [ 0.75, -0.75 ]      lon = [ 8.0, 10.0 ]
normalised cell-area weights = [[0.25, 0.25],
                                [0.25, 0.25]]
max deviation from uniform   = 0.0      (exactly, not approximately)
```

The basin is equatorial and its two latitude rows are **symmetric about the
equator**. `sin` is odd, so `|sin φ_n − sin φ_s|` is identical for both rows and
the area weights are exactly equal. The weighted mean *is* the unwe- ighted mean
here — bit for bit, not within tolerance. Confirmed on the data: `precip` and
`temp` basin means move by `+0.0000%` / `−0.0000%` (float noise).

The same 2×2 shape at other latitudes, for contrast:

| basin centre | weight spread |
|---|---|
| 0.75° (this fixture) | 8.6e-05 |
| 30° | 3.8e-03 |
| 45° | 6.5e-03 |
| 60° | 1.1e-02 |

**Consequences, and they are not small:**

1. The design's §8 row for 5a says "Re-record; diff **is** the weighting effect".
   On this fixture there is **no weighting effect to be the diff**. That row
   assumes a fixture the repo does not have.
2. A CLEAN value diff after 5a is the **correct** result here, not evidence the
   weighting failed to reach the reduction. F6's original form would have read
   correctness as failure.
3. Conversely — and this is now the real risk — a CLEAN diff proves *nothing*
   about D10. The fixture cannot distinguish a correct implementation from one
   that ignores latitude entirely.

**So 5a's correctness is carried by F1–F5 (unit tests, no fixture), not by the
tree diff.** The tree diff degrades from a correctness gate to a contamination
check.

**Restated F6:**

- **Falsified (contaminated) if** the diff shows changes in artifacts spatial
  reduction cannot touch — `raw/` slices, the config snapshot, `composition.csv`'s
  resolution columns.
- **Expected and required:** the series' `cst_weighting_scheme` attribute changes
  from `unweighted_mean_pre_5a` to `spherical_cell_area_midpoint_edges`, and the
  geometry-check result appears. Attributes move; values do not.
- **Falsified if VALUES move** on this fixture by more than float noise. Given
  exactly-equal weights, a value change means the reduction did something other
  than an area-weighted mean.

That last one is the useful inversion: on this fixture the strong test is that
5a changes **nothing numeric**, which is a sharper assertion than "something
changed".

**Carry-forward.** Gating 5a's actual effect needs a non-equatorial fixture. Either
add a synthetic mid-latitude case to the unit tests (cheap, done — F1/F2 cover the
arithmetic), or accept that no end-to-end evidence of the weighting exists in this
repo and say so rather than implying the tree diff supplies it.

## F7 — provenance must record what was done

**Falsified if:** after 5a the series still carry
`cst_weighting_scheme = "unweighted_mean_pre_5a"` (the current value, visible in
today's `gcm_timeseries.nc` attrs), or the geometry-check result is absent from the
series, `provenance.json` and the report.

A correct number with a stale provenance label is worse than a wrong number: it is
a wrong number that survives review.

---

## The cross-check 5a gets for free

5a is the **first real test of the fetch/reduce split**. Changing the reduction
changes `REDUCER_HASH`, so all 9 series re-derive. The split's acceptance criterion
(benchmark note §2) says that must cost **zero network requests**, because the raw
slices are already on disk and freshness is decided from their recorded digests.

**Observation to record:** run 5a's re-derivation and confirm no remote call is
made. The falsifier for the split itself was stated as "run it with the network
unavailable — it must succeed, not degrade". 5a is the first opportunity to
actually run that, and it costs nothing extra.

If 5a's re-derivation *does* hit the network, the split did not deliver what
`347638d` claimed, and that is a finding worth more than 5a.

---

## Order of work

1. Write the grid-geometry tests from F1–F5 against the new weighting function,
   **before** wiring it into the reducer. They need no fixture and no network.
2. Wire it in; confirm F7's attribute changes.
3. Re-derive, recording the F-cross-check observation (network or no network).
4. Run F6's diff and characterize every difference before accepting it.
5. Only then re-record the baseline — `check_baseline` will legitimately change,
   and a re-record before the diff is characterized destroys the attribution
   (risk-04).
