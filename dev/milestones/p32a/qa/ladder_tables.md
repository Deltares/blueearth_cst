# P3-2a ladder tables (era5, era5_20000101_20201231)

| edge | variable | mean | max-abs | rmse |
| --- | --- | --- | --- | --- |
| A2-A1 (correction component) | P_subcatchment (P_subcatchment) | 0 | 0 | 0 |
| A2-A1 (correction component) | T_subcatchment (T_subcatchment) | 0.159737 | 0.159763 | 0.159737 |
| A2-A1 (correction component) | EP_subcatchment (EP_subcatchment) | 0.00656873 | 0.00760937 | 0.00659924 |
| A2-A0 (sanctioned change) | P_subcatchment (P_subcatchment) | 3.97364e-06 | 3.8147e-05 | 1.13309e-05 |
| A2-A0 (sanctioned change) | T_subcatchment (T_subcatchment) | -9.82285e-05 | 0.000112534 | 9.88607e-05 |
| A2-A0 (sanctioned change) | EP_subcatchment (EP_subcatchment) | -1.10666e-05 | 7.03335e-05 | 4.7523e-05 |
| G (S3-S0 grid, masked) | P (precip) | 4.20732e-06 | 0.00499058 | 0.000368584 |
| G (S3-S0 grid, masked) | T (temp) | -0.000101933 | 0.00500107 | 0.0031278 |
| G (S3-S0 grid, masked) | EP (pet) | -1.0416e-05 | 0.00500011 | 0.00288957 |

## Persisted ladder states — untracked since `chore/dev-folder-tidy`

The two ladder states behind this table are **no longer tracked**:

| File | Size | SHA256 |
|---|---|---|
| `l1_regrid_only.nc` (S2, corrections OFF) | 35 428 895 B | `7a3f32c1cf95ee96e9070599acd8e30eb88e4ce51cbf37e5c5efb2121bc7a673` |
| `l2_parity.nc` (S3, corrections ON) | 35 436 291 B | `316e2c7ff38b3c30f87b78346453830e07a8825e4352a72dcff5cc135724ef7c` |

They were 68 MB of the repository's 80 MB `dev/` tree and the two largest blobs
in its history, committed alongside their gitignored sibling PNGs — an
accident, not a policy.

**They are not regenerable.** `../compare_climate_ladder.py` has been
superseded and non-runnable since R07 (its `climate_historical/wf1_raw/` input
was retired; see that module's docstring). Recover them from history instead —
the blobs remain reachable and byte-identical:

```bash
git checkout 75eb4d6 -- dev/milestones/p32a/qa/l1_regrid_only.nc dev/milestones/p32a/qa/l2_parity.nc
```

The table above is the milestone's finding; the states are the intermediates
behind it. R07 §2e (`dev/milestones/r07/migration_project-layout.md`) separately proved the
two stores element-wise identical, which retires the question the ladder was
built to answer.
