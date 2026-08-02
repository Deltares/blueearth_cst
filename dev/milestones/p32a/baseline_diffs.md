# P3-2a baseline record — characterized diff (comparison ladder) + manifest confirm

Recorded 2026-07-24 on the tracked fixture (`config/workflows/snake_config_model_test.yml`,
`project_dir = examples/test_local`, era5, window 2000-01-01..2020-12-31,
store key `era5_20000101_20201231`). Produced by
`dev/milestones/p32a/compare_climate_ladder.py` (design §5.5 ext1-3); persisted QA states
under `dev/milestones/p32a/qa/` (untracked): `l1_regrid_only.nc` (S2), `l2_parity.nc`
(S3), `ladder_tables.md`, `side_by_side_clim_wflow_1_{month,year}.png`.

## Ladder tables (per-subcatchment monthly climatology; every residual on a named edge)

| edge | variable | mean | max-abs | rmse |
| --- | --- | --- | --- | --- |
| A2−A1 (correction component) | P | 0 | 0 | 0 |
| A2−A1 (correction component) | T | 0.159737 | 0.159763 | 0.159737 |
| A2−A1 (correction component) | EP | 0.00656873 | 0.00760937 | 0.00659924 |
| A2−A0 (sanctioned change) | P | 3.97364e-06 | 3.8147e-05 | 1.13309e-05 |
| A2−A0 (sanctioned change) | T | −9.82285e-05 | 0.000112534 | 9.88607e-05 |
| A2−A0 (sanctioned change) | EP | −1.10666e-05 | 7.03335e-05 | 4.7523e-05 |
| G (S3−S0 grid, basin-masked) | P (precip) | 4.20732e-06 | 0.00499058 | 0.000368584 |
| G (S3−S0 grid, basin-masked) | T (temp) | −0.000101933 | 0.00500107 | 0.0031278 |
| G (S3−S0 grid, basin-masked) | EP (pet) | −1.0416e-05 | 0.00500011 | 0.00288957 |

## Edge-by-edge reading (era5 branch — the §5.5 acceptance logic)

- **A2−A1 (correction component).** T +0.160 °C uniform — the lapse term
  `−0.0065 °C/m × (mean model-DEM − era5-orography)` for this low-relief
  basin (≈ −25 m mean offset); EP the corresponding small PET response
  (+0.007 mm d⁻¹); **P ≡ 0 exactly** (null-check asserted `max-abs == 0.0`,
  bitwise — no correction touches precip).
- **A2−A0 (the sanctioned change).** ≤ 1.1e-4 °C / ≤ 4e-5 mm d⁻¹ — the
  parity path and the forcing build execute the same hydromt callables on the
  same sources, so the end-to-end plot-input change is numerically nil on
  era5, exactly as the design predicts. The visible plot difference is
  therefore presentation-free: the re-source decouples provenance (raw
  extraction vs stored inmaps artifact) without moving the plotted values
  beyond float noise.
- **G (grid-level parity, S3 vs S0 within the basin mask).** max-abs
  ≈ 0.005 for all three variables — within the §5.2 tolerances (0.05); the
  0.005 quantum is the inmaps storage packing/float32 rounding, uniform in
  sign-mixed noise (means ~1e-4). No missing/double correction (which would
  show as `lapse × Δdem` ≈ 1–3 °C).
- **wf1_raw vs keyed store (ext1-5 closure).** `allclose` (NaN-mask equal +
  values close) for all 7 extraction variables — the staticmaps-bbox vs
  region-bbox swap changed nothing. Unit-test-recorded per-edge bbox offsets:
  ~3.3e-7° (tolerance 2 × 0.00833° = 0.0167°).

## check_baseline (manifested wf1 slice) — no-op confirm + knowing divergence

`pixi run python dev/scripts/check_baseline.py check` (2026-07-24):

- **All value-carrying targets PASS**: `hydro_wflow_1.png`, `basin_area.png`,
  `precip.png`, and the `output.csv` discharge comparator are unchanged by
  P3-2a (C2 held).
- **2 pre-existing FAILs**: the `snake_config_model_creation.yml` /
  `snake_config_climate_projections.yml` copied-config rows — the manifest's
  stale yaml rows (mixed provenance since t260719a; tracked follow-up chore,
  `baseline-manifest-coverage` memory). Proven not-P3-2a: the current snapshot
  is byte-identical (sha256 `1335C7B6…`) to the pre-P3-2a tree snapshot.
- **Manifest re-record: none (knowing divergence from intake decision 4,
  repo-2).** The changed `clim_wflow_1_{month,year}.png` plots are not `rule
  all` targets and not manifested, so no re-record of the changed targets is
  possible; this ladder — not the manifest — carries the acceptance weight.
  `dev/baseline/manifest.json` is deliberately untouched. OQ-4 tracks the
  future manifest extension if the `clim_*` plots are ever promoted.

## wf2/wf3 value-identity evidence (C2/C3, recorded at commit 2)

Full 57-job wf3 re-run at the commit-2 tip (rule 3.08 executing via its
repointed `script:` path): `semantic_tree_diff` pre/post **CLEAN — 101 files,
0 failed, 0 missing, 0 extra**. Rule 3.02's diff across the milestone is
exactly its `script:` string; keyed-store/`.guard_ok` paths byte-identical.

## chirps note (ext2-2 defer-and-pin)

The fixture is era5; **no chirps tolerances are pinned in this record**. Per
the arbitrated procedure, chirps-basin plot acceptance stays **blocked** until
the first chirps/chirps_global basin runs this ladder and its per-variable
`A2−A1` / `A2−A0` / G values are pinned here (P exactly 0) with the
hop-mechanism justification. gabon's `clim_source` must be confirmed before
its diff is interpreted.
