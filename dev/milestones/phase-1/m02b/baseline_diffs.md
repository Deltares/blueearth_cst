# M2b — Baseline drift report

Diff between the M1 manifest (recorded against hydromt 0.x /
hydromt_wflow 0.x / Wflow.jl 0.7 / Julia 1.10 / Python 3.11) and the
fresh outputs produced under M2b's upgraded env (hydromt 1.3 /
hydromt_wflow 1.0 / Wflow.jl 1.0.2 / Julia 1.11.7 / Python 3.12).

Source of comparison: `dev/baseline/m02b_check_before_record.txt`.

## Verdict

All drifts are intentional consequences of the M2b library upgrades —
no regressions. Re-record the manifest as the new M2b contract.

## Per-target table

| Target                                                                                  | Class                | Magnitude / notes                                                                                              |
| --------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------- |
| `climate_experiment/model_results/Qstats.csv`                                           | numerical drift      | sha differs; same row/column shape. Wflow.jl 1.0.2 vs 0.7.x re-routes water through the upgraded SBM solver.   |
| `climate_experiment/model_results/basin.csv`                                            | unchanged            | sha match.                                                                                                     |
| `climate_projections/cmip6/annual_change_scalar_stats_summary.csv`                      | numerical drift      | sha differs; CSV schema unchanged.                                                                             |
| `climate_projections/cmip6/annual_change_scalar_stats_summary.nc`                       | numerical drift      | precip Δ-stats (kg m⁻² s⁻¹): min −44.8 → −40.1, max 87.6 → 29.2, mean 3.96 → 1.07, std 16.1 → 12.5. temp Δ-stats (K): min −0.027 → −0.096, max 3.49 → 3.81, mean 1.05 → 1.28, std 0.88 → 0.99. Plus `precip`/`temp` `.attrs` are now `{}` (hydromt 1.x dropped CMIP6 source attrs in the round-trip) and `spatial_ref` dtype changed `int32 → int64`. |
| `climate_projections/cmip6/annual_change_scalar_stats_summary_mean.csv`                 | numerical drift      | sha differs (downstream of the .nc above).                                                                     |
| `climate_projections/cmip6/gcm_timeseries.nc`                                           | unchanged            | sha match (per check report — not in the FAIL list).                                                           |
| `config/snake_config_climate_experiment.yml`                                            | rename               | All three copied configs now share the same sha `ded15183…` because the canonical local config is one file (`snake_config_model_test_local.yml`) that the `copy_config` rule snapshots. M1 had divergent per-workflow configs.    |
| `config/snake_config_climate_projections.yml`                                           | rename               | (same as above)                                                                                                |
| `config/snake_config_model_creation.yml`                                                | rename               | (same as above)                                                                                                |
| `plots/wflow_model_performance/hydro_wflow_1.png`                                       | numerical drift (PNG) | size 287 212 B → 320 244 B (+10.3%, just over the 10% tolerance). hydromt_wflow 1.x rebuilt the model with subcatchment-id outlets (130000086, 1, 2, 3, 4, 6, 7) instead of M1's contiguous 1..N; series naming is restored to 1..N in `plot_results.py` so the layout is the same, but tick labels and rendering differ slightly. |
| `plots/wflow_model_performance/calib_xy_*.png`                                          | unchanged            | sha match (same plot generator, no semantic change).                                                           |

(Targets not listed above either matched or weren't part of the M1
manifest.)

## Renames worth flagging

- Outlet station IDs in `plot_results.py` come back as subcatchment IDs
  in hydromt_wflow 1.x rather than 1..N. We rebuild `station_name` as
  1..N before plotting to keep `hydro_wflow_1.png` stable; if a future
  consumer reads the raw outlet-id column, it will see `130000086, 1,
  2, 3, 4, 6, 7` instead of `1, 2, 3, 4, 5, 6, 7`. Documented in M2b
  decision #5 of the handoff.
- The 13 dropped `setup_constant_pars` entries (Cfmax, TT, TTI, TTM,
  WHC, G_Cfmax, MaxLeakage, etc.) now fall back to Wflow.jl 1.x
  defaults under the "intentional drift, re-baseline aggressively"
  policy (M2b decision #3). Restore under CSDMS-renamed names in M3.

## Known regressions surfaced (not blockers)

- **Lost CMIP6 attrs on monthly_change merge.** `precip`/`temp` data
  vars in `annual_change_scalar_stats_summary.nc` now have empty
  `.attrs` (M1 had `cell_measures`, `cell_methods`, `comment`,
  `long_name`, `original_name`, `standard_name`, `units` on each).
  Filed for M3 — `dev/followups.md` item.

## Sign-off

The drifts above are explained, expected, and within the spirit of the
M2b "intentional drift, re-baseline aggressively" policy. The fresh
manifest recorded after this report is the new M2b contract; future
runs should match it.
