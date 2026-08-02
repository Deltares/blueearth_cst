# R04 change-factor chain audit — findings

The §5 audit deliverable of the R4 climate-projections milestone
(`dev/milestones/r04/climate-projections-design.md`). Records the outcome of the
audit-evidence matrix (§7), the M2b attrs localization probe, and the
disposition of every finding. Reading/diagnostic only — no baseline moves here;
deferred defects are enumerated at the seal (roadmap R4 section) with owner +
activation condition per risk-3.

Date: 2026-07-20. Evidence is executable: pytest tests in
`tests/test_get_change_climate_proj.py`,
`tests/test_get_change_climate_proj_summary.py`,
`tests/test_get_stats_climate_proj.py` (commit `770e48c`), and the probe
`dev/scripts/probe_attrs_chain.py`.

## Audit-evidence matrix outcome

| Row | Target | Norm | Evidence (test) | Outcome |
|---|---|---|---|---|
| U | `get_change_annual_clim_proj` | precip multiplicative `(c−h)/h*100`, temp additive `c−h` | `test_row_U_change_precip_multiplicative_temp_additive` | **PASS** — exact 20.0 / 2.0 on constant fields |
| C1 | `get_change_annual_clim_proj` on `cftime.Datetime360Day` | calendar-invariant change (20.0 / 2.0) | `test_row_C1_cftime_360day_change_invariant` | **XFAIL (defect)** — code raises `TypeError` on cftime slicing → deferred, see D-CAL |
| C2 | `get_change_annual_clim_proj` on `cftime.DatetimeNoLeap` | same | `test_row_C2_cftime_noleap_change_invariant` | **XFAIL (defect)** — same cause → D-CAL |
| C3 | `get_change_annual_clim_proj` on proleptic-Gregorian `datetime64` | same | `test_row_C3_gregorian_datetime64_change_invariant` | **PASS** — standard calendar decodes to a pandas-native index |
| H | `get_change_annual_clim_proj`, `start_month_hyd_year="Oct"` (+ Jan control) | correct Oct→Sep whole-year window | `test_row_H_october_hydro_year_window_and_change`, `test_row_H_january_control_matches_row_U` | **PASS** |
| V | `get_change_annual_clim_proj`, asymmetric variables | fail-loud `ValueError` naming the missing var | `test_row_V_asymmetric_variables_should_raise` (normative, strict-xfail) + `test_row_V_char_result_carries_only_precip` (characterization, PASS) | **XFAIL (defect)** → D-VAR |
| P | `get_change_annual_clim_proj`, asymmetric members | fail-loud `ValueError` naming the unshared member | `test_row_P_asymmetric_members_should_raise` (normative, strict-xfail) + `test_row_P_char_result_keeps_only_shared_member` (characterization, PASS) | **XFAIL (defect)** → D-MEM |
| D | dummy-skip merge decision | empty netCDF excluded, non-empty kept | `test_filter_nonempty_excludes_empty_keeps_nonempty`, `test_filter_nonempty_all_empty_returns_empty_list` | **PASS** (via the `filter_nonempty` extract) |

Unit tests (`_to_str_tuple`, `get_change_clim_projections`,
`get_stats_clim_projections` sum/mean + dim detection, `preprocess_coords`) all
**PASS**. Full suite after commit `770e48c`: 102 passed, 3 skipped, 6 xfailed.

## M2b attrs localization (unconditional obligation, ext1-4 / ext2-3)

Probe: `dev/scripts/probe_attrs_chain.py` — traces CF attrs
(`units`/`standard_name`/`long_name`) through the get_stats reduction (S0–S4),
the per-model get_change chain (P0–P6), and the summary merge (M0–M2) on a
synthetic known-attrs input.

**Result: no workflow-2 pure-Python operation drops CF attrs** in the current
pinned environment. Every checkpoint (S0–S4, P0–P6, M0–M2) retains the full
attribute set — including `resample`, the change arithmetic, `assign_coords`,
`xr.merge`, `open_mfdataset(coords="minimal", preprocess=preprocess_coords)`,
and the `to_netcdf`/reopen round-trips.

**Localization:** the only remaining candidate for the M2b `{}` attrs on
`annual_change_scalar_stats_summary.nc` is the hydromt catalog read
(`data_catalog.get_rasterdataset(...)` in `get_stats_climate_proj.py`) — a
**dependency operation**, matching the design's accepted "dependency reproducer"
end state (§5). The M2b defect was originally diagnosed under hydromt 1.3
(`dev/followups.md` § M2b); it is either produced by hydromt's raster-dataset
read stripping variable attrs, or no longer reproduces under the current pins.
Confirming which requires a real CMIP6 catalog read (network/GCS) and is part of
the deferred task, not this audit.

**Disposition:** the design's candidate #1 ruling stands — attrs restoration is
**OUT of R4** (the loss is not in workflow-2 code, so there is no mechanical
in-scope fix; a hydromt-interop fix or an explicit attrs re-attach is a
scientific/interop decision). Deferred to **D-ATTRS**.

## Deferred defects (enumerated at the seal, risk-3)

Each carries an owner (a dedicated task) and an activation condition. None
moves the R4 baseline.

- **D-CAL — cftime calendar defect.** `get_change_annual_clim_proj` builds
  hydrological-year slice bounds with `pd.to_datetime(...)` and applies them to
  a cftime-indexed dataset, raising `TypeError` on 360-day / noleap calendars
  (CMIP6-native, e.g. HadGEM/UKESM). Latent for the current seed (its models
  decode to `datetime64`; baseline clean) but real for the general CMIP6 model
  set. *Activation:* remove the strict-xfail markers on rows C1/C2 when the fix
  lands (a cftime-safe slice — e.g. string bounds or `.to_datetimeindex()`
  conversion mirroring `plot_proj_timeseries.py`). *Owner:* task **t260720c**.
- **D-VAR — variable-drop (row V).** Asymmetric hist/clim variables are silently
  intersected (`intersection()`), dropping a configured variable with no error.
  Norm: fail-loud `ValueError`. *Activation:* remove the row-V strict-xfail when
  fail-loud lands. *Owner:* task **t260720d**.
- **D-MEM — partial-member drop (row P).** Asymmetric members are silently
  inner-joined by xarray's default alignment, dropping an unshared member.
  Norm: fail-loud `ValueError`. *Activation:* remove the row-P strict-xfail when
  fail-loud lands. *Owner:* task **t260720d** (paired with D-VAR — same
  fail-loud hardening surface).
- **D-ATTRS — M2b CF-metadata loss. [RESOLVED 2026-07-21, t260720e — confirmed
  does-not-reproduce, no fix.]** The current-pins `annual_change_scalar_stats_
  summary.nc` (product of the real CMIP6 reads in the 2026-07-18 R01 rebuild and
  the 2026-07-20 R4 re-run) carries the **full 7-attr CF set** (`cell_measures`,
  `cell_methods`, `comment`, `long_name`, `original_name`, `standard_name`,
  `units`) on both `precip` and `temp`, and the recorded manifest fingerprint
  carries the same 7 (so `check --workflow climate_projections` passes on the
  `.nc`). With R4's probe having proved no wf2 code drops attrs, and the values
  being CMIP6-native (`original_name: 'pr'`, `standard_name: 'precipitation_flux'`),
  the hydromt read under the current pins (hydromt 1.3.1 / hydromt_wflow 1.0.2)
  demonstrably preserves them. The M2b `{}` diagnosis does **not** reproduce; its
  original root cause is not re-litigated (moot — no fix ships and the baseline
  already records attrs-present, so no re-record). *Owner (closed):* **t260720e**.

## Notes

- The strict-xfail wiring is the tripwire: when any deferred code defect (D-CAL,
  D-VAR, D-MEM) is fixed, its normative test flips xfail→xpass and fails the
  suite until the owning task removes the marker — so a fix cannot land silently.
- `filter_nonempty` was extracted from `summary_climate_proj` (output-neutral)
  to give row D a unit entry point without the hydromt `.raster` accessor, which
  collides with sibling test files' `sys.modules` stubs (M02c).
