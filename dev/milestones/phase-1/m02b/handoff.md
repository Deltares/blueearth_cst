# M2b — Mid-flight handoff

> **STATUS: SUPERSEDED — M2b shipped at tag `m02b-upgrades` (commit `16df134`).**
> Do **not** "resume" the steps below; they were a mid-flight resume guide for
> one paused session and have been fully executed. The as-shipped state is
> documented in `dev/milestones/phase-1/m02b/baseline_diffs.md`. Outstanding items surfaced
> during execution have been folded into `dev/followups.md`. This file is
> kept as historical record of the M2b decision log only.

Self-contained handoff so a fresh Claude Code session can pick up where the
previous one paused. Pair with `dev/milestones/phase-1/m02b/plan.md` (the original step plan)
and `dev/milestones/phase-1/m02b/audit.md` (upstream API breaks). This file overrides the plan
where they conflict — the plan is from before execution started; reality
diverged in several places that are documented below.

> **Branch:** `milestone/02b-library-upgrades`
> **HEAD at handoff:** `719b180`
> **Pixi:** 0.55.0 at `C:\Users\taner\AppData\Local\pixi\bin\pixi.exe`
> **Julia:** **1.11.7 (not 1.12.x — see Wflow.jl#884 below)**

---

## Snapshot of what's done

Commits since the audit (`06c54d0`):

```
719b180 m02b: eager .load() after cmip6 fetch in get_stats_climate_proj
9c3b43c m02b: climate_projections fixes — cmip6 catalog filesystem/ext_override, in-catalog check, hydrologic-year freq
7d59c9d m02b: pin Julia to 1.11.7, fix deferred staticmaps writes, pandas 3.x freq updates
6ebc46f m02b: rewrite src/ + configs for hydromt_wflow 1.x component API
95c4163 m02b: rename build/update CLI to wflow_sbm; merge model_resolution into build config; rename setup_reservoirs/setup_lakes
4df1fce m02b: rename WflowModel → WflowSbmModel across src/
1289526 m02b: bump Wflow.jl to 1.x via Project.toml + Manifest.toml
ee7b867 m02b: bump pixi.toml — Python 3.12, hydromt 1.3, hydromt_wflow 1.0
```

| Workflow | Code-level fixes | E2E verified |
|---|---|---|
| `Snakefile_model_creation` | ✅ done | ✅ all 4 rule_all targets produced (commit `7d59c9d` + later) |
| `Snakefile_climate_projections` | ✅ done in code | ⚠️ in-flight at handoff; reduced snake config, paused mid-run |
| `Snakefile_climate_experiment` | partial (downscale_climate_forcing rewrite committed; rest unattempted) | ❌ not run |

`pytest tests/` has not been re-run since the upgrades. The plan's exit
target is `13 passed, 2 xfailed` — preserve those two M3 xfails.

---

## Where execution paused

A reduced-scope projections run was killed mid-flight. Last state on disk:

- `examples/test_local/hydrology_model/` — fully built, full historical wflow
  run completed (`run_default/output.csv` ≈ 1.7 MB).
- `examples/test_local/plots/wflow_model_performance/` — 15 PNGs +
  `performance_metrics.csv`, all 4 rule_all targets present.
- `examples/test_local/climate_projections/cmip6/historical_stats_time_NOAA-GFDL/GFDL-ESM4.nc`
  — ~40 KB, real data 1950–2014 monthly precip+temp from a 5.5 h GCS fetch
  (the slow path before `.load()` was added). Acceptable as input to the
  next step; can be discarded if the resume re-runs everything.
- `config/snake_config_model_test_local.yml` — temporarily reduced to
  `models: ['NOAA-GFDL/GFDL-ESM4']`, `scenarios: [ssp245]`. Gitignored
  (per `*_local.yml`), so this is local-only state on this machine.
- `config/cmip6_data.yml.v0bak`, `config/deltares_data_local.yml.v0bak` —
  pre-migration safety copies from the v0→v1 catalog migration. Gitignored
  (`*.v0bak`). Keep until M2b ships.

No background processes left running (snakemake / julia / pixi all killed).

---

## Decisions and workarounds taken (read before resuming)

These are the non-obvious choices made during execution. Several were not
in the original audit/plan; if you disagree with one, that's where to push
back.

### 1. Julia 1.11.7 pin

**Wflow.jl#884** — Julia 1.12.x has a JIT compilation bug that hangs
Wflow.jl 1.0.x init right after the leaf_area_index cyclic-parameter
binding (no error message, julia.exe sits at ~1 GB indefinitely). Upstream
recommends Julia 1.11.7. Applied via:

- `Project.toml` `[compat] julia = "1.11"`
- `juliaup add 1.11.7` (installed on this machine)
- Both Snakefiles invoke `julia +1.11.7 ...`
- `Manifest.toml` regenerated on 1.11.7

**Action on resume:** if the dev machine doesn't already have 1.11.7 from
juliaup, `juliaup add 1.11.7` first. Wflow.jl version still pins to 1.0.2.

### 2. Data catalog v0 → v1 migration

The audit didn't anticipate the catalog-YAML schema rewrite. Upstream
schema changes (path→uri, meta→metadata, driver-string→object,
kwargs→driver.options, unit_*→data_adapter, alias inlining,
filesystem→driver.filesystem). Migrator is at
`dev/scripts/migrate_data_catalog_v0_to_v1.py` and is reusable.

Migrated catalogs:
- `config/cmip6_data.yml` (committed)
- `config/deltares_data_local.yml` (gitignored — local-only)

Driver-name map used:
| v0 driver | data_type | v1 driver.name |
|---|---|---|
| `vector` | GeoDataFrame | `pyogrio` |
| `csv`/`xlsx`/`xls`/`parquet` | GeoDataFrame | `geodataframe_table` |
| `raster` | RasterDataset | `rasterio` |
| `raster_tindex` | RasterDataset | `rasterio` (with mosaic options) |
| `netcdf` | RasterDataset | `raster_xarray` |
| `netcdf` | GeoDataset | `geodataset_xarray` |
| `zarr` | RasterDataset | `raster_xarray` |

The cmip6 catalog also needed `ext_override: .zarr` under `driver.options`
because the `gs://cmip6/.../*/*` URI ends in a glob with no extension.

**Action on resume:** if Linux/Docker work picks up later, run the
migrator on `config/deltares_data.yml` and `config/deltares_data_linux.yml`
the same way.

### 3. setup_constant_pars block dropped to one CSDMS-renamed param

`config/wflow_build_model.yml` originally had 14 constant pars
(`KsatHorFrac`, `Cfmax`, `cf_soil`, `EoverR`, `InfiltCapPath`,
`InfiltCapSoil`, `MaxLeakage`, `rootdistpar`, `TT`, `TTI`, `TTM`, `WHC`,
`G_Cfmax`, `G_SIfrac`, `G_TT`). Wflow.jl 1.x rejects these short names —
they all need CSDMS Standard Names. Wflow.jl 1.x further requires
`KsatHorFrac` to be set explicitly (errors at `LateralSsfParameters`
build time without it). Other 13 fall back to Wflow.jl 1.x defaults under
M2b's "intentional drift, re-baseline aggressively" policy.

Current block is just:
```yaml
- setup_constant_pars:
    subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio: 100
```

**Action on resume / M3:** map the other 13 to CSDMS names and decide
whether to restore them. CSDMS lookups in
`.pixi/envs/default/Lib/site-packages/hydromt_wflow/naming.py` and
`.../version_upgrade.py`. Examples already known: `Cfmax →
snowpack__degree_day_coefficient`, `WHC →
snowpack__liquid_water_holding_capacity`, `TT →
atmosphere_air__snowfall_temperature_threshold`, `TTI →
atmosphere_air__snowfall_temperature_interval`, `TTM →
snowpack__melting_temperature_threshold`, `G_Cfmax →
glacier_ice__degree_day_coefficient`, `MaxLeakage →
soil_water_saturated_zone_bottom__max_leakage_volume_flux`,
`InfiltCapPath → compacted_soil_surface_water__infiltration_capacity`.
`InfiltCapSoil` has `wflow_v1: None` in naming.py — deprecated, drop.

### 4. mod.close() bug

hydromt 1.x's `staticmaps.write()` writes to a `staticmaps_<hash>.nc`
temp file when the source nc is open in `r+` mode. The swap into the
canonical filename happens in `mod.close()`, **not** in `mod.write()`.
Without `mod.close()`, the staticmaps stay stale on disk and Wflow.jl
errors at run time on missing variables (`outlets`, etc).

`mod.close()` is now called after `mod.write()` in:
- `src/setup_gauges_and_outputs.py`
- `src/setup_reservoirs_lakes_glaciers.py`
- `src/downscale_climate_forcing.py`

If any new script writes to staticmaps via the model component API,
remember the close().

### 5. Snake config Snakefile_model_creation tweaks

- `model_resolution: 0.0062475` → `0.00833333` in
  `config/snake_config_model_test_local.yml`. hydromt 1.x rejects model
  resolution finer than the source hydrography (merit_hydro_ihu is
  1/120 ≈ 0.00833333). Pre-M2 baseline ran the finer resolution; this
  is intentional drift.
- Snakefile output `staticgeoms/gauges.geojson` → `outlets.geojson`.
  hydromt_wflow 1.x's `setup_outlets` writes to `outlets`, not `gauges`.
- plot_results.py — `mod.results` → `mod.output_csv.read()` +
  `mod.output_csv.data`. Outlet column name `Q_gauges` → `Q_outlets`.
  Outlet IDs are now subcatchment IDs (e.g. 130000086) instead of 1..N;
  `station_name` is rebuilt as 1..N to keep `hydro_wflow_1.png` stable.

### 6. Pandas 3.x frequency renames

Touched in `src/func_plot_signature.py`, `src/plot_map_forcing.py`,
`src/plot_proj_timeseries.py`, `src/get_change_climate_proj.py`,
`src/plot_results.py`. Mapping:
| Old | New |
|---|---|
| `"A"` | `"YE"` |
| `"M"` | `"ME"` |
| `f"AS-{Mon}"` | `f"YS-{MON.upper()[:3]}"` |

If you find another `resample(time="A")` etc., apply the same.

### 7. Reduced projections snake config

`snake_config_model_test_local.yml` is now `models: [NOAA-GFDL/GFDL-ESM4]`,
`scenarios: [ssp245]`. The full M2 baseline was 3 models × 2 scenarios.
Reduction is **only on the local test config** (gitignored). Full E2E
under the upgraded env would take ~6 h on this machine; reduced scope
gives ~1 h. M3 should investigate the GCS-throughput regression and
restore the full list. The eager `.load()` already cuts per-source time
from 5.5 h → 16 min — most of those 16 min is GCS metadata listing for
the `*/*` glob, not data transfer.

### 8. Things deliberately *not* fixed in M2b

- `src/get_change_climate_proj.py` — beyond the AS→YS frequency fix,
  hasn't been audited for hydromt 1.x API surface. May fail when M2b.5
  resumes past projections.
- `src/extract_historical_climate.py` — only `time_tuple → time_range`
  applied. Untested under upgraded env.
- `Snakefile_climate_experiment` — only the `julia +1.11.7` rename
  applied. The R weathergen layer hasn't been touched. The
  `downscale_climate_forcing.py` rewrite is committed but not exercised
  end-to-end yet (no realization run has been attempted).
- Dockerfile — explicitly out of scope per the plan.
- The stranded `staticmaps_<hash>.nc` temp files were cleaned by re-runs
  with `--forceall`; should not be present at handoff.

---

## To resume — next concrete steps in order

### Step 1 — verify env is sane

```bash
export PATH="/c/Users/taner/AppData/Local/pixi/bin:$PATH"
pixi run python -c "import hydromt, hydromt_wflow; print(hydromt.__version__, hydromt_wflow.__version__)"
# expect: 1.3.1 1.0.2
julia +1.11.7 --version
# expect: julia version 1.11.7
julia +1.11.7 --project=. -e 'using Pkg; Pkg.status()'
# expect: Wflow v1.0.2
```

### Step 2 — finish the reduced-scope projections E2E

Full chain on the reduced config (~1 h):
```bash
rm -rf examples/test_local/climate_projections
export PATH="/c/Users/taner/AppData/Local/pixi/bin:$PATH"
pixi run snakemake --unlock -s Snakefile_climate_projections \
  --configfile config/snake_config_model_test_local.yml
pixi run snakemake -s Snakefile_climate_projections \
  --configfile config/snake_config_model_test_local.yml \
  -c 1 --keep-going \
  > dev/baseline/m02b_run_projections.log 2>&1
echo "projections: $?"
```

Expect 4 jobs (1 hist + 1 fut + 2 changes for 2 horizons). First fetch
takes ~15 min (GCS metadata listing); subsequent fetches similar. Total
~1 h. If a script fails with a hydromt 1.x API surprise, fix it the
same way as previous commits — let pixi-run errors guide you.

If E2E goes green, restore the full M2 model/scenario list in the
local snake config so the M2b drift baseline isn't artificially small.
But that means a ~6 h run; only do it if the user signs off.

### Step 3 — climate_experiment E2E

```bash
pixi run snakemake --unlock -s Snakefile_climate_experiment \
  --configfile config/snake_config_model_test_local.yml
pixi run snakemake -s Snakefile_climate_experiment \
  --configfile config/snake_config_model_test_local.yml \
  -c 1 --keep-going \
  > dev/baseline/m02b_run_experiment.log 2>&1
echo "experiment: $?"
```

This will exercise `downscale_climate_forcing.py` (per realization) and
the R weathergen + Wflow.jl run loop. Expect breakage somewhere — the R
side hasn't been touched and the per-realization wflow runs on Julia
1.11.7 may hit a path that wasn't covered by the historical run.

The `weathergenr` R package was installed at the user level pre-M2;
verify with `R -e 'library(weathergenr); packageVersion("weathergenr")'`
before assuming it's there.

### Step 4 — pytest

```bash
pixi run pytest tests/
# expect: 13 passed, 2 xfailed
```

The two xfails are pre-existing M3 issues (cycle / cross-Snakefile dep).
Don't try to fix them in M2b. If anything else fails, debug it.

### Step 5 — re-baseline manifest (M2b.6 from the original plan)

```bash
pixi run python dev/scripts/check_baseline.py check 2>&1 \
  > dev/baseline/m02b_check_before_record.txt
# Expect a long drift list — this is intentional.

# Write dev/m02b_baseline_diffs.md per the plan template:
# - per-target table {unchanged/renamed/numerical drift/missing/new}
# - rough magnitudes for numerical drift
# - rename old → new variable names where applicable
# - sign-off line stating the new manifest is the M2b contract

pixi run python dev/scripts/check_baseline.py record
pixi run python dev/scripts/check_baseline.py check 2>&1 | tail -5
# Expect: OK — 14 target(s) match manifest.
```

### Step 6 — final exit checklist (M2b.7)

The original plan's exit checklist still applies. Notable tweaks under
the policies above:

- [ ] `pixi.toml` has hydromt ≥1.3,<2 and hydromt_wflow ≥1.0,<2; Python ≥3.12,<3.13. **Done.**
- [ ] `Project.toml` `[compat]` is `Wflow="1"`, **`julia="1.11"`** (not 1.12 as the plan says — Wflow.jl#884). **Done.**
- [ ] Every `WflowModel` renamed to `WflowSbmModel`. **Done.**
- [ ] `hydromt build/update wflow_sbm` is the only form. **Done.**
- [ ] `--opt setup_basemaps.res` and `--region` both gone; resolution + region merged via `prepare_build_config` rule. **Done.**
- [ ] `setup_reservoirs_simple_control` / `setup_reservoirs_no_control` in waterbodies config. **Done.**
- [ ] `wflow_outvars` semantic names → CSDMS via `WFLOW_VARS` map in `setup_gauges_and_outputs.py`. **Done.**
- [ ] All three workflows complete E2E. **Workflow 1 done; 2 + 3 pending.**
- [ ] `pytest tests/`: 13 passed, 2 xfailed. **Pending.**
- [ ] `dev/baseline/manifest.json` re-recorded; `check_baseline.py check` OK. **Pending.**
- [ ] `dev/m02b_baseline_diffs.md` committed. **Pending.**
- [ ] `dev/followups.md` updated with M3+ items surfaced during M2b. **Pending — list at end of this file.**
- [ ] Branch pushed; tag `m02b-upgrades` created and pushed. **Do NOT do this without user confirmation.**

---

## Followups for M3+ (write into dev/followups.md when M2b ships)

1. **GCS throughput regression** — single cmip6 source fetch went from
   "subsecond per chunk in 0.x" to "~15 min metadata listing in 1.x".
   Investigate whether hydromt 1.x's URI resolver is enumerating the
   `*/*` glob server-side inefficiently; consider switching to a
   directly-pointed zarr URL or staging cmip6 data locally.
2. **setup_constant_pars CSDMS remap** — restore the 13 dropped
   parameters under their CSDMS-renamed names (see decision #3 above).
3. **Reduced snake_config_model_test_local.yml** — restore full
   `models: [NOAA-GFDL/GFDL-ESM4, CMCC/CMCC-ESM2, INM/INM-CM5-0]` and
   `scenarios: [ssp245, ssp585]` once #1 is resolved.
4. **Outlet station naming** — `plot_results.py` now hardcodes
   `wflow_{i+1}` to keep `hydro_wflow_1.png` stable. M3 should decide
   whether to use real subcatchment IDs in the rule_all instead, or
   keep the 1..N convention as a documented pattern.
5. **func_plot_signature 'M' / 'A' / 'AS-' frequency callsites** — the
   pattern was repeated across multiple scripts; an audit of remaining
   pandas-3.x deprecation warnings would catch any I missed.
6. **plot_proj_timeseries.py** — `resample("A")` → `resample("YE")`
   committed but not exercised E2E. Re-verify when M2b.5 reaches the
   `plot_climate_proj_timeseries` rule.
7. **R weathergenr under upgraded env** — env upgrade may have broken
   the runtime install path. `pixi run install-rdeps` if needed.

---

## Files to keep in sync if anything in this handoff changes

- `dev/m02b_plan.md` — original step plan (do not rewrite; just note
  divergences here).
- `dev/m02b_audit.md` — original audit (do not rewrite; same).
- This file — single source of truth for "what's the actual state".
- `dev/m02b_baseline_diffs.md` — to be written in M2b.6.

If you (the resuming session) make new decisions, append a new section
to "Decisions and workarounds taken" rather than editing existing ones.

---

## One-line summary for `/loop` or quick handoff

> M2b model_creation E2E green on hydromt 1.3 / hydromt_wflow 1.0 /
> Wflow.jl 1.0.2 / Julia 1.11.7. Climate_projections green at code
> level; reduced-scope E2E paused mid-run. Resume via Step 2 above.
