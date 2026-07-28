# Migration — R07 project layout (old → new)

> Companion to `dev/r07/project-layout-design.md` (**ACCEPTED 2026-07-28**; not
> yet implemented). Required by `dev/conventions/naming.md` §7 — which commit 15
> **amends** to distinguish a *required* internal rename record under
> `dev/<milestone>/` from an *optional* user-facing guide under `docs/`. This file
> is R07's internal record. **R07 publishes no user-facing guide** (GA-2: pre-R07
> `project_dir` trees are unsupported and there is no production tree to migrate).
>
> **Not executable.** No `mv` script ships. A fresh run is the supported path
> (design § "Migrating an existing `project_dir`"). This document maps the paths
> for a reader and for the machinery.
>
> **Scope — corrected against the accepted design.** Most items are path moves,
> but *not all*, and the behaviour-preservation claim is scoped:
>
> - **Claim 1** — no computed value changes, **scoped to the baseline seed-fixture
>   class** (design § "Behaviour-preservation stance", ext1-03) — with **three**
>   named exceptions: (1) B4's new source-grid PET figures (additive), (2) O-22's
>   parse-time warning (warns, returns), and (3) **B1's bbox derivation, which
>   genuinely changes** and whose proof is the merge comparison in §2e. A stated
>   failure branch exists for exception 3.
> - **Claim 2** — four items **change code**, not just paths: **B1** (shared
>   producer spec + two declarations, input removals + one symmetric catalog
>   input, two `shared.basin` keys, rule-1.02 cross-check), **B6**
>   (`cst_*.csv` becomes a declared `input:` on rule 3.11), **B9**
>   (`copy_config_files.py` signature: one `output_dir` → four destinations), and
>   **O-22** (new helper + three call sites).
>
> **Commit plan: 15 commits.** Commit numbers in the tables below are that plan's.
> `check_baseline check` is **red by construction from commit 4 to commit 14**;
> `check_baseline.py`'s `TARGETS` + `PROJECT_DIR_DEFAULT` retarget rides in
> **commit 4**, atomically with the fixture rename (ext2-03), **not** commit 1.
>
> Reference tip: pre-R07 = `75eb4d6`. Fill in the milestone tip on landing.

**Rewrite classes** used in the tables, so `build_r07_path_map()` can be written
from this file without re-reading the design:

| Class | Meaning |
| --- | --- |
| `exact` | one literal path → one literal path |
| `prefix` | directory-prefix rewrite (the P3-1 map's only shape) |
| `pattern` | wildcard-bearing rewrite — a `rlz_<r>` / `cst_<c>` segment migrates between filename and directory, or is dropped. **Not expressible as a prefix rule.** |
| `merge` | many-to-one; needs the declared merge class, not a path-map entry |
| `new` / `deleted` | no counterpart on the other side |

---

## 1. Repository paths

| Old | New | Move | Commit | Class |
| --- | --- | --- | --- | --- |
| `data/observations/output-locations-test.csv` | *(deleted)* | O-01 | 2 | deleted |
| `data/observations/observations_timeseries_test.csv` | *(deleted)* | O-01 | 2 | deleted |
| `data/` (directory) | *(retired)* | O-01 | 2 | deleted |
| — | `config/templates/observations/output_locations.csv` | O-01 | 2 | new — header + 2 illustrative rows, comma-separated |
| — | `config/templates/observations/observations_timeseries.csv` | O-01 | 2 | new — header + ~3 rows, **semicolon**-separated |
| — | `config/templates/observations/README.md` | O-01 | 2 | new — schemas, the separator asymmetry, the `wflow_id` join rule |
| `docs/config/*` (16 tracked files) | *(deleted)* | O-05 | 3 | deleted — pre-R6 duplicates of `config/` |
| `examples/` | `test_case/` | O-20 | 4 | prefix — gitignored fixture root |
| `examples/test_local/` | `test_case/test_local/` | O-20 | 4 | prefix — the baseline seed's `project_dir` |
| `dag/` (repo root) | `<project_dir>/dag/` | O-02 | 5 | prefix — gitignored; `dag/` leaves `.gitignore:136` |
| `dag_model.png` (repo root) | *(deleted)* | O-02 | 5 | deleted — stale untracked artifact |
| `blueearth_cst/model/extract_climate_wf1.py` | *(deleted)* | B1 | 7 | deleted — rule 1.10's script, retired with the rule |
| `blueearth_cst/model/get_region_preview.py` | *(deleted)* | O-25 | 7 | deleted — dead; non-importable on hydromt 1.3.1 (`hydromt.cli.api` removed in 1.x) |
| — | `blueearth_cst/climate_analysis/plot_climate_source.py` | B4 | 12 | new — rule 1.15's script |
| — | `scripts/suggest_experiment_name.py` | B8 | 13 | new — thin CLI over `snake_utils.suggest_experiment_name` |
| `MIGRATION.md` (repo root) | `docs/migration-r06.md` | OQ-2 | 15 | exact — lowercased for `docs/` casing consistency (§8 row 4 carries no exemption inside `docs/`) |
| — | `dev/r06/migration_structural-refactor.md` | §7 amendment | 15 | new — the **required** internal R06 rename record, reconstructed from the moved file's rename tables |
| `blueearth_cst.Rproj` | *(deleted)* | O-13 | 15 | deleted — unreferenced, `Encoding: ISO8859-1`, unused (ruled at G1; commit assignment is this map's, see §7) |

### 1a. Config key *values* that change

| File | Key | Old | New | Move | Commit |
| --- | --- | --- | --- | --- | --- |
| `config/workflows/snake_config_model_test.yml` | `project.project_dir` | `examples/test_local` | `test_case/test_local` | O-20 | 4 |
| `config/workflows/snake_config_model_test_linux.yml` | `project.project_dir` | `examples/Gabon` | `test_case/gabon` | O-20 | 4 |
| `config/workflows/snake_config_model_test_linux.yml` | `…model_creation.output_locations` | `data/observations/output-locations-test.csv` | `None` | O-01 / ext1-05 | 2 |
| `config/workflows/snake_config_model_test_linux.yml` | `…model_creation.observations_timeseries` | `data/observations/observations_timeseries_test.csv` | `None` | O-01 / ext1-05 | 2 |
| `tests/snake_config_model_test.yml` | both observation keys | `tests/data/observations/…` *(dangling — no such dir; O-04)* | `None` | O-01 | 2 |
| `config/workflows/snake_config.template.yml` | `project.project_dir` | `examples/test` | outside-the-tree placeholder | O-21 | 6 |
| `config/workflows/*.yml` (as applicable) | `shared.basin.hydrography` | *(absent)* | `merit_hydro_ihu` (optional; default = the shipped template's value) | B1 / ext1-01 | 7 |
| `config/workflows/*.yml` (as applicable) | `shared.basin.basin_index` | *(absent)* | `merit_hydro_index` (optional; default as above) | B1 / ext1-01 | 7 |

**Sentinel warning.** Unquoted `None` parses to the Python **string** `"None"`,
not YAML `null`. `os.path.isfile("None")` / `os.path.exists("None")` are False, so
the existence guards at `setup_gauges_and_outputs.py:55` and `plot_results.py:127`
work — both read `if X is not None and os.path.<exists>(X):` and short-circuit, so
YAML `null` raises nothing there either (v1's `TypeError` claim was wrong; repo-4).
The value that actually misbehaves is the **string**: `plot_map.py:28-31` guards
only `if gauges_fn is not None:` and then builds `gauges_name =
f'gauges_{basename(gauges_fn).split(".")[0]}'`, yielding the bogus layer
`gauges_None` — that is **O-08**, fixed in commit 12. Every `None` written above
must stay byte-identical to `snake_config_model_test.yml:36-37`.

**Guard-digest note.** Absent `shared.basin.hydrography` / `basin_index` keys
leave rule 3.00b's guard digest byte-identical (it serializes the config dict
as-is), so the seed fixture and every existing config are unaffected.

### 1b. Reference sites to rewrite

`scripts/run_snake_test.cmd:32` (DAGDIR) · `scripts/run_snake_docker.sh:7` (drop
the `data/` mount, commit 2) and `:6` (`examples` → `test_case` mount, commit 4) ·
`.gitignore:124,135-136` · `README.rst:204,211,218,254,259,269,285,298` ·
`AGENTS.md` (repo map, `docs/` description, the invocation-model line for O-23) ·
`MIGRATION.md:173` (before its own move) · `docs/notebooks/*.ipynb` (six DAG
cells) · `docs/cst-toolbox-technical-note-2025.md:1278-1279` ·
`dev/workflows/model_creation.md:15-16,115` · `dev/scripts/check_baseline.py`
(commit 4) · `dev/scripts/semantic_tree_diff.py` (commit 1) ·
`dev/followups.md:171-184` · `dev/contracts/hydrological-model-seam.md:74,353` ·
`dev/contracts/weather-generator-seam.md:56,71,248,294`.

`tests/` path bindings (all must move with their targets):
`tests/test_model_creation.py:26-28` · `tests/test_interchange_contracts.py:39,484,529,570-571,592` ·
`tests/test_extract_climate_wf1.py:24,26` · `tests/test_check_baseline_scope.py:114,131,158,160` ·
`tests/test_semantic_tree_diff.py:332-388` · `tests/test_workflow_climate_experiment.py:114` ·
`tests/test_guard_invalidation.py:97` · `tests/test_check_project_consistency.py:30`.

---

## 2. `project_dir` paths

All paths project-root-relative. `<key>` = `<clim_source>_<YYYYMMDD>_<YYYYMMDD>`;
`<id>` = `experiment_name`; `<r>` / `<c>` = realization / stress-test index;
`<proj>` = `clim_project`.

### 2a. Project level

| Old | New | Move | Commit | Class |
| --- | --- | --- | --- | --- |
| `config/snake_config_model_creation.yml` | `config/runs/snake_config_model_creation.yml` | B9 | 10 | exact |
| `config/snake_config_climate_projections.yml` | `config/runs/snake_config_climate_projections.yml` | B9 | 10 | exact |
| `config/deltares_data.yml` | `config/catalogs/deltares_data.yml` | B9 | 10 | exact |
| `config/cmip6_data.yml` | `config/catalogs/cmip6_data.yml` | B9 | 10 | exact |
| `config/wflow_build_model.yml` | `config/templates/wflow_build_model.yml` | B9 | 10 | exact — verbatim snapshot of a shipped template |
| `config/wflow_update_waterbodies.yml` | `config/templates/wflow_update_waterbodies.yml` | B9 | 10 | exact — as above |
| `config/wflow_build_model_run.yml` | `config/generated/wflow_build_model_run.yml` | B9 | 10 | exact — **generated** at run time (rule 1.02 → 1.03) |
| `config/wflow_build_forcing_historical.yml` | `config/generated/wflow_build_forcing_historical.yml` | B9 | 10 | exact — **generated** at run time (rule 1.07 → 1.08) |
| `climate_historical/wf1_raw/extract_historical.nc` **+** `climate_historical/<key>/extract_historical.nc` | `climate_historical/<key>/extract_historical.nc` | B1 | 7 | **merge** — see §2e |
| `climate_historical/wf1_raw/orography.nc` **+** `climate_historical/<key>/<clim_source>_orography.nc` | `climate_historical/<key>/orography.nc` | B1 | 7 | **merge** — chirps / chirps_global branch only; see §2e |
| — | `climate_historical/<key>/store_region.geojson` | B1 | 7 | new — the model-free delineation the bbox came from |
| `climate_historical/<key>/.guard_ok` | *(path unchanged)* | B1 | 7 | unchanged path; its **DAG edge retires** — the artifact stays as the guard's store-level receipt |
| — | `climate_historical/<key>/plots/source_precip.png` | B4 | 12 | new — rule 1.15 `plot_climate_source` |
| — | `climate_historical/<key>/plots/source_temp.png` | B4 | 12 | new |
| — | `climate_historical/<key>/plots/source_pet.png` | B4 | 12 | new — source-grid PET; **may differ from the build's PET by design** |
| `climate_historical/wflow_data/inmaps_historical.nc` | `hydrology_model/forcing/inmaps_historical.nc` | B2 | 8 | exact |
| `plots/wflow_model_performance/precip.png` | `hydrology_model/forcing/plots/precip.png` | B10 | 12 | exact |
| `plots/wflow_model_performance/temp.png` | `hydrology_model/forcing/plots/temp.png` | B10 + O-24 | 12 | exact — **newly declared** on rule 1.13 |
| `plots/wflow_model_performance/pet.png` | `hydrology_model/forcing/plots/pet.png` | B10 + O-24 | 12 | exact — **newly declared** on rule 1.13 |
| `plots/wflow_model_performance/hydro_wflow_1.png` | `hydrology_model/evaluation/plots/hydro_wflow_1.png` | B10 | 12 | exact |
| `plots/wflow_model_performance/clim_wflow_1_month.png` | `hydrology_model/evaluation/plots/clim_wflow_1_month.png` | B10 + O-24 | 12 | exact — **newly declared** on rule 1.11 |
| `plots/wflow_model_performance/clim_wflow_1_year.png` | `hydrology_model/evaluation/plots/clim_wflow_1_year.png` | B10 + O-24 | 12 | exact — **newly declared** on rule 1.11 |
| `plots/wflow_model_performance/basin_area.png` | `hydrology_model/plots/basin_area.png` | B10 | 12 | exact — depicts the **model**, not its evaluation (P1) |
| `plots/wflow_model_performance/performance_metrics.csv` | `hydrology_model/evaluation/performance_metrics.csv` | B10 + O-24 | 12 | exact — leaves any `plots/` dir (P1: no CSVs in `plots/`); **newly declared** on rule 1.11 |
| `plots/` (project-level directory) | *(retired)* | B10 / P1 | 12 | deleted — there is no project-level `plots/` |
| `climate_projections/<proj>/gcm_timeseries.nc` | `climate_projections/<proj>/timeseries/gcm_timeseries.nc` | B3 | 9 | exact |
| `climate_projections/<proj>/annual_change_scalar_stats_summary.nc` | `climate_projections/<proj>/summary/annual_change_scalar_stats_summary.nc` | B3 | 9 | exact |
| `climate_projections/<proj>/annual_change_scalar_stats_summary.csv` | `climate_projections/<proj>/summary/annual_change_scalar_stats_summary.csv` | B3 | 9 | exact |
| `climate_projections/<proj>/annual_change_scalar_stats_summary_mean.csv` | `climate_projections/<proj>/summary/annual_change_scalar_stats_summary_mean.csv` | B3 | 9 | exact |

**Unchanged by design — stated explicitly, because the stale map mis-scoped them:**

| Path | Why it does not move |
| --- | --- |
| `climate_projections/<proj>/plots/projected_climate_statistics.png` | B3 / arch-10: only the **three summary files** move; the three PNGs stay |
| `climate_projections/<proj>/plots/precipitation_anomaly_projections_abs.png` | as above |
| `climate_projections/<proj>/plots/temperature_anomaly_projections_abs.png` | as above |
| `climate_projections/<proj>/annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc` | `temp()` per-member intermediates; no tier assigned, no `raw/` tier exists |
| `hydrology_model/{wflow_sbm.toml, staticmaps.nc, staticgeoms/, hydromt.log, hydromt_data.yml}` | the hydromt `model_root`'s immediate children — P3-exempt, `AGENTS.md` Hard Constraints |
| `hydrology_model/run_default/` (incl. `output.csv`) | wf1 historical simulation; exception 3(d) requires it **provably unmoved** — rule 1.08 builds `inmaps_historical.nc` via `hydromt update` from the catalog, not from the extraction |
| `hydrology_model/instate/instates.nc` | engine-mandated, inside the model root |
| project-level `logs/`, `benchmarks/` | wf1 + wf2; not content-bearing (`EXCLUDED_DIR_NAMES`) |

**Retired directories (project level):** `climate_historical/wf1_raw/`,
`climate_historical/wflow_data/`, `plots/` (and `plots/wflow_model_performance/`).

### 2b. Experiment subtree

| Old | New | Move | Commit | Class |
| --- | --- | --- | --- | --- |
| `experiments/<id>/model_results/Qstats.csv` | `experiments/<id>/indicators/Qstats.csv` | B7 | 11 | prefix |
| `experiments/<id>/model_results/basin.csv` | `experiments/<id>/indicators/basin.csv` | B7 | 11 | prefix |
| `experiments/<id>/model_results/RT_*.csv` | `experiments/<id>/indicators/RT_*.csv` | B7 | 11 | prefix |
| — | `experiments/<id>/indicators/plots/` | B7 | 11 | new — **reserved, no producer yet**; empty on both sides, so not an allowlist entry |
| `experiments/<id>/weathergen_config.yml` | `experiments/<id>/weather_generator/config/weathergen_config.yml` | B5 | 11 | exact |
| `experiments/<id>/realization_<r>/rlz_<r>_cst_<c>.nc` | `experiments/<id>/weather_generator/output/rlz_<r>_cst_<c>.nc` | B5 | 11 | **pattern** — filename unchanged, `realization_<r>/` dissolves; keeps `temp()` |
| `experiments/<id>/realization_<r>/weathergen_config_rlz_<r>_cst_<c>.yml` | `experiments/<id>/weather_generator/_work/weathergen_config_rlz_<r>_cst_<c>.yml` | B5 | 11 | **pattern** — filename unchanged, directory collapses |
| `experiments/<id>/stress_test/cst_<c>.csv` | `experiments/<id>/weather_generator/_work/cst_<c>.csv` | B6 | 11 | prefix — **retained, not deleted** (see note below) |
| `experiments/<id>/obs_power_spectra.png` | `experiments/<id>/weather_generator/plots/obs_power_spectra.png` | B5 | 11 | exact |
| `experiments/<id>/warm_annual_precip.png` | `experiments/<id>/weather_generator/plots/warm_annual_precip.png` | B5 | 11 | exact |
| `experiments/<id>/warm_annual_stats.png` | `experiments/<id>/weather_generator/plots/warm_annual_stats.png` | B5 | 11 | exact |
| `experiments/<id>/warm_annual_wavelet.png` | `experiments/<id>/weather_generator/plots/warm_annual_wavelet.png` | B5 | 11 | exact |
| `experiments/<id>/resampled_dates.csv` | `experiments/<id>/weather_generator/output/resampled_dates.csv` | B5 / OQ-4 | 11 | exact — **`output/`, not `_work/`** (ruled: products of the generator) |
| `experiments/<id>/sim_dates.csv` | `experiments/<id>/weather_generator/output/sim_dates.csv` | B5 / OQ-4 | 11 | exact — as above |
| `experiments/<id>/realization_<r>/inmaps_rlz_<r>_cst_<c>.nc` | `experiments/<id>/hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc` | B5 | 11 | **pattern** — **not** `weather_generator/output/` (repo-9 correction): this is wflow-grid downscaled forcing from rule 3.09, the per-realization twin of B2's move. Keeps `temp()` |
| `experiments/<id>/model_runs/wflow_sbm_rlz_<r>_cst_<c>.toml` | `experiments/<id>/hydrology_runs/rlz_<r>/config/cst_<c>.toml` | B5 | 11 | **pattern** |
| `experiments/<id>/model_runs/output_rlz_<r>_cst_<c>.csv` | `experiments/<id>/hydrology_runs/rlz_<r>/output/cst_<c>.csv` | B5 | 11 | **pattern** |
| `experiments/<id>/model_runs/outstates_rlz_<r>_cst_<c>.nc` | `experiments/<id>/hydrology_runs/rlz_<r>/output/outstates_cst_<c>.nc` | B5 | 11 | **pattern** — keeps `temp()` |

`experiments/<id>/realization_<r>/`, `experiments/<id>/model_runs/`, and
`experiments/<id>/stress_test/` **all dissolve**.

**Unchanged:** `experiments/<id>/config/snake_config_climate_experiment.yml`
(arch-10 — it does **not** join `config/runs/`; content changes only),
`experiments/<id>/data_catalog_climate_experiment.yml`,
`experiments/<id>/.project_consistency_ok`, `experiments/<id>/{logs,benchmarks}/`.

**B6 retention note.** `cst_*.csv` is demoted for legibility, **not** discarded:
it remains the only record of the `precip_variance` axis and of monthly
structure (`export_wflow_results.py:162-163` reads only the **January** row's
`temp_mean` / `precip_mean`). `_work/` is preserved on disk. The same move
promotes `cst_*.csv` from an undeclared runtime read
(`export_wflow_results.py:161`) to a declared `input:` on rule 3.11.

### 2c. In-TOML pointer strings that change value

Same targets, new depths; hydromt re-relativizes on write. Compare in a
project-root-relative namespace with the path-aware TOML comparator
(`dev/scripts/semantic_tree_diff.py`), never as raw strings. **The comparator
needs no change** — it already covers all five pointer fields generically
(including `csv.path` and `state.path_output`); B5 needs a new *path map*.

| Key | Old | New | Move | Commit |
| --- | --- | --- | --- | --- |
| `input.path_forcing` (project TOML) | `../climate_historical/wflow_data/inmaps_historical.nc` | `forcing/inmaps_historical.nc` | B2 | 8 |
| `input.path_static` (run TOMLs) | `../../../hydrology_model/staticmaps.nc` | `../../../../hydrology_model/staticmaps.nc` | B5 | 11 |
| `state.path_input` (run TOMLs) | `../../../hydrology_model/instate/instates.nc` | `../../../../hydrology_model/instate/instates.nc` | B5 | 11 |
| `input.path_forcing` (run TOMLs) | `../realization_<r>/inmaps_rlz_<r>_cst_<c>.nc` | `../forcing/inmaps_cst_<c>.nc` | B5 | 11 |

Depth derivation: run TOMLs move from `experiments/<id>/model_runs/` (3 levels
below the project root) to `experiments/<id>/hydrology_runs/rlz_<r>/config/` (4
levels), so every pointer that escapes the experiment gains one `../`; the forcing
pointer instead becomes a sibling-directory hop inside `rlz_<r>/`. hydromt
re-relativizes on write, so the comparator — not this map — is authoritative; the
strings above are the expected values, given for reviewability.

### 2d. `COPIED_CONFIG_PATH_MAP` additions (`semantic_tree_diff.py:90-110`)

`compare_copied_config` normalizes only keys present in this map and FAILs on any
residual difference, so without these entries the phase-B gate goes red for pure
path bookkeeping — indistinguishable from a real content regression (repo-6,
arch-11a). Landed in **commit 1**.

| Key | Old value | New value | Driver |
| --- | --- | --- | --- |
| `project_dir` | `examples/test_local` | `test_case/test_local` | O-20 |
| `project_dir` | `examples/Gabon` | `test_case/gabon` | O-20 |
| `project_dir` | `examples/test` | the outside-the-tree placeholder | O-21 |
| `output_locations` | `data/observations/output-locations-test.csv` | `None` | O-01 |
| `observations_timeseries` | `data/observations/observations_timeseries_test.csv` | `None` | O-01 |

Note also: `_is_copied_config` (`semantic_tree_diff.py:576`) matches **any** YAML
with a `config` path part, so the new
`experiments/<id>/weather_generator/config/weathergen_config.yml` is newly swept
into that directional policy. Intended — stated here so it is not read as a
surprise at the gate.

### 2e. The declared many-to-one merge class (blocking machinery, commit 1)

`diff_trees` keys the reference tree by mapped relpath and raises
`ValueError("path map collision: … both map to …")` when two reference files
translate to one key (`semantic_tree_diff.py:641-647`). B1 is exactly that, so
**no prefix rule can express it** and the gate aborts before it can report. R07
adds `--merge <survivor>=<src1>,<src2>`: the survivor is compared against **each**
collapsed source with the ordinary suffix-dispatched comparator, and the merge
passes only if **all** comparisons pass.

```
--merge climate_historical/<key>/extract_historical.nc=climate_historical/wf1_raw/extract_historical.nc,climate_historical/<key>/extract_historical.nc
# chirps / chirps_global branch only:
--merge climate_historical/<key>/orography.nc=climate_historical/wf1_raw/orography.nc,climate_historical/<key>/<clim_source>_orography.nc
```

**This merge comparison *is* exception 3's bbox proof** — `.nc` already dispatches
to the element-wise `compare_nc` (dims; coordinate labels **and stored order**, no
realignment; per-element values; NaN masks; non-volatile attrs). The two sides are
**not symmetric**:

- `survivor` vs `wf1_raw/…` — **expected exact.** The 2026-07-28 bounds probe puts
  the R07 `parse_region_basin` bbox **bit-identical** to today's wf1
  `staticmaps.nc` raster bounds on all four edges. A failure here means something
  other than the bbox changed.
- `survivor` vs the pre-R07 `<key>/…` — **the side that carries the risk.** That
  store was cut to `staticgeoms/region.geojson`'s 6-decimal-rounded bounds
  (Δ ≤ 3.4e-07°, four orders below the era5 cell and inside
  `prep_historical_climate`'s `buffer=1`).

Read a single failure against that asymmetry before invoking the branch below.

**If the merge comparison fails** (design § "Behaviour-preservation stance",
exception 3 branch):

1. List `clim_wflow_1_{month,year}.png` as **expected-to-move** — rule 1.11 reads
   the extraction at model parity, and PNG comparison is size-only with a 10% band
   (`check_baseline.PNG_TOLERANCE_FRAC = 0.10`), so it would not catch a shift.
2. Extend "expected-to-move" to the **wf3 indicator targets** — the store feeds
   weathergenr, so a shifted extraction shifts every downstream indicator. This
   falsifies the exit adjudication's "path-and-snapshot-only" claim, which must
   then be **rewritten, not annotated**.
3. Record the per-edge coordinate deltas **in this map** (§7 slot) and re-record
   the affected targets as a **stated** value change.
4. Confirm `hydrology_model/run_default/output.csv` is unmoved. **If discharge
   moves at all, stop and escalate to the owner** — outside every authorised
   branch.

### 2f. The store's freshness contract — boundary and escape hatch (B1 / ext2-01)

Documented here because the design requires the map, not the code, to carry it.

- **In scope:** the data catalog **file** (`project.data_sources`) is a declared,
  plain (not `ancient()`) `input:` on the producer in **both** DAGs. Editing it in
  place now mtime-triggers exactly one re-extraction — closing a staleness gap
  that **predates R07** (today rule 3.02 carries the catalog only as a `params`
  path string, so an in-place edit retriggers nothing).
- **Out of scope:** data *behind* an unchanged catalog entry — a local file the
  entry points at, or a remote store. Enumerating catalog-resolved sources as DAG
  inputs would need hydromt catalog semantics at DAG-parse time (outside CST's
  automation scope), and remote sources expose no usable mtime.
- **The supported signal** for a data change behind a stable entry is to **edit
  the entry** (path, version, or meta) — the catalog-conventional route, which the
  new input edge picks up.
- **The escape hatch** for a truly in-place data mutation is
  `snakemake --forcerun extract_climate_grid`.
- The producer's other rerun surface stays in `params`: the region string,
  `shared.basin.hydrography` / `basin_index`, `clim_source`, and the historical
  window. A region-specification change therefore re-extracts via the params
  trigger — which is what replaces the retired `ancient(.guard_ok)` edge.

### 2g. Non-seed derivation change — not a regression

Claim 1 is scoped to the baseline seed-fixture class (ext1-03). For another
region, resolution, or hydrography dataset, the new delineation bounds and today's
`staticmaps.nc` raster bounds can genuinely differ in principle — raster bounds are
snapped to the model grid, polygon bounds are not, and a polygon edge lying within
rounding distance of a source-grid cell boundary can shift the extracted extent by
one cell despite `buffer=1`. **Such divergence is the GA-1-accepted derivation
change, not a regression.** The scoping is lossless under GA-2 (the only pre-R07
`project_dir` tree in existence is the test fixture). The
configuration-independent invariant that must hold everywhere is the bbox-agreement
unit test: `store_region.geojson` bounds vs `staticmaps.nc` bounds, per-edge
tolerance 2 × model resolution, runnable on any project where both exist.

---

## 3. Baseline manifest impact

`dev/baseline/manifest.json` holds **18 rows**; `check_baseline.TARGETS` holds
**15 live templates**. **The inventory is re-derived from `TARGETS`, not from the
manifest file.** One re-record, at **commit 14**, after the discharge `compare`
gate.

Every manifest key is a literal path prefixed `examples/test_local/`, so **all 15
live targets change manifest key** (O-20, commit 4). Of those, **10 also move
within the tree** and **3 change content**.

### 3a. The 15 live targets

Paths below omit the `examples/test_local/` → `test_case/test_local/` prefix
change, which applies to **all** of them.

| # | Old (within tree) | New (within tree) | Move | Commit | Within-tree |
| --- | --- | --- | --- | --- | --- |
| 1 | `plots/wflow_model_performance/hydro_wflow_1.png` | `hydrology_model/evaluation/plots/hydro_wflow_1.png` | B10 | 12 | **moves** |
| 2 | `plots/wflow_model_performance/basin_area.png` | `hydrology_model/plots/basin_area.png` | B10 | 12 | **moves** |
| 3 | `plots/wflow_model_performance/precip.png` | `hydrology_model/forcing/plots/precip.png` | B10 | 12 | **moves** |
| 4 | `config/snake_config_model_creation.yml` | `config/runs/snake_config_model_creation.yml` | B9 | 10 | **moves** + content |
| 5 | `hydrology_model/run_default/output.csv` | *(same)* | O-20 only | 4 | prefix only |
| 6 | `climate_projections/cmip6/annual_change_scalar_stats_summary.nc` | `…/cmip6/summary/annual_change_scalar_stats_summary.nc` | B3 | 9 | **moves** |
| 7 | `climate_projections/cmip6/annual_change_scalar_stats_summary.csv` | `…/cmip6/summary/annual_change_scalar_stats_summary.csv` | B3 | 9 | **moves** |
| 8 | `climate_projections/cmip6/annual_change_scalar_stats_summary_mean.csv` | `…/cmip6/summary/annual_change_scalar_stats_summary_mean.csv` | B3 | 9 | **moves** |
| 9 | `climate_projections/cmip6/plots/projected_climate_statistics.png` | *(same)* | O-20 only | 4 | prefix only |
| 10 | `climate_projections/cmip6/plots/precipitation_anomaly_projections_abs.png` | *(same)* | O-20 only | 4 | prefix only |
| 11 | `climate_projections/cmip6/plots/temperature_anomaly_projections_abs.png` | *(same)* | O-20 only | 4 | prefix only |
| 12 | `config/snake_config_climate_projections.yml` | `config/runs/snake_config_climate_projections.yml` | B9 | 10 | **moves** + content |
| 13 | `experiments/experiment/model_results/Qstats.csv` | `experiments/experiment/indicators/Qstats.csv` | B7 | 11 | **moves** |
| 14 | `experiments/experiment/model_results/basin.csv` | `experiments/experiment/indicators/basin.csv` | B7 | 11 | **moves** |
| 15 | `experiments/experiment/config/snake_config_climate_experiment.yml` | *(same)* | O-20 only | 4 | prefix only + content |

**Derived counts: 15 live targets; 15 change manifest key; 10 move within the
tree; 3 change content (the config snapshots); 5 prefix-only.**

> The design's prose says "14 of them also changing path within the tree". That
> count is not reconstructible from `TARGETS` and appears to be residue from a
> draft in which the three wf2 PNGs moved; B3/arch-10 fixes them in place. The
> per-target derivation above gives **10**, and `check_baseline.TARGETS` must be
> written from this table, not from the "14".

### 3b. Three stale rows — expected deletions at the re-record

Pre-P3-1 orphans with no producer. `cmd_check` skips them (they are in neither
`current` nor `missing`) and a full `record` silently drops them. Listed here so
the exit adjudication accounts for them rather than reading them as R07 targets
that failed to be produced.

| Manifest key | Why orphaned |
| --- | --- |
| `examples/test_local/climate_experiment/model_results/Qstats.csv` | pre-P3-1 `climate_<name>/` layout; superseded by `experiments/<name>/` |
| `examples/test_local/climate_experiment/model_results/basin.csv` | as above |
| `examples/test_local/config/snake_config_climate_experiment.yml` | pre-P3-1 project-level wf3 snapshot; P3-1 moved it to `experiments/<name>/config/` |

### 3c. The discharge anchor and its sidecar (commit 14)

`check_baseline.py:384-385` derives the stored reference-series filename as
`sha1(resolved_path)[:16]`, so O-20 changes the **sidecar** name as well as the
manifest key. Because discharge is compared with a **tolerance comparator against
a stored series** — not a self-contained hash — a naive `record` would regenerate
the reference from the *post-R07* run and silently re-bless any drift.

| Step | Commit |
| --- | --- |
| Save `examples/test_local/hydrology_model/run_default/output.csv` to a run-local holding path | 1 |
| Gate **before** `record`: `python dev/scripts/check_baseline.py compare --ref <saved> --cur test_case/test_local/hydrology_model/run_default/output.csv` must exit 0 | 14 |
| Delete the orphaned `dev/baseline/discharge_ref/1f9f30a367de162f.csv`; a new `sha1(resolved_path)[:16]` sidecar is written by `record` | 14 |

### 3d. Newly declared targets — candidate additions at the re-record

| Candidate | Source | Rule |
| --- | --- | --- |
| `hydrology_model/forcing/plots/temp.png` | O-24 | 1.13 |
| `hydrology_model/forcing/plots/pet.png` | O-24 | 1.13 |
| `hydrology_model/evaluation/plots/clim_wflow_1_month.png` | O-24 | 1.11 |
| `hydrology_model/evaluation/plots/clim_wflow_1_year.png` | O-24 | 1.11 |
| `hydrology_model/evaluation/performance_metrics.csv` | O-24 | 1.11 |
| `climate_historical/<key>/plots/source_precip.png` | B4 | 1.15 — **explicitly added to rule `all`** |
| `climate_historical/<key>/plots/source_temp.png` | B4 | 1.15 — as above |
| `climate_historical/<key>/plots/source_pet.png` | B4 | 1.15 — as above |

**Reading (see §7):** the design's verification row says only "newly-declared
targets are added to the manifest in the single re-record" and does not
discriminate rule-level declaration from `rule all` / `TARGETS` membership. This
map lists **all 8** as candidates. The three B4 PNGs are the least ambiguous (rule
`all` members by design); the five O-24 artifacts are rule-declared, and whether
each also enters `TARGETS` is an implementation decision to be **recorded at
commit 14**. The stale map's "+4" is superseded — it omitted
`performance_metrics.csv` and the three `source_*.png`.

**O-24's declared set is the config-invariant subset only.** `plot_results.py`
additionally drives `plot_basavg` (one PNG per basin-average entry in
`wflow_outvars`) and `plot_signatures` (`signatures_{station}.png`, when
observations exist and `nb_years >= 5`), and `clim_{station}_{period}.png` is
per-station. `--delete-all-output` completeness is claimed for the **seed-config
class only**.

### 3e. Machinery to update alongside — the complete list

| # | Item | Commit |
| --- | --- | --- |
| 1 | `check_baseline.py` `TARGETS` + `PROJECT_DIR_DEFAULT` retarget | **4** (sole owner, atomically with the fixture rename — ext2-03) |
| 2 | `semantic_tree_diff.py` — `build_r07_path_map()` / `build_r07_allowlist()` **plus a generic `--map old=new` CLI option** (today the map is hardcoded milestone code: `build_p31_path_map()` / `build_p31_allowlist()`; `main()` exposes only `--experiment-name`, `--dataset-key`, `--no-path-map`, `--allow`) | 1 |
| 3 | `semantic_tree_diff.py` — the declared many-to-one **merge class** (§2e) | 1 |
| 4 | `semantic_tree_diff.py` — `COPIED_CONFIG_PATH_MAP` (§2d) | 1 |
| 5 | The TOML comparator — **no change needed** (§2c) | — |
| 6 | `tests/` path bindings — at least eight modules (§1b) | with each move |
| 7 | `dev/contracts/hydrological-model-seam.md`, `dev/contracts/weather-generator-seam.md` | with B1/B2/B5 |

The **Commit** column on every table above is what makes the per-slice diffs
constructible: the path map for the slice after commit *N* is exactly the rows
with commit ≤ *N*.

**Gate blackout.** `check_baseline check` is red by construction from **commit 4
to commit 14** (recorded keys are old paths → "target missing on disk"; every
current path → "target present but not in manifest"). Three things cover the
window: per-slice `semantic_tree_diff` runs against the retained pre-R07 reference
tree after each of commits **7, 8, 11, 12**; the discharge `compare` anchor; and
an explicit note in each commit message that a red `check` in the window is
expected. **Commit 6 is a pause point, not a safe cut** — the tree is runnable and
`pytest tests/` is green there, but the baseline gate is red and the holding
artifacts (the pre-R07 reference tree and the saved discharge series) must be
preserved. The milestone has exactly **one completed state: after commit 14**
(docs commit 15 may trail). Abandoning mid-flight means reverting the landed
`r07:` commits, after which the pre-R07 manifest is valid again — no re-record in
either direction.

`--dry-run` is blind to `params:`-string paths and to R `shell:` bodies: **B1, B4,
B5, and B6 need a real run** to be proven.

---

## 4. MISSING/EXTRA allowlist

To be confirmed at the milestone gate, in the form of
`dev/p31/migration_experiment-structure.md` §2: every entry carries a written
justification, and an entry not listed fails the gate.

**The allowlist is a full set per gate invocation, not an increment over P3-1's.**
`build_r07_allowlist()` must therefore return P3-1's three carried-forward entries
as well as R07's four new ones — R07 retires none of them.

| Entry | Kind | Justification |
| --- | --- | --- |
| `climate_historical/<key>/store_region.geojson` | EXTRA | B1's second declared output — the model-free delineation the store bbox came from. No pre-R07 counterpart. Safe inside the guarded store dir: rule 3.00b compares config digests and writes two *named* sentinels; it never enumerates the directory |
| `climate_historical/<key>/plots/source_precip.png` | EXTRA | B4's new producer (rule 1.15). Additive; no existing target changes value |
| `climate_historical/<key>/plots/source_temp.png` | EXTRA | as above |
| `climate_historical/<key>/plots/source_pet.png` | EXTRA | as above — source-grid PET did not previously exist |
| `hydrology_model/forcing/plots/{temp,pet}.png` | *(not EXTRA)* | Already written today under `plots/wflow_model_performance/`; O-24 only **declares** them. They are ordinary mapped rows (§2a), not allowlist entries |
| `hydrology_model/evaluation/{clim_wflow_1_month.png, clim_wflow_1_year.png, performance_metrics.csv}` | *(not EXTRA)* | as above |
| `experiments/<id>/indicators/plots/` | *(not an entry)* | Reserved with no producer — empty on both sides |
| `experiments/<name>/.project_consistency_ok` | EXTRA | **Carried forward from P3-1** (`dev/p31/migration_experiment-structure.md` §2), unchanged by R07 |
| `climate_historical/<key>/.guard_ok` | EXTRA | **Carried forward from P3-1**, unchanged by R07 — B1 retires its *DAG edge*, not the artifact, and the path does not move |
| `experiments/<name>/config/<data_catalog>.yml` (via `--allow`; `deltares_data.yml` on the seed config) | EXTRA | **Carried forward from P3-1**, unchanged by R07 |
| `<project_dir>/dag/` | *(not an entry, unless the gate capture rendered DAGs)* | O-02's destination has no pre-R07 counterpart, and `dag` is **not** in `EXCLUDED_DIR_NAMES` (`{logs, benchmarks, .snakemake}`). It is not produced by `snakemake all` — only by the explicit `--dag` renders in `scripts/run_snake_test.cmd:32,39` and the README / notebook cells — so a reference tree captured from a plain workflow run will not contain it. **If** a gate capture is taken after a DAG render, allowlist it as EXTRA rather than treating it as a regression |

**MISSING — the stale map's "none by design" line is retracted.**
`climate_historical/wf1_raw/*` does **not** disappear as a rename. It collapses
**many-to-one** into `climate_historical/<key>/`, which no path-map entry can
express and which would abort `diff_trees` with a path-map collision. It is
handled as a **declared merge** (§2e), where both comparisons must pass — the
alternative (`--retire` + allowlist-as-MISSING) was **rejected**, because it lets
the gate go green while proving nothing about the store that disappeared, exactly
where GA-1 demands proof.

Allowlist entries are `semantic_tree_diff` machinery; manifest membership (§3d) is
`check_baseline` machinery. They are different gates — an artifact can be an
allowlisted EXTRA in one and a recorded target in the other.

---

## 5. Row count

**Table rows, not distinct paths.** §3a and §3d re-state paths already mapped in
§2a / §2b from the `check_baseline` side, so the total below double-counts by
design; it is a completeness check on the tables, not a count of paths moved.
Distinct old→new path mappings across §1, §2a, and §2b: **65**.

| Table | Table rows |
| --- | --- |
| §1 Repository paths | 18 |
| §1a Config key values | 8 |
| §2a Project level (moves / new / merge / retired) | 29 |
| §2a Unchanged-by-design (stated) | 8 |
| §2b Experiment subtree | 18 |
| §2c In-TOML pointers | 4 |
| §2d `COPIED_CONFIG_PATH_MAP` | 5 |
| §2e Declared merges | 2 |
| §3a Manifest live targets | 15 |
| §3b Stale-row deletions | 3 |
| §3d Newly declared candidates | 8 |
| §4 Allowlist entries | 7 EXTRA — 4 new + 3 carried forward from P3-1 (+4 stated non-entries) |
| **Total** | **125** |

---

## 6. Move-ID index

| ID | Change | Commit |
| --- | --- | --- |
| O-01 | Delete `data/`; ship observation templates; Linux config + Docker retarget | 2 |
| O-02 | DAG renders → `<project_dir>/dag/` | 5 |
| O-05 | Delete `docs/config/` | 3 |
| O-08 | `plot_map.py` `"None"`-sentinel guard | 12 |
| O-13 | Delete `blueearth_cst.Rproj` | 15 *(this map's assignment)* |
| O-20 | `examples/` → `test_case/`; `check_baseline.py` retarget | 4 |
| O-21 | Template `project_dir` default | 6 |
| O-22 | `warn_if_project_dir_in_repo()` | 6 |
| O-24 | Declare config-invariant plot outputs (rules 1.11, 1.13) | 12 |
| O-25 | Retire `get_region_preview.py` | 7 |
| OQ-2 | `MIGRATION.md` → `docs/migration-r06.md`; `naming.md` §7 amendment; `dev/r06/` record | 15 |
| B1 | Single climate store, shared region+catalog producer | 7 |
| B2 | wflow forcing into the engine subtree | 8 |
| B3 | Tier `climate_projections/` | 9 |
| B4 | Climate figures from the store (rule 1.15) | 12 |
| B5 | Two symmetric engine subtrees inside the experiment | 11 |
| B6 | `stress_test/` → `weather_generator/_work/`; declared input | 11 |
| B7 | `model_results/` → `indicators/` | 11 |
| B8 | `experiment_id` suggestion helper | 13 |
| B9 | Split `<project_dir>/config/` into `runs/ catalogs/ templates/ generated/` | 10 |
| B10 | wf1 evaluation and model figures into the engine subtree | 12 |

---

## 7. Slots to fill at the gate

| Slot | Filled at | Status |
| --- | --- | --- |
| Milestone tip SHA (header) | landing | open |
| Final MISSING/EXTRA allowlist confirmed against the milestone diff, in `CLEAN: N files compared, …` form | commit 14 | open |
| Exception 3 outcome — merge comparison pass/fail, and **per-edge coordinate deltas if it fails** | commit 7 (per-slice diff) | open |
| Which of §3d's 8 candidates entered `TARGETS` / the manifest | commit 14 | open |

## 8. Judgement calls this map made

Recorded so a reader can see where the accepted design underdetermined the map.

1. **O-13's commit.** Ruled (delete `blueearth_cst.Rproj`) at G1, but it appears
   in neither the 15-commit plan nor the drive-by list (O-07 / O-09 / O-10).
   Assigned to commit 15 as repo housekeeping alongside the other doc-tier moves.
2. **The reconstructed R06 record's filename.** The amended §7 requires
   `dev/r06/migration_<topic>.md` but does not pin `<topic>`.
   `migration_structural-refactor.md` is this map's choice, matching
   `dev/r06/structural-refactor-design.md`.
3. **Manifest additions at the re-record** — 8 candidates enumerated in §3d, with
   the rule-`all`-vs-rule-declaration ambiguity flagged rather than resolved.
4. **"14 within-tree movers" superseded by 10** (§3a), derived per-target from
   `check_baseline.TARGETS`.
5. **Run-TOML `input.path_forcing`** is written as `../forcing/inmaps_cst_<c>.nc`
   in §2c — the depth-consistent form for
   `hydrology_runs/rlz_<r>/config/` → `hydrology_runs/rlz_<r>/forcing/`. The
   design states the target directory but not the emitted relative string;
   hydromt re-relativizes on write, so the comparator, not this map, is
   authoritative.
