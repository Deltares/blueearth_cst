# R9 P1 falsifier — the path map against the declared-tier inventory

Date: 2026-08-04. Phase: R9 P1 (comparator and tooling). Gate: master Gate 1.

The claim under test is *"the map covers every artifact"*. A passing unit test
cannot show that, because `apply_path_map` returns its input unchanged both when
an identity rule fires and when nothing matches — so a fall-through is
indistinguishable from a deliberate non-move, and a map with **no rules at all**
would report every path as mapped. `apply_path_map_matched` reports whether a
rule actually fired; `--check-map` classifies each path as MOVED / IDENTITY /
DELETED / UNMAPPED and exits 1 on any UNMAPPED.

## Inventory

`dev/milestones/r09/declared_inventory.txt` — 176 project-relative paths from
the three Snakefiles' `output:` declarations, expanded over the tracked seed
config `config/workflows/snake_config_model_test.yml` with `project_dir`
repointed at an empty temp dir so every job is planned rather than skipped.
Provenance and the regeneration recipe are in that file's header.

**Not run against `test_case/test_local`** — a mixed-era tree whose documented
orphans are deliberately unmapped (map doc, *Orphans in the fixture — do NOT
map*), so it fails by construction on paths the map is right to reject.

## Run 1 — the map AS FIRST ENCODED, before the amendment

```
$ python dev/scripts/semantic_tree_diff.py --check-map \
    dev/milestones/r09/declared_inventory.txt --milestone r09 \
    --experiment-name experiment --dataset-key era5_20000101_20201231
UNMAPPED config/runs/climate_projections/61868971c618
UNMAPPED experiments/experiment/config/runs/climate_experiment/278159763309
UNMAPPED spatial/geoms/region.geojson
UNMAPPED PATHS: 176 paths, 162 moved, 11 identity (by rule), 0 deleted-by-design, 3 unmapped

UNMAPPED PATHS: 176 paths, 162 moved, 11 identity (by rule), 0 deleted-by-design, 3 unmapped
```

Exit 1. Three declared artifacts had **no row in the migration map**. They were
findings against the map, not defects in the comparator: the map's `data/` row
enumerated five geoms layers when the code writes six, its `config/runs/` row
transcribed one workflow when the design tree says `<workflow>`, and no row at
all covered WF3's experiment-scoped digest bundle.

**Ruled by the owner, 2026-08-04** (phase-1 report F1a–F1c). The migration map
was amended — Finding 2's "correspond exactly" claim corrected, two rows
generalised, one row added — and the three rules moved from the opt-in
`build_r09_gap_rules` into `build_r09_path_map`, where they now belong.

Two further candidates remain **unruled** and stay opt-in: `hydrology_model/instate/`
and `hydrology_model/plots/` as a directory. Neither appears in any `output:`
declaration and neither has been observed, so neither can be ruled until the
observed tier exists.

## Run 2 — after the amendment

Exit 0. **Zero unmapped, with no opt-in rules.** Byte-identical to the
pre-amendment run with `--r09-gap-rules`, which is what confirms the amendment
encoded exactly the three ruled rows and nothing more.

The full old → new table follows: this is the map applied to a pre-migration
path set, showing the intended post-migration paths. `IDENTITY` means a rule
fired and resolved the path to itself — a deliberately unchanged artifact, not a
fall-through.

```
IDENTITY benchmarks/wf1_benchmarks.md
IDENTITY benchmarks/wf2_benchmarks.md
MOVED    climate_historical/era5_20000101_20201231/.guard_ok  ->  data/climate/historical/era5_20000101_20201231/.guard_ok
MOVED    climate_historical/era5_20000101_20201231/extract_historical.nc  ->  data/climate/historical/era5_20000101_20201231/extract_historical.nc
MOVED    climate_historical/era5_20000101_20201231/plots/source_pet_annual.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_pet_annual.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_pet_map.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_pet_map.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_pet_monthly.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_pet_monthly.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_precip_annual.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_precip_annual.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_precip_map.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_precip_map.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_precip_monthly.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_precip_monthly.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_temp_annual.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_temp_annual.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_temp_map.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_temp_map.png
MOVED    climate_historical/era5_20000101_20201231/plots/source_temp_monthly.png  ->  data/climate/historical/era5_20000101_20201231/plots/source_temp_monthly.png
MOVED    climate_projections/cmip6/plots/cmip6_change_factor_cloud.png  ->  data/climate/projections/cmip6/plots/cmip6_change_factor_cloud.png
MOVED    climate_projections/cmip6/plots/cmip6_precip_annual_absolute.png  ->  data/climate/projections/cmip6/plots/cmip6_precip_annual_absolute.png
MOVED    climate_projections/cmip6/plots/cmip6_precip_annual_change.png  ->  data/climate/projections/cmip6/plots/cmip6_precip_annual_change.png
MOVED    climate_projections/cmip6/plots/cmip6_precip_monthly_absolute.png  ->  data/climate/projections/cmip6/plots/cmip6_precip_monthly_absolute.png
MOVED    climate_projections/cmip6/plots/cmip6_precip_monthly_change.png  ->  data/climate/projections/cmip6/plots/cmip6_precip_monthly_change.png
MOVED    climate_projections/cmip6/plots/cmip6_temp_annual_absolute.png  ->  data/climate/projections/cmip6/plots/cmip6_temp_annual_absolute.png
MOVED    climate_projections/cmip6/plots/cmip6_temp_annual_change.png  ->  data/climate/projections/cmip6/plots/cmip6_temp_annual_change.png
MOVED    climate_projections/cmip6/plots/cmip6_temp_monthly_absolute.png  ->  data/climate/projections/cmip6/plots/cmip6_temp_monthly_absolute.png
MOVED    climate_projections/cmip6/plots/cmip6_temp_monthly_change.png  ->  data/climate/projections/cmip6/plots/cmip6_temp_monthly_change.png
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM4-8_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM5-0_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM5-0_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_INM_INM-CM5-0_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_INM_INM-CM5-0_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/raw/cmip6_NOAA-GFDL_GFDL-ESM4_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/report.md  ->  data/climate/projections/cmip6/report.md
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM4-8_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM4-8_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM4-8_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM5-0_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM5-0_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_INM_INM-CM5-0_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_INM_INM-CM5-0_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1.nc
MOVED    climate_projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_ssp585_r1i1p1f1.nc  ->  data/climate/projections/cmip6/scalar/cmip6_NOAA-GFDL_GFDL-ESM4_ssp585_r1i1p1f1.nc
MOVED    climate_projections/cmip6/summary/cmip6_change_factors_annual.csv  ->  data/climate/projections/cmip6/summary/cmip6_change_factors_annual.csv
MOVED    climate_projections/cmip6/summary/cmip6_change_factors_monthly.csv  ->  data/climate/projections/cmip6/summary/cmip6_change_factors_monthly.csv
MOVED    climate_projections/cmip6/summary/composition.csv  ->  data/climate/projections/cmip6/summary/composition.csv
MOVED    climate_projections/cmip6/summary/provenance.json  ->  data/climate/projections/cmip6/summary/provenance.json
MOVED    config/generated/wflow_build_forcing_historical.yml  ->  models/hydrology/wflow/config/build_historical_forcing.yml
IDENTITY config/runs/climate_projections/61868971c618
IDENTITY config/runs/model_creation/1a22a14838f3
IDENTITY config/runs/snake_config_climate_projections.yml
IDENTITY config/runs/snake_config_model_creation.yml
IDENTITY experiments/experiment/.project_consistency_ok
IDENTITY experiments/experiment/benchmarks/wf3_benchmarks.md
IDENTITY experiments/experiment/config/runs/climate_experiment/278159763309
IDENTITY experiments/experiment/config/snake_config_climate_experiment.yml
MOVED    experiments/experiment/data_catalog_climate_experiment.yml  ->  experiments/experiment/config/catalogs/data_catalog_climate_experiment.yml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_1.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_1.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_2.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_2.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_3.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_3.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_4.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_4.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_5.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_5.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/config/cst_6.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_1_cst_6.toml
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_1.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_1.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_2.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_2.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_3.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_3.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_4.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_4.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_5.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_5.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/forcing/inmaps_cst_6.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_cst_6.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_1.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_1.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_2.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_2.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_3.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_3.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_4.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_4.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_5.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_5.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/cst_6.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_1_cst_6.csv
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_1.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_1.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_2.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_2.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_3.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_3.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_4.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_4.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_5.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_5.nc
MOVED    experiments/experiment/hydrology_runs/rlz_1/output/outstates_cst_6.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_1_cst_6.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_1.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_1.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_2.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_2.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_3.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_3.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_4.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_4.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_5.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_5.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/config/cst_6.toml  ->  experiments/experiment/hydrology/wflow/config/rlz_2_cst_6.toml
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_1.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_1.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_2.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_2.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_3.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_3.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_4.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_4.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_5.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_5.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/forcing/inmaps_cst_6.nc  ->  experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_2_cst_6.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_1.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_1.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_2.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_2.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_3.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_3.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_4.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_4.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_5.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_5.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/cst_6.csv  ->  experiments/experiment/hydrology/wflow/output/rlz_2_cst_6.csv
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_1.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_1.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_2.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_2.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_3.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_3.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_4.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_4.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_5.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_5.nc
MOVED    experiments/experiment/hydrology_runs/rlz_2/output/outstates_cst_6.nc  ->  experiments/experiment/hydrology/wflow/output/outstates_rlz_2_cst_6.nc
MOVED    experiments/experiment/indicators/Qstats.csv  ->  experiments/experiment/results/q_indicators.csv
MOVED    experiments/experiment/indicators/basin.csv  ->  experiments/experiment/results/basin_indicators.csv
IDENTITY experiments/experiment/logs/wf3_climate_experiment.log
MOVED    experiments/experiment/weather_generator/_work/cst_1.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_1.csv
MOVED    experiments/experiment/weather_generator/_work/cst_2.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_2.csv
MOVED    experiments/experiment/weather_generator/_work/cst_3.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_3.csv
MOVED    experiments/experiment/weather_generator/_work/cst_4.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_4.csv
MOVED    experiments/experiment/weather_generator/_work/cst_5.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_5.csv
MOVED    experiments/experiment/weather_generator/_work/cst_6.csv  ->  experiments/experiment/climate/weathergenr/_work/cst_6.csv
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_1.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_1.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_2.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_2.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_3.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_3.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_4.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_4.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_5.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_5.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_1_cst_6.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_1_cst_6.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_1.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_1.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_2.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_2.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_3.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_3.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_4.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_4.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_5.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_5.yml
MOVED    experiments/experiment/weather_generator/_work/weathergen_config_rlz_2_cst_6.yml  ->  experiments/experiment/climate/weathergenr/_work/weathergen_config_rlz_2_cst_6.yml
MOVED    experiments/experiment/weather_generator/config/weathergen_config.yml  ->  experiments/experiment/climate/weathergenr/config/weathergen_config.yml
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_0.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_0.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_1.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_1.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_2.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_2.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_3.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_3.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_4.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_4.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_5.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_5.nc
MOVED    experiments/experiment/weather_generator/output/rlz_1_cst_6.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_1_cst_6.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_0.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_0.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_1.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_1.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_2.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_2.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_3.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_3.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_4.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_4.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_5.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_5.nc
MOVED    experiments/experiment/weather_generator/output/rlz_2_cst_6.nc  ->  experiments/experiment/climate/weathergenr/output/rlz_2_cst_6.nc
MOVED    hydrology_model/.model_built  ->  models/hydrology/wflow/.model_built
MOVED    hydrology_model/.outputs_configured  ->  models/hydrology/wflow/.outputs_configured
MOVED    hydrology_model/evaluation/performance_metrics.csv  ->  models/hydrology/wflow/evaluation/performance_metrics.csv
MOVED    hydrology_model/evaluation/plots/clim_wflow_1_month.png  ->  models/hydrology/wflow/evaluation/plots/clim_wflow_1_month.png
MOVED    hydrology_model/evaluation/plots/clim_wflow_1_year.png  ->  models/hydrology/wflow/evaluation/plots/clim_wflow_1_year.png
MOVED    hydrology_model/evaluation/plots/hydro_wflow_1.png  ->  models/hydrology/wflow/evaluation/plots/hydro_wflow_1.png
MOVED    hydrology_model/forcing/inmaps_historical.nc  ->  models/hydrology/wflow/forcing/inmaps_historical.nc
MOVED    hydrology_model/forcing/plots/forcing_pet_annual.png  ->  models/hydrology/wflow/forcing/plots/forcing_pet_annual.png
MOVED    hydrology_model/forcing/plots/forcing_pet_map.png  ->  models/hydrology/wflow/forcing/plots/forcing_pet_map.png
MOVED    hydrology_model/forcing/plots/forcing_pet_monthly.png  ->  models/hydrology/wflow/forcing/plots/forcing_pet_monthly.png
MOVED    hydrology_model/forcing/plots/forcing_precip_annual.png  ->  models/hydrology/wflow/forcing/plots/forcing_precip_annual.png
MOVED    hydrology_model/forcing/plots/forcing_precip_map.png  ->  models/hydrology/wflow/forcing/plots/forcing_precip_map.png
MOVED    hydrology_model/forcing/plots/forcing_precip_monthly.png  ->  models/hydrology/wflow/forcing/plots/forcing_precip_monthly.png
MOVED    hydrology_model/forcing/plots/forcing_temp_annual.png  ->  models/hydrology/wflow/forcing/plots/forcing_temp_annual.png
MOVED    hydrology_model/forcing/plots/forcing_temp_map.png  ->  models/hydrology/wflow/forcing/plots/forcing_temp_map.png
MOVED    hydrology_model/forcing/plots/forcing_temp_monthly.png  ->  models/hydrology/wflow/forcing/plots/forcing_temp_monthly.png
MOVED    hydrology_model/plots/basin_area.pdf  ->  models/hydrology/wflow/plots/basin_area.pdf
MOVED    hydrology_model/plots/basin_area.png  ->  models/hydrology/wflow/plots/basin_area.png
MOVED    hydrology_model/run_default/output.csv  ->  models/hydrology/wflow/run_default/output.csv
MOVED    hydrology_model/staticgeoms/outlet_index.csv  ->  models/hydrology/wflow/staticgeoms/outlet_index.csv
MOVED    hydrology_model/staticgeoms/outlets.geojson  ->  models/hydrology/wflow/staticgeoms/outlets.geojson
MOVED    hydrology_model/staticgeoms/region.geojson  ->  models/hydrology/wflow/staticgeoms/region.geojson
MOVED    hydrology_model/staticgeoms/reservoirs_lakes_glaciers.txt  ->  models/hydrology/wflow/staticgeoms/reservoirs_lakes_glaciers.txt
MOVED    hydrology_model/staticmaps.nc  ->  models/hydrology/wflow/staticmaps.nc
MOVED    hydrology_model/wflow_sbm.toml  ->  models/hydrology/wflow/wflow_sbm.toml
IDENTITY logs/wf1_model_creation.log
IDENTITY logs/wf2_climate_projections.log
MOVED    spatial/geoms/basins.geojson  ->  data/spatial/geoms/basins.geojson
MOVED    spatial/geoms/catchments.geojson  ->  data/spatial/geoms/catchments.geojson
MOVED    spatial/geoms/locations.geojson  ->  data/spatial/geoms/locations.geojson
MOVED    spatial/geoms/region.geojson  ->  data/spatial/geoms/region.geojson
MOVED    spatial/geoms/rivers.geojson  ->  data/spatial/geoms/rivers.geojson
MOVED    spatial/geoms/subbasins.geojson  ->  data/spatial/geoms/subbasins.geojson
MOVED    spatial/location_registry.csv  ->  data/spatial/location_registry.csv
MOVED    spatial/spatial_catalog.yml  ->  data/spatial/spatial_catalog.yml
MOVED    spatial/spatial_maps.nc  ->  data/spatial/spatial_maps.nc
MOVED    spatial/spatial_report.yml  ->  data/spatial/spatial_report.yml

MAP CLEAN: 176 paths, 163 moved, 13 identity (by rule), 0 deleted-by-design, 0 unmapped
```

## Tiers

- **Declared tier** — the above. Zero unmapped once the three gap rules are
  accepted.
- **Observed tier** — one clean three-workflow run from the primary checkout,
  snapshotted as a sorted path list. **UNVERIFIED.** It does not exist, and
  producing it is an owner action outside this phase's scope (map doc,
  *Sequencing*). It is the only tier carrying undeclared engine artifacts, which
  `--dry-run` structurally cannot see. Gate 1 does not close until the map has
  been applied to it with zero unmapped.
