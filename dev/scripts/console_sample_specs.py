"""One spec per workflow. Every rule row is the module's own literal.

Cited by `file:line` in the comments so a reviewer can check any of them.
Interpolated values are staged; the strings around them are not.
"""

from __future__ import annotations

from render_console_sample import Failure, Job, Rule, Workflow

PROJECT = "test_case/test_rapid"


# ==========================================================================
# WF0 -- analyze_climate
#
# A TWO-SOURCE comparison run (era5 + chirps), everything to build. Chosen
# because it exercises what WF2 could not:
#   * rules 0.03 and 0.04 are built in a Python loop, so three rule objects
#     share one number and collapse to one plan row (`_plan_rule_name`);
#   * a fresh run has NO up-to-date rules, so the gutter is dropped entirely;
#   * `0.04b` sorts between `0.03` and `0.04` on the lexicographic rule order;
#   * two real WARNING rows, which WF2's transcript never reached.
# ==========================================================================

WF0 = Workflow(
    name="wf0 analyze_climate",
    project=PROJECT,
    config="test_case/project_config_rapid.yml",
    elapsed=214,
    tokens={
        # analyze_climate.smk:280 -- the historical ROOT, not one store.
        "data": "p:/wflow_global/hydromt",
        "climate": f"{PROJECT}/data/climate/historical",
    },
    rules=[
        Rule("0.01", "delineate_region"),
        Rule("0.02", "delineate_subbasins_and_rivers"),
        Rule(
            "0.03",
            "extract_climate_datasets_era5",
            summary="clip global climate to the basin",
        ),
        Rule(
            "0.03",
            "extract_climate_datasets_chirps",
            summary="clip global climate to the basin",
        ),
        Rule(
            "0.03b",
            "derive_plot_scales",
            summary="one plotting scale per variable, across sources",
        ),
        Rule("0.04", "plot_climate_datasets_era5"),
        Rule("0.04", "plot_climate_datasets_chirps"),
        Rule(
            "0.05",
            "compare_climate_datasets",
            summary="compare the candidate climate datasets",
        ),
        Rule("0.06", "gather_benchmarks"),
        Rule("0.07", "gather_logs"),
    ],
    stats="""Job stats:
job                                count
-------------------------------  -------
all                                    1
delineate_region                       1
delineate_subbasins_and_rivers                1
extract_climate_datasets_era5        1
extract_climate_datasets_chirps      1
derive_plot_scales                     1
plot_climate_datasets_era5               1
plot_climate_datasets_chirps             1
compare_climate_datasets                1
gather_benchmarks                      1
gather_logs                            1
total                                 11
""",
    targets=[
        f"{PROJECT}/data/climate/historical/era5/plots/era5_precip_map_basin.png",
        f"{PROJECT}/data/climate/historical/chirps/plots/chirps_precip_map_basin.png",
        f"{PROJECT}/data/climate/historical/_comparison/climate_source_comparison.csv",
        f"{PROJECT}/data/climate/historical/_comparison/climate_source_comparison.md",
        f"{PROJECT}/data/spatial/basins.geojson",
        f"{PROJECT}/config/analyze_climate/run_record.yml",
        f"{PROJECT}/logs/wf0_analyze_climate.log",
        f"{PROJECT}/benchmarks/wf0_benchmarks.md",
    ],
    jobs=[
        # spatial/products.py: `Delineating region ...`, `Wrote region: ...`
        Job(
            "delineate_region",
            seconds=6,
            body=(
                (
                    "spatial",
                    "Delineating region {'subbasin': [37.6, 38.1]} on merit_hydro",
                    1,
                ),
                ("spatial", "Wrote region: <project>/data/spatial/region.geojson", 4),
            ),
        ),
        # spatial/products.py:873, :868 (WARNING), plus the basin-cells row
        Job(
            "delineate_subbasins_and_rivers",
            seconds=7,
            body=(
                (
                    "spatial",
                    "River network: 48 reach(es) at 25 km2, reaching all 3 location(s)",
                    2,
                ),
                # products.py:863 -- the REAL text, which is an invariant-violation
                # notice, not a "we fixed it" one.
                (
                    "spatial",
                    "1 location(s) are not on the derived river network: outlet_2. "
                    "They were snapped onto river_mask, so this should be "
                    "unreachable -- the network vectorization and the mask "
                    "have diverged.",
                    3,
                    "WARNING",
                ),
            ),
        ),
        # climate_analysis/extract_historical_climate.py
        Job(
            "extract_climate_datasets_era5",
            seconds=22,
            body=(
                ("climate", "Extracting historical climate grid", 1),
                (
                    "extract",
                    "Basin touches 84 of 240 store cells -> basin_cells.csv",
                    3,
                ),
                ("climate", "Saving to netcdf", 16),
            ),
        ),
        Job(
            "extract_climate_datasets_chirps",
            seconds=34,
            body=(
                ("climate", "Extracting historical climate grid", 1),
                (
                    "climate",
                    "chirps only contains precipitation data. Combining with "
                    "climate data from era5",
                    2,
                ),
                (
                    "climate",
                    "Downscaling era5 variables to the resolution of chirps",
                    2,
                ),
                ("climate", "Orography for chirps from merit_hydro (downscaling)", 6),
                (
                    "extract",
                    "Basin touches 312 of 900 store cells -> basin_cells.csv",
                    4,
                ),
                ("climate", "Saving to netcdf", 15),
            ),
        ),
        Job(
            "derive_plot_scales",
            seconds=4,
            body=(
                ("scales", "Wrote shared climate scales -> shared_plot_scales.json", 3),
            ),
        ),
        # climate_analysis/plot_climate_source.py:276, and the figure bundle
        # row from snake_utils.py:4055
        Job(
            "plot_climate_datasets_era5",
            seconds=31,
            body=(
                (
                    "plot",
                    "Reading store (era5): <climate>/era5/extract_historical.nc",
                    1,
                ),
                ("plot", "Aggregating over 4 area(s): basin + 3 subbasin(s)", 3),
                ("plot", "12 figures -> <climate>/era5/plots", 22),
                ("plot", "9 figures -> <climate>/era5/plots/subbasins", 3),
            ),
        ),
        Job(
            "plot_climate_datasets_chirps",
            seconds=19,
            body=(
                (
                    "plot",
                    "Reading store (chirps): <climate>/chirps/extract_historical.nc",
                    1,
                ),
                (
                    "plot",
                    "chirps is precipitation-only: precip figures only, "
                    "no temperature or PET",
                    2,
                ),
                ("plot", "Aggregating over 4 area(s): basin + 3 subbasin(s)", 1),
                ("plot", "4 figures -> <climate>/chirps/plots", 12),
            ),
        ),
        # climate_analysis/compare_sources.py -- including its WARNING at :713
        Job(
            "compare_climate_datasets",
            seconds=12,
            body=(
                ("compare", "Comparing 2 sources (chirps, era5) on precip", 1),
                (
                    "compare",
                    "The sources' extracted periods do not overlap; each is drawn "
                    "over its own record and the figures say so",
                    2,
                    "WARNING",
                ),
                (
                    "compare",
                    "Common ground: 1991-01-01..2020-12-31; 4 area(s); basin cells: chirps 312, era5 84",
                    2,
                ),
                ("compare", "Pooled precip over 2 source(s): chirps, era5", 2),
                (
                    "compare",
                    "Comparison table (2 sources) -> climate_source_comparison.csv, "
                    "climate_source_comparison.md",
                    3,
                ),
            ),
        ),
        Job("gather_benchmarks", seconds=1),
        Job("gather_logs", seconds=1),
        Job("all", seconds=0),
    ],
)


# ==========================================================================
# WF2 -- analyze_projections
#
# A RE-RUN: the region and spatial-unit rules are already satisfied, so the
# plan block shows the partial case (a `>` gutter and two `-` rows) that WF0's
# fresh run cannot. Fans out x4 over the CMIP6 series.
# ==========================================================================

WF2 = Workflow(
    name="wf2 analyze_projections",
    project=PROJECT,
    config="test_case/project_config_rapid.yml",
    elapsed=94,
    rules=[
        Rule("2.01", "delineate_region"),
        Rule("2.02", "delineate_subbasins_and_rivers"),
        Rule(
            "2.03",
            "fetch_cmip6_projections",
            "series {series_key}",
            "download one CMIP6 slice",
        ),
        Rule(
            "2.04",
            "reduce_to_basin_averages",
            "series {series_key}",
            "reduce the slice to a basin-average series",
        ),
        Rule(
            "2.05",
            "derive_change_factors",
            summary="compare each horizon against the reference period",
        ),
        Rule(
            "2.05b",
            "prepare_projection_analysis",
            summary="prepare analysis-window diagnostics",
        ),
        Rule("2.06", "plot_climate_projections"),
        Rule("2.07", "gather_benchmarks"),
        Rule("2.08", "gather_logs"),
    ],
    stats="""Job stats:
job                       count
----------------------  -------
all                           1
fetch_cmip6_projections               4
reduce_to_basin_averages             4
derive_change_factors         1
prepare_projection_analysis  1
plot_climate_projections           1
gather_benchmarks             1
gather_logs                   1
total                        14
""",
    targets=[
        f"{PROJECT}/data/climate/projections/cmip6/summary/cmip6_change_factors_annual.csv",
        f"{PROJECT}/data/climate/projections/cmip6/summary/cmip6_change_factors_monthly.csv",
        f"{PROJECT}/data/climate/projections/cmip6/_engine/analysis_record.json",
        f"{PROJECT}/data/climate/projections/cmip6/plots/full_period",
        f"{PROJECT}/data/climate/projections/cmip6/plots/period_2036_2065",
        f"{PROJECT}/data/climate/projections/cmip6/plots/figure_captions.md",
        f"{PROJECT}/data/climate/projections/cmip6/_engine/figure_captions.json",
        f"{PROJECT}/logs/wf2_analyze_projections.log",
        f"{PROJECT}/benchmarks/wf2_analyze_projections.csv",
    ],
    jobs=[
        # fetch_gcm_raw.py:766, :848, :1001, :1032
        Job(
            "fetch_cmip6_projections",
            seconds=3,
            wildcards={"series_key": "cmip6_CSIRO-ARCCSS_ACCESS-CM2_ssp245_r1i1p1f1"},
            body=(
                ("fetch", "Fetching cmip6_CSIRO-ARCCSS/ACCESS-CM2_ssp245_r1i1p1f1", 1),
                ("fetch", "Pinned gn/v20210317, no bucket listing", 2),
                ("fetch", "store calendar=proleptic_gregorian (no pin)", 1),
                (
                    "fetch",
                    "Wrote raw cmip6_CSIRO-ARCCSS_ACCESS-CM2_ssp245_r1i1p1f1.nc "
                    "(0.07 MB, 1032 steps)",
                    14,
                ),
            ),
        ),
        Job(
            "fetch_cmip6_projections",
            seconds=24,
            wildcards={"series_key": "cmip6_CSIRO-ARCCSS_ACCESS-CM2_ssp585_r1i1p1f1"},
        ),
        Job(
            "fetch_cmip6_projections",
            seconds=19,
            wildcards={
                "series_key": "cmip6_EC-Earth-Consortium_EC-Earth3_ssp245_r1i1p1f1"
            },
        ),
        Job(
            "fetch_cmip6_projections",
            seconds=23,
            wildcards={
                "series_key": "cmip6_EC-Earth-Consortium_EC-Earth3_ssp585_r1i1p1f1"
            },
        ),
        # get_stats_climate_proj.py: `{model} {scenario} {member} reducing raw -> series`
        Job(
            "reduce_to_basin_averages",
            seconds=3,
            wildcards={"series_key": "cmip6_CSIRO-ARCCSS_ACCESS-CM2_ssp245_r1i1p1f1"},
            body=(("reduce", "ACCESS-CM2 ssp245 r1i1p1f1 reducing raw -> series", 1),),
        ),
        Job(
            "reduce_to_basin_averages",
            seconds=3,
            wildcards={"series_key": "cmip6_CSIRO-ARCCSS_ACCESS-CM2_ssp585_r1i1p1f1"},
            body=(("reduce", "ACCESS-CM2 ssp585 r1i1p1f1 reducing raw -> series", 1),),
        ),
        Job(
            "reduce_to_basin_averages",
            seconds=2,
            wildcards={
                "series_key": "cmip6_EC-Earth-Consortium_EC-Earth3_ssp245_r1i1p1f1"
            },
            body=(("reduce", "EC-Earth3 ssp245 r1i1p1f1 reducing raw -> series", 1),),
        ),
        Job(
            "reduce_to_basin_averages",
            seconds=3,
            wildcards={
                "series_key": "cmip6_EC-Earth-Consortium_EC-Earth3_ssp585_r1i1p1f1"
            },
            body=(("reduce", "EC-Earth3 ssp585 r1i1p1f1 reducing raw -> series", 1),),
        ),
        # derive_change_factors.py:409, :429 (keys from reference_window.py:172), :562, :567
        Job(
            "derive_change_factors",
            seconds=1,
            body=(
                ("change", "Deriving change factors for 2 points x 1 horizon", 1),
                (
                    "change",
                    "reference_window reference_window_clipped=False "
                    "reference_window_effective=1985-2014 "
                    "reference_window_requested=1985-2014 "
                    "reference_window_years=30",
                    1,
                ),
                (
                    "change",
                    "Tidy monthly change-factor table: 144 rows "
                    "-> cmip6_change_factors_monthly.csv",
                    2,
                ),
                (
                    "change",
                    "Tidy annual change-factor table: 12 rows "
                    "-> cmip6_change_factors_annual.csv",
                    1,
                ),
            ),
        ),
        Job(
            "prepare_projection_analysis",
            seconds=2,
            body=(
                ("analysis", "Used 1 window: 2036-2065", 1),
                ("analysis", "Available 14 products; not produced 2", 1),
            ),
        ),
        # plot_proj_timeseries.py:render_suite
        Job(
            "plot_climate_projections",
            seconds=4,
            body=(
                (
                    "plot_proj_timeseries",
                    "Windows used: 2036-2065 (reference 2000–2014)",
                    1,
                ),
                (
                    "plot_proj_timeseries",
                    "14 figures written; 2 products unavailable",
                    3,
                ),
            ),
        ),
        Job("gather_benchmarks", seconds=1),
        Job("gather_logs", seconds=1),
        Job("all", seconds=0),
    ],
)


# ==========================================================================
# WF1 -- build_model
#
# The RE-RUN the plan block was designed against: `_plan_block`'s own docstring
# measures it -- "WF1 declares 19 rules and a completed build leaves 5 with
# work" (rapid fixture, 2026-09-05). Chosen because it is the workflow with the
# widest plan and the narrowest run, and because 1.14 is the long rule: under
# `[logging] silent = true` Wflow prints nothing, so the console shows our own
# progress bar and the heartbeat instead.
# ==========================================================================

WF1 = Workflow(
    name="wf1 build_model",
    project=PROJECT,
    config="test_case/project_config_rapid.yml",
    elapsed=436,
    tokens={
        "data": "p:/wflow_global/hydromt",
        "model": f"{PROJECT}/hydrology_model",
        "climate": f"{PROJECT}/data/climate/historical/era5",
    },
    rules=[
        Rule("1.01", "delineate_region"),
        Rule("1.02", "delineate_subbasins_and_rivers"),
        Rule(
            "1.03",
            "extract_climate_datasets",
            summary="clip the global climate dataset to the basin",
        ),
        Rule("1.04", "plot_climate_datasets"),
        Rule("1.05", "prepare_land_and_soil_maps"),
        Rule(
            "1.06",
            "build_wflow_model",
            summary="parameterize Wflow-SBM from global data",
        ),
        Rule(
            "1.07",
            "add_reservoirs_lakes_glaciers",
            summary="add waterbodies to the model",
        ),
        Rule("1.08", "declare_gauges_and_outputs"),
        Rule(
            "1.09",
            "add_climate_forcing",
            summary="build the forcing netCDF for the run period",
        ),
        Rule("1.10", "write_gauge_index"),
        Rule("1.11", "plot_basin_map"),
        Rule("1.12", "plot_model_forcing"),
        Rule(
            "1.13",
            "run_historical_simulation",
            summary="run the model over the simulation window",
        ),
        Rule("1.14", "export_simulation_tables"),
        Rule("1.15", "plot_model_evaluation"),
        Rule("1.16", "write_run_metadata"),
        Rule("1.17", "gather_benchmarks"),
        Rule("1.18", "gather_logs"),
    ],
    stats="""Job stats:
job                      count
---------------------  -------
all                          1
run_historical_simulation                    1
export_simulation_tables          1
plot_model_evaluation        1
gather_benchmarks            1
gather_logs                  1
total                        6
""",
    targets=[
        f"{PROJECT}/hydrology_model/run_default/output_q.csv",
        f"{PROJECT}/hydrology_model/plots/evaluation/discharge_timeseries.png",
        f"{PROJECT}/hydrology_model/run_default/run_metadata.json",
        f"{PROJECT}/logs/wf1_build_model.log",
        f"{PROJECT}/benchmarks/wf1_benchmarks.md",
    ],
    jobs=[
        # 1.14 -- the long one. Wflow is silent, so what the console shows is
        # our own bar (shared/progress.py: `render_bar`) redrawn in place, and
        # the heartbeat when a redraw gap gets long. Only the LAST bar frame
        # survives on a captured transcript; one mid-run frame is shown here.
        Job(
            "run_historical_simulation",
            seconds=180,
            body=(
                # snake_utils.py:2655 -- `format_elapsed`, i.e. h:mm:ss, the same
                # spelling as the DONE lines, the run summary, the benchmark
                # tables and the progress bar.
                # INFO, not WARNING, and the difference is visible: the
                # watchdog paints this row yellow but emits it at INFO, so
                # `_log_row_text` prints no level token. Spelling it WARNING
                # here put a word on the transcript that no run prints.
                (
                    "heartbeat",
                    "Rule 1.13: run_historical_simulation still running, 0:04:00 elapsed",
                    240,
                ),
            ),
        ),
        Job(
            "export_simulation_tables",
            seconds=9,
            body=(("tables", "Wrote 4 table(s) -> <model>/run_default", 6),),
        ),
        Job(
            "plot_model_evaluation",
            seconds=26,
            body=(
                (
                    "plot",
                    "Spatial overlay absent, skipped: <model>/staticgeoms/gauges.geojson",
                    2,
                ),
                ("plot", "6 figures -> <model>/plots/evaluation", 18),
            ),
        ),
        Job("gather_benchmarks", seconds=2),
        Job("gather_logs", seconds=2),
        Job("all", seconds=0),
    ],
    # WF1 carries the failure shape because 1.14 is where a run of this
    # toolbox actually dies: Wflow is the one step running foreign code over
    # data the build assembled, and a bad forcing window or an unsolvable
    # state surfaces as a non-zero exit with nothing above it but our own
    # heartbeat. The block below is Snakemake's, not ours -- what the console
    # does to it is relativize its paths onto the run's tokens, which is the
    # behaviour the failure transcript exists to show.
    failure=Failure(
        rule="run_historical_simulation",
        seconds=9,
        body=(
            (
                "heartbeat",
                "Rule 1.13: run_historical_simulation still running, 0:04:00 elapsed",
                240,
            ),
        ),
        error="""Error in rule run_historical_simulation:
    jobid: 1
    input: test_case/test_rapid/hydrology_model/wflow_sbm.toml
    output: test_case/test_rapid/hydrology_model/run_default/output.csv
    log: test_case/test_rapid/logs/_parts/1.13_run_wflow.log (check log file(s) for error details)
    shell:
        julia --project=julia_env -e "using Wflow; Wflow.run()" test_case/test_rapid/hydrology_model/wflow_sbm.toml
        (one of the commands exited with non-zero exit code; note that snakemake uses bash strict mode!)

Shutting down, this might take some time.
Exiting because a job execution failed. Look above for error message""",
        log_parts_dir="logs/_parts",
    ),
)


# ==========================================================================
# WF3 -- generate_scenarios
#
# The FAN-OUT workflow: where a real run spends hours, and the one the
# `rule_banner` context and `_trim_summary` were designed for. 2 realizations x
# 4 stress-test members here; a production grid is 10 x 20. Two rules fan out,
# and they interleave, so the console alternates between two identities.
# ==========================================================================

WF3 = Workflow(
    name="wf3 generate_scenarios",
    project=PROJECT,
    config="test_case/project_config_rapid.yml",
    elapsed=203,
    details={"experiment": "experiment_rapid"},
    tokens={
        "data": "p:/wflow_global/hydromt",
        "climate": f"{PROJECT}/data/climate/historical/era5",
    },
    rules=[
        Rule("3.01", "delineate_region"),
        Rule("3.02", "extract_climate_datasets"),
        Rule("3.03", "prepare_perturbation_grid"),
        Rule(
            "3.04",
            "snapshot_generation_inputs",
            summary="fingerprint the generation inputs into a scenario request",
        ),
        Rule("3.05", "claim_scenario_collection"),
        Rule("3.06", "prepare_weather_generator_settings"),
        Rule(
            "3.07",
            "generate_weather_realizations",
            summary="generate stochastic weather with weathergenr",
        ),
        Rule(
            "3.08",
            "perturb_climate_realization",
            "rlz {rlz_num} | st {st_num}",
            "apply one stress-test member to one realization",
        ),
        Rule(
            "3.08b",
            "retain_scenario_forcing",
            "collection {collection_id} | run {run_id}",
        ),
        Rule(
            "3.09",
            "publish_scenario_collection",
            "collection {collection_id}",
            "seal the collection and make it immutable",
        ),
        Rule("3.11", "gather_logs"),
        Rule("3.10", "gather_benchmarks"),
    ],
    stats="""Job stats:
job                              count
-----------------------------  -------
all                                  1
snapshot_generation_inputs           1
claim_scenario_collection       1
prepare_weather_generator_settings            1
generate_weather_realizations        1
perturb_climate_realization          8
retain_scenario_forcing              8
publish_scenario_collection          1
gather_logs                          1
gather_benchmarks                    1
total                               24
""",
    targets=[
        f"{PROJECT}/scenarios/collections/da182e806b6b/collection.json",
        f"{PROJECT}/logs/wf3_generate_scenarios.log",
        f"{PROJECT}/benchmarks/wf3_benchmarks.md",
    ],
    jobs=[
        # generation_plan.py / generation_publication.py
        Job(
            "snapshot_generation_inputs",
            seconds=5,
            body=(
                (
                    "collection",
                    "Collection da182e806b6b: claiming 8 scenario row(s), "
                    "3 portable input(s)",
                    3,
                ),
            ),
        ),
        Job("claim_scenario_collection", seconds=2),
        Job(
            "prepare_weather_generator_settings",
            seconds=3,
            body=(
                (
                    "weathergen",
                    "Preparing and writing the weather generator config file "
                    "<project>/scenarios/weathergen_config.yml",
                    2,
                ),
            ),
        ),
        Job(
            "generate_weather_realizations",
            seconds=26,
            body=(
                (
                    "request",
                    "Request scenario_request_9f2c: 2 realization(s) x 4 design "
                    "point(s) from 3 source file(s)",
                    2,
                ),
            ),
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "1", "st_num": "1"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "01"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "1", "st_num": "2"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "02"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "1", "st_num": "3"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "03"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "1", "st_num": "4"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "04"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "2", "st_num": "1"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "05"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "2", "st_num": "2"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "06"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "2", "st_num": "3"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "07"},
        ),
        Job(
            "perturb_climate_realization",
            seconds=8,
            wildcards={"rlz_num": "2", "st_num": "4"},
        ),
        Job(
            "retain_scenario_forcing",
            seconds=1,
            wildcards={"collection_id": "da182e806b6b", "run_id": "08"},
        ),
        Job(
            "publish_scenario_collection",
            seconds=4,
            wildcards={"collection_id": "da182e806b6b"},
            body=(
                (
                    "experiment",
                    "Experiment config recorded for 'experiment_rapid' (7 setting(s))",
                    2,
                ),
            ),
        ),
        Job("gather_logs", seconds=1),
        Job("gather_benchmarks", seconds=1),
        Job("all", seconds=0),
    ],
)


# ==========================================================================
# WF4 -- simulate_system
#
# The simulation half: 8 retained runs downscaled, then run through Wflow in
# two batches, then reduced. Driven by `scripts/simulate_system.py`, and the
# only workflow that passes TWO details to the header (`experiment` and
# `operation`) -- which is what makes the `PATHS` naming question concrete.
# It also has no `rule all`: `4.07 responses` and `4.10 indicators` are its
# target aggregators, and they carry ordinary rule banners.
# ==========================================================================

WF4 = Workflow(
    name="wf4 simulate_system",
    project=PROJECT,
    config="test_case/project_config_rapid.yml",
    elapsed=487,
    # `operation` is a config setting -- `simulate-and-indicators` or
    # `indicators-only` (simulation_runner.py:29) -- not the `--target` value.
    details={"experiment": "experiment_rapid", "operation": "simulate-and-indicators"},
    rules=[
        Rule("4.01", "write_model_fingerprint"),
        Rule("4.02", "check_model_unchanged"),
        Rule(
            "4.03",
            "snapshot_simulation_inputs",
            summary="pin the simulation's inputs so the experiment cannot drift",
        ),
        Rule(
            "4.04",
            "downscale_scenario_series",
            "run {run_id}",
            "downscale the perturbed climate onto the model grid",
        ),
        Rule(
            "4.05",
            "run_wflow_simulations_batch_0",
            "4 members",
            "run Wflow for one batch",
        ),
        Rule(
            "4.05",
            "run_wflow_simulations_batch_1",
            "4 members",
            "run Wflow for one batch",
        ),
        Rule(
            "4.06",
            "publish_wflow_outputs",
            summary="inventory the retained native runs",
        ),
        Rule(
            "4.07", "prepare_indicator_plan", "indicator_request {indicator_request_id}"
        ),
        Rule(
            "4.08",
            "derive_system_indicators",
            "indicator_set {indicator_set_id}",
            "reduce the retained responses to the immutable indicator set",
        ),
        Rule("4.09", "gather_logs"),
        Rule("4.10", "gather_benchmarks"),
    ],
    stats="""Job stats:
job                             count
----------------------------  -------
write_model_fingerprint               1
check_model_unchanged               1
snapshot_simulation_inputs             1
downscale_scenario_series       8
run_wflow_simulations_batch_0                   1
run_wflow_simulations_batch_1                   1
publish_wflow_outputs            1
simulations_only                    1
prepare_indicator_plan                 1
derive_system_indicators                  1
simulations_and_indicators          1
gather_logs                         1
gather_benchmarks                   1
total                              20
""",
    targets=[],
    jobs=[
        Job("write_model_fingerprint", seconds=3),
        # check_model_reference.py:103
        Job(
            "check_model_unchanged",
            seconds=1,
            body=(
                (
                    "experiment",
                    "Model reference matches the live model; simulation may proceed",
                    1,
                ),
            ),
        ),
        # simulation_record.py:321
        Job(
            "snapshot_simulation_inputs",
            seconds=11,
            body=(
                (
                    "simulation",
                    "Froze simulation 5c1a9e0b77d2: collection da182e806b6b "
                    "(exact) against model 84fe10c3a9bb",
                    8,
                ),
            ),
        ),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "01"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "02"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "03"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "04"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "05"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "06"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "07"}),
        Job("downscale_scenario_series", seconds=14, wildcards={"run_id": "08"}),
        Job("run_wflow_simulations_batch_0", seconds=96),
        Job("run_wflow_simulations_batch_1", seconds=93),
        # response_inventory.py:429
        Job(
            "publish_wflow_outputs",
            seconds=7,
            body=(
                (
                    "responses",
                    "Inventoried 8 native run(s), 24 series -> 6b2f70a1c934",
                    5,
                ),
            ),
        ),
        Job("simulations_only", seconds=1),
        # indicator_plan.py:431
        Job(
            "prepare_indicator_plan",
            seconds=3,
            wildcards={"indicator_request_id": "3ad81f5c2e70"},
            body=(
                (
                    "indicators",
                    "Indicator plan 9f41c7d0b6a2: 4 declaration(s) over 12 result key(s)",
                    2,
                ),
            ),
        ),
        # indicator_plan.py:718
        Job(
            "derive_system_indicators",
            seconds=19,
            wildcards={"indicator_set_id": "9f41c7d0b6a2"},
            body=(
                (
                    "indicators",
                    "Published indicator set 9f41c7d0b6a2: q_mean (240 rows), "
                    "q_p10 (240 rows), q_p90 (240 rows)",
                    12,
                ),
            ),
        ),
        Job("simulations_and_indicators", seconds=1),
        Job("gather_logs", seconds=2),
        Job("gather_benchmarks", seconds=2),
    ],
)


WORKFLOWS = {"wf2": WF2, "wf0": WF0, "wf1": WF1, "wf3": WF3, "wf4": WF4}
