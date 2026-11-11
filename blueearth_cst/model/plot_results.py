# Current guidance: dev/reference/wf1-model-construction.md
# -*- coding: utf-8 -*-
"""
Plot wflow results and compare to observations if any
"""

import os
from os.path import join
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.readers import open_timeseries_from_table
from hydromt_wflow import WflowSbmModel

from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
from blueearth_cst.model.observation_validation import (
    validate_observation_station_ids,
)
from blueearth_cst.shared.func_plot_signature import (
    compute_indicators,
    plot_basavg,
)
from blueearth_cst.shared.plot_evaluation import (
    STATION_PLOT_DIRNAME,
    Station,
    plot_station_evaluation,
)
from blueearth_cst.shared.response_tables import open_table_responses, read_table_record
from blueearth_cst.shared.snake_utils import log_row, plural
from blueearth_cst.shared.wflow_outputs import (
    SUBCATCHMENT_SUFFIX,
)


def _log(message):
    """Emit one standard-format log row (module tag ``plot``) for this rule."""
    log_row(message, module="plot")


def evaluation_arrays(record_path, location_registry):
    """Read rounded location tables and recover basin support from the registry."""
    from blueearth_cst.experiment.wflow_response_reader import NATIVE_VARIABLES

    record = read_table_record(record_path)
    responses = open_table_responses(
        "historical", record_path, ResponseRequest(tuple(record["variables"]))
    )
    registry = pd.read_csv(location_registry, dtype={"wflow_id": str})
    if registry.wflow_id.duplicated().any():
        raise ValueError("Location registry has duplicate wflow_id values")
    registry = registry.set_index("wflow_id")
    q = [item for item in responses if item.variable == "q"]
    if not q:
        raise ValueError("Historical evaluation requires retained discharge")
    unknown = {item.location_id for item in responses} - set(registry.index)
    if unknown:
        raise ValueError(f"Response locations absent from registry: {sorted(unknown)}")
    qsim = xr.DataArray(
        np.column_stack([item.values for item in q]),
        dims=("time", "index"),
        coords={
            "time": q[0].time,
            "index": [int(item.location_id) for item in q],
            "station_name": (
                "index",
                [str(registry.loc[item.location_id, "station_name"]) for item in q],
            ),
        },
        name="Q",
    )
    basin = xr.Dataset()
    for variable in record["variables"]:
        if variable in {"q", "precip"}:
            continue
        series = [item for item in responses if item.variable == variable]
        ids = [int(registry.loc[item.location_id, "subbasin_id"]) for item in series]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate subbasin response support: {variable}")
        basin[f"{NATIVE_VARIABLES[variable][0].lower()}{SUBCATCHMENT_SUFFIX}"] = (
            xr.DataArray(
                np.column_stack([item.values for item in series]),
                dims=("time", "index"),
                coords={"time": series[0].time, "index": ids},
            )
        )
    return qsim, basin


def resolve_stations(qsim, location_registry, log=_log):
    """Map every plotted series onto the ``wflow_id`` the rest of the run uses.

    The series arrive keyed two different ways. A gauge series is indexed by its
    ``wflow_id`` already (``Q_1010``); the OUTLET series is indexed by its
    subcatchment id (``Q_101``), which is not a wflow_id at all. The location
    registry closes the gap: every subbasin has exactly one PRIMARY location,
    and that location's ``wflow_id`` is what names the outlet.

    Two consequences worth stating, because both used to be defects:

    * A gauge sitting on a model outlet is now recognised as ONE point even when
      the two series carry different index values — 101 and 1010 are the same
      cell here. Raw indices can differ, so a project with output_locations at its
      outlet drew that station's figures twice, under two names.
    * A series whose index resolves to nothing keeps its own index as the id and
      says so. Guessing would be worse: the figures would be named after
      something no other artifact of the run recognises.

    Returns ``{index: Station}``.
    """
    by_subbasin, by_wflow_id = {}, {}
    if location_registry is not None:
        registry = pd.read_csv(location_registry)
        # `.get(col, True)` returns the SCALAR default when the column is
        # absent, which has no `.astype` — so the fallback is built explicitly.
        # Every registry this repo writes carries `is_primary`; a defensive
        # branch that raises is worse than no branch at all.
        if "is_primary" in registry:
            primary = registry.loc[registry["is_primary"].astype(bool)]
        else:
            primary = registry
        for row in primary.itertuples(index=False):
            by_subbasin[int(row.subbasin_id)] = row
        for row in registry.itertuples(index=False):
            by_wflow_id[int(row.wflow_id)] = row

    stations, seen = {}, {}
    for index, name in zip(qsim["index"].values, qsim["station_name"].values):
        index = int(index)
        # wflow_id first: a series already keyed that way is not ambiguous, and
        # the subbasin lookup is the fallback the outlet needs. Explicit
        # membership rather than `a or b` — a row is a namedtuple, and relying
        # on its truthiness is a trap the day one has no fields.
        row = by_wflow_id.get(index)
        if row is None:
            row = by_subbasin.get(index)
        if row is None:
            log(
                f"Series {index} has no matching location-registry identity; "
                "using its original ID for figure names"
            )
            stations[index] = Station(index, station_name=str(name))
            continue
        station = Station(
            wflow_id=int(row.wflow_id),
            subbasin_id=int(row.subbasin_id),
            # The registry's own name, not the synthetic `wflow_N` the outlet
            # series carries -- that counter exists nowhere else.
            station_name=None if pd.isna(row.station_name) else str(row.station_name),
        )
        if station.wflow_id in seen:
            log(
                f"Series {index} and {seen[station.wflow_id]} resolve to the same "
                "location; rendering one figure set"
            )
            continue
        seen[station.wflow_id] = index
        stations[index] = station
    return stations


def subcatchment_cell_counts(staticmaps: xr.Dataset) -> dict[int, int]:
    """Count cells in the disjoint native output zones for basin reconstruction."""
    values = np.asarray(staticmaps["subcatchment"].values)
    valid = values[np.isfinite(values) & (values > 0)]
    if not valid.size or np.any(valid != np.floor(valid)):
        raise ValueError("subcatchment must contain positive integer output-zone IDs")
    ids, counts = np.unique(valid.astype(int), return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def analyse_wflow_historical(
    project_dir: Path,
    model_dir: Union[Path, str] = None,
    observations_fn: Union[Path, str] = None,
    gauges_locs: Union[Path, str] = None,
    location_registry: Union[Path, str] = None,
    response_record: Union[Path, str] = None,
    subbasin_plots: bool = False,
    annotation: str = "none",
    toml_name: str = "wflow_sbm.toml",
    evaluation_dir: Union[Path, str, None] = None,
    station_plots: bool = False,
):
    """
    Analyse and plot wflow model performance for historical run.

    ``toml_name`` and ``evaluation_dir`` select one named forcing recipe's run
    configuration and evaluation folder; defaults are the single-recipe layout.

    The model root is supplied by the RULE as ``model_dir``, not rebuilt here
    from ``project_dir``: where the model lives is the Snakefile's fact, and
    spelling it in both places is how the two drift (R9 P2 commit 1).
    Results are read exclusively from verified per-variable tables. The model supplies
    spatial cell-count weights, not a second native time-series reader.

    Per-subcatchment climate figures were removed in the climate-plot consolidation: the
    climate a reader wants is the map and series family under
    ``models/hydrology/wflow/forcing/plots/``, and drawing it twice from two
    code paths was the duplication that decision retires.


    Outputs:

    - optional station hydrographs with daily, annual and seasonal discharge, daily agreement
      where observations exist, and log flow duration.
    - peak/low signature plots when the record and observations support them, with
      return-period frequency panels and wet/dry years selected by annual mean flow.
    - separate daily/monthly performance tables with metrics in rows.
    - basin-average annual series and seasonal cycles for configured zonal outputs;
      optionally the same two figures individually for each subbasin.
    - compute performance indicators (daily and monthly KGE, NSE, NSElog, RMSE, MSE, Pbias,
      VE) if observations are available and if wflow run is longer than a year. Indicators
      are saved to a csv file.

    Parameters
    ----------
    project_dir : Path
        path to CST project directory
    model_dir : Union[Path, str], optional
        The wflow model root, passed by the rule. Defaults to the v10 location
        under ``project_dir`` when omitted, which is what a standalone call
        gets.
    observations_fn : Union[Path, str], optional
        Path to observations timeseries file, by default None
        Required columns: time, wflow_id IDs of the locations as in ``gauges_locs``.
        Separator is **;** and decimal is . -- deliberately different from
        ``gauges_locs`` below, which is comma-separated. Both are read with an
        explicit ``sep=``; see ``config/templates/README.md``.
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    location_registry : Union[Path, str], optional
        Resolved P1 registry used to validate observation station IDs.
    station_plots : bool, optional
        Render per-location discharge figures, by default False. Performance
        indicators are computed regardless of this presentation setting.
    """
    ### 1. Prepare output and plotting options ###

    # Create output folders. R07 B10: the project-level
    # plots/wflow_model_performance/ tree is retired; rule 1.10's artifacts
    # live inside the engine subtree, split by KIND (P1) — figures under
    # evaluation/plots/, the indicators table one level up in evaluation/,
    # because plots/ holds figures only.
    model_dir = model_dir or f"{project_dir}/models/hydrology/wflow"
    Folder_eval = str(evaluation_dir or f"{model_dir}/evaluation")
    Folder_plots = f"{Folder_eval}/plots"
    # The per-station bin. Its members are named for wflow_ids, which are a
    # product of the model build and so unknowable when the Snakefile is
    # parsed -- rule 1.15 therefore declares this DIRECTORY rather than the
    # files, which is what lets `--delete-all-output` reach them (R7-5 /
    # t2608071206). Same device, and the same reason, as WF0's `subbasins/`
    # bin for its per-subbasin figures.
    Folder_stations = f"{Folder_plots}/{STATION_PLOT_DIRNAME}"

    # makedirs, not mkdir: evaluation/plots/ is two levels deep, and only the
    # DECLARED outputs get their parents pre-created by Snakemake. The stations
    # bin IS declared, but as a directory() -- Snakemake creates a declared
    # directory's PARENT, not the directory itself, so it is created here too.
    os.makedirs(Folder_plots, exist_ok=True)
    if station_plots:
        os.makedirs(Folder_stations, exist_ok=True)

    # Style is not decided here any more. Font sizes, line weights, colours and
    # the printed page all live in `shared/plot_evaluation.py` beside the code
    # that draws with them; this rule chooses WHAT to plot, not how.

    ### 2. Read the observations ###
    # check if user provided observations
    has_observations = False

    if observations_fn is not None and os.path.exists(observations_fn):
        has_observations = True

        if location_registry is None or not os.path.isfile(location_registry):
            raise ValueError(
                "configured observations require the resolved location_registry"
            )
        validate_observation_station_ids(observations_fn, location_registry)
        da_ts_obs = open_timeseries_from_table(
            observations_fn, name="Q", index_dim="wflow_id", sep=";"
        )
        qobs = da_ts_obs.rename({"wflow_id": "index"}).load()

    ### 3. Read the wflow model and results ###
    # Instantiate wflow model
    Folder_run = str(model_dir)
    mod = WflowSbmModel(root=Folder_run, mode="r", config_filename=toml_name)
    if response_record is None or location_registry is None:
        raise ValueError(
            "Historical evaluation requires response tables and a location registry"
        )
    qsim, ds_basin = evaluation_arrays(response_record, location_registry)

    # Section 4 (per-subcatchment climate plots) removed 2026-08-09 — see
    # dev/reference/wf1-model-construction.md (outputs and consumers). The climate
    # a reader wants is now the map/series family under forcing/plots/.

    # Apply one common warm-up boundary before any hydrological summary.
    warmup_excluded = len(qsim.time) > 365
    if warmup_excluded:
        _log(
            "Excluding the first simulation year as warm-up from hydrological summaries and evaluation"
        )
        start = pd.Timestamp(qsim.time.values[0]) + pd.DateOffset(years=1)
        qsim = qsim.sel(time=slice(start, None))
        if ds_basin.data_vars:
            ds_basin = ds_basin.sel(time=slice(start, None))
    else:
        _log(
            "Short simulation: retaining the warm-up period in plots and hydrological summaries"
        )

    ### 5. Plot other basin average outputs ###
    if ds_basin.data_vars:
        _log("Rendering basin-average Wflow simulation outputs")
        plot_basavg(
            ds_basin,
            Folder_plots,
            cell_counts=subcatchment_cell_counts(mod.staticmaps.data),
            subbasin_plots=subbasin_plots,
            annotation=annotation,
            warmup_excluded=warmup_excluded,
        )
        plt.close()
    else:
        _log(
            "Basin-average figures omitted: no basin-average output variables are configured"
        )

    ### 6. Plot hydrographs and compute performance indicators ###
    # Initialise the output performance table
    df_perf_all = pd.DataFrame()
    # Flag for plot signatures
    # (True if wflow run is longer than a year and observations are available)
    do_signatures = warmup_excluded and has_observations
    # Sel qsim and qobs so that they have the same time period
    if has_observations:
        start = max(qsim.time.values[0], qobs.time.values[0])
        end = min(qsim.time.values[-1], qobs.time.values[-1])
        qsim = qsim.sel(time=slice(start, end))
        qobs = qobs.sel(time=slice(start, end))

    # Every plotted series, keyed by the wflow_id the rest of the run uses. A
    # series that resolves onto a wflow_id already claimed is dropped here, so
    # an outlet and a gauge on the same cell produce one set of figures.
    stations = resolve_stations(qsim, location_registry)

    # ONE announcement for the whole set, not one per station. Every `log_row`
    # flushes the open figure bundle (`snake_utils.save_figure`), so a row
    # before each station cut sixteen figures into four identical
    # `4 figures -> <model>/evaluation/plots/stations` rows. The ids the
    # per-station rows carried are kept here, on the row that opens the set.
    plotted_ids = [
        stations[int(sid)].wflow_id
        for sid in qsim.index.values
        if stations.get(int(sid)) is not None
    ]
    if station_plots:
        _log(
            f"Rendering discharge figures for {plural(len(plotted_ids), 'model location')} (wflow_id): "
            + ", ".join(str(wflow_id) for wflow_id in plotted_ids)
        )

    for station_id in qsim.index.values:
        station = stations.get(int(station_id))
        if station is None:
            continue  # a duplicate of a cell already plotted; resolve_stations said so
        qsim_i = qsim.sel(index=station_id)
        # Observations are keyed by wflow_id (the file's `wflow_id` column), so
        # they are matched on the RESOLVED id, not on the series index. Those
        # differ for the outlet — index 101, wflow_id 1010 — which is why an
        # observation at the basin outlet used to match nothing at all and the
        # station was silently evaluated as if it had none. Found 2026-08-10 by
        # running this rule against a synthetic observation file; the fixture
        # ships none, so nothing here could have surfaced it.
        qobs_i = None
        if has_observations and station.wflow_id in qobs.index.values:
            qobs_i = qobs.sel(index=station.wflow_id)

        # The indicators table is computed BEFORE the figures because the
        # performance sheet renders it rather than recomputing it — the figure
        # and performance_indicators.csv cannot disagree that way.
        indicators = None
        if do_signatures and qobs_i is not None:
            df_perf = compute_indicators(
                qsim=qsim_i,
                qobs=qobs_i,
                # The CSV is keyed by wflow_id too, so a reader can join the
                # table to a figure, to a map label, to an output.csv column.
                station_name=str(station.wflow_id),
            )
            indicators = df_perf[str(station.wflow_id)].unstack("time_type")
            df_perf_all = df_perf if df_perf_all.empty else df_perf_all.join(df_perf)

        if station_plots:
            plot_station_evaluation(
                simulated=qsim_i,
                observed=qobs_i,
                station=station,
                plot_dir=Folder_stations,
                indicators=indicators,
                signatures=do_signatures and qobs_i is not None,
                log=_log,
                annotation=annotation,
            )
            plt.close("all")

    # Save performance indicators beside the other evaluation tables.
    Folder_tables = os.path.join(Folder_eval, "tables")
    os.makedirs(Folder_tables, exist_ok=True)
    df_perf_all.to_csv(os.path.join(Folder_tables, "performance_indicators.csv"))
    _log(
        (
            "Performance indicator table written to"
            if len(df_perf_all.columns)
            else "No observation-based performance indicators computed"
        )
        + " → performance_indicators.csv"
    )

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            analyse_wflow_historical(
                project_dir=sm.params.project_dir,
                model_dir=sm.params.model_dir,
                observations_fn=getattr(sm.input, "observations_timeseries", None),
                gauges_locs=getattr(sm.input, "output_locations", None),
                location_registry=sm.input.location_registry,
                response_record=sm.input.response_record,
                subbasin_plots=sm.params.subbasin_plots,
                station_plots=getattr(sm.params, "station_plots", False),
                annotation=getattr(sm.params, "annotation", "none"),
                toml_name=getattr(sm.params, "toml_name", "wflow_sbm.toml"),
                evaluation_dir=getattr(sm.params, "evaluation_dir", None),
                # declared only on the chirps/chirps_global branch (ext2-1)
            )
    else:
        analyse_wflow_historical(
            project_dir=join(os.getcwd(), "test_case", "my_project"),
            observations_fn=None,
            gauges_locs=None,
            location_registry=None,
        )
