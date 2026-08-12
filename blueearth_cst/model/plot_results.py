# -*- coding: utf-8 -*-
"""
Plot wflow results and compare to observations if any
"""

import os
from os.path import join
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from hydromt.readers import open_timeseries_from_table
from hydromt_wflow import WflowSbmModel

from blueearth_cst.model.observation_validation import (
    validate_observation_station_ids,
)
from blueearth_cst.shared.func_plot_signature import (
    compute_metrics,
    plot_basavg,
)
from blueearth_cst.shared.gauges import gauges_layer_name, gauges_variable_name
from blueearth_cst.shared.plot_evaluation import Station, plot_station_evaluation
from blueearth_cst.shared.snake_utils import log_row
from blueearth_cst.shared.wflow_outputs import (
    SUBCATCHMENT_SUFFIX,
    code_for,
    is_basin_average,
)


def _log(message):
    """Emit one standard-format log row (module tag ``plot``) for this rule."""
    log_row(message, module="plot")


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
      cell here. ``merge_outlet_and_gauge_series`` can only see the collision
      when the raw indices coincide, so a project with output_locations at its
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
                f"series {index} is in neither the location registry's wflow_id "
                f"nor its subbasin_id column; naming its figures after {index}."
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
                f"series {index} and {seen[station.wflow_id]} are the same model "
                f"cell (wflow_id {station.wflow_id}); plotting it once."
            )
            continue
        seen[station.wflow_id] = index
        stations[index] = station
    return stations


def merge_outlet_and_gauge_series(qsim, qsim_gauges, log=_log):
    """Combine the outlet and user-gauge discharge series into one station set.

    Outlet ids and user gauge ids share ONE namespace — both are values burned
    into wflow's output maps (``blueearth_cst/shared/gauges.py``, MIN_GAUGE_ID)
    — and the two sets overlap whenever a gauge sits on the basin outlet. That
    is not an exotic case: delineating the model as a subbasin at the outlet
    gauge (`region: {'subbasin': [x, y]}`) is the normal way to build one, and
    it makes the outlet's subcatchment id equal to that gauge's ``wflow_id``.

    The two series then carry the same ``index`` with DIFFERENT ``station_name``
    values — the synthetic ``wflow_1`` on the outlet side, the user's own name
    on the gauge side — and ``xr.merge`` refuses to reconcile a coordinate that
    disagrees ("conflicting values for variable 'station_name'", observed
    2026-08-02 on a real basin whose outlet gauge is id 101).

    The OUTLET series wins on a collision. Until 2026-08-10 that mattered for
    a filename — rule 1.15 declared ``hydro_wflow_1.png`` and every figure was
    written as ``<kind>_{station_name}.png``, so a user name winning on the
    first outlet traded a MergeError for a MissingOutputException. The figures
    are keyed by ``wflow_id`` now and nothing is declared, so the choice is no
    longer load-bearing; it stays because a merge needs a deterministic winner
    and nothing is lost by it — the colliding entries are the same model cell,
    so the two series hold the same discharge.

    This only sees a collision when the two INDEX values coincide. An outlet
    and a gauge on one cell can also carry different indices (101 and 1010 are
    the same point on the test fixture), and that case is caught downstream by
    ``resolve_stations``, which dedupes on the resolved ``wflow_id``.

    Parameters
    ----------
    qsim, qsim_gauges : xr.DataArray
        Discharge on dim ``index`` with a ``station_name`` coordinate.
    log : callable
        One-line reporter; the collision is announced rather than silent,
        because the station a user named disappears from the figure filenames.
    """
    outlet_ids = set(qsim["index"].values.tolist())
    keep = [i for i in qsim_gauges["index"].values if i not in outlet_ids]
    dropped = [i for i in qsim_gauges["index"].values if i in outlet_ids]
    if dropped:
        # Plain Python in the message: numpy scalars render as np.int32(101),
        # which is noise in a user-facing log line.
        gauge_names = [
            str(n) for n in qsim_gauges["station_name"].sel(index=dropped).values
        ]
        outlet_names = [str(n) for n in qsim["station_name"].sel(index=dropped).values]
        log(
            f"Gauge(s) {[int(i) for i in dropped]} ({', '.join(gauge_names)}) "
            f"sit on a model outlet and are already in Q_outlets; plotted "
            f"under the outlet label(s) {outlet_names}."
        )
    if not keep:
        return qsim
    # Both kwargs are xarray's CURRENT defaults, spelled out because both are
    # scheduled to change and this merge depends on both: the station sets are
    # disjoint by construction, so it needs join="outer" (the announced
    # "exact" would raise on unequal indexes), and it needs compat that FILLS
    # across the alignment NaNs (the announced "override" would take the
    # outlet side's station_name and blank every gauge label).
    return xr.merge(
        [qsim, qsim_gauges.sel(index=keep)], join="outer", compat="no_conflicts"
    )["Q"]


def analyse_wflow_historical(
    project_dir: Path,
    model_dir: Union[Path, str] = None,
    observations_fn: Union[Path, str] = None,
    gauges_locs: Union[Path, str] = None,
    location_registry: Union[Path, str] = None,
):
    """
    Analyse and plot wflow model performance for historical run.

    The model root is supplied by the RULE as ``model_dir``, not rebuilt here
    from ``project_dir``: where the model lives is the Snakefile's fact, and
    spelling it in both places is how the two drift (R9 P2 commit 1).
    Model results should include the discharge keys Q_outlets and, if gauges are
    provided, Q_gauges_{basename(gauges_locs)}.

    Per-subcatchment climate figures were removed 2026-08-09 (ADR 0006): the
    climate a reader wants is the map and series family under
    ``models/hydrology/wflow/forcing/plots/``, and drawing it twice from two
    code paths was the duplication that decision retires.


    Outputs:

    - plot of hydrographs at the outlet(s) and gauges_locs if provided. If wflow run is
      three years or less, only the daily hydrograph will be plotted. If wflow run is
      longer than three years, plots will also include the yearly hydrograph, the
      monthly average and hyddrographs for the wettest and driest years. If observations
      are available, they are added as well.
    - plot of signature plots if wflow run is longer than a year and if observations
      are available.
    - plot of basin average outputs (e.g. soil moisture, snow, etc.). The variables to
      include should have the postfix _basavg in the wflow output file.
    - compute performance metrics (daily and monthly KGE, NSE, NSElog, RMSE, MSE, Pbias,
      VE) if observations are available and if wflow run is longer than a year. Metrics
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
    """
    ### 1. Prepare output and plotting options ###

    # Create output folders. R07 B10: the project-level
    # plots/wflow_model_performance/ tree is retired; rule 1.11's artifacts
    # live inside the engine subtree, split by KIND (P1) — figures under
    # evaluation/plots/, the metrics table one level up in evaluation/,
    # because plots/ holds figures only.
    model_dir = model_dir or f"{project_dir}/models/hydrology/wflow"
    Folder_eval = f"{model_dir}/evaluation"
    Folder_plots = f"{Folder_eval}/plots"

    # makedirs, not mkdir: evaluation/plots/ is two levels deep, and only the
    # DECLARED outputs get their parents pre-created by Snakemake.
    os.makedirs(Folder_plots, exist_ok=True)

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
    mod = WflowSbmModel(root=Folder_run, mode="r")
    mod.output_csv.read()
    results = mod.output_csv.data
    geoms = mod.geoms.data

    # Discharge at the outlet(s) (was Q_gauges in 0.x; now Q_outlets).
    qsim = results["Q_outlets"].rename("Q")
    # In hydromt_wflow 1.x outlet ids come from the subcatchment map
    # (e.g. 130000086) instead of the 1..N counter that 0.x used. Keep the
    # 1..N station_name so rule_all and downstream plots stay stable.
    qsim = qsim.assign_coords(
        station_name=(
            "index",
            [f"wflow_{i + 1}" for i in range(qsim["index"].size)],
        )
    )
    # Discharge at the gauges_locs if present. Both names are resolved from the
    # MODEL rather than from the filename: hydromt_wflow renames
    # output_locations -> output-locations, so deriving them here found neither
    # the variable nor the layer, and the membership tests turned that into
    # silence -- no gauge hydrographs, no signatures, and an EMPTY
    # performance_metrics.csv on a config that supplied observations
    # (2026-08-01). See blueearth_cst/shared/gauges.py.
    if gauges_locs is not None and os.path.exists(gauges_locs):
        gauges_var = gauges_variable_name(results, gauges_locs, "Q")
        gauges_layer = gauges_layer_name(geoms, gauges_locs)
        if gauges_var is not None and gauges_layer is not None:
            qsim_gauges = results[gauges_var].rename("Q")
            gdf_gauges = (
                geoms[gauges_layer]
                .rename(columns={"wflow_id": "index"})
                .set_index("index")
            )
            qsim_gauges = qsim_gauges.assign_coords(
                station_name=(
                    "index",
                    list(gdf_gauges["station_name"][qsim_gauges.index.values].values),
                )
            )
            qsim = merge_outlet_and_gauge_series(qsim, qsim_gauges, log=_log)

    # Identified by hydromt's `<header>_<mapname>` suffix, not by the header:
    # the header is now a short code (`gwr`), so a `"_basavg" in dvar` test
    # would silently match nothing and skip every basin-average figure.
    basavg_vars = [dvar for dvar in results if is_basin_average(dvar)]
    if basavg_vars:
        # NOT .squeeze(drop=True). Squeezing collapsed a single subcatchment's
        # index away, which is the only reason plot_basavg's 1-D assumption ever
        # held; keeping the dim makes the shape uniform at 1 subcatchment and at
        # four, and plot_basavg now facets over it.
        ds_basin = xr.merge([results[dvar] for dvar in basavg_vars])
        precip = f"{code_for('precipitation')}{SUBCATCHMENT_SUFFIX}"
        if precip in ds_basin:
            ds_basin = ds_basin.drop_vars(precip)
    else:
        ds_basin = xr.Dataset()

    # Section 4 (per-subcatchment climate plots) removed 2026-08-09 — see
    # dev/decisions/0006-retire-subcatchment-climate-plots.md. The climate
    # a reader wants is now the map/series family under forcing/plots/.

    ### 5. Plot other basin average outputs ###
    if ds_basin.data_vars:
        _log("Plot basin average wflow outputs")
        plot_basavg(ds_basin, Folder_plots)
        plt.close()
    else:
        _log("No basin-average outputs configured; skipping plot_basavg.")

    ### 6. Plot hydrographs and compute performance metrics ###
    # Initialise the output performance table
    df_perf_all = pd.DataFrame()
    # Flag for plot signatures
    # (True if wflow run is longer than a year and observations are available)
    do_signatures = False

    # If possible, skip the first year of the wflow run (warm-up period)
    if len(qsim.time) > 365:
        _log("Skipping the first year of the wflow run (warm-up period)")
        qsim = qsim.sel(
            time=slice(
                f"{qsim['time.year'][0].values + 1}-{qsim['time.month'][0].values}-{qsim['time.day'][0].values}",
                None,
            )
        )
        if has_observations:
            do_signatures = True
    else:
        _log("Simulation is less than a year so model warm-up period will be plotted.")
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

        # The metrics table is computed BEFORE the figures because the
        # performance sheet renders it rather than recomputing it — the figure
        # and performance_metrics.csv cannot disagree that way.
        metrics = None
        if do_signatures and qobs_i is not None:
            df_perf = compute_metrics(
                qsim=qsim_i,
                qobs=qobs_i,
                # The CSV is keyed by wflow_id too, so a reader can join the
                # table to a figure, to a map label, to an output.csv column.
                station_name=str(station.wflow_id),
            )
            metrics = df_perf[str(station.wflow_id)].unstack("time_type")
            df_perf_all = df_perf if df_perf_all.empty else df_perf_all.join(df_perf)

        _log(f"Plot evaluation figures for wflow_id {station.wflow_id}")
        plot_station_evaluation(
            simulated=qsim_i,
            observed=qobs_i,
            station=station,
            plot_dir=Folder_plots,
            metrics=metrics,
            signatures=do_signatures and qobs_i is not None,
            log=_log,
        )
        plt.close("all")

    # Save performance metrics to csv (evaluation/, not evaluation/plots/ — P1)
    df_perf_all.to_csv(os.path.join(Folder_eval, "performance_metrics.csv"))

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
                # declared only on the chirps/chirps_global branch (ext2-1)
            )
    else:
        analyse_wflow_historical(
            project_dir=join(os.getcwd(), "test_case", "my_project"),
            observations_fn=None,
            gauges_locs=None,
            location_registry=None,
        )
