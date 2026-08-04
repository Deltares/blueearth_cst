# -*- coding: utf-8 -*-
"""
Export wflow results for easy plotting of the climate response plots
"""
import re
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from typing import List, Union

import blueearth_cst.shared.metrics_definition as md
from blueearth_cst.shared.snake_utils import log_row


#: ``rlz_<n>_cst_<m>`` in a wflow run CSV's stem. Anchored at the start so a
#: directory component can never satisfy it.
_RLZ_IN_STEM = re.compile(r"^rlz_(\d+)_cst_\d+$")


def realization_from_run_csv(csv_fn: Union[str, Path]) -> int:
    """Realization index of a wflow run CSV, read from its FILENAME.

    The index has moved twice. R07 B5 took it out of the filename and into a
    run directory (``hydrology_runs/rlz_<n>/output/cst_<m>.csv``), so this read
    the grandparent directory. R9 P2 dissolves that level and puts the index
    back in the stem (``hydrology/wflow/output/rlz_<n>_cst_<m>.csv``), so the
    directory read stopped working -- caught by the first full WF3 run on the
    migrated tree, at rule 3.11, after all twelve members had run.

    Matched against a full-stem pattern rather than split on ``_``: the stem
    now contains two indices, and ``split("_")[-1]`` would silently return the
    CST member number as the realization. A wrong-but-plausible integer is far
    worse here than an exception, because it would mislabel every result row.
    """
    stem = Path(csv_fn).stem
    match = _RLZ_IN_STEM.match(stem)
    if match is None:
        raise ValueError(
            f"cannot derive the realization index from {csv_fn!r}: expected a "
            f"'rlz_<n>_cst_<m>' filename, got {stem!r}"
        )
    return int(match.group(1))


def analyze_wflow_results(
    csv_fns: List[Union[str, Path]],
    st_csv_fns: List[Union[str, Path]],
    indicators_dir: Union[str, Path],
    st_num: int,
    qstats_fn: Union[str, Path] = None,
    bas_fn: Union[str, Path] = None,
    Tpeak: int = 10,
    Tlow: int = 2,
    aggr_rlz: bool = True,
):
    """
    Analyze wflow results and computes statistics for each realization/stress test.

    Parameters
    ----------
    csv_fns : List[Union[str, Path]]
        List of csv files containing the wflow results for each realization/stress test
    st_csv_fns : List[Union[str, Path]]
        Paths to the per-stress-test ``cst_<m>.csv`` perturbation files (the temp
        and precip changes each stress test imposes). Passed in from the rule's
        declared inputs rather than reconstructed from a directory convention
        (R07 B6), so the read is on the DAG and visible to ``--dry-run``.
    indicators_dir : Union[str, Path]
        Directory the response-surface tables are written to (``indicators/``).
    st_num : int
        Number of stress tests (increments) per realization
    qstats_fn : Union[str, Path], optional
        Path to the output csv file with the discharge statistics for each
        realization/stress test.
        If None will be saved in `indicators_dir/q_indicators.csv`.
    bas_fn : Union[str, Path], optional
        Path to the output csv file with the basin average statistics for each
        realization/stress test.
        If None will be saved in `indicators_dir/basin_indicators.csv`.
    Tpeak : int, optional
        Return period for high flows (in years), by default 10
    Tlow : int, optional
        Return period for low flows (in years), by default 2
    aggr_rlz : bool, optional
        If True, aggregate all realizations for each stress test, by default True.
    """
    # Output file paths
    if qstats_fn is None:
        qstats_fn = f"{indicators_dir}/q_indicators.csv"
    if bas_fn is None:
        bas_fn = f"{indicators_dir}/basin_indicators.csv"

    # cst_<m>.csv lookup, keyed by the stress-test index in the file name.
    # cst_0 is the reserved unperturbed baseline and has no file (naming.md §4),
    # so the declared set is exactly 1..st_num. Check it here rather than letting
    # a Snakefile/script disagreement surface as a KeyError deep in the loop.
    st_csv_by_num = {Path(p).stem.split("_")[-1]: p for p in st_csv_fns}
    expected = {str(n) for n in range(1, st_num + 1)}
    if set(st_csv_by_num) != expected:
        raise ValueError(
            f"stress-test parameter files do not cover 1..{st_num}: got "
            f"{sorted(st_csv_by_num)}, expected {sorted(expected)}"
        )

    # Get output discharge columns
    sim = pd.read_csv(csv_fns[0], index_col=0, parse_dates=True)
    Q_vars = [x for x in sim.columns if x.startswith("Q_")]
    basavg_vars = [x for x in sim.columns if "basavg" in x]

    # Initialise emtpy output dataframes
    if aggr_rlz:
        col_names = ["statistic", "tavg", "prcp"]
        col_names.extend(Q_vars)
        df_out_mean = pd.DataFrame(
            data=np.zeros((st_num, len(col_names))),
            columns=col_names,
            dtype="str",
        )
        # Update the dtype of the statistics columns into string
        df_out_mean["statistic"] = df_out_mean["statistic"].astype(str)
    else:
        col_names = ["statistic", "realization", "tavg", "prcp"]
        col_names.extend(Q_vars)
        df_out_mean = pd.DataFrame(
            data=np.zeros((len(csv_fns), len(col_names))),
            columns=col_names,
            dtype="str",
        )
        # Update the dtype of the statistics columns into string
        df_out_mean["statistic"] = df_out_mean["statistic"].astype(str)
    df_out_max = df_out_mean.copy()
    df_out_min = df_out_mean.copy()
    df_out_q95 = df_out_mean.copy()
    df_out_RT = df_out_mean.copy()
    df_out_Q7dmax = df_out_mean.copy()
    # df_out_highpulse = df_out_mean.copy()
    df_out_wetmonth = df_out_mean.copy()
    df_out_RT7d = df_out_mean.copy()
    df_out_Q7dmin = df_out_mean.copy()
    # df_out_lowpulse = df_out_mean.copy()
    df_out_drymonth = df_out_mean.copy()
    df_out_BFI = df_out_mean.copy()

    # Other variables than discharge
    if aggr_rlz:
        col_names = ["tavg", "prcp"]
        col_names.extend(basavg_vars)
        df_out_basavg = pd.DataFrame(
            data=np.zeros((st_num, len(col_names))),
            columns=col_names,
            dtype="float32",
        )
    else:
        col_names = ["realization", "tavg", "prcp"]
        col_names.extend(basavg_vars)
        df_out_basavg = pd.DataFrame(
            data=np.zeros((len(csv_fns), len(col_names))),
            columns=col_names,
            dtype="float32",
        )

    log_row("Computing discharge stats for each realization/stress test", module="export")
    Q_rps = []
    for i in range(np.size(df_out_mean, 0)):
        # Read csv file
        if not aggr_rlz:
            st_nb = os.path.basename(csv_fns[i]).split(".")[0].split("_")[-1]
            sim_all = pd.read_csv(csv_fns[i], index_col=0, parse_dates=True)
            sim = sim_all[Q_vars]
        else:
            # read and concat several files
            st_nb = i + 1
            csv_fns_i = [x for x in csv_fns if x.endswith("cst_" + str(i + 1) + ".csv")]
            csv_rlz = []
            for j in range(len(csv_fns_i)):
                sim_j = pd.read_csv(csv_fns_i[j], index_col=0, parse_dates=True)
                csv_rlz.append(sim_j)
            sim_all = pd.concat(csv_rlz)
            sim_all.index = pd.date_range(
                start=sim_all.index[0], periods=len(sim_all), name="time"
            )
            sim = sim_all[Q_vars]
        # Get statistics
        # Average Yearly statistics
        df_mean = sim.resample("YE").mean().mean()
        df_max = sim.resample("YE").max().mean()
        df_min = sim.resample("YE").min().mean()
        df_q95 = sim.resample("YE").quantile(0.95).mean()
        # High flows
        df_RT = md.returninterval(sim, Tpeak)
        df_Q7dmax = md.Q7d_maxyear(sim)
        # df_highpulse = md.highpulse(sim)
        df_wetmonth = md.wetmonth_mean(sim)
        # Low flows
        df_RT7d = md.returninterval_Q7d(sim, Tlow)
        df_Q7dmin = md.Q7d_min(sim)
        # df_lowpulse = md.lowpulse(sim)
        df_drymonth = md.drymonth_mean(sim)
        df_BFI = md.BFI(sim)

        # Get stress test stats
        rlz_nb = realization_from_run_csv(csv_fns[i])
        if st_nb == "0":
            tavg = 0
            prcp = 0
        else:
            df_st = pd.read_csv(st_csv_by_num[str(st_nb)])
            tavg = df_st["temp_mean"].iloc[0]
            prcp = df_st["precip_mean"].iloc[0] * 100 - 100  # change in %
        if not aggr_rlz:
            cst_stat = (rlz_nb, tavg, prcp)
        else:
            cst_stat = (tavg, prcp)

        # Update discharge statistics tableslen
        df_out_mean.iloc[i, :] = np.concatenate(
            [["mean"], cst_stat, df_mean.values.round(2)]
        )
        df_out_max.iloc[i, :] = np.concatenate(
            [["max"], cst_stat, df_max.values.round(2)]
        )
        df_out_min.iloc[i, :] = np.concatenate(
            [["min"], cst_stat, df_min.values.round(4)]
        )
        df_out_q95.iloc[i, :] = np.concatenate(
            [["q95"], cst_stat, df_q95.values.round(2)]
        )
        df_out_RT.iloc[i, :] = np.concatenate(
            [["returninterval"], cst_stat, df_RT.values.round(2)]
        )
        df_out_Q7dmax.iloc[i, :] = np.concatenate(
            [["Q7day_max"], cst_stat, df_Q7dmax.values.round(2)]
        )
        # df_out_highpulse.iloc[i, :] = np.concatenate([['highpulse'], cst_stat, df_highpulse.values.round(2)])
        df_out_wetmonth.iloc[i, :] = np.concatenate(
            [["wetmonth_mean"], cst_stat, df_wetmonth.values.round(2)]
        )
        df_out_RT7d.iloc[i, :] = np.concatenate(
            [["returninternval_min_7day"], cst_stat, df_RT7d.values.round(4)]
        )
        df_out_Q7dmin.iloc[i, :] = np.concatenate(
            [["Q7day_min"], cst_stat, df_Q7dmin.values.round(4)]
        )
        # df_out_lowpulse.iloc[i, :] = np.concatenate([['lowpulse'], cst_stat, df_lowpulse.values.round(2)])
        df_out_drymonth.iloc[i, :] = np.concatenate(
            [["drymonth_mean"], cst_stat, df_drymonth.values.round(4)]
        )
        df_out_BFI.iloc[i, :] = np.concatenate(
            [["BaseFlowIndex"], cst_stat, df_BFI.values.round(4)]
        )

        # Update return interval dataset
        Q_rp = md.returnintervalmulti(sim)
        # Add realization as new coords
        Q_rp = Q_rp.assign_coords(scenario=i)
        # Add a new dim for realization number
        Q_rp = Q_rp.expand_dims("scenario")
        # Add tavg coords that are function of scenario dim
        if not aggr_rlz:
            Q_rp = Q_rp.assign_coords(realization=("scenario", [rlz_nb]))
        Q_rp = Q_rp.assign_coords(tavg=("scenario", [tavg]))
        Q_rp = Q_rp.assign_coords(prcp=("scenario", [prcp]))
        Q_rps.append(Q_rp)

        # Update basin average statistics table
        if not aggr_rlz:
            stats_basavg = np.array([rlz_nb, tavg, prcp])
        else:
            stats_basavg = np.array([tavg, prcp])
        sim = sim_all[basavg_vars]
        for v in basavg_vars:
            if v == "snow_basavg":
                # Maximum snow water equivalent per year (mm/yr)
                stats_basavg = np.append(
                    stats_basavg, (sim[v].resample("YE").max().mean())
                )
            else:
                # actual evapotranspiration_basavg or groundwater recharge_basavg
                # or overland_flow_basavg
                # Total evaporation or recharge or overland flow volume (mm/yr)
                stats_basavg = np.append(
                    stats_basavg, (sim[v].resample("YE").sum().mean())
                )
        df_out_basavg.iloc[i, :] = np.float32(stats_basavg.round(1))

    log_row("Writting tables for 2D stress tests plots", module="export")
    if not os.path.isdir(os.path.dirname(bas_fn)):
        os.makedirs(bas_fn)

    df_out_basavg.to_csv(bas_fn, index=False)
    # One file for all stats
    df_out_Qstats = pd.concat(
        [
            df_out_mean,
            df_out_max,
            df_out_min,
            df_out_q95,
            df_out_RT,
            df_out_Q7dmax,
            df_out_wetmonth,
            df_out_RT7d,
            df_out_Q7dmin,
            df_out_drymonth,
            df_out_BFI,
        ]
    )
    df_out_Qstats.to_csv(qstats_fn, index=False)

    # Merge Qrps list and save as one csv per loc
    Q_rps = xr.concat(Q_rps, dim="scenario")
    for v in Q_rps.data_vars:
        df_rp = Q_rps[v].to_pandas().round(1)
        # Reorder dims of Q_rp
        if not aggr_rlz:
            df_rp["realization"] = Q_rps["realization"].values
        df_rp["tavg"] = Q_rps["tavg"].values
        df_rp["prcp"] = Q_rps["prcp"].values
        # Change column order of df
        cols = df_rp.columns.tolist()
        if not aggr_rlz:
            cols = cols[-3:] + cols[:-3]
        else:
            cols = cols[-2:] + cols[:-2]
        df_rp = df_rp[cols]
        # Save to csv
        df_rp.to_csv(os.path.join(indicators_dir, f"RT_{v}.csv"), index=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            analyze_wflow_results(
                csv_fns=sm.input.rlz_csv_fns,
                st_csv_fns=sm.input.st_csv_fns,
                indicators_dir=sm.params.indicators_dir,
                st_num=sm.params.st_num,
                qstats_fn=sm.output.q_indicators,
                bas_fn=sm.output.basin_indicators,
                Tpeak=sm.params.Tpeak,
                Tlow=sm.params.Tlow,
                aggr_rlz=sm.params.aggr_rlz,
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
