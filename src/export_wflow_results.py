# -*- coding: utf-8 -*-
"""
Export wflow results for easy plotting of the climate response plots
"""
import os
import pandas as pd
import numpy as np
from xclim.indices.stats import frequency_analysis
import xarray as xr

import hydromt
from hydromt_wflow import WflowModel
from hydromt_wflow.utils import read_csv_results

# Snakemake options
csv_fns = snakemake.input.rlz_csv_fns
exp_dir = snakemake.params.exp_dir

model_dir = snakemake.params.model_dir
Tpeak = snakemake.params.Tpeak
Tlow = snakemake.params.Tlow
aggr_rlz = snakemake.params.aggr_rlz
rlz_num = snakemake.params.rlz_num 
st_num = snakemake.params.st_num

Qstats_fn = snakemake.output.Qstats
bas_fn = snakemake.output.basin

# Read the wflow models and results
print("Reading wflow model")
mod = WflowModel(root=model_dir, mode="r")

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
        dtype="float32",
    )
else: 
    col_names = ["statistic", "realization", "tavg", "prcp"]
    col_names.extend(Q_vars)
    df_out_mean = pd.DataFrame(
        data=np.zeros((len(csv_fns), len(col_names))),
        columns=col_names,
        dtype="float32",
    )
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



## High flows
def returninterval(df, T):
    ds = xr.Dataset.from_dataframe(df)
    Q_interval = frequency_analysis(ds, t=T, dist="genextreme", mode="max", freq="YS")
    df_interval = xr.Dataset.to_dataframe(Q_interval)
    return df_interval.transpose().iloc[:,0]

def returnintervalmulti(df):
    ds = xr.Dataset.from_dataframe(df)
    all_T = [2, 5, 10, 20, 50, 100,200]
    Q_rps = frequency_analysis(ds, t=all_T, dist="genextreme", mode="max", freq="YS")
    return Q_rps

def Q7d_max(df): 
    return df.rolling(7).mean().resample('a').max().mean()

def highpulse(df): 
    return df[df>df.quantile(.75)].resample('a').count().mean()

def wetmonth_mean(df): 
    monthlysum = df.groupby(df.index.month).sum()
    wetmonth = monthlysum.idxmax()[0] 
    df_wetmonth = df[df.index.month == wetmonth]
    return df_wetmonth.mean()

## Low flows
def returninterval_Q7d(df, T):
    df7D = df.rolling(7).mean()
    ds = xr.Dataset.from_dataframe(df7D)
    Q_interval = frequency_analysis(ds, t=T, dist="genextreme", mode="min", freq="YS")
    df_interval = xr.Dataset.to_dataframe(Q_interval)
    return df_interval.transpose().iloc[:,0]

def Q7d_min(df): 
    return df.rolling(7).mean().resample('a').min().mean()

def lowpulse(df): 
    return df[df<df.quantile(.25)].resample('a').count().mean()

def drymonth_mean(df): 
    monthlysum = df.groupby(df.index.month).sum()
    drymonth = monthlysum.idxmin()[0]
    df_drymonth = df[df.index.month == drymonth]
    return df_drymonth.mean()

def BFI(df): 
    Q7d = df.rolling(7).mean().resample('a').min()
    annmean = df.resample('a').mean()
    return (Q7d / annmean).mean()

print("Computing discharge stats for each realization/stress test")
Q_rps = []
for i in range(np.size(df_out_mean,0)):
    # Read csv file
    if not aggr_rlz:
        st_nb = os.path.basename(csv_fns[i]).split(".")[0].split("_")[-1]
        sim_all = pd.read_csv(csv_fns[i], index_col=0, parse_dates=True)
        sim = sim_all[Q_vars]
    else:
        # read and concat several files
        st_nb = i+1
        csv_fns_i = [x for x in csv_fns if x.endswith(str(i+1)+'.csv')]
        csv_rlz = []
        for j in range(len(csv_fns_i)): 
            sim_j = pd.read_csv(csv_fns_i[j], index_col=0, parse_dates=True)
            csv_rlz.append(sim_j)
        sim_all = pd.concat(csv_rlz)
        sim_all.index = pd.date_range(start=sim_all.index[0], periods=len(sim_all), name='time')
        sim = sim_all[Q_vars]
    # Get statistics
    # Average Yearly statistics
    df_mean = sim.resample('a').mean().mean()
    df_max = sim.resample('a').max().mean()
    df_min = sim.resample('a').min().mean()
    df_q95 = sim.resample('a').quantile(0.95).mean()
    # High flows
    df_RT = returninterval(sim, Tpeak)
    df_Q7dmax = Q7d_max(sim)
    # df_highpulse = highpulse(sim)
    df_wetmonth = wetmonth_mean(sim)
    # Low flows
    df_RT7d = returninterval_Q7d(sim, Tlow)
    df_Q7dmin = Q7d_min(sim)
    # df_lowpulse = lowpulse(sim)
    df_drymonth = drymonth_mean(sim)
    df_BFI = BFI(sim)

    # Get stress test stats
    rlz_nb = int(os.path.basename(csv_fns[i]).split(".")[0].split("_")[2])
    if st_nb == "0":
        tavg = 0
        prcp = 0
    else:
        df_st = pd.read_csv(f"{exp_dir}/stress_test/cst_{st_nb}.csv")
        tavg = df_st["temp_mean"].iloc[0]
        prcp = df_st["precip_mean"].iloc[0] * 100 - 100  # change in %
    if not aggr_rlz:
        cst_stat = (rlz_nb, tavg, prcp)
    else:
        cst_stat = (tavg, prcp)

    # Update discharge statistics tableslen
    df_out_mean.iloc[i, :] = np.concatenate([['mean'], cst_stat, df_mean.values.round(2)]) 
    df_out_max.iloc[i, :] = np.concatenate([['max'], cst_stat, df_max.values.round(2)]) 
    df_out_min.iloc[i, :] = np.concatenate([['min'], cst_stat, df_min.values.round(4)])
    df_out_q95.iloc[i, :] = np.concatenate([['q95'], cst_stat, df_q95.values.round(2)])
    df_out_RT.iloc[i, :] = np.concatenate([['returninterval'], cst_stat, df_RT.values.round(2)])
    df_out_Q7dmax.iloc[i, :] = np.concatenate([['Q7day_max'], cst_stat, df_Q7dmax.values.round(2)])
    # df_out_highpulse.iloc[i, :] = np.concatenate([['highpulse'], cst_stat, df_highpulse.values.round(2)])
    df_out_wetmonth.iloc[i, :] = np.concatenate([['wetmonth_mean'], cst_stat, df_wetmonth.values.round(2)])
    df_out_RT7d.iloc[i, :] = np.concatenate([['returninternval_min_7day'], cst_stat, df_RT7d.values.round(4)])
    df_out_Q7dmin.iloc[i, :] = np.concatenate([['Q7day_min'], cst_stat, df_Q7dmin.values.round(4)])
    # df_out_lowpulse.iloc[i, :] = np.concatenate([['lowpulse'], cst_stat, df_lowpulse.values.round(2)])
    df_out_drymonth.iloc[i, :] = np.concatenate([['drymonth_mean'], cst_stat, df_drymonth.values.round(4)])
    df_out_BFI.iloc[i, :] = np.concatenate([['BaseFlowIndex'], cst_stat, df_BFI.values.round(4)])

    # Update return interval dataset
    Q_rp = returnintervalmulti(sim)
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
            stats_basavg = np.append(stats_basavg, (sim[v].resample('a').max().mean()))
        else: #actual evapotranspiration_basavg or groundwater recharge_basavg or overland_flow_basavg
            # Total evaporation or recharge or overland flow volume (mm/yr)
            stats_basavg = np.append(stats_basavg, (sim[v].resample('a').sum().mean()))
    df_out_basavg.iloc[i, :] = stats_basavg.round(1)

print("Writting tables for 2D stress tests plots")
df_out_basavg.to_csv(bas_fn, index=False)

df_out_Qstats = pd.concat([df_out_mean, df_out_max, df_out_min, df_out_q95, 
                           df_out_RT, df_out_Q7dmax, df_out_wetmonth, df_out_RT7d, 
                           df_out_Q7dmin, df_out_drymonth, df_out_BFI])
df_out_Qstats.to_csv(Qstats_fn, index=False)

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
    df_rp.to_csv(os.path.join(f"{exp_dir}/model_results", f"RT_{v}.csv"), index=False)
