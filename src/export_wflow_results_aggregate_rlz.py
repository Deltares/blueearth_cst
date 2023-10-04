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
from metrics_def import *
# Snakemake options
csv_fns = snakemake.input.rlz_csv_fns
exp_dir = snakemake.params.exp_dir

model_dir = snakemake.params.model_dir
Tpeak = snakemake.params.Tpeak
Tlow = snakemake.params.Tlow
aggr_rlz = snakemake.params.aggr_rlz
rlz_num = snakemake.params.rlz_num 
st_num = snakemake.params.st_num

mean_fn = rf'{snakemake.output.output_fn}/mean.csv'
max_fn = rf'{snakemake.output.output_fn}/max.csv'
min_fn = rf'{snakemake.output.output_fn}/min.csv'
q95_fn = rf'{snakemake.output.output_fn}/q95.csv'
RT_fn = rf'{snakemake.output.output_fn}/returninterval_discharge.csv'
Q7dmax_fn = rf'{snakemake.output.output_fn}/Q7dmax_discharge.csv'
Q7dmax_total_fn = rf'{snakemake.output.output_fn}/Q7dmax_total_discharge.csv'
highpulse_fn = rf'{snakemake.output.output_fn}/highpulse_discharge.csv'
wetmonth_fn = rf'{snakemake.output.output_fn}/wetmonth_discharge.csv'
wetmonth_total_fn = rf'{snakemake.output.output_fn}/wetmonth_total_discharge.csv'
RT7d_fn = rf'{snakemake.output.output_fn}/returninterval7d_discharge.csv'
Q7dmin_fn = rf'{snakemake.output.output_fn}/Q7dmin_discharge.csv'
Q7dmin_total_fn = rf'{snakemake.output.output_fn}/Q7dmin_total_discharge.csv'
lowpulse_fn = rf'{snakemake.output.output_fn}/lowpulse_discharge.csv'
drymonth_fn = rf'{snakemake.output.output_fn}/drymonth_discharge.csv'
drymonth_total_fn = rf'{snakemake.output.output_fn}/drymonth_total_discharge.csv'
BFI_fn = rf'{snakemake.output.output_fn}/BFI_discharge.csv'
bas_fn = rf'{snakemake.output.output_fn}/basin.csv'

# Read the wflow models and results
print("Reading wflow model")
mod = WflowModel(root=model_dir, mode="r")

# Get output discharge columns
sim = pd.read_csv(csv_fns[0], index_col=0, parse_dates=True)
Q_vars = [x for x in sim.columns if x.startswith("Q_")]
basavg_vars = [x for x in sim.columns if "basavg" in x]

# Initialise emtpy output dataframes
if aggr_rlz == True: 
    col_names = ["tavg", "prcp"]
    col_names.extend(Q_vars)
    df_out_mean = pd.DataFrame(
        data=np.zeros((st_num, len(col_names))),
        columns=col_names,
        dtype="float32",
    )
else: 
    col_names = ["realization", "tavg", "prcp"]
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
df_out_Q7dmax_total = df_out_mean.copy()
df_out_highpulse = df_out_mean.copy()
df_out_wetmonth = df_out_mean.copy()
df_out_wetmonth_total = df_out_mean.copy()
df_out_RT7d = df_out_mean.copy()
df_out_Q7dmin = df_out_mean.copy()
df_out_Q7dmin_total = df_out_mean.copy()
df_out_lowpulse = df_out_mean.copy()
df_out_drymonth = df_out_mean.copy()
df_out_drymonth_total = df_out_mean.copy()
df_out_BFI = df_out_mean.copy()
# Other variables than discharge
if aggr_rlz == True: 
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

print("Computing discharge stats for each realization/stress test")
# Dataset with multiple RT
if aggr_rlz==False:
    Q_rps = []
    for i in range(len(csv_fns)):
        # Read csv file
        sim_all = pd.read_csv(csv_fns[i], index_col=0, parse_dates=True)
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
        df_highpulse = highpulse(sim)
        df_wetmonth = wetmonth_mean(sim)
        # Low flows
        df_RT7d = returninterval_Q7d(sim, Tlow)
        df_Q7dmin = Q7d_min(sim)
        df_lowpulse = lowpulse(sim)
        df_drymonth = drymonth_mean(sim)
        df_BFI = BFI(sim)

        # Get stress test stats
        rlz_nb = int(os.path.basename(csv_fns[i]).split(".")[0].split("_")[2])
        st_nb = os.path.basename(csv_fns[i]).split(".")[0].split("_")[-1]
        if st_nb == "0":
            tavg = 0
            prcp = 0
        else:
            df_st = pd.read_csv(f"{exp_dir}/stress_test/cst_{st_nb}.csv")
            tavg = df_st["temp_mean"].iloc[0]
            prcp = df_st["precip_mean"].iloc[0] * 100 - 100  # change in %
        
        # Update discharge statistics tableslen
        df_out_mean.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_mean.values.round(2))
        df_out_max.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_max.values.round(2))
        df_out_min.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_min.values.round(4))
        df_out_q95.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_q95.values.round(2))
        df_out_RT.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_RT.values.round(2))
        df_out_Q7dmax.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_Q7dmax.values.round(2))
        df_out_highpulse.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_highpulse.values.round(2))
        df_out_wetmonth.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_wetmonth.values.round(2))
        df_out_RT7d.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_RT7d.values.round(4))
        df_out_Q7dmin.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_Q7dmin.values.round(4))
        df_out_lowpulse.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_lowpulse.values.round(2))
        df_out_drymonth.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_drymonth.values.round(4))
        df_out_BFI.iloc[i, :] = np.append((rlz_nb, tavg, prcp), df_BFI.values.round(4))

        # Update return interval dataset
        Q_rp = returnintervalmulti(sim)
        # Add realization as new coords
        Q_rp = Q_rp.assign_coords(scenario=i)
        # Add a new dim for realization number
        Q_rp = Q_rp.expand_dims("scenario")
        # Add tavg coords that are function of scenario dim
        Q_rp = Q_rp.assign_coords(realization=("scenario", [rlz_nb]))
        Q_rp = Q_rp.assign_coords(tavg=("scenario", [tavg]))
        Q_rp = Q_rp.assign_coords(prcp=("scenario", [prcp]))
        Q_rps.append(Q_rp)

        # Update basin average statistics table
        stats_basavg = np.array([rlz_nb, tavg, prcp])
        # # wflow saves basin average and not sum, use basin area to get total water volume in m3
        # geom = mod.staticgeoms["basins"]
        # crs = hydromt.gis_utils.utm_crs(geom.total_bounds)
        # geom = geom.to_crs(crs)
        # bas_area = geom.geometry.area[0] # assume only one basin
        sim = sim_all[basavg_vars]
        for v in basavg_vars:
            if v == "snow_basavg":
                # Maximum snow water equivalent per year (mm/yr)
                stats_basavg = np.append(stats_basavg, (sim[v].resample('a').max().mean())) # * bas_area))
            else: #actual evapotranspiration_basavg or groundwater recharge_basavg or overland_flow_basavg
                # Total evaporation or recharge or overland flow volume (mm/yr)
                stats_basavg = np.append(stats_basavg, (sim[v].resample('a').sum().mean())) # * bas_area))
        df_out_basavg.iloc[i, :] = stats_basavg.round(1)

    print("Writting tables for 2D stress tests plots")
    df_out_mean.to_csv(mean_fn, index=False)
    df_out_max.to_csv(max_fn, index=False)
    df_out_min.to_csv(min_fn, index=False)
    df_out_q95.to_csv(q95_fn, index=False)
    df_out_RT.to_csv(RT_fn, index=False)
    df_out_Q7dmax.to_csv(Q7dmax_fn, index=False)
    df_out_highpulse.to_csv(highpulse_fn, index=False)
    df_out_wetmonth.to_csv(wetmonth_fn, index=False)
    df_out_RT7d.to_csv(RT7d_fn, index=False)
    df_out_Q7dmin.to_csv(Q7dmin_fn, index=False)
    df_out_lowpulse.to_csv(lowpulse_fn, index=False)
    df_out_drymonth.to_csv(drymonth_fn, index=False)
    df_out_BFI.to_csv(BFI_fn, index=False)
    df_out_basavg.to_csv(bas_fn, index=False)

    # Merge Qrps list and save as one csv per loc
    Q_rps = xr.concat(Q_rps, dim="scenario")
    for v in Q_rps.data_vars:
        df_rp = Q_rps[v].to_pandas().round(1)
        # Reorder dims of Q_rp
        df_rp["realization"] = Q_rps["realization"].values
        df_rp["tavg"] = Q_rps["tavg"].values
        df_rp["prcp"] = Q_rps["prcp"].values
        # Change column order of df
        cols = df_rp.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        df_rp = df_rp[cols]
        # Save to csv
        df_rp.to_csv(os.path.join(os.path.dirname(RT_fn), f"RT_{v}.csv"), index=False)

if aggr_rlz==True:
    Q_rps = []
    for i in range(st_num):
        csv_fns_i = [x for x in csv_fns if x.endswith(str(i+1)+'.csv')]
        sim_i = sim[Q_vars]
        Q7dmax_list = []
        Q7dmin_list = []
        wetmonth_list = []
        drymonth_list = []
        for j in range(len(csv_fns_i)): 
            sim_all = pd.read_csv(csv_fns_i[j], index_col=0, parse_dates=True)
            sim_j = sim_all[Q_vars]
            Q7dmax_list.append(Q7d_total(sim_j).max())
            Q7dmin_list.append(Q7d_total(sim_j).min())
            drymonth_list.append(drymonth_mean(sim_j).min())
            wetmonth_list.append(wetmonth_mean(sim_j).max())
            sim_i = pd.concat([sim_i, sim_j])
        sim_i.index = pd.date_range(start=sim_i.index[0], periods=len(sim_i), name='time')
        # Get statistics
        # Average Yearly statistics
        df_mean = sim_i.resample('a').mean().mean()
        df_max = sim_i.resample('a').max().mean()
        df_min = sim_i.resample('a').min().mean()
        df_q95 = sim_i.resample('a').quantile(0.95).mean()
        # High flows
        df_RT = returninterval(sim_i, Tpeak)
        df_Q7dmax = Q7d_max(sim_i)
        df_highpulse = highpulse(sim_i)
        df_wetmonth = wetmonth_mean(sim_i)
        # Low flows
        df_RT7d = returninterval_Q7d(sim_i, Tlow)
        df_Q7dmin = Q7d_min(sim_i)
        df_lowpulse = lowpulse(sim_i)
        df_drymonth = drymonth_mean(sim_i)
        df_BFI = BFI(sim_i)

        # Get stress test stats
        #rlz_nb = int(os.path.basename(csv_fns[i]).split(".")[0].split("_")[2])
        st_nb = i+1 #os.path.basename(csv_fns[i]).split(".")[0].split("_")[-1]
        if st_nb == 0:
            tavg = 0
            prcp = 0
        else:
            df_st = pd.read_csv(f"{exp_dir}/stress_test/cst_{st_nb}.csv")
            tavg = df_st["temp_mean"].iloc[0]
            prcp = df_st["precip_mean"].iloc[0] * 100 - 100  # change in %
        
        # Update discharge statistics tableslen
        df_out_mean.iloc[i, :] = np.append((tavg, prcp), df_mean.values.round(2))
        df_out_max.iloc[i, :] = np.append((tavg, prcp), df_max.values.round(2))
        df_out_min.iloc[i, :] = np.append((tavg, prcp), df_min.values.round(4))
        df_out_q95.iloc[i, :] = np.append((tavg, prcp), df_q95.values.round(2))
        df_out_RT.iloc[i, :] = np.append((tavg, prcp), df_RT.values.round(2))
        df_out_Q7dmax.iloc[i, :] = np.append((tavg, prcp), df_Q7dmax.values.round(2))
        df_out_highpulse.iloc[i, :] = np.append((tavg, prcp), df_highpulse.values.round(2))
        df_out_wetmonth.iloc[i, :] = np.append((tavg, prcp), df_wetmonth.values.round(2))
        df_out_RT7d.iloc[i, :] = np.append((tavg, prcp), df_RT7d.values.round(4))
        df_out_Q7dmin.iloc[i, :] = np.append((tavg, prcp), df_Q7dmin.values.round(4))
        df_out_lowpulse.iloc[i, :] = np.append((tavg, prcp), df_lowpulse.values.round(2))
        df_out_drymonth.iloc[i, :] = np.append((tavg, prcp), df_drymonth.values.round(4))
        df_out_BFI.iloc[i, :] = np.append((tavg, prcp), df_BFI.values.round(4))

        # Update return interval dataset
        Q_rp = returnintervalmulti(sim_i)
        # Add realization as new coords
        Q_rp = Q_rp.assign_coords(scenario=i)
        # Add a new dim for realization number
        Q_rp = Q_rp.expand_dims("scenario")
        # Add tavg coords that are function of scenario dim
        #Q_rp = Q_rp.assign_coords(realization=("scenario", [rlz_nb]))
        Q_rp = Q_rp.assign_coords(tavg=("scenario", [tavg]))
        Q_rp = Q_rp.assign_coords(prcp=("scenario", [prcp]))
        Q_rps.append(Q_rp)

        # Update basin average statistics table
        stats_basavg = np.array([tavg, prcp])
        # # wflow saves basin average and not sum, use basin area to get total water volume in m3
        # geom = mod.staticgeoms["basins"]
        # crs = hydromt.gis_utils.utm_crs(geom.total_bounds)
        # geom = geom.to_crs(crs)
        # bas_area = geom.geometry.area[0] # assume only one basin
        sim_v = sim_all[basavg_vars]
        for v in basavg_vars:
            if v == "snow_basavg":
                # Maximum snow water equivalent per year (mm/yr)
                stats_basavg = np.append(stats_basavg, (sim_v[v].resample('a').max().mean())) # * bas_area))
            else: #actual evapotranspiration_basavg or groundwater recharge_basavg or overland_flow_basavg
                # Total evaporation or recharge or overland flow volume (mm/yr)
                stats_basavg = np.append(stats_basavg, (sim_v[v].resample('a').sum().mean())) # * bas_area))
        df_out_basavg.iloc[i, :] = stats_basavg.round(1)

    print("Writting tables for 2D stress tests plots")
    df_out_mean.to_csv(mean_fn, index=False)
    df_out_max.to_csv(max_fn, index=False)
    df_out_min.to_csv(min_fn, index=False)
    df_out_q95.to_csv(q95_fn, index=False)
    df_out_RT.to_csv(RT_fn, index=False)
    df_out_Q7dmax.to_csv(Q7dmax_fn, index=False)
    df_out_highpulse.to_csv(highpulse_fn, index=False)
    df_out_wetmonth.to_csv(wetmonth_fn, index=False)
    df_out_RT7d.to_csv(RT7d_fn, index=False)
    df_out_Q7dmin.to_csv(Q7dmin_fn, index=False)
    df_out_lowpulse.to_csv(lowpulse_fn, index=False)
    df_out_drymonth.to_csv(drymonth_fn, index=False)
    df_out_BFI.to_csv(BFI_fn, index=False)
    df_out_basavg.to_csv(bas_fn, index=False)

    # Merge Qrps list and save as one csv per loc
    Q_rps = xr.concat(Q_rps, dim="scenario")
    for v in Q_rps.data_vars:
        df_rp = Q_rps[v].to_pandas().round(1)
        # Reorder dims of Q_rp
        #df_rp["realization"] = Q_rps["realization"].values
        df_rp["tavg"] = Q_rps["tavg"].values
        df_rp["prcp"] = Q_rps["prcp"].values
        # Change column order of df
        cols = df_rp.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        df_rp = df_rp[cols]
        # Save to csv
        df_rp.to_csv(os.path.join(os.path.dirname(RT_fn), f"RT_{v}.csv"), index=False)