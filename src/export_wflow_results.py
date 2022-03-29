# -*- coding: utf-8 -*-
"""
Export wflow results for easy plotting of the climate response plots
"""
import os
import pandas as pd
import numpy as np

import hydromt
from hydromt_wflow.utils import read_csv_results

# Snakemake options
csv_fns = snakemake.input.rlz_csv_fns
exp_dir = snakemake.params.exp_dir

model_dir = snakemake.params.model_dir

mean_fn = snakemake.output.mean
max_fn = snakemake.output.max
min_fn = snakemake.output.min
q95_fn = snakemake.output.q95

# Read the wflow models and results
print("Reading wflow model")
mod = hydromt.WflowModel(root=model_dir, mode="r")

# Get output discharge columns
sim = pd.read_csv(csv_fns[0], index_col=0, parse_dates=True)
Q_vars = [x for x in sim.columns if x.startswith('Q_')]
col_names = ['tavg', 'prcp']
col_names.extend(Q_vars)

df_out_mean = pd.DataFrame(
    data = np.zeros((len(csv_fns),len(Q_vars)+2)),
    columns = col_names, #["tavg", "prcp", "Mean", "Max", "Min", "Q95"],
    dtype = "float32",
)
df_out_max = df_out_mean.copy()
df_out_min = df_out_mean.copy()
df_out_q95 = df_out_mean.copy()

print("Computing discharge stats for each realization/stress test")
for i in range(len(csv_fns)):
    # Read csv file
    sim = pd.read_csv(csv_fns[i], index_col=0, parse_dates=True)
    sim = sim[Q_vars]
    # Get statistics
    df_mean = sim.mean()
    df_max = sim.max()
    df_min = sim.min()
    df_q95 = sim.quantile(0.95)
    # Get stress test stats
    st_nb = os.path.basename(csv_fns[i]).split(".")[0].split("_")[-1]
    if st_nb == '0':
        tavg = 0
        prcp = 0
    else:
        df_st = pd.read_csv(f"{exp_dir}/stress_test/cst_{st_nb}.csv")
        tavg = df_st['temp_mean'].iloc[0]
        prcp = df_st['precip_mean'].iloc[0]*100 - 100 # change in %
    df_out_mean.iloc[i, :] = np.append((tavg, prcp), df_mean.values).round(2)
    df_out_max.iloc[i, :] = np.append((tavg, prcp), df_max.values).round(2)
    df_out_min.iloc[i, :] = np.append((tavg, prcp), df_min.values).round(2)
    df_out_q95.iloc[i, :] = np.append((tavg, prcp), df_q95.values).round(2)

print("Writting tables for 2D stress tests plots")
df_out_mean.to_csv(mean_fn, index=False)
df_out_max.to_csv(max_fn, index=False)
df_out_min.to_csv(min_fn, index=False)
df_out_q95.to_csv(q95_fn, index=False)
    