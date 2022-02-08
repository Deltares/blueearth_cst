import os
import pandas as pd
import numpy as np
import yaml

#%% Snake parameters
config_fn =  snakemake.input.config
csv_fns = snakemake.output.st_csv_fns

#%% Read the yaml config
with open(config_fn, "r") as stream: 
    yml = yaml.load(stream, Loader=yaml.FullLoader)

# Temperature change attributes
delta_temp_mean_min = yml["temp"]["mean"]["min"]
delta_temp_mean_max = yml["temp"]["mean"]["max"]
temp_step_num = yml["temp"]["step_num"]

# Precip change attributes
delta_precip_mean_min = yml["precip"]["mean"]["min"]
delta_precip_mean_max = yml["precip"]["mean"]["max"]
delta_precip_variance_min = yml["precip"]["variance"]["min"]
delta_precip_variance_max = yml["precip"]["variance"]["min"]
precip_step_num = yml["precip"]["step_num"]

# Number of stress tests
ST_NUM = temp_step_num * precip_step_num
#%% Stress test values per variables
temp_values = np.linspace(delta_temp_mean_min, delta_temp_mean_max , temp_step_num, axis=1)
precip_values = np.linspace(delta_precip_mean_min, delta_precip_mean_max , precip_step_num, axis=1)
precip_var_values = np.linspace(delta_precip_variance_min, delta_precip_variance_max , precip_step_num, axis=1)

#%% Generate csv file for each stress test scenario
i = 0
for j in range(temp_step_num):
    temp_j = temp_values[:,j]
    for k in range(precip_step_num):
        precip_k = precip_values[:,k]
        precip_var_k = precip_var_values[:,k]

        #Create df and save to csv
        data = {'temp_mean': temp_j, 'precip_mean': precip_k, 'precip_variance': precip_var_k}
        df = pd.DataFrame(data=data, dtype=np.float32, index=np.arange(1, 13))
        df.index.name = "month"
        csv_fn = csv_fns[i]
        df.to_csv(csv_fn)

        i+=1