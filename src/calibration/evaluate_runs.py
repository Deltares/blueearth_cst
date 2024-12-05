from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
# import hydromt_wflow
import hydromt
from setuplog import setup_logging
import traceback
from timeit import default_timer as timer
from metrics import *
from metrics import _obs_peaks, _sim_peaks
from filelock import FileLock
import pdb 
from rich.console import Console
from rich.traceback import Traceback
console = Console()
# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from src.wflow.wflow_utils import get_wflow_results
else:
    from wflow.wflow_utils import get_wflow_results

def create_index_from_params(params: dict) -> pd.Index:
    """
    Create a Pandas Index from a single set of parameters.
    
    Args:
        params (dict): Dictionary containing parameter names as keys and a single set of values.

    Returns:
        pd.Index: A Pandas Index object representing the single set of parameters.
    """
    keys = list(params.keys())
    params_values = list(params.values())
    
    # index = pd.MultiIndex.from_tuples([tuple(params_values)], names=keys)
    index = pd.Index([tuple(params_values)], name=tuple(keys))
    return index

def parse_params_from_path(file_path):
    # Extract the part of the path that contains the parameters
    path_parts = Path(file_path).parts
    params_dict = {part.split('~')[0]:np.float64(part.split('~')[1]) for part in path_parts if '~' in part}  # Assuming the parameters are in the third last part of the path
    return params_dict

def create_index_from_params(params: dict) -> pd.Index:
    """
    Create a Pandas Index from a single set of parameters.
    
    Args:
        params (dict): Dictionary containing parameter names as keys and a single set of values.

    Returns:
        pd.Index: A Pandas Index object representing the single set of parameters.
    """
    keys = list(params.keys())
    params_values = list(params.values())
    
    # index = pd.MultiIndex.from_tuples([tuple(params_values)], names=keys)
    index = pd.Index([tuple(params_values)], name=tuple(keys))
    return index

def parse_params_from_path(file_path):
    # Extract the part of the path that contains the parameters
    if isinstance(file_path, Path):
        file_path = str(file_path)
    remove = [".toml", ".csv", ".nc"]
    
    if '/' in file_path:
        file_path = file_path.split('/')[-1]

    for r in remove:
        if file_path.endswith(r):
            file_path = file_path[:-len(r)]
            break
    
    remove = ["output_", "wflow_sbm_"]
    
    for r in remove:
        if file_path.startswith(r):
            file_path = file_path[len(r):]
            break
    param_str = file_path
    path_parts = file_path.split("_")
    parts = [part.split("~") for part in path_parts if "~" in part]
    params_dict = {part[0]:np.float64(part[1]) for part in parts}
    
    return params_dict, param_str
    

def main(
    l,
    modelled: Path | str,
    observed: Path | str,
    gauges: tuple | list | Path,
    params: dict | str,
    starttime: str | None,
    splittime: str,
    endtime: str | None,
    metrics: tuple | list,
    weights: tuple | list,
    out: Path | str,
    gid: str | None,
    outflow: dict | None,
    dry_month: list | None,
    window: int | None,
):
    """
    Perform evaluation of model parameters.

    Args:
        modelled (tuple | list): List of paths to modelled data files.
        observed (Path | str): Path to observed data file.
        dry_month (list): List of dry months.
        window (int): Window size for peak calculation.
        gauges (tuple | list): List of gauge IDs.
        params (dict | str): Parameters for the model.
        starttime (str): Start time for the evaluation period.
        endtime (str): End time for the evaluation period.
        metrics (tuple | list): List of metrics to evaluate.
        weights (tuple | list): List of weights for the metrics.
        out (Path | str): Output directory.
        gid (str): Gauge key for simulation data.
    """
    #original example indexed by 'Q_gauges_obs' and 'time'
    #we use gid 
    #our data is indexed by 'wflow_id' 'runs' and 'time'

    if starttime is None:
        starttime = splittime
        perf_str = "cal"
    if endtime is None:
        endtime = splittime
        perf_str = "eval"
    METRICS = {metric: globals()[metric] for metric in metrics}
    
    out_dir = Path(out).parent
    os.makedirs(out_dir, exist_ok=True)
    
    if gid == None:
        gid = "index"
    
    #Old version read the results file directly
    # if observed:
    if observed.endswith(".nc"):
        #TODO: make coords compatible with the standard hmt obs reads
        obs = xr.open_dataset(observed)
    elif observed.endswith(".csv") or observed.endswith(".xlsx"):
        obs = hydromt.io.open_timeseries_from_table(
            observed, name="Q", index_dim="wflow_id", sep=";"
        )
    else:
        raise ValueError(f"Unknown file type for observed data: {observed}")

    #CST compatible import
    if modelled.endswith(".toml") and isinstance(gauges, str | Path):
        if isinstance(gauges, list):
            raise ValueError("Gauges should be a path to a csv with wflow_id and station_name when using modelled data")
        # wflow_root = os.path.dirname(modelled)
        # config_fn = os.path.basename(modelled)
        # gauges_locs = os.path.join(wflow_root, gauges)

        md, _, _ = get_wflow_results(
            wflow_root=os.path.dirname(modelled),
            config_fn=os.path.basename(modelled),
            gauges_locs=gauges,
            remove_warmup=True,
        )
    
    #Older 
    elif modelled.endswith(".nc"):
        md = xr.open_dataset(modelled)
        
    if isinstance(obs, xr.Dataset):
        obs = obs.sel(runs='Obs.', time=slice(starttime, endtime)) 
    else:
        obs = obs.sel(time=slice(starttime, endtime))
        
    metric_values = {
        item: [] for item in metrics
    }
    
    evals=[]
    
    md = md.sel(time=slice(starttime, endtime))

    if outflow:
        gauges = [outflow["wflow_id"]]

    if len(md.time.values) == 0:
        l.warning(f"\n{'*'*10}\n{modelled}\nIs not a complete time series, Skipping...\n{'*'*10}")
        with open(Path(out_dir, "failed.nc"), 'a') as f:
            f.write(f"")
        raise ValueError(f"{modelled} is not a complete time series")
    
    params, param_str = parse_params_from_path(modelled)

    if md[gid].dtype != int:
        md[gid] = md[gid].astype(np.int64)

    #make sure a dataset with Q variable is returned
    if isinstance(md, xr.DataArray):
        md = xr.Dataset({'Q': md})

    if isinstance(obs, xr.DataArray):
        obs = xr.Dataset({'Q': obs})
    
    if any("peak" in metric for metric in metrics):
        start = timer()
        obs_peaks = {
            g: _obs_peaks(obs.sel(wflow_id=g).Q)
            for g in gauges
        }
        peaks = {
            g: _sim_peaks(sim=md.sel({gid: g}).Q, 
                          obs=obs_peaks[g],
                          window=window,)
            for g in gauges
        }
        end = timer()
        l.info(f"Calculated peaks in {end-start} seconds")
    else:
        peaks = None
    
    md, obs = xr.align(md,obs, join='inner')
    
    # Calculate the metric values
    for metric in metrics:
        start = timer()
        metric_func = METRICS.get(metric)
        if not metric_func:
            raise ValueError(f"Metric '{metric}' is not defined in METRICS.")
        
        # Check if additional parameters are needed based on the metric type
        if metric == "nselog_mm7q":
            e = metric_func(md, obs, dry_month, gauges, gid)
        elif metric in {"mae_peak_timing", "mape_peak_magnitude"}:
            e = metric_func(peaks, window)
        else:
            e = metric_func(md, obs, gauges, gid)
        
        # Special case for the 'kge' metric to extract specific component    
        if metric == "kge":
            e = e["kge"]
        
        metric_values[metric].append(e)
        end = timer()
        evals.append(e)
    
    # l.info(f"Calculated metrics for {modelled}")
    
    # Get the euclidean distance
    res = weighted_euclidean(
        np.array(evals),
        weights=weights,
        weighted=True,
    )
    
    # Extract the level from the modelled file path
    # allowing easy access to eucldean 
    results_file = Path(out_dir, f"performance_{perf_str}.txt")
    l.info(f"appending results to: {results_file}")
     
    append_results_to_file(results_file, gauges, modelled, res, metric_values, param_str, l)  # Call the new function

    param_coords = create_index_from_params(params)

    ds = None
    for metric in metrics:
        #metric_values[metric] has an array of len 60 at level 0
        da = xr.DataArray(
            metric_values[metric], 
            coords=[('params', param_coords), ('gauges', gauges)], 
            dims=['params', 'gauges'],
            attrs={"metric": metric}
        )
        # l.info(da)
        da.name = metric
        if ds is None:
            ds = da.to_dataset()
            continue

        ds[metric] = da
    
    
    if res.ndim == 1:
        res = res.reshape((len(param_coords), len(gauges)))
    
    ds["euclidean"] = xr.DataArray(
        res, 
        coords=[('params', param_coords), ('gauges', gauges)], 
        dims=['params', 'gauges']
    )

    ds = ds.assign_attrs(
        {
            "metrics": metrics,
            "weights": weights,
        }
    )

    ds = ds.unstack()
    ds.to_netcdf(
        Path(str(out).format(perf_str))
    )
    
    l.info(f"for params: {param_str}")
    for i, gauge in enumerate(gauges):
        l.info(f"gauge: {i}")
        for j, (key, value) in enumerate(metric_values.items()):
            l.info(f"{gauge} -- o_i * w_i: {key} * {weights[j]} = {value[i][0]}, euclidean: {res[i][0]}")

    l.info(f"Saved the performance dataset to {out_dir}")
    assert Path(str(out).format(perf_str)).exists(), f"Performance file not found: {str(out).format(perf_str)}"
    assert Path(Path(out).parent, f"performance_{perf_str}.txt").exists(), f"Performance appended file not found: {Path(out).parent, f'performance_{perf_str}.txt'}"
    


def append_results_to_file(results_file: Path, 
                           gauges: list, 
                           modelled: Path,
                           res: np.ndarray,
                           metric_values: dict, 
                           param_str: str,
                           l) -> None:
    """
    Append results to the specified results file with file locking.
    This method is preferred over multidimensional evaluation files,
    Multiple parameters multiplied by runs == a LOT of dimensions.
    Prefer the evaluate to individual files and then append the summary.
    """
    # Use FileLock to ensure safe file access
    lock_path = results_file.with_suffix('.lock')
    with FileLock(str(lock_path)):
        # Append results to the specified results file
        with open(results_file, 'a') as f:
            # Write the header if the file is empty
            if os.stat(results_file).st_size == 0:
                # Create the header with parameters and metrics
                header = "params,gauge,euclidean," + ",".join(metric_values.keys()) + "\n"
                f.write(header)
                
            for i, gauge in enumerate(gauges):
                # Prepare the row with parameters, followed by gauge results and metric values
                row = f"{param_str},{gauge},{res[i]}," + ",".join([','.join(map(str, metric_values[metric][i])) for metric in metric_values.keys()]) + "\n"
                f.write(row)

if __name__ == "__main__":
    
    """
    This module is used to evaluate parameters for a model. 

    it will evaluate per run returning a netcdf file with the performance metrics for each run.
    
    This can be grouped across multiple cpus and thus is much more efficient than the old file loop. 

    """
    
    l = setup_logging("data/0-log", "04-evaluate_param_per_run.log")
    try:
        if "snakemake" in globals():
            mod = globals()["snakemake"]
            l.info(f"{'*'*10}\nEvaluating {mod.input.sim} for calibration period\n{'*'*10}")
            main(
                l,
                modelled=mod.input.sim,
                observed=mod.params.observed,
                gauges=mod.params.gauges,
                params=mod.params.params,
                starttime=mod.params.starttime,
                splittime=mod.params.caleval_split,
                endtime=None, #CALIBRATION SPLIT
                metrics=mod.params.metrics,
                weights=mod.params.weights,
                out=mod.output.cal_file,
                outflow=mod.params.outflow,
                gid=None, #gauge index for sim, defaults to index
                dry_month=None, #for NM7Q
                window=None, #for peak timing
            )
            l.info(f"{'*'*10}\nEvaluating {mod.input.sim} for evaluation period\n{'*'*10}")
            main(
                l,
                modelled=mod.input.sim,
                observed=mod.params.observed,
                gauges=mod.params.gauges,
                params=mod.params.params,
                starttime=None, #EVALUATION SPLIT
                splittime=mod.params.caleval_split,
                endtime=mod.params.endtime,
                metrics=mod.params.metrics,
                weights=mod.params.weights,
                out=mod.output.eval_file, #EVALUATION SPLIT
                outflow=mod.params.outflow,
                gid=None, #gauge index for sim, defaults to index
                dry_month=None, #for NM7Q
                window=None, #for peak timing
            )

        else:
            # raise ValueError("NO TEST DATA SET UP IN EVALUATE")
            from src.calibration.create_params import create_set
            import yaml
            from snakemake.utils import Paramspace
            import shutil
            if sys.platform.startswith("win"):
                DRIVE="p:"
            elif sys.platform.startswith("lin"):
                DRIVE="/p"
            else:
                raise ValueError(f"Unsupported platform for formatting drive location: {sys.platform}")
            
            config = "{}/11210673-fao/14 Subbasins/run_configs/2_calibration/snake_calibration_config_damchhu_soil_cal.yml".format(DRIVE)
            with open(config) as f:
                cfg = yaml.safe_load(f)
            
            root = cfg["wflow_root"].format(DRIVE)
            calibration_parameters = os.path.join(root, cfg["calibration_parameters"])
            lnames, methods, df, wflow_vars = create_set(calibration_parameters)
            paramspace = Paramspace(df, filename_params="*")
            wildcard_pattern = paramspace.wildcard_pattern
            params = [part.split('~')[0] for part in wildcard_pattern.split('_')]
            param_values = {}
            
            for param in params:
                param_values[param] = paramspace.get(param)
            
            formatted_string = wildcard_pattern
            
            for param in params:
                value = param_values[param].iloc[0]  # Get the first value
                formatted_string = formatted_string.replace(f'{{{param}}}', str(value))
            
            calib_folder = os.path.join(root, cfg["calibration_runs_folder"])
            calib_file = os.path.join(calib_folder, f"wflow_sbm_{formatted_string}.toml")
            test_file = os.path.join(calib_folder, f"test_{formatted_string}.toml")

            #make a testcopy
            shutil.copy(calib_file, test_file)
            l.info(f"{'*'*10}\nEvaluating {test_file} for calibration period\n{'*'*10}")
            main(
                l,
                modelled=test_file,
                observed=cfg["observations_timeseries"].format(DRIVE),
                gauges=cfg["observations_locations"].format(DRIVE),
                params=formatted_string,
                starttime=cfg["starttime"],
                splittime=cfg["caleval"],
                endtime=None,
                metrics=cfg["metrics"],
                weights=cfg["weights"],
                out=os.path.join(calib_folder, f"performance_{formatted_string}.nc"),
                gid=None,
                outflow=cfg["outflow"],
                dry_month=None,
                window=None,
            )
            l.info(f"{'*'*10}\nEvaluating {test_file} for evaluation period\n{'*'*10}")
            main(
                l,
                modelled=test_file,
                observed=cfg["observations_timeseries"].format(DRIVE),
                gauges=cfg["observations_locations"].format(DRIVE),
                params=formatted_string,
                starttime=None,
                splittime=cfg["caleval"],  
                endtime=cfg["endtime"],
                metrics=cfg["metrics"],
                weights=cfg["weights"],
                out=os.path.join(calib_folder, f"performance_{formatted_string}.nc"),
                gid=None,
                outflow=cfg["outflow"],
                dry_month=None,
                window=None,
            )


    except Exception:
        console.print_exception()
        # l.exception(e)
        # l.error(traceback.format_exc())
        # raise e
