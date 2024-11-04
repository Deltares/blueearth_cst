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
from metrics import kge, nselog_mm7q, mae_peak_timing, mape_peak_magnitude, weighted_euclidean
from metrics import _obs_peaks, _sim_peaks
from filelock import FileLock
import pdb 

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from wflow.wflow_utils import get_wflow_results
else:
    from .wflow.wflow_utils import get_wflow_results

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
    path_parts = Path(file_path).parts
    params_dict = {part.split('~')[0]:np.float64(part.split('~')[1]) for part in path_parts if '~' in part}  # Assuming the parameters are in the third last part of the path
    return params_dict

def main(
    l,
    modelled: Path | str,
    observed: Path | str,
    dry_month: list,
    window: int,
    gauges: tuple | list | Path,
    params: dict | str,
    starttime: str,
    endtime: str,
    metrics: tuple | list,
    weights: tuple | list,
    out: Path | str,
    gid: str,
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
    
    METRICS = {metric: globals()[metric] for metric in metrics}
    
    out_dir = Path(out).parent
    os.makedirs(out_dir, exist_ok=True)
    
    
    #Old version read the results file directly
    # if observed:
    if observed.endswith(".nc"):
        #TODO: make coords compatible with the standard hmt obs reads
        obs = xr.open_dataset(observed)
    elif observed.endswith(".csv") or observed.endswith(".xlsx"):
        da_ts_obs = hydromt.io.open_timeseries_from_table(
            observed, name="Q", index_dim="wflow_id", sep=";"
        )
    else:
        raise ValueError(f"Unknown file type for observed data: {observed}")

    #CST compatible import
    if modelled.endswith(".toml"):
        if isinstance(gauges, list):
            raise ValueError("Gauges should be a path to a csv with wflow_id and station_name when using modelled data")
        
        md, _, _ = get_wflow_results(
            wflow_root=os.path.dirname(modelled),
            config_fn=os.path.basename(modelled),
            gauges_locs=gauges,
            remove_warmup=True,
        )
    
    #Older 
    elif modelled.endswith(".nc"):
        md = xr.open_dataset(modelled)
        
    
    obs = obs.sel(runs='Obs.', time=slice(starttime, endtime)) 

    metric_values = {
        item: [] for item in metrics
    }
    evals=[]
    
    md = md.sel(time=slice(starttime, endtime))
    
    if len(md.time.values) == 0:
        l.warning(f"\n{'*'*10}\n{modelled}\nIs not a complete time series, Skipping...\n{'*'*10}")
        with open(Path(out_dir, "failed.nc"), 'a') as f:
            f.write(f"")
        raise ValueError(f"{modelled} is not a complete time series")
    
    params = parse_params_from_path(modelled)
    
    pdb.set_trace()

    if md[gid].dtype != int:
        md[gid] = md[gid].astype(np.int64)

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
    
    l.info(f"sim time: {md.time}")
    l.info(f"obs time: {obs.time}")
    
    
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
        l.info(f"Calculated {metric} in {end-start} seconds")
        l.info(f"{metric} value: {e}")
        evals.append(e)
    
    l.info(f"Calculated metrics for {modelled}")
    
    # Get the euclidean distance
    res = weighted_euclidean(
        np.array(evals),
        weights=weights,
        weighted=True,
    )
    
    # Extract the level from the modelled file path
    # allowing easy access to eucldean 
    level = None
    for part in Path(modelled).parts:
        if "level" in part:
            level = part
            break
    split = str(out).split(level)[0]
    results_file = Path(split) / level / f"results_{level}.txt"
    l.info(f"appending results to: {results_file}")
     
    append_results_to_file(results_file, gauges, modelled, res, l)  # Call the new function

    param_coords = create_index_from_params(params)
    
    ds = None
    for metric in metrics:
        l.info(metric)
        #metric_values[metric] has an array of len 60 at level 0
        da = xr.DataArray(
            metric_values[metric], 
            coords=[('params', param_coords), ('gauges', gauges)], 
            dims=['params', 'gauges'],
            attrs={"metric": metric}
        )
        l.info(da)
        da.name = metric
        if ds is None:
            ds = da.to_dataset()
            continue

        ds[metric] = da
    
    # l.info(ds)
    
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
        Path(out)
    )
    l.info(f"Saved the performance dataset to {out_dir}")
    #
    
    with open(out_dir / "evaluate.done", "w") as f:
        f.write("")


def append_results_to_file(results_file: Path, gauges: list, modelled: Path, res: np.ndarray, l) -> None:
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
                header = "params," + ",".join(map(str, gauges)) + "\n"
                f.write(header)

            # Write the file path and distances
            f.write(f"{parse_params_from_path(Path(modelled))}," + ",".join(map(str, res)) + "\n")


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
            main(
                l,
                modelled=mod.input.sim,
                observed=mod.params.observed,
                dry_month=mod.params.dry_month,
                window=mod.params.window,
                gauges=mod.params.gauges,
                params=mod.params.params,
                starttime=mod.params.starttime,
                endtime=mod.params.endtime,
                metrics=mod.params.metrics,
                weights=mod.params.weights,
                out=mod.output.performance,
                gid=mod.params.gaugeset,
            )

        else:
            raise ValueError("NO TEST DATA SET UP IN EVALUATE")
            from src.calibration.create_params import create_set
            import yaml
            from snakemake.utils import Paramspace

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
            calibration_parameters = cfg["calibration_parameters"].format(DRIVE)
            calibration_parameters = calibration_parameters.format(DRIVE)
            lnames, methods, df, wflow_vars = create_set(calibration_parameters)
            paramspace = Paramspace(df, filename_params="*")
            print(paramspace)

    except Exception as e:
        l.exception(e)
        l.error(traceback.format_exc())
        raise e
