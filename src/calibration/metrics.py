import hydromt.stats as stats
from hydromt.stats import kge as kge_ds, skills
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import argrelmax, argrelmin, find_peaks


"""
:: CONTRIBUTORS :: 
michaelohanrahan
jingdeng
dalmijn

originally written for RWSOS Wflow Calibration 2024.

"""

def kge(
    sim: xr.Dataset,
    obs: xr.Dataset,
    gauges: tuple | list,
    gid: str,
):
    """_summary_"""
    res = {
        "kge": [],
        "kge_pearson_coef": [],
        "kge_rel_var": [],
        "kge_bias": [],
    }

    for g in gauges:
        da = kge_ds(
            sim.sel({gid:g}).Q,
            obs.sel(wflow_id=g).Q,
        )

        for var in da.data_vars:
            res[var].append(
                round(float(da[var].values),4)
            )

    return res


def nse(
    sim: xr.Dataset,
    obs: xr.Dataset,
    gauges: tuple | list,
    gid: str,
):
    res = []
    
    for g in gauges:
        da = skills.nashsutcliffe(
            sim.sel({gid:g}).Q,
            obs.sel(wflow_id=g).Q,
        )
        res.append(round(float(da.values),4))
    
    return res


def nse_log(
    sim: xr.Dataset,
    obs: xr.Dataset,
    gauges: tuple | list,
    gid: str,
):
    res = []
    
    for g in gauges:
        da = skills.lognashsutcliffe(
            sim.sel({gid:g}).Q,
            obs.sel(wflow_id=g).Q,
        )
        res.append(round(float(da.values),4))
    
    return res
    

def mm7q(
    ds: xr.Dataset | xr.DataArray,
    dry_month: list,
):
    """
    Monthly minimum 7-day discharge for selected dry months.
    
    Args:
        ds (xr.Dataset | xr.DataArray): The dataset that contains the discharge time series for single station.
        dry_month (list): List of dry months. 
                          You can specify only the start and end months of dry period, e.g. [6,9]
                          Or you can specify each month, e.g., [6,7,8,9]
        
    Returns:
        xr.Dataset | xr.DataArray: Monthly minimum 7-day discharge for selected dry month.
    """
    # get the rolling mean window based on timescale in dataset
    if xr.infer_freq(ds.time).lower() == "D":
        window = 7
    elif xr.infer_freq(ds.time).lower() == "h":
        window = 7 * 24
    elif xr.infer_freq(ds.time).lower() == "3h":
        window = 7 * int(24 / 3)
    
    # calculate the MM7Q for each month
    _mm7q = ds.rolling(time=window).mean().resample(time='M').min('time').compute()
    
    # select out the MM7Q for selected dry months
    dry_month_start = dry_month[0]
    dry_month_end = dry_month[-1]
    months = _mm7q['time'].dt.month
    mm7q_dry_month = _mm7q.sel(time=_mm7q['time'].where((months>=dry_month_start)
                                                        &(months<=dry_month_end),
                                                        drop=True))
    
    return mm7q_dry_month
    
    
def nselog_mm7q(
    sim: xr.Dataset,
    obs: xr.Dataset,
    dry_month: list,
    gauges: tuple | list,
    gid: str,
):
    """nse-log of mm7q of modeled discharge compared to observations for selected dry months and gauges

    Args:
        sim (xr.Dataset): Model dataset containing discharge values.
        obs (xr.Dataset): Observed dataset containing discharge values.
        dry_month (list): List of dry months.
        gauges (tuple | list): Tuple or list of gauges wflow_id for which needs to be calculated.

    Returns:
        List: List of nselog_mm7q values for each gauge (wflow_id).
    """
    
    res = []
    
    for g in gauges:
        sim_mm7q = mm7q(sim.sel({gid:g}).Q, dry_month)
        obs_mm7q = mm7q(obs.sel(wflow_id=g).Q, dry_month)
        nselog_mm7q = skills.lognashsutcliffe(sim_mm7q, obs_mm7q)
        res.append(round(float(nselog_mm7q.values),4))
    
    return res

def _obs_peaks(
    obs: xr.DataArray,
    distance: int = None,
    prominence: float = None, 
    datetime_coord: str = None,
):
    """
    Finds peaks in the observed time series using scipy.find_peaks.

    Parameters
    ----------
    obs : xr.DataArray
        Observed time series.
    distance: int, optional
        Required minimal horizontal distance (>= 1) in samples between neighbouring peaks. 
        Larger distance will filter out peaks that are close in time.
        Default value is 24*3=72
    prominence: float, optional
        Required prominence of peaks. The peaks with a prominence less than this are discarded.
        Larger prominence will filter out peaks that are close in time and magnitude.
        Default value is np.std(obs.values).
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically as 'time' if not specified.

    Returns
    -------
    peaks : numpy array (datetime64)
        Datetime indices of peaks in obs. 
    """
    if distance is None:
        distance = 24*3  # default value as 24*3
    
    if prominence is None:
        prominence = np.nanstd(obs.values)  # default value as 0.5 * np.std(obs.values)
        
    if datetime_coord is None:
        datetime_coord = 'time'  # default value as 'time'

    peak_indices, _ = find_peaks(obs.values, distance=distance, prominence=prominence)
    
    # Convert peak indices to datetime indices
    peaktime = obs[datetime_coord].values[peak_indices]
    peakval = obs[peak_indices].values

    return {'t': peaktime, 'Q': peakval}


def _sim_peaks(
    sim: xr.DataArray,
    obs: dict,
    window: int,
    datetime_coord: str = None,
) -> dict:
    """
    Finds the corresponding peaks in the simulated time series within a specified window around the observed peaks.

    Parameters
    ----------
    sim : xr.DataArray
        Simulated time series.
    peaks : np.array
        Datetime indices of peaks in the observed time series.
    window : int
        Size of window to consider on each side of the observed peak for finding the simulated peak.
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically as 'time' if not specified.

    Returns
    -------
    sim_peaks : dict
        A dictionary with observed peak times as keys and corresponding simulated peak times as clean string values.
    """
    if datetime_coord is None:
        datetime_coord = 'time'  # default value as 'time'

    if isinstance(sim, tuple):
        sim = sim[0]
    
    sim = sim.set_index({datetime_coord: 'time'})

    obs_t = []
    sim_t = []
    obs_q = []
    sim_q = []
    window = pd.Timedelta(hours=int(window))
    
    peaks = obs['t']
    Qdict = {k: v for k, v in zip(obs['t'], obs['Q'])}
    for idx in peaks:
        # Skip peaks at the start and end of the sequence and peaks around missing observations
        if (idx - window < sim.time.min()) or (idx + window > sim.time.max()):
            continue

        # Define peak around idx as the max value inside of the window
        values = sim.loc[idx - window: idx + window]
        if not values.isnull().all():
            peak_sim_time = values.idxmax().values
            peak_sim_val = values.loc[peak_sim_time].values
        else:
            # Handle the case when all values are NaN
            peak_sim_val = np.nan
            peak_sim_time = np.nan
            continue

        if not pd.isnull(peak_sim_time):
            obs_t.append(idx)
            obs_q.append(Qdict[idx])
            sim_t.append(peak_sim_time)
            sim_q.append(peak_sim_val)
    sim_peaks = {'t_obs': obs_t, 'Q_obs': obs_q, 't_sim': sim_t, 'Q_sim': sim_q}

    return sim_peaks


#TODO: mae has very different value ranges compared to KGE, NSE, MAPE. How to integrate to a weighted euclidean?
# 1-log(mae+1)? or 1-mae/window?
def normalize_mae(
    val:float,
    window: int,
    ):
    """
    window: Size of window to consider on each side of the observed peak for finding the simulated peak.
    Assuming less than 1 hour lead lag is as good as perfect
    """
    
    if val < 1:
        norm = 1
    
    #best values approach 1
    else:
        norm  = 1 - val / window
    
    return norm

def normalize_mape(val:float):
    """__summary__"""
    norm = 1 - val
    return norm

def mae_peak_timing(
    peaks: dict,
    window: int,
) -> list:
    """
    Calculates the Mean Absolute Error (MAE) of peak timing errors for the given gauges.

    Args:
        sim (dict): Dictionary containing simulated peak times for each gauge.
        obs (dict): Dictionary containing observed peak times for each gauge.
        window (int): Size of window to consider on each side of the observed peak for finding the simulated peak.
        gauges (tuple or list): Tuple or list of gauges (wflow_id) for which the MAE needs to be calculated.
        gid (str): Identifier for the gauge ID in the dataset (not used in this function, but kept for consistency).

    Returns:
        List: List of MAE of peak timing errors for each gauge (wflow_id).
    """
    
    res = []
    
    for g in peaks.keys():
        obs_g = np.array(peaks[g]['t_obs'])
        sim_g = np.array(peaks[g]['t_sim'])
        
        # Ensure observed and simulated peaks have the same length
        if len(sim_g) != len(obs_g):
            raise ValueError(f"Length of observed and simulated peaks should be equal for gauge {g}")

        # Calculate timing errors as the difference between observed and simulated peaks
        timing_errors = [(sim_time-obs_time) / np.timedelta64(1,'h') for obs_time, sim_time in zip(obs_g, sim_g)]
        
        # Filter out timing errors that are exactly equal to the window size (as per your original intention)
        timing_errors = np.array([error for error in timing_errors if error != window])
        
        # Compute MAE of timing errors
        mae_peak_timing = np.mean(np.abs(timing_errors))
        
        # Normalize MAE and store the result
        res.append(normalize_mae(round(float(mae_peak_timing), 4), window))
    
    return res

def mape_peak_magnitude(
    peaks: dict,
    window: int,
) -> list:
    """
    Calculates the Mean Absolute Percentage Error (MAPE) of peak magnitudes for the given gauges.

    Args:
        peaks (dict): Dictionary containing both observed and simulated peak times and magnitudes for each gauge.
        window (int): Size of window to consider on each side of the observed peak for finding the simulated peak.

    Returns:
        List: List of MAPE of peak magnitudes for each gauge (wflow_id).
    """
    
    res = []
    
    for g in peaks.keys():
        obs_magnitudes = np.array(peaks[g]['Q_obs'])
        sim_magnitudes = np.array(peaks[g]['Q_sim'])

        # Ensure observed and simulated magnitudes have the same length
        if len(sim_magnitudes) != len(obs_magnitudes):
            raise ValueError(f"Length of observed and simulated magnitudes should be equal for gauge {g}")
        
        # Compute MAPE of peak magnitudes
        mape_peak_magnitude = np.mean(np.abs((sim_magnitudes - obs_magnitudes) / obs_magnitudes))
        
        # Store the result
        res.append(normalize_mape(round(float(mape_peak_magnitude), 4)))
    
    return res


    
    
def _rld(
    data: np.ndarray,
    thres: float = 95,
    buildup: bool = False,
    flow_based: bool = False,
):

    """
    Calculate the Rising Limb Density (RLD) metric.

    Args:
        data (np.ndarray): The input data array.
        thres (float, optional): The threshold percentile for filtering smaller events. Defaults to 95.
        buildup (bool, optional): Whether to group pre-events with larger events. Defaults to False.
        flow_based (bool, optional): Whether to calculate RLD based on flow. Defaults to False.

    Returns:
        float: The calculated RLD value.

    Raises:
        ValueError: If the input data is empty.

    """
    
    minx = find_peaks(data * -1)[0]
    maxx = find_peaks(data)[0]

    if len(maxx) == 0 or len(minx) == 0:
        return np.nan

    if maxx[0] < minx[0]: 
        maxx = maxx[1:]
    if len(maxx) == 0:
        return np.nan
    
    if maxx[-1] < minx[-1]:
        minx = minx[:-1]
    if len(minx) == 0:
        return np.nan
    
    # TODO look at this later on
    # Need to figure out how bad data can be 
    # and the influence on peak finding
    shape = (len(minx), len(maxx))
    shape_old = (np.nan,np.nan)
    while True:
        if shape == shape_old and shape[0] == shape[1]:
            break
        shape_old = shape
        if shape[0] > shape[1]:
            minx = fix_minmax(minx, maxx, shape)
        elif shape[1] > shape[0]:
            maxx = fix_maxmin(maxx, minx, shape)
        else:
            minx = fix_minmax(minx, maxx, shape)
            maxx = fix_maxmin(maxx, minx, shape)

        shape = (len(minx), len(maxx))
        if 0 in shape:
            return np.nan

    # Filter all smaller events
    temp = data[maxx] - data[minx]
    red = np.where(temp < np.percentile(temp, thres))[0]
    maxx = np.delete(maxx, red)
    minx = np.delete(minx, red)

    # Group pre-events with the larger events
    if buildup:
        idx = 0
        while idx < len(maxx):
            if idx == 0:
                idx += 1
                continue
            betw = data[maxx[idx - 1]] - data[minx[idx]]
            prev = data[maxx[idx - 1]] - data[minx[idx - 1]]

            q_prev = data[maxx[idx-1]] - data[minx[idx-1]]
            q_next = data[maxx[idx]] - data[minx[idx-1]]

            if (2 * betw < prev) and (q_next > 2.5 * q_prev):
                if sum(np.isnan(data[minx[idx-1]: minx[idx]])) > 0:
                    pass
                else:
                    maxx = np.delete(maxx, idx - 1)
                    minx = np.delete(minx, idx)
                    idx -= 1
            idx += 1

    res = maxx - minx

    if flow_based:
        q_diff = data[maxx] - data[minx]
        res = q_diff / (maxx - minx)
        return np.mean(res)
    
    return len(res) / sum(res)


def rld(
    sim: xr.Dataset,
    obs: xr.Dataset,
    gauges: tuple | list,
    gid: str,
):
    """_summary_"""
    res = []

    for g in gauges:
        sim_e = _rld(
            sim.sel({gid:g}).Q,
        )
        obs_e = _rld(
            obs.sel(wflow_id=g).Q.values,
        )
        if np.isnan(sim_e) or np.isnan(obs_e):
            e = 1
        else:
            e = 1 - abs(1 - (sim_e / obs_e))
        res.append(round(e, 4))

    return res


def _peakdis(
    data: tuple | list,
    upper: float | int = 90,
    lower: float | int = 50,
):
    """_summary_"""

    peaks = (
        np.nanpercentile(data, upper) - np.nanpercentile(data, lower)
    )
    e = peaks / (0.9 - 0.5)

    return e


def peakdis(
    sim: xr.Dataset,
    obs: xr.Dataset,
    gauges: tuple | list,  
    gid: str,  
):
    """
    Calculate the peak discharge discrepancy between model (sim) and observed (obs) datasets for a given set of gauges.

    Parameters:
    sim (xr.Dataset): Model dataset containing discharge values.
    obs (xr.Dataset): Observed dataset containing discharge values.
    gauges (tuple | list): Tuple or list of gauge wflow_id for which the peak discharge discrepancy needs to be calculated.

    Returns:
    list: List of peak discharge discrepancies for each gauge.

    """
    res = []
    
    for g in gauges:
        sim_val = _peakdis(sim.sel({gid:g}).Q.values)
        obs_val = _peakdis(obs.sel(wflow_id=g).Q.values)

        if np.isnan(sim_val) or np.isnan(obs_val):
            r = 1
        else:
            r = 1 - abs(1 - (sim_val/ obs_val))
        res.append(r)

    return res


def fix_minmax(
    minx: tuple | list,
    maxx: tuple | list,
    shape: tuple | list,    
):
    """
    Adjusts the minx array to match the size of the maxx array based on the given shape.

    Args:
        minx (tuple | list): The minx array.
        maxx (tuple | list): The maxx array.
        shape (tuple | list): The desired shape of the minx array.

    Returns:
        tuple | list: The adjusted minx array.

    Raises:
        None

    """
    while True:
        size_diff = len(minx)-len(maxx)
        temp = maxx - minx[size_diff:]
        mask = temp < 0
        loc_rv = np.argmax(np.flip(mask))
        if loc_rv == 0:
            break
        
        loc = len(temp) - loc_rv - 1 + size_diff
        minx = np.delete(minx, loc)

        if shape[0] == shape[1]:
            return minx
        
        shape = (len(minx), len(maxx))
    return minx


def fix_maxmin(
    maxx: tuple | list,
    minx: tuple | list,
    shape: tuple | list,
):
    """
    Adjusts the `maxx` array by removing elements based on the `minx` array.
    
    Args:
        maxx (tuple | list): The maximum values array.
        minx (tuple | list): The minimum values array.
        shape (tuple | list): The shape of the arrays.
        
    Returns:
        tuple | list: The adjusted `maxx` array.
    """
    while True:
        temp = maxx[0:len(minx)] - minx
        loc = np.argmax(temp < 0)

        if loc == 0:
            break

        maxx = np.delete(maxx, loc)

        if shape[0] == shape[1]:
            return maxx
        
        shape = (len(minx), len(maxx))
    return maxx


def fix_gap():
    pass
     

def weighted_euclidean(
    coef: tuple | list,
    weights: tuple | list,
    weighted = True,
):
    """_summary_"""

    if weighted and len(weights) != len(coef):
        raise ValueError("The length of weights should be equal to the length of coef")
    
    if weighted and sum(weights) != 1:
        raise ValueError("The sum of weights should be equal to 1")
    
    if not weighted:
        weights = [1] * len(coef)

    dist = [
        w * (1-item)**2 for item, w in zip(coef, weights) 
    ]

    res = np.sqrt(sum(dist))

    return np.array(res.round(4))
