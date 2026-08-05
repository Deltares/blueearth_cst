"""Model-parity transform of raw gridded climate (P3-2a design §5.2).

Reproduces, on the shared historical-climate store, exactly the regrid +
corrections + PET
the forcing build applies (``setup_precip_forcing`` / ``setup_temp_pet_forcing``
under our config: ``temp_correction=True``, ``press_correction=True``,
``pet_method`` from the setup_time_horizon mapping), via the same public
``hydromt.model.processes.meteo`` functions the build delegates to
(wflow_sbm.py:3310, :3667, :3707, :3732). Plain xarray in and out; never a
model object (criterion C1) — the callers load ``staticmaps.nc`` variables and
persist results. The corrections-off variant backs the commit-5 QA ladder's
regrid-only state S2 (design §5.5).

Lives in ``shared/`` because it is genuinely engine-neutral: two callers now
use it from opposite sides — ``model/plot_results.py`` at model parity, and
``climate_analysis/plot_climate_source.py`` on the source grid, where passing
the SAME dem for ``dem_model`` and ``dem_forcing`` makes both model-specific
steps degenerate. It sat under ``model/`` until R7-1's follow-up (R7-4); the
import direction was backwards for the one module whose headline claim is
model-independence.

The build's final ``where(mask)`` (basins mask) is deliberately not
reproduced: the downstream per-subcatchment groupby restricts to exactly those
cells (design §5.2).
"""

from typing import Optional

import pandas as pd
import xarray as xr
from hydromt.model.processes import meteo

#: The build's forcing timestep and time-resampling kwargs
#: (setup_config time.timestepsecs=86400; wflow_sbm.py:3315,:3717,:3737).
_FREQ = pd.to_timedelta(86400, unit="s")
_RESAMPLE_KWARGS = dict(label="right", closed="right")


def model_parity_climate(
    ds_raw: xr.Dataset,
    dem_model: xr.DataArray,
    dem_forcing: Optional[xr.DataArray],
    pet_method: str = "debruin",
    *,
    corrections: bool = True,
) -> xr.Dataset:
    """Regrid + correct + derive PET on raw climate, exactly as the build does.

    Parameters
    ----------
    ds_raw : xr.Dataset
        Extraction from the shared store
        (``data/climate/historical/<key>/extract_historical.nc``): ``precip``,
        ``temp``, ``kin``, ``press_msl`` (+ ``kout`` for debruin), daily,
        on the extraction grid.
    dem_model : xr.DataArray
        The model DEM (``staticmaps.nc["land_elevation"]`` — the build's
        ``dem_model``, hydromt_wflow naming.py "elevtn").
    dem_forcing : xr.DataArray, optional
        The forcing-grid DEM the temperature lapse correction references. On
        era5: the ``era5_orography`` catalog source fetched as the build does.
        On chirps/chirps_global: MANDATORILY the extraction's orography
        sidecar (the MERIT-on-chirps-grid DEM) — passing ``era5_orography``
        there leaves the extraction's embedded correction unreversed and
        double-corrects (design ext1-1 ledger).
    pet_method : str
        ``"debruin"`` for the P3-2a-supported sources (era5/chirps/
        chirps_global map to debruin per the setup_time_horizon mapping).
    corrections : bool, keyword-only
        True (default): the production parity transform (ladder state S3).
        False: the regrid-only ladder state S2 — ``lapse_correction=False``,
        ``press_correction=False`` (``press_msl`` renamed uncorrected,
        meteo.py:404-406), same methods and resampling (design §5.5).

    Returns
    -------
    xr.Dataset
        Lazy ``precip``, ``temp``, ``pet`` on the model grid, daily.
    """
    # The build casts inputs float32 before the meteo calls (wflow_sbm.py:3295,
    # :3654-3655, :3665); mirror that so parity holds bit-for-bit in dtype.
    precip_out = meteo.precip(
        precip=ds_raw["precip"].astype("float32"),
        da_like=dem_model,
        clim=None,  # no precip_clim_fn in our build config
        freq=_FREQ,
        reproj_method="nearest_index",
        resample_kwargs=dict(_RESAMPLE_KWARGS),
    )

    if dem_forcing is not None:
        dem_forcing = dem_forcing.astype("float32")
    temp_in = meteo.temp(
        ds_raw["temp"].astype("float32"),
        dem_model=dem_model,
        dem_forcing=dem_forcing,
        lapse_correction=corrections,
        freq=None,  # resample time after the pet workflow, as the build does
        reproj_method="nearest_index",
    )

    pet_vars = ["press_msl", "kin", "kout"] if pet_method == "debruin" else ["press_msl", "kin"]
    pet_out = meteo.pet(
        ds_raw[pet_vars].astype("float32"),
        temp=temp_in,
        dem_model=dem_model,
        method=pet_method,
        press_correction=corrections,
        wind_correction=True,
        wind_altitude=10,
        reproj_method="nearest_index",
        freq=_FREQ,
        resample_kwargs=dict(_RESAMPLE_KWARGS),
    )

    temp_out = meteo.resample_time(
        temp_in,
        _FREQ,
        upsampling="bfill",  # we assume right labeled original data
        downsampling="mean",
        label="right",
        closed="right",
        conserve_mass=False,
    )

    return xr.Dataset({"precip": precip_out, "temp": temp_out, "pet": pet_out})
