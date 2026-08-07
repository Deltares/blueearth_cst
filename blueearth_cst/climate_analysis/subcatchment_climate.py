"""Aggregate gridded wflow forcing to per-subcatchment climate timeseries.

Kept dependency-light (xarray only) and separate from the plotting/model code so
the same aggregation can back a model-independent climate-analysis component
later — see the R6 functional-modularization note in ``dev/tasks/`` and
ADR 0002 (``dev/decisions/0002-revive-subcatchment-climate-plots.md``).
"""

# wflow forcing variable -> the key ``plot_clim`` (func_plot_signature) expects.
FORCING_TO_CLIM = {
    "precip": "P_subcatchment",
    "temp": "T_subcatchment",
    "pet": "EP_subcatchment",
}


def climate_forcing_by_subcatchment(forcing, subcatchment):
    """Spatial-mean the wflow forcing per subcatchment -> ``(index, time)`` dataset.

    The climate variables plotted by ``plot_clim`` come from the model's climate
    **input** (the ERA5-derived ``inmaps`` forcing: ``precip``/``temp``/``pet``),
    not from wflow outputs (ADR 0002). Each variable is reduced to a per-
    subcatchment spatial mean timeseries so it can be plotted per station.

    Parameters
    ----------
    forcing : xr.Dataset
        wflow forcing (``inmaps``) with variables ``precip``, ``temp``, ``pet``
        on the model grid (a spatial ``(latitude, longitude)`` pair plus
        ``time``).
    subcatchment : xr.DataArray
        wflow ``subcatchment`` staticmap on the same grid; each cell holds its
        integer subcatchment id, with a nodata value (``_FillValue``, 0 in the
        current build) outside the basin.

    Returns
    -------
    xr.Dataset
        Variables ``P_subcatchment`` / ``T_subcatchment`` / ``EP_subcatchment``
        with dims ``(index, time)``; ``index`` is the integer subcatchment id,
        matching the discharge-station index. Consumed directly by ``plot_clim``.

    Notes
    -----
    The spatial reduction is an **unweighted** cell mean (matching the
    ``plot_forcing`` precedent). Fine for small equatorial basins; for
    higher-latitude or large basins, cell-area weighting would be more accurate.
    """
    nodata = subcatchment.attrs.get("_FillValue", 0)
    # nodata cells become NaN; xarray groupby skips NaN group labels, so only
    # real subcatchments form groups.
    groups = subcatchment.where(subcatchment != nodata)
    ds = forcing[list(FORCING_TO_CLIM)]
    # groupby a 2D (lat, lon) label array reduces the spatial dims, keeping time.
    grouped = ds.groupby(groups.rename("index")).mean()
    # where(...) promoted the ids to float; restore integer ids for clean labels.
    grouped = grouped.assign_coords(index=grouped["index"].astype("int64"))
    return grouped.rename(FORCING_TO_CLIM)
