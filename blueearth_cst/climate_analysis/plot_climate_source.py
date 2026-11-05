"""Source-grid orography and PET derivation for WF0 diagnostics."""

from pathlib import Path
from typing import Optional, Union

import xarray as xr

from blueearth_cst.shared.climate_parity import model_parity_climate


def _drop_nonspatial(dem: xr.DataArray) -> xr.DataArray:
    """Strip scalar leftovers (notably ``time``) from a DEM, keeping ``spatial_ref``.

    The shipped ``era5_orography`` source is a single-timestep field, so
    ``get_rasterdataset(...).squeeze()`` leaves a **scalar ``time``
    coordinate** behind. hydromt's ``reproject_like`` copies the reference
    grid's coordinates onto its result, so handing that DEM to
    ``meteo.precip``/``temp``/``pet`` as ``da_like``/``dem_model`` replaces the
    climate array's 7671-step time axis with that scalar — and the first
    ``resample_time`` then dies inside ``np.diff(da.time)`` with "diff requires
    input that is at least one dimensional". Found on the first real wf1 run;
    the synthetic fixture in ``tests/test_plot_climate_source.py`` now carries
    the same scalar coordinate so it cannot regress unnoticed.
    """
    extra = [c for c in dem.coords if c not in dem.dims and c != "spatial_ref"]
    return dem.drop_vars(extra) if extra else dem


def load_source_orography(
    ds_raw: xr.Dataset,
    oro_nc: Optional[Union[str, Path]] = None,
    data_sources: Optional[Union[str, Path]] = None,
) -> xr.DataArray:
    """Return the source-grid DEM, on ``ds_raw``'s own grid.

    Two branches, resolved by the caller from ``clim_historical`` — the same
    split ``extract_historical_climate.prep_historical_climate`` and
    ``plot_results.analyse_wflow_historical`` already make:

    * chirps / chirps_global — ``oro_nc`` is the store's declared ``orography.nc``
      sidecar (MERIT, already reprojected onto the extraction grid by the
      producer), received as a rule input rather than discovered as a sibling.
    * era5 — the store carries no sidecar, so ``era5_orography`` is fetched from
      the data catalog exactly as the forcing build fetches it.

    Either way the result is reprojected onto ``ds_raw``'s grid, so the figures
    are computed on the extraction's own cells and nothing is regridded, and
    every non-spatial coordinate is stripped first — see ``_drop_nonspatial``.
    """
    # Both branches end in ``.raster.reproject_like``, so the accessor is
    # needed here regardless of which one runs. Imported at the top of the
    # function rather than at module scope: hydromt is the heaviest dependency,
    # so it stays off the import path
    # of anything that merely imports the module (t2608202307).
    import hydromt  # noqa: F401 -- registers the xarray .raster accessor

    if oro_nc is not None:
        dem = xr.open_dataarray(oro_nc)
    else:
        if data_sources is None:
            raise ValueError(
                "load_source_orography: the era5 branch needs a data catalog "
                "(WF0 params.data_sources) to resolve era5_orography"
            )
        data_catalog = hydromt.DataCatalog(data_libs=data_sources)
        dem = data_catalog.get_rasterdataset(
            "era5_orography",
            geom=ds_raw.raster.box,  # clip with the extraction bbox for full coverage
            buffer=2,
            variables=["elevtn"],
        ).squeeze()
    return _drop_nonspatial(dem).raster.reproject_like(ds_raw, method="average")


def source_grid_climate(
    ds_raw: xr.Dataset,
    dem_source: xr.DataArray,
    pet_method: str = "debruin",
) -> xr.Dataset:
    """Derive ``precip`` / ``temp`` / ``pet`` on the extraction grid.

    Reuses the build's own PET machinery rather than inventing a second one:
    ``climate_parity.model_parity_climate`` wraps exactly the
    ``hydromt.model.processes.meteo`` calls the forcing build delegates to. It
    is called here with ``dem_model == dem_forcing == dem_source``, which makes
    the two model-specific steps degenerate:

    * the regrid targets ``dem_source``'s grid, i.e. the extraction's own grid,
      so it is the identity;
    * the temperature lapse correction shifts by ``dem_model - dem_forcing``,
      which is zero — correct, because the extraction's ``temp`` is already
      stated at ``dem_source``'s elevations on both branches (era5 temp is at
      era5 orography; the chirps branch's producer already lapse-corrected onto
      the sidecar DEM).

    What survives is the de Bruin PET workflow with the pressure correction
    referenced to the source elevations — source-grid PET. It is **not** the
    build's PET and is not required to equal it.
    """
    # Idempotent, and applied here as well as at fetch time: this is the single
    # funnel into the meteo machinery, and a stray scalar coordinate on the DEM
    # silently rewrites the climate array's time axis (see _drop_nonspatial).
    dem_source = _drop_nonspatial(dem_source)
    return model_parity_climate(
        ds_raw,
        dem_model=dem_source,
        dem_forcing=dem_source,
        pet_method=pet_method,
    )
