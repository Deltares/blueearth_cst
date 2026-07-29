"""Source-grid climate figures from the shared climate store (R07 B4 / P4).

Rule ``plot_climate_source``'s script (``Snakefile_model_creation`` 1.15 — a
single declaration; none of B1's two-DAG machinery applies). It answers *"what
does the source climate look like?"* from the store alone, so its whole
subgraph is the B1 producer (whose sole input is the tracked data catalog) plus
itself: the three figures build with **neither** ``hydrology_model/`` **nor**
``config/templates/wflow_build_model.yml`` on disk. That is the P4 assertion,
pinned by ``tests/test_plot_climate_source.py``.

Three climate-figure families now coexist (design § B4); this module owns the
first, and its filenames are prefixed ``source_`` deliberately — a ``pet.png``
copied into a report or picked up by a GUI collector loses its parent directory,
and the ``hydrology_model/forcing/plots/pet.png`` it would collide with is a
**deliberately different** value:

===============================  ======  ==================================
Product                          Grid    Home
===============================  ======  ==================================
source climate (this module)     source  ``climate_historical/<key>/plots/``
model-parity climate (rule 1.11) model   ``hydrology_model/evaluation/plots/``
forcing / model-input QA (1.13)  model   ``hydrology_model/forcing/plots/``
===============================  ======  ==================================

**Source-grid PET need not match the build's PET, by design.** These are
approximate quick assessments computed on the extraction grid against the
*source* orography; the build's PET is the refined model input, derived on the
model grid. The figures say so on their face.

Plain matplotlib only: no cartopy basemap tiles, so the rule needs no network.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from blueearth_cst.shared.climate_parity import model_parity_climate
from blueearth_cst.shared.snake_utils import log_row, save_figure

#: Variables the parity/PET machinery needs off the extraction.
PARITY_VARS = ("precip", "temp", "press_msl", "kin", "kout")

#: Rendered on every figure, so the caveat survives the file being copied out
#: of its directory (design risk-9).
_CAVEAT = (
    "Approximate quick assessment on the source extraction grid "
    "(source orography); not the model's forcing."
)
_PET_CAVEAT = (
    "Source-grid PET: differs from the model's PET input by design — that one "
    "is derived on the model grid from the model DEM."
)

#: One spec per figure: (variable, stem, label, unit, how to aggregate in time).
#: "sum" reports a climatological annual total; "mean" a climatological mean.
_FIGURES = (
    ("precip", "source_precip", "precipitation", "mm y$^{-1}$", "sum"),
    ("temp", "source_temp", "air temperature", "$\\degree$C", "mean"),
    ("pet", "source_pet", "potential evaporation", "mm y$^{-1}$", "sum"),
)


def _space_dims(da: xr.DataArray):
    """The non-time dimensions of ``da`` (the spatial ones, on this grid)."""
    return [d for d in da.dims if d != "time"]


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
    if oro_nc is not None:
        dem = xr.open_dataarray(oro_nc)
    else:
        if data_sources is None:
            raise ValueError(
                "load_source_orography: the era5 branch needs a data catalog "
                "(rule 1.15 params.data_sources) to resolve era5_orography"
            )
        import hydromt

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


def _plot_variable(da: xr.DataArray, out_path, label: str, unit: str, how: str, extra=None):
    """Render one figure: climatological map + monthly climatology."""
    space = _space_dims(da)
    yearly = da.resample(time="YE")
    field = (yearly.sum("time") if how == "sum" else yearly.mean("time")).mean("time")
    if how == "sum":
        field = field.where(field > 0)
    field = field.compute()

    domain = da.mean(dim=space)
    monthly = domain.resample(time="ME")
    monthly = monthly.sum("time") if how == "sum" else monthly.mean("time")
    monthly = monthly.groupby("time.month").mean("time").compute()

    fig, (ax_map, ax_clim) = plt.subplots(1, 2, figsize=(11, 4.2))

    field.attrs.update(long_name=label, units=unit)
    field.plot(ax=ax_map, cbar_kwargs=dict(aspect=30, shrink=0.85, label=f"{label} [{unit}]"))
    ax_map.set_title("climatological mean")
    ax_map.set_xlabel("longitude [degree east]")
    ax_map.set_ylabel("latitude [degree north]")

    months = np.arange(1, 13)
    values = monthly.reindex(month=months).values
    if how == "sum":
        ax_clim.bar(months, values, color="steelblue")
        ax_clim.set_ylabel(f"{label} [mm month$^{{-1}}$]")
    else:
        ax_clim.plot(months, values, color="firebrick", marker="o", lw=0.9, ms=3)
        ax_clim.set_ylabel(f"{label} [{unit}]")
    ax_clim.set_xticks(months)
    ax_clim.set_xlabel("month")
    ax_clim.set_title("monthly climatology, domain mean")
    ax_clim.grid(alpha=0.3)

    caveat = _CAVEAT if extra is None else f"{_CAVEAT}\n{extra}"
    fig.text(0.01, 0.01, caveat, fontsize=6.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(out_path, dpi=300)
    plt.close(fig)


def plot_climate_source(
    climate_nc: Union[str, Path],
    plot_dir: Union[str, Path],
    oro_nc: Optional[Union[str, Path]] = None,
    data_sources: Optional[Union[str, Path]] = None,
    clim_source: str = "era5",
):
    """Write ``source_{precip,temp,pet}.png`` from the shared climate store.

    Parameters
    ----------
    climate_nc : str | Path
        ``climate_historical/<key>/extract_historical.nc`` — the store's
        extraction (rule 1.10 ``extract_climate_grid``).
    plot_dir : str | Path
        ``climate_historical/<key>/plots/``. Created if absent.
    oro_nc : str | Path, optional
        chirps / chirps_global only: the store's declared ``orography.nc``
        sidecar. None on era5, where the orography comes from the catalog.
    data_sources : str | Path, optional
        hydromt data catalog(s). Required on the era5 branch (``era5_orography``).
    clim_source : str
        ``shared.clim_historical``; recorded in the log for traceability.

    Raises
    ------
    ValueError
        If the extraction is missing any variable the PET workflow needs. This
        is deliberately loud: the rule declares three outputs, so a silent skip
        would surface as an opaque ``MissingOutputException`` instead.
    """
    log_row(f"Reading climate store extraction ({clim_source}): {climate_nc}", module="plot")
    ds_raw = xr.open_dataset(climate_nc)
    missing = [v for v in PARITY_VARS if v not in ds_raw]
    if missing:
        raise ValueError(
            f"plot_climate_source: {climate_nc} is missing {missing}; the "
            f"source-grid PET workflow needs {list(PARITY_VARS)}"
        )

    dem_source = load_source_orography(ds_raw, oro_nc=oro_nc, data_sources=data_sources)
    # setup_time_horizon.py maps every source supported on this path
    # (era5/chirps/chirps_global) to debruin; eobs is rejected at DAG-parse time.
    ds_src = source_grid_climate(ds_raw, dem_source, pet_method="debruin")

    plot_dir = Path(plot_dir)
    for var, stem, label, unit, how in _FIGURES:
        log_row(f"Plot source-grid {label}", module="plot")
        _plot_variable(
            ds_src[var],
            plot_dir / f"{stem}.png",
            label=label,
            unit=unit,
            how=how,
            extra=_PET_CAVEAT if var == "pet" else None,
        )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_climate_source(
                climate_nc=sm.input.climate_nc,
                plot_dir=sm.params.plot_dir,
                # declared only on the chirps/chirps_global branch, mirroring
                # rule 1.11's input split
                oro_nc=getattr(sm.input, "oro_nc", None),
                data_sources=sm.params.data_sources,
                clim_source=sm.params.clim_source,
            )
