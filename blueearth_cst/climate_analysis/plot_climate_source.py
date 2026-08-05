"""Source-grid climate figures from the shared climate store (R07 B4 / P4).

Rule ``plot_climate_source``'s script (``Snakefile_model_creation`` 1.15 — a
single declaration; none of B1's two-DAG machinery applies). It answers *"what
does the source climate look like?"* from the store alone, so its whole
subgraph is the B1 producer (whose sole input is the tracked data catalog) plus
itself: the three figures build with **neither** ``models/hydrology/wflow/``
**nor**
``config/templates/wflow_build_model.yml`` on disk. That is the P4 assertion,
pinned by ``tests/test_plot_climate_source.py``.

Three climate-figure families coexist (design § B4). This module owns the first
and rule 1.13 owns the second, and since 2026-08 BOTH are drawn by the same
canonical set (``climate_figures``) so the two are directly comparable — every
filename is prefixed by its dataset, because a bare ``pet.png`` copied into a
report or picked up by a GUI collector loses its parent directory and the two
are **deliberately different** values:

================================  ======  ============================================
Product                           Grid    Home
================================  ======  ============================================
source climate (this module)      source  ``data/climate/historical/<key>/plots/``
forcing / model-input QA (1.13)   model   ``models/hydrology/wflow/forcing/plots/``
model-parity climate (rule 1.11)  model   ``models/hydrology/wflow/evaluation/plots/``
================================  ======  ============================================

The third stays outside the canonical set: it is keyed by STATION rather than by
grid and answers a different question (climate beside the discharge it drove).

**Source-grid PET need not match the build's PET, by design.** These are
approximate quick assessments computed on the extraction grid against the
*source* orography; the build's PET is the refined model input, derived on the
model grid. The figures say so on their face — which is exactly what the
canonical set makes readable, since the same figure now exists on both sides.

Plain matplotlib only: no cartopy basemap tiles, so the rule needs no network.
"""

from pathlib import Path
from typing import Optional, Union

import xarray as xr

from blueearth_cst.climate_analysis.climate_figures import plot_climate_figures
from blueearth_cst.shared.climate_parity import model_parity_climate
from blueearth_cst.shared.snake_utils import log_row

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


def plot_climate_source(
    climate_nc: Union[str, Path],
    plot_dir: Union[str, Path],
    oro_nc: Optional[Union[str, Path]] = None,
    data_sources: Optional[Union[str, Path]] = None,
    clim_source: str = "era5",
):
    """Write the canonical climate figure set from the shared climate store.

    Derives ``precip``/``temp``/``pet`` on the extraction grid, then hands them
    to ``climate_figures.plot_climate_figures`` as dataset ``source``. The file
    names are that module's to define (``climate_figures.figure_names``).

    Parameters
    ----------
    climate_nc : str | Path
        ``data/climate/historical/<key>/extract_historical.nc`` — the store's
        extraction (rule 1.10 ``extract_climate_grid``).
    plot_dir : str | Path
        ``data/climate/historical/<key>/plots/``. Created if absent.
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

    # The canonical set (climate_figures) draws these; this module's job ends at
    # producing the dataset. The PET caveat rides along on every figure rather
    # than only on the pet ones -- one caveat block per figure, and a reader
    # comparing precip across the two directories should also know the PET on
    # this side is source-grid.
    return plot_climate_figures(
        ds_src,
        plot_dir,
        "source",
        caveat=f"{_CAVEAT}\n{_PET_CAVEAT}",
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
