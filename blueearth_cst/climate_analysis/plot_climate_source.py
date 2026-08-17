"""Source-grid climate figures from the shared climate store (R07 B4 / P4).

Rule ``plot_climate_source``'s script (``build_model.smk`` 1.15 — a
single declaration; none of B1's two-DAG machinery applies). It answers *"what
does the source climate look like?"* from the store alone, so its whole
subgraph is the B1 producer (whose sole input is the tracked data catalog) plus
itself: the three figures build with **neither** ``models/hydrology/wflow/``
**nor**
``config/defaults/wflow_build_model.yml`` on disk. That is the P4 assertion,
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

from blueearth_cst.climate_analysis.climate_figures import (
    load_spatial_overlays,
    plot_climate_figures,
    source_climate_vars,
)
from blueearth_cst.climate_analysis.climate_levels import read_climate_levels
from blueearth_cst.climate_analysis.figure_naming import subbasin_scope
from blueearth_cst.shared.climate_parity import model_parity_climate
from blueearth_cst.shared.grid_cells import cells_csv_mask, subbasin_masks
from blueearth_cst.shared.snake_utils import (
    DEFAULT_WATER_YEAR_ANCHOR,
    log_row,
    water_year_end_anchor,
)

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


def area_masks_for(
    ds: xr.Dataset,
    basin_cells: Optional[Union[str, Path]] = None,
    subbasins=None,
) -> dict:
    """``{spatial_scope: mask}`` for every area the series figures are drawn over.

    The basin comes from the store's own ``basin_cells.csv`` and the subbasins
    from the shared vector foundation, both through
    :mod:`blueearth_cst.shared.grid_cells` — so one predicate decides every
    domain, and rule 0.06's comparison of the same source lands on the same
    number as this rule's own figure.

    **The basin scope is always present, even with nothing to build it from**:
    it then maps to ``None``, which draws the full extraction. Dropping it
    instead would leave a source with no annual figure at all, and the rule
    declares one.
    """
    masks = {"basin_avg": cells_csv_mask(ds, basin_cells)}
    for subbasin_id, mask in subbasin_masks(ds, subbasins).items():
        masks[subbasin_scope(subbasin_id)] = mask
    return masks


def plot_climate_source(
    climate_nc: Union[str, Path],
    plot_dir: Union[str, Path],
    oro_nc: Optional[Union[str, Path]] = None,
    data_sources: Optional[Union[str, Path]] = None,
    clim_source: str = "era5",
    geoms_dir: Optional[Union[str, Path]] = None,
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    levels_json: Optional[Union[str, Path]] = None,
    basin_cells: Optional[Union[str, Path]] = None,
    subbasin_dir: Optional[Union[str, Path]] = None,
):
    """Write the canonical climate figure set from the shared climate store.

    Derives ``precip``/``temp``/``pet`` on the extraction grid, then hands them
    to ``climate_figures.plot_climate_figures`` as dataset ``source``. The file
    names are that module's to define (``climate_figures.figure_names``).

    **A precipitation-only source draws precipitation only** (owner ruling
    2026-08-16). Its store carries temperature, radiation and pressure, but the
    extraction borrowed them from era5 -- a model cannot be forced without them
    -- so drawing them here would put era5's values under this source's name, in
    the workflow whose whole job is telling sources apart. The set comes from
    ``climate_figures.source_climate_vars``; on that branch the PET derivation
    and the orography read are skipped entirely rather than computed and
    discarded, since both exist only to serve the temperature and PET figures.

    Parameters
    ----------
    climate_nc : str | Path
        ``data/climate/historical/<key>/extract_historical.nc`` — the store's
        extraction (rule 0.04 / 1.04 / 3.08 ``extract_historical_climate``).
    plot_dir : str | Path
        ``data/climate/historical/<key>/plots/``. Created if absent.
    oro_nc : str | Path, optional
        chirps / chirps_global only: the store's declared ``orography.nc``
        sidecar. None on era5, where the orography comes from the catalog.
    data_sources : str | Path, optional
        hydromt data catalog(s). Required on the era5 branch (``era5_orography``).
    clim_source : str
        ``shared.clim_historical``; recorded in the log for traceability.
    geoms_dir : str | Path, optional
        ``data/spatial/geoms/`` — the ENGINE-NEUTRAL vector foundation from rule
        1.03. Supplies the basin outline, subcatchment divides, river network
        and points of interest the maps are drawn over. Optional, and
        model-independent by construction: this rule runs off the climate store
        and must not wait on a wflow build to plot the source climate.

    Raises
    ------
    ValueError
        If the extraction is missing any variable the PET workflow needs. This
        is deliberately loud: the rule declares three outputs, so a silent skip
        would surface as an opaque ``MissingOutputException`` instead.
    """
    log_row(f"Reading store ({clim_source}): {climate_nc}", module="plot")
    ds_raw = xr.open_dataset(climate_nc)
    variables = source_climate_vars(clim_source)

    if variables == ("precip",):
        # No PET derivation and no orography read: both exist only to produce
        # the temperature and PET figures, which this source does not get.
        # Checked against what is DRAWN rather than against PARITY_VARS -- the
        # store does carry the era5 companions, but needing them here would be
        # a claim this branch no longer makes.
        missing = [v for v in variables if v not in ds_raw]
        if missing:
            raise ValueError(
                f"plot_climate_source: {climate_nc} is missing {missing}; "
                f"{clim_source} is precipitation-only and needs {list(variables)}"
            )
        # The parenthetical explaining WHOSE temperature the store carries was
        # the larger half of a 180-character row that wrapped to two console
        # lines. It is a property of the store, not news about this run, and it
        # is stated on the figures themselves (`_CAVEAT`) where a reader who
        # needs it actually is.
        log_row(
            f"{clim_source} is precipitation-only: precip figures only, "
            "no temperature or PET",
            module="plot",
        )
        ds_src, caveat = ds_raw, _CAVEAT
    else:
        missing = [v for v in PARITY_VARS if v not in ds_raw]
        if missing:
            raise ValueError(
                f"plot_climate_source: {climate_nc} is missing {missing}; the "
                f"source-grid PET workflow needs {list(PARITY_VARS)}"
            )

        dem_source = load_source_orography(
            ds_raw, oro_nc=oro_nc, data_sources=data_sources
        )
        # setup_time_horizon.py maps every source supported on this path
        # (era5/chirps/chirps_global) to debruin; eobs is rejected at DAG-parse
        # time.
        ds_src = source_grid_climate(ds_raw, dem_source, pet_method="debruin")
        # The PET caveat rides along on every figure rather than only on the pet
        # ones -- one caveat block per figure, and a reader comparing precip
        # across the two directories should also know the PET on this side is
        # source-grid. It is dropped above, where no PET figure is drawn.
        caveat = f"{_CAVEAT}\n{_PET_CAVEAT}"

    overlays = load_spatial_overlays(geoms_dir)
    # The areas the series figures reduce to. The MAP ignores them -- it draws
    # the field, and cropping it per subbasin would be one raster shown N ways.
    masks = area_masks_for(ds_src, basin_cells, overlays.get("subbasins"))
    subbasins_drawn = [s for s in masks if s.startswith("subbasin_")]
    log_row(
        f"Aggregating over {len(masks)} area(s): basin"
        + (f" + {len(subbasins_drawn)} subbasin(s)" if subbasins_drawn else "")
        + (
            ""
            if masks.get("basin_avg") is not None
            else " (no basin cell mask available -- the full extraction is used)"
        ),
        module="plot",
    )

    # The canonical set (climate_figures) draws these; this module's job ends at
    # producing the dataset. `variables` MUST be the same list the rule declared
    # its outputs from, or the job ends in MissingOutputException.
    return plot_climate_figures(
        ds_src,
        plot_dir,
        "source",
        caveat=caveat,
        overlays=overlays,
        variables=variables,
        # Absent for a single-source run (WF1), where there is nothing to share
        # a scale WITH -- each figure then classifies from its own data.
        levels=read_climate_levels(levels_json),
        # Present ⇒ the WF0 filename grammar, with the source id as the
        # dataset_scope (`era5_precip_annual_ts_basin_avg.png`).
        clim_source=clim_source,
        area_masks=masks,
        subbasin_dir=subbasin_dir,
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
                geoms_dir=sm.params.geoms_dir,
                anchor=water_year_end_anchor(sm.params.water_year_start),
                # Declared by WF0's multi-source path only; absent in WF1.
                levels_json=getattr(sm.input, "levels_json", None),
                # Rule 0.04's own output: the cells the basin touches, which is
                # what the basin-average figures reduce over -- and the same
                # file weathergenr averages over, so the two agree.
                basin_cells=getattr(sm.input, "basin_cells", None),
                subbasin_dir=getattr(sm.params, "subbasin_plot_dir", None),
            )
