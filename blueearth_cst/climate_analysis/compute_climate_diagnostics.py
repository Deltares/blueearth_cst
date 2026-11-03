"""Rule 0.04: source-owned retained numerical diagnostic tables."""

from pathlib import Path

import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis import diagnostics as dg
from blueearth_cst.climate_analysis.diagnostic_outputs import diagnostic_source_outputs
from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.diagnostic_tables import (
    compute,
    daily_series,
    write_tables,
)


def run(sm):
    settings = dict(sm.params.settings)
    source = sm.params.source
    m0 = int(sm.params.m0)
    paths = diagnostic_source_outputs(sm.params.store_dir, source, settings)
    daily, lineage = daily_series(
        sm.input.climate_nc, sm.input.basin_cells, source, sm.params.data_sources
    )
    ry = dg.reporting_year(daily.index, m0)
    period = (int(ry.min()), int(ry.max()))
    _, tables = compute(
        daily,
        settings,
        m0,
        period,
        temperature_source=source if carries_temp(source) else None,
    )
    Path(paths["compute"]["daily"]).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(paths["compute"]["daily"])
    if settings["subbasin_figures"]:
        import geopandas as gpd

        from blueearth_cst.shared.grid_cells import subbasin_masks

        with xr.open_dataset(sm.input.climate_nc) as ds:
            masks = subbasin_masks(ds, gpd.read_file(sm.input.subbasins))
            folder = Path(sm.output.subbasin_tables)
            folder.mkdir(parents=True, exist_ok=True)
            for id_, mask in masks.items():
                frame = pd.DataFrame(index=pd.DatetimeIndex(ds.time.values))
                for variable in ("precip", "temp"):
                    frame[variable] = (
                        ds[variable].where(mask).mean(["latitude", "longitude"]).values
                        if variable in ds
                        and (
                            variable == "precip"
                            or (carries_temp(source) and daily.temp.notna().any())
                        )
                        else float("nan")
                    )
                frame.index.name = "date"
                subfolder = folder / f"subbasin_{id_}"
                subfolder.mkdir(parents=True, exist_ok=True)
                frame.to_csv(subfolder / "daily_basin.csv")
                _, subset = compute(
                    frame,
                    settings,
                    m0,
                    period,
                    temperature_source=source if carries_temp(source) else None,
                )
                for name in (
                    "precip_climatology",
                    "temp_climatology",
                    "annual_indices",
                    "trends",
                ):
                    subset[name].to_csv(subfolder / f"{name}.csv", index=False)
    write_tables(paths, {source: tables}, settings, m0, period, {source: lineage})
    from blueearth_cst.shared.snake_utils import log_row

    for warning in lineage["warnings"]:
        log_row(warning, module="diagnostics", level="WARNING")
    log_row(
        f"Wrote diagnostic tables for {source}",
        module="diagnostics",
    )


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
