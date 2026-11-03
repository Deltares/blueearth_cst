"""Prepare native source-map fields with the existing WF0 scientific methods.

The caller owns dependency resolution, identity sealing and publication. No retained
field file is reused here, and source roles do not remove extracted auxiliary arrays.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr

from blueearth_cst.climate_analysis import diagnostics as dg
from blueearth_cst.climate_analysis.diagnostic_maps import annual_fields
from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.diagnostic_tables import daily_series


@dataclass(frozen=True)
class SourceMapSpec:
    """Declared source inputs; catalog DEM files are dependencies, not sidecars.

    ``orography_nc`` is the extraction's optional already prepared sidecar.
    ``orography_inputs`` identifies concrete files behind catalog source PET;
    HydroMT still loads those through the ordered catalogs using its own adapter.
    """

    source: str
    climate_nc: Path
    basin_cells: Path
    catalogs: tuple[Path, ...]
    requested_window: tuple[str, str]
    orography_nc: Path | None = None
    orography_inputs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PreparedSourceMapFields:
    """Eager fields, existing basin-unit lineage and verified preparation identity."""

    fields: xr.Dataset
    lineage: dict[str, Any]
    preparation_identity: dict[str, Any]


def prepare_source_map_fields(
    source_spec: SourceMapSpec,
    m0: int,
    completeness: Mapping[str, int],
) -> PreparedSourceMapFields:
    """Retain raw precipitation/temperature and derive source-grid PET as WF0 does.

    Genuine temperature uses the existing basin-mean unit/value gate. Its failure
    suppresses temperature and PET after the same PET derivation as the old caller.
    Precipitation-only sources retain any raw auxiliary arrays without becoming
    temperature/PET contributors. Completeness uses ``annual_fields`` unchanged.
    """
    from blueearth_cst.shared.map_preparation import preparation_identity

    identity = preparation_identity(source_spec, m0, completeness)
    daily, lineage = daily_series(
        source_spec.climate_nc,
        source_spec.basin_cells,
        source_spec.source,
        source_spec.catalogs,
    )
    reporting_years = dg.reporting_year(daily.index, m0)
    with xr.open_dataset(source_spec.climate_nc) as raw:
        ds = raw
        if carries_temp(source_spec.source):
            from blueearth_cst.climate_analysis.plot_climate_source import (
                load_source_orography,
                source_grid_climate,
            )

            dem = load_source_orography(
                raw,
                oro_nc=source_spec.orography_nc,
                data_sources=source_spec.catalogs,
            )
            derived = source_grid_climate(raw, dem)
            ds = raw[["precip", "temp"]].assign(pet=derived.pet)
            if not daily.temp.notna().any():
                ds = ds.assign(
                    temp=raw.temp * float("nan"), pet=derived.pet * float("nan")
                )
        fields = annual_fields(ds, m0, completeness)
        fields.attrs.update(
            reporting_year_label="start",
            period_start=int(reporting_years.min()),
            period_end=int(reporting_years.max()),
        )
        fields.load()
    if preparation_identity(source_spec, m0, completeness) != identity:
        raise ValueError(
            f"{source_spec.source}: source inputs changed during map preparation"
        )
    return PreparedSourceMapFields(fields, lineage, identity)
