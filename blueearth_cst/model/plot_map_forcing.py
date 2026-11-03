"""Canonical WF0-style figures for Wflow model-grid forcing (rule 1.12).

Load the actual model forcing and basin mask independently of WF0 execution.
The reviewed WF0 artists and summary methods are shared; filenames retain the
``forcing_<variable>_<kind>.png`` contract. Maps use offline spatial overlays
and captions live separately beside the nine figures.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr

from blueearth_cst.model.forcing_figures import render_forcing_figures
from blueearth_cst.shared.map_reference import load_reference, reference_arguments
from blueearth_cst.shared.snake_utils import DEFAULT_WATER_YEAR_ANCHOR


def plot_forcing(
    wflow_root: Union[str, Path],
    plot_dir: Optional[Union[str, Path]] = None,
    gauges_fn: Optional[Union[str, Path]] = None,
    geoms_dir: Optional[Union[str, Path]] = None,
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    map_reference: Optional[Union[str, Path]] = None,
    map_reference_params: Optional[dict] = None,
    annotation: str = "none",
    reference_record: Optional[dict] = None,
):
    """Write the canonical climate figure set for the wflow forcing.

    Parameters
    ----------
    wflow_root : str | Path
        The wflow model root (``models/hydrology/wflow/``).
    plot_dir : str | Path, optional
        Destination. Defaults to ``<wflow_root>/plots``; rule 1.12 passes
        ``models/hydrology/wflow/forcing/plots`` so the figures sit beside the
        forcing they describe (R07 B10).
    gauges_fn : str | Path, optional
        Retained for the rule's signature; the map overlays no longer come from
        the model's own gauge layer (see ``geoms_dir``).
    geoms_dir : str | Path, optional
        ``data/spatial/geoms/`` — the ENGINE-NEUTRAL vector foundation from rule
        1.02, and the SAME layers the source-grid maps draw. Both climate map
        families read it so the two differ only in the raster underneath, which
        is what makes "what did downscaling change?" answerable by putting the
        two directories side by side.
    anchor : str, optional
        Year-end reporting anchor. The figures use WF0's start-year labels,
        full-year coverage and default tolerated gaps (3 days/month, 15/year).
    map_reference : str | Path
        Declared historical reference JSON, required before figure publication.
    map_reference_params : dict
        Executable params from the shared historical reference rule factory.
    reference_record : dict, optional
        Replaces the two arguments above for a re-render of retained products
        (``forcing_rerender``); the caller has checked its integrity.
    """
    if reference_record is None and (
        map_reference is None or map_reference_params is None
    ):
        raise ValueError(
            "Forcing figures require the declared historical map reference"
        )
    arguments = (
        reference_arguments(map_reference_params) if reference_record is None else None
    )
    reference = None

    def validate_current():
        if reference_record is not None:
            return reference_record
        current = load_reference(map_reference, **arguments)
        if (
            reference is not None
            and current["reference_id"] != reference["reference_id"]
        ):
            raise ValueError(
                "Historical map reference changed during forcing figure preparation"
            )
        return current

    reference = validate_current()
    from hydromt_wflow import WflowSbmModel

    mod = WflowSbmModel(str(wflow_root), mode="r")
    if plot_dir is None:
        plot_dir = os.path.join(str(wflow_root), "plots")

    staticmaps = mod.staticmaps.data

    # Mask to the modelled basin. Cells outside it carry forcing values (the
    # forcing grid is rectangular) that are not part of the model's climate, and
    # including them shifts every domain mean.
    inside = staticmaps["subcatchment"] >= 0
    overlays = (
        {
            name: Path(geoms_dir) / f"{name}.geojson"
            for name in ("basins", "subbasins", "rivers", "locations")
        }
        if geoms_dir is not None
        else {}
    )
    # HydroMT's forcing component loads the entire cube eagerly. Opening the
    # configured file with dask keeps this figure-only reduction bounded.
    forcing_path = Path(
        mod.config.get_value("dir_input", fallback=""),
        mod.config.get_value("input.path_forcing") or "inmaps.nc",
    )
    if not forcing_path.is_absolute():
        forcing_path = Path(wflow_root) / forcing_path
    offset = pd.tseries.frequencies.to_offset(anchor)
    if not isinstance(offset, pd.offsets.YearEnd):
        raise ValueError(f"Expected a year-end reporting anchor, got {anchor!r}")
    m0 = offset.month % 12 + 1
    with xr.open_mfdataset(str(forcing_path), chunks={"time": 365}) as forcing:
        return render_forcing_figures(
            forcing,
            inside,
            plot_dir,
            overlays,
            m0,
            reference=reference,
            validate_current=validate_current,
            annotation=annotation,
        )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import (
            tee_to_log,
            water_year_end_anchor,
        )

        with tee_to_log(sm.log[0]):
            project_dir = sm.params.project_dir

            # R07 B10: forcing / model-input QA figures live beside the forcing
            # they describe, inside the engine subtree.
            # Rule-owned model root (R9 P2 commit 1): the script no longer
            # rebuilds `{project_dir}/hydrology_model` for itself.
            model_dir = sm.params.model_dir
            plot_forcing(
                wflow_root=model_dir,
                plot_dir=f"{model_dir}/forcing/plots",
                gauges_fn=getattr(sm.input, "output_locations", None),
                geoms_dir=sm.params.geoms_dir,
                anchor=water_year_end_anchor(sm.params.water_year_start),
                map_reference=sm.input.map_reference,
                map_reference_params=dict(sm.params.map_reference_params),
                annotation=getattr(sm.params, "annotation", "none"),
            )
