"""One plotting scale per variable per figure kind, shared across sources.

Rule ``derive_climate_levels``'s script (``analyze_climate.smk`` 0.04b). WF0 draws
one figure set per candidate source, each in its own job, so without this no two
of those figures share an axis: a reader comparing ERA5's precipitation map with
CHIRPS's would be comparing two different colour ramps and could not see a
difference that is actually there. This module reads every candidate's store,
pools the values each figure would plot, and writes the boundaries all of them
then draw against.

**Why this is not the sidecar retired on 2026-08-16.** That one shared a scale
between the SOURCE and FORCING families, and it was retired because the two frame
different footprints -- a bar classified on a raster-framed extraction is wrong
for a basin-cropped forcing field (``climate_figures.MAP_EXTENT``). This shares a
scale between two SOURCES, which are both raster-framed extractions over the same
bbox plus buffer. The objection does not transfer; the mechanism
(``cartographic_map.RasterStyle.levels``) is the same one, and its docstring
already describes this use.

**The resolution difference is a finding, not noise to normalise away.** ERA5 at
0.25 degrees against CHIRPS at 0.05 degrees means the coarser grid averages over
more area and damps its own extremes. A shared bar makes that visible, which is
the point. Nothing here rescales a dataset to make the panels agree.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from blueearth_cst.climate_analysis.climate_figures import (
    CLIMATE_VARS,
    FIGURE_KINDS,
    VALUE_DERIVATIONS,
)
from blueearth_cst.shared.grid_cells import cells_csv_mask, masked
from blueearth_cst.shared.snake_utils import DEFAULT_WATER_YEAR_ANCHOR, log_row

#: Quantiles the pooled range is clipped to before classing, mirroring
#: ``RasterStyle.clip_quantiles``' intent: one anomalous cell must not spend the
#: whole ramp. Applied to the POOLED values so the clip is the same decision for
#: every dataset, rather than each clipping to its own tail.
_CLIP_QUANTILES = (0.0, 0.98)


def _plotted_values(ds: xr.Dataset, var: str, kind: str, anchor: str) -> np.ndarray:
    """Every value the ``(var, kind)`` figure would draw, flattened.

    Derived through ``climate_figures.VALUE_DERIVATIONS``, so a scale is always
    computed over the same quantity the figure plots. Deriving it here
    independently is the defect this indirection exists to prevent.
    """
    derive = VALUE_DERIVATIONS[kind]
    drawn = derive(ds[var], CLIMATE_VARS[var], anchor)
    if kind == "monthly":
        parts = [np.asarray(p, dtype=float).ravel() for p in drawn if len(p)]
        values = np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        values = np.asarray(drawn.values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _pooled_range(values: np.ndarray) -> Optional[tuple[float, float]]:
    """Clipped (lower, upper) of the pooled values, or None if there are none."""
    if values.size == 0:
        return None
    lower, upper = (float(v) for v in np.quantile(values, list(_CLIP_QUANTILES)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if upper <= lower:
        # A degenerate range -- a spatially uniform field, or a single year.
        # Widened rather than dropped, matching what `_class_levels` does for
        # the same case: dropping would send every source back to classifying
        # on its own, which is the inconsistency this file exists to remove.
        upper = lower + 1.0
    return lower, upper


def _map_levels(var: str, lower: float, upper: float) -> list[float]:
    """Class boundaries for a map, on the pooled range and the style's ladder.

    Reuses the toolbox's own equal-interval classifier, so the boundaries land
    on the same readable ladder a single-dataset figure would have used -- only
    the range they span is pooled.

    The area-share refinement ``_class_levels`` applies for a single raster is
    deliberately NOT pooled: it weights classes by the fraction of basin AREA
    each covers, and two grids at different resolutions have no common cell area
    to weight by. Equal interval over the pooled range is the honest answer;
    inventing a shared weighting would be a number nobody could interpret.
    """
    from blueearth_cst.shared.cartographic_map import (
        RASTER_STYLES,
        _equal_interval_levels,
    )

    style = RASTER_STYLES[CLIMATE_VARS[var]["style"]]
    levels = _equal_interval_levels(
        lower,
        upper,
        getattr(style, "zero_baseline", None),
        getattr(style, "step_ladder", None),
    )
    return [float(v) for v in levels]


def compute_climate_levels(
    stores: Mapping[str, Union[str, Path]],
    variables: Sequence[str],
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    basin_cells: Optional[Mapping] = None,
) -> dict:
    """Pool every store's plotted values and derive one scale per (var, kind).

    Parameters
    ----------
    stores : mapping
        ``{source_name: extract_historical.nc}`` for every candidate to be drawn
        on a shared scale.
    variables : sequence of str
        The variables to derive scales for. A variable is only pooled across the
        stores that actually CARRY it -- a precipitation-only source contributes
        to ``precip`` and to nothing else, which is what keeps its borrowed era5
        fields out of a scale as surely as it keeps them out of a figure.
    anchor : str
        Water-year resample anchor, as the figures use.
    basin_cells : mapping, optional
        ``{source_name: basin_cells.csv}``. The SERIES kinds are pooled over the
        basin's cells, because that is what rule 0.05 now draws -- a scale
        computed over the buffered extraction and applied to a basin mean is
        exactly the "computed over one quantity, applied to another" defect the
        indirection through ``VALUE_DERIVATIONS`` exists to prevent.

        The ``map`` kind is pooled UNMASKED, and that is not an oversight: the
        map draws the field itself, over the frame ``MAP_EXTENT`` chooses, so
        its colourbar must describe those cells rather than the basin's.

    Returns
    -------
    dict
        ``{var: {"map": [boundaries...], "annual": [lo, hi], "monthly": [lo, hi]}}``.
        A ``(var, kind)`` no store could supply is OMITTED rather than given a
        placeholder, so a consumer falls back to per-figure classification
        instead of drawing against an invented scale.
    """
    levels: dict = {}
    basin_cells = basin_cells or {}
    opened = {name: xr.open_dataset(path) for name, path in stores.items()}
    # The basin-masked twin of each store, for the SERIES kinds. Built once per
    # store rather than per (var, kind): the mask is a property of the grid.
    on_basin = {
        name: masked(ds, cells_csv_mask(ds, basin_cells.get(name)))
        for name, ds in opened.items()
    }
    try:
        for var in variables:
            carriers = [name for name, ds in opened.items() if var in ds]
            if not carriers:
                continue
            per_kind: dict = {}
            for kind in FIGURE_KINDS:
                # `map` from the full field, the series kinds from the basin --
                # each scale pooled over the domain its own figure draws.
                source_of = opened if kind == "map" else on_basin
                pooled = np.concatenate(
                    [
                        _plotted_values(source_of[name], var, kind, anchor)
                        for name in carriers
                    ]
                    or [np.array([], dtype=float)]
                )
                bounds = _pooled_range(pooled)
                if bounds is None:
                    continue
                lower, upper = bounds
                per_kind[kind] = (
                    _map_levels(var, lower, upper) if kind == "map" else [lower, upper]
                )
            if per_kind:
                levels[var] = per_kind
            log_row(
                f"Pooled {var} over {len(carriers)} source(s): "
                f"{', '.join(sorted(carriers))}",
                module="levels",
            )
    finally:
        for ds in opened.values():
            ds.close()
    return levels


def write_climate_levels(levels: Mapping, out_path: Union[str, Path]) -> Path:
    """Write the scales as JSON, sorted so the file is diffable."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(levels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_row(f"Wrote shared climate scales -> {out_path.name}", module="levels")
    return out_path


def read_climate_levels(path: Optional[Union[str, Path]]) -> dict:
    """Read the scales, or an empty mapping when there are none.

    ``None`` is the ordinary case for WF1, which draws ONE source and therefore
    has nothing to share a scale with. An absent file must degrade to per-figure
    classification rather than raise, or the model build would be coupled to a
    candidate set it does not have.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log, water_year_end_anchor

        with tee_to_log(sm.log[0]):
            write_climate_levels(
                compute_climate_levels(
                    dict(zip(sm.params.sources, sm.input.climate_ncs)),
                    sm.params.variables,
                    anchor=water_year_end_anchor(sm.params.water_year_start),
                    # The series scales are pooled over the SAME cells rule 0.05
                    # averages, or the shared axis describes a domain no figure
                    # draws.
                    basin_cells=dict(zip(sm.params.sources, sm.input.basin_cells)),
                ),
                sm.output.levels,
            )
