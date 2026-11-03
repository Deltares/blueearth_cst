"""Rule 0.04b: canonical source figures and recoverable captions."""

from blueearth_cst.climate_analysis.diagnostic_outputs import diagnostic_source_outputs
from blueearth_cst.climate_analysis.diagnostic_render import render


def run(sm):
    settings = dict(sm.params.settings)
    paths = diagnostic_source_outputs(sm.params.store_dir, sm.params.source, settings)
    render(
        paths,
        settings,
        int(sm.params.m0),
        [sm.params.source],
        list(sm.params.pooled_roots),
        {key: sm.input[key] for key in ("basins", "subbasins", "rivers", "locations")},
        map_reference=sm.input.map_reference,
        map_reference_params=dict(sm.params.map_reference_params),
        annotation=getattr(sm.params, "annotation", "none"),
    )
    if settings["subbasin_figures"]:
        from blueearth_cst.climate_analysis.diagnostic_subbasins import render_subbasins

        render_subbasins(sm, settings, [sm.params.source])


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
