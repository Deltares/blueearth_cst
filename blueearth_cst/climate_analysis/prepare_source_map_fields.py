"""Shared WF0/WF1 numerical field producer, independent of rendering."""

from blueearth_cst.climate_analysis.source_map_fields import prepare_source_map_fields
from blueearth_cst.shared.map_preparation import (
    publish_source_fields,
    source_spec_from_dict,
)


def run(sm):
    """Prepare, seal and atomically publish the declared source fields."""
    spec = source_spec_from_dict(sm.params.source_spec)
    completeness = dict(sm.params.completeness)
    prepared = prepare_source_map_fields(spec, sm.params.m0, completeness)
    publish_source_fields(
        sm.output.fields,
        prepared.fields,
        spec,
        sm.params.m0,
        completeness,
        prepared.preparation_identity,
    )
    from blueearth_cst.shared.snake_utils import log_row

    log_row(f"Wrote native-grid map fields for {spec.source}", module="maps")


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
