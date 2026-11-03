"""Scientific source boundary shared by WF4 planning and completed-run reuse."""

from pathlib import Path

from blueearth_cst.experiment.content_identity import repository_code_inventory

SIMULATION_CODE_ENTRIES = (
    "blueearth_cst/experiment/simulation_code.py",
    "blueearth_cst/experiment/simulator_adapter.py",
    "blueearth_cst/experiment/downscale_climate_forcing.py",
    "blueearth_cst/experiment/run_wflow_batch.jl",
    # This helper also constructs Wflow.Config and invokes Wflow.run.
    "blueearth_cst/shared/wflow_progress.jl",
    "blueearth_cst/shared/resource_timing.jl",
)

# Stop traversal at non-numerical boundaries, rather than maintaining a second
# list of every transitive scientific import. This module itself is inventoried.
SIMULATION_INVENTORY_EXCLUDED = (
    # Generation is already sealed by collection id/revision and forcing bytes.
    # The collection reader still validates its schema and generator projection.
    "blueearth_cst/experiment/generation_plan.py",
    # Records and archives validate/persist values hashed separately by the
    # simulation intent: model, collection, settings, response and preparation.
    "blueearth_cst/experiment/simulation_record.py",
    "blueearth_cst/shared/workflow_config_snapshot.py",
    # The scientific entry points use snake_utils only for tee_to_log. Following
    # its unrelated helpers pulls rendering/export code into simulation identity.
    "blueearth_cst/shared/snake_utils.py",
    "blueearth_cst/shared/progress.py",
    "blueearth_cst/shared/run_logged.py",
    # Environment probes use these only to capture their console output.
    "blueearth_cst/shared/output_capture.py",
    "blueearth_cst/shared/run_log_core.py",
)


def simulation_code_inventory(repo: Path) -> list[dict[str, str]]:
    """Observe scientific sources freshly; scheduling remains with Snakemake."""
    return repository_code_inventory(
        repo, SIMULATION_CODE_ENTRIES, excluded=SIMULATION_INVENTORY_EXCLUDED
    )
