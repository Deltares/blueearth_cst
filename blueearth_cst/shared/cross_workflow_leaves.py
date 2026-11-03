"""Consumer-local external leaves shared by runners and scheduling fixtures.

WF3 generation declares its own source producers. WF4 simulation consumes a
validated ready collection and WF1 model files; indicators-only consumes retained
simulation and response records. Typed readers verify content beyond existence.
"""

from __future__ import annotations

# Archive marker retained for fixture staging, not a WF4 dependency.
LEAF_WF1_SNAPSHOT = "config/build_model/run_record.yml"

# WF4's model reference writer consumes these WF1 artifacts.
LEAF_MODEL_TOML = "models/hydrology/wflow/wflow_sbm.toml"
LEAF_MODEL_READY = "models/hydrology/wflow/.outputs_configured"

#: The full set. Order is stable so failure messages read the same way twice —
#: and it is DAG order, so the first entry is the one Snakemake would have
#: reported had the run been allowed to start.
LEAVES: tuple[str, ...] = (LEAF_MODEL_TOML, LEAF_MODEL_READY)

#: The workflow that produces every leaf above. Named rather than spelled at
#: each call site, so a message can say what to RUN and not only what is absent.
LEAF_PRODUCER = "build_model"


def consumer_leaves(
    consumer, *, operation="simulate-and-indicators", experiment_name=None
):
    """Return static external leaves for the selected consumer and operation.

    Collection readiness and complete native inventories require their typed
    readers; a constant path list cannot certify either one.
    """
    if consumer in {
        "analyze_climate",
        "generate_scenarios",
        "build_model",
        "analyze_projections",
    }:
        return ()
    if consumer != "simulate_system":
        raise ValueError(f"unknown workflow consumer: {consumer}")
    if operation == "simulate-and-indicators":
        return LEAVES
    if operation == "indicators-only":
        if not experiment_name:
            raise ValueError("indicators-only leaves require experiment_name")
        return (
            f"experiments/{experiment_name}/_engine/simulation.json",
            f"experiments/{experiment_name}/_engine/response_inventory.json",
        )
    raise ValueError(f"unknown simulation operation: {operation}")
