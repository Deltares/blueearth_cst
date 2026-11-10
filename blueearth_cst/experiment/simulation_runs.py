"""Select WF4 members and their initialization without changing WF3 collections."""

import csv
from pathlib import Path

INITIAL_STATES = ("auto", "historical", "cold")
#: Retained experiments frozen before t010 ran the model TOML's own state
#: with no discarded warm-up.
LEGACY_INITIALIZATION = {"state": "model", "source": "model", "warmup_years": 0}


def include_unperturbed(settings: dict) -> bool:
    """Read the boolean WF4 switch, retaining all members by default."""
    value = settings.get("include_unperturbed", True)
    if type(value) is not bool:
        raise ValueError("simulate_system.include_unperturbed must be true or false")
    return value


def initialization(settings: dict) -> dict:
    """Validate the WF4 ``initialization`` block, defaulting to auto + one year."""
    block = settings.get("initialization")
    block = {} if block is None else block
    if type(block) is not dict:
        raise ValueError("simulate_system.initialization must be a mapping")
    unknown = set(block) - {"state", "warmup_years"}
    if unknown:
        raise ValueError(
            f"unknown simulate_system.initialization fields: {sorted(unknown)}"
        )
    state = block.get("state", "auto")
    if state not in INITIAL_STATES:
        raise ValueError(
            "simulate_system.initialization.state must be one of "
            + ", ".join(INITIAL_STATES)
        )
    years = block.get("warmup_years", 1)
    if type(years) is not int or years < 0:
        raise ValueError(
            "simulate_system.initialization.warmup_years must be a nonnegative integer"
        )
    return {"state": state, "warmup_years": years}


def resolve_initialization(
    model_root, model_config: dict, requested: dict, project_root
) -> dict:
    """Bind the requested initialization to the state file each run will read.

    An explicit warm state in the source model TOML (``auto`` keeps it) wins
    over the WF1 historical end state; an explicit ``historical`` or ``cold``
    request that contradicts it is refused rather than silently overridden.
    The historical state is outside the model digest, so its bytes are bound
    here and a changed WF1 end state changes the simulation identity.
    """
    from blueearth_cst.experiment.batch_sizing import STATE_ANCHOR
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    cold = model_config.get("model", {}).get("cold_start__flag", True)
    if type(cold) is not bool:
        raise ValueError("model.cold_start__flag must be boolean")
    state = requested["state"]
    if state == "auto":
        source = "historical" if cold else "model"
    elif not cold:
        raise ValueError(
            f"initialization.state: {state} conflicts with the warm state configured "
            "in the model TOML (model.cold_start__flag = false); use auto to keep it"
        )
    else:
        source = state
    resolved = {
        "state": state,
        "source": source,
        "warmup_years": requested["warmup_years"],
    }
    if source == "historical":
        path = Path(model_root, *STATE_ANCHOR)
        if not path.is_file():
            raise ValueError(
                f"WF1 historical end state is missing: {path}. Run build_model's "
                "historical simulation, or set simulate_system.initialization.state: cold"
            )
        if not model_config.get("state", {}).get("variables"):
            raise ValueError("historical initialization requires state.variables")
        resolved["state_file"] = file_reference(path, "project_root", project_root)
    return resolved


def frozen_initialization(settings: dict) -> dict:
    """Return a frozen intent's initialization, mapping legacy intents explicitly."""
    return settings.get("initialization", LEGACY_INITIALIZATION)


def validate_frozen_initialization(value: dict) -> None:
    """Check the closed shape recorded in simulation intents."""
    if type(value) is not dict:
        raise ValueError("simulation initialization must be a mapping")
    keys = {"state", "source", "warmup_years"}
    if value.get("source") == "historical":
        keys.add("state_file")
    if set(value) != keys:
        raise ValueError("simulation initialization has unclassified fields")
    if (
        value["state"] not in INITIAL_STATES
        or value["source"] not in ("historical", "cold", "model")
        or type(value["warmup_years"]) is not int
        or value["warmup_years"] < 0
    ):
        raise ValueError("simulation initialization is invalid")


def selected_run_ids(
    collection: dict, manifest_path, *, include: bool = True
) -> list[str]:
    """Keep collection order and IDs; exclude only rows with an empty st_id."""
    runs = [item["run_id"] for item in collection["series"]]
    if include:
        return runs
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    path = Path(manifest_path).resolve()
    lookup = resolve_file_reference(
        collection["scenario_run_lookup"],
        {"project_root": path.parent.parents[3], "record_directory": path.parent},
    )
    with lookup.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        if collection.get("schema_version") == "delta-scenario-collection/1":
            perturbed = {row["run_id"] for row in rows if row["role"] == "projection"}
        else:
            perturbed = {row["run_id"] for row in rows if row["st_id"]}
    selected = [run for run in runs if run in perturbed]
    if not selected:
        raise ValueError(
            "include_unperturbed: false leaves no perturbed runs to simulate"
        )
    return selected
