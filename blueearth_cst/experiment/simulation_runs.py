"""Select WF4 members without changing the durable WF3 collection."""

import csv
from pathlib import Path


def include_unperturbed(settings: dict) -> bool:
    """Read the boolean WF4 switch, retaining all members by default."""
    value = settings.get("include_unperturbed", True)
    if type(value) is not bool:
        raise ValueError("simulate_system.include_unperturbed must be true or false")
    return value


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
        perturbed = {row["run_id"] for row in csv.DictReader(handle) if row["st_id"]}
    selected = [run for run in runs if run in perturbed]
    if not selected:
        raise ValueError(
            "include_unperturbed: false leaves no perturbed runs to simulate"
        )
    return selected
