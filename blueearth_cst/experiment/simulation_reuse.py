"""Reuse compact frozen responses only when all simulation inputs match."""

from pathlib import Path
from typing import Any

from blueearth_cst.experiment.simulation_record import (
    SimulationFrozenError,
)


def check_simulation_reuse(retained: dict[str, Any], current: dict[str, Any]) -> None:
    """Require identical frozen inputs; changed implementations need a new experiment."""
    if current["simulation_id"] != retained["simulation_id"]:
        raise SimulationFrozenError(
            "simulation inputs changed; use a new experiment name"
        )


def check_completed_simulation_inputs(
    experiment_root,
    *,
    project_root,
    model_root,
    collection_path,
    collection,
    resolution_mode,
    preparation,
    simulation_window,
    julia_command,
    include_unperturbed=True,
    netcdf_grid=None,
    csv_output=None,
    initialization=None,
    historical_forcing=None,
    _ready_simulation=None,
) -> None:
    """Recheck a completed simulation without regenerating its native layout.

    Wflow's output layout is derived from the model and installed simulator.
    Retain that frozen layout only if freshly observed model bytes, code,
    environment, preparation, settings and selected collection still match.
    This saves Julia startup/JIT without caching any input observation.
    """
    from blueearth_cst.experiment.content_identity import (
        content_sha256,
        read_canonical_json,
        stage_environment,
    )
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory
    from blueearth_cst.experiment.simulation_record import (
        _assert_model_reference_current,
        read_simulation_v2,
        simulation_intent_v2,
    )
    from blueearth_cst.experiment.simulation_runs import selected_run_ids
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.experiment.write_model_reference import build_model_reference
    from blueearth_cst.shared.workflow_config_snapshot import (
        file_reference,
        resolve_file_reference,
    )

    root = Path(experiment_root).resolve()
    project = Path(project_root).resolve()
    model = Path(model_root).resolve()
    ready = read_simulation_v2(root) if _ready_simulation is None else _ready_simulation
    retained = read_canonical_json(
        resolve_file_reference(ready["intent"], {"project_root": project})
    )
    from blueearth_cst.shared.named_forcings import HistoricalLayout

    historical = HistoricalLayout(historical_forcing)
    reference = build_model_reference(model, project, historical.toml)
    _assert_model_reference_current(
        retained["documents"]["model_reference"], reference, model
    )
    repo = Path(__file__).resolve().parents[2]
    code = simulation_code_inventory(repo)
    environment = stage_environment(
        ["hydromt-wflow", "xarray", "netCDF4", "pyproj"],
        julia_manifest=repo / "Manifest.toml",
        julia_command=julia_command,
    )
    from blueearth_cst.shared.csv_output import normalize_csv_output

    documents = {
        "model_reference": reference,
        "settings": {
            "simulation_window": dict(simulation_window),
            "csv": normalize_csv_output(
                {"csv": csv_output} if csv_output is not None else None
            ),
        },
        "environment": environment,
        "simulator_adapter_code": code,
        "response_request": retained["documents"]["response_request"],
        "preparation": preparation,
    }
    if documents["settings"]["csv"] != retained["documents"]["settings"]["csv"]:
        raise SimulationFrozenError(
            "simulation CSV precision changed; use a new experiment name"
        )
    if not include_unperturbed:
        documents["settings"]["include_unperturbed"] = False
    if netcdf_grid is not None:
        documents["settings"]["netcdf_grid"] = netcdf_grid
    if initialization is not None:
        import tomllib

        from blueearth_cst.experiment.simulation_runs import resolve_initialization

        with (model / reference["model_toml"]).open("rb") as handle:
            model_config = tomllib.load(handle)
        documents["settings"]["initialization"] = resolve_initialization(
            model, model_config, initialization, project, historical.end_state
        )
    selection = {
        "resolution_mode": resolution_mode,
        "manifest": file_reference(collection_path, "project_root", project),
        "collection_id": collection["collection_id"],
        "collection_revision": collection["collection_revision"],
        "run_ids": selected_run_ids(
            collection, collection_path, include=include_unperturbed
        ),
    }
    simulator = {
        "name": "wflow",
        "revision": content_sha256(
            {"julia:Wflow": environment["packages"]["julia:Wflow"]}
        ),
    }
    current = simulation_intent_v2(
        root.name,
        selection,
        simulator,
        documents,
        preparation_identity_v2(
            preparation, project_root=project, experiment_root=root
        ),
    )
    check_simulation_reuse(retained, current)
