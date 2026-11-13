"""Read each forcing recipe's current historical run for comparison (R16, D18).

A run lives with its model, in ``<model>/historical/<recipe>/run/``; there are
no named retained copies (t017 ruling D18). Changed model settings go to a new
model folder, so a model's recipe runs only change when rerun unchanged.
"""

from __future__ import annotations

import copy
import json
import tomllib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe
from blueearth_cst.shared.provenance import collection_sha256, file_sha256

if TYPE_CHECKING:
    import pandas as pd


def warn_retired_historical_run(settings: object) -> None:
    """``historical_run`` named retained copies; it is accepted and ignored."""
    if settings is not None:
        warnings.warn(
            "build_model.historical_run is no longer used: each recipe's run stays "
            "in models/hydrology/<model>/historical/<recipe>/run/, and changed model "
            "settings belong in a new model folder. Remove the setting.",
            FutureWarning,
            stacklevel=2,
        )


def _json_bytes(document: dict) -> bytes:
    """Serialize JSON without silently accepting nonfinite metadata."""
    return (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False, default=str)
        + "\n"
    ).encode("utf-8")


def _input_path(toml_path: Path, config: dict, value: str) -> Path:
    """Apply Wflow's TOML-relative input anchor."""
    return (toml_path.parent / config.get("dir_input", ".") / value).resolve()


def execution_inputs(toml_path: Path, registry_path: Path) -> tuple[dict, dict]:
    """Fingerprint executed files and a forcing-independent scientific model.

    Input pointers are discovered structurally, including nested waterbody inputs.
    Missing referenced files fail rather than entering the identity as absent.
    """
    from blueearth_cst.experiment.content_identity import netcdf_content_sha256

    config = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    files = {
        str(toml_path.resolve()): file_sha256(toml_path),
        str(registry_path.resolve()): file_sha256(registry_path),
    }
    model = copy.deepcopy(config)
    for key in ("dir_input", "dir_output", "time", "logging", "output"):
        model.pop(key, None)
    model.get("input", {}).pop("path_forcing", None)
    # Variable bindings and transformations are scientific model semantics.
    # Only the forcing file itself varies independently between comparisons.
    model.get("state", {}).pop("path_output", None)
    if model.get("model", {}).get("cold_start__flag", True):
        # HydroMT declares an input-state path even for cold runs. Wflow does not
        # read it in this mode; its absence must not prevent retaining a cold run.
        model.get("state", {}).pop("path_input", None)

    def visit(node: object, trail: tuple[str, ...], *, scientific: bool) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                visit(item, (*trail, str(i)), scientific=scientific)
        elif isinstance(node, dict):
            for key, value in list(node.items()):
                if isinstance(value, str) and (
                    key == "path" or key.startswith("path_")
                ):
                    path = _input_path(toml_path, config, value)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"missing executed input {'.'.join((*trail, key))}: {path}"
                        )
                    files[str(path)] = file_sha256(path)
                    if scientific:
                        node[key] = {
                            "content_sha256": netcdf_content_sha256(path)
                            if path.suffix.lower() == ".nc"
                            else files[str(path)]
                        }
                else:
                    visit(value, (*trail, key), scientific=scientific)

    visit(model, (), scientific=True)
    forcing = _input_path(toml_path, config, config["input"]["path_forcing"])
    files[str(forcing)] = file_sha256(forcing)
    # Native location labelling also reads geometry crosswalks; pin those reads.
    static = _input_path(toml_path, config, config["input"]["path_static"])
    for path in sorted((static.parent / "staticgeoms").glob("*.geojson")):
        files[str(path.resolve())] = file_sha256(path)
    return files, {
        "model_sha256": collection_sha256(model),
        "model": model,
        "time": config["time"],
        "forcing_sha256": files[str(forcing)],
    }


def write_forcing_record(
    toml_path: Path, destination: Path, selected: str, forcing: dict | None
) -> None:
    """Bind the prepared forcing bytes to the recipe at their producer."""
    config = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    path = _input_path(toml_path, config, config["input"]["path_forcing"])
    recipe = normalize_forcing_recipe(selected, forcing)
    destination.write_bytes(
        _json_bytes({"recipe": recipe.to_dict(), "forcing_sha256": file_sha256(path)})
    )


def _check_inputs(files: dict[str, str]) -> None:
    """Detect source mutation between preflight and completed publication."""
    for name, digest in files.items():
        if not Path(name).is_file() or file_sha256(name) != digest:
            raise ValueError(
                f"executed input changed during historical simulation: {name}"
            )


def read_recipe_run(
    model_root: Path, forcing_id: str, registry_path: Path
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """One recipe's current run: identity record, discharge, stations, input hashes.

    The identity fingerprints the executed TOML and its inputs now; the caller
    re-checks the returned hashes after use so a concurrent change is refused.
    """
    import pandas as pd

    from blueearth_cst.experiment.response_series import response_frame
    from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
    from blueearth_cst.shared.named_forcings import HistoricalLayout
    from blueearth_cst.shared.response_tables import open_table_responses

    model_root = Path(model_root)
    layout = HistoricalLayout(forcing_id)
    toml_path = model_root / layout.toml
    files, identity = execution_inputs(toml_path, registry_path)
    forcing_record_path = model_root / layout.config_dir / "historical_forcing.json"
    files[str(forcing_record_path.resolve())] = file_sha256(forcing_record_path)
    forcing_record = json.loads(forcing_record_path.read_text(encoding="utf-8"))
    if forcing_record["forcing_sha256"] != identity["forcing_sha256"]:
        raise ValueError(f"recipe {forcing_id}: forcing differs from its record")
    table_record = model_root / layout.response_record
    files[str(table_record.resolve())] = file_sha256(table_record)
    frame = response_frame(
        open_table_responses(forcing_id, table_record, ResponseRequest(("q",)))
    )
    registry = pd.read_csv(registry_path, dtype={"wflow_id": str})
    if registry["wflow_id"].duplicated().any() or not set(frame.columns) <= set(
        registry["wflow_id"]
    ):
        raise ValueError(f"recipe {forcing_id}: stations do not match the registry")
    stations = (
        registry[registry.wflow_id.isin(frame.columns)]
        .sort_values("wflow_id")
        .reset_index(drop=True)
    )
    native = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    record = {
        **identity,
        "recipe": forcing_record["recipe"],
        "forcing_mapping": native.get("input", {}).get("forcing", {}),
        "units": "m3 s-1",
        "time_label": "interval_end",
        "samples": len(frame),
        "stations": sorted(frame.columns),
    }
    return record, frame, stations, files
