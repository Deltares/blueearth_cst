"""Publish and read immutable historical responses without the mutable model."""

from __future__ import annotations

import copy
import csv
import json
import re
import shutil
import tomllib
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe
from blueearth_cst.shared.native_locations import native_location_labels
from blueearth_cst.shared.provenance import (
    collection_sha256,
    environment_file_hashes,
    file_sha256,
    toolbox_identity,
)
from blueearth_cst.shared.workflow_config_snapshot import snapshot_bytes

SCHEMA = "historical-run/1"

if TYPE_CHECKING:
    import pandas as pd


def historical_run_name(settings: object) -> str | None:
    """Validate the optional closed retention mapping and portable label."""
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != {"name"}:
        raise ValueError("historical_run must be null or a mapping with only name")
    name = settings["name"]
    if not isinstance(name, str) or not re.fullmatch("[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("historical_run.name must match [a-z][a-z0-9_]{0,63}")
    if name in {
        "con",
        "prn",
        "aux",
        "nul",
        *[f"com{i}" for i in range(10)],
        *[f"lpt{i}" for i in range(10)],
    }:
        raise ValueError("historical_run.name is a reserved device name")
    return name


def run_directory(project_dir: str | Path, name: str) -> Path:
    """Resolve one named historical run beneath the project publication root."""
    historical_run_name({"name": name})
    root = Path(project_dir).resolve() / "simulations" / "historical"
    target = root / name
    if target.resolve().parent != root.resolve() or target.is_symlink():
        raise ValueError("historical run must not resolve outside its publication root")
    return target


def require_new_run(project_dir: str | Path, name: str) -> Path:
    """Refuse reuse, including damaged publications, before running Wflow."""
    target = run_directory(project_dir, name)
    if target.exists():
        raise FileExistsError(
            f"historical run {name!r} already exists; choose a new name"
        )
    return target


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
    model.get("input", {}).pop("forcing", None)
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


@contextmanager
def retain_historical_run(
    *,
    project_dir: str | Path,
    name: str,
    toml_path: Path,
    registry_path: Path,
    forcing_record_path: Path,
    config: dict,
    config_path: Path,
    workflow_config_path: Path | None,
    repo_root: Path,
) -> Iterator[None]:
    """Capture before execution, then publish only after a successful caller body.

    The caller runs Wflow and validates optional native grids inside this context.
    Staging is a sibling of the final directory, so rename is one filesystem action.
    The reservation is exclusive and never expires automatically after a crash.
    """
    import pandas as pd

    from blueearth_cst.experiment.response_series import response_frame
    from blueearth_cst.experiment.wflow_response_reader import (
        NativeRunArtifacts,
        ResponseRequest,
        open_responses,
    )

    target = require_new_run(project_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    reservation = target.parent / f".{name}.lock"
    reservation.mkdir()
    stage = target.parent / f".{name}.{uuid.uuid4().hex}.partial"
    try:
        require_new_run(project_dir, name)
        stage.mkdir()
        files, identity = execution_inputs(toml_path, registry_path)
        files[str(forcing_record_path.resolve())] = file_sha256(forcing_record_path)
        record = json.loads(forcing_record_path.read_text(encoding="utf-8"))
        recipe = normalize_forcing_recipe(
            config["climate"]["selected"], config["climate"].get("forcing")
        )
        if record != {
            "recipe": recipe.to_dict(),
            "forcing_sha256": identity["forcing_sha256"],
        }:
            raise ValueError(
                "prepared forcing provenance disagrees with executed recipe or forcing"
            )
        stage.joinpath("composed_config.yml").write_bytes(
            snapshot_bytes("build_model", config_path, workflow_config_path, config)
        )
        shutil.copyfile(toml_path, stage / "wflow_sbm.toml")
        shutil.copyfile(registry_path, stage / "location_registry.csv")
        evidence = {
            "schema_version": SCHEMA,
            "name": name,
            **identity,
            "recipe": record["recipe"],
            "inputs": files,
            "toolbox": toolbox_identity(repo_root),
            "environment": environment_file_hashes(repo_root),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _check_inputs(files)
        yield
        _check_inputs(files)
        native_config = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        output_path = (
            toml_path.parent
            / native_config.get("dir_output", ".")
            / native_config["output"]["csv"]["path"]
        )
        with output_path.open(encoding="utf-8", newline="") as stream:
            native_headers = next(csv.reader(stream))
        duplicates: dict[str, str] = {}
        crosswalk = native_location_labels(
            native_config["output"]["csv"]["column"],
            native_headers,
            _input_path(
                toml_path, native_config, native_config["input"]["path_static"]
            ),
            duplicates,
        )
        unknown = (
            {key for key in native_headers if key.startswith("Q_")}
            - crosswalk.keys()
            - duplicates.keys()
        )
        if unknown:
            raise ValueError(
                f"historical native discharge stations are unrecognized: {sorted(unknown)}"
            )
        responses = open_responses(
            name, NativeRunArtifacts(output_path, toml_path), ResponseRequest(("q",))
        )
        frame = response_frame(responses)
        registry = pd.read_csv(registry_path, dtype={"wflow_id": str})
        if registry["wflow_id"].duplicated().any() or not set(frame.columns) <= set(
            registry["wflow_id"]
        ):
            raise ValueError(
                "historical response stations do not match the stable registry"
            )
        frame.to_csv(stage / "discharge.csv", index_label="time", float_format="%.17g")
        registry[registry.wflow_id.isin(frame.columns)].sort_values("wflow_id").to_csv(
            stage / "stations.csv", index=False
        )
        shutil.copyfile(output_path, stage / "output.csv")
        evidence.update(
            native_crosswalk=crosswalk,
            duplicate_native_columns=duplicates,
            units="m3 s-1",
            time_label="interval_end",
            completed_at=datetime.now(timezone.utc).isoformat(),
            samples=len(frame),
            stations=sorted(frame.columns),
        )
        (stage / "run.json").write_bytes(_json_bytes(evidence))
        _check_inputs(files)
        payloads = {path.name: file_sha256(path) for path in stage.iterdir()}
        (stage / "complete.json").write_bytes(
            _json_bytes({"schema_version": SCHEMA, "payloads": payloads})
        )
        # rename refuses an existing nonempty publication; never replace it.
        require_new_run(project_dir, name)
        stage.rename(target)
    finally:
        # Interrupted stages remain unavailable and inspectable; never delete evidence.
        reservation.rmdir()


def read_historical_run(
    project_dir: str | Path, name: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Verify a complete publication and read it without accessing original inputs."""
    import pandas as pd

    target = run_directory(project_dir, name)
    marker = target / "complete.json"
    if marker.is_symlink() or marker.resolve().parent != target.resolve():
        raise ValueError("historical completion marker escapes the retained run")
    complete = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "run.json",
        "output.csv",
        "discharge.csv",
        "stations.csv",
        "location_registry.csv",
        "wflow_sbm.toml",
        "composed_config.yml",
    }
    if (
        complete.get("schema_version") != SCHEMA
        or set(complete.get("payloads", {})) != expected
    ):
        raise ValueError("unsupported or incomplete historical run")
    for filename, digest in complete["payloads"].items():
        path = target / filename
        if (
            path.is_symlink()
            or path.resolve().parent != target.resolve()
            or file_sha256(path) != digest
        ):
            raise ValueError(f"damaged historical run payload: {filename}")
    record = json.loads((target / "run.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA or record.get("name") != name:
        raise ValueError("historical run identity mismatch")
    with (target / "discharge.csv").open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if len(set(header)) != len(header):
        raise ValueError("duplicate retained discharge columns")
    frame = pd.read_csv(
        target / "discharge.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    stations = pd.read_csv(target / "stations.csv", dtype={"wflow_id": str})
    if sorted(frame.columns) != record["stations"] or len(frame) != record["samples"]:
        raise ValueError("retained discharge shape disagrees with completion evidence")
    return record, frame, stations
