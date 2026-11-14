"""Named hydrology models under ``models/hydrology/<name>/`` (t017 R16, t2609281102).

Each folder is one model. The default ``wflow`` is built by WF1. Further models,
for example a calibration prepared outside the toolbox, are registered as
``user_managed`` siblings and selected by name. A model records itself in
``_engine/model.json``: origin, parent, notes, and the identity of its static
inputs.

**What the registration identity covers.** The ``model_digest`` v1 pointer graph
of the model TOML plus HydroMT's ``staticgeoms/``. It excludes everything the
toolbox writes per forcing recipe (``historical/``, ``historical_comparison/``,
``_engine/``, ``wflow_sbm_<recipe>.toml``). WF1 runs a registered model's
recipes without touching these files, so the identity stays fixed, and any
difference later is a change the toolbox did not make. Experiments record their
recipe's full inputs, forcing included, separately in ``model_reference.yml``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from blueearth_cst.shared.model_digest import (
    ABSENT,
    DIGEST_VERSION,
    MODEL_TOML_NAME,
    model_digest_entries,
    model_digest_from_entries,
)

DEFAULT_MODEL_NAME = "wflow"
MODELS_DIR = "models/hydrology"
RECORD_PATH = "_engine/model.json"
RECORD_SCHEMA_VERSION = 1
ORIGINS = ("built", "user_managed")
#: Folders beside the models that are not models.
RESERVED_NAMES = frozenset({"comparisons"})
_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: Paths the toolbox writes inside a model folder; never part of its identity.
_TOOLBOX_OWNED = ("historical/", "historical_comparison/", "_engine/")
_GEOMS_DIR = "staticgeoms"


class ModelSelectionError(ValueError):
    """A model name is invalid, missing, or ambiguous."""


class ModelChangedError(RuntimeError):
    """A registered model's files changed after registration."""


def validate_model_name(name: object) -> str:
    """Return ``name`` when it is a usable model folder name, else raise."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ModelSelectionError(
            f"model name {name!r} must be lowercase snake_case: a letter, then "
            "letters, digits or underscores (at most 64 characters)"
        )
    if name in RESERVED_NAMES or "_staging_" in name:
        raise ModelSelectionError(
            f"model name {name!r} is reserved for toolbox folders under {MODELS_DIR}/"
        )
    return name


def models_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / MODELS_DIR


def model_root(project_dir: str | Path, name: str = DEFAULT_MODEL_NAME) -> Path:
    """The folder of model ``name`` inside ``project_dir``."""
    return models_root(project_dir) / validate_model_name(name)


def _is_toolbox_owned(rel: str) -> bool:
    return (
        rel.startswith(_TOOLBOX_OWNED)
        or re.fullmatch(r"wflow_sbm_.+\.toml", rel) is not None
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registration_entries(
    root: str | Path, toml_name: str = MODEL_TOML_NAME
) -> dict[str, str]:
    """``{relative_path: sha256 or ABSENT}`` of the files a registration covers."""
    root = Path(root)
    entries = {
        rel: value
        for rel, value in model_digest_entries(root, toml_name)
        if not _is_toolbox_owned(rel)
    }
    geoms = root / _GEOMS_DIR
    if geoms.is_dir():
        for path in sorted(geoms.rglob("*")):
            if path.is_file():
                entries[path.relative_to(root).as_posix()] = _sha256(path)
    return dict(sorted(entries.items()))


def _static_input(root: Path, toml_name: str) -> str | None:
    """The TOML's ``input.path_static`` target, root-relative."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    with open(root / toml_name, "rb") as handle:
        static = (tomllib.load(handle).get("input") or {}).get("path_static")
    return None if static is None else Path(static.replace("\\", "/")).as_posix()


def build_record(
    root: str | Path,
    *,
    name: str,
    origin: str,
    parent: str | None = None,
    notes: str = "",
    toml_name: str = MODEL_TOML_NAME,
    registered_at: str | None = None,
) -> dict[str, Any]:
    """The ``model.json`` document for the model at ``root``."""
    root = Path(root)
    validate_model_name(name)
    if origin not in ORIGINS:
        raise ValueError(f"model origin must be one of {ORIGINS}, not {origin!r}")
    if parent is not None:
        validate_model_name(parent)
    if not (root / toml_name).is_file():
        raise FileNotFoundError(f"no {toml_name} in model folder {root}")
    inventory = registration_entries(root, toml_name)
    static = _static_input(root, toml_name)
    if static is None or inventory.get(static, ABSENT) == ABSENT:
        raise FileNotFoundError(
            f"{root / toml_name} must name an existing input.path_static "
            f"(found {static!r})"
        )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "name": name,
        "origin": origin,
        "parent": parent,
        "notes": notes,
        "registered_at": registered_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_toml": toml_name,
        "digest_version": DIGEST_VERSION,
        "digest": model_digest_from_entries(inventory.items()),
        "inventory": inventory,
    }


def write_record(root: str | Path, record: Mapping[str, Any]) -> Path:
    target = Path(root) / RECORD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return target


def read_record(root: str | Path) -> dict[str, Any] | None:
    """The model's ``model.json``, or ``None`` for a legacy model without one."""
    path = Path(root) / RECORD_PATH
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has unsupported schema_version {record.get('schema_version')!r}"
        )
    return record


def origin(root: str | Path) -> str:
    """``built`` or ``user_managed``; a model without a record was built by WF1."""
    record = read_record(root)
    return "built" if record is None else record["origin"]


def registration_changes(
    root: str | Path, record: Mapping[str, Any] | None = None
) -> list[str]:
    """How the model's files differ from its record; ``[]`` when they agree."""
    root = Path(root)
    record = read_record(root) if record is None else record
    if record is None:
        return []
    if record.get("digest_version") != DIGEST_VERSION:
        return [
            f"{root / RECORD_PATH} uses digest_version {record.get('digest_version')!r}; "
            f"this toolbox computes {DIGEST_VERSION}. Register the model again."
        ]
    was = dict(record["inventory"])
    try:
        now = registration_entries(root, record.get("model_toml", MODEL_TOML_NAME))
    except FileNotFoundError as error:
        return [str(error)]
    changes = []
    for rel in sorted(set(was) | set(now)):
        old, new = was.get(rel), now.get(rel)
        if old == new:
            continue
        if old is None or old == ABSENT:
            changes.append(f"added: {rel}")
        elif new is None or new == ABSENT:
            changes.append(f"removed: {rel}")
        else:
            changes.append(f"changed: {rel}")
    return changes


def require_unchanged(root: str | Path) -> dict[str, Any] | None:
    """Refuse a ``user_managed`` model whose files changed after registration."""
    record = read_record(root)
    if record is None or record["origin"] != "user_managed":
        return record
    changes = registration_changes(root, record)
    if changes:
        raise ModelChangedError(
            f"model {record['name']!r} changed after it was registered "
            f"({'; '.join(changes)}). Register the edited model under a new name "
            "(scripts/register_model.py register), or restore the registered files."
        )
    return record


@dataclass(frozen=True)
class ModelEntry:
    name: str
    root: Path
    origin: str
    record: dict[str, Any] | None


def list_models(project_dir: str | Path) -> list[ModelEntry]:
    """Every model folder in the project, sorted by name.

    A folder is a model when it has a record or a model TOML. Comparisons,
    transient staging folders and names that are not model names are skipped.
    """
    home = models_root(project_dir)
    if not home.is_dir():
        return []
    entries = []
    for path in sorted(home.iterdir()):
        if not path.is_dir():
            continue
        try:
            validate_model_name(path.name)
        except ModelSelectionError:
            continue
        record = read_record(path)
        if record is None and not (path / MODEL_TOML_NAME).is_file():
            continue
        entries.append(
            ModelEntry(
                path.name, path, "built" if record is None else record["origin"], record
            )
        )
    return entries


def _describe(entries: list[ModelEntry]) -> str:
    return ", ".join(f"{entry.name} ({entry.origin})" for entry in entries) or "none"


def resolve_model_name(project_dir: str | Path, requested: str | None) -> str:
    """The model a configuration selects.

    An explicit name is used as given, existing or not: WF1 builds a new name,
    and a consumer that needs the model reports a missing one. Without a name
    the only model is used; with several the choice is the user's, so this
    stops and lists them.
    """
    if requested is not None:
        return validate_model_name(requested)
    entries = list_models(project_dir)
    if len(entries) > 1:
        raise ModelSelectionError(
            f"{models_root(project_dir)} holds several models: {_describe(entries)}. "
            "Set model.name in the project configuration to choose one."
        )
    return entries[0].name if entries else DEFAULT_MODEL_NAME


def reference_model_path(reference: Mapping[str, Any]) -> str:
    """The project-relative model folder an experiment's model reference records.

    References without ``model_path`` used the only model there was, ``wflow``.
    """
    return reference.get("model_path", f"{MODELS_DIR}/{DEFAULT_MODEL_NAME}")


def recipe_key(root: str | Path, forcing_id: str | None) -> str | None:
    """The historical recipe key of ``forcing_id`` within the model at ``root``.

    A registered model's single recipe is keyed ``default`` (its TOML is
    ``wflow_sbm_default.toml``); a built model's single recipe is ``None``.
    """
    if forcing_id is None and origin(root) == "user_managed":
        return "default"
    return forcing_id


def selected_model_root(project_dir: str | Path, config: Mapping[str, Any]) -> Path:
    """The model folder a project configuration selects for an experiment."""
    requested = (config.get("model") or {}).get("name")
    return model_root(project_dir, resolve_model_name(project_dir, requested))


def copy_model_inputs(
    parent_root: str | Path, target_root: str | Path, toml_name: str = MODEL_TOML_NAME
) -> list[str]:
    """Fill a model folder with the parent's registered files it lacks.

    Copies model inputs only (never runs or records) and never overwrites, so a
    user can bring back just the files a calibration changed.
    """
    parent_root, target_root = Path(parent_root), Path(target_root)
    copied = []
    for rel, value in registration_entries(parent_root, toml_name).items():
        destination = target_root / rel
        if value == ABSENT or destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_root / rel, destination)
        copied.append(rel)
    return copied
