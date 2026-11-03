"""Product-local production records for launcher-observed WF2 executions.

A product keeps one record of which observed invocation produced each current
file, beside retained copies of those invocations' diagnostics. The launcher
inventories the product (bytes and write times) before its child starts and
after its diagnostics are merged, inside the project execution lock. A file is
credited to the invocation when its bytes or write time changed, so an
identical rewrite (a deterministic plot rerun) is credited too. A no-op writes
nothing and changes nothing here, whatever its exit status. Bytes that changed without an
observation (older products, bare Snakemake, a killed launcher) are recorded
with no producer rather than credited to a later run.

Central invocation records and `logs/` remain the attempt history; this module
only copies from them. It is deliberately outside `execution_outputs` and
`invocation_history`, whose code identity WF3/WF4 records include.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "product-production/1"
RECORD_DIRECTORY = "_engine/execution"
RECORD_NAME = "production.json"
# Product-local provenance this module and imports own; never product bytes.
EXCLUDED = (RECORD_DIRECTORY, "_engine/imported")
PRODUCTS = {"analyze_projections": "wf2"}
# Shared spatial producers run inside WF2 but write data/spatial/, not the product.
SHARED_STEPS = {
    "analyze_projections": [
        {"rule": "2.01", "name": "delineate_region", "outputs": "data/spatial/geoms"},
        {
            "rule": "2.02",
            "name": "delineate_subbasins_and_rivers",
            "outputs": "data/spatial/geoms",
        },
    ]
}


@dataclass(frozen=True)
class Observation:
    workflow: str
    project_root: Path
    product: Path
    before: dict[str, dict[str, Any]]
    # Write times only detect rewrites under the lock; they are never recorded.
    before_mtimes: dict[str, int]


def product_root(project_root: Path, workflow: str) -> Path | None:
    """The product directory named by the current captured configuration."""
    if workflow not in PRODUCTS:
        return None
    from blueearth_cst.shared.workflow_config_snapshot import read_archive

    record = read_archive(project_root, workflow, f"config/{workflow}")
    ensemble = record["loaded_config"]["workflows"][workflow]["ensemble"]
    if not isinstance(ensemble, str) or Path(ensemble).name != ensemble:
        raise ValueError(f"ensemble is not a single directory name: {ensemble!r}")
    return Path(project_root) / "data/climate/projections" / ensemble


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(product: Path) -> dict[str, dict[str, Any]]:
    """Digest every product file, excluding product-local provenance."""
    return {
        name: {
            "sha256": _sha256(product / name),
            "size_bytes": (product / name).stat().st_size,
        }
        for name in _product_files(product)
    }


def write_times(product: Path) -> dict[str, int]:
    """Nanosecond write times of the files :func:`inventory` digests."""
    return {
        name: (product / name).stat().st_mtime_ns for name in _product_files(product)
    }


def _product_files(product: Path) -> list[str]:
    names = []
    if not product.exists():
        return names
    for path in sorted(product.rglob("*")):
        relative = path.relative_to(product).as_posix()
        if any(
            relative == item or relative.startswith(f"{item}/") for item in EXCLUDED
        ):
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"product contains a linked path: {relative}")
        if path.is_file():
            names.append(relative)
    return names


def read_record(product: Path) -> dict[str, Any] | None:
    path = product / RECORD_DIRECTORY / RECORD_NAME
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported production record: {path}")
    return record


def observe(workflow: str, project_root: Path) -> Observation | None:
    """Inventory a product before its producing child starts.

    Fails open: without an observation the run's bytes stay unattributed.
    """
    if workflow not in PRODUCTS:
        return None
    try:
        project_root = Path(project_root).resolve()
        product = product_root(project_root, workflow)
        return Observation(
            workflow, project_root, product, inventory(product), write_times(product)
        )
    except Exception as error:  # noqa: BLE001 -- diagnostics cannot block a run
        _unavailable(workflow, error)
        return None


def _unavailable(label: str, error: BaseException) -> None:
    try:
        print(f"(production record unavailable for {label}: {error})", file=sys.stderr)
    except Exception:  # noqa: BLE001 -- diagnostic streams can fail too
        return


def attribute(
    written: set[str],
    after: Mapping[str, Mapping[str, Any]],
    previous: Mapping[str, Mapping[str, Any]],
    invocation_id: str,
) -> dict[str, dict[str, Any]]:
    """Credit written files to this invocation; keep only still-true credits."""
    files = {}
    for name, state in after.items():
        if name in written:
            producer = invocation_id
        else:
            prior = previous.get(name, {})
            producer = (
                prior.get("producer")
                if prior.get("sha256") == state["sha256"]
                else None
            )
        files[name] = {**state, "producer": producer}
    return files


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _region_sha256(project_root: Path) -> str | None:
    from blueearth_cst.shared.wf3_science import region_geojson_path

    path = Path(region_geojson_path(project_root))
    return _sha256(path) if path.is_file() else None


def _generation(
    observation: Observation,
    invocation: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    written: set[str],
    diagnostics: list[str],
) -> dict[str, Any]:
    configuration = invocation.get("configuration", {})
    return {
        "status": invocation["status"],
        "entry_point": invocation.get("entry_point"),
        "parent_invocation_id": invocation.get("parent_invocation_id"),
        "started_at_utc": invocation.get("started_at_utc"),
        "ended_at_utc": invocation.get("ended_at_utc"),
        "invocation_record": f"config/_engine/invocations/{invocation['invocation_id']}.json",
        "configuration": {
            "run_record": configuration.get("run_record"),
            "effective_config_sha256": configuration.get("effective_config_sha256"),
            "configuration_inputs_sha256": configuration.get(
                "configuration_inputs_sha256"
            ),
        },
        "dependencies": {
            "data/spatial/geoms/region.geojson": _region_sha256(
                observation.project_root
            )
        },
        "changes": {
            "added": sum(name not in before for name in after),
            "changed": sum(
                name in before and before[name] != state
                for name, state in after.items()
            ),
            "rewritten_identical": sum(
                name in written and before.get(name) == state
                for name, state in after.items()
            ),
            "removed": sum(name not in after for name in before),
        },
        "diagnostics": diagnostics,
    }


def _retain_diagnostics(
    observation: Observation, invocation: Mapping[str, Any]
) -> list[str]:
    """Copy this invocation's merged workflow log and benchmark report."""
    relative = invocation.get("execution_directory")
    if not relative:
        return []
    source = observation.project_root / relative
    label = PRODUCTS[observation.workflow]
    target = observation.product / RECORD_DIRECTORY / invocation["invocation_id"]
    retained = []
    for name in (f"{label}.log", f"{label}-benchmarks.md", f"{label}-benchmarks.tsv"):
        if (source / name).is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target / name)
            retained.append(f"{RECORD_DIRECTORY}/{invocation['invocation_id']}/{name}")
    return retained


def record_production(
    observation: Observation, invocation: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Publish product attribution after the launcher finished the invocation."""
    if invocation["status"] not in {"succeeded", "failed"}:
        raise ValueError("production is recorded only for a finished invocation")
    after = inventory(observation.product)
    times = write_times(observation.product)
    written = {
        name
        for name, state in after.items()
        if observation.before.get(name) != state
        or observation.before_mtimes.get(name) != times.get(name)
    }
    if not written and set(after) == set(observation.before):
        return None
    from blueearth_cst.shared.workflow_config_snapshot import _atomic_json

    previous = read_record(observation.product) or {"files": {}, "generations": {}}
    identity = invocation["invocation_id"]
    files = attribute(written, after, previous["files"], identity)
    # The changing invocation stays described even when it only removed files.
    owners = {item["producer"] for item in files.values()} | {identity}
    generations = {
        key: value for key, value in previous["generations"].items() if key in owners
    }
    generations[identity] = _generation(
        observation,
        invocation,
        observation.before,
        after,
        written,
        _retain_diagnostics(observation, invocation),
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "workflow": observation.workflow,
        "product": observation.product.relative_to(observation.project_root).as_posix(),
        # Only a successful observed run establishes a coherent product.
        "state": "complete" if invocation["status"] == "succeeded" else "incomplete",
        "latest_invocation_id": identity,
        "updated_at_utc": _now(),
        "shared_steps": SHARED_STEPS.get(observation.workflow, []),
        "unattributed_files": sum(item["producer"] is None for item in files.values()),
        "generations": generations,
        "files": files,
    }
    directory = observation.product / RECORD_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / RECORD_NAME, record)
    for child in directory.iterdir():
        if child.is_dir() and child.name not in generations:
            # Retained copies only; the central execution logs stay intact.
            shutil.rmtree(child)
    return record


def publish(
    observation: Observation | None,
    invocation_path: Path,
    invocation: Mapping[str, Any],
) -> None:
    """Record production without ever replacing the execution outcome."""
    if observation is None:
        return
    try:
        record_production(observation, invocation)
    except Exception as error:  # noqa: BLE001 -- diagnostics cannot fail a run
        _unavailable(invocation_path.stem, error)
