"""Declared method revisions and the code pins that guard them.

A producer's identity carries a hand-bumped method revision, never its code
bytes. The pin file beside this module stores, per producer, that revision with
the digest of the code closure it was declared for. ``method_revision`` refuses
a closure that differs from its pin, so every identity is computed under a
revision someone confirmed for the code that is actually running. Re-record a
pin with ``dev/scripts/record_method_pins.py``; ``--bump`` when produced values
can change, ``--no-numerical-change`` otherwise.

The pin is JSON, so no closure (which follows ``.py`` imports and explicit
entries only) can contain it.
"""

import json
from functools import cache
from pathlib import Path
from typing import Any

from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
    repository_code_inventory,
)

PIN_PATH = Path(__file__).with_name("method_revisions.json")
PIN_SCHEMA = "method-revisions/1"
REPO_ROOT = Path(__file__).resolve().parents[2]

STOCHASTIC = "stochastic-collection"
DELTA = "delta-collection"
SIMULATION = "wf4-simulation"
INDICATORS = "indicator-set"
PRODUCERS = (STOCHASTIC, DELTA, SIMULATION, INDICATORS)
RECORD_KINDS = ("bump", "no-numerical-change")
RECORD_COMMAND = "python dev/scripts/record_method_pins.py"

# The delta producer and its WF2 adapter; imports are followed. Logging,
# archiving and configuration composition stop traversal as in WF3, because
# resolved settings reach the identity by value.
DELTA_CODE_ENTRIES = (
    "blueearth_cst/experiment/delta_scenarios.py",
    "blueearth_cst/experiment/delta_wf2.py",
)
DELTA_INVENTORY_EXCLUDED = (
    "blueearth_cst/shared/advanced_settings.py",
    "blueearth_cst/shared/config_composition.py",
    "blueearth_cst/shared/invocation_history.py",
    "blueearth_cst/shared/provenance.py",
    "blueearth_cst/shared/run_log_core.py",
    "blueearth_cst/shared/windows_job.py",
    "blueearth_cst/shared/workflow_archive_launch.py",
    "blueearth_cst/shared/workflow_config_snapshot.py",
)


class MethodPinMismatch(RuntimeError):
    """The running code differs from the code its method revision was pinned for."""


def producer_closure(producer: str, repo: Path = REPO_ROOT) -> list[dict[str, str]]:
    """Inventory the code a producer's method revision stands for."""
    if producer == STOCHASTIC:
        from blueearth_cst.experiment.generation_plan import (
            GENERATION_CODE_ENTRIES,
            GENERATION_INVENTORY_EXCLUDED,
        )

        return repository_code_inventory(
            repo, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
        )
    if producer == DELTA:
        return repository_code_inventory(
            repo, DELTA_CODE_ENTRIES, DELTA_INVENTORY_EXCLUDED
        )
    if producer == SIMULATION:
        from blueearth_cst.experiment.simulation_code import simulation_code_inventory

        return simulation_code_inventory(repo)
    if producer == INDICATORS:
        # Every installed set: one revision covers the engine and all sets.
        from blueearth_cst.experiment.indicator_plan import _indicator_code_inventory
        from blueearth_cst.indicators.loader import available_sets

        return _indicator_code_inventory(available_sets())
    raise ValueError(f"unknown producer {producer!r}; expected one of {PRODUCERS}")


def read_pins(path: Path = PIN_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != PIN_SCHEMA:
        raise ValueError(f"{path}: expected {PIN_SCHEMA}")
    if set(document["producers"]) != set(PRODUCERS):
        raise ValueError(f"{path}: pins must cover exactly {sorted(PRODUCERS)}")
    return document


def closure_drift(
    pin: dict[str, Any], files: list[dict[str, str]]
) -> dict[str, list[str]]:
    """Name the added, removed and changed paths between a pin and a closure."""
    pinned = {item["path"]: item["sha256"] for item in pin["files"]}
    live = {item["path"]: item["sha256"] for item in files}
    return {
        "added": sorted(set(live) - set(pinned)),
        "removed": sorted(set(pinned) - set(live)),
        "changed": sorted(
            path for path in set(live) & set(pinned) if live[path] != pinned[path]
        ),
    }


def drift_message(producer: str, drift: dict[str, list[str]]) -> str:
    listed = "; ".join(
        f"{kind}: {', '.join(paths)}" for kind, paths in drift.items() if paths
    )
    return (
        f"{producer} code differs from its method-revision pin ({listed}). Record "
        f'the change: {RECORD_COMMAND} --producer {producer} --bump "<reason>" if '
        'produced values can change, otherwise --no-numerical-change "<reason>"'
    )


@cache
def method_revision(producer: str) -> dict[str, Any]:
    """Return the producer's identity record after checking the live code pin.

    Identity builders take the revision only from here, so no identity is ever
    computed under an unconfirmed revision. Checked once per process.
    """
    pin = read_pins(PIN_PATH)["producers"][producer]
    files = producer_closure(producer)
    if content_sha256(files) != pin["closure_sha256"]:
        raise MethodPinMismatch(drift_message(producer, closure_drift(pin, files)))
    return {"producer": producer, "revision": pin["revision"]}


def record_pin(
    producer: str, kind: str, reason: str, path: Path = PIN_PATH
) -> dict[str, Any]:
    """Re-record one producer's pin; ``bump`` also increments its revision."""
    if kind not in RECORD_KINDS:
        raise ValueError(f"kind must be one of {RECORD_KINDS}")
    if not reason.strip():
        raise ValueError("a record needs a reason")
    document = read_pins(path)
    pin = document["producers"][producer]
    files = producer_closure(producer)
    document["producers"][producer] = {
        "revision": pin["revision"] + (kind == "bump"),
        "closure_sha256": content_sha256(files),
        "files": files,
        "last_record": {"kind": kind, "reason": reason.strip()},
    }
    write_pins(document, path)
    return document["producers"][producer]


def write_pins(document: dict[str, Any], path: Path = PIN_PATH) -> None:
    # Indented for reviewable diffs; key order is canonical.
    path.write_text(
        json.dumps(json.loads(canonical_json_bytes(document)), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
