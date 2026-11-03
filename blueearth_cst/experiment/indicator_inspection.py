"""Separate saved-artifact integrity from eligible fresh numerical checks.

Importing this module never imports the current calculation registries. Hash
consistency detects accidental corruption, not coordinated forgery.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.indicator_artifacts import read_indicator_set
from blueearth_cst.experiment.legacy_indicator_names import indicator_names


class NumericalRecheckUnavailable(ValueError):
    """Saved execution evidence cannot establish numerical eligibility."""


def inspect_indicator_set(
    experiment_root, manifest_path, *, recheck=False, traceback=False
) -> dict:
    """Inspect saved artifacts; optionally require a fresh numerical recheck.

    Old publications remain inspectable without their inputs. Numerical work
    requires the /5 execution context and exact dependency/source identities.
    New receipts are written beside the engine objects, never over old evidence.
    """
    root = Path(experiment_root).resolve()
    marker = Path(manifest_path).absolute()
    manifest = read_indicator_set(root, marker, integrity_only=True)
    binding = {
        "indicator_set_id": manifest["indicator_set_id"],
        "indicators_manifest_sha256": manifest["indicators_manifest_sha256"],
    }
    historical = []
    historical_issues = []
    for path in sorted((marker.parent / "verification").glob("*.json")):
        try:
            receipt = read_canonical_json(path)
            if marker.name == "metrics.json":
                receipt = indicator_names(receipt)
        except (OSError, ValueError) as exc:
            historical_issues.append({"path": path.name, "reason": str(exc)})
            continue
        if (
            isinstance(receipt, dict)
            and receipt.get("publication") == binding
            and receipt.get("schema_version") == "indicator-verification/1"
            and receipt.get("kind") == "fresh numerical recheck"
            and receipt.get("status") == "passed"
            and receipt.get("checker_sources")
            and receipt.get("runtime")
        ):
            historical.append(
                {"path": path.name, "label": "historical", "receipt": receipt}
            )
    report = {
        "schema_version": "indicator-inspection/1",
        "publication": binding,
        "integrity": {
            "status": "passed",
            "basis": "saved artifact hashes and structure",
        },
        "historical_checks": historical,
        "historical_check_issues": historical_issues,
        "numerical": {
            "status": "skipped",
            "reason": "integrity-only inspection requested",
        },
        "authenticity": "hash consistency does not establish authenticity",
    }
    if not recheck:
        return report
    if marker.name == "metrics.json":
        report["numerical"] = {
            "status": "unavailable",
            "reason": "legacy publication predates the indicator source identities; saved integrity checks passed",
        }
        return report
    context = manifest.get("execution_context")
    if context is None:
        report["numerical"]["reason"] = (
            "legacy publication has no complete execution context; numerical eligibility unestablished"
        )
        return report
    try:
        _recheck(root, marker, manifest, context)
    except (OSError, NumericalRecheckUnavailable) as exc:
        if traceback:
            raise
        report["numerical"] = {"status": "unavailable", "reason": str(exc)}
        return report
    except (ValueError, AssertionError) as exc:
        if traceback:
            raise
        report["numerical"] = {"status": "failed", "reason": str(exc)}
        return report
    checker_root = Path(__file__).resolve().parents[1] / "indicators"
    checkers = [
        Path(__file__),
        checker_root / "checking.py",
        *sorted(checker_root.rglob("checks.py")),
    ]
    receipt = {
        "schema_version": "indicator-verification/1",
        "publication": binding,
        "status": "passed",
        "kind": "fresh numerical recheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checker_sources": [
            {
                "path": p.relative_to(checker_root.parent).as_posix(),
                "sha256": hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            }
            for p in checkers
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    directory = marker.parent / "verification"
    directory.mkdir(exist_ok=True)
    destination = directory / f"verification-{uuid.uuid4().hex}.json"
    atomic_record(destination, receipt)
    report["numerical"] = {"status": "passed", "receipt": str(destination)}
    return report


def _recheck(root, marker, manifest, context):
    """Check eligibility before loading neutral data and independent checkers."""
    request = context["request"]
    definition = context["indicator_definition"]
    identity = manifest["identity_projection"]
    if content_sha256(definition) != identity["indicator_definition_sha256"]:
        raise NumericalRecheckUnavailable(
            "saved execution definitions differ from publication identity"
        )
    if (
        request["declarations"] != manifest["declarations"]
        or request["calculations"] != manifest["calculations"]
    ):
        raise NumericalRecheckUnavailable(
            "saved execution declarations differ from publication"
        )
    repo = Path(__file__).resolve().parents[2]
    for source in request["code_inventory"]:
        path = (repo / source["path"]).resolve()
        if not path.is_relative_to(repo):
            raise NumericalRecheckUnavailable(
                "execution source path escapes repository"
            )
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != source["sha256"]:
            raise NumericalRecheckUnavailable(
                f"execution source changed: {source['path']}"
            )
    # Import execution code only after saved-artifact inspection and source checks.
    from blueearth_cst.experiment.indicator_plan import (
        _plain,
        build_indicator_plan,
        reduce_indicator_plan,
        resolve_indicator_environment,
    )

    if resolve_indicator_environment() != request["indicator_environment"]:
        raise NumericalRecheckUnavailable(
            "recorded indicator dependency versions differ from current runtime"
        )
    try:
        read_indicator_set(root, marker)
        plan = build_indicator_plan(root, request)
    except (OSError, ValueError) as exc:
        raise NumericalRecheckUnavailable(
            f"current retained inputs are unavailable or changed: {exc}"
        ) from exc
    if plan["indicator_set_id"] != manifest["indicator_set_id"]:
        raise NumericalRecheckUnavailable(
            "current inputs or definitions differ from publication identity"
        )
    diagnostics = []
    tables, _, _ = reduce_indicator_plan(
        root, plan, numerical_checks=True, diagnostics=diagnostics
    )
    saved_diagnostics = read_canonical_json(
        marker.parent / manifest["diagnostics"]["path"]
    )
    if content_sha256(saved_diagnostics) != content_sha256(
        {
            "schema_version": "indicator-diagnostics/1",
            "results": _plain(diagnostics),
        }
    ):
        raise ValueError("fresh numerical diagnostics differ from saved publication")
    for table in manifest["tables"]:
        with (root / table["file"]["path"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            saved = list(csv.DictReader(handle))
        scope = table["scope"]
        current = sorted(
            [
                {
                    "indicator": row["indicator"],
                    "location": row["location"],
                    f"{scope}_id": row["run_group_id"],
                    "value": row["value"],
                    "status": row["status"],
                    "reason": row["reason"],
                }
                for row in tables[table["token"]]
                if row["run_group_id"].startswith(f"{scope}_")
            ],
            key=lambda row: (row["indicator"], row["location"], row[f"{scope}_id"]),
        )
        if saved != current:
            raise ValueError(f"fresh numerical values differ: {table['file']['path']}")
