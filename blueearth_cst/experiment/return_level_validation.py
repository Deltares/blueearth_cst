"""Shipped GF15 benchmark report and the closed validation declaration it backs.

Accepted design D4-D5. The report is loaded as a package resource from the
importable source tree, so this module works from any checkout or relocated copy
that carries the package and its asset. It never consults `dev/`, an absolute
source path, session scratch or the network.

Nothing here imports the estimator or its numerical dependencies: a reader of a
retained indicator set must not need `lmoments3` installed (D6), and the declaration
is pure data.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from blueearth_cst.experiment.content_identity import canonical_json_bytes

SCHEMA_VERSION = "return-level-validation/1"
REPORT_SCHEMA_VERSION = "gf15-benchmark-report/1"
ESTIMATOR_ID = "gf15-lmoments-c/1"
POLICY_ID = "gf15-accuracy-8x-v1"

#: Name the copied report takes inside a published indicator set.
REPORT_FILENAME = "return_level_benchmark.json"

#: Name of the packaged asset within `blueearth_cst.experiment.data`.
REPORT_RESOURCE = "gf15-accuracy-8x-v1.json"

#: The exact retained validation record of every pre-C indicator set. D6 dispatches
#: on this value and it must never be written into a new request.
LEGACY_VALIDATION = {"status": "provisional_operational", "benchmark": "not assessed"}

#: Closed field set of `return-level-validation/1`. A declaration carrying more
#: or fewer keys is refused rather than defaulted.
DECLARATION_FIELDS = frozenset(
    {
        "schema_version",
        "estimator_id",
        "policy_id",
        "screening_ratio",
        "screening_floor",
        "screening_policy_status",
        "benchmark_status",
        "benchmark_record",
        "tested_domain",
        "criteria",
        "application_scope",
        "relative_error_near_zero",
        "actual_bundle_applicability",
        "evidence_use",
        "screening_policy_validated",
        "fit_uncertainty",
    }
)

_DOMAIN_FIELDS = (
    "shapes_c",
    "counts",
    "probabilities",
    "ratios",
    "relative_qualified_ratios",
    "draws_per_base_cell",
)


class ReturnLevelValidationError(ValueError):
    """The shipped report or a declaration built from it is missing or altered.

    Callers at an indicator-set boundary translate this into `ImmutableIndicatorSetError`;
    it is raised separately here so this module stays independent of `indicator_plan`.
    """


def load_report_bytes() -> bytes:
    """Read the packaged report exactly as shipped, without interpreting it."""
    resource = files("blueearth_cst.experiment").joinpath("data", REPORT_RESOURCE)
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise ReturnLevelValidationError(
            f"shipped return-level benchmark {REPORT_RESOURCE} is unavailable: {error}"
        ) from error


def report_digest(data: bytes) -> str:
    """Digest the report bytes themselves; there is no digest inside the report."""
    return hashlib.sha256(data).hexdigest()


def decode_report(data: bytes) -> dict[str, Any]:
    """Decode and structurally check a report, installed or copied.

    Checks only what makes the report usable as evidence for a declaration: its
    schema and identity, and that the duplicated criteria agree with the embedded
    source they were packaged from. It deliberately does not re-derive the
    benchmark.
    """
    try:
        report = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReturnLevelValidationError(
            f"return-level benchmark is not valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(report, dict):
        raise ReturnLevelValidationError("return-level benchmark is not a JSON object")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReturnLevelValidationError(
            f"unknown return-level benchmark schema {report.get('schema_version')!r}"
        )
    if (
        report.get("estimator_id") != ESTIMATOR_ID
        or report.get("policy_id") != POLICY_ID
    ):
        raise ReturnLevelValidationError(
            "return-level benchmark does not identify the accepted estimator and policy"
        )
    for field in ("criteria", "tested_domain", "embedded_sources", "verdict"):
        if field not in report:
            raise ReturnLevelValidationError(
                f"return-level benchmark is missing {field!r}"
            )

    embedded = report["embedded_sources"]
    if not isinstance(embedded, list) or not embedded:
        raise ReturnLevelValidationError("return-level benchmark embeds no sources")
    for item in embedded:
        if not isinstance(item, dict) or set(item) != {"source_path", "sha256", "text"}:
            raise ReturnLevelValidationError("embedded source record is malformed")
        if hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item["sha256"]:
            raise ReturnLevelValidationError(
                f"embedded source {item['source_path']} does not match its digest"
            )

    criteria_source = [
        item for item in embedded if item["source_path"].endswith("/criteria.json")
    ]
    if len(criteria_source) != 1:
        raise ReturnLevelValidationError(
            "return-level benchmark does not embed exactly one criteria source"
        )
    if json.loads(criteria_source[0]["text"]) != report["criteria"]:
        raise ReturnLevelValidationError(
            "return-level benchmark criteria differ from their embedded source"
        )
    return report


def build_declaration(data: bytes | None = None) -> dict[str, Any]:
    """Build the closed `return-level-validation/1` record from shipped evidence.

    The declaration is derived from the report, never from user annotation. Pass
    `data` to declare against specific bytes (the copy inside an indicator set);
    omit it to declare against the installed asset.
    """
    payload = load_report_bytes() if data is None else data
    report = decode_report(payload)
    domain = report["tested_domain"]
    missing = [field for field in _DOMAIN_FIELDS if field not in domain]
    if missing:
        raise ReturnLevelValidationError(
            f"return-level benchmark domain is missing {missing}"
        )
    declaration = {
        "schema_version": SCHEMA_VERSION,
        "estimator_id": ESTIMATOR_ID,
        "policy_id": POLICY_ID,
        "screening_ratio": 1.0,
        "screening_floor": 10,
        "screening_policy_status": "provisional_operational",
        "benchmark_status": "reviewed_bounded",
        "benchmark_record": {
            "path": REPORT_FILENAME,
            "sha256": report_digest(payload),
        },
        "tested_domain": {field: domain[field] for field in _DOMAIN_FIELDS},
        "criteria": report["criteria"],
        "application_scope": "operational_screen_only",
        "relative_error_near_zero": "unvalidated",
        "actual_bundle_applicability": "unestablished",
        "evidence_use": "post_results_development_rescore",
        "screening_policy_validated": False,
        "fit_uncertainty": "point_estimates_only",
    }
    # Fail here rather than at an indicator-set boundary if the closed set drifts.
    canonical_json_bytes(declaration)
    if set(declaration) != DECLARATION_FIELDS:
        raise ReturnLevelValidationError(
            "built declaration is not the closed field set"
        )
    return declaration


def is_legacy(validation: Any) -> bool:
    """Report whether a retained record is the exact pre-C legacy validation."""
    return validation == LEGACY_VALIDATION


def verify_declaration(validation: Any, report_bytes: bytes) -> dict[str, Any]:
    """Check a retained declaration against the report bytes retained beside it.

    This is the reader's check (D5, D6). It uses only the copied report, so a
    historical set stays readable when the installed asset is a different
    version, or absent entirely.
    """
    if not isinstance(validation, dict):
        raise ReturnLevelValidationError("return-level validation is not a record")
    if validation.get("schema_version") != SCHEMA_VERSION:
        raise ReturnLevelValidationError(
            f"unknown return-level validation schema {validation.get('schema_version')!r}"
        )
    if set(validation) != DECLARATION_FIELDS:
        raise ReturnLevelValidationError(
            "return-level validation is not the closed declaration field set"
        )
    record = validation.get("benchmark_record")
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ReturnLevelValidationError("benchmark_record is malformed")
    if record["path"] != REPORT_FILENAME:
        raise ReturnLevelValidationError(
            f"benchmark_record names {record['path']!r}, not {REPORT_FILENAME!r}"
        )
    digest = report_digest(report_bytes)
    if record["sha256"] != digest:
        raise ReturnLevelValidationError(
            f"retained benchmark digest {record['sha256']} differs from the "
            f"retained report bytes {digest}"
        )
    report = decode_report(report_bytes)
    if validation["criteria"] != report["criteria"]:
        raise ReturnLevelValidationError(
            "retained declaration criteria differ from the retained report"
        )
    rebuilt = build_declaration(report_bytes)
    if rebuilt != validation:
        raise ReturnLevelValidationError(
            "retained declaration differs from the declaration its report supports"
        )
    return report
