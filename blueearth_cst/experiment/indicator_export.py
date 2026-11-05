"""Publish each selected named indicator set after a WF4 invocation.

Sets commit independently: a set whose immutable indicator set is complete is
published even when another set failed, and a failed set keeps its prior
publication. This is per-set completion, not an all-set transaction.
"""

from pathlib import Path

from blueearth_cst.experiment import indicator_sets
from blueearth_cst.experiment.console_reporting import (
    record_indicator_detail,
    result_counts,
    unavailable_summary,
    undefined_result_summary,
)
from blueearth_cst.experiment.content_identity import read_canonical_json
from blueearth_cst.experiment.indicator_plan import (
    current_indicator_request,
    current_indicator_requests,
    indicator_anchor,
    verify_indicator_plan,
)
from blueearth_cst.experiment.named_export import (
    NamedExportError,
    legacy_flat_export,
    publish_named_set,
    read_current_publication,
)
from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2
from blueearth_cst.shared.snake_utils import listed, plural, resolve_water_year_start


def publish_selected_sets(
    root: Path, project: dict, settings: dict, selection
) -> list[dict]:
    """Publish every selected set whose indicator set is ready; report each outcome.

    Returns one record per set: ``status`` is ``current``, ``partial`` (published
    with unavailable indicators; only under ``indicator_evaluation: partial``)
    or ``failed``, with the indicator-set and publication ids, the failure
    reason, and any prior publication a failed set keeps (a mixed-age export).
    """
    root = Path(root).resolve()

    def retained_variables():
        request = read_simulation_intent_v2(root)["documents"]["response_request"]
        return [item["variable"] for item in request["variables"]]

    month = resolve_water_year_start(
        (project.get("climate") or {}).get("water_year_start")
    )
    sets = indicator_sets.selected_sets(
        project,
        settings,
        selection,
        simulation_path=settings["operation"] == "simulate-and-indicators",
        retained_variables=retained_variables,
        legacy_context="export",
    )
    # Each publication phase observes live code/environment once for all sets.
    # If one request is invalid, retain the independent per-set failure path.
    try:
        requests = current_indicator_requests(
            root,
            [indicator_sets.calculation_records(item.indicators) for item in sets],
            indicator_anchor(project),
        )
    except (OSError, KeyError, ValueError):
        requests = None
    outcomes = []
    for index, item in enumerate(sets):
        try:
            previous = read_current_publication(root, item.set_id)["publication_id"]
        except (OSError, KeyError, ValueError, NamedExportError):
            previous = None
        try:
            request = (
                requests[index]
                if requests is not None
                else current_indicator_request(
                    root,
                    indicator_sets.calculation_records(item.indicators),
                    indicator_anchor(project),
                )
            )
            plan = verify_indicator_plan(
                root, request, _observed_code=request["code_inventory"]
            )
            manifest = read_canonical_json(Path(plan["targets"]["manifest"]))
            counts = result_counts(root, manifest)
            expected_undefined, undefined_indicators = undefined_result_summary(
                root, manifest
            )
            unavailable = unavailable_summary(manifest)
            if unavailable and not selection.partial:
                raise ValueError(
                    f"strict evaluation refuses unavailable indicators: {listed(unavailable)}"
                )
            record = publish_named_set(
                root, item, Path(plan["targets"]["manifest"]), water_year_start=month
            )
            outcomes.append(
                {
                    "set_id": item.set_id,
                    "status": "partial" if unavailable else "current",
                    "indicator_set_id": record["indicator_set_id"],
                    "publication_id": record["publication_id"],
                    "reused": previous == record["publication_id"],
                    "available_results": counts[0],
                    "missing_results": counts[1],
                    "expected_undefined_results": expected_undefined,
                    "undefined_indicators": undefined_indicators,
                    "unavailable_results": counts[2],
                    "unavailable": unavailable,
                }
            )
        except (OSError, KeyError, ValueError) as exc:
            try:
                prior = read_current_publication(root, item.set_id)["publication_id"]
            except (OSError, KeyError, ValueError, NamedExportError):
                prior = None
            outcomes.append(
                {
                    "set_id": item.set_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "prior_publication_id": prior,
                }
            )
    return outcomes


def report_publication(root: Path, outcomes) -> bool:
    """Log one row per set and any preserved legacy flat export; True if none failed."""
    from blueearth_cst.shared.run_log_core import log_row

    legacy = legacy_flat_export(root)
    if legacy:
        log_row(
            f"Earlier results kept at {legacy[0]}; updated results are saved separately "
            "under results/indicator_sets/<set-id>/",
            module="indicators",
        )
    for outcome in outcomes:
        if outcome["status"] in ("current", "partial"):
            action = (
                "results unchanged; reusing saved results"
                if outcome.get("reused")
                else "results saved"
            )
            destination = f"<experiment>/results/indicator_sets/{outcome['set_id']}/"
            log_row(
                f"{outcome['set_id']}: {action} → {destination}",
                module="indicators",
            )
            expected = outcome.get("expected_undefined_results", 0)
            if expected:
                log_row(
                    f"{outcome['set_id']}: {plural(expected, 'zero-flow timing result')} undefined; "
                    "no zero-flow event in one or more contributing simulations; "
                    f"{plural(outcome['available_results'], 'result')} available",
                    module="indicators",
                )
            unexpected = outcome.get("missing_results", 0) - expected
            if unexpected:
                affected = listed(outcome.get("undefined_indicators", []))
                log_row(
                    f"{outcome['set_id']}: {plural(unexpected, 'result')} undefined; "
                    f"{plural(outcome['available_results'], 'result')} available; "
                    f"affected indicators: {affected or 'see result tables'}; "
                    "check indicator definitions and coverage evidence",
                    module="indicators",
                    level="WARNING",
                )
            if outcome["status"] == "partial":
                log_row(
                    f"{outcome['set_id']}: partial results; unavailable: "
                    f"{listed(outcome['unavailable'])}; see the availability column "
                    "of indicators.csv",
                    module="indicators",
                    level="WARNING",
                )
            record_indicator_detail(
                root,
                f"{outcome['set_id']}; indicator set {outcome['indicator_set_id']}; "
                f"publication {outcome['publication_id']}; "
                f"available results={outcome.get('available_results')}; "
                f"missing or invalid results={outcome.get('missing_results')}",
            )
        else:
            kept = (
                "previous results kept; they have not been updated"
                if outcome["prior_publication_id"]
                else "no saved results available"
            )
            log_row(
                f"{outcome['set_id']}: could not update results; {kept}; {outcome['reason'].splitlines()[0]}",
                module="indicators",
                level="ERROR",
            )
    # A partial set is only ever published under indicator_evaluation: partial.
    return all(outcome["status"] in ("current", "partial") for outcome in outcomes)
