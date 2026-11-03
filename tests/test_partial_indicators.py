"""Partial indicator evaluation: availability, explicit unavailable rows, recovery."""

import csv
import os
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment import indicator_sets
from blueearth_cst.experiment.console_reporting import (
    prepare_indicator_plan,
    result_counts,
)
from blueearth_cst.experiment.indicator_plan import (
    build_indicator_plan,
    current_indicator_request,
    publish_indicator_set,
    read_indicator_set,
    split_availability,
)
from blueearth_cst.experiment.indicator_sets import IndicatorSetError, SetSource
from blueearth_cst.experiment.simulation_runner import check_required_variables
from tests._v2_experiment import FIXTURE_ENVIRONMENT, build_v2_experiment
from tests.test_crop_water_integration import _responses

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


REASON = "missing_parameter:thresholds"


def _crop_water(path, thresholds):
    overrides = {"summer_low_flow_days": {"season": "summer", "thresholds": thresholds}}
    text = yaml.safe_dump(
        {
            "schema_version": 1,
            "id": "crop-water",
            "title": "Crop-water",
            "description": "Partial evaluation probe",
            "selection": {"registry": {"set": "crop-water", "overrides": overrides}},
        }
    )
    return indicator_sets.resolve_set(SetSource("crop-water", path, False, text))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from blueearth_cst.experiment import indicator_plan

    responses, thresholds = _responses(locations=("9", "10"))
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    return build_v2_experiment(tmp_path, responses), thresholds


def _publish(root, resolved):
    request = current_indicator_request(
        root, indicator_sets.calculation_records(resolved.indicators), "YE-SEP"
    )
    plan = build_indicator_plan(root, request)
    return request, plan, publish_indicator_set(root, plan)


def _rows(root, manifest, indicator):
    rows = []
    for table in manifest["tables"]:
        with (root / table["file"]["path"]).open(encoding="utf-8", newline="") as f:
            rows += [row for row in csv.DictReader(f) if row["indicator"] == indicator]
    return rows


def test_missing_location_qcrit_publishes_explicit_unavailable_rows(experiment):
    built, thresholds = experiment
    root = built.root
    resolved = _crop_water(root / "crop-water.yml", thresholds)
    request, plan, manifest = _publish(root, resolved)
    assert request["unavailable_results"] == [
        {"indicator": "summer_low_flow_days", "location": "10", "reason": REASON}
    ]
    assert manifest["schema_version"] == "indicator-set/6"
    rows = _rows(root, manifest, "summer_low_flow_days")
    by_location = {}
    for row in rows:
        by_location.setdefault(row["location"], set()).add(
            (row["status"], row["reason"], row["value"])
        )
    # Location 10 is unavailable at every run and group; 9 keeps its result.
    assert by_location["10"] == {("unavailable", REASON, "")}
    assert {status for status, _, _ in by_location["9"]} == {"available"}
    assert len(rows) == 2 * len(plan["units"])
    assert read_indicator_set(root, plan["targets"]["manifest"]) == manifest
    assert result_counts(root, manifest)[2] == len(plan["units"])
    # Strict evaluation refuses the same request at the plan job.
    with pytest.raises(ValueError, match="summer_low_flow_days at 1 location"):
        prepare_indicator_plan(root, request, [resolved], "strict")


def test_supplying_qcrit_later_publishes_a_new_complete_set(experiment):
    built, thresholds = experiment
    root = built.root
    _, partial_plan, partial = _publish(
        root, _crop_water(root / "crop-water.yml", thresholds)
    )
    completed = {**thresholds, "10": thresholds["9"]}
    request, plan, complete = _publish(
        root, _crop_water(root / "crop-water.yml", completed)
    )
    assert request["unavailable_results"] == []
    assert complete["indicator_set_id"] != partial["indicator_set_id"]
    assert all(
        row["status"] == "available"
        for row in _rows(root, complete, "summer_low_flow_days")
    )
    # The earlier immutable partial set stays readable beside the new one.
    assert read_indicator_set(root, partial_plan["targets"]["manifest"]) == partial


def test_empty_qcrit_is_refused_before_simulation_only_in_strict_mode(tmp_path):
    resolved = _crop_water(tmp_path / "crop-water.yml", {})
    variables = {"q", "precip", "pet", "aet", "gwr"}
    with pytest.raises(ValueError, match="lacks missing_parameter:thresholds"):
        check_required_variables([resolved], variables, "outvars")
    gaps = check_required_variables(
        [resolved], variables, "outvars", evaluation="partial"
    )
    assert [gap["text"] for gap in gaps] == [
        "crop-water:summer_low_flow_days (seasonal_low_flow_days/1) lacks "
        f"{REASON} at every location"
    ]


def test_absent_response_drops_the_indicator_with_a_new_simulation_reason(tmp_path):
    resolved = _crop_water(tmp_path / "crop-water.yml", {})
    records = indicator_sets.calculation_records(resolved.indicators)
    request = {
        "variables": [
            {"variable": variable, "locations": ["9"]}
            for variable in ("q", "precip", "aet", "gwr")
        ]
    }
    available, dropped, results = split_availability(records, request)
    names = {item["calculation"]["indicator"]: item["reasons"] for item in dropped}
    assert names == {
        f"{season}_climatic_deficit": ["missing_response:pet"]
        for season in ("winter", "summer")
    } | {
        f"{season}_{kind}": ["missing_response:pet"]
        for season in ("winter", "summer")
        for kind in ("relative_et_deficit", "et_limitation_days")
    }
    assert len(available) == 4
    assert results == [
        {"indicator": "summer_low_flow_days", "location": "9", "reason": REASON}
    ]
    with pytest.raises(ValueError, match="needs response pet"):
        check_required_variables(
            [resolved], {"q", "precip", "aet", "gwr"}, "the retained responses"
        )


def test_malformed_supplied_qcrit_still_fails(tmp_path):
    with pytest.raises(ValueError, match="thresholds: got"):
        _crop_water(tmp_path / "crop-water.yml", {"9": {"value": 1.0}})


def test_evaluation_mode_is_validated_and_snapshotted(tmp_path):
    settings = {"indicator_sets": ["crop-water"], "indicator_evaluation": "partial"}
    selection = indicator_sets.parse_selection(settings, tmp_path / "wf4.yml")
    assert selection.partial
    path = indicator_sets.write_selection_snapshot(selection, tmp_path / "s.json")
    assert indicator_sets.read_selection_snapshot(path).evaluation == "partial"
    assert indicator_sets.parse_selection({}, tmp_path / "wf4.yml").evaluation == (
        "strict"
    )
    with pytest.raises(IndicatorSetError, match="indicator_evaluation"):
        indicator_sets.parse_selection(
            {"indicator_evaluation": "lenient"}, Path(tmp_path / "wf4.yml")
        )


@pytest.mark.parametrize(
    "execute",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name != "nt",
                reason="run_project_child refuses project writers off Windows",
            ),
        ),
    ],
    ids=["preflight-and-dry-run", "publication"],
)
def test_cli_partial_run_succeeds_with_warning_and_strict_refuses(experiment, execute):
    import subprocess
    import sys

    import pandas as pd

    built, thresholds = experiment
    source = built.config.parent / "crop-water.yml"
    source.write_text(_crop_water(source, thresholds).source.text, encoding="utf-8")
    workflow = built.config.parent / "simulate_system.yml"
    environment = os.environ.copy()
    environment.pop("SNAKEMAKE_WORKFLOW_PROFILE", None)

    def run(evaluation, *extra):
        workflow.write_text(
            yaml.safe_dump(
                {
                    "experiment_name": built.root.name,
                    "operations": ["indicators"],
                    "indicator_sets": ["crop-water"],
                    "indicator_set_definitions": {
                        "crop-water": {"config_path": source.name}
                    },
                    "indicator_evaluation": evaluation,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                "scripts/simulate_system.py",
                "--config",
                str(built.config),
                "--cores",
                "1",
                *extra,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=1200,
        )

    strict = run("strict")
    assert strict.returncode != 0
    assert "summer_low_flow_days" in strict.stdout + strict.stderr
    assert "indicator_evaluation: partial" in strict.stdout + strict.stderr
    partial = run("partial", *(() if execute else ("--dry-run",)))
    assert partial.returncode == 0, partial.stdout + partial.stderr
    if not execute:
        assert not (built.root / "results/indicator_sets/crop-water").exists()
        return
    assert "partial results; unavailable: summer_low_flow_days" in partial.stdout
    dictionary = pd.read_csv(
        built.root / "results/indicator_sets/crop-water/indicators.csv"
    )
    availability = dict(zip(dictionary.indicator, dictionary.availability))
    assert availability.pop("summer_low_flow_days") == "partial"
    assert set(availability.values()) == {"available"}


def test_a_set_with_nothing_calculable_publishes_only_unavailable_entries(
    tmp_path, monkeypatch
):
    from blueearth_cst.experiment import indicator_plan

    responses, _ = _responses(without_pet=True)
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    root = build_v2_experiment(tmp_path, responses).root
    resolved = _crop_water(root / "crop-water.yml", {})
    settings = {
        "indicator_sets": ["crop-water"],
        "indicator_selection": {
            "crop-water": {"include": ["summer_relative_et_deficit"]}
        },
    }
    narrowed = indicator_sets.apply_choice(
        resolved, indicator_sets.parse_selection(settings, tmp_path / "w.yml").choices
    )
    request, plan, manifest = _publish(root, narrowed)
    assert request["calculations"] == [] and request["code_inventory"] == []
    assert manifest["tables"] == []
    assert manifest["unavailable_calculations"][0]["reasons"] == [
        "missing_response:pet"
    ]
    assert read_indicator_set(root, plan["targets"]["manifest"]) == manifest
