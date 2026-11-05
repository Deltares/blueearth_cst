"""Independent-module boundary, actionable validation and local authoring journeys."""

import ast
import importlib
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from statistics import median as sample_median

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.indicators import (
    DiagnosticTable,
    IndicatorResult,
    ResponseSeries,
    check,
    declarations,
    evaluate,
    loader,
    resolve,
    sets,
)
from blueearth_cst.indicators.extremes import (
    InsufficientReturnLevelBlocks,
    ReturnLevelProbabilityNotCovered,
)
from blueearth_cst.indicators.validation import validate_result


def response(values, *, start="2000-01-01", run="r1"):
    values = np.asarray(values, dtype=float)
    return ResponseSeries(
        run,
        "q",
        "gauge",
        0,
        pd.date_range(start, periods=len(values)),
        "standard",
        timedelta(days=1),
        "interval_end",
        "m3 s-1",
        values,
        np.isnan(values),
    )


def test_local_rate_declarations_preserve_output_units():
    rates = [
        item
        for item in declarations("flow_regime")
        if item.statistic in {"rise_rate", "fall_rate"}
    ]
    assert len(rates) == 2
    assert all(loader.registration(item).units == "m3 s-1 d-1" for item in rates)
    assert all(item.required_responses.units == "m3 s-1" for item in rates)


@pytest.mark.parametrize("set_id", ["default", "flow_regime", "flood_risk"])
def test_unimplemented_method_check_cannot_fall_back_to_another_formula(set_id):
    indicator = replace(
        declarations(set_id)[0], calculation_method="new_method_without_check/1"
    )
    result = IndicatorResult(pd.Series({"gauge": 2.0}))
    with pytest.raises(ValueError, match=f"{set_id}/.*missing independent check"):
        check(indicator, [response([1, 2, 3])], result, anchor="YE-DEC")


def test_independent_module_has_no_workflow_imports():
    """Guard actual import dependencies rather than documentation keywords."""
    root = Path(loader.__file__).parent
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                name.startswith(
                    ("blueearth_cst.experiment", "snakemake", "blueearth_cst.shared")
                )
                for name in names
            ), source
    program = """
import sys
from datetime import timedelta
import numpy as np
import pandas as pd
from blueearth_cst.indicators import ResponseSeries, resolve, evaluate
values = np.array([1., 2., 3., 4.])
series = [ResponseSeries('member', 'q', 'gauge', 0,
    pd.date_range('2001-01-01', periods=4), 'standard', timedelta(days=1),
    'interval_end', 'm3 s-1', values, np.isnan(values))]
indicator = resolve('default', 'annual_summary/1', {'variable': 'q', 'statistic': 'mean'})
assert evaluate(indicator, series, anchor='YE-SEP').values['gauge'] == 2.5
assert not any(name.startswith(('blueearth_cst.experiment', 'snakemake')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.parametrize("anchor", ["YE-DEC", "YE-SEP"])
def test_plain_python_native_missingness_and_independent_checks(anchor):
    values = np.arange(800, dtype=float)
    values[12:15] = np.nan
    series = [response(values)]
    for indicator in declarations("default"):
        if (
            indicator.required_responses.variable != "q"
            or indicator.grain != "run"
            or indicator.reference
        ):
            continue
        result = evaluate(indicator, series, anchor=anchor)
        assert check(indicator, series, result, anchor=anchor)["status"] == "passed"
        wrong = replace(result, values=result.values + 1)
        with pytest.raises(
            ValueError, match=f"default/{indicator.name}/location=.*numerical mismatch"
        ):
            check(indicator, series, wrong, anchor=anchor)


def test_default_shares_rolling_blocks_and_keeps_member_boundaries(monkeypatch):
    """Extrema, BFI and pooled lows reuse each member's seven-day windows."""
    values = np.arange(11 * 366, dtype=float) % 101 + 1
    values[400:410] = np.nan
    series = [response(values, run="r1"), response(values * 2, run="r2")]
    registered = {item.name: item for item in declarations("default")}
    rolling = pd.DataFrame.rolling
    calls = []

    def counted(frame, *args, **kwargs):
        calls.append(len(frame))
        return rolling(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "rolling", counted)
    prepared = {}
    validation = {"tested_domain": {"probabilities": [0.5, 0.9]}}
    for name in (
        "q_mean_annual_7day_min",
        "q_mean_annual_7day_max",
        "q_baseflow_index",
        "q_return_level_2yr_7day_min",
    ):
        indicator = registered[name]
        result = evaluate(
            indicator,
            series,
            anchor="YE-SEP",
            group=True,
            expected_run_ids=("r1", "r2"),
            validation=validation,
            _prepared=prepared,
        )
        check(
            indicator,
            series,
            result,
            anchor="YE-SEP",
            group=True,
            expected_run_ids=("r1", "r2"),
            validation=validation,
        )
    assert calls == [len(values), len(values)]


@pytest.mark.parametrize(
    "name", ["q_return_level_10yr_max", "q_return_level_2yr_7day_min"]
)
def test_bundle_preserves_distinct_time_grids_between_locations(name):
    registered = {item.name: item for item in declarations("default")}
    series = []
    for run in ("r1", "r2"):
        for ordinal, start in enumerate(("2000-01-01", "2001-01-01")):
            values = 10 + np.arange(11 * 366, dtype=float) / 366
            series.append(
                replace(
                    response(values, start=start, run=run),
                    location_id=f"gauge-{ordinal}",
                    location_ordinal=ordinal,
                )
            )
    validation = {"tested_domain": {"probabilities": [0.5, 0.9]}}
    indicator = registered[name]
    result = evaluate(indicator, series, anchor="YE-SEP", validation=validation)
    assert (
        check(indicator, series, result, anchor="YE-SEP", validation=validation)[
            "status"
        ]
        == "passed"
    )


@pytest.mark.parametrize(
    "method,parameters,fragment",
    [
        ("flood_return_level/1", {"retrun_period": 10}, "required keys.*return_period"),
        ("flood_return_level/1", {"return_period": 1}, "greater than one year"),
        ("missing", {}, "accepted calculations"),
    ],
)
def test_identity_bearing_authoring_failures(method, parameters, fragment):
    with pytest.raises(ValueError, match=f"flood_risk/my_flood.*{fragment}"):
        resolve("flood_risk", method, parameters, indicator_name="my_flood")
    with pytest.raises(
        ValueError, match="discoverable sets.*default.*flood_risk.*flow_regime"
    ):
        declarations("missing_set")


def test_flood_screening_precedes_qualification_and_qualification_is_not_expanded():
    time = pd.date_range("2000-01-01", "2019-12-31")
    series = [response(1 + np.arange(len(time)) % 101)]
    validation = {"tested_domain": {"probabilities": [0.5, 0.9]}}
    indicator = resolve(
        "flood_risk",
        "flood_return_level/1",
        {"return_period": 25},
        indicator_name="too_long",
    )
    with pytest.raises(
        InsufficientReturnLevelBlocks,
        match="flood_risk/too_long/location=.*required=25, actual=20",
    ):
        evaluate(indicator, series, anchor="YE-DEC", validation=validation)
    indicator = resolve(
        "flood_risk",
        "flood_return_level/1",
        {"return_period": 5},
        indicator_name="unqualified",
    )
    with pytest.raises(
        ReturnLevelProbabilityNotCovered,
        match="flood_risk/unqualified/location=.*outside the verified",
    ):
        evaluate(indicator, series, anchor="YE-DEC", validation=validation)


@pytest.mark.parametrize("cell", [[1, 2], {"nested": 1}, (1, 2), complex(1, 2)])
def test_diagnostic_cells_reject_nested_objects(cell):
    result = IndicatorResult(
        pd.Series({"gauge": 1.0}),
        (DiagnosticTable("events", pd.DataFrame({"bad": [cell]})),),
    )
    with pytest.raises(ValueError, match="events/bad: scalar cells required"):
        validate_result(result)


FIXTURE_CALCULATIONS = '''"""Isolated local authoring examples, without a global inventory."""
import numpy as np
import pandas as pd
from blueearth_cst.indicators.contract import DiagnosticTable, IndicatorResult
from blueearth_cst.indicators.series import response_frame

def median(indicator, series, context):
    return IndicatorResult(response_frame(series).median())

def events(indicator, series, context):
    daily = response_frame(series).iloc[:, 0]
    above = daily > indicator.calculation_parameters["threshold"]
    groups = above.ne(above.shift()).cumsum()
    rows = [{"location": daily.name,
             "start_date": group.index[0].date().isoformat(),
             "end_date": group.index[-1].date().isoformat(),
             "duration_days": int(len(group))}
            for _, group in daily[above].groupby(groups[above])]
    table = pd.DataFrame(rows, columns=["location", "start_date", "end_date", "duration_days"])
    value = table.duration_days.mean() if rows else np.nan
    return IndicatorResult(pd.Series({daily.name: value}),
        (DiagnosticTable("events", table),), {"undefined_reason": None if rows else "no events"})

PARAMETERS = {median: {}, events: {"threshold": lambda value: isinstance(value, (int, float)) and np.isfinite(value)}}
'''

FIXTURE_REGISTRY = '''"""New trusted fixture set uses only its local callable registration."""
from blueearth_cst.indicators.contract import Indicator, ResponseRequirement
from .calculations import median, events
ASSUMPTIONS = {"events": "consecutive daily values strictly above threshold; native NaN breaks events"}
INDICATORS = (
    Indicator(id="daily_median", calculate=median, parameters={}, input=ResponseRequirement("q", "m3 s-1"), units="m3 s-1", description="Median of retained daily values"),
    Indicator(id="event_duration", calculate=events, parameters={"threshold": 1.0}, input=ResponseRequirement("q", "m3 s-1"), units="days", description="Mean consecutive exceedance duration"),
)
METHODS = {"median/1": median, "events/1": events}
'''

FIXTURE_CHECKS = '''"""Expected values never call the fixture's production calculations."""
from statistics import median

import numpy as np
import pandas as pd

def event_rows(item, threshold):
    result, count, start = [], 0, None
    for date, value in zip(item.time, item.values):
        if value > threshold:
            if not count:
                start = date.date().isoformat()
            count += 1
            end = date.date().isoformat()
        elif count:
            result.append({"location": item.location_id, "start_date": start,
                           "end_date": end, "duration_days": count})
            count = 0
    if count:
        result.append({"location": item.location_id, "start_date": start,
                       "end_date": end, "duration_days": count})
    return result

def expected(indicator, series, context, *, group=False):
    result = {}
    for item in series:
        if indicator.calculation_method == "median/1":
            valid = [value for value in item.values if not np.isnan(value)]
            value = median(valid) if valid else np.nan
        else:
            counts = [row["duration_days"] for row in event_rows(item, indicator.calculation_parameters["threshold"])]
            value = sum(counts)/len(counts) if counts else np.nan
        result[item.location_id] = value
    return pd.Series(result)

def check_result(indicator, series, result, context, *, group=False):
    if indicator.calculation_method == "events/1":
        assert result.diagnostics[0].name == "events"
        assert result.diagnostics[0].data.to_dict("records") == event_rows(series[0], indicator.calculation_parameters["threshold"]), "event diagnostic boundaries/durations differ"
'''


def test_new_set_median_and_nonannual_event_extension(tmp_path, monkeypatch):
    """Select a new installed fixture with unchanged orchestration and checker."""
    from blueearth_cst.experiment import indicator_registry

    existing = indicator_registry.declarations(["q", "gwr"])
    optional = indicator_registry.optional_declarations()
    local = declarations("default")
    package = tmp_path / "authoring_fixture"
    package.mkdir()
    for name, content in {
        "__init__.py": "",
        "calculations.py": FIXTURE_CALCULATIONS,
        "registry.py": FIXTURE_REGISTRY,
        "checks.py": FIXTURE_CHECKS,
    }.items():
        (package / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(sets, "__path__", [*sets.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    series = [response([0, 2, 3, 0, 5, 6, 7, 0])]
    median, event = declarations("authoring_fixture")
    assert "authoring_fixture" in loader.available_sets()
    assert indicator_registry.declarations(["q", "gwr"]) == existing
    assert indicator_registry.optional_declarations() == optional
    assert declarations("default") == local
    for indicator in (median, event):
        result = evaluate(indicator, series, anchor="YE-SEP")
        expected = (
            sample_median(series[0].values) if indicator is median else (2 + 3) / 2
        )
        assert result.values["gauge"] == expected
        assert check(indicator, series, result, anchor="YE-SEP")["status"] == "passed"
        with pytest.raises(
            ValueError,
            match=f"authoring_fixture/{indicator.name}/location=.*numerical mismatch",
        ):
            check(
                indicator,
                series,
                replace(result, values=result.values + 1),
                anchor="YE-SEP",
            )
    result = evaluate(event, series, anchor="YE-SEP")
    expected_events = pd.DataFrame(
        [
            {
                "location": "gauge",
                "start_date": "2000-01-02",
                "end_date": "2000-01-03",
                "duration_days": 2,
            },
            {
                "location": "gauge",
                "start_date": "2000-01-05",
                "end_date": "2000-01-07",
                "duration_days": 3,
            },
        ]
    )
    pd.testing.assert_frame_equal(result.diagnostics[0].data, expected_events)
    wrong_events = result.diagnostics[0].data.copy()
    wrong_events.loc[0, "duration_days"] = 9
    with pytest.raises(
        AssertionError,
        match="authoring_fixture/event_duration/location=.*event diagnostic boundaries/durations differ",
    ):
        check(
            event,
            series,
            replace(result, diagnostics=(DiagnosticTable("events", wrong_events),)),
            anchor="YE-SEP",
        )
    independent_checker = importlib.import_module(
        "blueearth_cst.indicators.sets.authoring_fixture.checks"
    )
    with monkeypatch.context() as missing_checker:
        missing_checker.delattr(independent_checker, "expected")
        with pytest.raises(
            AttributeError,
            match="authoring_fixture/event_duration/location=.*has no attribute 'expected'",
        ):
            check(event, series, result, anchor="YE-SEP")
    assert loader.registration(event).units == "days"
    empty = [response([0, 0, 0])]
    result = evaluate(event, empty, anchor="YE-DEC")
    assert result.statuses == {"gauge": "undefined"}
    assert result.metadata["undefined_reason"] == "no events"
    assert result.diagnostics[0].data.empty
    assert check(event, empty, result, anchor="YE-DEC")["status"] == "passed"
