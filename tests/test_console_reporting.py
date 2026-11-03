"""Publication summaries describe retained estimates without changing their values."""

import io
import logging
import sys
from dataclasses import replace
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

import pytest

from blueearth_cst.experiment import indicator_sets
from blueearth_cst.experiment.console_reporting import (
    _PlanStream,
    matching_sets,
    prepare_indicator_plan,
    record_indicator_detail,
    report_indicator_set,
    result_counts,
    undefined_result_summary,
)
from blueearth_cst.experiment.indicator_export import report_publication
from blueearth_cst.shared.run_log_core import _compact_log_line, _Tee, log_row


@pytest.mark.parametrize("tee", [False, True])
def test_legacy_console_folds_arrows_without_corrupting_utf8_log(monkeypatch, tee):
    data = io.BytesIO()
    live = io.TextIOWrapper(data, encoding="cp1252")
    log = io.StringIO()
    stream = _Tee(live, log) if tee else live
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", stream)
        log_row("Written → sample.csv", module="indicators")
        stream.flush()
        live.flush()
    assert b"Written -> sample.csv" in data.getvalue()
    if tee:
        assert "Written → sample.csv" in log.getvalue()


def test_indicator_details_survive_queued_cp1252_logging(
    tmp_path, monkeypatch, capsys, caplog
):
    """Snakemake's asynchronous file logger may use the Windows code page."""
    native_log = tmp_path / "snakemake.log"
    handler = logging.FileHandler(native_log, encoding="cp1252")
    queue = Queue()
    listener = QueueListener(queue, handler)
    logger = logging.getLogger("snakemake.logging")
    monkeypatch.setattr(logger, "handlers", [QueueHandler(queue)])
    # Use setLevel (through caplog) to clear decisions cached by earlier tests.
    caplog.set_level(logging.INFO, logger="snakemake.logging")
    monkeypatch.setattr(logger, "propagate", False)
    message = "Indicator set abc; manifest → C:/cases/café/水/indicators.json"
    listener.start()
    try:
        record_indicator_detail(tmp_path, message)
        report_indicator_set(
            tmp_path, {"indicator_set_id": "abc", "engine_root": "_engine/sets"}
        )
    finally:
        listener.stop()
        handler.close()
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
    assert "UnicodeEncodeError" not in captured.err
    assert r"\u2192" in native_log.read_text(encoding="cp1252")
    assert r"\u6c34" in native_log.read_text(encoding="cp1252")
    assert message in (tmp_path / "logs/wf4_indicators.log").read_text(encoding="utf-8")


def test_unwritable_detail_log_uses_encoding_safe_console_fallback(
    tmp_path, monkeypatch
):
    (tmp_path / "logs").write_text("blocks log directory", encoding="utf-8")
    data = io.BytesIO()
    live = io.TextIOWrapper(data, encoding="cp1252")
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", live)
        record_indicator_detail(tmp_path, "Manifest → café/水/indicators.json")
        live.flush()
    assert b"Indicator details: Manifest -> caf\xe9/" in data.getvalue()


def test_indicator_summary_counts_actual_finite_and_missing_estimates(tmp_path, capsys):
    table = tmp_path / "indicators.csv"
    payload = "indicator,location,run_group_id,value\nmean,a,run_1,1.5\nmean,b,run_1,\nmean,c,run_1,nan\nmean,d,run_1,-2\n"
    table.write_text(payload, encoding="utf-8")
    manifest = {
        "indicator_set_id": "selected",
        "engine_root": "_engine/indicator_sets/selected",
        "tables": [{"file": {"path": table.name}}],
    }
    assert result_counts(tmp_path, manifest) == (2, 2, 0)
    assert table.read_text(encoding="utf-8") == payload


def test_shared_calculations_keep_both_set_names_and_report_reuse(tmp_path, capsys):
    source = indicator_sets.legacy_selection({"operation": "indicators-only"}).sets[0]
    default = indicator_sets.resolve_set(source, ["gwr"])
    alias = replace(default, set_id="same_numbers")
    records = indicator_sets.calculation_records(default.indicators)
    sets = [default, alias]
    assert matching_sets(sets, {"calculations": records}) == sets
    manifest = {
        "indicator_set_id": "opaque",
        "engine_root": "_engine/indicators",
        "calculations": records,
    }
    report_indicator_set(tmp_path, manifest, sets, reused=True)
    output = capsys.readouterr().out
    assert output == ""
    detail = (tmp_path / "logs/wf4_indicators.log").read_text(encoding="utf-8")
    assert "default: reusing calculated results" in detail
    assert "same_numbers: reusing calculated results" in detail


@pytest.mark.parametrize("producer", [False, True, "fragmented"])
def test_planning_stream_keeps_warnings_and_detailed_log(
    caplog, tmp_path, monkeypatch, producer
):
    monkeypatch.setattr(
        logging.getLogger("snakemake.logging"), "handlers", [caplog.handler]
    )
    stream = io.StringIO()
    proxy = _PlanStream(stream, tmp_path)
    detail = "12:00:00 - indicators - Indicator set abc planned: 12 indicator declarations; 896 expected result keys\n"
    with caplog.at_level("INFO", logger="snakemake.logging"):
        if producer == "fragmented":
            for char in detail.replace("\n", "\r\n"):
                proxy.write(char)
        elif producer:
            monkeypatch.setattr(sys, "stdout", proxy)
            log_row("abc: 12 declarations over 896 result keys", module="indicators")
        else:
            proxy.write(detail)
        proxy.write("12:00:00 - indicators - WARNING - input coverage incomplete\n")
    assert "896 expected result keys" in caplog.text
    assert "896 expected result keys" in (
        tmp_path / "logs/wf4_indicators.log"
    ).read_text(encoding="utf-8")
    assert (
        stream.getvalue()
        == "12:00:00 - indicators - WARNING - input coverage incomplete\n"
    )


def test_publication_reports_saved_reused_missing_and_failed_results(
    tmp_path, capsys, monkeypatch, caplog
):
    console = io.StringIO()
    monkeypatch.setattr(
        logging.getLogger(), "handlers", [logging.StreamHandler(console)]
    )
    logger = logging.getLogger("snakemake.logging")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="snakemake.logging")
    outcomes = [
        {
            "set_id": "default",
            "status": "current",
            "indicator_set_id": "hidden1",
            "publication_id": "hidden2",
            "missing_results": 2,
            "available_results": 4,
        },
        {
            "set_id": "flow_regime",
            "status": "current",
            "indicator_set_id": "hidden3",
            "publication_id": "hidden4",
            "reused": True,
        },
        {
            "set_id": "failed_set",
            "status": "failed",
            "prior_publication_id": "hidden5",
            "reason": "MissingInput: daily discharge unavailable",
        },
        {
            "set_id": "new_set",
            "status": "failed",
            "prior_publication_id": None,
            "reason": "MissingInput: recharge unavailable",
        },
    ]
    assert report_publication(tmp_path, outcomes) is False
    output = capsys.readouterr().out
    assert "default: results saved" in output
    assert "2 results undefined; 4 results available" in output
    assert "flow_regime: results unchanged; reusing saved results" in output
    assert "previous results kept; they have not been updated" in output
    assert "no saved results available" in output
    assert "daily discharge unavailable" in output
    assert "hidden" not in output
    assert console.getvalue() == ""
    assert "hidden1" in (tmp_path / "logs/wf4_indicators.log").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "module,current,proposed",
    [
        (
            "experiment",
            "Model reference matches the live model",
            "Live model matches the recorded experiment identity",
        ),
        (
            "experiment",
            "Model reference: models/hydrology/wflow digest 1594b6dd303d... over 6 inputs",
            "Using hydrological model: models/hydrology/wflow; 6 input files; checksum 1594b6dd303d…",
        ),
        (
            "indicators",
            "abcdef123456: 1 declaration over 2 result keys",
            "Indicator set abcdef123456 planned: 1 indicator declaration; 2 expected result keys",
        ),
        (
            "responses",
            "Duplicate series on one model point, kept once: Q_2 as Q_1",
            "Duplicate series at the same model location will be used once: Q_2 as Q_1",
        ),
        (
            "wflow",
            "FAILED [2/3] Run run_002 batch=0  Missing native output",
            "FAILED run run_002 in batch 0 [2/3]: Missing native output",
        ),
    ],
)
def test_simulation_presentation_keeps_identity_and_failure_detail(
    module, current, proposed
):
    assert (
        _compact_log_line(f"12:34:56 - {module} - {current}\n")
        == f"12:34:56 - {module} - {proposed}\n"
    )


def test_plan_reports_names_and_indicator_results(tmp_path, monkeypatch, capsys):
    from blueearth_cst.experiment import indicator_plan

    source = indicator_sets.legacy_selection({"operation": "indicators-only"}).sets[0]
    default = indicator_sets.resolve_set(source, ["gwr"])
    alias = replace(default, set_id="same_numbers")
    request = {"calculations": indicator_sets.calculation_records(default.indicators)}
    plan = {
        "indicator_request_id": "a" * 64,
        "indicator_set_id": "b" * 64,
        "request": {"declarations": ["gwr_annual_total"]},
        "expected_result_keys": ["first", "second"],
    }
    monkeypatch.setattr(indicator_plan, "build_indicator_plan", lambda *args: plan)
    assert prepare_indicator_plan(tmp_path, request, [default, alias]) == plan
    output = capsys.readouterr().out
    assert "default: 1 indicator; 2 results across simulations and locations" in output
    assert (
        "same_numbers: 1 indicator; 2 results across simulations and locations"
        in output
    )
    assert "expected result keys" not in output
    assert "bbbbbbbbbbbb" not in output


def test_undefined_results_explain_absent_zero_flow_without_hiding_other_gaps(
    tmp_path, capsys
):
    table = tmp_path / "results.csv"
    table.write_text(
        "indicator,location,value,status,reason\n"
        "timing_med,a,,undefined,\n"
        "timing_iqr,a,,undefined,\n"
        "other,a,,undefined,\n"
        "count,a,0,available,\n"
        "absent,a,,unavailable,missing_response:q\n",
        encoding="utf-8",
    )
    manifest = {
        "tables": [{"file": {"path": table.name}}],
        "calculations": [
            {
                "indicator": name,
                "method": "erfa_annual_statistic/1",
                "parameters": {"metric": "time_main_zero_seq", "statistic": statistic},
            }
            for name, statistic in (("timing_med", "med"), ("timing_iqr", "iqr"))
        ],
    }
    expected, other = undefined_result_summary(tmp_path, manifest)
    assert (expected, other) == (2, ["other (1 result)"])
    outcome = {
        "set_id": "eflow-erfa",
        "status": "current",
        "indicator_set_id": "hidden",
        "publication_id": "hidden",
        "missing_results": 3,
        "available_results": 1,
        "expected_undefined_results": expected,
        "undefined_indicators": other,
    }
    assert report_publication(tmp_path, [outcome])
    output = capsys.readouterr().out
    assert "2 zero-flow timing results undefined; no zero-flow event" in output
    assert "WARNING - eflow-erfa: 1 result undefined" in output
    assert "other (1 result)" in output
    assert "missing or invalid" not in output
    assert "WARNING - eflow-erfa: 2" not in output


def test_overlapping_plans_keep_details_off_console_and_restore_stdout(
    tmp_path, capsys, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from blueearth_cst.experiment import indicator_plan

    first_inside, second_inside, first_left = Event(), Event(), Event()
    original = sys.stdout

    def write_plan(root, request):
        if root.name == "first":
            first_inside.set()
            assert second_inside.wait(10)
            log_row("abc1: 1 declaration over 2 result keys", module="indicators")
        else:
            second_inside.set()
            assert first_left.wait(10)
            log_row("abc2: 1 declaration over 2 result keys", module="indicators")
        return {"expected_result_keys": []}

    monkeypatch.setattr(indicator_plan, "write_indicator_plan", write_plan)

    def first():
        prepare_indicator_plan(tmp_path / "first", {"calculations": []}, ())
        first_left.set()

    def second():
        assert first_inside.wait(10)
        prepare_indicator_plan(tmp_path / "second", {"calculations": []}, ())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first), pool.submit(second)]
        for future in futures:
            future.result(timeout=30)
    assert sys.stdout is original
    assert capsys.readouterr().out == ""
    for name, identity in (("first", "abc1"), ("second", "abc2")):
        detail = (tmp_path / name / "logs/wf4_indicators.log").read_text(
            encoding="utf-8"
        )
        assert identity in detail


def test_planning_capture_preserves_unrelated_thread_output(tmp_path, capsys):
    from concurrent.futures import ThreadPoolExecutor

    from blueearth_cst.experiment.console_reporting import _planning_output

    with _planning_output(tmp_path), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: sys.stdout.write("unrelated warning fragment")).result()
    assert capsys.readouterr().out == "unrelated warning fragment"
