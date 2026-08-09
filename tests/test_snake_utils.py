"""Exact-equivalence tests for the shared get_config helper (R3 §3, §8).

Pins the semantics the four inline get_config copies (three Snakefiles +
conftest) had before they were collapsed into blueearth_cst/shared/snake_utils.py, so the move
is provably identity-preserving rather than merely green on a smoke test.
"""

import io
import os
import sys
import time
import warnings
from pathlib import Path

import pytest

import re

import blueearth_cst.shared.snake_utils as su  # noqa: E402
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    _Heartbeat,
    _compact_log_line,
    _cr_overwrite,
    _fmt_elapsed,
    _log_path_parts,
    _relativize_paths,
    get_config,
    log_row,
    patch_psutil_windows_benchmark,
    rule_banner,
    save_figure,
    target_banner,
    tee_to_log,
)


def test_missing_required_raises():
    with pytest.raises(ValueError):
        get_config({}, "absent", optional=False)


def test_missing_optional_returns_none_by_default():
    assert get_config({}, "absent") is None


def test_missing_optional_returns_explicit_default():
    assert get_config({}, "absent", default="fallback") == "fallback"


def test_present_key_returned():
    assert get_config({"k": 42}, "k") == 42


def test_present_required_key_returned():
    assert get_config({"k": "v"}, "k", optional=False) == "v"


def test_none_value_returned_not_treated_as_missing():
    # A key explicitly set to None returns None, not the default — the key IS
    # present. This is the subtle semantic the inline copies all shared.
    assert get_config({"k": None}, "k", default="fallback") is None


@pytest.mark.parametrize("falsey", [0, "", False, []])
def test_falsey_values_returned_as_is(falsey):
    result = get_config({"k": falsey}, "k", default="fallback")
    assert result == falsey and type(result) is type(falsey)


# --- _compact_log_line (hydromt format) --------------------------------------


def test_compact_shortens_timestamp_and_drops_dotted_name():
    line = (
        "2026-07-21 18:03:38,474 - hydromt.model.model - model - INFO - "
        "Initializing wflow_sbm model.\n"
    )
    # date + milliseconds dropped -> HH:MM:SS; dotted name dropped; module kept
    assert _compact_log_line(line) == (
        "18:03:38 - model - INFO - Initializing wflow_sbm model.\n"
    )


def test_compact_preserves_dashes_in_message():
    line = (
        "2026-07-21 18:03:20,505 - hydromt.model.model - model - INFO - "
        "setup_rivers.river_routing=kinematic - wave - x\n"
    )
    # message (with its own ' - ') is kept whole; only ts + dotted name change
    assert _compact_log_line(line) == (
        "18:03:20 - model - INFO - setup_rivers.river_routing=kinematic - wave - x\n"
    )


def test_compact_keeps_level_and_no_trailing_newline():
    line = (
        "2026-07-21 18:03:18,884 - hydromt.hydromt_wflow.workflows.basemaps"
        " - basemaps - WARNING - Model resolution mismatch"
    )  # no trailing newline
    assert _compact_log_line(line) == (
        "18:03:18 - basemaps - WARNING - Model resolution mismatch"
    )


@pytest.mark.parametrize(
    "line",
    [
        "[ Info: Wflow version v1.0.2\n",  # Julia log, no timestamp
        "Traceback (most recent call last):\n",  # traceback
        "just a plain print line\n",
        "",  # empty
    ],
)
def test_compact_passes_through_non_hydromt(line):
    assert _compact_log_line(line) == line


# --- save_figure -------------------------------------------------------------


def test_save_figure_writes_creates_parent_and_announces(tmp_path, capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_path / "plots" / "basin_area.png"  # parent does not exist yet
    plt.figure()
    plt.plot([0, 1], [0, 1])
    save_figure(str(out), dpi=50)
    assert out.exists()
    assert f"Saved figure: {out}" in capsys.readouterr().out


# --- log_row -----------------------------------------------------------------


def test_log_row_standard_format(capsys):
    log_row("hello world", module="plot")
    out = capsys.readouterr().out.strip()
    assert re.match(r"^\d{2}:\d{2}:\d{2} - plot - INFO - hello world$", out)


def test_log_row_row_survives_compaction_unchanged():
    # a log_row line is already compact -> the tee's _compact_log_line is a no-op
    row = "21:56:12 - plot - INFO - Saved figure: x.png\n"
    assert _compact_log_line(row) == row


# --- psutil benchmark shim ---------------------------------------------------


def test_patch_psutil_exposes_pss():
    # Snakemake's benchmark sampler reads meminfo.pss; on Windows psutil omits it
    # (only uss), which NAs every metric. The shim must expose pss (= uss proxy).
    if sys.platform != "win32":
        import pytest as _pytest

        _pytest.skip("Windows-only shim")
    import psutil

    patch_psutil_windows_benchmark()
    meminfo = psutil.Process().memory_full_info()
    assert hasattr(meminfo, "pss")
    assert meminfo.pss == meminfo.uss  # Windows proxy


# --- rule_banner (console header) --------------------------------------------


class _FakeTTY:
    def isatty(self):
        return True


def test_rule_banner_bold_on_tty(monkeypatch):
    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert rule_banner("1.09", "run_wflow") == "\033[1;36m1.09  run_wflow\033[0m"


def test_rule_banner_plain_when_not_tty(monkeypatch):
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())  # isatty() -> False
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = rule_banner("1.09", "run_wflow")
    assert out == "1.09  run_wflow" and "\033" not in out  # no escape codes


def test_rule_banner_respects_no_color_env(monkeypatch):
    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    monkeypatch.setenv("NO_COLOR", "1")
    assert rule_banner("2.04", "monthly_change") == "2.04  monthly_change"


# --- target_banner (rule `all` message) --------------------------------------


def test_target_banner_puts_one_target_per_line(monkeypatch):
    """The whole point: Snakemake's own `input:` joins with ", "; this does not."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())  # isatty() -> False
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("2.00", "all", ["a/x.csv", "b/y.png"])
    assert out == "2.00  all\n    a/x.csv\n    b/y.png"
    assert ", " not in out


def test_target_banner_keeps_the_rule_banner_colouring(monkeypatch):
    """The banner half is rule_banner's, escape codes and all -- only the banner.

    A target path must never be wrapped in escape codes: the message is what a
    reader copies a path out of.
    """
    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("1.00", "all", ["p/q.nc"])
    assert out.startswith("\033[1;36m1.00  all\033[0m\n")
    assert "\033" not in out.split("\n", 1)[1]


def test_target_banner_accepts_a_dict_values_view(monkeypatch):
    """WF2 and WF3 pass `TARGETS.values()`, not a list."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    targets = {"a": "one.csv", "b": "two.csv"}
    assert target_banner("3.00", "all", targets.values()) == (
        "3.00  all\n    one.csv\n    two.csv"
    )


def test_target_banner_with_no_targets_is_just_the_banner(monkeypatch):
    """No trailing blank line -- an empty list must not print an empty row."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert target_banner("1.00", "all", []) == "1.00  all"


def test_target_banner_relativizes_against_project_dir(monkeypatch):
    """The root moves to the banner; the paths below it lose the prefix."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner(
        "2.00",
        "all",
        ["C:/TESTS/CST/gabonx/climate_projections/cmip6/summary/x.csv"],
        "C:/TESTS/CST/gabonx",
    )
    assert out == (
        "2.00  all  [C:/TESTS/CST/gabonx]\n    climate_projections/cmip6/summary/x.csv"
    )


def test_target_banner_relativizes_a_native_separator_root(monkeypatch):
    """Snakefiles build targets with `/`; project_dir may arrive either way."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("1.00", "all", ["proj/logs/wf1.log"], os.path.join("proj"))
    assert out.endswith("    logs/wf1.log")


def test_target_banner_leaves_a_path_outside_the_project_absolute(monkeypatch):
    """Only the project prefix is stripped -- a catalog elsewhere stays whole."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("2.00", "all", ["D:/data/catalog.yml"], "C:/TESTS/CST/gabonx")
    assert "    D:/data/catalog.yml" in out


def test_target_banner_without_project_dir_keeps_paths_verbatim(monkeypatch):
    """The default is unchanged: no root given, nothing stripped, no bracket."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("3.00", "all", ["C:/p/q.csv"])
    assert out == "3.00  all\n    C:/p/q.csv"
    assert "[" not in out


# --- path relativization -----------------------------------------------------


def test_log_path_parts_project_root_and_id(tmp_path):
    lp = tmp_path / "gabon" / "logs" / "1.03_create_model.log"
    root, log_id = _log_path_parts(str(lp))
    assert root == os.path.normpath(str(tmp_path / "gabon"))
    assert log_id == "1.03_create_model.log"
    # wildcard sub-log: project root unchanged, id is the path below logs/
    lp2 = tmp_path / "gabon" / "logs" / "3.10_run_wflow" / "rlz_1_st_1.log"
    root2, log_id2 = _log_path_parts(str(lp2))
    assert root2 == os.path.normpath(str(tmp_path / "gabon"))
    assert log_id2 == "3.10_run_wflow/rlz_1_st_1.log"


def test_relativize_strips_project_root_both_separators():
    root = os.path.normpath("C:/TESTS/gabon")
    native = f"Writing geoms to {root}{os.sep}hydrology_model{os.sep}basins.geojson.\n"
    assert _relativize_paths(native, root) == (
        f"Writing geoms to hydrology_model{os.sep}basins.geojson.\n"
    )
    fwd_root = root.replace(os.sep, "/")
    forward = f"Writing config to {fwd_root}/hydrology_model/wflow_sbm.toml.\n"
    assert _relativize_paths(forward, root) == (
        "Writing config to hydrology_model/wflow_sbm.toml.\n"
    )


def test_relativize_leaves_out_of_project_paths_absolute():
    root = os.path.normpath("C:/TESTS/gabon")
    line = f"Reading data from {os.path.normpath('C:/data/wflow_global/x.tif')}\n"
    assert _relativize_paths(line, root) == line  # not under project -> untouched


def test_tee_to_log_relativizes_project_paths(tmp_path):
    proj = tmp_path / "gabon"
    log = proj / "logs" / "1.15_plot_wflow_evaluation.log"
    abs_png = os.path.join(str(proj), "plots", "map.png")
    with tee_to_log(log):
        print(f"Saved figure: {abs_png}")
    text = log.read_text(encoding="utf-8")
    assert "Saved figure: " + os.path.join("plots", "map.png") in text
    assert abs_png not in text  # absolute project path relativized away


# --- tee_to_log (R3 §6) ------------------------------------------------------


def test_tee_to_log_writes_and_restores_streams(tmp_path):
    log = tmp_path / "sub" / "rule.log"  # parent does not exist yet
    out0, err0 = sys.stdout, sys.stderr
    with tee_to_log(log):
        print("hello-stdout")
        print("hello-stderr", file=sys.stderr)
    # streams restored to exactly what they were on entry
    assert sys.stdout is out0 and sys.stderr is err0
    text = log.read_text(encoding="utf-8")
    assert "hello-stdout" in text and "hello-stderr" in text


def test_tee_to_log_captures_preexisting_console_logging(tmp_path):
    # A library (like hydromt) installs a StreamHandler bound to the real stdout
    # BEFORE tee_to_log runs. Its records must be compacted AND land in the log
    # file, not bypass the tee (regression: the earlier print()-based test only
    # exercised the regex, not this wiring).
    import logging

    lg = logging.getLogger("cst_test_lib")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # isolate: only our handler emits, so no double-count
    handler = logging.StreamHandler(sys.stdout)  # bound to the current console
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
        )
    )
    lg.addHandler(handler)
    log = tmp_path / "rule.log"
    try:
        with tee_to_log(log):  # note: we do NOT print() — only the logger emits
            lg.info("built model grid")
    finally:
        lg.removeHandler(handler)
    body = log.read_text(encoding="utf-8")
    # compacted row present exactly once, and the full hydromt timestamp is gone
    assert (
        len(re.findall(r"\d{2}:\d{2}:\d{2} - \w+ - INFO - built model grid", body)) == 1
    )
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", body)


def test_tee_to_log_compacts_hydromt_format(tmp_path):
    log = tmp_path / "rule.log"
    with tee_to_log(log):
        # a hydromt-format record (as hydromt's Python API emits) and a plain line
        print("2026-07-21 18:03:38,474 - hydromt.model.model - model - INFO - built")
        print("plain progress line")
    text = log.read_text(encoding="utf-8")
    # the record row is exactly the compacted form: HH:MM:SS, no date/ms/name
    row = next(line for line in text.splitlines() if "INFO - built" in line)
    assert row == "18:03:38 - model - INFO - built"
    assert "hydromt.model.model" not in text  # dotted name dropped
    assert "plain progress line" in text  # non-hydromt line untouched


def test_tee_to_log_writes_project_header(tmp_path):
    # a `.../<project>/logs/<rule>.log` path yields a header naming the project,
    # the full project dir, and the rule-log id; the date lives here (dropped
    # from each row), followed by a blank line before the body.
    log = tmp_path / "gabon" / "logs" / "1.07_build_wflow_model.log"
    with tee_to_log(log):
        print("body line")
    head = log.read_text(encoding="utf-8").splitlines()
    assert head[0].startswith("# BlueEarth-CST")
    assert "project: gabon" in head[0]
    assert head[1].startswith("# project dir:") and head[1].rstrip().endswith("gabon")
    assert "1.07_build_wflow_model.log" in head[2] and "started" in head[2]
    assert head[3] == ""  # blank line separates header from body
    assert head[4] == "body line"


def test_tee_to_log_reraises_and_still_restores(tmp_path):
    log = tmp_path / "rule.log"
    out0, err0 = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match="boom"):
        with tee_to_log(log):
            print("before-error")
            raise RuntimeError("boom")
    # exception propagated (not swallowed) AND streams restored in finally
    assert sys.stdout is out0 and sys.stderr is err0
    assert "before-error" in log.read_text(encoding="utf-8")


# --- carriage-return progress-bar collapse -----------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("plain line", "plain line"),  # no CR: untouched
        ("\r[## ] 10%\r[####] 20%", "[####] 20%"),  # keep last redraw
        # dask ends a redrawn line with a bare CR before the newline; the empty
        # trailing segment must be dropped, not kept (else the bar blanks out).
        (
            "\r[#] 0%\r[####] 100% Completed | 7.08 s\r",
            "[####] 100% Completed | 7.08 s",
        ),
        ("\r", ""),  # only a bare CR -> nothing visible
    ],
)
def test_cr_overwrite_keeps_last_nonempty_segment(line, expected):
    assert _cr_overwrite(line) == expected


def test_tee_to_log_collapses_progress_bar_to_final_line(tmp_path):
    log = tmp_path / "rule.log"
    with tee_to_log(log):
        # mimic dask's ProgressBar: many \r-redraws written as separate chunks,
        # the last ending "\r\n" (a bare \r right before the newline).
        for pct in (0, 42, 100):
            done = "#" * (pct // 10)
            state = "Completed" if pct == 100 else "In progress"
            sys.stdout.write(f"\r[{done:<10}] | {pct}% {state} | 7.08 s")
        sys.stdout.write("\r\n")
    body = [ln for ln in log.read_text(encoding="utf-8").splitlines() if "%" in ln]
    # exactly one progress line survives, and it is the final 100% redraw
    assert len(body) == 1
    assert "100% Completed" in body[0]
    # no intermediate redraw ("In progress" / "42%") leaked into the log
    assert "In progress" not in log.read_text(encoding="utf-8")
    assert "42%" not in body[0]


def test_tee_to_log_close_flushes_interrupted_bar(tmp_path):
    # a bar cut short (no final newline) must still land its last state in the log
    log = tmp_path / "rule.log"
    with tee_to_log(log):
        sys.stdout.write("\r[## ] 50% In progress")  # no trailing newline
    assert "50% In progress" in log.read_text(encoding="utf-8")


# --- heartbeat watchdog ------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0s"), (45, "45s"), (134, "2m14s"), (3600, "1h00m00s"), (3980, "1h06m20s")],
)
def test_fmt_elapsed(seconds, expected):
    assert _fmt_elapsed(seconds) == expected


def test_heartbeat_fires_on_silence_and_summarizes():
    stream = io.StringIO()
    hb = _Heartbeat("2.05_merge", stream, interval=0.05).start()
    time.sleep(0.16)  # stay silent well past the interval
    hb.stop()
    out = stream.getvalue()
    assert "still running" in out and "2.05_merge" in out  # heartbeat fired
    assert "done in" in out  # stop() prints the summary


def test_heartbeat_suppressed_while_active():
    stream = io.StringIO()
    hb = _Heartbeat("busy_rule", stream, interval=0.2).start()
    for _ in range(6):  # keep touching so it never stays silent for 0.2s
        hb.touch()
        time.sleep(0.02)
    hb.stop()
    assert "still running" not in stream.getvalue()  # never beeped


def test_heartbeat_disabled_when_interval_zero():
    stream = io.StringIO()
    hb = _Heartbeat("off", stream, interval=0).start()
    time.sleep(0.05)
    hb.stop()
    assert stream.getvalue() == ""  # nothing at all, not even a summary


def test_heartbeat_reports_systemexit_zero_as_done(tmp_path, capsys):
    # A `script:` module ends its cache-hit path with `raise SystemExit(0)`, which
    # IS a success -- Snakemake reports the job Finished. The console summary must
    # agree; it used to print "failed after" on every cached WF2 fetch.
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log, heartbeat_interval=0.05):
            raise SystemExit(0)
    err = capsys.readouterr().err
    assert "done in" in err and "failed after" not in err


def test_heartbeat_still_reports_real_failures(tmp_path, capsys):
    # The other half of the same test: only exit code 0 is forgiven.
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log, heartbeat_interval=0.05):
            raise SystemExit(1)
    with pytest.raises(RuntimeError):
        with tee_to_log(tmp_path / "rule2.log", heartbeat_interval=0.05):
            raise RuntimeError("boom")
    err = capsys.readouterr().err
    assert err.count("failed after") == 2 and "done in" not in err


def test_tee_to_log_heartbeat_goes_to_console_not_log(tmp_path, capsys):
    # THE key requirement: the heartbeat must not populate the log file
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0.05):
        time.sleep(0.16)  # silence triggers a console heartbeat
    err = capsys.readouterr().err
    logged = log.read_text(encoding="utf-8")
    assert "still running" in err and "done in" in err  # console got them
    assert "still running" not in logged and "done in" not in logged  # log stayed clean


# ---------------------------------------------------------------------------
# O-22: warn_if_project_dir_in_repo
#
# The design's stated verification for this feature was "tests/test_cli.py
# matches on combined stdout+stderr -- confirm its assertions are undisturbed".
# Review found that false: no test in test_cli.py asserts on output text (the
# three CLI tests assert only returncode == 0, using the combined stream as the
# assertion MESSAGE). The feature would have shipped with zero coverage, and
# the exemption branch -- the case most likely to regress -- would never have
# been exercised. These are the replacement.
# ---------------------------------------------------------------------------


def test_warn_in_repo_project_dir_warns(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scratch_run").mkdir(parents=True)
    with pytest.warns(UserWarning, match="inside the repository tree"):
        fired = su.warn_if_project_dir_in_repo(repo / "scratch_run", repo)
    assert fired is True


def test_warn_exempt_test_case_is_silent(tmp_path):
    """The fixture exemption: the baseline seed config is TRACKED, and a
    tracked config cannot carry a machine-specific absolute path."""
    repo = tmp_path / "repo"
    (repo / "test_case" / "test_local").mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        fired = su.warn_if_project_dir_in_repo(repo / "test_case" / "test_local", repo)
    assert fired is False


def test_warn_absolute_out_of_tree_is_silent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere" / "my_project"
    outside.mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fired = su.warn_if_project_dir_in_repo(outside, repo)
    assert fired is False


def test_warn_uses_containment_not_string_prefix(tmp_path):
    """`test_caseX` must not read as inside `test_case`, and a sibling repo
    directory must not read as inside the repo. A startswith() implementation
    passes the three cases above and fails both of these."""
    repo = tmp_path / "repo"
    (repo / "test_caseX").mkdir(parents=True)
    with pytest.warns(UserWarning):
        assert su.warn_if_project_dir_in_repo(repo / "test_caseX", repo) is True

    sibling = tmp_path / "repo_other"
    sibling.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert su.warn_if_project_dir_in_repo(sibling, repo) is False


# --- climate_store_rule (R07 B1) ---------------------------------------------

_WINDOW = {"starttime": "2000-01-01T00:00:00", "endtime": "2020-12-31T00:00:00"}


def _spec(**overrides):
    kwargs = dict(
        project_dir="/proj",
        model_region="{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        clim_source="era5",
        historical_window=_WINDOW,
        data_sources="config/catalogs/deltares_data.yml",
    )
    kwargs.update(overrides)
    return su.climate_store_rule(**kwargs)


def test_climate_store_rule_key_matches_the_pre_r07_wf3_construction():
    """The store KEY must be byte-identical to the one wf3 built inline.

    R9 P2 moved the store under `data/climate/`, but the key is the load-bearing
    half: it is a CACHE key (R9 design Finding 3), so two experiments sharing a
    source and a window must still land on one directory. The assertion is split
    accordingly -- the root moved, the key did not.
    """
    spec = _spec()
    assert spec.store_dir == "/proj/data/climate/historical/era5_20000101_20201231"
    assert spec.store_dir.endswith("/era5_20000101_20201231")
    assert spec.outputs["climate_nc"] == f"{spec.store_dir}/extract_historical.nc"
    # ADR 0003: the polygon is no longer a per-store-key output. It is the one
    # project artifact, and the store declares it as an INPUT.
    assert "region_geojson" not in spec.outputs
    assert spec.inputs["region_geojson"] == "/proj/data/spatial/geoms/region.geojson"


def test_climate_store_rule_inputs_are_the_catalog_and_the_region():
    """ext2-01 + ADR 0003: two inputs, and BOTH symmetric across the workflows.

    The catalog stays the store's freshness boundary. The region joined it when
    the delineation moved out of this producer; what ext2-01 forbids is a
    workflow-LOCAL input, not a second shared one.
    """
    spec = _spec()
    assert spec.inputs == {
        "catalog": "config/catalogs/deltares_data.yml",
        "region_geojson": "/proj/data/spatial/geoms/region.geojson",
    }


def test_climate_store_region_input_is_the_region_rule_output():
    """One owner for the path: the two helpers cannot disagree about it."""
    spec = _spec()
    region = su.region_rule(
        project_dir="/proj",
        model_region="{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        data_sources="config/catalogs/deltares_data.yml",
    )
    assert spec.inputs["region_geojson"] == region.region_geojson
    assert region.outputs == {"region_geojson": region.region_geojson}


def test_climate_store_rule_params_carry_the_content_surface():
    spec = _spec()
    assert set(spec.params) == {
        "model_region",
        "clim_source",
        "starttime",
        "endtime",
        "hydrography",
        "basin_index",
    }
    # The catalog moved OUT of params and into the declared input.
    assert "data_sources" not in spec.params
    assert spec.params["starttime"] == _WINDOW["starttime"]
    assert spec.params["endtime"] == _WINDOW["endtime"]


def test_climate_store_rule_hydrography_defaults_match_the_spatial_contract():
    """Climate extraction and P1 share one model-neutral source default."""
    from blueearth_cst.spatial.config import parse_spatial_config

    spec = _spec()
    spatial = parse_spatial_config({"region": {"basin": [0, 0]}}, {})

    assert spec.params["hydrography"] == spatial.hydrography == "merit_hydro_ihu"
    assert spec.params["basin_index"] == spatial.basin_index == "merit_hydro_index"


def test_climate_store_rule_overrides_are_carried_through():
    spec = _spec(hydrography="merit_hydro_1k", basin_index="my_index")
    assert spec.params["hydrography"] == "merit_hydro_1k"
    assert spec.params["basin_index"] == "my_index"


@pytest.mark.parametrize("source", ["chirps", "chirps_global"])
def test_chirps_branch_declares_the_standardised_orography_sidecar(source):
    """R07 standardises on `orography.nc` (was `<clim_source>_orography.nc`)."""
    spec = _spec(clim_source=source)
    assert spec.outputs["oro_nc"] == f"{spec.store_dir}/orography.nc"
    assert list(spec.outputs) == ["climate_nc", "oro_nc"]


def test_no_orography_output_outside_the_chirps_branch():
    assert "oro_nc" not in _spec(clim_source="era5").outputs


def test_climate_store_rule_script_is_relative_to_the_repo_root():
    """One relative path serves both Snakefiles (`script:` resolves to basedir)."""
    spec = _spec()
    assert spec.script == "blueearth_cst/climate_analysis/extract_historical_climate.py"
    assert (Path(__file__).resolve().parents[1] / spec.script).is_file()


def test_climate_store_rule_rejects_a_non_mapping_window():
    with pytest.raises(TypeError, match="historical_window"):
        _spec(historical_window=("2000-01-01T00:00:00", "2020-12-31T00:00:00"))


def test_climate_store_rule_rejects_a_sub_day_window():
    """The day-resolution store key cannot represent a sub-day window."""
    with pytest.raises(ValueError, match="time-of-day"):
        _spec(
            historical_window={
                "starttime": "2000-01-01T06:00:00",
                "endtime": "2020-12-31T00:00:00",
            }
        )


def test_climate_store_rule_is_frozen():
    """The two Snakefiles share one contract object; it must not be mutable."""
    spec = _spec()
    with pytest.raises(Exception):
        spec.store_dir = "/elsewhere"


# --------------------------------------------------------------------------- #
# Log-line path compaction (2026-08-01)
# --------------------------------------------------------------------------- #


def _rel(text, project_root=r"C:\TESTS\CST\gabon_0108"):
    from blueearth_cst.shared.snake_utils import _relativize_paths

    return _relativize_paths(text, project_root)


def test_an_installed_dependency_path_collapses_to_its_package():
    """The reported case. Which package the file came from is the information;
    where pixi put the env is not, and it differs per machine."""
    line = (
        r"Parsing data catalog from C:\Users\x\workspace\blueearth_cst\.pixi"
        r"\envs\default\Lib\site-packages\hydromt_wflow\data\parameters_data.yml"
    )
    assert _rel(line) == (
        "Parsing data catalog from <site-packages>/hydromt_wflow\data"
        "\parameters_data.yml"
    )


# The three cases below assert how a WINDOWS path renders: a drive letter, a
# backslash separator, or pixi's win-64 `Lib/site-packages` (linux-64 lays that
# out as `lib/python3.12/site-packages`, so the match falls through to the
# <repo> branch instead). The abbreviation LOGIC is platform-neutral and stays
# covered on both legs by the other cases in this section; only these spellings
# are Windows-specific.
#
# They ran red on the ubuntu leg for three CI runs before anyone looked
# (t2608071205). Skipping is a deliberate coverage reduction, not a fix --
# t2608071221 tracks Linux being unexercised, and these are the first thing to
# revisit when a real Linux run becomes available.
windows_path_spelling = pytest.mark.skipif(
    sys.platform != "win32",
    reason="asserts Windows path spelling; revisit under t2608071221",
)


@windows_path_spelling
def test_a_project_path_becomes_project_relative():
    line = r"Writing geoms to C:\TESTS\CST\gabon_0108\hydrology_model\basins.geojson"
    assert _rel(line) == r"Writing geoms to hydrology_model\basins.geojson"


def test_a_repo_path_is_marked_rather_than_bared():
    """Marked so a repo-relative path and a project-relative one stay
    distinguishable in the same line."""
    from blueearth_cst.shared.snake_utils import _REPO_ROOT

    line = f"script at {_REPO_ROOT}{os.sep}blueearth_cst{os.sep}shared{os.sep}x.py"
    assert _rel(line) == f"script at <repo>/blueearth_cst{os.sep}shared{os.sep}x.py"


def test_a_path_outside_all_three_is_left_alone():
    """A staged data path's location IS the information — never shorten it."""
    line = r"Reading era5 from C:\data\wflow_global\hydromt\meteo\era5_daily.zarr"
    assert _rel(line) == line


@windows_path_spelling
def test_site_packages_is_matched_before_the_repo():
    """The pixi env lives INSIDE the repo, so the order is load-bearing: a
    repo-relative rewrite would otherwise hide the package name."""
    from blueearth_cst.shared.snake_utils import _REPO_ROOT

    line = f"x {_REPO_ROOT}{os.sep}.pixi{os.sep}envs{os.sep}default{os.sep}Lib{os.sep}site-packages{os.sep}hydromt{os.sep}a.py"
    assert _rel(line) == f"x <site-packages>/hydromt{os.sep}a.py"


@windows_path_spelling
def test_forward_slash_spelling_is_handled_too():
    """hydromt emits either separator."""
    line = "Writing to C:/TESTS/CST/gabon_0108/hydrology_model/basins.geojson"
    assert _rel(line) == "Writing to hydrology_model/basins.geojson"


def test_an_already_relative_line_is_untouched():
    line = "Parsing data catalog from config/catalogs/deltares_data.yml"
    assert _rel(line) == line


# ---------------------------------------------------------------------------
# R9 P2 commit 3: per-member pointer keying (the concurrency falsifier's cheap
# half). Removing the `rlz_<r>/` directory level puts every member's artifacts
# in ONE directory, and rule 3.10 batches members concurrently -- so the
# filename is now the only thing keeping them apart.
# ---------------------------------------------------------------------------

_MEMBER_CONFIG = "experiments/e/hydrology/wflow/config/{member}.toml"


def test_member_pointer_base_derives_the_stem_and_the_sibling_output_hop():
    run_name, out_prefix = su.member_pointer_base(
        _MEMBER_CONFIG.format(member="rlz_1_st_2")
    )
    assert run_name == "rlz_1_st_2"
    # Relative, POSIX, trailing-separated -- config/ -> sibling output/.
    assert out_prefix == "../output/"


def test_every_member_gets_a_distinct_log_and_output_pointer():
    """The property the flattening put at risk, asserted over a real grid.

    Pre-R9 each realization owned a directory, so wflow's default `log.txt`
    beside the TOML was one shared log PER REALIZATION -- measured on the P1
    observed tier as exactly two logs for twelve members. Flattening the level
    would make that one log for all twelve unless every pointer is keyed by
    member. Checked for the three member-keyed pointers at once.
    """
    members = [f"rlz_{r}_st_{c}" for r in (1, 2) for c in range(1, 7)]
    logs, csvs, states = set(), set(), set()
    for member in members:
        run_name, out_prefix = su.member_pointer_base(
            _MEMBER_CONFIG.format(member=member)
        )
        logs.add(f"{out_prefix}{run_name}.log")
        csvs.add(f"{out_prefix}{run_name}.csv")
        states.add(f"{out_prefix}outstates_{run_name}.nc")
    assert len(logs) == len(members), "two members would share one log"
    assert len(csvs) == len(members)
    assert len(states) == len(members)
    # ...and the realization index is genuinely back IN the filename, which is
    # what the R7 -> R9 inversion means.
    assert "../output/rlz_2_st_6.log" in logs


def test_the_log_pointer_is_keyed_the_same_way_as_the_other_two():
    """A guard on the guard: `path_log` must be derived, not hardcoded.

    If `downscale_climate_forcing.py` ever spells the log path literally
    instead of building it from `member_pointer_base`, this stays green while
    the race returns -- so the source is asserted too.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "blueearth_cst"
        / "experiment"
        / "downscale_climate_forcing.py"
    ).read_text(encoding="utf-8")
    assert '"logging.path_log": f"{out_prefix}{run_name}.log"' in src
    # Comments legitimately NAME the wflow default while explaining why it is
    # overridden, so strip them first: what must not reappear is `log.txt` as a
    # VALUE. A blunter substring check fails on the rationale for the fix.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert "log.txt" not in code, "the wflow default log name must not be a value"
