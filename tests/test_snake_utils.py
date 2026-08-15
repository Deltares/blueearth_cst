"""Exact-equivalence tests for the shared get_config helper (R3 §3, §8).

Pins the semantics the four inline get_config copies (three Snakefiles +
conftest) had before they were collapsed into blueearth_cst/shared/snake_utils.py, so the move
is provably identity-preserving rather than merely green on a smoke test.
"""

import gc
import io
import os
import re
import sys
import time
import warnings
import zlib
from pathlib import Path

import pytest

import blueearth_cst.shared.snake_utils as su  # noqa: E402
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    _compact_log_line,
    _cr_overwrite,
    _fmt_elapsed,
    _Heartbeat,
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
        "18:03:38 - model - Initializing wflow_sbm model.\n"
    )


def test_compact_preserves_dashes_in_message():
    line = (
        "2026-07-21 18:03:20,505 - hydromt.model.model - model - INFO - "
        "setup_rivers.river_routing=kinematic - wave - x\n"
    )
    # message (with its own ' - ') is kept whole; only ts + dotted name change
    assert _compact_log_line(line) == (
        "18:03:20 - model - setup_rivers.river_routing=kinematic - wave - x\n"
    )


def test_compact_keeps_level_and_no_trailing_newline():
    line = (
        "2026-07-21 18:03:18,884 - hydromt.hydromt_wflow.workflows.basemaps"
        " - basemaps - WARNING - Model resolution mismatch"
    )  # no trailing newline
    assert _compact_log_line(line) == (
        "18:03:18 - basemaps - WARNING - Model resolution mismatch"
    )


def test_compact_drops_a_component_prefix_that_repeats_the_module():
    """`geoms - INFO - wflow_sbm.geoms:` names one subsystem twice."""
    line = (
        "2026-08-14 11:13:01,204 - hydromt_wflow.wflow_base - geoms - INFO - "
        "wflow_sbm.geoms: Writing geoms to staticgeoms/basins.geojson.\n"
    )
    assert _compact_log_line(line) == (
        "11:13:01 - geoms - Writing geoms to staticgeoms/basins.geojson.\n"
    )


def test_compact_keeps_a_component_prefix_that_says_something_else():
    """`spatial` is hydromt's module, `staticmaps` the component -- two facts."""
    line = (
        "2026-08-14 11:13:01,204 - hydromt_wflow.wflow_base - spatial - INFO - "
        "wflow_sbm.staticmaps: Writing region to staticgeoms/region.geojson.\n"
    )
    assert _compact_log_line(line) == (
        "11:13:01 - spatial - wflow_sbm.staticmaps: Writing region to "
        "staticgeoms/region.geojson.\n"
    )


def test_compact_leaves_an_ordinary_colon_in_the_message_alone():
    """Only a dotted prefix in the leading position is a component prefix."""
    line = (
        "2026-08-14 11:13:01,204 - hydromt.model.model - model - INFO - "
        "Reading model config file from wflow_sbm.toml: found 4 sections\n"
    )
    assert _compact_log_line(line) == (
        "11:13:01 - model - Reading model config file from "
        "wflow_sbm.toml: found 4 sections\n"
    )


def test_compact_keeps_a_bare_prefix_with_no_message_after_it():
    """`wflow_sbm.geoms:` alone is the whole message; dropping it leaves nothing."""
    line = (
        "2026-08-14 11:13:01,204 - hydromt_wflow.wflow_base - geoms - INFO - "
        "wflow_sbm.geoms:\n"
    )
    assert _compact_log_line(line) == "11:13:01 - geoms - wflow_sbm.geoms:\n"


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


def _figure(tmp_path, name, subdir="plots"):
    """Write one throwaway figure through `save_figure`, returning its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_path / subdir / name
    plt.figure()
    plt.plot([0, 1], [0, 1])
    save_figure(str(out), dpi=50)
    return out


def test_save_figure_writes_creates_parent_and_announces(tmp_path, capsys):
    su._LAST_FIGURE_DIR = None
    out = _figure(tmp_path, "basin_area.png")  # parent does not exist yet
    assert out.exists()
    printed = capsys.readouterr().out.strip()
    assert printed.endswith(f"- plot - {out}"), printed


def test_save_figure_states_a_directory_once(tmp_path, capsys):
    """A plotting rule writes 9-17 figures into ONE directory."""
    su._LAST_FIGURE_DIR = None
    first = _figure(tmp_path, "hydrograph_1030.png")
    _figure(tmp_path, "performance_1030.png")
    _figure(tmp_path, "signatures_peaks_1030.png")
    rows = [line.split(" - ", 2)[2] for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        str(first),
        "  performance_1030.png",
        "  signatures_peaks_1030.png",
    ], rows


def test_save_figure_restates_the_directory_when_it_changes(tmp_path, capsys):
    """Elision must never make a row name a directory it did not go to."""
    su._LAST_FIGURE_DIR = None
    _figure(tmp_path, "a.png", subdir="plots")
    other = _figure(tmp_path, "b.png", subdir="maps")
    back = _figure(tmp_path, "c.png", subdir="plots")
    rows = [line.split(" - ", 2)[2] for line in capsys.readouterr().out.splitlines()]
    assert rows[1] == str(other) and rows[2] == str(back), rows


def test_a_new_log_restates_the_directory(tmp_path, capsys):
    """The invariant is per FILE: a log's first figure row always has its path."""
    su._LAST_FIGURE_DIR = None
    _figure(tmp_path, "first.png")
    capsys.readouterr()
    with tee_to_log(tmp_path / "logs" / "rule.log"):
        second = _figure(tmp_path, "second.png")
    body = (tmp_path / "logs" / "rule.log").read_text(encoding="utf-8")
    assert os.path.basename(str(second)) in body
    assert "  second.png" not in body  # not elided into a bare continuation row


# --- log_row -----------------------------------------------------------------


def test_log_row_standard_format(capsys):
    log_row("hello world", module="plot")
    out = capsys.readouterr().out.strip()
    assert re.match(r"^\d{2}:\d{2}:\d{2} - plot - hello world$", out)


def test_log_row_row_survives_compaction_unchanged():
    # a log_row line is already compact -> the tee's _compact_log_line is a no-op
    row = "21:56:12 - plot - plots/x.png\n"
    assert _compact_log_line(row) == row


# --- _log_row_text (the level is shown only when it is not INFO) --------------


def test_log_row_text_omits_info():
    """259 of 272 rows on a full WF1 build are INFO -- a constant column."""
    assert su._log_row_text("11:13:01", "geoms", "INFO", "writing") == (
        "11:13:01 - geoms - writing"
    )


@pytest.mark.parametrize("level", ["WARNING", "ERROR", "CRITICAL", "DEBUG"])
def test_log_row_text_keeps_every_other_level(level):
    """The point of dropping INFO is that these stop hiding among it."""
    assert su._log_row_text("11:13:04", "states", level, "cold start") == (
        f"11:13:04 - states - {level} - cold start"
    )


def test_log_row_text_recognizes_info_in_any_spelling():
    """A caller passing `info` must not produce a row shaped unlike its peers."""
    assert su._log_row_text("11:13:01", "cst", " info ", "x") == "11:13:01 - cst - x"


def test_log_row_prints_a_warning_with_its_level(capsys):
    out = capsys.readouterr()  # drain
    log_row("state file not found", module="states", level="WARNING")
    out = capsys.readouterr().out.strip()
    assert re.match(r"^\d{2}:\d{2}:\d{2} - states - WARNING - state file", out), out


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


# --- target_banner (rule `all` message) --------------------------------------


def test_target_banner_puts_one_target_per_line(monkeypatch):
    """The whole point: Snakemake's own `input:` joins with ", "; this does not."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())  # isatty() -> False
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("2.00", "all", ["a/x.csv", "b/y.png"])
    assert out == "Rule 2.00: all\n    a/x.csv\n    b/y.png"
    assert ", " not in out


def test_target_banner_accepts_a_dict_values_view(monkeypatch):
    """WF2 and WF3 pass `TARGETS.values()`, not a list."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    targets = {"a": "one.csv", "b": "two.csv"}
    assert target_banner("3.00", "all", targets.values()) == (
        "Rule 3.00: all\n    one.csv\n    two.csv"
    )


def test_target_banner_with_no_targets_is_just_the_banner(monkeypatch):
    """No trailing blank line -- an empty list must not print an empty row."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert target_banner("1.00", "all", []) == "Rule 1.00: all"


def test_target_banner_relativizes_against_project_dir(monkeypatch):
    """The root moves to the banner; the paths below it lose the prefix."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner(
        "2.00",
        "all",
        ["C:/TESTS/CST/gabonx/analyze_projections/cmip6/summary/x.csv"],
        "C:/TESTS/CST/gabonx",
    )
    assert out == (
        "Rule 2.00: all  [C:/TESTS/CST/gabonx]\n    analyze_projections/cmip6/summary/x.csv"
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
    assert out == "Rule 3.00: all\n    C:/p/q.csv"
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
    """Both spellings of the root are stripped, and the REMAINDER is normalized.

    The remainder used to keep whatever separator the producing library
    happened to use, so a single log carried two spellings of one tree and they
    read as different locations at a glance. Forward slashes on both platforms.
    """
    root = _abs("TESTS/gabon")
    native = f"Writing geoms to {root}{os.sep}hydrology_model{os.sep}basins.geojson.\n"
    assert _relativize_paths(native, root) == (
        "Writing geoms to hydrology_model/basins.geojson.\n"
    )
    fwd_root = root.replace(os.sep, "/")
    forward = f"Writing config to {fwd_root}/hydrology_model/wflow_sbm.toml.\n"
    assert _relativize_paths(forward, root) == (
        "Writing config to hydrology_model/wflow_sbm.toml.\n"
    )


def _abs(path):
    """A path that is genuinely absolute on BOTH platforms.

    `declare_path_tokens` stores `os.path.abspath(...)`, and a `C:/...` literal
    is NOT absolute on POSIX -- abspath silently prepends the runner's CWD, so
    the stored token stops matching the line under test and the tokenizer does
    nothing at all.

    CI caught this on 2026-08-14: three failures on ubuntu-latest, green on
    windows-latest. Worse than the three, and the reason every affected test is
    converted rather than only the ones that went red: two more passed for the
    WRONG REASON. Both assert that something is left UNCHANGED, which is
    exactly what a token that never matches produces -- so on Linux they were
    asserting the absence of an effect that could not have occurred.

    `C:` on Windows, `/` elsewhere; pass a root-relative path.
    """
    prefix = "C:/" if os.name == "nt" else "/"
    return os.path.normpath(prefix + str(path).lstrip("/"))


def test_abs_helper_survives_abspath_on_this_platform():
    """The property whose absence caused the 2026-08-14 ubuntu failures.

    `declare_path_tokens` stores `os.path.abspath(folder)`. If a test's literal
    is not already absolute, abspath rewrites it and the stored token no longer
    matches the line under test -- silently, because a token that matches
    nothing simply leaves the text alone.

    Asserting the fixed point is platform-independent, so this guard runs on
    BOTH legs and would have gone red on windows too had `_abs` been wrong
    there. A test that only checked `os.path.isabs` would not: `C:/x` IS
    absolute on Windows, which is precisely why the defect was invisible here.
    """
    for raw in ("TESTS/gabon", "data/wflow_global/hydromt"):
        got = _abs(raw)
        assert os.path.abspath(got) == got, (
            f"_abs({raw!r}) -> {got!r} is rewritten by abspath to "
            f"{os.path.abspath(got)!r}; a token declared from it would never match"
        )


@pytest.fixture
def declare_folders(monkeypatch):
    """Declare key folders with the process-wide effect undone afterwards.

    `declare_path_tokens` writes an env var by design -- a rule's output is
    written by a CHILD of the process that parses the Snakefile. Isolation is
    `monkeypatch.setenv`, NOT `delenv`: delenv on an absent variable records
    nothing to undo, so a declaration made after it leaks into every test that
    follows (three run-header tests, measured).
    """
    monkeypatch.setenv(su._PATH_TOKENS_ENV, "")
    return su.declare_path_tokens


def test_declared_folders_become_tokens_in_one_line(declare_folders):
    """The interleaved case: an external data root and a model dir in one line.

    This is the shape a build actually prints -- `Reading <x> data from <data
    root>` and `Writing grid data to <model dir>` -- and the one that breaks if
    tokens are applied after the project strip, because by then the model path
    has already lost the root a token is registered with.
    """
    root = _abs("TESTS/gabon")
    declare_folders(
        data=_abs("data/wflow_global/hydromt"),
        model=os.path.join(root, "models", "hydrology", "wflow"),
    )
    tokens = su._path_tokens()
    line = (
        f"copied {_abs('data/wflow_global/hydromt/rgi/rgi.gpkg')} "
        f"to {os.path.join(root, 'models', 'hydrology', 'wflow', 'staticmaps.nc')}\n"
    )
    assert su._relativize_paths(line, root, tokens) == (
        "copied <data>/rgi/rgi.gpkg to <model>/staticmaps.nc\n"
    )


def test_a_declared_folder_named_on_its_own_is_tokenized(declare_folders):
    """`Write model data to <model dir>` -- no trailing separator, and the one
    line that states the folder rather than something under it."""
    model = _abs("TESTS/gabon/models/hydrology/wflow")
    declare_folders(model=model)
    out = su._relativize_paths(f"Write model data to {model}\n", "", su._path_tokens())
    assert out == "Write model data to <model>\n"


def test_a_token_does_not_claim_a_sibling_that_starts_with_it(declare_folders):
    """`.../wflow` must not turn `.../wflow_extra` into `<model>_extra`."""
    model = _abs("TESTS/gabon/models/hydrology/wflow")
    declare_folders(model=model)
    out = su._relativize_paths(f"reading {model}_extra/x.nc\n", "", su._path_tokens())
    assert out == f"reading {model}_extra/x.nc\n"


def test_the_longest_declared_folder_wins(declare_folders):
    """An experiment dir sits under the project and a model dir can sit under
    either; a shorter root claiming a longer one would mislabel every path."""
    root = _abs("TESTS/gabon")
    declare_folders(
        climate=os.path.join(root, "data"),
        experiment=os.path.join(root, "data", "experiments", "exp_a"),
    )
    line = f"wrote {os.path.join(root, 'data', 'experiments', 'exp_a', 'q.csv')}\n"
    assert (
        su._relativize_paths(line, root, su._path_tokens())
        == "wrote <experiment>/q.csv\n"
    )


def test_declare_path_tokens_drops_what_a_workflow_does_not_have(declare_folders):
    """WF2 declares no model. A blank must not become a token matching everything."""
    tokens = declare_folders(data="", model=None, projections="proj/x")
    assert set(tokens) == {"projections"}
    assert os.path.isabs(tokens["projections"])


def test_path_tokens_fails_open_on_a_malformed_declaration(monkeypatch):
    """Every path prints in full -- which is what this mechanism improves on,
    so a broken variable can only cost the improvement, never the run."""
    for raw in ("", "not json", "[1, 2]", '{"model": 7}'):
        monkeypatch.setenv(su._PATH_TOKENS_ENV, raw)
        assert su._path_tokens() == ()


def test_a_relative_project_root_still_strips_an_absolute_path():
    """The shipped configs' shape, and it used to produce a path that lied.

    `project_dir` is relative in every shipped config (`test_case/test_rapid`)
    while hydromt resolves before it logs, so the text is absolute and the root
    is not. `_strip_prefix` is unanchored, so the root came out of the MIDDLE
    and left the head -- which the repo rewrite then labelled, turning
    `<...>/test_case/test_rapid/models/x.nc` into `<repo>/models/x.nc`: a path
    that does not exist, spelled like one that does.
    """
    root = os.path.normpath("test_case/test_rapid")
    absolute = os.path.abspath(os.path.join(root, "models", "hydrology", "wflow"))
    out = su._relativize_paths(f"Write model data to {absolute}\n", root)
    assert out == "Write model data to models/hydrology/wflow\n"
    assert "<repo>" not in out


def test_the_header_defines_a_token_under_a_relative_project_dir(declare_folders):
    """Same mismatch, one level up: tokens are stored ABSOLUTE.

    Stripping a relative root off an absolute token left the head in place, so
    the row offered `C:/.../pipeline/models/hydrology/wflow` as the definition
    of `<model>` -- pointing outside the project it belongs to.
    """
    declare_folders(model="test_case/test_rapid/models/hydrology/wflow")
    rows = su.run_header("wf1 build_model", "test_case/test_rapid").splitlines()
    assert rows[-1].split() == ["<model>", "models/hydrology/wflow"]


def test_relativize_leaves_out_of_project_paths_absolute():
    root = _abs("TESTS/gabon")
    line = f"Reading data from {_abs('data/wflow_global/x.tif')}\n"
    assert _relativize_paths(line, root) == line  # not under project -> untouched


def test_tee_to_log_relativizes_project_paths(tmp_path):
    proj = tmp_path / "gabon"
    log = proj / "logs" / "1.15_plot_wflow_evaluation.log"
    abs_png = os.path.join(str(proj), "plots", "map.png")
    with tee_to_log(log):
        print(abs_png)
    text = log.read_text(encoding="utf-8")
    # Forward slash regardless of platform: the stripped remainder is normalized.
    assert "plots/map.png" in text
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
    assert len(re.findall(r"\d{2}:\d{2}:\d{2} - \w+ - built model grid", body)) == 1
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", body)


def test_tee_to_log_compacts_hydromt_format(tmp_path):
    log = tmp_path / "rule.log"
    with tee_to_log(log):
        # a hydromt-format record (as hydromt's Python API emits) and a plain line
        print("2026-07-21 18:03:38,474 - hydromt.model.model - model - INFO - built")
        print("plain progress line")
    text = log.read_text(encoding="utf-8")
    # the record row is exactly the compacted form: HH:MM:SS, no date/ms/name
    row = next(line for line in text.splitlines() if line.endswith("- built"))
    assert row == "18:03:38 - model - built"
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
    assert head[3].startswith("# rows:")  # the row-grammar legend
    assert head[4] == ""  # blank line separates header from body
    assert head[5] == "body line"


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


# --- failure tracebacks land in the log part (t2608071219 / [R10-13]) --------
#
# Snakemake prints `check log file(s) for error details` and nothing else when a
# `script:` rule fails. Before this, the named log stopped mid-rule with no
# reason: the traceback reached the interactive console and was absent from the
# artifact a user would send you, so the message actively misdirected.


def test_tee_to_log_writes_the_traceback_into_the_log(tmp_path):
    log = tmp_path / "rule.log"
    with pytest.raises(KeyError):
        with tee_to_log(log):
            print("progress line")
            raise KeyError("missing_column")
    text = log.read_text(encoding="utf-8")
    assert "progress line" in text  # body still there
    assert "Traceback (most recent call last):" in text
    assert "KeyError" in text and "missing_column" in text


def test_the_logged_traceback_carries_the_exception_message(tmp_path):
    """[R10-13]'s specific ask: the useful part must survive, not just a type.

    `check_model_reference` raises `ModelDriftError` naming the changed input,
    and that naming is the part an operator needs. It rides in ``str(exc)``, so
    the formatted traceback carries it -- asserted here so a future exception
    that hides its detail in unrendered attributes fails this instead of quietly
    reinstating the useless-log problem.
    """

    class ModelDriftLike(RuntimeError):
        pass

    log = tmp_path / "rule.log"
    with pytest.raises(ModelDriftLike):
        with tee_to_log(log):
            raise ModelDriftLike("staticmaps.nc changed since the reference was taken")
    assert "staticmaps.nc changed since the reference was taken" in log.read_text(
        encoding="utf-8"
    )


def test_a_clean_systemexit_writes_no_traceback(tmp_path):
    """The regression this fix could cause, on the busiest path in the repo.

    Every WF2 cache-hit job leaves its body via ``raise SystemExit(0)`` -- see
    ``_is_clean_exit``, which exists because treating that as a failure printed
    ``failed after Ns`` for jobs Snakemake then reported as ``Finished``. A
    SystemExit(0) is a SUCCESS and must leave the log traceback-free.
    """
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log):
            print("cache hit, nothing to do")
            raise SystemExit(0)
    text = log.read_text(encoding="utf-8")
    assert "cache hit, nothing to do" in text
    assert "Traceback" not in text
    assert "SystemExit" not in text


def test_a_nonzero_systemexit_is_a_failure_and_is_logged(tmp_path):
    """Only the exit CODE decides -- ``SystemExit(1)`` is a real failure."""
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log):
            raise SystemExit(1)
    text = log.read_text(encoding="utf-8")
    assert "Traceback" in text and "SystemExit" in text


def test_the_logged_traceback_has_paths_relativized(tmp_path):
    """The traceback is written straight to the file, bypassing the tee.

    So it must be relativized here, or the log would mix shortened paths in its
    body with full absolute ones in its traceback. The repo root is the prefix
    every frame of an in-repo traceback carries.
    """
    log = tmp_path / "rule.log"
    with pytest.raises(ValueError):
        with tee_to_log(log):
            raise ValueError("boom")
    text = log.read_text(encoding="utf-8")
    assert "Traceback" in text
    assert str(su._REPO_ROOT) not in text, "absolute repo paths must not reach the log"
    assert "<repo>" in text


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
    #
    # The sleep is REQUIRED, not padding: the success verdict is printed only
    # once the watchdog has beeped (`_Heartbeat.stop`), so a job that returns
    # before the first interval prints nothing at all and this test would assert
    # on whether the machine happened to be slow.
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log, heartbeat_interval=0.05):
            time.sleep(0.16)
            raise SystemExit(0)
    err = capsys.readouterr().err
    assert "done in" in err and "failed after" not in err


def test_tee_mutes_the_catalog_line_on_the_console_but_keeps_it_in_the_log(
    tmp_path, capsys
):
    """Nine parallel jobs read the same two catalogs and say so eighteen times.

    Each job is its own process, so no in-process buffer can collapse them --
    muting the console copy is the only lever, and the log part is where the
    question "what did this job read?" is actually answered.
    """
    # Written as `logging.StreamHandler.emit` writes -- `msg + terminator` in ONE
    # call -- because that is the emitter this mute exists for, and it is the
    # only shape the predicate accepts. `print` splits the text and the newline
    # into two writes, which is a partial chunk and therefore prints, by design.
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(
            "14:02:03 - data_catalog - Parsing data catalog from a/b.yml\n"
        )
        sys.stdout.write("14:02:03 - data_catalog - Resolved 12 sources\n")
    out = capsys.readouterr().out
    logged = log.read_text(encoding="utf-8")
    assert "Parsing data catalog from" not in out
    assert "Parsing data catalog from a/b.yml" in logged
    # Only the muted row goes; its neighbours from the same module stay.
    assert "Resolved 12 sources" in out and "Resolved 12 sources" in logged


def test_tee_mute_never_silences_a_warning_or_a_partial_write(tmp_path, capsys):
    """A prefix muted for VOLUME must not be able to hide a raised level.

    `_log_row_text` renders INFO by omitting the level, so a WARNING row carries
    a third field and cannot match. A chunk that is not exactly one terminated
    line cannot match either -- a tee is handed chunks, not lines.
    """
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(
            "14:02:03 - data_catalog - WARNING - "
            "Parsing data catalog from a/b.yml failed\n"
        )
        sys.stdout.write("14:02:03 - data_catalog - Parsing data catalog from c.yml")
        sys.stdout.write("\n")
    out = capsys.readouterr().out
    assert "WARNING" in out and "failed" in out
    assert "Parsing data catalog from c.yml" in out  # split write, so not matched


def test_heartbeat_says_nothing_when_a_quiet_job_simply_finishes(tmp_path, capsys):
    """The DONE line already carries this job's identity and its duration.

    A watchdog that never beeped has nothing to close, so its `done in` was a
    second duration in a second grammar for every member of a fan-out -- and
    printed from the job's own process, so it did not even land beside the
    Snakemake line it repeated.
    """
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=10.0):  # never fires
        print("real output the console still wants")
    err = capsys.readouterr()
    assert "done in" not in err.err and "done in" not in err.out
    assert "real output the console still wants" in err.out


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
    # `basin_cells` joined the contract on 2026-08-10 and is source-independent:
    # every extraction carries the mask, the chirps branch adds orography on top.
    assert list(spec.outputs) == ["climate_nc", "basin_cells", "oro_nc"]


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
    assert _rel(line) == "Writing geoms to hydrology_model/basins.geojson"


def test_a_repo_path_is_marked_rather_than_bared():
    """Marked so a repo-relative path and a project-relative one stay
    distinguishable in the same line."""
    from blueearth_cst.shared.snake_utils import _REPO_ROOT

    line = f"script at {_REPO_ROOT}{os.sep}blueearth_cst{os.sep}shared{os.sep}x.py"
    assert _rel(line) == "script at <repo>/blueearth_cst/shared/x.py"


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


# --- the tee outlives its log file (2026-08-10) -------------------------------


def test_a_tee_write_after_close_does_not_raise(tmp_path):
    """A tee OUTLIVES its log file, and a late write must not blow up.

    `tee_to_log` closes the log when its `with open(...)` exits; anything still
    holding the tee then points at a dead sink. Raising there is the expensive
    kind of failure: these writes happen during interpreter finalization, where
    the exception cannot be reported and CPython prints the bare
    `Error in sys.excepthook:` / `Original exception was:` pair instead.
    """
    from blueearth_cst.shared.snake_utils import _Tee

    console = io.StringIO()
    handle = open(tmp_path / "t.log", "w", encoding="utf-8")
    tee = _Tee(console, handle)
    tee.write("during\n")
    tee.close()
    handle.close()

    tee.write("after close\n")  # must not raise
    tee.flush()  # logging.shutdown() flushes every handler at exit
    assert "after close" in console.getvalue()  # console is still the right sink


def test_close_is_idempotent(tmp_path):
    from blueearth_cst.shared.snake_utils import _Tee

    handle = open(tmp_path / "t2.log", "w", encoding="utf-8")
    tee = _Tee(io.StringIO(), handle)
    tee.close()
    handle.close()
    tee.close()  # second close must not touch the closed handle


def test_a_handler_created_inside_the_block_is_not_left_on_the_tee(tmp_path):
    """The snapshot `_redirect_console_log_handlers` takes cannot see it.

    Libraries configure logging lazily inside the rule body -- hydromt installs
    a StreamHandler per data catalog -- binding to whatever `sys.stdout` was at
    that moment, which is the tee. Nothing restored those, so they outlived the
    log file they wrote into.
    """
    import logging

    from blueearth_cst.shared.snake_utils import _Tee, tee_to_log

    logger = logging.getLogger("cst_probe_lazy_handler")
    logger.handlers.clear()
    try:
        with tee_to_log(tmp_path / "h.log", heartbeat_interval=0):
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            assert isinstance(handler.stream, _Tee)  # bound to the tee, as feared
        assert not isinstance(handler.stream, _Tee), (
            "handler still points at the tee after the block"
        )
        logger.warning("must not raise")  # the record has somewhere real to go
    finally:
        logger.handlers.clear()


def test_a_reference_cycle_from_the_body_is_collected_on_the_way_out(tmp_path):
    """`tee_to_log` collects cycles while the interpreter is still healthy.

    hydromt's catalog and model objects reference each other, so what holds a
    rule's GDAL/rasterio handles is a reference CYCLE — freed only by the cyclic
    collector. Left to interpreter finalization, tearing those handles down at
    once on Windows makes a stderr write fail, and CPython prints the bare
    `Error in sys.excepthook:` / `Original exception was:` pair with empty
    bodies after a rule that SUCCEEDED (rules 1.03/1.04/1.07, observed
    2026-08-11).

    A rule body's catalog is a frame local of the function it called, so it is
    unreachable-but-uncollected by the time this block exits — exactly what the
    collect is for. Modelled here with a self-referencing object holding a
    finalizer, which only the cyclic collector can reach.
    """
    from blueearth_cst.shared.snake_utils import tee_to_log

    released = []

    class _CatalogLike:
        """Self-referencing, like a hydromt catalog <-> model pair."""

        def __init__(self):
            self.self_ref = self  # the cycle: refcounting alone never frees it

        def __del__(self):
            released.append("closed")

    def rule_body():
        _CatalogLike()  # a frame local, dropped when this function returns

    gc.disable()  # so nothing collects it incidentally before the block exits
    try:
        with tee_to_log(tmp_path / "cycle.log", heartbeat_interval=0):
            rule_body()
            assert not released, "refcounting alone must not free a cycle"
        assert released == ["closed"], (
            "the cycle survived tee_to_log and would be torn down during "
            "interpreter finalization instead"
        )
    finally:
        gc.enable()


# --- shared.seed resolution ------------------------------------------------
#
# One seed for every stochastic step, defaulted in advanced_settings.yml and
# overridable per project. `auto` derives from the experiment name so that a
# re-run of one experiment is bit-identical while a different experiment gets
# a different draw -- the property the whole design rests on.


def test_seed_absent_takes_the_advanced_settings_default():
    assert su.resolve_seed(None, "any_experiment") == su.DEFAULT_SEED


def test_seed_accepts_an_explicit_integer():
    assert su.resolve_seed(7, "any_experiment") == 7


def test_seed_accepts_zero():
    """0 is a legitimate seed; `_positive_int` would have rejected it."""
    assert su.resolve_seed(0, "any_experiment") == 0


@pytest.mark.parametrize("token", ["auto", "AUTO", " auto "])
def test_seed_auto_is_case_and_space_insensitive(token):
    assert su.resolve_seed(token, "exp_a") == su.derive_seed("exp_a")


def test_seed_auto_is_stable_for_one_experiment():
    """The idempotence property: same name in, same seed out, every process.

    If this ever fails, WF3 stops converging -- rule 3.10 rewrites its output,
    3.11 regenerates every realization, and every wflow run below re-executes.
    """
    assert su.derive_seed("gabon_20260812") == su.derive_seed("gabon_20260812")


def test_seed_auto_differs_between_experiments():
    assert su.derive_seed("gabon_20260812") != su.derive_seed("gabon_20260901")


def test_seed_auto_does_not_use_the_salted_builtin_hash():
    """`hash()` is salted per process by PYTHONHASHSEED, so a derivation built
    on it would return a different seed in every interpreter while looking
    reproducible within one. Pin the actual value to catch that substitution."""
    assert su.derive_seed("experiment") == zlib.crc32(b"experiment") % 2**31


def test_seed_auto_fits_r_integer_range():
    """R's integer tops out at 2**31 - 1 and set.seed takes an integer; a wider
    value would arrive as a double and warn or truncate."""
    for name in ("a", "experiment", "gabon_20260812", "x" * 500):
        assert 0 <= su.derive_seed(name) <= 2**31 - 1


@pytest.mark.parametrize("bad", ["random", "123", "", "auto-ish"])
def test_seed_refuses_a_string_that_is_not_auto(bad):
    """Refused rather than coerced: `"123"` would work by accident and
    `random` would reach weathergenr as NULL."""
    with pytest.raises(ValueError, match="shared.seed"):
        su.resolve_seed(bad, "exp")


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_seed_refuses_non_integers_and_negatives(bad):
    with pytest.raises(ValueError, match="shared.seed"):
        su.resolve_seed(bad, "exp")


def test_seed_auto_needs_an_experiment_name():
    with pytest.raises(ValueError, match="experiment_name"):
        su.resolve_seed("auto", "")


# --- stress_test spell factors ---------------------------------------------
#
# Moved out of the weathergen template into the project config, beside the two
# perturbation axes. The LENGTH check is the point: weathergenr indexes these
# by month, so R would recycle or truncate a wrong-length list rather than
# reject it, and the run would silently perturb the wrong months.


def test_spell_factor_absent_is_the_identity():
    assert su.validate_spell_factor(None, "x") == [1.0] * 12


def test_spell_factor_accepts_twelve_numbers_and_floats_them():
    out = su.validate_spell_factor([1] * 12, "x")
    assert out == [1.0] * 12
    assert all(isinstance(v, float) for v in out)


@pytest.mark.parametrize("n", [0, 11, 13, 24])
def test_spell_factor_refuses_a_wrong_length(n):
    with pytest.raises(ValueError, match="12 entries"):
        su.validate_spell_factor([1.0] * n, "stress_test.dry_spell_factor")


@pytest.mark.parametrize("bad", ["1.0", 1.0, 5, {"jan": 1.0}])
def test_spell_factor_refuses_a_non_list(bad):
    with pytest.raises(ValueError, match="12 monthly coefficients"):
        su.validate_spell_factor(bad, "stress_test.dry_spell_factor")


def test_spell_factor_refuses_a_non_numeric_entry():
    values = [1.0] * 12
    values[6] = "high"
    with pytest.raises(ValueError, match=r"\[7\]"):
        su.validate_spell_factor(values, "stress_test.wet_spell_factor")


# --- stress_test axis sub-key closure (defect D) ----------------------------


def test_stress_test_refuses_temperature_variance():
    """Owner ruling 2026-08-13: temperature variance is not a dimension.

    It was ACCEPTED and silently ignored -- only `precip.variance` reaches the
    generator. The axis guard checked top-level axes only, so a sub-key of a
    valid axis passed unexamined and a user could configure it and get
    unchanged results.
    """
    cfg = {
        "temp": {
            "step_num": 1,
            "transient_change": True,
            "variance": {"min": [1.0] * 12, "max": [1.0] * 12},
        },
        "precip": {"step_num": 1, "transient_change": True},
    }
    with pytest.raises(ValueError, match="not a supported stress dimension"):
        su.stress_test_grid(cfg)


def test_stress_test_refuses_a_typo_in_an_axis():
    """The closure is general, not a one-off ban on `variance`."""
    cfg = {
        "temp": {"step_num": 1, "transient_change": True, "meen": {}},
        "precip": {"step_num": 1, "transient_change": True},
    }
    with pytest.raises(ValueError, match=r"unsupported key\(s\) \['meen'\]"):
        su.stress_test_grid(cfg)


def test_precip_variance_is_still_accepted():
    """Only TEMPERATURE variance is refused; precip variance is live."""
    cfg = {
        "temp": {"step_num": 1, "transient_change": True},
        "precip": {
            "step_num": 1,
            "transient_change": True,
            "variance": {"min": [1.0] * 12, "max": [1.0] * 12},
        },
    }
    assert su.stress_test_grid(cfg) == (2, 2, 4)


# --- one wflow_outvars default (defect A) -----------------------------------


def test_wflow_outvars_default_is_not_empty():
    """`[]` meant zero indicator tables and no error.

    WF3 defaulted to it while WF1 defaulted to two variables, so a config
    omitting the key ran to completion and wrote nothing --
    `snake_config_baseline_linux.yml` omits it, so that was shipped.
    """
    assert su.DEFAULT_WFLOW_OUTVARS
    assert "river discharge" in su.DEFAULT_WFLOW_OUTVARS


# --- console style (_ConsoleHandler / install_console_style) -----------------
#
# Every case here is a synthetic LogRecord carrying the `extra=` shape
# Snakemake's own emitters attach (`scheduler.py`, `workflow.py`, `dag.py`), so
# the grammar is pinned without a pipeline run. Events are plain strings on
# purpose: `LogEvent` is a StrEnum, so `LogEvent.JOB_INFO == "job_info"`, and
# the handler never imports it.

import logging as _logging  # noqa: E402 -- module-level imports are grouped above


def _console_handler():
    """A handler over a StringIO, standing in for Snakemake's own."""
    base = _logging.StreamHandler(io.StringIO())
    base.name = "DefaultStreamHandler"
    base.setFormatter(_logging.Formatter("%(message)s"))
    return su._ConsoleHandler(base)


def _console_record(msg="", level=_logging.INFO, **extra):
    record = _logging.LogRecord("snakemake", level, __file__, 0, msg, (), None)
    record.__dict__.update(extra)
    return record


def _job_info(jobid, rule_name, rule_msg, wildcards=None):
    return _console_record(
        event="job_info",
        jobid=jobid,
        rule_name=rule_name,
        rule_msg=rule_msg,
        wildcards=dict(wildcards or {}),
    )


def _emit(handler, *records):
    for record in records:
        handler.emit(record)
    return handler.stream.getvalue()


def test_console_start_line_is_one_line_with_a_short_stamp():
    """Snakemake spends three lines here: blank, `[Thu Aug 13 ...]`, `Job 8: ...`."""
    handler = _console_handler()
    out = _emit(
        handler, _job_info(8, "run_wflow", "Rule 1.14: run_wflow - run the model")
    )
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d - RUN  Rule 1\.14: run_wflow - run the model\n", out
    ), out


def test_console_marker_column_is_the_same_on_a_start_and_a_finish():
    """`RUN ` is padded to `DONE`'s width, so both identities start at one column.

    Asserted on rendered lines rather than on the constants, because the padding
    only pays off if nothing between the stamp and the marker differs either.
    """
    su._RULE_NUMBERS["seed"] = "9.01"
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "seed", "Rule 9.01: seed"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=2),
    )
    start, finish = out.splitlines()
    assert start.index("Rule 9.01:") == finish.index("Rule 9.01:"), out


def test_console_finish_line_carries_number_wildcards_elapsed_and_counter():
    """The finish record names only a jobid, so every other field is recovered."""
    su._RULE_NUMBERS["downscale_climate_realization"] = "3.14"
    handler = _console_handler()
    _emit(
        handler,
        _job_info(
            3,
            "downscale_climate_realization",
            "Rule 3.14: downscale_climate_realization  [rlz 1 | st 0]",
            {"rlz": "1", "st": "0"},
        ),
    )
    # Backdate the memoized start rather than sleeping: the duration is the
    # point of the assertion, and 71 seconds of it is not affordable in a test.
    name, wildcards, started = handler._started[3]
    handler._started[3] = (name, wildcards, started - 71)
    out = _emit(
        handler,
        _console_record(event="job_finished", job_id=3),
        _console_record(event="progress", done=27, total=37),
    )
    done = out.splitlines()[1]
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d - DONE Rule 3\.14: downscale_climate_realization  "
        r"\[rlz 1 \| st 0\]  0:01:11  \[27/37\]",
        done,
    ), done


def test_console_finish_line_uses_the_banner_wildcard_grammar():
    """`[rlz 1 | st 0]`, not Snakemake's `rlz=1, st=0` -- one grammar, two lines."""
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "x", {"rlz": "1", "st": "0"}),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "[rlz 1 | st 0]" in out and "rlz=1" not in out


def test_console_finish_line_drops_the_num_suffix_the_banner_drops():
    """`rlz_num` is the WILDCARD's name; `rlz` is the console's, on both lines.

    WF3's banners write `rlz {wildcards.rlz_num}`, so a finish line rendering
    the raw key put `[rlz_num 1 | st_num 2]` directly under `[rlz 1 | st 2]` --
    one fact in two spellings, which is what this grammar exists to prevent.
    """
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "x", {"rlz_num": "1", "st_num": "2"}),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "[rlz 1 | st 2]" in out and "rlz_num" not in out


def test_console_finish_line_keeps_a_wildcard_actually_named_num():
    """Only a TRAILING `_num` with something in front of it is a suffix."""
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "x", {"num": "7", "num_bands": "3"}),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "[num 7 | num_bands 3]" in out


def test_console_a_wave_of_finishes_counts_up_to_the_progress_total():
    """Three held lines flushed at `done=15` are 13, 14, 15 -- in finish order."""
    handler = _console_handler()
    records = []
    for jobid in (10, 11, 12):
        records.append(
            _job_info(jobid, f"rule_{jobid}", f"Rule 9.0{jobid}: rule_{jobid}")
        )
    for jobid in (10, 11, 12):
        records.append(_console_record(event="job_finished", job_id=jobid))
    records.append(_console_record(event="progress", done=15, total=37))
    out = _emit(handler, *records)
    finished = [line for line in out.splitlines() if " - DONE " in line]
    assert [line.split("  ")[-1] for line in finished] == [
        "[13/37]",
        "[14/37]",
        "[15/37]",
    ], finished
    assert [line.split(" - DONE ")[1].split("  ")[0] for line in finished] == [
        "rule_10",
        "rule_11",
        "rule_12",
    ]


def test_console_a_held_finish_is_flushed_uncounted_if_progress_never_comes():
    """A held line must not be lost when the next record is not a progress one."""
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(4, "generate_weather_realizations", "Rule 3.11: generate"),
        _console_record(event="job_finished", job_id=4),
        _job_info(5, "next_rule", "Rule 3.12: next"),
    )
    lines = out.splitlines()
    assert " - DONE " in lines[1] and "[" not in lines[1], lines
    assert " - RUN  Rule 3.12: next" in lines[2]


def test_console_progress_alone_prints_nothing():
    """The counter is a property of a finish line, never a line of its own."""
    handler = _console_handler()
    assert _emit(handler, _console_record(event="progress", done=1, total=9)) == ""


def test_console_scheduler_chatter_is_muted():
    handler = _console_handler()
    muted = [
        _console_record("Select jobs to execute..."),
        _console_record("Execute 8 jobs...", event="job_started", jobs=[1]),
        _console_record("Removing temporary output x/y/rlz_1_st_0.nc."),
        _console_record("Touching output file x/.model_reference_ok."),
    ]
    assert _emit(handler, *muted) == ""


def test_console_the_preamble_is_left_alone():
    """It is printed before `onstart`, so muting it would be a dead rule."""
    handler = _console_handler()
    out = _emit(
        handler, _console_record("Assuming unrestricted shared filesystem usage.")
    )
    assert out == "Assuming unrestricted shared filesystem usage.\n"


def test_console_muting_never_reaches_a_warning_or_an_error():
    """A prefix match must not be able to silence a diagnostic."""
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record("Select jobs to execute...", level=_logging.WARNING),
        _console_record("Removing temporary output boom.", level=_logging.ERROR),
    )
    assert out.splitlines() == [
        "Select jobs to execute...",
        "Removing temporary output boom.",
    ]


def test_console_unknown_records_are_delegated_verbatim():
    """Errors and anything unrecognized keep Snakemake's own formatting."""
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record("Building DAG of jobs..."),
        _console_record("Error in rule x:\n    jobid: 3", event="job_error", jobid=3),
    )
    assert out == "Building DAG of jobs...\nError in rule x:\n    jobid: 3\n"


def test_console_a_rule_without_a_message_keeps_the_default_block():
    """Its input/output/jobid block is all a reader gets for that rule."""
    handler = _console_handler()
    record = _job_info(2, "some_rule", "")
    record.msg = "rule some_rule:\n    output: a.nc"
    assert _emit(handler, record) == "rule some_rule:\n    output: a.nc\n"


def test_console_a_rule_without_a_message_still_gets_a_named_finish_line():
    """Delegating the START line must not cost the job its identity at the END."""
    handler = _console_handler()
    record = _job_info(2, "some_rule", "")
    record.msg = "rule some_rule:"
    out = _emit(
        handler,
        record,
        _console_record(event="job_finished", job_id=2),
        _console_record(event="progress", done=9, total=10),
    )
    assert out.splitlines()[-1].endswith("- DONE some_rule  [9/10]"), out


def test_console_a_sub_second_job_shows_no_duration():
    """`0:00:00` reads as a broken clock, and bookkeeping rules are most of them."""
    su._RULE_NUMBERS["write_experiment_config"] = "3.07"
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "write_experiment_config", "Rule 3.07: write_experiment_config"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=3, total=37),
    )
    done = out.splitlines()[1]
    assert done.endswith("Rule 3.07: write_experiment_config  [3/37]"), done


def test_console_no_escape_codes_when_the_stream_is_not_a_tty():
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "Rule 1.01: r"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "\033" not in out


def test_console_a_finish_with_no_start_falls_back_to_snakemakes_own_text():
    """`--quiet rules` drops the start record; the finish must still name the rule.

    Rendering `done  job 5` there would be strictly worse than what Snakemake
    prints unaided, which is the one outcome a console layer must not produce.
    """
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            "Finished jobid: 5 (Rule: seed)", event="job_finished", job_id=5
        ),
        _console_record(event="progress", done=4, total=10),
    )
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d - DONE Finished jobid: 5 \(Rule: seed\)  \[4/10\]\n", out
    ), out


def test_console_the_start_memo_is_bounded():
    """`--quiet progress` drops every finish, so nothing would ever pop an entry."""
    handler = _console_handler()
    for jobid in range(su._CONSOLE_MAX_TRACKED_JOBS + 50):
        handler.emit(_job_info(jobid, "member", f"Rule 9.02: member  n {jobid}"))
    assert len(handler._started) == su._CONSOLE_MAX_TRACKED_JOBS
    # The oldest go first: the most recent job is always still tracked.
    assert su._CONSOLE_MAX_TRACKED_JOBS + 49 in handler._started
    assert 0 not in handler._started


class _TTYStringIO(io.StringIO):
    """A StringIO the console handler treats as a terminal."""

    def isatty(self):
        return True


def _colour_handler():
    base = _logging.StreamHandler(_TTYStringIO())
    base.name = "DefaultStreamHandler"
    base.setFormatter(_logging.Formatter("%(message)s"))
    return su._ConsoleHandler(base)


def test_rule_banner_never_colours_even_on_a_tty():
    """The banner string reaches a log file and an error block, not just a
    console -- so colour belongs to whoever writes the line, not to this."""
    import blueearth_cst.shared.snake_utils as _su

    real = sys.stderr
    sys.stderr = _FakeTTY()
    try:
        out = _su.rule_banner("1.09", "run_wflow", summary="run it", context="rlz 1")
    finally:
        sys.stderr = real
    assert out == "Rule 1.09: run_wflow - run it  [rlz 1]"
    assert "\033" not in out


def test_console_paints_a_start_and_a_finish_in_two_different_tiers():
    """Three tiers by LINE KIND -- started, finished, and everything between.

    Written against the CONSTANTS, not against the codes they hold. The three
    values are documented as the one place to swap when a terminal renders them
    badly, and a test spelling `\\033[94m` would make that swap a test failure --
    which is how a cosmetic knob acquires a gate it does not deserve. What must
    hold is structural: each line wrapped WHOLE, one pair of escapes, and a
    start distinguishable from a finish.
    """
    handler = _colour_handler()
    su._RULE_NUMBERS["seed"] = "9.01"
    out = _emit(
        handler,
        _job_info(1, "seed", "Rule 9.01: seed - prepare one realization"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=4),
    )
    run_line, done_line = out.splitlines()
    assert su._ANSI_RUN != su._ANSI_DONE != su._ANSI_BODY
    for line, code in ((run_line, su._ANSI_RUN), (done_line, su._ANSI_DONE)):
        opener = f"\033[{code}m"
        assert line.startswith(opener) and line.endswith("\033[0m"), line
        assert "\033" not in line[len(opener) : -4], line


def test_console_paints_an_informational_snakemake_line_in_the_body_tier():
    handler = _colour_handler()
    out = _emit(handler, _console_record("Building DAG of jobs..."))
    assert out == f"\033[{su._ANSI_BODY}mBuilding DAG of jobs...\033[0m\n", repr(out)


def test_console_never_recolours_a_warning_or_an_error():
    """Snakemake's own red must stay the loudest thing on the console."""
    handler = _colour_handler()
    out = _emit(
        handler,
        _console_record("state file missing", level=_logging.WARNING),
        _console_record("Error in rule x:", level=_logging.ERROR, event="job_error"),
    )
    assert "\033" not in out, repr(out)


def test_rule_banner_registers_its_number_for_the_finish_line(monkeypatch):
    """The finish record carries a rule NAME only; the number comes from here."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rule_banner("2.07", "a_freshly_named_rule")
    assert su._RULE_NUMBERS["a_freshly_named_rule"] == "2.07"


def test_rule_banner_registers_its_summary_for_the_once_per_rule_trim(monkeypatch):
    """The console trims by the EXACT clause inserted here, never by parsing."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rule_banner("2.08", "a_summarized_rule", summary="do the slow thing")
    assert su._RULE_SUMMARIES["a_summarized_rule"] == "do the slow thing"
    # A rule without one stays absent, so the trim has nothing to remove.
    rule_banner("2.09", "a_bare_rule")
    assert "a_bare_rule" not in su._RULE_SUMMARIES


def test_rule_banner_brackets_its_context():
    """Without ANSI the fields rest on whitespace alone once the summary goes."""
    assert rule_banner("3.14", "downscale", context="rlz 1 | st 0").endswith(
        "  [rlz 1 | st 0]"
    )


def test_console_prints_a_rule_summary_once_and_trims_it_from_the_fan_out():
    """A constant clause on 400 fan-out lines is the widest column saying least.

    The messages come from `rule_banner` itself, and the registry it fills is
    not written by hand here. The mechanism IS the agreement between the two --
    the handler removes the exact substring the banner inserted -- so a test
    that typed both sides would keep passing after they drifted apart.
    """
    handler = _console_handler()

    def banner(context):
        return rule_banner(
            "3.14",
            "downscale_climate_realization",
            context,
            summary="downscale the climate",
        )

    out = _emit(
        handler,
        _job_info(1, "downscale_climate_realization", banner("rlz 2 | st 0")),
        _job_info(2, "downscale_climate_realization", banner("rlz 2 | st 2")),
        _job_info(3, "downscale_climate_realization", banner("rlz 2 | st 3")),
    )
    first, second, third = out.splitlines()
    assert first.endswith(
        "RUN  Rule 3.14: downscale_climate_realization "
        "- downscale the climate  [rlz 2 | st 0]"
    ), first
    for line, context in ((second, "[rlz 2 | st 2]"), (third, "[rlz 2 | st 3]")):
        assert line.endswith(
            f"RUN  Rule 3.14: downscale_climate_realization  {context}"
        ), line
        # The identity survives: one grep on `3.14` still finds every member.
        assert "downscale the climate" not in line, line


def test_console_summary_trim_is_per_rule_not_global():
    """Each rule gets its own first line; one rule's does not consume another's."""
    su._RULE_SUMMARIES["rule_a"] = "do a"
    su._RULE_SUMMARIES["rule_b"] = "do b"
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "rule_a", "Rule 9.01: rule_a - do a"),
        _job_info(2, "rule_b", "Rule 9.02: rule_b - do b"),
        _job_info(3, "rule_a", "Rule 9.01: rule_a - do a"),
    )
    first, second, third = out.splitlines()
    assert first.endswith("Rule 9.01: rule_a - do a"), first
    assert second.endswith("Rule 9.02: rule_b - do b"), second
    assert third.endswith("Rule 9.01: rule_a"), third


def test_console_summary_seen_set_is_per_handler_not_module_level():
    """`run_workflows.py` drives four Snakefiles; the registries outlive one run.

    A module-level seen-set would leave the second and later workflows with a
    console on which no summary was ever printed -- so the state that says
    "already shown" belongs to the handler, like `_started`.
    """
    su._RULE_SUMMARIES["shared_rule"] = "do the thing"
    banner = "Rule 9.03: shared_rule - do the thing"
    first_run = _emit(_console_handler(), _job_info(1, "shared_rule", banner))
    second_run = _emit(_console_handler(), _job_info(1, "shared_rule", banner))
    assert first_run.rstrip().endswith(banner), first_run
    assert second_run.rstrip().endswith(banner), second_run


def test_console_leaves_a_summaryless_rule_alone_on_every_line():
    """An unregistered rule is the fail-open case: print exactly what came in."""
    handler = _console_handler()
    su._RULE_SUMMARIES.pop("unregistered_rule", None)
    out = _emit(
        handler,
        _job_info(1, "unregistered_rule", "Rule 9.04: unregistered_rule - do it"),
        _job_info(2, "unregistered_rule", "Rule 9.04: unregistered_rule - do it"),
    )
    assert all(
        line.endswith("Rule 9.04: unregistered_rule - do it")
        for line in out.splitlines()
    ), out


def test_install_console_style_replaces_only_the_stream_handler(monkeypatch):
    """The log-file handler writes the verbose durable record -- leave it alone."""
    import types

    import snakemake.logging as snakemake_logging

    stream = _logging.StreamHandler(io.StringIO())
    stream.name = "DefaultStreamHandler"
    logfile = _logging.StreamHandler(io.StringIO())
    logfile.name = "DefaultLogFileHandler"
    listener = types.SimpleNamespace(handlers=(stream, logfile))
    monkeypatch.setattr(
        snakemake_logging.logger_manager, "queue_listener", listener, raising=False
    )

    assert su.install_console_style() is True
    assert isinstance(listener.handlers[0], su._ConsoleHandler)
    assert listener.handlers[1] is logfile

    # Idempotent: a second call (a second Snakefile in one process) is a no-op.
    already = listener.handlers[0]
    assert su.install_console_style() is True
    assert listener.handlers[0] is already


def test_install_console_style_fails_open(monkeypatch):
    """No console styling is worth ending a run over, so every miss is silent."""
    import types

    import snakemake.logging as snakemake_logging

    for listener in (None, types.SimpleNamespace(handlers=()), object()):
        monkeypatch.setattr(
            snakemake_logging.logger_manager, "queue_listener", listener, raising=False
        )
        assert su.install_console_style() is False


def test_run_header_shape_matches_run_summary():
    """Same head-then-indented-rows block, so a run opens and closes alike.

    With no declared folders there is one group, so the block is the `run`
    label and its rows -- the legend appears only when there is one to give.
    """
    out = su.run_header(
        "wf3 run_stress_test",
        "test_case/test_rapid2",
        "test_case/snake_config_rapid.yml",
        experiment="experiment_rapid",
    )
    assert out.splitlines() == [
        "wf3 run_stress_test",
        "",
        "  run",
        "    project     test_case/test_rapid2",
        "    config      test_case/snake_config_rapid.yml",
        "    experiment  experiment_rapid",
    ]


def test_run_header_states_the_declared_folders(declare_folders):
    """Every token that appears in a body line is defined here, in full.

    A folder under the project is shown project-relative because that is how
    paths below it print; the external data root is shown absolute for the same
    reason. The rows keep DECLARATION order -- matching wants longest-path
    first, and that order in a header is arbitrary to a reader.
    """
    project = _abs("TESTS/gabon")
    declare_folders(
        data=_abs("data/wflow_global/hydromt"),
        model=os.path.join(project, "models", "hydrology", "wflow"),
    )
    lines = su.run_header("wf1 build_model", project).splitlines()
    rows = [line for line in lines if line.startswith("    ")]
    assert [row.split()[0] for row in rows] == ["project", "<data>", "<model>"]
    assert rows[1].endswith("data/wflow_global/hydromt")
    assert rows[2].endswith("models/hydrology/wflow")
    # The two kinds of row are separated and each group says what it is: the
    # `<name>` rows are a legend for the body, not more facts about the run.
    assert "  run" in lines
    assert any(line.startswith("  path tokens") for line in lines), lines
    assert "" in lines  # blank-line separation, not a wall


def test_run_header_aligns_both_groups_on_one_value_column():
    """One width across the block; a per-group width restarts the column."""
    out = su.run_header(
        "wf3 run_stress_test", "test_case/test_rapid", experiment="experiment_rapid"
    )
    rows = [line for line in out.splitlines() if line.startswith("    ")]
    # Where the VALUE starts: past the indent, the key, and the gutter.
    columns = {re.match(r" {4}\S+ +", row).end() for row in rows}
    assert len(columns) == 1, rows


def test_run_header_forward_slashes_and_shortens_the_config_path(monkeypatch):
    """The one row that kept OS separators made the block read as two trees."""
    monkeypatch.setattr(su, "_REPO_ROOT", os.path.normpath(_abs("repo")))
    config = os.path.join(_abs("repo"), "test_case", "snake_config_rapid.yml")
    out = su.run_header("wf1 build_model", "test_case/test_rapid", config)
    row = next(line for line in out.splitlines() if line.strip().startswith("config"))
    assert row.split() == ["config", "<repo>/test_case/snake_config_rapid.yml"]
    assert "\\" not in out


def test_a_rule_log_header_defines_every_token_its_rows_use(declare_folders, tmp_path):
    """The console scrolls away; the log is read months later. A `<model>/x.nc`
    row in a file that never says what `<model>` was is worse than the long
    path it replaced -- so the definition travels with the rows."""
    project = tmp_path / "gabon"
    declare_folders(model=project / "models" / "hydrology" / "wflow")
    header = su._log_header_lines(str(project / "logs" / "1.07_build.log"))
    assert "# <model>: models/hydrology/wflow" in header.splitlines()


def test_run_summary_closes_the_run_in_the_shape_it_opened_in():
    """Head, blank, labelled group, rows indented one step further."""
    out = su.run_summary(
        "wf3 run_stress_test",
        "test_case/test_rapid",
        "wf3_run_stress_test.log",
        "wf3_benchmarks.md",
        elapsed_seconds=176,
    )
    assert out.splitlines() == [
        "wf3 run_stress_test done in 0:02:56",
        "",
        "  wrote",
        "    log         test_case/test_rapid/logs/wf3_run_stress_test.log",
        "    benchmarks  test_case/test_rapid/benchmarks/wf3_benchmarks.md",
    ]


def test_run_summary_paints_only_the_failed_verdict(monkeypatch):
    """Red marks the exceptional run; a success verdict stays unpainted.

    Written against the CONSTANT, like the three-tier test: the hue is a knob,
    what must hold is that one verdict is marked and the other is not.
    """
    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    failed = su.run_summary("wf3", "p", "l.log", "b.md", failed=True)
    ok = su.run_summary("wf3", "p", "l.log", "b.md")
    assert failed.splitlines()[0] == su._ansi("wf3 FAILED", su._ANSI_FAIL)
    assert "\033" not in ok


def test_run_summary_verdict_is_plain_when_stderr_is_not_a_console(monkeypatch):
    """Piped or redirected, the block must carry no escape codes at all."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    out = su.run_summary("wf3", "p", "l.log", "b.md", failed=True)
    assert "\033" not in out, repr(out)


def test_heartbeat_paints_the_alarm_and_not_the_all_clear():
    """A stall notice says the console does not KNOW that anything is wrong.

    Yellow is what that means everywhere else. The `done in` that closes it is
    the resolution, and painting it too would make it as loud as the alarm.
    """
    stream = _TTYStringIO()
    hb = su._Heartbeat("2.05_merge", stream, interval=0.05).start()
    time.sleep(0.16)
    hb.stop()
    out = stream.getvalue()
    stall = next(line for line in out.splitlines() if "still running" in line)
    done = next(line for line in out.splitlines() if "done in" in line)
    assert stall.startswith(f"\033[{su._ANSI_WARN}m"), stall
    assert done.startswith(f"\033[{su._ANSI_BODY}m"), done


def test_heartbeat_paints_the_failure_verdict(monkeypatch):
    """There is no DONE line for a job that raised -- this is the only report."""
    stream = _TTYStringIO()
    hb = su._Heartbeat("3.15_run_wflow", stream, interval=10.0).start()
    hb.stop(failed=True)
    assert stream.getvalue().startswith(f"\033[{su._ANSI_WARN}m")


def test_run_summary_failure_keeps_its_note_out_of_the_key_column():
    """The note names no artifact, so it is not a row under `wrote`."""
    out = su.run_summary(
        "wf3 run_stress_test",
        "test_case/test_rapid",
        "wf3.log",
        "wf3.md",
        failed=True,
        log_parts_dir="test_case/test_rapid/logs/_parts/experiment_rapid",
    )
    assert out.splitlines() == [
        "wf3 run_stress_test FAILED",
        "",
        "  wrote",
        "    log parts  test_case/test_rapid/logs/_parts/experiment_rapid/",
        "",
        "  the failing job's own log is printed above",
    ]


def test_console_shortens_paths_in_snakemakes_own_lines(declare_folders):
    """`Complete log(s): <100 chars>` sat above a block of short relative paths.

    The tee already shortens a rule's own output this way; a run must not end
    with the one line that spells a path differently from every other.
    """
    declare_folders(model=_abs("TESTS/gabon/models/hydrology/wflow"))
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            f"Writing to {os.path.join(_abs('TESTS/gabon/models/hydrology/wflow'), 'staticmaps.nc')}"
        ),
    )
    assert "<model>/staticmaps.nc" in out, out


def test_run_header_omits_rows_a_workflow_does_not_have():
    """WF1 and WF2 pass no experiment; the block shrinks rather than showing a blank."""
    out = su.run_header("wf1 build_model", "test_case/test_rapid")
    assert out.splitlines() == [
        "wf1 build_model",
        "",
        "  run",
        "    project  test_case/test_rapid",
    ]


# --- console body tier for rule output (_Tee / _paint_body) ------------------


class _TTYWriter(io.StringIO):
    def isatty(self):
        return True


def test_paint_body_leaves_a_progress_bar_alone():
    """A carriage-return chunk is an in-place redraw: an SGR pair per frame
    would flicker, and a reset landing mid-bar would break it."""
    bar = "\r[####################] | 100% Completed | 102.06 ms"
    assert su._paint_body(bar, True) == bar


def test_paint_body_leaves_whitespace_alone():
    """`print` arrives as two writes; colouring a bare newline wraps nothing."""
    assert su._paint_body("\n", True) == "\n"


def test_paint_body_wraps_each_line_of_a_multi_line_chunk():
    """A chunk can hold several lines; no colour may span a newline."""
    out = su._paint_body("first\nsecond\n", True)
    body = su._ANSI_BODY
    assert out == f"\033[{body}mfirst\033[0m\n\033[{body}msecond\033[0m\n", repr(out)


def test_paint_body_is_a_no_op_without_colour():
    assert su._paint_body("plain text", False) == "plain text"


def test_tee_paints_the_console_and_never_the_log_file(tmp_path):
    """The split this rests on: one call, two sinks, one of them a file read
    months later by merge_logs -- where an escape code is corruption."""
    log = tmp_path / "rule.log"
    console = _TTYWriter()
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(console, handle)
        tee.write("13:42:17 - stats - deriving digest=7dc9315280ff\n")
        tee.close()
    # The reset lands BEFORE the newline: a colour spanning the break would
    # carry across a terminal reflow.
    assert console.getvalue() == (
        f"\033[{su._ANSI_BODY}m13:42:17 - stats - deriving digest=7dc9315280ff\033[0m\n"
    )
    assert "\033" not in log.read_text(encoding="utf-8")


def test_tee_does_not_colour_a_console_that_is_not_a_terminal(tmp_path):
    log = tmp_path / "rule.log"
    console = io.StringIO()  # isatty() -> False
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(console, handle)
        tee.write("plain row\n")
        tee.close()
    assert console.getvalue() == "plain row\n"
