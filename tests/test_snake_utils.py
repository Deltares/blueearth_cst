"""Exact-equivalence tests for the shared get_config helper (R3 §3, §8).

Pins the semantics the four inline get_config copies (three Snakefiles +
conftest) had before they were collapsed into blueearth_cst/shared/snake_utils.py, so the move
is provably identity-preserving rather than merely green on a smoke test.
"""

import gc
import io
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import pytest

import blueearth_cst.shared.console_style as cs  # noqa: E402
import blueearth_cst.shared.run_log_core as log_core  # noqa: E402
import blueearth_cst.shared.snake_utils as su  # noqa: E402
from blueearth_cst.shared.console_style import (  # noqa: E402
    rule_banner,
    target_banner,
)
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    _compact_log_line,
    _cr_overwrite,
    _drop_redraw_frames,
    _Heartbeat,
    _log_path_parts,
    _relativize_paths,
    get_config,
    log_row,
    patch_psutil_windows_benchmark,
    save_figure,
    tee_to_log,
)


@pytest.fixture(autouse=True)
def _no_ambient_no_color(monkeypatch):
    """Clear ambient console state for every test in this module.

    The painting tests hand the console handler a fake TTY stream, but that is
    only half of what decides colour: ``_ConsoleHandler`` also consults
    ``NO_COLOR``, which it honours per the no-color.org convention. Leaving that
    half ambient made the result depend on which shell launched pytest --
    ``NO_COLOR=1`` in the environment turned seven painting assertions red with
    no code change, and the same commit passed from a shell without it.

    A test that wants the variable set uses ``monkeypatch.setenv`` and overrides
    this, which runs after the fixture.

    ``test_cli.py`` parses Snakefiles in this same pytest process, while real
    workflow invocations each use a fresh process. Clear the dynamic-progress
    registry so a prior WF3 dry-run cannot turn an unrelated synthetic handler
    into a checkpoint-aware one.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cs, "_DYNAMIC_PROGRESS_RULES", set())
    monkeypatch.setattr(cs, "_HEADER_PRINTED", False)


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


@pytest.mark.parametrize(
    "module,current,proposed",
    [
        (
            "weathergen",
            "Reading weather netcdf: source.nc",
            "Reading historical climate → source.nc",
        ),
        (
            "weathergen",
            "Resampling on 1 basin cell(s) of 9 in the store",
            "Weather resampling uses 1 basin-intersecting cell from the 9-cell extracted grid",
        ),
        (
            "weathergen",
            "Generating 1 weather realization(s)",
            "Generating 1 stochastic weather realization",
        ),
        (
            "weathergen",
            "Moving 2 figure(s) to plots",
            "Moving diagnostic figures → plots",
        ),
        (
            "weathergen",
            "Saving realization 1 of 2",
            "Writing stochastic realization 1 of 2",
        ),
        (
            "weathergen",
            "run_001 reading realization root.nc",
            "run_001: reading source realization root.nc",
        ),
        (
            "weathergen",
            "run_001 applying perturbations (temp=+2)",
            "run_001: applying climate perturbations (temp=+2)",
        ),
        (
            "experiment",
            "Wrote lookup.csv (12 rows, 1 member)",
            "Perturbation lookup written → lookup.csv (12 monthly rows; 1 perturbation point)",
        ),
    ],
)
def test_provider_console_wording_preserves_identity_and_level(
    module, current, proposed
):
    # Provider source is hashed into the seed. Change presentation, not that source.
    row = f"12:34:56 - {module} - WARNING - {current}\n"
    expected = f"12:34:56 - {module} - WARNING - {proposed}\n"
    assert _compact_log_line(row) == expected
    assert _compact_log_line(expected) == expected


def test_compact_shortens_timestamp_and_drops_dotted_name():
    line = (
        "2026-07-21 18:03:38,474 - hydromt.model.model - model - INFO - "
        "Initializing wflow_sbm model.\n"
    )
    # date + milliseconds dropped -> HH:MM:SS; dotted name dropped; module kept
    assert _compact_log_line(line) == (
        "18:03:38 - model - Initializing wflow_sbm model\n"
    )


def test_compact_shortens_hydromts_forcing_exists_warning():
    """Three sentences and ~290 characters, two of them an inapplicable remedy.

    hydromt tells the reader to change `input.path_forcing` in "the build
    inifile"; this pipeline sets that key programmatically in rule 1.10, so the
    remedy names a file nobody here edits. The fact and its consequence stay,
    because a duplicate forcing file written under a generated name is real and
    is what the reader has to act on.
    """
    line = (
        "2026-09-05 15:20:18,001 - hydromt_wflow.components.forcing - forcing - "
        "WARNING - Netcdf forcing file `m/forcing/inmaps_historical.nc` already "
        "exists and overwriting is not enabled. To overwrite netcdf forcing file: "
        "change name `input.path_forcing` in setup_config section of the build "
        "inifile or allow overwriting with `overwrite` flag. A default name will "
        "be generated.\n"
    )
    assert _compact_log_line(line) == (
        "15:20:18 - forcing - WARNING - m/forcing/inmaps_historical.nc exists and "
        "overwriting is off; writing under a generated name\n"
    )


def test_compact_leaves_a_reworded_forcing_warning_alone():
    """Cosmetic filters fail toward PRINTING; an upstream reword is not silenced."""
    line = (
        "2026-09-05 15:20:18,001 - hydromt_wflow.components.forcing - forcing - "
        "WARNING - Netcdf forcing file `m/f.nc` already exists; overwriting "
        "disabled. A default name will be generated.\n"
    )
    assert "overwriting disabled" in _compact_log_line(line)


def test_compact_drops_a_single_trailing_stop():
    """hydromt ends its messages with a full stop; our rows end with none.

    One convention across the log, and a stop after a path (`staticmaps.nc.`)
    is the one place it read as part of the path. Only ONE stop goes, so an
    ellipsis survives.
    """
    stamp = "2026-07-21 18:03:38,474 - hydromt.model.model - model - INFO - "
    assert _compact_log_line(stamp + "Writing grid data to a/staticmaps.nc.\n") == (
        "18:03:38 - model - Writing grid data to a/staticmaps.nc\n"
    )
    assert (
        _compact_log_line(stamp + "Writing...\n") == "18:03:38 - model - Writing...\n"
    )


def test_compact_drops_the_data_type_from_a_catalog_read():
    """``Reading <name> <Type> data from <uri>`` -> ``Reading <name> from <uri>``.

    hydromt's commonest row (``data_source.py:85``). The type is already implied
    by the ``data_source`` column and by the catalog entry itself, so it sits
    between the two facts a reader wants: WHICH source, from WHERE.
    """
    line = (
        "2026-08-17 08:12:03,441 - hydromt.data_catalog.sources.data_source - "
        "data_source - INFO - Reading merit_hydro_ihu RasterDataset data from "
        "C:/data/topography/merit_hydro_ihu/30sec/*.tif\n"
    )
    assert _compact_log_line(line) == (
        "08:12:03 - data_source - Reading merit_hydro_ihu from "
        "C:/data/topography/merit_hydro_ihu/30sec/*.tif\n"
    )


@pytest.mark.parametrize(
    "data_type",
    ["RasterDataset", "GeoDataFrame", "GeoDataset", "DataFrame", "Dataset"],
)
def test_compact_drops_every_data_type_hydromt_can_log(data_type):
    """The five concrete ``data_type`` values in ``hydromt.data_catalog.sources``."""
    line = (
        "2026-08-17 08:12:03,441 - x - data_source - INFO - "
        f"Reading src {data_type} data from a.nc\n"
    )
    assert _compact_log_line(line) == "08:12:03 - data_source - Reading src from a.nc\n"


def test_compact_leaves_an_unknown_data_type_visible():
    """A cosmetic filter on someone else's message must not eat what it cannot name.

    An upstream addition should surface as a slightly noisy row, never as a
    quietly mangled one — which is why the five names are enumerated.
    """
    line = (
        "2026-08-17 08:12:03,441 - x - data_source - INFO - "
        "Reading src QuantumBlob data from a.nc\n"
    )
    assert _compact_log_line(line) == (
        "08:12:03 - data_source - Reading src QuantumBlob data from a.nc\n"
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
        "11:13:01 - geoms - Writing to staticgeoms/basins.geojson\n"
    )


def test_compact_keeps_a_component_prefix_that_says_something_else():
    """`spatial` is hydromt's module, `staticmaps` the component -- two facts."""
    line = (
        "2026-08-14 11:13:01,204 - hydromt_wflow.wflow_base - spatial - INFO - "
        "wflow_sbm.staticmaps: Writing region to staticgeoms/region.geojson.\n"
    )
    assert _compact_log_line(line) == (
        "11:13:01 - spatial - wflow_sbm.staticmaps: Writing region to "
        "staticgeoms/region.geojson\n"
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


def _figure_console_path(path):
    """Use the row formatter's path contract for any pytest temp location."""
    return _relativize_paths(
        str(path),
        os.environ.get(log_core._PROJECT_ROOT_ENV, ""),
        log_core._path_tokens(),
    )


def test_save_figure_writes_creates_parent_and_announces(tmp_path, capsys):
    su._FIGURE_BUNDLES.clear()
    out = _figure(tmp_path, "elevation_basin.png")  # parent does not exist yet
    assert out.exists()
    su.flush_figure_bundles()
    printed = capsys.readouterr().out.strip()
    assert printed.endswith(f"- plot - Figure written → {_figure_console_path(out)}"), (
        printed
    )


def test_save_figure_bundles_one_directory_into_one_row(tmp_path, capsys):
    """A plotting rule writes 9-33 figures into ONE directory; that is one row."""
    su._FIGURE_BUNDLES.clear()
    first = _figure(tmp_path, "hydrograph_1030.png")
    _figure(tmp_path, "performance_1030.png")
    _figure(tmp_path, "signatures_peaks_1030.png")
    assert capsys.readouterr().out == ""  # nothing announced until the flush
    su.flush_figure_bundles()
    rows = [line.split(" - ", 2)[2] for line in capsys.readouterr().out.splitlines()]
    assert rows == [f"Wrote 3 figures → {_figure_console_path(first.parent)}"], rows


def test_save_figure_bundles_alternating_directories_separately(tmp_path, capsys):
    """The rules that motivated this alternate between a dir and its child."""
    su._FIGURE_BUNDLES.clear()
    _figure(tmp_path, "a.png", subdir="plots")
    _figure(tmp_path, "b.png", subdir="maps")
    _figure(tmp_path, "c.png", subdir="plots")
    _figure(tmp_path, "d.png", subdir="maps")
    su.flush_figure_bundles()
    rows = [line.split(" - ", 2)[2] for line in capsys.readouterr().out.splitlines()]
    # Two rows, in the order the directories were FIRST written to -- not four,
    # and not one per alternation.
    assert rows == [
        f"Wrote 2 figures → {_figure_console_path(tmp_path / 'plots')}",
        f"Wrote 2 figures → {_figure_console_path(tmp_path / 'maps')}",
    ], rows


def test_an_ordinary_row_flushes_the_bundle_first(tmp_path, capsys):
    """A module's own summary must read AFTER the figures it summarizes."""
    su._FIGURE_BUNDLES.clear()
    _figure(tmp_path, "a.png")
    _figure(tmp_path, "b.png")
    log_row("Wrote 2 canonical climate figures", module="plot")
    rows = [line.split(" - ", 2)[2] for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        f"Wrote 2 figures → {_figure_console_path(tmp_path / 'plots')}",
        "Wrote 2 canonical climate figures",
    ], rows


def test_a_closing_log_drains_pending_figures(tmp_path, capsys):
    """A rule whose LAST act is a figure must still report it."""
    su._FIGURE_BUNDLES.clear()
    with tee_to_log(tmp_path / "logs" / "rule.log"):
        _figure(tmp_path, "second.png")
    body = (tmp_path / "logs" / "rule.log").read_text(encoding="utf-8")
    assert "second.png" in body


def test_a_new_log_starts_a_new_bundle(tmp_path, capsys):
    """The grouping is per FILE: an earlier block's figures are not counted in."""
    su._FIGURE_BUNDLES.clear()
    _figure(tmp_path, "first.png")
    capsys.readouterr()
    with tee_to_log(tmp_path / "logs" / "rule.log"):
        _figure(tmp_path, "second.png")
    body = (tmp_path / "logs" / "rule.log").read_text(encoding="utf-8")
    assert "second.png" in body
    assert "first.png" not in body
    assert "2 figures" not in body


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
    # (only uss), which NAs every indicator. The shim must expose pss (= uss proxy).
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


def test_compact_target_banner_summarizes_requested_outputs(tmp_path):
    present = tmp_path / "present.csv"
    missing = tmp_path / "missing.csv"
    present.touch()
    targets = {"present": present, "missing": missing}
    assert target_banner("all", targets.values(), compact=True) == (
        "Requested outputs: 2/2 provided"
    )
    assert target_banner("all", [], compact=True) == ("Requested outputs: 0/0 provided")
    assert target_banner("all", None, compact=True) == "Confirm requested outputs"


def test_target_banner_puts_one_target_per_line(monkeypatch):
    """The whole point: Snakemake's own `input:` joins with ", "; this does not."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())  # isatty() -> False
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("all", ["a/x.csv", "b/y.png"])
    assert out == "Target: all - confirm requested outputs\n    a/x.csv\n    b/y.png"
    assert ", " not in out


def test_target_banner_accepts_a_dict_values_view(monkeypatch):
    """WF2 and WF3 pass `TARGETS.values()`, not a list."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    targets = {"a": "one.csv", "b": "two.csv"}
    assert target_banner("all", targets.values()) == (
        "Target: all - confirm requested outputs\n    one.csv\n    two.csv"
    )


def test_target_banner_with_no_targets_is_just_the_banner(monkeypatch):
    """No trailing blank line -- an empty list must not print an empty row."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert target_banner("all", []) == "Target: all - confirm requested outputs"


def test_target_banner_relativizes_against_project_dir(monkeypatch):
    """The root moves to the banner; the paths below it lose the prefix.

    Through `_abs`, not a `C:/...` literal: since t2609162114 the bracket prints
    `display_root(project_dir)`, which absolutizes -- and a `C:/...` literal is
    NOT absolute on POSIX, so abspath prepended the runner's CWD and the bracket
    read `<repo>/C:/TESTS/CST/gabonx`. Caught by CI on ubuntu, green on windows:
    the same class `_abs` was introduced for on 2026-08-14.
    """
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    root = _abs("TESTS/CST/gabonx")
    shown = root.replace(os.sep, "/")
    out = target_banner(
        "all",
        [f"{shown}/analyze_projections/cmip6/summary/x.csv"],
        root,
    )
    assert out == (
        f"Target: all - confirm requested outputs  [{shown}]\n    analyze_projections/cmip6/summary/x.csv"
    )


def test_target_banner_relativizes_a_native_separator_root(monkeypatch):
    """Snakefiles build targets with `/`; project_dir may arrive either way."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("all", ["proj/logs/wf1.log"], os.path.join("proj"))
    assert out.endswith("    logs/wf1.log")


def test_target_banner_leaves_a_path_outside_the_project_absolute(monkeypatch):
    """Only the project prefix is stripped -- a catalog elsewhere stays whole.

    Converted to `_abs` alongside the test above even though this one stayed
    GREEN on ubuntu, for the reason `_abs`'s own docstring gives: it asserts
    something is left UNCHANGED, which is exactly what a root that never matches
    produces. With a `C:/...` root on POSIX it asserted the absence of an effect
    that could not have occurred.
    """
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    outside = _abs("elsewhere/data/catalog.yml").replace(os.sep, "/")
    out = target_banner("all", [outside], _abs("TESTS/CST/gabonx"))
    assert f"    {outside}" in out


def test_target_banner_without_project_dir_keeps_paths_verbatim(monkeypatch):
    """The default is unchanged: no root given, nothing stripped, no bracket."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = target_banner("all", ["C:/p/q.csv"])
    assert out == "Target: all - confirm requested outputs\n    C:/p/q.csv"
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
    managed = (
        tmp_path
        / "gabon/logs/20261019T120000Z-abcdef012345/_parts"
        / "wf3-child/logs/3.10_run_wflow/rlz_1_st_1.log"
    )
    root3, log_id3 = _log_path_parts(managed)
    assert root3 == root
    assert log_id3 == log_id2


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


@pytest.mark.parametrize("backend", ["tee", "child"])
@pytest.mark.parametrize("separator", ["/", "\\"])
def test_console_aliases_preserve_saved_commands(
    tmp_path, monkeypatch, capsys, declare_folders, backend, separator
):
    project = str(tmp_path)
    model = tmp_path / "models/hydrology/wflow"
    catalog = (
        tmp_path
        / "config/_engine/execution-configs/_shared/project_catalog_0/459d6135261e/project_catalog_0.yml"
    )
    declare_folders(
        model=model,
        console_only={"model-config": model / "config", "project-catalog": catalog},
    )
    monkeypatch.setenv(su._PROJECT_ROOT_ENV, project)
    recipe = str(model / "config/build_historical_forcing.yml").replace("\\", "/")
    source = str(catalog).replace("\\", "/")
    command = f"$ hydromt update wflow_sbm {model} -i {recipe} -d {source} -vv"
    command = command.replace("\\", "/").replace("/", separator)
    if backend == "tee":
        live, saved = io.StringIO(), io.StringIO()
        tee = su._Tee(live, saved, project_root=project)
        tee.write(command + "\n")
        tee.close()
        shown, logged = live.getvalue(), saved.getvalue()
    else:
        log = tmp_path / "rule.log"
        assert su.run_and_tee([sys.executable, "-c", f"print({command!r})"], log) == 0
        shown = capsys.readouterr().out
        logged = log.read_text(encoding="utf-8")
    assert shown == (
        "$ hydromt update wflow_sbm <model> "
        "-i <model-config>/build_historical_forcing.yml -d <project-catalog> -vv\n"
    )
    assert "<model>/config/build_historical_forcing.yml" in logged
    assert (
        "config/_engine/execution-configs/_shared/project_catalog_0/459d6135261e/project_catalog_0.yml"
        in logged
    )
    assert "<model-config>" not in logged
    assert "<project-catalog>" not in logged
    rows = dict(cs.run_meta_rows(project))
    assert rows["<model-config>"] == "<project>/models/hydrology/wflow/config"
    assert rows["<project-catalog>"].endswith("459d6135261e/project_catalog_0.yml")
    # A subsequent workflow must not inherit these run-specific aliases.
    declare_folders(climate=tmp_path / "climate")
    assert "<model-config>" not in dict(cs.run_meta_rows(project))


@pytest.mark.parametrize("backend", ["tee", "child"])
def test_scenario_console_paths_use_exported_tree(
    tmp_path, monkeypatch, capsys, declare_folders, backend
):
    declare_folders(scenarios=tmp_path / "scenarios")
    monkeypatch.setenv(su._PROJECT_ROOT_ENV, str(tmp_path))
    path = tmp_path / "scenarios/_engine/data/da6324c8b66b/weathergenr/output"
    message = f"12:39:49 - eval - Completed in 18.5 seconds | Saved to: {path}\n"
    if backend == "tee":
        live, saved = io.StringIO(), io.StringIO()
        tee = su._Tee(live, saved, project_root=str(tmp_path))
        tee.write(message)
        tee.close()
        shown, logged = live.getvalue(), saved.getvalue()
    else:
        log = tmp_path / "rule.log"
        assert (
            su.run_and_tee([sys.executable, "-c", f"print({message!r}, end='')"], log)
            == 0
        )
        shown = capsys.readouterr().out
        logged = log.read_text(encoding="utf-8")
    assert "Saved to: <scenarios>/weathergenr/output" in shown
    assert "<scenarios>/_engine/data/da6324c8b66b/weathergenr/output" in logged


def test_template_notice_is_once_across_generation_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CST_WARNING_TALLY", str(tmp_path / "warnings"))
    monkeypatch.setenv("CST_GENERATION_INVOCATION_ID", "a" * 32)
    message = "12:39:49 - io - Template loaded: nx=3, ny=2, nt=15,330"
    warning = message.replace(" - Template", " - WARNING - Template")
    command = [
        sys.executable,
        "-c",
        f"print({message!r}); print({message!r}); print({warning!r})",
    ]
    children = []
    try:
        for job in ("roots", "perturbations"):
            log = tmp_path / f"{job}.log"
            code = (
                "from blueearth_cst.shared.run_log_core import run_and_tee; "
                f"raise SystemExit(run_and_tee({command!r}, {str(log)!r}))"
            )
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", code],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
            )
        output = ""
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            output += stdout
        for job in ("roots", "perturbations"):
            assert (tmp_path / f"{job}.log").read_text(encoding="utf-8").count(
                message
            ) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    assert output.count(message) == 1
    assert output.count(warning) == 2
    monkeypatch.setenv("CST_GENERATION_INVOCATION_ID", "b" * 32)
    assert su.run_and_tee(command, tmp_path / "retry.log") == 0
    assert capsys.readouterr().out.count(message) == 1


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


def test_declare_project_root_lets_a_row_shorten_without_a_rule_log(monkeypatch):
    """Rule 0.01 declares no `log:`, so nothing tees its output. Its rows were
    the only ones in a run still printing the project dir in full."""
    monkeypatch.setenv(su._PATH_TOKENS_ENV, "{}")
    monkeypatch.delenv(su._PROJECT_ROOT_ENV, raising=False)
    su.declare_project_root("test_case/test_rapid")
    assert os.environ[su._PROJECT_ROOT_ENV] == "test_case/test_rapid"
    native = os.path.join("test_case", "test_rapid", "config", "cfg.yml")
    assert (
        su._relativize_paths(f"snapshot -> {native}", os.environ[su._PROJECT_ROOT_ENV])
        == "snapshot -> config/cfg.yml"
    )


def test_declare_project_root_unsets_on_a_blank(monkeypatch):
    """A caller with no project dir must not leave a stale root behind for the
    next process to shorten against."""
    monkeypatch.setenv(su._PROJECT_ROOT_ENV, "old/root")
    assert su.declare_project_root("") == ""
    assert su._PROJECT_ROOT_ENV not in os.environ


def test_log_row_shortens_the_project_dir_it_was_given(capsys, monkeypatch):
    monkeypatch.setenv(su._PATH_TOKENS_ENV, "{}")
    monkeypatch.setenv(su._PROJECT_ROOT_ENV, "test_case/test_rapid")
    native = os.path.join("test_case", "test_rapid", "config", "record.yml")
    su.log_row(f"Run record -> {native}", module="config")
    assert "Run record -> config/record.yml" in capsys.readouterr().out


def test_log_row_leaves_a_row_alone_when_no_project_root_is_declared(
    capsys, monkeypatch
):
    """A bare script run by hand keeps the behaviour this improves on."""
    monkeypatch.setenv(su._PATH_TOKENS_ENV, "{}")
    monkeypatch.delenv(su._PROJECT_ROOT_ENV, raising=False)
    su.log_row("Run record -> test_case/test_rapid/config/record.yml")
    assert "test_case/test_rapid/config/record.yml" in capsys.readouterr().out


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
    rows = cs.run_header("wf1 build_model", "test_case/test_rapid").splitlines()
    assert rows[-1].split() == ["<model>", "<project>/models/hydrology/wflow"]


def test_run_header_project_row_is_absolute_and_repo_marked():
    """One spelling of the project root, whatever form the Snakefile holds.

    WF0-WF3 hold the config's relative `project_dir`; WF4 resolves it. Printed
    as held, one `run_workflows.py` run stated one fact two ways a few lines
    apart (t2609162114). Both now print the absolute form, `<repo>`-marked when
    the tree is inside the checkout -- shorter than the relative spelling AND
    unambiguous about which of six worktrees it names.
    """
    relative = cs.run_header("wf3 generate_scenarios", "test_case/test_rapid")
    absolute = cs.run_header(
        "wf4 simulate_system", os.path.abspath("test_case/test_rapid")
    )

    def project_row(text):
        return next(
            row.split() for row in text.splitlines() if row.split()[:1] == ["<project>"]
        )

    assert project_row(relative) == project_row(absolute)
    assert project_row(relative) == ["<project>", "<repo>/test_case/test_rapid"]


def test_display_root_leaves_a_project_outside_the_repo_whole():
    """No shared root to imply, so nothing is marked and nothing is shortened.

    The production case: `project_dir` lives outside the repository tree, where
    the `<repo>` rewrite matches nothing and the full path IS the information.
    """
    outside = _abs("TESTS/CST/gabonx")
    assert cs.display_root(outside) == outside.replace(os.sep, "/")
    assert "<repo>" not in cs.display_root(outside)


def test_target_banner_bracket_and_header_row_agree(monkeypatch):
    """The bracket and the `project` row are the same fact; they must match."""
    import io

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.delenv("NO_COLOR", raising=False)
    banner = target_banner(
        "all", ["test_case/test_rapid/logs/wf3.log"], "test_case/test_rapid"
    )
    header = cs.run_header("wf3 generate_scenarios", "test_case/test_rapid")
    root = next(
        row.split()[1]
        for row in header.splitlines()
        if row.split()[:1] == ["<project>"]
    )
    assert f"[{root}]" in banner
    # The STRIP still uses the caller's form, so the target stays short.
    assert banner.endswith("    logs/wf3.log")


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
    assert head[1].startswith("# <project>:") and head[1].rstrip().endswith("gabon")
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

    `check_model_unchanged` raises `ModelDriftError` naming the changed input,
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


# --- covering the frame a carriage return did not erase -----------------------


def test_pad_line_over_covers_a_wider_frame():
    """A short row over a long frame must clear the frame's tail, not sit in it."""
    padded, dirty = su._pad_line_over("done\n", 12)
    assert padded == "done        \n"  # padding BEFORE the newline
    assert dirty == 0  # the newline moved past the frame


def test_pad_line_over_reports_what_is_left_standing():
    """A write with no newline leaves its own width on the line for the next one."""
    padded, dirty = su._pad_line_over("bar 50%", 20)
    assert dirty == 20  # the padding is what now stands there
    assert len(padded) == 20


def test_pad_line_over_leaves_a_line_wider_than_the_frame_alone():
    """Padding is a floor, never a width: a long row is not truncated or grown."""
    padded, dirty = su._pad_line_over("a considerably longer row\n", 5)
    assert padded == "a considerably longer row\n"
    assert dirty == 0


def test_pad_line_over_pads_the_first_line_of_a_multi_line_chunk():
    """The frame is on the line the CURSOR is on, not the one the text ends on."""
    padded, _ = su._pad_line_over("first\nsecond\n", 10)
    assert padded == "first     \nsecond\n"


# --- console-side redraw suppression -----------------------------------------


def test_drop_redraw_frames_swallows_a_bar_and_its_closing_newline():
    """A dask bar reaches the console as N frames plus a bare newline: show none."""
    shown, state = "", False
    for pct in (0, 50, 100):
        text, state = _drop_redraw_frames(f"\r[{'#' * pct}] | {pct}% Completed", state)
        shown += text
    assert state is True
    text, state = _drop_redraw_frames("\n", state)  # dask's `_finish`
    shown += text
    assert shown == ""
    assert state is False


def test_drop_redraw_frames_keeps_content_that_follows_a_redraw():
    shown, state = _drop_redraw_frames(
        "\r[###] | 100% Completed\n08:12:03 - a - b\n", False
    )
    assert shown == "08:12:03 - a - b\n"
    assert state is False


def test_drop_redraw_frames_passes_ordinary_text_through():
    shown, state = _drop_redraw_frames("08:12:03 - a - b\n", False)
    assert (shown, state) == ("08:12:03 - a - b\n", False)


def test_drop_redraw_frames_keeps_a_blank_line_that_is_not_closing_a_bar():
    """The bare-newline rule must fire only while a redraw is open."""
    assert _drop_redraw_frames("\n", False) == ("\n", False)


def test_drop_redraw_frames_swallows_a_final_frame_that_carries_no_cr():
    """dask ends a bar with the completed frame and a newline, no `\\r`.

    Looking for `\\r` alone let exactly one `[####...] | 100% Completed` row
    per bar through, which is what a WF1 run still showed after the per-frame
    writes were suppressed.
    """
    _text, state = _drop_redraw_frames("\r[##  ] | 50% Completed", False)
    assert state is True
    shown, state = _drop_redraw_frames("[####] | 100% Completed |  1.4 s\n", state)
    assert (shown, state) == ("", False)


def test_drop_redraw_frames_keeps_content_after_a_cr_less_final_frame():
    """Only the bar's own tail is swallowed; the next row must survive."""
    _text, state = _drop_redraw_frames("\r[##  ] | 50%", False)
    shown, state = _drop_redraw_frames("[####] | 100%\n08:12:03 - a - b\n", state)
    assert (shown, state) == ("08:12:03 - a - b\n", False)


def test_tee_keeps_a_bar_out_of_the_console_but_in_the_log(tmp_path, capsys):
    # The sentinel is the row hydromt prints right after a dask bar. It was
    # `forcing - Write forcing file` until that row joined `_TEE_CONSOLE_MUTED`,
    # at which point this test failed for the wrong cause: a muted sentinel
    # proves nothing about bars. Keep this one UNMUTED -- delivering it is what
    # the assertion below actually tests.
    log = tmp_path / "rule.log"
    row = "08:12:03 - forcing - Writing file <model>/forcing/inmaps.nc\n"
    with tee_to_log(log):
        for pct in (0, 42, 100):
            sys.stdout.write(f"\r[{'#' * (pct // 10):<10}] | {pct}% Completed | 7.08 s")
        sys.stdout.write("\n")
        sys.stdout.write(row)
    console = capsys.readouterr().out
    assert "% Completed" not in console
    assert row.rstrip() in console
    persisted = log.read_text(encoding="utf-8")
    assert "100% Completed" in persisted  # the durable record keeps the final state


def test_tee_keeps_our_own_bar_visible_through_write_redraw(tmp_path, capsys):
    """`shared.progress` is the sanctioned exception; a library bar is not."""
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write_redraw("\r[####] ours 50%")
        sys.stdout.write_redraw("\r[########] ours 100%")
        sys.stdout.write_redraw("\n")
        sys.stdout.write("\r[####] theirs 50%")
        sys.stdout.write("\n")
    console = capsys.readouterr().out
    assert "ours 100%" in console
    assert "theirs" not in console
    assert "theirs 50%" in log.read_text(encoding="utf-8")


class _LiveConsole:
    """A console sink that can declare itself a terminal, or not."""

    def __init__(self, tty):
        self._buffer = io.StringIO()
        self._tty = tty
        self.encoding = "utf-8"

    def write(self, text):
        return self._buffer.write(text)

    def flush(self):
        pass

    def isatty(self):
        return self._tty

    def getvalue(self):
        return self._buffer.getvalue()


@pytest.mark.parametrize("closed", [False, True])
def test_tee_flush_publishes_live_output_without_draining_partial_log(closed):
    class BufferedConsole(io.StringIO):
        visible = ""

        def flush(self):
            self.visible = self.getvalue()

    live = BufferedConsole()
    logfile = io.StringIO()
    tee = su._Tee(live, logfile)
    if closed:
        tee.close()
        logfile.close()
    tee.write("partial")
    assert live.visible == ""
    tee.flush()
    assert live.visible == "partial"
    if not closed:
        assert logfile.getvalue() == ""


def test_empty_tee_replays_do_not_postpone_heartbeat(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(log_core.time, "monotonic", lambda: clock[0])
    monkeypatch.delenv("CST_HEARTBEAT_SECS", raising=False)
    console = io.StringIO()
    heartbeat = _Heartbeat("plot_climate_diagnostics", console, interval=15)
    tee = su._Tee(console, io.StringIO(), on_activity=heartbeat.touch)
    for seconds in (15.0, 30.0, 45.0):
        clock[0] = seconds
        # Successful captured_output() replays an empty string between figures.
        assert tee.write("") == 0
        heartbeat._tick(seconds)
    assert console.getvalue().count("elapsed") == 3
    assert "0:00:45 elapsed" in console.getvalue()


def _drive_a_bar(live, logfile):
    """Exactly what ``shared.progress`` emits: frames, then a closing newline."""
    tee = su._Tee(live, logfile)
    tee.write_redraw("\rforcing  ====------  40.0%  0:03 | eta 0:04")
    tee.write_redraw("\rforcing  ==========  100.0%  0:07 elapsed")
    tee.write_redraw("\n")
    tee.close()


def test_tee_animates_our_bar_on_a_terminal(tmp_path):
    """A carriage return overwrites here, so the frames are worth streaming."""
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=True)
    with open(log, "w", encoding="utf-8") as handle:
        _drive_a_bar(live, handle)

    console = live.getvalue()
    assert console.count("\r") == 2  # every frame reached the screen
    assert "40.0%" in console


def test_tee_holds_our_bar_to_one_summary_row_off_a_terminal(tmp_path):
    """`snakemake ... *> run.txt` and the GUI capture: a `\\r` APPENDS there.

    Streaming would put one row per frame into the very artifact someone reads
    afterwards, which is the defect ``run_and_tee`` has always guarded against
    for `shell:` rules. The bar still reports — once, as an ordinary row.
    """
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=False)
    with open(log, "w", encoding="utf-8") as handle:
        _drive_a_bar(live, handle)

    console = live.getvalue()
    assert "\r" not in console
    assert console.count("\n") == 1
    assert "100.0%" in console
    assert "40.0%" not in console  # intermediate frames stay off the record


def test_tee_persists_the_same_summary_whatever_the_console_is(tmp_path):
    """The log file is the durable record and must not depend on the terminal."""
    written = []
    for tty in (True, False):
        log = tmp_path / f"rule-{tty}.log"
        with open(log, "w", encoding="utf-8") as handle:
            _drive_a_bar(_LiveConsole(tty=tty), handle)
        written.append(log.read_text(encoding="utf-8"))

    assert written[0] == written[1]
    assert written[0].count("100.0%") == 1
    assert "40.0%" not in written[0]


# --- line reset over a standing progress frame -------------------------------
#
# A frame is left standing between redraws (`\r<frame>`, cursor at its end), so
# the next writer to start a logical line must clear it or its row is APPENDED
# to the bar. The writers are in different processes under `-c 3`, which is why
# the fix is cursor state rather than a shared flag.

_LINE_RESET = "\r\033[2K"


def _unreset(text):
    """Drop the line reset these writers prefix to every row they start.

    Used by the COLOUR tests, whose subject is the paint tier: the reset is
    unconditional on a terminal and would otherwise have to be spelled into
    every one of them, which is how an assertion about one thing quietly
    starts gating another. It also carries a carriage return, and
    ``str.splitlines`` splits on that -- so a test reading ``out.splitlines()``
    would see a phantom empty row before every line.
    """
    return text.replace(_LINE_RESET, "")


def test_line_reset_is_the_erase_sequence_on_a_terminal():
    assert su._line_reset(_LiveConsole(tty=True)) == _LINE_RESET


def test_line_reset_is_empty_off_a_terminal():
    """Off a terminal the escape would be literal text in a captured run."""
    assert su._line_reset(_LiveConsole(tty=False)) == ""


def test_line_reset_survives_no_color(monkeypatch):
    """`NO_COLOR` asks for no colour, not for an unmanaged cursor."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert su._line_reset(_LiveConsole(tty=True)) == _LINE_RESET


def test_tee_clears_a_standing_frame_before_an_ordinary_row(tmp_path, monkeypatch):
    """The defect: `13:18:23 - DONE ...` landing on the tail of an open bar."""
    monkeypatch.setenv("NO_COLOR", "1")  # assert on the escape under test alone
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=True)
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(live, handle)
        tee.write_redraw("\rforcing  ====------  40.0%  0:03 | eta 0:04")
        tee.write("08:12:03 - a - b\n")
        tee.close()

    console = live.getvalue()
    assert console.index("08:12:03") > console.index(_LINE_RESET)
    assert "40.0%08:12:03" not in console


def test_tee_keeps_the_reset_off_the_log_file(tmp_path):
    """The escape is console-only; no escape code may ever reach `logs/`."""
    log = tmp_path / "rule.log"
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(_LiveConsole(tty=True), handle)
        tee.write_redraw("\rforcing  ====------  40.0%  0:03 | eta 0:04")
        tee.write("08:12:03 - a - b\n")
        tee.close()

    assert "\033" not in log.read_text(encoding="utf-8")


def test_tee_never_resets_part_way_through_its_own_row(tmp_path, monkeypatch):
    """The one state the reset must not fire in.

    An ordinary partial write leaves the cursor mid-line too, but there the
    next write is the REST of that line -- a library writing one row in two
    calls -- and erasing it would destroy the half already shown. Every other
    line start is reset unconditionally, including this row's own.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=True)
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(live, handle)
        tee.write("08:12:03 - a - ")
        tee.write("b\n")
        tee.close()

    assert live.getvalue() == _LINE_RESET + "08:12:03 - a - b\n"


def test_tee_resets_even_after_the_bar_closed_its_own_line(tmp_path, monkeypatch):
    """`finish` terminated OUR bar; it says nothing about anyone else's.

    Asserted the other way round until 2026-09-17, on the reasoning that a
    closed line has nothing standing on it. True of this tee and irrelevant
    to the defect: under `-c 3` the bar on that line can belong to a sibling
    job in another process. Erasing an empty line costs nothing, so the reset
    is unconditional and this test pins that it stayed unconditional.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=True)
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(live, handle)
        tee.write_redraw("\rforcing  ==========  100.0%  0:07 elapsed")
        tee.write_redraw("\n")
        tee.write("08:12:03 - a - b\n")
        tee.close()

    assert live.getvalue().endswith(_LINE_RESET + "08:12:03 - a - b\n")


def test_a_row_clears_a_frame_drawn_by_a_sibling_job(tmp_path, monkeypatch):
    """The reported defect, in the shape it actually occurs in.

    wf0's `extract_climate_datasets` fans out per source, so `-c 3` runs
    era5 and chirps as two jobs -- two PROCESSES, two tees, one console. One
    holds a bar; the other writes rows, and until 2026-09-17 those rows landed
    on the tail of a frame their own tee had never drawn and could not know
    about (`era5 store ... eta 1:55:2922:04:49 - extract - Downscaling ...`).
    Two tees over one stream is the closest a unit test gets to that, and it
    fails against a per-tee flag for the same reason the run did.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    live = _LiveConsole(tty=True)
    with (
        open(tmp_path / "era5.log", "w", encoding="utf-8") as era5_log,
        open(tmp_path / "chirps.log", "w", encoding="utf-8") as chirps_log,
    ):
        era5 = su._Tee(live, era5_log)
        chirps = su._Tee(live, chirps_log)
        era5.write_redraw("\rera5 store  ----------    0.8%  0:00:54 | eta 1:55:29")
        chirps.write("22:04:49 - extract - Downscaling era5 variables\n")
        era5.close()
        chirps.close()

    console = live.getvalue()
    assert "eta 1:55:2922:04:49" not in console, console
    assert console.endswith(
        _LINE_RESET + "22:04:49 - extract - Downscaling era5 variables\n"
    ), console


def test_tee_writes_no_escape_over_a_frame_off_a_terminal(tmp_path):
    """Nothing overwrote anything there, so there is nothing to clear."""
    log = tmp_path / "rule.log"
    live = _LiveConsole(tty=False)
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(live, handle)
        tee.write_redraw("\rforcing  ====------  40.0%  0:03 | eta 0:04")
        tee.write("08:12:03 - a - b\n")
        tee.close()

    assert "\033" not in live.getvalue()


# --- repeated source name ----------------------------------------------------


def test_compact_drops_a_source_name_that_repeats_the_file_stem():
    line = (
        "2026-08-17 22:54:30,101 - x - data_source - INFO - "
        "Reading rlz_1_st_3 Dataset data from climate/weathergenr/output/rlz_1_st_3.nc\n"
    )
    assert _compact_log_line(line) == (
        "22:54:30 - data_source - Reading climate/weathergenr/output/rlz_1_st_3.nc\n"
    )


def test_compact_keeps_a_source_name_that_differs_from_the_file_stem():
    """``era5_orography`` is not the stem of ``era5_orography_2018.nc`` — keep both."""
    line = (
        "2026-08-17 22:54:32,001 - x - data_source - INFO - "
        "Reading era5_orography from meteo/era5/meta/era5_orography_2018.nc\n"
    )
    assert _compact_log_line(line) == (
        "22:54:32 - data_source - "
        "Reading era5_orography from meteo/era5/meta/era5_orography_2018.nc\n"
    )


def test_compact_matches_a_windows_spelt_path_stem():
    line = (
        "2026-08-17 22:54:30,101 - x - data_source - INFO - "
        "Reading rlz_1_st_3 from output\\rlz_1_st_3.nc\n"
    )
    assert _compact_log_line(line) == (
        "22:54:30 - data_source - Reading output\\rlz_1_st_3.nc\n"
    )


# --- heartbeat watchdog ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2.05_merge", "Rule 2.05: merge"),
        (
            "2.04_fetch_cmip6_projections/cmip6_INM_x",
            "Rule 2.04: fetch_cmip6_projections  [cmip6_INM_x]",
        ),
        (
            "3.12_perturb_climate_realization/rlz_1_st_2",
            "Rule 3.12: perturb_climate_realization  [rlz 1 | st 2]",
        ),
        ("3.15_run_wflow/batch_0", "Rule 3.15: run_wflow  [batch 0]"),
        ("1.14_export_simulation_tables", "Rule 1.14: export_simulation_tables"),
        ("busy_rule", "busy_rule"),
    ],
)
def test_heartbeat_identity_spells_the_job_as_the_run_and_done_lines_do(
    label, expected
):
    """The watchdog knows the job by its log-parts label; the console does not."""
    assert su._heartbeat_identity(label) == expected


def _driven(label, stream, interval=1.0, **kwargs):
    """An unstarted watchdog on a synthetic clock: silence begins at t=0.

    Tests advance it with ``hb._tick(t)`` instead of sleeping, so the schedule
    is exact and a slow runner cannot move a notice across an assertion.
    """
    hb = _Heartbeat(label, stream, interval=interval, **kwargs)
    hb._start = hb._last = hb._seen = 0.0
    return hb


def _tick_through(hb, *times):
    for t in times:
        hb._tick(float(t))


def _wait_for_notice(capsys, seconds=2.0):
    """Collect stderr until a watchdog notice appears (or ``seconds`` pass)."""
    err = ""
    deadline = time.monotonic() + seconds
    while "elapsed" not in err and time.monotonic() < deadline:
        time.sleep(0.02)
        err += capsys.readouterr().err
    return err


def test_noninteractive_heartbeat_reports_elapsed_without_summary():
    stream = io.StringIO()
    hb = _driven("2.05_merge", stream)
    _tick_through(hb, *range(1, 7))
    lines = stream.getvalue().splitlines()
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d - execution - Rule 2\.05: merge "
        r"\d:\d\d:\d\d elapsed",
        lines[0],
    ), lines[0]
    assert len(lines) == 6
    assert "no output for" not in stream.getvalue()
    assert "done in" not in stream.getvalue()
    assert "2.05_merge" not in stream.getvalue()


def test_heartbeat_suppressed_while_active():
    stream = io.StringIO()
    hb = _driven("busy_rule", stream)
    for t in range(1, 7):  # output lands just before every tick
        hb._last = t - 0.1
        hb._tick(float(t))
    assert "elapsed" not in stream.getvalue()


def _still_running(stream):
    return [line for line in stream.getvalue().splitlines() if "elapsed" in line]


def test_heartbeat_notices_keep_a_fixed_interval():
    """Redirected streams receive an elapsed row on every quiet interval."""
    stream = io.StringIO()
    hb = _driven("2.04_fetch_cmip6_projections", stream)
    _tick_through(hb, *range(1, 19))
    assert len(_still_running(stream)) == 18, stream.getvalue()


def test_heartbeat_defaults_to_fifteen_second_updates(monkeypatch):
    monkeypatch.delenv("CST_HEARTBEAT_SECS", raising=False)
    stream = io.StringIO()
    hb = _Heartbeat("2.05b_prepare_projection_analysis", stream)
    hb._start = hb._last = hb._seen = 0.0
    _tick_through(hb, 14, 15, 29, 30, 44, 45)
    lines = _still_running(stream)
    assert len(lines) == 3
    assert [line.rsplit(" ", 2)[-2] for line in lines] == [
        "0:00:15",
        "0:00:30",
        "0:00:45",
    ]


def test_heartbeat_first_notice_lands_at_one_interval():
    stream = io.StringIO()
    hb = _driven("2.04_fetch_cmip6_projections", stream)
    hb._tick(0.9)
    assert not _still_running(stream)
    hb._tick(1.0)
    assert len(_still_running(stream)) == 1


def test_heartbeat_thread_reports_a_real_silence():
    """End to end on the real clock: the thread does drive `_tick`.

    Polls with half a second of slack rather than sleeping a fixed time; a
    0.13 s sleep for a notice due at 0.10 s flaked on a slow Windows runner
    (CI 36123429933).
    """
    stream = io.StringIO()
    hb = _Heartbeat("2.04_fetch_cmip6_projections", stream, interval=0.2).start()
    deadline = time.monotonic() + 0.9
    while not _still_running(stream) and time.monotonic() < deadline:
        time.sleep(0.01)
    hb.stop()
    assert len(_still_running(stream)) == 1


def test_heartbeat_interval_resets_when_output_resumes():
    """Output resuming resets the next notice to one quiet interval later."""
    stream = io.StringIO()
    hb = _driven("2.04_fetch_cmip6_projections", stream)
    _tick_through(hb, *range(1, 11))
    assert len(_still_running(stream)) == 10
    hb._last = 10.5
    hb._tick(11.0)
    assert len(_still_running(stream)) == 10
    hb._tick(11.5)
    assert len(_still_running(stream)) == 11


def test_heartbeat_hands_a_stall_to_the_bar_when_one_is_open():
    """`on_stall` answering the silence replaces the watchdog status."""
    stream = io.StringIO()
    answered = []
    hb = _driven(
        "3.15_run_wflow",
        stream,
        on_stall=lambda: (answered.append(1), True)[1],
    )
    _tick_through(hb, 1, 2, 3)

    assert answered  # the hook was consulted rather than bypassed
    assert stream.getvalue() == ""


def test_heartbeat_still_beeps_when_the_hook_declines():
    """A rule with no bar to redraw -- 3.06, 3.12, 3.14 -- keeps the notice."""
    stream = io.StringIO()
    hb = _driven("3.06_weathergen", stream, on_stall=lambda: False)
    _tick_through(hb, 1, 2)

    assert "elapsed" in stream.getvalue()


def test_heartbeat_records_the_quiet_period_even_when_the_bar_answered_it():
    """The console presentation changed; the silence itself did not.

    `quiet_rows` is the durable record a log reader gets, and a stall covered by
    a redrawn bar is still a stretch in which the rule produced no output.
    """
    stream = io.StringIO()
    hb = _driven("3.15_run_wflow", stream, on_stall=lambda: True)
    _tick_through(hb, 1, 2, 3)
    hb._last = 3.2  # output resumes, closing the period
    hb._tick(4.0)

    assert any("quiet for" in row for row in hb.quiet_rows())


def test_heartbeat_disabled_when_interval_zero():
    stream = io.StringIO()
    hb = _Heartbeat("off", stream, interval=0).start()
    time.sleep(0.05)
    hb.stop()
    assert stream.getvalue() == ""  # nothing at all, not even a summary


def test_heartbeat_reports_no_failure_for_systemexit_zero(tmp_path, capsys):
    # A `script:` module ends its cache-hit path with `raise SystemExit(0)`, which
    # IS a success -- Snakemake reports the job Finished. The watchdog must not
    # contradict that with its failure-only terminal verdict.
    log = tmp_path / "rule.log"
    with pytest.raises(SystemExit):
        with tee_to_log(log, heartbeat_interval=0.05):
            err = _wait_for_notice(capsys)
            raise SystemExit(0)
    err += capsys.readouterr().err
    assert "elapsed" in err and "failed after" not in err
    assert "done in" not in err


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


@pytest.mark.parametrize(
    "row",
    [
        "14:02:03 - model - Initializing wflow_sbm model from hydromt_wflow (v1.0.2).\n",
        "14:02:03 - config - Reading model config file from <model>/wflow_sbm.toml.\n",
        "14:02:03 - config - Reading default config file from a/b.toml.\n",
        "14:02:03 - wflow_base - Supported Wflow.jl version v1+\n",
        "14:02:03 - tables - Reading model table files.\n",
        "14:02:03 - tables - No tables found, skip writing.\n",
        "14:02:03 - grid - No grid data found, skip writing.\n",
    ],
)
def test_tee_mutes_hydromt_model_open_boilerplate(tmp_path, capsys, row):
    """Every rule that touches the built model reopens it and re-announces itself.

    Ten WF3 downscale members print each of these once; none of them names the
    member, so the console cannot use them to tell one job from another.
    """
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    assert row.split(" - ", 2)[2].rstrip() not in capsys.readouterr().out
    assert row.split(" - ", 2)[2].rstrip() in log.read_text(encoding="utf-8")


def test_tee_mutes_hydromt_boilerplate_after_its_full_stop_is_dropped(tmp_path, capsys):
    """The mute sees the COMPACTED row, which has no closing stop.

    A prefix spelled with hydromt's `.` matched nothing once the compactor
    started dropping it -- three rows came back on the console for a day.
    """
    log = tmp_path / "rule.log"
    raw = (
        "2026-09-05 12:24:52,001 - hydromt_wflow.wflow_base - tables - INFO - "
        "Reading model table files.\n"
    )
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(raw)
    assert "Reading model table files" not in capsys.readouterr().out
    assert "tables - Reading model table files" in log.read_text(encoding="utf-8")


def test_tee_mutes_a_row_whose_component_prefix_survived_compaction(tmp_path, capsys):
    """`grid - wflow_sbm.states: No grid data found` keeps its prefix by design.

    `_compact_log_line` drops the prefix only when it repeats the module column;
    here it does not, so the mute has to look past it or never fire.
    """
    log = tmp_path / "rule.log"
    row = "14:02:03 - grid - wflow_sbm.states: No grid data found, skip writing.\n"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    assert "No grid data found" not in capsys.readouterr().out
    assert "wflow_sbm.states: No grid data found" in log.read_text(encoding="utf-8")


def test_tee_mutes_the_forcing_write_announcement_but_not_the_path(tmp_path, capsys):
    """`Write forcing file` announces what the NEXT row states with its target.

    The pair is one fact twice, so the opener is muted -- but the row that names
    the file is the one a reader is actually looking for, and a mute that took
    both would be the defect. WF3 writes one forcing file per member, so this is
    a row per downscale rule, not a one-off.
    """
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write("14:02:03 - forcing - Write forcing file\n")
        sys.stdout.write(
            "14:02:03 - forcing - Writing file <experiment>/inmaps_rlz_1_st_3.nc\n"
        )
    out = capsys.readouterr().out
    logged = log.read_text(encoding="utf-8")
    assert "Write forcing file" not in out
    assert "Writing file <experiment>/inmaps_rlz_1_st_3.nc" in out
    # The durable record keeps everything; only the console mirror drops a row.
    assert "Write forcing file" in logged
    assert "Writing file <experiment>/inmaps_rlz_1_st_3.nc" in logged


def test_forcing_mute_cannot_silence_a_raised_level(tmp_path, capsys):
    """The rename case announces itself at WARNING and must survive the mute.

    That row is the reason the opener is safe to drop: it names both paths
    itself, so it never depended on `Write forcing file` as its anchor.
    """
    log = tmp_path / "rule.log"
    row = (
        "14:02:03 - forcing - WARNING - Write forcing file skipped: "
        "inmaps_historical.nc already exists.\n"
    )
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    assert "already exists" in capsys.readouterr().out
    assert "already exists" in log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "uri,logged",
    [
        (
            "gs://cmip6/CMIP6/CMIP/INM/INM-CM4-8/historical/r1i1p1f1/Amon/{variable}"
            "/gr1/v20190530",
            # The log keeps the row, in the log's own spelling: the CMIP6 store
            # prefix is abbreviated in BOTH sinks, exactly as a project path is.
            "<cmip6>/CMIP/INM/INM-CM4-8/historical/r1i1p1f1/Amon/{variable}"
            "/gr1/v20190530",
        ),
        ("s3://some-bucket/a/b/c.zarr", "s3://some-bucket/a/b/c.zarr"),
    ],
)
def test_tee_mutes_hydromts_object_store_read_echo(tmp_path, capsys, uri, logged):
    """WF2 prints the same URI one row earlier, in both catalog branches.

    hydromt's echo is 162-175 characters -- the longest rows WF2 produces --
    and it repeats `fetch_gcm_raw`'s own row, so the console drops it.
    """
    log = tmp_path / "rule.log"
    row = (
        f"14:02:03 - data_source - Reading cmip6_INM/INM-CM4-8_historical from {uri}\n"
    )
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    out = capsys.readouterr().out
    # Asserted on the ENTRY NAME, not on the URI. `uri not in out` passes for a
    # `gs://` row that merely got abbreviated to `<cmip6>/`, so it could not tell
    # a muted row from a printed one -- and for two months it did not: the
    # `<cmip6>` rewrite runs BEFORE the mute is tested, and defeated the
    # scheme-only pattern the mute was written as.
    assert "INM-CM4-8_historical" not in out
    assert logged in log.read_text(encoding="utf-8")


def test_tee_keeps_a_local_data_source_read_on_the_console(tmp_path, capsys):
    """The mute is scoped to a URI SCHEME; a filesystem read still prints.

    WF1's `Reading merit_hydro_index from <data>/.../basin_index.gpkg` says
    which basin dataset a build actually opened, and nothing else repeats it.
    """
    log = tmp_path / "rule.log"
    row = (
        "14:02:03 - data_source - Reading merit_hydro_index from "
        "<data>/topography/merit_hydro/basin_index.gpkg\n"
    )
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    assert "merit_hydro_index" in capsys.readouterr().out


@pytest.mark.parametrize(
    "message",
    [
        "gcsfs extended-filesystem switch = 'false'",
        "store calendar=noleap (tas)",
        "Wrote raw cmip6_NCC_NorESM2-MM_ssp245_r1i1p1f1.nc (0.07 MB, 1032 steps)",
    ],
)
def test_tee_mutes_the_fetch_rows_that_repeat_the_run_or_the_artifact(
    tmp_path, capsys, message
):
    """WF2's fetch prints these once per slice, and a staging run is 161 slices.

    Driven through `log_row` rather than a hand-written string, because the
    emitter is half of what is being pinned: `print` splits a row into two
    writes and `_muted_on_console` refuses a partial chunk, so a row emitted
    that way could never match however the table is spelled.
    """
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        log_row(message, module="fetch")
    assert message not in capsys.readouterr().out
    assert message in log.read_text(encoding="utf-8")


def test_tee_keeps_the_fetch_row_that_says_which_slice_is_being_read(tmp_path, capsys):
    """Several fetch jobs run at once and one store open can take twenty minutes.

    This is the only row that attributes that wait to a source -- under
    Snakemake's `-c 3` and under `stage_cmip6.py`'s worker pool alike -- so it
    stays on the console whatever else from `fetch` is muted for volume.
    """
    log = tmp_path / "rule.log"
    row = "Fetching cmip6_NCC/NorESM2-MM_ssp245_r1i1p1f1"
    with tee_to_log(log, heartbeat_interval=0):
        log_row(row, module="fetch")
    assert "NorESM2-MM_ssp245" in capsys.readouterr().out


def test_tee_keeps_the_gcsfs_switch_row_that_explains_a_slow_run(tmp_path, capsys):
    """The muted spelling is the INFO case; every other value is a WARNING.

    `hns_switch_row` is called rather than quoted, so the test fails if the
    module ever reports an unexpected value at INFO -- which is the one way the
    volume mute above could start hiding the row that explains a 14x slowdown.
    """
    from blueearth_cst.projections.fetch_gcm_raw import hns_switch_row

    message, level = hns_switch_row("true")
    assert level == "WARNING"
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        log_row(message, module="fetch", level=level)
    out = capsys.readouterr().out
    assert "WARNING" in out and "remote reads may be slower" in out


def test_log_row_emits_one_write_so_the_console_mute_can_see_it(monkeypatch):
    """A row is handed to the stream whole, newline included.

    `print` would send the text and the newline separately, and the mute
    predicate rejects a chunk that is not exactly one terminated line. That is
    not a detail of this function: it is what decides whether OUR rows can be
    muted at all, so it is pinned here rather than only through a tee.
    """
    chunks = []

    class _Recorder:
        def write(self, text):
            chunks.append(text)
            return len(text)

        def flush(self):
            pass

    monkeypatch.setattr(su.sys, "stdout", _Recorder())
    log_row("one chunk", module="fetch")
    assert len(chunks) == 1
    assert chunks[0].endswith("one chunk\n")


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
        err = _wait_for_notice(capsys)  # silence triggers a console heartbeat
    err += capsys.readouterr().err
    logged = log.read_text(encoding="utf-8")
    assert "elapsed" in err and "done in" not in err
    assert "elapsed" not in logged and "done in" not in logged


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


def test_warn_in_repo_project_dir_warns(tmp_path, capsys):
    repo = tmp_path / "repo"
    (repo / "scratch_run").mkdir(parents=True)
    fired = cs.warn_if_project_dir_in_repo(repo / "scratch_run", repo)
    assert fired is True
    err = capsys.readouterr().err
    assert "inside the repo tree" in err and "scratch_run" in err


def test_warn_exempt_test_case_is_silent(tmp_path, capsys):
    """The fixture exemption: the baseline seed config is TRACKED, and a
    tracked config cannot carry a machine-specific absolute path."""
    repo = tmp_path / "repo"
    (repo / "test_case" / "test_local").mkdir(parents=True)
    fired = cs.warn_if_project_dir_in_repo(repo / "test_case" / "test_local", repo)
    assert fired is False
    assert capsys.readouterr().err == ""


def test_warn_absolute_out_of_tree_is_silent(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere" / "my_project"
    outside.mkdir(parents=True)
    fired = cs.warn_if_project_dir_in_repo(outside, repo)
    assert fired is False
    assert capsys.readouterr().err == ""


def test_warn_uses_containment_not_string_prefix(tmp_path):
    """`test_caseX` must not read as inside `test_case`, and a sibling repo
    directory must not read as inside the repo. A startswith() implementation
    passes the three cases above and fails both of these."""
    repo = tmp_path / "repo"
    (repo / "test_caseX").mkdir(parents=True)
    assert cs.warn_if_project_dir_in_repo(repo / "test_caseX", repo) is True

    sibling = tmp_path / "repo_other"
    sibling.mkdir()
    assert cs.warn_if_project_dir_in_repo(sibling, repo) is False


# --- climate_store_rule (R07 B1) ---------------------------------------------

#: R14 `C-70`: `climate.window` is a pair of INCLUSIVE YEARS. The ISO strings
#: below are what `climate_store_rule` still puts in `params:`, and they are
#: BYTE-IDENTICAL to the v1 values -- which is the property the two assertions
#: about them exist to hold.
_WINDOW = {"start": 2000, "end": 2020}
_WINDOW_ISO = {"starttime": "2000-01-01T00:00:00", "endtime": "2020-12-31T00:00:00"}


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
    assert spec.params["starttime"] == _WINDOW_ISO["starttime"]
    assert spec.params["endtime"] == _WINDOW_ISO["endtime"]


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
        _spec(historical_window=(2000, 2020))


def test_climate_store_rule_rejects_a_non_year_window():
    """R14 `C-70` made the sub-day case unrepresentable, so this is its heir.

    The store key is day-resolution, and a sub-day window could not be spelled
    in it. Under ISO endpoints that had to be REFUSED (`slugify_window` still
    carries the guard, now unreachable from `climate.window`); under inclusive
    YEARS it cannot be written down at all. What can still go wrong is an
    endpoint that is not a year, so that is what this asserts.
    """
    with pytest.raises(ValueError, match="not a year"):
        _spec(historical_window={"start": "2000-01-01T06:00:00", "end": 2020})


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
        r"Parsing data catalog from C:\workspace\blueearth_cst\.pixi"
        r"\envs\default\Lib\site-packages\hydromt_wflow\data\parameters_data.yml"
    )
    assert _rel(line) == (
        r"Parsing data catalog from <site-packages>/hydromt_wflow\data"
        r"\parameters_data.yml"
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


def test_the_cmip6_store_prefix_is_abbreviated():
    """17 constant characters in front of the only part that identifies a run."""
    line = (
        "entry=cmip6_MPI-ESM1-2-HR_ssp585 uri=gs://cmip6/CMIP6/ScenarioMIP/"
        "DKRZ/MPI-ESM1-2-HR/ssp585/r1i1p1f1/Amon/pr/gn/v20190710/"
    )
    assert _rel(line) == (
        "entry=cmip6_MPI-ESM1-2-HR_ssp585 uri=<cmip6>/ScenarioMIP/"
        "DKRZ/MPI-ESM1-2-HR/ssp585/r1i1p1f1/Amon/pr/gn/v20190710/"
    )


def test_a_non_cmip6_remote_uri_is_left_alone():
    """The table is a closed list of stores we actually read, not a URI rule."""
    line = "uri=s3://somebucket/CMIP6/ScenarioMIP/x.nc"
    assert _rel(line) == line


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
    """The preparation uses the explicitly declared native log path."""
    import ast

    src = (
        Path(__file__).resolve().parents[1]
        / "blueearth_cst"
        / "experiment"
        / "downscale_climate_forcing.py"
    ).read_text(encoding="utf-8")
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node)
        == "os.path.relpath(settings.native_log_path, config_out_root)"
        for node in ast.walk(ast.parse(src))
    )
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


# --- seed resolution (generate_scenarios `seed:`) ------------------------------------------------
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
    with pytest.raises(ValueError, match=r"`seed:`"):
        su.resolve_seed(bad, "exp")


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_seed_refuses_non_integers_and_negatives(bad):
    with pytest.raises(ValueError, match=r"`seed:`"):
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
            "n_levels": 2,
            "trajectory": "transient",
            "variance": {"min": [1.0] * 12, "max": [1.0] * 12},
        },
        "precip": {"n_levels": 2, "trajectory": "transient"},
    }
    with pytest.raises(ValueError, match="not a supported stress dimension"):
        su.stress_test_grid(cfg)


def test_stress_test_refuses_a_typo_in_an_axis():
    """The closure is general, not a one-off ban on `variance`."""
    cfg = {
        "temp": {"n_levels": 2, "trajectory": "transient", "meen": {}},
        "precip": {"n_levels": 2, "trajectory": "transient"},
    }
    with pytest.raises(ValueError, match=r"unsupported key\(s\) \['meen'\]"):
        su.stress_test_grid(cfg)


def test_precip_variance_is_still_accepted():
    """Only TEMPERATURE variance is refused; precip variance is live."""
    cfg = {
        "temp": {"n_levels": 2, "trajectory": "transient"},
        "precip": {
            "n_levels": 2,
            "trajectory": "transient",
            "variance": {"min": [1.0] * 12, "max": [1.0] * 12},
        },
    }
    assert su.stress_test_grid(cfg) == (2, 2, 4)


# --- one wflow_outvars default (defect A) -----------------------------------


def test_wflow_outvars_default_is_not_empty():
    """`[]` meant zero indicator tables and no error.

    WF3 defaulted to it while WF1 defaulted to two variables, so a config
    omitting the key ran to completion and wrote nothing --
    `project_config_baseline_linux.yml` omits it, so that was shipped.
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
    return cs._ConsoleHandler(base)


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
    cs._RULE_NUMBERS["seed"] = "9.01"
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
    cs._RULE_NUMBERS["downscale_scenario_series"] = "3.14"
    handler = _console_handler()
    _emit(
        handler,
        _job_info(
            3,
            "downscale_scenario_series",
            "Rule 3.14: downscale_scenario_series  [rlz 1 | st 0]",
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
        r"\d\d:\d\d:\d\d - DONE Rule 3\.14: downscale_scenario_series  "
        r"\[rlz 1 \| st 0\]  0:01:11  \[27/37\]",
        done,
    ), done


def test_console_start_line_carries_no_progress_counter():
    """The counter is a FINISH-line fact, and was a start-line lie.

    Replayed from the last progress record, it is the tally BEFORE the job the
    line names: two consecutive starts repeat a number, and the last start of a
    run reads `[13/14]` with nothing finished. The finish line keeps it, where
    it is true.
    """
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "perturb", "Rule 3.12: perturb  [rlz 1 | st 1]"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=8, total=37),
        _job_info(2, "perturb", "Rule 3.12: perturb  [rlz 1 | st 2]"),
    )
    lines = out.splitlines()
    assert lines[0].endswith("[rlz 1 | st 1]"), lines[0]
    assert lines[-1].endswith("[rlz 1 | st 2]"), lines[-1]
    # Once, on the finish line -- not three times across the pair.
    assert out.count("[8/37]") == 1, out


def test_console_quiet_start_prints_only_the_finish_line(monkeypatch):
    """A bookkeeping rule's start line is the one it can afford to lose.

    WF3's 3.09 copies one payload per member, interleaved with the rule that
    produced it -- two lines per member, 800 on a 10 x 20 grid.
    """
    monkeypatch.setattr(cs, "_QUIET_START_RULES", {"retain"})
    cs._RULE_NUMBERS["retain"] = "3.09"
    out = _emit(
        _console_handler(),
        _job_info(1, "retain", "Rule 3.09: retain  [run 01]"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=6, total=24),
    )
    assert "RUN " not in out, out
    assert re.search(r"DONE Rule 3\.09: retain .*\[6/24\]", out), out


def test_console_quiet_start_is_opt_in(monkeypatch):
    """Every other rule keeps its start line."""
    monkeypatch.setattr(cs, "_QUIET_START_RULES", set())
    out = _emit(_console_handler(), _job_info(1, "perturb", "Rule 3.12: perturb"))
    assert "RUN  Rule 3.12: perturb" in out, out


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


def test_indicator_start_and_finish_show_set_names_instead_of_hashes(monkeypatch):
    monkeypatch.setattr(cs, "_INDICATOR_CONTEXTS", {})
    monkeypatch.setitem(cs._RULE_NUMBERS, "prepare_indicator_plan", "4.07")
    cs.register_indicator_context(
        "indicator_request_id", "opaque", ["default", "flow_regime"]
    )
    out = _emit(
        _console_handler(),
        _job_info(
            1,
            "prepare_indicator_plan",
            "Rule 4.07: prepare_indicator_plan [indicator_request opaque]",
            {"indicator_request_id": "opaque"},
        ),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert out.count("Check simulation outputs for indicators") == 2
    assert out.count("[default, flow_regime]") == 2
    assert "opaque" not in out


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


def test_console_finish_line_shows_series_identifier_without_label():
    """CMIP6 identifiers remain visible without a redundant series label."""
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "x", {"series_key": "cmip6_x"}),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "[cmip6_x]" in out
    assert "[series " not in out and "series_key" not in out


def test_console_finish_line_drops_the_id_suffix_the_banner_drops():
    """`run_id` is the WILDCARD's name; `run` is the console's, on both lines.

    The `_num` defect two suffixes along, and the one with the most wildcards
    behind it: `collection_id`, `run_id`, `indicator_request_id` and
    `indicator_set_id` are every fan-out wildcard WF3 and WF4 have. WF4's 4.04
    banner wrote `run {wildcards.run_id}` against a finish line reading
    `[run_id 07]` from the day it was written.
    """
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "x", {"collection_id": "967615227842", "run_id": "07"}),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "[collection 967615227842 | run 07]" in out
    assert "run_id" not in out and "collection_id" not in out


@pytest.mark.parametrize(
    "name", ["all", "simulations_only", "simulations_and_indicators"]
)
def test_console_target_counts_resolved_inputs(name):
    """Count actual job inputs, including targets resolved after checkpoints."""
    handler = _console_handler()
    record = _job_info(1, name, "Confirm requested outputs")
    record.input = ["a/x.csv", "logs/wf1.log"]
    out = _emit(
        handler,
        _console_record(event="progress", done=19, total=20),
        record,
    )
    lines = out.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("Requested outputs: 2/2 provided")
    assert "a/x.csv" not in out and "logs/wf1.log" not in out


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


def test_console_checkpoint_workflow_omits_provisional_job_totals(monkeypatch):
    """A dynamic DAG keeps one stable counter while Snakemake expands its total."""
    monkeypatch.setattr(cs, "_DYNAMIC_PROGRESS_RULES", {"checkpoint", "downstream"})
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "checkpoint", "Rule 3.04: checkpoint"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=2, total=5),
        _job_info(2, "downstream", "Rule 3.05: downstream"),
        _console_record(event="job_finished", job_id=2),
        _console_record(event="progress", done=3, total=35),
    )

    finished = [line for line in out.splitlines() if " - DONE " in line]
    assert [line.split("  ")[-1] for line in finished] == ["[2]", "[3]"]


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


@pytest.mark.parametrize(
    "row",
    [
        "14:02:03 - log - HydroMT version: 1.3.1\n",
        "14:02:03 - model - update: setup_precip_forcing\n",
        "14:02:03 - model - setup_temp_pet_forcing.pet_method=debruin\n",
    ],
)
def test_tee_mutes_the_hydromt_cli_update_chatter(tmp_path, capsys, row):
    """Rule 1.10's `hydromt update -vv` announces its version three times, then
    names each step and echoes every keyword of it. The log part keeps them."""
    log = tmp_path / "rule.log"
    with tee_to_log(log, heartbeat_interval=0):
        sys.stdout.write(row)
    assert row.split(" - ", 2)[2].rstrip() not in capsys.readouterr().out
    assert row.split(" - ", 2)[2].rstrip() in log.read_text(encoding="utf-8")


def test_paint_body_demotes_the_empty_forcing_warning():
    """Painted as body, text and level untouched; any other WARNING stays orange."""
    demoted = [
        "14:02:03 - forcing - WARNING - Write forcing skipped: dataset is empty (no variables or data)\n",
    ]
    for row in demoted:
        painted = su._paint_body(row, True)
        # Body tier: the scaffolding dims and the message keeps the terminal's
        # own foreground, so the only escape is the stamp's.
        assert painted == (
            su._ansi("14:02:03 - " + row.split(" - ")[1] + " - ", su._ANSI_DIM)
            + row.split(" - ", 2)[2].rstrip("\n")
            + "\n"
        ), painted
    kept = "14:02:03 - states - WARNING - CRS not found in states data, setting to model CRS\n"
    assert su._paint_body(kept, True) == (
        su._ansi(kept.replace("WARNING - ", "").rstrip("\n"), su._ANSI_ALERT) + "\n"
    )
    # The match is module AND prefix: the same words under another module keep
    # their colour, so the list cannot widen by accident.
    other = "14:02:03 - model - WARNING - Write forcing skipped: dataset is empty (no variables or data)\n"
    assert su._ANSI_ALERT in su._paint_body(other, True)


def test_log_row_flushes_after_its_one_write(monkeypatch):
    """Off a terminal stdout is block-buffered, so an unflushed row lands after
    the DONE line Snakemake writes from the parent process."""
    events = []

    class _Recorder:
        def write(self, text):
            events.append("write")
            return len(text)

        def flush(self):
            events.append("flush")

    monkeypatch.setattr(su.sys, "stdout", _Recorder())
    log_row("one chunk", module="fetch")
    assert events == ["write", "flush"]


def test_console_job_stats_collapse_to_one_line(monkeypatch):
    """The FALLBACK, for a workflow whose rules declared no numbers.

    `run_info` renders the plan block whenever `_RULE_NUMBERS` can name the
    rules; this one-line collapse is what is left when it cannot. The registry
    is cleared explicitly rather than trusted to be empty -- it is module
    state, and any earlier test in this file that calls `rule_banner` fills it,
    which is exactly how this test started exercising the wrong branch.
    """
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {})
    handler = _console_handler()
    table = (
        "Job stats:\njob                              count\n"
        "-----------------------------  -------\n"
        "all                                  1\n"
        "downscale_scenario_series       10\n"
        "perturb_climate_realization          8\n"
        "run_wflow_batch_0                    1\n"
        "total                               20\n"
    )
    out = _emit(handler, _console_record(table, event="run_info"))
    assert out == (
        "20 jobs across 4 rules  "
        "(downscale_scenario_series x10, perturb_climate_realization x8)\n"
    ), out
    flat = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\nall  1\nx  1\ntotal  2\n",
            event="run_info",
        ),
    )
    assert flat == "2 jobs across 2 rules\n", flat
    one = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\nall  1\ntotal  1\n", event="run_info"
        ),
    )
    assert one == "1 job across 1 rule\n", one


def test_console_run_info_renders_the_plan_block(monkeypatch):
    """The other branch: a registry that CAN name the rules.

    End to end through the handler, because `tests/test_plan_block.py` covers
    the renderer in isolation and this is the only place the wiring -- the
    `run_info` event reaching `_plan_block`, and the whole block arriving as
    one string so its blank line survives `_render`'s truthiness filter -- is
    exercised.
    """
    monkeypatch.setattr(
        cs,
        "_RULE_NUMBERS",
        {"snapshot_config": "1.01", "run_wflow": "1.14"},
    )
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\nall  1\nsnapshot_config  1\ntotal  2\n",
            event="run_info",
        ),
    )
    assert out == (
        "status  step  rule             jobs\n"
        "run     1.01  snapshot_config     1\n"
        "reuse   1.14  run_wflow           -\n"
        "1 job  |  1 rule scheduled  |  1 rule up to date\n"
        "\n"
    ), out


# --- the opening block (title + plan + metadata, assembled at run_info) ------


def _declared_header(monkeypatch, **details):
    """Declare a run header the way `open_run_header` does from `onstart:`."""
    monkeypatch.setattr(
        cs, "_RUN_HEADER", ("wf1 build_model", "test_case/test_rapid", None, details)
    )
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {"a": "1.01", "b": "1.02"})


def test_console_opening_puts_the_rules_under_the_title(monkeypatch):
    _declared_header(monkeypatch)
    out = _emit(
        _console_handler(),
        _console_record("Job stats:\na 1\ntotal 1", event="run_info"),
    )
    assert (
        out.index("wf1 build_model")
        < out.index("Snakemake command")
        < out.index("Work plan")
        < out.index("Paths")
        < out.index("Execution log")
    )
    assert "run     1.01  a" in out and "reuse   1.02  b" in out


def test_the_opening_block_draws_no_rules_at_all(monkeypatch):
    """No band, no `-- SECTION --`, no footer under the table.

    Six full-width rules in thirty lines, and under `run_workflows` a seventh
    three lines above them at a DIFFERENT width -- the runner draws a fixed 80
    and this block sized itself to its longest value. What the rules were
    carrying the content carries: numbered rows are a plan, a `key  value`
    column is metadata, and a blank line separates them.
    """
    _declared_header(monkeypatch, experiment="experiment_rapid")
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\nb  1\ntotal  2\n",
            event="run_info",
        ),
    )
    lines = out.split("\n")
    assert not [line for line in lines if set(line) in ({"="}, {"-"})], lines
    assert not [line for line in lines if line.startswith("-- ")], lines
    assert lines[1] == "wf1 build_model"
    assert "2 jobs" in out and "2 rules scheduled" in out


def test_console_opening_prints_once(monkeypatch):
    """A second record must not repeat the header."""
    _declared_header(monkeypatch)
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
        _job_info(1, "a", "Rule 1.01: a"),
    )
    assert out.count("wf1 build_model") == 1, out


def test_console_opening_survives_a_run_without_run_info(monkeypatch):
    """Whichever record is first opens the run.

    `run_info` normally is, and carries the plan. If anything beats it the
    header still prints -- losing it entirely is the outcome worth guarding
    against, and a header is not something a styling detail may drop.
    """
    _declared_header(monkeypatch)
    out = _emit(_console_handler(), _job_info(1, "a", "Rule 1.01: a"))
    lines = out.split("\n")
    # No plan means no table to caption, so the summary would go back onto the
    # title -- and none was reported here.
    assert lines[1] == "wf1 build_model"
    assert "<project>  <repo>/test_case/test_rapid" in lines


def test_child_keeps_identity_when_parent_announced_sequence(monkeypatch):
    _declared_header(monkeypatch)
    monkeypatch.setenv(cs.ANNOUNCED_ENV, "1")
    out = _emit(
        _console_handler(),
        _console_record("Job stats:\na 1\ntotal 1", event="run_info"),
    )
    assert out.count("wf1 build_model") == 1
    assert out.index("wf1 build_model") < out.index("Work plan")
    assert "<project>  <repo>/test_case/test_rapid" in out


def test_the_title_stays_for_a_standalone_snakemake_run(monkeypatch):
    """`snakemake -s build_model.smk` has no runner above it to do the naming."""
    _declared_header(monkeypatch)
    monkeypatch.delenv(cs.ANNOUNCED_ENV, raising=False)
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
    )
    assert out.split("\n")[1] == "wf1 build_model"


def test_an_announced_run_keeps_the_title_when_it_carries_the_plan_summary(
    monkeypatch,
):
    """With no table, the summary rides the title -- so the title cannot go.

    Reached when Snakemake's job table cannot be parsed. Suppressing the title
    there would take the run's size with it, and a repeated name is a smaller
    loss than an unstated one.
    """
    monkeypatch.setenv(cs.ANNOUNCED_ENV, "1")
    block = cs.opening_block(
        "wf1 build_model", "test_case/test_rapid", plan=("7 of 9 rules to run", [])
    )
    assert ("wf1 build_model", "title") in block
    assert ("7 of 9 rules to run", "body") in block


def test_console_plan_stands_alone_when_no_header_was_declared(monkeypatch):
    """A bare `snakemake -s` without `onstart:`, and every test above."""
    monkeypatch.setattr(cs, "_RUN_HEADER", None)
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {"a": "1.01", "b": "1.02"})
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
    )
    assert out.startswith("status  step  rule  jobs")
    assert "1 job  |  1 rule scheduled  |  1 rule up to date" in out


def test_open_run_header_writes_immediately_without_a_console(monkeypatch):
    """A styling failure must never be the reason a run loses its header."""
    monkeypatch.setattr(cs, "_CONSOLE_STYLE_ACTIVE", False)
    monkeypatch.setattr(cs, "_RUN_HEADER", None)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    assert cs.open_run_header("wf1 build_model", "test_case/test_rapid") is False
    assert "wf1 build_model" in stream.getvalue()
    assert cs._RUN_HEADER is None  # nothing held for a handler that is not there


def test_open_run_header_holds_it_for_the_console(monkeypatch):
    monkeypatch.setattr(cs, "_CONSOLE_STYLE_ACTIVE", True)
    monkeypatch.setattr(cs, "_RUN_HEADER", None)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    assert cs.open_run_header("wf1 build_model", "test_case/test_rapid") is True
    assert stream.getvalue() == ""  # held, not written
    assert cs._RUN_HEADER[0] == "wf1 build_model"


def test_deferred_warning_lands_under_the_run_block(monkeypatch):
    """Parse-time rows belong in the run's structure, not above its title.

    They are emitted while Snakemake is still parsing the workflow -- before
    the DAG, before `onstart:`, before this handler exists -- so the only place
    they can appear in the block is where the block is BUILT.
    """
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    _declared_header(monkeypatch)
    cs.defer_warning("two members did not resolve", module="resolution")
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
    )
    lines = out.split("\n")
    row = next(i for i, line in enumerate(lines) if "did not resolve" in line)
    meta = next(i for i, line in enumerate(lines) if line.startswith("<project>  "))
    # Under the metadata rows, and last in the block: the section rule that
    # used to close it is gone, so what bounds the warning below is the blank
    # line before the first `HH:MM:SS - RUN` row.
    assert meta < row, lines
    assert [line for line in lines[row + 1 :] if line] == [], lines
    assert lines[row].endswith(
        " - resolution - WARNING - two members did not resolve"
    ), lines[row]
    assert lines[row - 1] == ""  # air between the path tokens and the warning


def test_deferred_warning_keeps_a_multi_line_block_together(monkeypatch):
    """WF2's resolution report is seven lines, and `_render` filters blanks.

    The block therefore has to travel as ONE element, its internal air inside
    the string -- the same constraint `_plan_block` states.
    """
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    _declared_header(monkeypatch)
    cs.defer_warning("head\n  one\n  two", module="resolution")
    out = _emit(
        _console_handler(),
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
    )
    lines = out.split("\n")
    row = next(i for i, line in enumerate(lines) if line.endswith("WARNING - head"))
    assert lines[row + 1 : row + 3] == ["  one", "  two"], lines


def test_deferred_warning_is_painted_as_a_warning(monkeypatch):
    """By severity, not by the opening block's body tier.

    A warning that reads as routine is the one thing this must not do -- the
    module's whole colour scheme rests on severity outranking the tier.
    """
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    _declared_header(monkeypatch)
    cs.defer_warning("two members did not resolve", module="resolution")
    handler = _console_handler()
    handler._color = True
    out = _emit(
        handler,
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
    )
    row = next(line for line in out.split("\n") if "did not resolve" in line)
    assert row.startswith(f"\x1b[{su._ANSI_ALERT}m"), repr(row)


def test_deferred_warnings_drain_once(monkeypatch):
    """Popped, not read: the opening block is assembled on the first record."""
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    _declared_header(monkeypatch)
    cs.defer_warning("held once", module="resolution")
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            "Job stats:\njob  count\n----  ---\na  1\ntotal  1\n", event="run_info"
        ),
        _job_info(1, "a", "Rule 1.01: a"),
    )
    assert out.count("held once") == 1, out
    assert cs._DEFERRED_WARNINGS == []


def test_checkpoint_warning_prints_before_workflow_completion(monkeypatch, capsys):
    """Warnings discovered after startup must not wait until process exit."""
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    monkeypatch.setattr(cs, "_HEADER_PRINTED", True)
    cs.defer_rows("19:02:02 - responses - WARNING - Duplicate location kept once\n")
    assert "Duplicate location kept once" in capsys.readouterr().err
    assert cs._DEFERRED_WARNINGS == []


def test_short_reference_warning_is_a_plain_note(monkeypatch):
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    message = (
        "Reference period is short: 11 years (1995-2005); recommended minimum: 20 years"
    )
    cs.defer_warning(message, module="reference_window")
    assert cs._drain_deferred_warnings(False) == [message]


def test_owned_wf4_defers_done_all_to_its_launcher(monkeypatch):
    monkeypatch.setenv("CST_SIMULATION_FINALIZE", "1")
    handler = _console_handler()
    output = _emit(
        handler,
        _job_info(0, "all", "Requested outputs: 1/1 provided"),
        _console_record(event="job_finished", job_id=0),
        _console_record(event="progress", done=1, total=1),
    )
    assert "DONE all" not in output


def test_deferred_warning_flushes_when_the_console_style_did_not_take(monkeypatch):
    """The unstyled path prints its own header, so it flushes on the spot."""
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    monkeypatch.setattr(cs, "_CONSOLE_STYLE_ACTIVE", False)
    monkeypatch.setattr(cs, "_RUN_HEADER", None)
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    cs.defer_warning("two members did not resolve", module="resolution")
    assert cs.open_run_header("wf1 build_model", "test_case/test_rapid") is False
    out = stream.getvalue()
    assert out.index("wf1 build_model") < out.index("did not resolve")
    assert cs._DEFERRED_WARNINGS == []


def test_undrained_warning_is_printed_out_of_position_not_dropped(monkeypatch):
    """`--dry-run` never fires `onstart:`, so no flush site is ever reached.

    Measured 2026-09-17: a dry run shows Snakemake's own `Job stats:` and
    nothing of this module. A deferred row must not be lost there -- the
    resolution report is one of the things a dry-run reader came for.
    """
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    cs.defer_warning("two members did not resolve", module="resolution")
    cs._flush_undrained_warnings()
    assert "resolution - WARNING - two members did not resolve" in stream.getvalue()
    # Drained, so a healthy run -- one that reached an ordered flush site --
    # writes nothing here.
    quiet = io.StringIO()
    monkeypatch.setattr(sys, "stderr", quiet)
    cs._flush_undrained_warnings()
    assert quiet.getvalue() == ""
    assert cs._DEFERRED_WARNINGS == []


def test_console_an_unparsed_run_info_passes_through():
    handler = _console_handler()
    out = _emit(handler, _console_record("Job counts unavailable.", event="run_info"))
    assert out == "Job counts unavailable.\n"


@pytest.mark.parametrize("colour", [False, True])
def test_nothing_to_be_done_is_restated_as_an_up_to_date_verdict(monkeypatch, colour):
    """A logger filter, because onstart -- and so the handler -- never runs then."""
    import logging as _logging

    monkeypatch.setattr(cs, "_console_colour", lambda stream: colour)
    record = _logging.LogRecord(
        "snakemake.logging",
        _logging.INFO,
        __file__,
        1,
        "Nothing to be done (all requested files are present and up to date).",
        (),
        None,
    )
    assert cs._UpToDateVerdict().filter(record) is True
    expected = f"\033[{cs._ANSI_DONE}mUp to date\033[0m" if colour else "Up to date"
    assert record.getMessage() == "\n" + expected


def test_the_verdict_filter_is_on_snakemakes_logger_once():
    import snakemake.logging as sm_logging

    cs._install_up_to_date_verdict()
    cs._install_up_to_date_verdict()
    installed = [
        f for f in sm_logging.logger.filters if isinstance(f, cs._UpToDateVerdict)
    ]
    assert len(installed) == 1


def test_console_the_preamble_is_left_alone():
    """It is printed before `onstart`, so muting it would be a dead rule."""
    handler = _console_handler()
    out = _emit(
        handler, _console_record("Assuming unrestricted shared filesystem usage.")
    )
    assert out == ""


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
        _console_record("Workflow diagnostic"),
        _console_record("Error in rule x:\n    jobid: 3", event="job_error", jobid=3),
    )
    assert out == "Workflow diagnostic\nError in rule x:\n    jobid: 3\n"


def test_console_a_rule_without_a_message_keeps_the_default_block():
    """Its input/output/jobid block is all a reader gets for that rule."""
    handler = _console_handler()
    record = _job_info(2, "some_rule", "")
    record.msg = "rule some_rule:\n    output: a.nc"
    assert _emit(handler, record) == "rule some_rule:\n    output: a.nc\n"


def test_console_failure_names_the_actual_job_and_preserves_diagnosis(monkeypatch):
    monkeypatch.setitem(cs._RULE_NUMBERS, "build_model", "1.03")
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(9, "another_rule", "Another job running concurrently"),
        _console_record(
            "Error in rule build_model:\nFileNotFoundError: unavailable data",
            level=_logging.ERROR,
            event="job_error",
            rule_name="build_model",
            jobid=0,
            log=["project/logs/build.log"],
        ),
    )
    assert "FAILED Rule 1.03: build_model  |  job 0" in out
    assert "  log  project/logs/build.log" in out
    assert "FileNotFoundError: unavailable data" in out
    assert out.index("FAILED") < out.index("FileNotFoundError")
    assert "FAILED another_rule" not in out


def test_console_group_failure_lists_all_members_without_requiring_rule_numbers():
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            "Original group error",
            level=_logging.ERROR,
            event="group_error",
            job_error_info=[
                {"name": "unregistered_a", "jobid": 2, "log": ["a.log", "b.log"]},
                {"name": "unregistered_b", "jobid": 3, "log": []},
            ],
        ),
    )
    assert "FAILED unregistered_a  |  job 2" in out
    assert "FAILED unregistered_b  |  job 3" in out
    assert "  log  a.log" in out and "  log  b.log" in out
    assert "Original group error" in out


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
    cs._RULE_NUMBERS["snapshot_generation_inputs"] = "3.04"
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(
            1, "snapshot_generation_inputs", "Rule 3.04: snapshot_generation_inputs"
        ),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=3, total=37),
    )
    done = out.splitlines()[1]
    assert done.endswith("Rule 3.04: snapshot_generation_inputs  [3/37]"), done


def test_console_no_escape_codes_when_the_stream_is_not_a_tty():
    handler = _console_handler()
    out = _emit(
        handler,
        _job_info(1, "r", "Rule 1.01: r"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=1),
    )
    assert "\033" not in out


def test_console_clears_the_line_before_a_row_on_a_terminal(monkeypatch):
    """The finish line is printed by the PARENT while a job's bar is open.

    The bar is drawn in the job's own process, so nothing in this handler can
    know one is standing; it clears the line unconditionally instead. Asserted
    without colour so the escape under test is the only one in the output.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    base = _logging.StreamHandler(_LiveConsole(tty=True))
    base.name = "DefaultStreamHandler"
    base.setFormatter(_logging.Formatter("%(message)s"))
    handler = cs._ConsoleHandler(base)
    _emit(handler, _job_info(1, "r", "Rule 1.01: r"))

    assert handler.stream.getvalue().startswith(_LINE_RESET)


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
    for jobid in range(cs._CONSOLE_MAX_TRACKED_JOBS + 50):
        handler.emit(_job_info(jobid, "member", f"Rule 9.02: member  n {jobid}"))
    assert len(handler._started) == cs._CONSOLE_MAX_TRACKED_JOBS
    # The oldest go first: the most recent job is always still tracked.
    assert cs._CONSOLE_MAX_TRACKED_JOBS + 49 in handler._started
    assert 0 not in handler._started


class _TTYStringIO(io.StringIO):
    """A StringIO the console handler treats as a terminal."""

    def isatty(self):
        return True


def _colour_handler():
    base = _logging.StreamHandler(_TTYStringIO())
    base.name = "DefaultStreamHandler"
    base.setFormatter(_logging.Formatter("%(message)s"))
    return cs._ConsoleHandler(base)


def test_rule_banner_never_colours_even_on_a_tty():
    """The banner string reaches a log file and an error block, not just a
    console -- so colour belongs to whoever writes the line, not to this."""
    import blueearth_cst.shared.console_style as _su

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
    cs._RULE_NUMBERS["seed"] = "9.01"
    out = _emit(
        handler,
        _job_info(1, "seed", "Rule 9.01: seed - prepare one realization"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=4),
    )
    run_line, done_line = _unreset(out).splitlines()
    assert cs._ANSI_RUN != cs._ANSI_DONE
    # The FINISH line stays whole-line green: it is the one a reader scrolls
    # back for, and the one carrying the numbers.
    opener = f"\033[{cs._ANSI_DONE}m"
    assert done_line.startswith(opener) and done_line.endswith("\033[0m"), done_line
    assert "\033" not in done_line[len(opener) : -4], done_line
    # The START line is three tiers: a dim stamp, a blue marker, and a payload
    # in the terminal's own foreground.
    assert run_line.startswith(f"\033[{su._ANSI_DIM}m09"[:7]) or run_line.startswith(
        f"\033[{su._ANSI_DIM}m"
    ), run_line
    assert f"\033[{cs._ANSI_RUN}mRUN " in run_line, run_line
    assert run_line.endswith("prepare one realization"), run_line


def test_console_honours_no_color_even_on_a_tty(monkeypatch):
    """``NO_COLOR`` suppresses painting on a stream that reports as a terminal.

    The convention (no-color.org) is honoured by ``_ConsoleHandler``, and this
    is the only test that says so. Without it the variable is invisible: the
    painting tests below assert the opposite branch, so a regression that
    stopped honouring ``NO_COLOR`` would leave the suite green while a user who
    set it still got escapes.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    out = _emit(_colour_handler(), _console_record("Workflow diagnostic"))
    # The line reset stays: `NO_COLOR` asks for no colour, not for a cursor
    # left sitting on whatever a concurrent job's progress bar drew there.
    assert out == _LINE_RESET + "Workflow diagnostic" + "\n", repr(out)
    assert "\033[0m" not in out
    assert f"\033[{su._ANSI_BODY}m" not in out


def test_console_paints_an_informational_snakemake_line_in_the_body_tier():
    handler = _colour_handler()
    out = _emit(handler, _console_record("Workflow diagnostic"))
    # The body tier emits NO SGR: this is the terminal's own foreground.
    assert _unreset(out) == "Workflow diagnostic\n", repr(out)


def test_console_never_recolours_a_warning_or_an_error():
    """Snakemake's own red must stay the loudest thing on the console."""
    handler = _colour_handler()
    out = _emit(
        handler,
        _console_record("state file missing", level=_logging.WARNING),
        _console_record("Error in rule x:", level=_logging.ERROR, event="job_error"),
    )
    assert "\033" not in _unreset(out), repr(out)


def test_rule_banner_registers_its_number_for_the_finish_line(monkeypatch):
    """The finish record carries a rule NAME only; the number comes from here."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rule_banner("2.07", "a_freshly_named_rule")
    assert cs._RULE_NUMBERS["a_freshly_named_rule"] == "2.07"


def test_rule_banner_registers_its_summary_for_the_once_per_rule_trim(monkeypatch):
    """The console trims by the EXACT clause inserted here, never by parsing."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rule_banner("2.08", "a_summarized_rule", summary="do the slow thing")
    assert cs._RULE_SUMMARIES["a_summarized_rule"] == "do the slow thing"
    # A rule without one stays absent, so the trim has nothing to remove.
    rule_banner("2.09", "a_bare_rule")
    assert "a_bare_rule" not in cs._RULE_SUMMARIES


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
            "downscale_scenario_series",
            context,
            summary="downscale the climate",
        )

    out = _emit(
        handler,
        _job_info(1, "downscale_scenario_series", banner("rlz 2 | st 0")),
        _job_info(2, "downscale_scenario_series", banner("rlz 2 | st 2")),
        _job_info(3, "downscale_scenario_series", banner("rlz 2 | st 3")),
    )
    first, second, third = out.splitlines()
    assert first.endswith(
        "RUN  Rule 3.14: downscale_scenario_series "
        "- downscale the climate  [rlz 2 | st 0]"
    ), first
    for line, context in ((second, "[rlz 2 | st 2]"), (third, "[rlz 2 | st 3]")):
        assert line.endswith(f"RUN  Rule 3.14: downscale_scenario_series  {context}"), (
            line
        )
        # The identity survives: one grep on `3.14` still finds every member.
        assert "downscale the climate" not in line, line


def test_console_summary_trim_is_per_rule_not_global():
    """Each rule gets its own first line; one rule's does not consume another's."""
    cs._RULE_SUMMARIES["rule_a"] = "do a"
    cs._RULE_SUMMARIES["rule_b"] = "do b"
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
    """`run_workflows.py` drives five Snakefiles; the registries outlive one run.

    A module-level seen-set would leave the second and later workflows with a
    console on which no summary was ever printed -- so the state that says
    "already shown" belongs to the handler, like `_started`.
    """
    cs._RULE_SUMMARIES["shared_rule"] = "do the thing"
    banner = "Rule 9.03: shared_rule - do the thing"
    first_run = _emit(_console_handler(), _job_info(1, "shared_rule", banner))
    second_run = _emit(_console_handler(), _job_info(1, "shared_rule", banner))
    assert first_run.rstrip().endswith(banner), first_run
    assert second_run.rstrip().endswith(banner), second_run


def test_console_leaves_a_summaryless_rule_alone_on_every_line():
    """An unregistered rule is the fail-open case: print exactly what came in."""
    handler = _console_handler()
    cs._RULE_SUMMARIES.pop("unregistered_rule", None)
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

    assert cs.install_console_style() is True
    assert isinstance(listener.handlers[0], cs._ConsoleHandler)
    assert listener.handlers[1] is logfile

    # Idempotent: a second call (a second Snakefile in one process) is a no-op.
    already = listener.handlers[0]
    assert cs.install_console_style() is True
    assert listener.handlers[0] is already


def test_install_console_style_fails_open(monkeypatch):
    """No console styling is worth ending a run over, so every miss is silent."""
    import types

    import snakemake.logging as snakemake_logging

    for listener in (None, types.SimpleNamespace(handlers=()), object()):
        monkeypatch.setattr(
            snakemake_logging.logger_manager, "queue_listener", listener, raising=False
        )
        assert cs.install_console_style() is False


def test_run_header_shape_matches_run_summary():
    """Ruled title, blank, one aligned column -- opening and closing alike.

    The `run` and `path tokens` group labels went on 2026-09-06: with the
    console handler owning the block, the token rows sit directly above the
    lines that use them and the angle brackets already say what they are.
    """
    out = cs.run_header(
        "wf3 run_stress_test",
        "test_case/test_rapid2",
        "test_case/project_config_rapid.yml",
        experiment="experiment_rapid",
    )
    lines = out.splitlines()
    assert lines[0] == ""
    assert lines[1] == "wf3 run_stress_test"
    assert lines[2] == ""
    assert "Snakemake command" in lines
    assert lines[lines.index("Run settings") :] == [
        "Run settings",
        "experiment  experiment_rapid",
        "",
        "Paths",
        # `<repo>`-marked since t2609162114; the `config` row beside it is not,
        # only because this test passes a CWD-relative path. Snakemake hands a
        # Snakefile an ABSOLUTE `configfiles[0]`, so in a real run both rows
        # carry the marking -- which is the point of routing them the same way.
        "config      test_case/project_config_rapid.yml",
        "<project>   <repo>/test_case/test_rapid2",
    ]


def test_run_header_states_the_declared_folders(declare_folders):
    """Every token that appears in a body line is defined here, in full.

    A folder under the project names <project> as its base; the external data
    root stays absolute. The rows keep DECLARATION order -- matching wants longest-path
    first, and that order in a header is arbitrary to a reader.
    """
    project = _abs("TESTS/gabon")
    declare_folders(
        data=_abs("data/wflow_global/hydromt"),
        model=os.path.join(project, "models", "hydrology", "wflow"),
    )
    lines = cs.run_header("wf1 build_model", project).splitlines()
    rows = [line for line in lines if re.match(r"^\S+ {2,}\S", line)]
    assert [row.split()[0] for row in rows] == ["<project>", "<data>", "<model>"]
    assert rows[1].endswith("data/wflow_global/hydromt")
    assert rows[2].endswith("models/hydrology/wflow")
    # One column now, no group labels: the `<name>` rows announce themselves.
    assert lines[1] == "wf1 build_model"
    assert "" in lines  # blank-line separation, not a wall


def test_run_header_aligns_both_groups_on_one_value_column():
    """One width across the block; a per-group width restarts the column."""
    out = cs.run_header(
        "wf3 run_stress_test", "test_case/test_rapid", experiment="experiment_rapid"
    )
    rows = [line for line in out.splitlines() if re.match(r"^\S+ {2,}\S", line)]
    # Where the VALUE starts: past the key and the gutter.
    columns = {re.match(r"\S+ +", row).end() for row in rows}
    assert len(columns) == 1, rows


def test_run_header_forward_slashes_and_shortens_the_config_path(monkeypatch):
    """The one row that kept OS separators made the block read as two trees."""
    monkeypatch.setattr(log_core, "_REPO_ROOT", os.path.normpath(_abs("repo")))
    config = os.path.join(_abs("repo"), "test_case", "project_config_rapid.yml")
    out = cs.run_header("wf1 build_model", "test_case/test_rapid", config)
    row = next(line for line in out.splitlines() if line.strip().startswith("config"))
    assert row.split() == ["config", "<repo>/test_case/project_config_rapid.yml"]
    assert "\\" not in out


def test_run_header_marks_a_staged_execution_config_with_the_project_token():
    """A generated config under `execution-configs/` reads `<project>/...`.

    Regression: `prepare_workflow` stages the composed config INSIDE the
    project root (`config/_engine/execution-configs/<workflow>/<digest>/
    project_config.yml`), so on a production run -- where the project lives
    outside the repo and matches neither the `<repo>` nor `<site-packages>`
    rewrite -- this row used to print the same absolute prefix as `<project>`
    above it, in full, twice. It must read as project-relative, the same
    convention `<data>`/`<climate>` rows already use, and stay distinguishable
    from a bare relative output path.
    """
    project = os.path.join(_abs("elsewhere"), "gabon-ntoum-v2")
    config = os.path.join(
        project,
        "config",
        "_engine",
        "execution-configs",
        "analyze_projections",
        "91d954f93e17",
        "project_config.yml",
    )
    out = cs.run_header("wf2 analyze_projections", project, config)
    row = next(line for line in out.splitlines() if line.startswith("config"))
    assert row.split(None, 1)[1] == (
        "<project>/config/_engine/execution-configs/"
        "analyze_projections/91d954f93e17/project_config.yml"
    )
    assert "\\" not in row


def test_run_header_forward_slashes_a_config_outside_the_repo():
    """The production case: a config in a project tree, under neither rewrite.

    `<repo>` and `<site-packages>` normalise a config that lives under one of
    them, and `display_root` normalises the `project` row -- so a config
    outside both kept its OS separators and the block printed
    `project C:/a/b` above `config C:\a\b`, which reads as two trees.
    """
    config = os.path.join(_abs("elsewhere"), "gabon", "project_config.yml")
    out = cs.run_header("wf2 analyze_projections", "test_case/test_rapid", config)
    row = next(line for line in out.splitlines() if line.startswith("config"))
    assert "\\" not in row, row
    assert row.endswith("elsewhere/gabon/project_config.yml"), row


def test_a_rule_log_header_defines_every_token_its_rows_use(declare_folders, tmp_path):
    """The console scrolls away; the log is read months later. A `<model>/x.nc`
    row in a file that never says what `<model>` was is worse than the long
    path it replaced -- so the definition travels with the rows."""
    project = tmp_path / "gabon"
    declare_folders(model=project / "models" / "hydrology" / "wflow")
    header = su._log_header_lines(str(project / "logs" / "1.07_build.log"))
    assert "# <model>: <project>/models/hydrology/wflow" in header.splitlines()


def test_run_summary_closes_a_success_in_one_line():
    """The duration, and nothing else.

    The `log` and `benchmarks` rows this used to carry are already among rule
    `all`'s targets, printed just above. What survives is the one fact stated
    nowhere else -- and two readers need it: a workflow run directly has no
    other duration, and `run_workflows.py` prints no success band of its own
    precisely because this line exists.
    """
    out = cs.run_summary(
        "wf3 run_stress_test",
        "test_case/test_rapid",
        "wf3_run_stress_test.log",
        "wf3_benchmarks.md",
        elapsed_seconds=176,
    )
    assert out.splitlines() == ["wf3 run_stress_test completed in 0:02:56"]


def test_warning_tally_counts_across_processes(tmp_path, monkeypatch):
    """The tally is a FILE because the producers are separate processes.

    A rule prints its warnings from a process of its own and the verdict is
    written by the parent, so an in-memory counter can only ever see half the
    run. Two independent appenders standing in for two jobs here.
    """
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / ".wf1.log.tally"))
    su.note_warning("heartbeat")
    su.note_warning("plot")
    assert su.warning_count() == 2


def test_warning_tally_is_none_when_no_run_declared_one(monkeypatch):
    """`None` and `0` are different answers: untracked, versus a clean run."""
    monkeypatch.delenv(su._WARNING_TALLY_ENV, raising=False)
    assert su.warning_count() is None


def test_declare_warning_tally_truncates_so_a_rerun_counts_its_own(
    tmp_path, monkeypatch
):
    # `setenv` FIRST, although `declare_warning_tally` overwrites it: monkeypatch
    # records a key's original state when it is first touched and undoes only
    # that. Letting `declare` be the first writer leaks the variable into every
    # later test in the session -- which is how one suite run ended with
    # `48 warnings` on the verdict that asserts it closes in one line.
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "unused"))
    monkeypatch.chdir(tmp_path)
    su.declare_warning_tally(".", "wf1_build_model.log")
    su.note_warning("plot")
    assert su.warning_count() == 1
    su.declare_warning_tally(".", "wf1_build_model.log")
    assert su.warning_count() == 0
    assert not os.path.exists(su.warning_tally_path(".", "wf1_build_model.log")), (
        "a parse leaves no artifact in the project"
    )


def test_warning_tally_lives_beside_the_merged_log_not_in_the_parts_dir():
    """`merge_logs` prunes `logs/_parts/`, and the verdict reads the tally after."""
    path = su.warning_tally_path("p", "wf1_build_model.log").replace(os.sep, "/")
    assert path == "p/logs/.wf1_build_model.log.tally"
    assert "_parts" not in path


def test_runner_warning_tally_survives_workflow_phase_resets(tmp_path, monkeypatch):
    from blueearth_cst.shared.run_log_core import RUN_WARNING_TALLY_ENV

    runner_path = tmp_path / "runner.tally"
    monkeypatch.setenv(RUN_WARNING_TALLY_ENV, str(runner_path))
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "unused"))
    su.declare_warning_tally(tmp_path, "wf3.log")
    su.note_warning("source")
    su.declare_warning_tally(tmp_path, "wf3.log")
    su.note_warning("generation")
    assert su.warning_count() == 1
    assert runner_path.read_text(encoding="utf-8").splitlines() == [
        "source",
        "generation",
    ]


def test_log_row_counts_a_warning_it_prints_and_not_one_it_suppresses(
    tmp_path, monkeypatch, capsys
):
    """The tally must match what the reader was SHOWN.

    A row dropped by `CST_LOG_LEVEL` never reached the console, so counting it
    would make the verdict report warnings nobody can find.
    """
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "t"))
    su.log_row("disk is nearly full", module="stage", level="WARNING")
    su.log_row("wrote 3 files", module="stage")
    assert su.warning_count() == 1
    monkeypatch.setenv("CST_LOG_LEVEL", "ERROR")
    su.log_row("disk is nearly full", module="stage", level="WARNING")
    assert su.warning_count() == 1, "a suppressed row is not a shown warning"
    capsys.readouterr()


def test_run_summary_verdict_carries_the_warning_count(tmp_path, monkeypatch):
    """A run that warned must not end in a line a clean run could have written."""
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "t"))
    su.note_warning("heartbeat")
    out = cs.run_summary(
        "wf1", "p", "l.log", "b.md", elapsed_seconds=436, warnings=su.warning_count()
    )
    assert out.splitlines()[0] == "wf1 completed in 0:07:16  |  1 warning"
    su.note_warning("plot")
    out = cs.run_summary(
        "wf1", "p", "l.log", "b.md", elapsed_seconds=436, warnings=su.warning_count()
    )
    assert out.splitlines()[0].endswith("|  2 warnings"), out


def test_run_summary_says_nothing_when_a_run_printed_no_warning(tmp_path, monkeypatch):
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "t"))
    su.declare_warning_tally(str(tmp_path), "l.log")
    assert su.warning_count() == 0, (
        "declared and unwritten is a clean run, not an absent one"
    )
    out = cs.run_summary(
        "wf1", "p", "l.log", "b.md", elapsed_seconds=436, warnings=su.warning_count()
    )
    assert out.splitlines() == ["wf1 completed in 0:07:16"]


def test_run_summary_ignores_a_tally_it_was_not_handed(tmp_path, monkeypatch):
    """Ambient state must not reach the verdict.

    `declare_warning_tally` sets a process-global variable at Snakefile PARSE,
    so a DAG contract test or a dry run leaves one set. A `run_summary` that
    read it would report another run's warnings -- which is what a full suite
    did, closing a one-line verdict with `| 48 warnings`.
    """
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "t"))
    su.note_warning("someone else's run")
    out = cs.run_summary("wf1", "p", "l.log", "b.md", elapsed_seconds=436)
    assert out.splitlines() == ["wf1 completed in 0:07:16"]


def test_run_summary_failure_verdict_carries_it_too(tmp_path, monkeypatch):
    """A failed run's warnings are not less interesting than a clean one's."""
    monkeypatch.setenv(su._WARNING_TALLY_ENV, str(tmp_path / "t"))
    su.note_warning("heartbeat")
    out = cs.run_summary(
        "wf1",
        "p",
        "l.log",
        "b.md",
        failed=True,
        elapsed_seconds=249,
        warnings=su.warning_count(),
    )
    head, rule = out.splitlines()[0], out.splitlines()[1]
    assert head == "wf1 FAILED in 0:04:09  |  1 warning"
    # The band under the title is sized on the head it underlines, field and all.
    assert len(rule) == len(head)


def test_run_summary_paints_only_the_failed_verdict(monkeypatch):
    """Red marks the exceptional run; a success verdict stays unpainted.

    Written against the CONSTANT, like the three-tier test: the hue is a knob,
    what must hold is that one verdict is marked and the other is not.
    """
    monkeypatch.setattr(sys, "stderr", _FakeTTY())
    failed = cs.run_summary("wf3", "p", "l.log", "b.md", failed=True)
    ok = cs.run_summary("wf3", "p", "l.log", "b.md")
    # A failure is WHOLLY red: one loud signal, not red competing with bold.
    assert failed.splitlines()[0] == su._ansi("wf3 FAILED", su._ANSI_FAIL)
    assert su._ANSI_FAIL not in ok
    # A success bolds the NAME only, matching the opening title.
    assert ok.splitlines()[0] == su._ansi("wf3", cs._ANSI_TITLE) + " completed"


def test_run_summary_verdict_is_plain_when_stderr_is_not_a_console(monkeypatch):
    """Piped or redirected, the block must carry no escape codes at all."""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    out = cs.run_summary("wf3", "p", "l.log", "b.md", failed=True)
    assert "\033" not in out, repr(out)


def test_heartbeat_redraws_a_routine_status_frame_without_an_all_clear():
    """Silence is factual status, not a warning, and completion clears it."""
    stream = _TTYStringIO()
    hb = su._Heartbeat("2.05_merge", stream, interval=1.0)
    hb._start = hb._last = hb._seen = 0.0
    for t in (1.0, 2.0, 3.0, 4.0):
        hb._tick(t)
    hb._clear_status()  # what stop() does on a clean finish
    out = _unreset(stream.getvalue())
    assert out.count("elapsed") >= 2
    assert "done in" not in out
    assert f"\033[{su._ANSI_FAIL}m" not in out
    assert "\r" in stream.getvalue()


def test_heartbeat_paints_the_failure_verdict(monkeypatch):
    """There is no DONE line for a job that raised -- this is the only report."""
    stream = _TTYStringIO()
    hb = su._Heartbeat("3.15_run_wflow", stream, interval=10.0).start()
    hb.stop(failed=True)
    assert _unreset(stream.getvalue()).startswith(f"\033[{su._ANSI_FAIL}m")


def test_run_summary_failure_keeps_the_block_a_success_no_longer_needs():
    """On failure rule `all` never ran, so this is the only record there is.

    The merged log does not exist yet -- rule W.17 is a normal rule and does
    not run when an upstream job fails -- so the log-parts directory is named
    nowhere else. The note beneath it names no artifact, so it is not a row.
    """
    out = cs.run_summary(
        "wf3 run_stress_test",
        "test_case/test_rapid",
        "wf3.log",
        "wf3.md",
        failed=True,
        log_parts_dir="test_case/test_rapid/logs/_parts/experiment_rapid",
    )
    assert out.splitlines() == [
        "wf3 run_stress_test FAILED",
        "--------------------------",
        "",
        "log parts  test_case/test_rapid/logs/_parts/experiment_rapid/",
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


def test_console_shortens_an_error_block_under_a_relative_project_dir(
    declare_folders, monkeypatch
):
    """Snakemake's error block prints the RELATIVE spelling the Snakefile
    declared, which an absolute token alone cannot match (t2609172119)."""
    declare_folders(model="test_case/test_rapid/models/hydrology/wflow")
    monkeypatch.setenv(su._PROJECT_ROOT_ENV, "test_case/test_rapid")
    handler = _console_handler()
    out = _emit(
        handler,
        _console_record(
            "    output: test_case/test_rapid/models/hydrology/wflow/run_default/output.csv",
            level=_logging.ERROR,
        ),
    )
    assert "<model>/run_default/output.csv" in out, out


def test_run_header_omits_rows_a_workflow_does_not_have():
    """WF1 and WF2 pass no experiment; the block shrinks rather than showing a blank."""
    out = cs.run_header("wf1 build_model", "test_case/test_rapid")
    lines = out.splitlines()
    assert lines[1] == "wf1 build_model"
    assert lines[-1] == "<project>  <repo>/test_case/test_rapid"
    assert not any("experiment" in line for line in lines)


# --- console body tier for rule output (_Tee / _paint_body) ------------------


class _TTYWriter(io.StringIO):
    def isatty(self):
        return True


def test_log_row_dims_its_prefix_when_written_straight_to_a_terminal(monkeypatch):
    """A `run:` rule's row reaches the terminal with no tee to paint it."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    console = _TTYWriter()
    monkeypatch.setattr(su.sys, "stdout", console)
    log_row("default: calculations complete", module="indicators")
    out = console.getvalue()
    assert re.fullmatch(
        rf"\033\[{su._ANSI_DIM}m\d\d:\d\d:\d\d - indicators - {re.escape(su._ANSI_RESET)}"
        r"default: calculations complete\n",
        out,
    ), repr(out)


def test_log_row_stays_plain_off_a_terminal(monkeypatch):
    console = io.StringIO()  # isatty() -> False, as a tee reports
    monkeypatch.setattr(su.sys, "stdout", console)
    log_row("default: calculations complete", module="indicators")
    assert "\033[" not in console.getvalue()


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
    out = su._paint_body("first\nWARNING: second\n", True)
    # Neither line is in our row grammar, so the body one is left alone
    # entirely and only the warning is wrapped -- and its reset lands before
    # the newline, or the colour would carry across a terminal reflow.
    assert out == f"first\n\033[{su._ANSI_ALERT}mWARNING: second\033[0m\n", repr(out)


def test_paint_body_is_a_no_op_without_colour():
    assert su._paint_body("plain text", False) == "plain text"


def test_tee_paints_the_console_and_never_the_log_file(tmp_path):
    """The split this rests on: one call, two sinks, one of them a file read
    months later by merge_logs -- where an escape code is corruption."""
    log = tmp_path / "rule.log"
    console = _TTYWriter()
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(console, handle)
        tee.write("13:42:17 - stats - MPI-ESM1-2-HR ssp585 deriving\n")
        tee.close()
    # The reset lands BEFORE the newline: a colour spanning the break would
    # carry across a terminal reflow.
    assert console.getvalue() == (
        _LINE_RESET
        + su._ansi("13:42:17 - stats - ", su._ANSI_DIM)
        + "MPI-ESM1-2-HR ssp585 deriving\n"
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


# --- severity colouring -------------------------------------------------------
# A line's own text outranks the tier its emitter assumed, because most of what
# scrolls past arrives as undifferentiated body text from a foreign process.


@pytest.mark.parametrize(
    "line",
    [
        "13:42:17 - forcing - WARNING: precip units look wrong",
        "WARN reservoir table is empty",
        "Warning message:",  # R
        "FutureWarning: 'M' is deprecated",  # Python warnings
    ],
)
def test_a_warning_line_is_painted_orange_whatever_emitted_it(line):
    out = su._paint_body(line, True)
    assert out == f"\033[{su._ANSI_ALERT}m{line}\033[0m", repr(out)


def test_structured_warning_uses_colour_without_a_console_level(tmp_path):
    row = "13:42:17 - spatial - WARNING - Output location coordinates adjusted:\n"
    console = _TTYWriter()
    log = tmp_path / "rule.log"
    with log.open("w", encoding="utf-8") as handle:
        tee = su._Tee(console, handle)
        tee.write(row)
        tee.close()
    assert console.getvalue().endswith(
        f"\033[{su._ANSI_ALERT}m13:42:17 - spatial - "
        "Output location coordinates adjusted:\033[0m\n"
    )
    assert "WARNING - " not in console.getvalue()
    assert log.read_text(encoding="utf-8") == row
    assert su._paint_body(row, False) == row


@pytest.mark.parametrize(
    "line",
    [
        "ERROR could not open the store",
        "13:42:17 - fetch - FAILURE after 3 attempts",
        "08:03:16 - wflow - FAILED [2/3] rlz_1_st_2  boom",
        "CRITICAL julia exited 1",
        "Traceback (most recent call last):",
        "Error in eval(expr) : object 'x' not found",  # R
    ],
)
def test_a_failure_line_is_painted_red_whatever_emitted_it(line):
    out = su._paint_body(line, True)
    assert out == f"\033[{su._ANSI_FAIL}m{line}\033[0m", repr(out)


def test_failure_outranks_warning_on_one_line():
    """A line carrying both reads as the worse of the two."""
    line = "ERROR while handling WARNING backlog"
    assert su._paint_body(line, True).startswith(f"\033[{su._ANSI_FAIL}m")


@pytest.mark.parametrize(
    "line",
    [
        "13:42:17 - plot - wrote error_indicators.csv",
        "n_errors=0 n_warnings=0",
        "running compute_errors on 4 members",
        "13:42:17 - stats - MPI-ESM1-2-HR ssp585 deriving",
    ],
)
def test_ordinary_lines_keep_the_body_tier(line):
    """Case-sensitive and word-bounded is what keeps this off ordinary text.

    The body tier emits no SGR, so an ordinary line that is not in our row
    grammar comes back untouched, and one that is keeps its message untouched
    with only the stamp and module column dimmed.
    """
    out = su._paint_body(line, True)
    match = su._ROW_PREFIX_RE.match(line)
    if match:
        assert out == su._ansi(match.group(1), su._ANSI_DIM) + match.group(2), repr(out)
    else:
        assert out == line, repr(out)


def test_severity_paints_only_the_offending_line_of_a_chunk():
    out = su._paint_body("routine row\nWARNING: something\n", True)
    assert out == (f"routine row\n\033[{su._ANSI_ALERT}mWARNING: something\033[0m\n"), (
        repr(out)
    )


def test_severity_never_reaches_the_log_file(tmp_path):
    """The whole constraint: merge_logs reads these files months later, where
    an escape code is corruption rather than colour."""
    log = tmp_path / "rule.log"
    console = _TTYWriter()
    with open(log, "w", encoding="utf-8") as handle:
        tee = su._Tee(console, handle)
        tee.write("13:42:17 - fetch - WARNING: retrying\n")
        tee.write("13:42:18 - fetch - ERROR: gave up\n")
        tee.close()
    assert f"\033[{su._ANSI_ALERT}m" in console.getvalue()
    assert f"\033[{su._ANSI_FAIL}m" in console.getvalue()
    assert log.read_text(encoding="utf-8") == (
        "13:42:17 - fetch - WARNING: retrying\n13:42:18 - fetch - ERROR: gave up\n"
    )


# --- rule identity (t2608071213) ---------------------------------------------


def _registry():
    return cs.RuleRegistry("PROJ/logs/_parts", "PROJ/benchmarks/_parts")


def test_a_logged_rule_registers_its_label_in_declaration_order():
    """`LOG_RULES` IS the merge order, so the registry must build it in order."""
    reg = _registry()
    reg.logged("0.02", "delineate_region")
    reg.logged("0.03", "delineate_subbasins_and_rivers")
    assert reg.log_rules == [
        "0.02_delineate_region",
        "0.03_delineate_subbasins_and_rivers",
    ]


def test_log_rules_is_the_live_list_a_snakefile_aliases():
    """A Snakefile does `LOG_RULES = RULES.log_rules` BEFORE the last rule.

    It must stay the same object, or a conditionally-registered rule would be
    missing from the registered log labels.
    """
    reg = _registry()
    alias = reg.log_rules
    reg.logged("0.06", "compare_climate_datasets")
    assert alias == ["0.06_compare_climate_datasets"]
    assert alias is reg.log_rules


def test_a_banner_only_rule_is_not_in_log_rules():
    """Banner-only rules write no part; nothing must merge one."""
    reg = _registry()
    reg.banner_only("0.01", "snapshot_config")
    assert reg.log_rules == []


def test_asking_a_banner_only_rule_for_a_log_path_raises():
    """The [R10-8] defect, refused at PARSE time instead of by a test.

    A part written under a label `LOG_RULES` does not carry is merged by nothing
    and cleaned up by nothing, and neither failure raises at run time.
    """
    identity = _registry().banner_only("0.01", "snapshot_config")
    with pytest.raises(ValueError, match="banner-only"):
        identity.log()


def test_a_flat_rule_spells_its_paths_from_one_label():
    reg = _registry()
    r = reg.logged("0.02", "delineate_region")
    assert r.label == "0.02_delineate_region"
    assert r.job_name() == "delineate_region"
    assert r.log() == "PROJ/logs/_parts/0.02_delineate_region.log"
    assert r.benchmark() == "PROJ/benchmarks/_parts/0.02_delineate_region.tsv"


def test_a_fan_out_rule_suffixes_its_NAME_and_nests_its_PARTS():
    """The divergence the label derivation exists for.

    A parse-time fan-out names each generated rule for its part, while every
    part lands in ONE directory named for the singular label -- which is what
    keeps `LOG_RULES` singular and lets `merge_logs` list the directory.
    """
    r = _registry().logged("0.04", "extract_climate_datasets")
    assert r.label == "0.04_extract_climate_datasets"
    assert r.job_name("chirps") == "extract_climate_datasets_chirps"
    assert (
        r.log("chirps") == "PROJ/logs/_parts/0.04_extract_climate_datasets/chirps.log"
    )
    assert r.benchmark("era5") == (
        "PROJ/benchmarks/_parts/0.04_extract_climate_datasets/era5.tsv"
    )


def test_the_banner_carries_the_rules_summary_and_the_fanned_name():
    reg = _registry()
    plain = reg.logged("0.02", "delineate_region")
    assert plain.banner() == cs.rule_banner("0.02", "delineate_region")
    described = reg.logged("0.04b", "derive_plot_scales", summary="one scale")
    assert described.banner() == cs.rule_banner(
        "0.04b", "derive_plot_scales", summary="one scale"
    )
    fanned = reg.logged("0.05", "plot_climate_datasets")
    assert fanned.banner("chirps") == cs.rule_banner(
        "0.05", "plot_climate_datasets_chirps"
    )


def test_benchmark_is_available_to_a_banner_only_rule():
    """Logging and benchmarking declarations remain independent."""
    assert _registry().banner_only("0.10", "x").benchmark() == (
        "PROJ/benchmarks/_parts/0.10_x.tsv"
    )


def test_path_legend_makes_the_project_base_explicit(declare_folders, tmp_path):
    project = tmp_path / "project"
    external = tmp_path / "data"
    declare_folders(
        data=external,
        climate=project / "data" / "climate" / "historical",
    )
    rows = dict(cs.run_meta_rows(project, project / "project_config.yml"))
    assert rows["<project>"] == cs.display_root(project)
    assert "config" in rows and "<config>" not in rows
    assert rows["<data>"] == external.as_posix()
    assert rows["<climate>"] == "<project>/data/climate/historical"
    header = su._log_header_lines(str(project / "logs" / "extract.log"))
    assert f"# <project>: {project.as_posix()}" in header
    assert "# <climate>: <project>/data/climate/historical" in header


def test_console_target_jobs_are_not_counted_as_work(monkeypatch):
    """`all` finished as `[15/15]`; the counter now covers work jobs only.

    A mid-run target (WF4's `simulations_only`) shifts every later counter down
    by one, so the last work job still ends at N/N.
    """
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {"work": "1.01"})
    # Left by another parsed Snakefile; not this workflow's rule, so not counted.
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"derive_system_indicators": 2})
    stats = "Job stats:\njob  count\n----  ---\nall  1\nsimulations_only  1\nwork  2\ntotal  4\n"
    out = _emit(
        _console_handler(),
        _console_record(stats, event="run_info"),
        _job_info(1, "work", "Rule 1.01: work"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=4),
        _job_info(2, "simulations_only", "Target: simulations_only"),
        _console_record(event="job_finished", job_id=2),
        _console_record(event="progress", done=2, total=4),
        _job_info(3, "work", "Rule 1.01: work"),
        _console_record(event="job_finished", job_id=3),
        _console_record(event="progress", done=3, total=4),
        _job_info(4, "all", "Target: all - confirm requested outputs"),
        _console_record(event="job_finished", job_id=4),
        _console_record(event="progress", done=4, total=4),
    )
    done = [line for line in out.splitlines() if "- DONE " in line]
    assert done[0].endswith("[1/2]"), done
    assert "[job" not in done[1], done
    assert done[2].endswith("[2/2]"), done
    assert done[3].endswith("- DONE all"), done


def test_console_counter_includes_jobs_planned_after_a_checkpoint(monkeypatch):
    """WF4 read `[112/112]` then `[113/114]` once 4.08 was added.

    The plan already lists a checkpoint-gated rule's declared jobs, so the
    denominator starts there instead of growing mid-run.
    """
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {"plan": "4.07", "derive": "4.08"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": 1, "derive": 1})
    stats = "Job stats:\njob  count\n----  ---\nall  1\nplan  1\ntotal  2\n"
    out = _emit(
        _console_handler(),
        _console_record(stats, event="run_info"),
        _job_info(1, "plan", "Rule 4.07: plan"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=2),
        _job_info(2, "derive", "Rule 4.08: derive"),
        _console_record(event="job_finished", job_id=2),
        _console_record(event="progress", done=2, total=3),
    )
    done = [line for line in out.splitlines() if "- DONE " in line]
    assert done[0].endswith("[1/2]"), done
    assert done[1].endswith("[2/2]"), done


def test_console_counter_plans_only_the_checkpoint_jobs_this_run_lists(monkeypatch):
    """Adding a set beside a current one ended WF4's counter at `[2/3]`.

    Only the new set's 4.07 plan runs, so 4.08 publishes one set, not two.
    """
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {"plan": "4.07", "derive": "4.08"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": 2, "derive": 2})
    monkeypatch.setattr(cs, "_PLANNED_FROM", {"derive": "plan"})
    counts = {"all": 1, "plan": 1}
    assert [row[2] for row in cs._plan_rows(counts)] == [1, 1]
    assert [row[2] for row in cs._plan_rows({"all": 1})] == [2, 2]
    stats = "Job stats:\njob  count\n----  ---\nall  1\nplan  1\ntotal  2\n"
    out = _emit(
        _console_handler(),
        _console_record(stats, event="run_info"),
        _job_info(1, "plan", "Rule 4.07: plan"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=2),
        _job_info(2, "derive", "Rule 4.08: derive"),
        _console_record(event="job_finished", job_id=2),
        _console_record(event="progress", done=2, total=3),
    )
    done = [line for line in out.splitlines() if "- DONE " in line]
    assert done[0].endswith("[1/2]"), done
    assert done[1].endswith("[2/2]"), done


def test_console_counter_and_plan_skip_checkpoint_rules_a_target_excludes(monkeypatch):
    """`--target simulations_only` never reaches 4.07/4.08: no planned jobs."""
    monkeypatch.setattr(
        cs, "_RULE_NUMBERS", {"simulate": "4.05", "plan": "4.07", "derive": "4.08"}
    )
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": 1, "derive": 1})
    counts = {"simulations_only": 1, "simulate": 1}
    assert [row[2] for row in cs._plan_rows(counts)] == [1, 0, 0]
    stats = "Job stats:\njob  count\n----  ---\nsimulations_only  1\nsimulate  1\ntotal  2\n"
    out = _emit(
        _console_handler(),
        _console_record(stats, event="run_info"),
        _job_info(1, "simulate", "Rule 4.05: simulate"),
        _console_record(event="job_finished", job_id=1),
        _console_record(event="progress", done=1, total=2),
    )
    done = [line for line in out.splitlines() if "- DONE " in line]
    assert done[0].endswith("[1/1]"), done
