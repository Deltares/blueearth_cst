# -*- coding: utf-8 -*-
"""Unit tests for ``blueearth_cst/model/add_climate_forcing.py`` (rules 1.07+1.08).

The module had NO direct coverage until 2026-08-11
(`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §4). What existed pinned
only its RULE WIRING — ``test_model_rebuild_cascade`` proves re-firing the build
reschedules it, ``test_model_root_ordering`` proves it runs after its last
writer — and neither reads the command it issues.

That command is the whole point of the module. `[R10-1]` merged a `steps:`-YAML
writer and a `hydromt update` `shell:` body into one `script:`, and the merge's
correctness claim is that the CLI invocation stayed **byte-identical** to what
the shell rule issued. Nothing checked that claim. Three properties follow:

* the argv is exactly the old shell line, in order, ending in ``-vv``;
* several catalogs render as REPEATED ``-d`` flags — the shell rule interpolated
  ``-d "{DATA_SOURCES}"`` once and would have handed hydromt a Python list repr;
* the store branch appends a GENERATED catalog and passes the file stem as
  ``store_source``, because ``prepare_clim_data_catalog`` keys entries on it.

``_run_streaming`` is tested against a real child process: the reason it exists
is that ``tee_to_log`` redirects Python-level ``sys.stdout`` and cannot see a
child's file descriptors, so a bare ``subprocess.run`` would leave the rule's log
part empty. A mock cannot fail that way, so it would not be a test of it.

The recipe builder is NOT retested here — it lives in
``shared/setup_time_horizon.py`` with ``tests/test_setup_time_horizon.py`` on it.
What is asserted is that this module hands it the right arguments.
"""

from __future__ import annotations

import sys

import pytest

from blueearth_cst.model import add_climate_forcing as acf
from blueearth_cst.model.add_climate_forcing import (
    _as_list,
    _catalog_flags,
    _run_streaming,
    add_climate_forcing,
)

# ---------------------------------------------------------------------------
# catalog rendering
# ---------------------------------------------------------------------------


def test_a_single_catalog_becomes_a_one_element_list(tmp_path):
    assert _as_list("config/catalogs/deltares_data.yml") == [
        "config/catalogs/deltares_data.yml"
    ]
    assert _as_list(tmp_path / "a.yml") == [str(tmp_path / "a.yml")]


def test_several_catalogs_survive_as_several(tmp_path):
    assert _as_list(["a.yml", tmp_path / "b.yml"]) == ["a.yml", str(tmp_path / "b.yml")]
    assert _as_list(("a.yml", "b.yml")) == ["a.yml", "b.yml"]


def test_one_catalog_renders_as_one_d_flag():
    assert _catalog_flags("deltares_data.yml") == ["-d", "deltares_data.yml"]


def test_several_catalogs_render_as_REPEATED_d_flags():
    """The defect the merge fixed, stated directly.

    ``-d "{DATA_SOURCES}"`` interpolated a list as ``['a.yml', 'b.yml']`` — one
    flag carrying a Python repr, which hydromt would reject obscurely or read as
    a filename. Repeating the flag is what hydromt actually accepts.
    """
    assert _catalog_flags(["a.yml", "b.yml", "c.yml"]) == [
        "-d",
        "a.yml",
        "-d",
        "b.yml",
        "-d",
        "c.yml",
    ]


def test_a_path_object_reaches_the_flag_as_a_string(tmp_path):
    """argv entries must be strings; a ``Path`` in the list is a caller reality."""
    flags = _catalog_flags([tmp_path / "a.yml"])
    assert flags == ["-d", str(tmp_path / "a.yml")]
    assert all(isinstance(item, str) for item in flags)


# ---------------------------------------------------------------------------
# _run_streaming — against a real child process
# ---------------------------------------------------------------------------


def test_child_output_is_reprinted_so_the_tee_can_capture_it(capsys):
    """``tee_to_log`` redirects ``sys.stdout``, not the child's file descriptor.

    Inheriting stdout would put hydromt's ``-vv`` output on the console and leave
    the rule's log part empty — silently losing what ``run_logged`` used to
    capture. Re-printing is what puts it back inside the tee.
    """
    _run_streaming([sys.executable, "-c", "print('first line'); print('second line')"])

    out = capsys.readouterr().out
    assert "first line" in out
    assert "second line" in out


def test_the_command_itself_is_echoed_before_it_runs(capsys):
    """So a log part names the invocation, not just its output."""
    _run_streaming([sys.executable, "-c", "pass"])

    assert capsys.readouterr().out.splitlines()[0].startswith(f"$ {sys.executable} -c")


def test_stderr_is_folded_into_the_same_stream(capsys):
    """``stderr=STDOUT``: a hydromt warning belongs in the log part, in order."""
    _run_streaming(
        [sys.executable, "-c", "import sys; sys.stderr.write('a warning\\n')"]
    )

    assert "a warning" in capsys.readouterr().out


def test_a_nonzero_exit_raises_and_names_the_code_and_the_command():
    """A failed ``hydromt update`` must fail the RULE.

    Popen does not raise on its own, so without this check the rule would
    complete green having written no forcing.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _run_streaming([sys.executable, "-c", "raise SystemExit(3)"])

    assert "exit code 3" in str(excinfo.value)
    assert sys.executable in str(excinfo.value)


def test_a_zero_exit_returns_quietly():
    assert _run_streaming([sys.executable, "-c", "pass"]) is None


# ---------------------------------------------------------------------------
# add_climate_forcing — the orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    """Capture the three collaborators without running hydromt.

    Each is a module-level binding imported by name, so patching the module's
    namespace is what the call site actually resolves.
    """
    calls: dict[str, list] = {"catalog": [], "recipe": [], "command": []}

    monkeypatch.setattr(
        acf, "prepare_clim_data_catalog", lambda **kw: calls["catalog"].append(kw)
    )
    monkeypatch.setattr(
        acf,
        "prep_hydromt_update_forcing_config",
        lambda **kw: calls["recipe"].append(kw),
    )
    monkeypatch.setattr(acf, "_run_streaming", lambda cmd: calls["command"].append(cmd))
    return calls


def _invoke(tmp_path, spy, **overrides):
    kwargs = dict(
        starttime="2000-01-01",
        endtime="2010-12-31",
        clim_source="era5",
        basin_dir=tmp_path / "hydrology_model",
        data_catalog="config/catalogs/deltares_data.yml",
        forcing_yml=tmp_path / "forcing.yml",
    )
    kwargs.update(overrides)
    add_climate_forcing(**kwargs)
    return spy


def test_the_argv_is_the_shell_rule_it_replaced(tmp_path, spy):
    """The merge's correctness claim: byte-identical to rule 1.08's command.

    Order matters — hydromt reads ``update <MODEL> <ROOT>`` positionally, and
    ``-vv`` is what makes the log part non-empty.
    """
    calls = _invoke(tmp_path, spy)

    (command,) = calls["command"]
    assert command == [
        "hydromt",
        "update",
        "wflow_sbm",
        str(tmp_path / "hydrology_model"),
        "-i",
        str(tmp_path / "forcing.yml"),
        "-d",
        "config/catalogs/deltares_data.yml",
        "-vv",
    ]


def test_the_recipe_is_written_before_the_command_that_reads_it(tmp_path, spy):
    """``-i <forcing_yml>`` is only meaningful if the recipe already exists."""
    calls = _invoke(tmp_path, spy)

    (recipe,) = calls["recipe"]
    assert recipe["fn_yml"] == tmp_path / "forcing.yml"
    assert recipe["starttime"] == "2000-01-01"
    assert recipe["endtime"] == "2010-12-31"
    assert recipe["precip_source"] == "era5"
    assert recipe["wflow_root"] == tmp_path / "hydrology_model"


def test_without_a_store_no_catalog_is_generated_and_no_store_source_is_named(
    tmp_path, spy
):
    """The catalog-only path: hydromt reads the global dataset directly."""
    calls = _invoke(tmp_path, spy)

    assert calls["catalog"] == []
    assert calls["recipe"][0]["store_source"] is None


def test_a_store_is_handed_over_as_a_generated_catalog_entry(tmp_path, spy):
    """Why the store is not passed as a path.

    Handing it over as a catalog entry generated FROM the real source's entry is
    what makes units and renames inherited rather than hand-written.
    """
    calls = _invoke(
        tmp_path,
        spy,
        climate_nc=tmp_path / "era5_20000101_20101231.nc",
        store_catalog=tmp_path / "store.yml",
    )

    (catalog,) = calls["catalog"]
    assert catalog["fns"] == [tmp_path / "era5_20000101_20101231.nc"]
    assert catalog["source_like"] == "era5"
    assert catalog["data_libs_like"] == "config/catalogs/deltares_data.yml"
    assert catalog["fn_out"] == tmp_path / "store.yml"


def test_the_store_source_is_the_file_STEM(tmp_path, spy):
    """``prepare_clim_data_catalog`` keys each entry on the stem.

    Passing the path, or the name with its suffix, would name an entry that does
    not exist and hydromt would fall back to the global source — a second full
    pass over the same data, silently.
    """
    calls = _invoke(
        tmp_path,
        spy,
        climate_nc=tmp_path / "era5_20000101_20101231.nc",
        store_catalog=tmp_path / "store.yml",
    )

    assert calls["recipe"][0]["store_source"] == "era5_20000101_20101231"


def test_the_generated_catalog_is_APPENDED_after_the_real_one(tmp_path, spy):
    """Order is the inheritance direction.

    The store entry is derived from the real source's entry, so the real catalog
    must be readable when hydromt resolves it.
    """
    calls = _invoke(
        tmp_path,
        spy,
        climate_nc=tmp_path / "era5_20000101_20101231.nc",
        store_catalog=tmp_path / "store.yml",
    )

    (command,) = calls["command"]
    assert command[command.index("-d") :] == [
        "-d",
        "config/catalogs/deltares_data.yml",
        "-d",
        str(tmp_path / "store.yml"),
        "-vv",
    ]


def test_several_catalogs_plus_a_store_all_reach_the_command(tmp_path, spy):
    """The list path and the store path compose rather than shadowing each other."""
    calls = _invoke(
        tmp_path,
        spy,
        data_catalog=["a.yml", "b.yml"],
        climate_nc=tmp_path / "era5_x.nc",
        store_catalog=tmp_path / "store.yml",
    )

    (command,) = calls["command"]
    assert command.count("-d") == 3
    assert command[command.index("-d") :] == [
        "-d",
        "a.yml",
        "-d",
        "b.yml",
        "-d",
        str(tmp_path / "store.yml"),
        "-vv",
    ]


def test_a_failing_update_propagates_rather_than_completing_green(
    tmp_path, monkeypatch
):
    """End to end through the real ``_run_streaming``: the rule must fail."""
    monkeypatch.setattr(acf, "prep_hydromt_update_forcing_config", lambda **kw: None)
    monkeypatch.setattr(
        acf,
        "_run_streaming",
        lambda cmd: _run_streaming([sys.executable, "-c", "raise SystemExit(1)"]),
    )

    with pytest.raises(RuntimeError, match="exit code 1"):
        add_climate_forcing(
            starttime="2000-01-01",
            endtime="2010-12-31",
            clim_source="era5",
            basin_dir=tmp_path,
            data_catalog="a.yml",
            forcing_yml=tmp_path / "forcing.yml",
        )


# The module's OTHER contract — that a `script:` target carries no
# `from __future__ import annotations`, because Snakemake's preamble displaces it
# and the rule then dies at RUN time — is a repo-wide rule rather than a property
# of this module, and one this module's header comment is merely the clearest
# statement of. It is enforced over every `script:` target at
# `tests/test_script_module_importability.py::test_no_script_target_carries_a_future_import`.
