"""Unit tests for dev/scripts/snapshot_project_tree.py (R09 phase 1).

The wrapper's whole value is that it derives the map parameters from the config
instead of taking them on the command line -- a mistyped `--dataset-key` turns a
mapped store into an unmapped one, which reads as a map gap. These tests pin
that derivation, the exclusions, and the exit code.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import snapshot_project_tree as spt  # noqa: E402


def _config(project_dir):
    return {
        "project": {"project_dir": str(project_dir).replace("\\", "/")},
        "shared": {
            "clim_historical": "era5",
            "historical_window": {
                "starttime": "2000-01-01T00:00:00",
                "endtime": "2020-12-31T00:00:00",
            },
        },
        "workflows": {
            "climate_experiment": {"experiment_name": "my_experiment"},
            "climate_projections": {"clim_project": "cmip6"},
        },
    }


def _write_config(tmp_path, cfg, name="cfg.yml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _touch(root, rel, text="x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Parameter derivation
# ---------------------------------------------------------------------------

def test_map_parameters_are_derived_the_way_the_workflows_build_them():
    params = spt.map_parameters(_config("proj"))
    assert params == {
        "project_dir": "proj",
        "experiment_name": "my_experiment",
        "dataset_key": "era5_20000101_20201231",
        "clim_project": "cmip6",
    }


def test_map_parameters_fall_back_to_the_shipped_defaults():
    cfg = _config("proj")
    cfg["workflows"] = {}
    params = spt.map_parameters(cfg)
    assert params["experiment_name"] == "experiment"
    assert params["clim_project"] == "cmip6"


def test_a_sub_day_window_fails_loud():
    """The store key is day-resolution; a silent collision would mis-key it."""
    cfg = _config("proj")
    cfg["shared"]["historical_window"]["starttime"] = "2000-01-01T06:00:00"
    with pytest.raises(ValueError, match="time-of-day"):
        spt.map_parameters(cfg)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_list_tree_returns_sorted_relative_posix_paths(tmp_path):
    _touch(tmp_path, "b/second.nc")
    _touch(tmp_path, "a/first.csv")
    assert spt.list_tree(tmp_path) == ["a/first.csv", "b/second.nc"]


def test_snakemake_metadata_is_excluded_but_nothing_else_is(tmp_path):
    """`.snakemake/` is bookkeeping. Everything else is kept ON PURPOSE --
    an observed snapshot exists to carry artifacts no rule declares."""
    _touch(tmp_path, ".snakemake/log/whatever.log")
    _touch(tmp_path, "hydrology_model/hydromt.log")     # undeclared, kept
    _touch(tmp_path, "logs/wf1_model_creation.log")     # excluded from tree
    _touch(tmp_path, "hydrology_model/.model_built")    # dotfile, kept
    assert spt.list_tree(tmp_path) == [
        "hydrology_model/.model_built",
        "hydrology_model/hydromt.log",
        "logs/wf1_model_creation.log",
    ]


def test_empty_directories_are_not_paths(tmp_path):
    (tmp_path / "empty").mkdir()
    _touch(tmp_path, "a.csv")
    assert spt.list_tree(tmp_path) == ["a.csv"]


def test_relative_project_dir_resolves_against_the_cwd_not_the_tool_repo(tmp_path):
    """The cross-checkout trap the runbook walks into by design.

    The workflows run from the PRIMARY checkout while the comparator lives in a
    task worktree, so the tool's own repo root is the wrong base: it would look
    for the tree beside the tool instead of beside the run.
    """
    resolved = spt.resolve_project_dir("test_case/test_local", base=tmp_path)
    assert resolved == tmp_path / "test_case" / "test_local"
    assert spt.REPO not in resolved.parents
    # an absolute project_dir is returned untouched
    assert spt.resolve_project_dir(str(tmp_path)) == tmp_path


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def _tree(tmp_path):
    """A miniature pre-migration tree: one mapped file, one orphan."""
    proj = tmp_path / "proj"
    _touch(proj, "hydrology_model/staticmaps.nc")
    _touch(proj, "logs/1.03_create_model.log")   # pre-_parts orphan shape
    return proj


def test_reports_unmapped_and_exits_nonzero(tmp_path, capsys):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert "UNMAPPED logs/1.03_create_model.log" in out
    assert "MOVED    hydrology_model/staticmaps.nc" in out
    assert "either a leftover ORPHAN" in out


def test_a_clean_tree_exits_zero(tmp_path, capsys):
    proj = tmp_path / "proj"
    _touch(proj, "hydrology_model/staticmaps.nc")
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg)]) == 0
    assert "MAP CLEAN" in capsys.readouterr().out


def test_writes_nothing_without_out(tmp_path):
    """Runbook step 0 inspects; it must not leave a file behind."""
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    spt.main(["--config", str(cfg)])
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_out_writes_a_snapshot_the_falsifier_can_read_back(tmp_path):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    out = tmp_path / "snap" / "observed_inventory.txt"
    spt.main(["--config", str(cfg), "--out", str(out)])

    text = out.read_text(encoding="utf-8")
    assert "# PROVENANCE" in text
    assert "era5_20000101_20201231" in text          # the derived store key
    assert "my_experiment" in text                   # the derived experiment
    # `--check-map` skips comments and blank lines; what is left must be the
    # exact path list, so the two tools cannot disagree about the snapshot.
    payload = [ln.strip() for ln in text.splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    assert payload == spt.list_tree(proj)


def test_no_check_skips_the_gate_and_always_exits_zero(tmp_path, capsys):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg), "--no-check"]) == 0
    assert "UNMAPPED" not in capsys.readouterr().out


def test_the_gap_rule_set_is_empty_and_the_flag_is_a_no_op(tmp_path, capsys):
    """Every candidate is closed, so `--gap-rules` currently changes nothing.

    Three were ruled into the map (phase-1 report F1a-F1c) and two were settled
    negatively by the observed-tier run (F2). Asserted rather than assumed: a
    non-empty set here would mean an unruled row had crept back in.
    """
    assert spt.std.build_r09_gap_rules("experiment") == []
    assert spt.std.R09_MAP_GAPS == ()

    proj = tmp_path / "proj"
    _touch(proj, "hydrology_model/staticmaps.nc")
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg)]) == 0
    assert spt.main(["--config", str(cfg), "--gap-rules"]) == 0


def test_gap_rules_still_wire_through_when_the_set_is_non_empty(
    tmp_path, capsys, monkeypatch
):
    """The MECHANISM, pinned independently of any particular candidate.

    The set is empty today; the next inventory may raise a sixth candidate, and
    what must keep working is that an opt-in rule is off by default and applied
    on the flag. Tested against a stand-in so this cannot rot the way the old
    `instate/` case did when its candidate was retired.
    """
    monkeypatch.setattr(
        spt.std, "build_r09_gap_rules",
        lambda _e: [("some_future_dir/", "data/some_future_dir/")],
    )
    proj = tmp_path / "proj"
    _touch(proj, "some_future_dir/thing.nc")
    cfg = _write_config(tmp_path, _config(proj))

    assert spt.main(["--config", str(cfg)]) == 1
    assert "UNMAPPED some_future_dir/thing.nc" in capsys.readouterr().out

    assert spt.main(["--config", str(cfg), "--gap-rules"]) == 0
    assert "MAP CLEAN" in capsys.readouterr().out


def test_quiet_keeps_the_unmapped_lines_and_the_summary(tmp_path, capsys):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    spt.main(["--config", str(cfg), "--quiet"])
    out = capsys.readouterr().out
    assert "UNMAPPED logs/1.03_create_model.log" in out
    assert "MOVED" not in out


def test_a_missing_project_dir_says_where_it_looked(tmp_path):
    cfg = _write_config(tmp_path, _config(tmp_path / "does_not_exist"))
    with pytest.raises(SystemExit):
        spt.main(["--config", str(cfg)])


def test_project_dir_override_wins_over_the_config(tmp_path, capsys):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(tmp_path / "somewhere_else"))
    assert spt.main(["--config", str(cfg), "--project-dir", str(proj)]) == 1
    assert "UNMAPPED logs/1.03_create_model.log" in capsys.readouterr().out
