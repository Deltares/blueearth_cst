"""Unit tests for dev/scripts/snapshot_project_tree.py.

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
            "run_stress_test": {"experiment_name": "my_experiment"},
            "analyze_projections": {"clim_project": "cmip6"},
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
    _touch(tmp_path, "hydrology_model/hydromt.log")  # undeclared, kept
    _touch(tmp_path, "logs/wf1_build_model.log")  # excluded from tree
    _touch(tmp_path, "hydrology_model/.model_built")  # dotfile, kept
    assert spt.list_tree(tmp_path) == [
        "hydrology_model/.model_built",
        "hydrology_model/hydromt.log",
        "logs/wf1_build_model.log",
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
    """A miniature project tree: one declared artifact, one orphan.

    Both sides are in TODAY's layout. Until 2026-08-11 this fixture was
    pre-migration and these tests passed `--map r09`; that one-way map is
    retired (`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §6a) and
    the only map left is the post-migration inventory.
    """
    proj = tmp_path / "proj"
    _touch(proj, "data/spatial/spatial_maps.nc")
    _touch(proj, "logs/1.03_create_model.log")  # pre-_parts orphan shape
    return proj


def test_reports_unmapped_and_exits_nonzero(tmp_path, capsys):
    """An undeclared artifact fails the gate and is named in the report."""
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert "UNMAPPED logs/1.03_create_model.log" in out
    assert "IDENTITY data/spatial/spatial_maps.nc" in out
    assert "either a leftover ORPHAN" in out


def test_a_clean_tree_exits_zero(tmp_path, capsys):
    """A tree holding only declared artifacts passes."""
    proj = tmp_path / "proj"
    _touch(proj, "data/spatial/spatial_maps.nc")
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg)]) == 0
    assert "MAP CLEAN" in capsys.readouterr().out


def test_the_map_is_the_post_migration_inventory(tmp_path, capsys):
    """`[R10-11]`: the gate must pass on a tree in today's layout.

    Before that finding, `tree-check` ran the one-way R9 migration map against
    a migrated tree and reported every relocated artifact as unmapped -- exit 1
    on every correct tree. The map was retired on 2026-08-11, so the era
    mismatch is now structurally impossible rather than merely defaulted away;
    what survives is the half that can still regress, and the banner naming
    which map ran.
    """
    proj = tmp_path / "proj"
    _touch(proj, "data/spatial/spatial_maps.nc")
    cfg = _write_config(tmp_path, _config(proj))

    assert spt.main(["--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "map           : current" in out
    assert "MAP CLEAN" in out


def test_the_default_map_still_reports_an_undeclared_artifact(tmp_path, capsys):
    """The inventory is a gate, not a rubber stamp."""
    proj = tmp_path / "proj"
    _touch(proj, "data/spatial/spatial_maps.nc")
    _touch(proj, "data/spatial/leftover_intermediate.nc")
    cfg = _write_config(tmp_path, _config(proj))

    assert spt.main(["--config", str(cfg)]) == 1
    assert "UNMAPPED data/spatial/leftover_intermediate.nc" in capsys.readouterr().out


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
    assert "era5_20000101_20201231" in text  # the derived store key
    assert "my_experiment" in text  # the derived experiment
    # `--check-map` skips comments and blank lines; what is left must be the
    # exact path list, so the two tools cannot disagree about the snapshot.
    payload = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert payload == spt.list_tree(proj)


def test_no_check_skips_the_gate_and_always_exits_zero(tmp_path, capsys):
    proj = _tree(tmp_path)
    cfg = _write_config(tmp_path, _config(proj))
    assert spt.main(["--config", str(cfg), "--no-check"]) == 0
    assert "UNMAPPED" not in capsys.readouterr().out


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
