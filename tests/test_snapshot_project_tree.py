"""Unit tests for dev/scripts/snapshot_project_tree.py.

The wrapper's whole value is that it derives the map parameters from the config
instead of taking them on the command line -- a mistyped `--dataset-key` turns a
mapped store into an unmapped one, which reads as a map gap. These tests pin
that derivation, the exclusions, and the exit code.
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import snapshot_project_tree as spt  # noqa: E402

from tests.conftest import write_config  # noqa: E402


def _config(project_dir):
    return {
        "schema_version": 2,
        "project": {"project_dir": str(project_dir).replace("\\", "/")},
        "climate": {
            "selected": "era5",
            "sources": ["era5"],
            # `C-70`: INCLUSIVE YEARS. The store key stays ISO at day
            # resolution, and the tool converts through the same helper a
            # real run uses.
            "window": {"start": 2000, "end": 2020},
        },
        "workflows": {
            "simulate_system": {"experiment_name": "my_experiment"},
            "analyze_projections": {"ensemble": "cmip6"},
        },
    }


def _write_config(tmp_path, cfg, name="cfg"):
    """Write the whole mapping back as the T1 + T2 set the tool reads.

    Split on disk since R13, because the tool composes: this fixture writes the
    shape a real project has, so a regression in the tool's own reading is
    visible here rather than hidden behind a shape no run can produce.
    """
    return write_config(tmp_path, cfg, stem=name)


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
        "experiment_name": "my_experiment_01",
        "dataset_key": "era5_20000101_20201231",
        "clim_project": "cmip6",
    }


def test_map_parameters_fall_back_to_the_shipped_defaults():
    cfg = _config("proj")
    cfg["workflows"] = {}
    params = spt.map_parameters(cfg)
    assert params["experiment_name"] == "stress_test_01"
    assert params["clim_project"] == "cmip6"


def test_a_sub_day_window_fails_loud():
    """The store key is day-resolution; a silent collision would mis-key it.

    **Asserted against `slugify_window` directly, not through a config.** `C-70`
    retyped `climate.window` to INCLUSIVE YEARS, so a sub-day window is no longer
    expressible in a project file â€” `map_parameters` converts through
    `historical_window_bounds`, which yields midnight on both ends by
    construction, and the guard is unreachable from that direction.

    The guard still matters and still fires: `slugify_window` is shared, and any
    future caller handing it a time-of-day would collide two windows onto one
    store key. So the case moved down to the level where it is real rather than
    being deleted with the shape that used to reach it.
    """
    from blueearth_cst.shared.snake_utils import slugify_window

    with pytest.raises(ValueError, match="time-of-day"):
        slugify_window("2000-01-01T06:00:00", "2020-12-31T00:00:00")


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


def test_relative_project_dir_resolves_against_the_cwd_not_the_tool_repo(
    tmp_path, monkeypatch
):
    """The cross-checkout trap the runbook walks into by design.

    The workflows run from the PRIMARY checkout while the comparator lives in a
    task worktree, so the tool's own repo root is the wrong base: it would look
    for the tree beside the tool instead of beside the run.
    """
    monkeypatch.setattr(spt, "REPO", tmp_path / "tool-checkout")
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
    retired (`dev/records/reviews/2026-08-11_test-suite-bloat-assessment.md` Â§6a) and
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


def test_numbered_retained_owners_and_override_are_read_only(tmp_path, capsys):
    from blueearth_cst.experiment import reservation
    from blueearth_cst.shared.workflow_config_snapshot import (
        archive_lock,
        file_reference,
    )
    from tests import _v2_experiment

    fixture = _v2_experiment.build_v2_experiment(tmp_path)
    root = fixture.project
    request = {
        "schema_version": "experiment-request/1",
        "generation": None,
        "selection": {
            "mode": "explicit-manifest",
            "manifest": file_reference(fixture.collection_marker, "project_root", root),
        },
        "input_catalogs": [],
        "simulation": {
            "climate_source": "era5",
            "model_root": "models/hydrology/wflow",
            "outvars": ["gwr"],
            "model_request": None,
        },
    }
    captures = [
        reservation.CapturedSource(role, path, path.read_bytes())
        for role, path in (
            ("project", fixture.config),
            ("simulate_system", tmp_path / "simulate_system.yml"),
        )
    ]
    with archive_lock(root, "project-execution"):
        for new in (False, True):
            reservation.admit_experiment_owned(
                root,
                request=request,
                sources=captures,
                creator_invocation_id=uuid.uuid4().hex,
                new_experiment=new,
            )
    configured = _config(tmp_path / "not-the-inspected-root")
    configured["workflows"]["simulate_system"]["experiment_name"] = "stress_test"
    config_path = _write_config(tmp_path, configured, name="inspection")
    _touch(root, "experiments/unknown_run/_engine/reservation.json", "unowned bytes")
    _touch(root, "experiments/stress_test_01/orphan_table.csv")
    before = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert spt.map_parameters(configured, root)["experiment_name"] == "stress_test_02"
    assert spt.retained_experiment_names(root) == (
        _v2_experiment.EXPERIMENT,
        "stress_test_01",
        "stress_test_02",
    )
    legacy_config = _config(root)
    legacy_config["workflows"]["simulate_system"]["experiment_name"] = (
        _v2_experiment.EXPERIMENT
    )
    assert (
        spt.map_parameters(legacy_config)["experiment_name"]
        == _v2_experiment.EXPERIMENT
    )
    assert (
        spt.main(["--config", str(config_path), "--project-dir", str(root), "--quiet"])
        == 1
    )
    output = capsys.readouterr().out
    assert "experiment    : stress_test_02" in output
    assert "UNMAPPED experiments/unknown_run/_engine/reservation.json" in output
    assert "UNMAPPED experiments/stress_test_01/orphan_table.csv" in output
    assert "UNMAPPED experiments/stress_test_01/_engine/reservation.json" not in output
    assert "UNMAPPED experiments/stress_test_02/_engine/reservation.json" not in output
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_scaffold_refuses_dynamic_workflows_before_config_read_or_output(tmp_path):
    import scaffold_project_tree as scaffold

    output = tmp_path / "not-created"
    with pytest.raises(SystemExit):
        scaffold.main(
            [
                "--config",
                str(tmp_path / "absent.yml"),
                "--workflows",
                "3,4",
                "--out",
                str(output),
            ]
        )
    assert not output.exists()


def test_scaffold_refuses_nonempty_destination_without_touching_it(tmp_path, capsys):
    import scaffold_project_tree as scaffold

    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "valuable.txt"
    sentinel.write_bytes(b"keep these bytes")
    with pytest.raises(SystemExit) as error:
        scaffold.main(["--config", str(tmp_path / "absent.yml"), "--out", str(output)])
    assert error.value.code == 2
    assert sentinel.read_bytes() == b"keep these bytes"
    assert list(output.iterdir()) == [sentinel]
    assert "--out is not empty" in capsys.readouterr().err


@pytest.mark.parametrize("destination", ["new", "empty", "keep"])
def test_scaffold_adds_placeholders_without_replacing_files(
    tmp_path, monkeypatch, destination
):
    import scaffold_project_tree as scaffold

    output = tmp_path / "scaffold"
    if destination != "new":
        output.mkdir()
    retained = output / scaffold.cross_workflow_inputs.LEAF_WF1_SNAPSHOT
    if destination == "keep":
        retained.parent.mkdir(parents=True)
        retained.write_bytes(b"existing prerequisite")
        (output / "existing.txt").write_bytes(b"existing output")
    config = tmp_path / "config.yml"
    config.write_text("project: {}\nworkflows: {}\n", encoding="utf-8")
    monkeypatch.setattr(scaffold, "_summary", lambda *_: ["existing.txt", "new.txt"])
    monkeypatch.setattr(scaffold, "_log_paths", lambda *_: [])
    args = [
        "--config",
        str(config),
        "--out",
        str(output),
        "--workflows",
        "0",
        "--extras",
        str(tmp_path / "no-extras.yml"),
    ]
    if destination == "keep":
        args.append("--keep")
    assert scaffold.main(args) == 0
    assert (output / "new.txt").is_file()
    assert (output / "existing.txt").is_file()
    if destination == "keep":
        assert retained.read_bytes() == b"existing prerequisite"
        assert (output / "existing.txt").read_bytes() == b"existing output"


def test_scaffold_stops_on_summary_failure_without_reporting_success(
    tmp_path, monkeypatch, capsys
):
    import subprocess

    import scaffold_project_tree as scaffold

    output = tmp_path / "scaffold"
    config = tmp_path / "config.yml"
    config.write_text("project: {}\nworkflows: {}\n", encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                command, 0, "new.txt\tdate\trule\tstatus\n", ""
            )
        return subprocess.CompletedProcess(
            command, 1, "Building DAG of jobs...\n", "MissingInputException: missing.nc"
        )

    monkeypatch.setattr(scaffold.subprocess, "run", run)
    monkeypatch.setattr(scaffold, "_log_paths", lambda *_: [])
    assert scaffold.main(["--config", str(config), "--out", str(output)]) == 1
    captured = capsys.readouterr()
    assert len(calls) == 2  # Stop at WF1; never attempt WF2.
    assert "build_model.smk --summary failed (exit 1)" in captured.err
    assert "Building DAG of jobs" in captured.err
    assert "MissingInputException: missing.nc" in captured.err
    assert "scaffolded" not in captured.out + captured.err
    assert not (output / "new.txt").exists()
