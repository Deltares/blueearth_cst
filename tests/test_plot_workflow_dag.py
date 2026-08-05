"""Contract tests for scripts/plot_workflow_dag.py.

Covers the two derivations the tool exists for -- WHERE the graph is written and
WHAT it is called -- plus the graphviz-missing message. No real snakemake or
graphviz runs; the subprocess boundary is monkeypatched.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import plot_workflow_dag as pwd  # noqa: E402


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_r01_cfg(path, project_dir, experiment=None):
    """An R01 config; `experiment` adds the key WF3's DAG scope reads.

    Optional on purpose -- both WF3 branches are exercised below: the
    experiment scope, and the fallback when the key is absent.
    """
    text = (
        "project:\n"
        f"  project_dir: {project_dir}\n"
        "  static_dir: config\n"
    )
    if experiment is not None:
        text += (
            "workflows:\n"
            "  climate_experiment:\n"
            f"    experiment_name: {experiment}\n"
        )
    path.write_text(text, encoding="utf-8")


# --- workflow number -------------------------------------------------------

@pytest.mark.parametrize(
    "snakefile, number",
    [
        ("Snakefile_model_creation", 1),
        ("Snakefile_climate_projections", 2),
        ("Snakefile_climate_experiment", 3),
    ],
)
def test_workflow_number_covers_all_three(snakefile, number):
    assert pwd.workflow_number(pwd.Path(snakefile)) == number


def test_workflow_number_is_by_basename_not_full_path():
    assert pwd.workflow_number(pwd.Path("/some/repo/Snakefile_model_creation")) == 1


def test_unknown_snakefile_names_the_valid_ones():
    with pytest.raises(pwd.DagPlotError) as excinfo:
        pwd.workflow_number(pwd.Path("Snakefile_nope"))
    assert "Snakefile_model_creation" in str(excinfo.value)


# --- project_dir / project_name derivation ---------------------------------

def test_absolute_project_dir_is_used_verbatim(tmp_path):
    cfg = tmp_path / "cfg.yml"
    _write_r01_cfg(cfg, "C:/TESTS/CST/gabon_0108")
    project_dir, name, _ = pwd.read_project(cfg)
    assert project_dir == pwd.Path("C:/TESTS/CST/gabon_0108")
    assert name == "gabon_0108"


def test_relative_project_dir_resolves_against_the_repo_root(tmp_path):
    """Snakemake runs with cwd=REPO_ROOT, so the plot must land where its
    outputs do -- not next to the config or the caller's cwd."""
    cfg = tmp_path / "cfg.yml"
    _write_r01_cfg(cfg, "test_case/test_local")
    project_dir, name, _ = pwd.read_project(cfg)
    assert project_dir == pwd.REPO_ROOT / "test_case" / "test_local"
    assert name == "test_local"


def test_legacy_flat_config_uses_top_level_keys(tmp_path):
    """The single-workflow projections configs have no `project:` section but
    do carry an explicit project_name, which wins over the basename."""
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        "project_name: Gabon\nproject_dir: test_case/gabon\n", encoding="utf-8"
    )
    project_dir, name, _ = pwd.read_project(cfg)
    assert project_dir == pwd.REPO_ROOT / "test_case" / "gabon"
    assert name == "Gabon"


def test_explicit_project_name_wins_in_the_r01_schema(tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        "project_name: Gabon\nproject:\n  project_dir: C:/TESTS/CST/gabon_0108\n",
        encoding="utf-8",
    )
    _, name, _ = pwd.read_project(cfg)
    assert name == "Gabon"


def test_missing_project_dir_is_a_hard_error(tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("project:\n  static_dir: config\n", encoding="utf-8")
    with pytest.raises(pwd.DagPlotError, match="no project_dir"):
        pwd.read_project(cfg)


def test_missing_config_file_is_a_hard_error(tmp_path):
    with pytest.raises(pwd.DagPlotError, match="not found"):
        pwd.read_project(tmp_path / "absent.yml")


# --- end-to-end output path ------------------------------------------------

def test_output_lands_in_logs_dag_named_project_wfN(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "cfg.yml"
    project_dir = tmp_path / "gabon_0108"
    _write_r01_cfg(cfg, project_dir.as_posix())

    rendered = {}

    def fake_run(cmd, cwd=None, input=None, **kwargs):
        if cmd[0] == "dot-stub":
            rendered["argv"] = cmd
            rendered["dot"] = input
            return FakeResult()
        assert cwd == pwd.REPO_ROOT, "snakemake must run from the repo root"
        return FakeResult(stdout="digraph snakemake_dag {\n}\n")

    monkeypatch.setattr(pwd.subprocess, "run", fake_run)
    monkeypatch.setattr(pwd.shutil, "which", lambda _: "dot-stub")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_workflow_dag.py", "-s", "Snakefile_model_creation",
         "--configfile", str(cfg)],
    )

    assert pwd.main() == 0
    expected = project_dir / "logs" / "dag" / "gabon_0108_wf1_dag.png"
    assert rendered["argv"] == ["dot-stub", "-Tpng", "-o", str(expected)]
    assert rendered["dot"].startswith("digraph")
    assert capsys.readouterr().out.strip() == str(expected)
    # render() creates the subdir; the project root stays free of the plot.
    assert expected.parent.is_dir()
    assert not list(project_dir.glob("*.png"))


def test_rulegraph_mode_and_format_reach_the_filename(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    project_dir = tmp_path / "gabon_0108"
    _write_r01_cfg(cfg, project_dir.as_posix(), experiment="experiment")

    seen = {}

    def fake_run(cmd, cwd=None, input=None, **kwargs):
        if cmd[0] == "dot-stub":
            seen["out"] = cmd[-1]
            return FakeResult()
        seen["snakemake"] = cmd
        return FakeResult(stdout="digraph x {}")

    monkeypatch.setattr(pwd.subprocess, "run", fake_run)
    monkeypatch.setattr(pwd.shutil, "which", lambda _: "dot-stub")
    monkeypatch.setattr(
        sys, "argv",
        ["plot_workflow_dag.py", "-s", "Snakefile_climate_experiment",
         "--configfile", str(cfg), "--mode", "rulegraph", "--format", "svg"],
    )

    assert pwd.main() == 0
    assert "--rulegraph" in seen["snakemake"]
    assert seen["out"] == str(
        project_dir / "experiments" / "experiment" / "logs" / "dag"
        / "gabon_0108_wf3_rulegraph.svg"
    )


def test_missing_graphviz_reports_how_to_get_it(monkeypatch):
    monkeypatch.setattr(pwd.shutil, "which", lambda _: None)
    with pytest.raises(pwd.DagPlotError, match="pixi"):
        pwd.render("digraph x {}", pwd.Path("out.png"), "png")


def test_snakemake_failure_surfaces_its_stderr(monkeypatch):
    monkeypatch.setattr(
        pwd.subprocess, "run",
        lambda *a, **k: FakeResult(returncode=1, stderr="MissingInputException: boom"),
    )
    with pytest.raises(pwd.DagPlotError, match="MissingInputException"):
        pwd.build_graph("dag", pwd.Path("Snakefile_model_creation"),
                        pwd.Path("cfg.yml"), "all", [])


# --- R9 P2 commit 4: the render sits at the PRODUCING RUN's scope (P7) ------

def test_wf1_and_wf2_renders_are_project_scoped():
    """Project-scoped producers write to the project's own logs/dag/."""
    cfg = {"workflows": {"climate_experiment": {"experiment_name": "e"}}}
    assert pwd.plot_subdir(1, cfg) == pwd.Path("logs") / "dag"
    assert pwd.plot_subdir(2, cfg) == pwd.Path("logs") / "dag"


def test_wf3_render_is_experiment_scoped():
    """WF3's DAG describes ONE experiment's run, so it goes with that
    experiment's other run records -- beside wf3_climate_experiment.log."""
    cfg = {"workflows": {"climate_experiment": {"experiment_name": "gabon_dry"}}}
    assert pwd.plot_subdir(3, cfg) == (
        pwd.Path("experiments") / "gabon_dry" / "logs" / "dag"
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"workflows": None},
        {"workflows": {"climate_experiment": None}},
        {"workflows": {"climate_experiment": {}}},
    ],
)
def test_wf3_falls_back_to_project_scope_without_an_experiment_name(cfg):
    """A convenience render must not fail a user's command over a missing
    optional key -- including the several ways YAML spells "absent"."""
    assert pwd.plot_subdir(3, cfg) == pwd.Path("logs") / "dag"


def test_the_render_never_lands_under_the_editable_config_root():
    """The P4 property the move exists for, asserted directly."""
    cfg = {"workflows": {"climate_experiment": {"experiment_name": "e"}}}
    for number in (1, 2, 3):
        assert "config" not in pwd.plot_subdir(number, cfg).parts
