"""WF2 product-local production attribution across launcher-observed runs."""

import json
import time
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import execution_outputs, product_record
from blueearth_cst.shared.config_paths import resolve_project_paths
from blueearth_cst.shared.product_record import RECORD_DIRECTORY, RECORD_NAME
from scripts import run_workflows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def make_project(tmp_path):
    """A rapid project with only WF2 enabled and its catalogs local."""
    config = resolve_project_paths(
        yaml.safe_load((ROOT / "test_case/project_config_rapid.yml").read_text()),
        ROOT / "test_case/project_config_rapid.yml",
    )
    output = tmp_path / "output"
    config["project"]["project_dir"] = str(output)
    config["project"]["catalog"] = [str(ROOT / "config/catalogs/deltares_data.yml")]
    for name, stanza in config["workflows"].items():
        stanza["enabled"] = name == "analyze_projections"
        if "config_path" in stanza:
            stanza["config_path"] = str(
                (ROOT / "test_case" / stanza["config_path"]).resolve()
            )
    path = tmp_path / "project_config_test.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, output


def _run(monkeypatch, config, child):
    """Run the all-workflow launcher with `child(env)` standing in for Snakemake."""

    def fake(cmd, **kwargs):
        return child(kwargs["env"])

    monkeypatch.setattr(run_workflows, "run_project_child", fake)
    return run_workflows.run(str(config), cores=1, extra=[])


def _writer(product, files, *, remove=(), exit_code=0):
    def child(env):
        for name, text in files.items():
            (product / name).parent.mkdir(parents=True, exist_ok=True)
            (product / name).write_text(text, encoding="utf-8")
        for name in remove:
            (product / name).unlink()
        directory = Path(env[execution_outputs.DIRECTORY_ENV])
        (directory / "wf2.log").write_text(f"log {sorted(files)}\n", encoding="utf-8")
        return exit_code

    return child


def _record(product):
    return json.loads((product / RECORD_DIRECTORY / RECORD_NAME).read_text())


def _producers(record):
    return {name: item["producer"] for name, item in record["files"].items()}


def test_runs_keep_truthful_attribution(project, monkeypatch):
    config, output = project
    product = output / "data/climate/projections/cmip6"
    first = {"summary/provenance.json": "p1", "analysis/coverage.csv": "c1"}
    first |= {"plots/a/fig.png": "f1"}
    assert _run(monkeypatch, config, _writer(product, first)) == 0
    record = _record(product)
    (success,) = record["generations"]
    assert record["state"] == "complete"
    assert set(_producers(record).values()) == {success}
    generation = record["generations"][success]
    assert generation["status"] == "succeeded"
    assert generation["changes"] == {
        "added": 3,
        "changed": 0,
        "rewritten_identical": 0,
        "removed": 0,
    }
    retained = product / RECORD_DIRECTORY / success / "wf2.log"
    assert retained.read_text() == f"log {sorted(first)}\n"
    reference = generation["configuration"]["run_record"]
    assert reference["path"].startswith(
        "config/_engine/configuration-snapshots/analyze_projections/"
    )
    assert record["shared_steps"][0]["outputs"] == "data/spatial/geoms"

    # A no-op, successful or failed, leaves the production evidence untouched.
    before = (product / RECORD_DIRECTORY / RECORD_NAME).read_bytes()
    assert _run(monkeypatch, config, _writer(product, {})) == 0
    assert _run(monkeypatch, config, _writer(product, {}, exit_code=1)) == 1
    assert (product / RECORD_DIRECTORY / RECORD_NAME).read_bytes() == before

    # A failed retry that replaced some files marks mixed generations.
    retry = _writer(
        product,
        {"analysis/coverage.csv": "c2"},
        remove=["plots/a/fig.png"],
        exit_code=1,
    )
    assert _run(monkeypatch, config, retry) == 1
    record = _record(product)
    failed = record["latest_invocation_id"]
    assert record["state"] == "incomplete"
    assert _producers(record) == {
        "summary/provenance.json": success,
        "analysis/coverage.csv": failed,
    }
    assert record["generations"][failed]["status"] == "failed"
    assert record["generations"][failed]["changes"]["removed"] == 1

    # A partial plotting rerun completes the product without re-crediting tables.
    plots = _writer(product, {"plots/a/fig.png": "f2"})
    assert _run(monkeypatch, config, plots) == 0
    record = _record(product)
    rerun = record["latest_invocation_id"]
    assert record["state"] == "complete"
    assert _producers(record) == {
        "summary/provenance.json": success,
        "analysis/coverage.csv": failed,
        "plots/a/fig.png": rerun,
    }
    assert set(record["generations"]) == {success, failed, rerun}

    # Generations that own no current file lose only their retained copies.
    tables = _writer(
        product,
        {"summary/provenance.json": "p3", "analysis/coverage.csv": "c3"},
    )
    assert _run(monkeypatch, config, tables) == 0
    record = _record(product)
    assert set(record["generations"]) == {rerun, record["latest_invocation_id"]}
    assert not (product / RECORD_DIRECTORY / success).exists()
    assert not (product / RECORD_DIRECTORY / failed).exists()
    assert (output / f"config/_engine/invocations/{success}.json").is_file()


def test_identical_rewrite_is_credited_but_untouched_files_are_not(
    project, monkeypatch
):
    """A deterministic plot rerun rewrites equal bytes; that is production."""
    config, output = project
    product = output / "data/climate/projections/cmip6"
    first = {"summary/provenance.json": "p", "plots/a/fig.png": "f"}
    assert _run(monkeypatch, config, _writer(product, first)) == 0
    tables = _record(product)["latest_invocation_id"]
    before = (product / RECORD_DIRECTORY / RECORD_NAME).read_bytes()
    assert _run(monkeypatch, config, _writer(product, {})) == 0
    assert (product / RECORD_DIRECTORY / RECORD_NAME).read_bytes() == before

    time.sleep(0.01)  # distinct write time on coarse-clock filesystems
    assert _run(monkeypatch, config, _writer(product, {"plots/a/fig.png": "f"})) == 0
    record = _record(product)
    plots = record["latest_invocation_id"]
    assert plots != tables
    assert _producers(record) == {
        "summary/provenance.json": tables,
        "plots/a/fig.png": plots,
    }
    assert record["generations"][plots]["changes"]["rewritten_identical"] == 1


def test_unobserved_bytes_are_not_credited(project, monkeypatch):
    config, output = project
    product = output / "data/climate/projections/cmip6"
    # A product written before records existed, or by bare Snakemake.
    (product / "summary").mkdir(parents=True)
    (product / "summary/provenance.json").write_text("legacy", encoding="utf-8")
    assert _run(monkeypatch, config, _writer(product, {"plots/x.png": "x"})) == 0
    record = _record(product)
    assert _producers(record) == {
        "summary/provenance.json": None,
        "plots/x.png": record["latest_invocation_id"],
    }
    assert record["unattributed_files"] == 1

    (product / "plots/x.png").write_text(
        "edited outside the launcher", encoding="utf-8"
    )
    assert _run(monkeypatch, config, _writer(product, {"report.md": "r"})) == 0
    record = _record(product)
    assert _producers(record)["plots/x.png"] is None
    assert record["unattributed_files"] == 2


def test_single_workflow_launcher_records_production(project, monkeypatch):
    from blueearth_cst.shared import windows_job, workflow_archive_launch

    config, output = project
    product = output / "data/climate/projections/cmip6"
    child = _writer(product, {"report.md": "r"}, exit_code=1)
    monkeypatch.setattr(
        windows_job, "run_project_child", lambda cmd, **kwargs: child(kwargs["env"])
    )
    result = workflow_archive_launch.run_workflow(
        "analyze_projections", config, output, cores=1
    )
    assert result == 1
    record = _record(product)
    assert record["state"] == "incomplete"
    assert _producers(record) == {"report.md": record["latest_invocation_id"]}


def test_record_failure_never_replaces_the_outcome(project, monkeypatch, capsys):
    config, output = project
    product = output / "data/climate/projections/cmip6"

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(product_record, "record_production", broken)
    assert _run(monkeypatch, config, _writer(product, {"report.md": "r"})) == 0
    assert "production record unavailable" in capsys.readouterr().err
    assert not (product / RECORD_DIRECTORY).exists()


def test_other_workflows_have_no_product_record(tmp_path):
    assert product_record.product_root(tmp_path, "analyze_climate") is None
    assert product_record.observe("build_model", tmp_path) is None
