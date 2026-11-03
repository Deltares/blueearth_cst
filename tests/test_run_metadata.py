"""WF1 metadata follows output inputs, without relabelling reuse after a commit."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import workflow_archive_launch as launch
from blueearth_cst.shared.config_paths import (
    resolve_project_paths,
    resolve_workflow_paths,
)
from tests.test_snapshot_config_rules import _rule_block

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.process_isolation
def test_metadata_repeat_ignores_git_state_but_tracks_configuration_inputs(
    tmp_path, monkeypatch
):
    project = resolve_project_paths(
        yaml.safe_load((REPO / "test_case/project_config_rapid.yml").read_text()),
        REPO / "test_case/project_config_rapid.yml",
    )
    root = tmp_path / "project"
    project["project"]["project_dir"] = str(root)
    catalog = tmp_path / "catalog.yml"
    catalog.write_bytes((REPO / "config/catalogs/deltares_data.yml").read_bytes())
    project["project"]["catalog"] = [str(catalog)]
    settings = tmp_path / "workflow.yml"
    stanza = project["workflows"]["build_model"]
    settings.write_text(
        yaml.safe_dump(
            resolve_workflow_paths(
                yaml.safe_load(
                    (REPO / "test_case" / stanza["config_path"]).read_bytes()
                ),
                REPO / "test_case" / stanza["config_path"],
                "build_model",
                project["schema_version"],
            )
        ),
        encoding="utf-8",
    )
    stanza["config_path"] = str(settings)
    source = tmp_path / "project.yml"
    source.write_text(yaml.safe_dump(project), encoding="utf-8")
    toolbox = {"commit": "a" * 40, "commit_source": "git", "dirty": False}
    environment = {"pixi.lock": "a" * 64, "Manifest.toml": "b" * 64}
    monkeypatch.setattr(launch, "toolbox_identity", lambda: dict(toolbox))
    monkeypatch.setattr(launch, "environment_file_hashes", lambda: dict(environment))

    model = root / "models/hydrology/wflow"
    (model / "evaluation").mkdir(parents=True)
    product = model / "evaluation/performance_indicators.csv"
    product.write_text("indicator,value\nfixture,1\n", encoding="utf-8")
    (model / ".model_final").touch()
    rule = "rule write_run_metadata:\n" + _rule_block(
        REPO / "build_model.smk", "write_run_metadata"
    )
    rule = rule.replace(
        '"blueearth_cst/shared/write_run_metadata.py"',
        repr(str(REPO / "blueearth_cst/shared/write_run_metadata.py")),
    )
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        "import os\nfrom types import SimpleNamespace\n"
        "from blueearth_cst.shared.provenance import configuration_inputs_digest\n"
        "from blueearth_cst.shared.workflow_archive_launch import require_capture\n"
        "config_path = os.environ['TEST_EXECUTION_CONFIG']\n"
        "CAPTURE_RECORD = require_capture('build_model', config_path, dry_run=False)\n"
        "EFFECTIVE_CONFIG_DIGEST = CAPTURE_RECORD['configuration_projection']['effective_config_sha256']\n"
        "CONFIGURATION_INPUTS_DIGEST = CAPTURE_RECORD['configuration_projection']['configuration_inputs_sha256']\n"
        f"basin_dir = {str(model)!r}\n"
        "WRITE_RUN_METADATA = SimpleNamespace(banner=lambda: 'metadata')\n"
        "rule all:\n    input: basin_dir + '/run_metadata.json'\n" + rule,
        encoding="utf-8",
    )

    def run(*, preview=False):
        execution, context = launch.prepare_workflow(
            "build_model", source, root, command=["snakemake"], targets=["all"]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "snakemake",
                "-s",
                str(snakefile),
                "-c",
                "1",
                *(["--dry-run"] if preview else []),
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO),
                "TEST_EXECUTION_CONFIG": str(execution),
                launch.CONTEXT_ENV: str(context),
            },
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    run()
    metadata = model / "run_metadata.json"
    before = metadata.read_bytes(), metadata.stat().st_mtime_ns
    first = json.loads(before[0])
    toolbox.update(commit="b" * 40, dirty=True)
    assert "Nothing to be done" in run()
    assert (metadata.read_bytes(), metadata.stat().st_mtime_ns) == before
    archive = launch.read_archive(root, "build_model", "config/build_model")
    assert (
        first["configuration_inputs_sha256"]
        != archive["configuration_projection"]["configuration_inputs_sha256"]
    )
    assert "Nothing to be done" in run()

    environment["pixi.lock"] = "c" * 64
    assert "write_run_metadata" in run(preview=True)
    run()
    assert (
        json.loads(metadata.read_bytes())["configuration_inputs_sha256"]
        == (
            launch.read_archive(root, "build_model", "config/build_model")[
                "configuration_projection"
            ]["configuration_inputs_sha256"]
        )
    )
    catalog.write_bytes(catalog.read_bytes() + b"\n# changed referenced input\n")
    assert "write_run_metadata" in run(preview=True)
    run()
    project["basin"]["resolution"] *= 2
    source.write_text(yaml.safe_dump(project), encoding="utf-8")
    assert "write_run_metadata" in run(preview=True)
