"""End to end: all five workflows through the owned runner, from a fresh root.

Drives ``scripts/run_workflows.py`` -- the user's entry point -- over the rapid
config set, rewritten to write into a temporary project root, so no standing
tree under ``test_case/`` is touched and nothing carries over from an earlier
run. Each workflow's deliverable is then checked through the production
readers, not just for existence.

This replaces three per-workflow smoke tests that had drifted until none could
test anything (t2610041227): two pointed at pre-R9 paths, one force-rebuilt the
standing baseline tree, and one drove WF3 with bare Snakemake, which skips the
runner's pinned plan.

Opt-in and slow (tens of minutes): ``pixi run test-e2e`` exercises live
acquisition; ``pixi run test-e2e-local`` reuses a sealed source-only bundle.
Both need juliaup-managed Julia and Rscript with ``weathergenr``. CI cannot
provide their data, so the validation ladder says when to run either gate.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr
import yaml

from blueearth_cst.shared.config_paths import (
    resolve_project_paths,
    resolve_workflow_paths,
)
from dev.scripts.local_e2e_bundle import RAW_NAMES, verify_bundle

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "test_case" / "project_config_rapid.yml"


def _catalog_root():
    """First data root the rapid project's first catalog declares, or None."""
    project = resolve_project_paths(
        yaml.safe_load(CONFIG.read_text(encoding="utf-8")), CONFIG
    )
    catalogs = project["project"]["catalog"]
    first = catalogs[0] if isinstance(catalogs, list) else catalogs
    meta = (yaml.safe_load((REPO / first).read_text(encoding="utf-8")) or {}).get(
        "meta"
    ) or {}
    roots = meta.get("roots") or ([meta["root"]] if "root" in meta else [])
    return roots[0] if roots else None


def _weathergenr_available():
    if shutil.which("Rscript") is None:
        return False
    result = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "quit(status=as.integer(!requireNamespace('weathergenr', quietly=TRUE)))",
        ],
        capture_output=True,
    )
    return result.returncode == 0


def _gcs_reachable():
    try:
        socket.create_connection(("storage.googleapis.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


def _skip_without_prerequisites():
    root = _catalog_root()
    if root is None or not Path(root).exists():
        pytest.skip(f"data mirror not found (catalog root: {root})")
    if shutil.which("julia") is None:
        pytest.skip("julia not on PATH (juliaup-managed Julia required)")
    if not _weathergenr_available():
        pytest.skip("weathergenr not loadable via Rscript (run `pixi run install`)")
    if not _gcs_reachable():
        pytest.skip("Google Cloud Storage (storage.googleapis.com:443) not reachable")


def _fresh_config_set(
    base: Path, project: Path, local_hydromt: Path | None = None
) -> Path:
    """Copy the rapid config set beside ``base`` with ``project_dir`` redirected.

    Resolve case inputs at their original YAML locations before copying the
    configuration set into the isolated test directory.
    """
    for source in sorted(CONFIG.parent.glob(f"{CONFIG.stem}*.yml")):
        shutil.copyfile(source, base / source.name)
    target = base / CONFIG.name
    document = resolve_project_paths(
        yaml.safe_load(target.read_text(encoding="utf-8")), CONFIG
    )
    for name, stanza in document["workflows"].items():
        if "config_path" in stanza:
            original = CONFIG.parent / stanza["config_path"]
            workflow = resolve_workflow_paths(
                yaml.safe_load(original.read_text(encoding="utf-8")),
                original,
                name,
                document["schema_version"],
            )
            (base / stanza["config_path"]).write_text(
                yaml.safe_dump(workflow), encoding="utf-8"
            )
    document["project"]["project_dir"] = project.as_posix()
    if local_hydromt is not None:
        catalogs = []
        for source in document["project"]["catalog"]:
            catalog = yaml.safe_load((REPO / source).read_text(encoding="utf-8"))
            catalog["meta"].pop("root", None)
            catalog["meta"]["roots"] = [local_hydromt.as_posix()]
            local_catalog = base / Path(source).name
            local_catalog.write_text(
                yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8"
            )
            catalogs.append(local_catalog.as_posix())
        document["project"]["catalog"] = catalogs
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return target


def _one(paths):
    found = sorted(paths)
    assert len(found) == 1, found
    return found[0]


def test_local_config_catalogs_have_only_the_bundle_root(tmp_path):
    hydromt = tmp_path / "bundle/hydromt"
    config = _fresh_config_set(tmp_path, tmp_path / "project", hydromt)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    for catalog_path in document["project"]["catalog"]:
        catalog = yaml.safe_load(Path(catalog_path).read_text(encoding="utf-8"))
        assert catalog["meta"]["roots"] == [hydromt.as_posix()]
        assert "root" not in catalog["meta"]


def _run_and_assert_all_five(
    project: Path, config: Path, log: Path, *, local_inputs: bool = False
) -> None:
    from blueearth_cst.experiment.indicator_plan import read_indicator_set
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_record import read_simulation_v2

    environment = os.environ.copy()
    if local_inputs:
        environment["BLUEEARTH_CST_LOCAL_INPUTS"] = "1"
    else:
        environment.pop("BLUEEARTH_CST_LOCAL_INPUTS", None)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_workflows.py",
                "--config",
                str(config),
                "--project-dir",
                str(project),
                "--cores",
                "3",
            ],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3 * 3600,
            env=environment,
        )
    output = log.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, output[-6000:]

    # WF0 builds one historical store per compared source and compares them;
    # WF1 forces from the selected source's store and runs Wflow.
    historical = project / "data/climate/historical"
    selected = yaml.safe_load(config.read_text(encoding="utf-8"))["climate"]["selected"]
    assert _one(historical.glob(f"{selected}_*/extract_historical.nc"))
    with xr.open_dataset(
        _one(historical.glob("chirps_*/extract_historical.nc"))
    ) as chirps:
        assert chirps.time.size == 6210
        assert str(chirps.time.values[0])[:10] == "2000-01-01"
        assert str(chirps.time.values[-1])[:10] == "2016-12-31"
    assert (historical / "comparison/dataset_comparison.csv").stat().st_size > 0
    discharge = project / "models/hydrology/wflow/run_default/output_q.csv"
    assert discharge.stat().st_size > 0
    # WF2: the CMIP6 change factors.
    annual = project / "data/climate/projections/cmip6/summary"
    assert (annual / "cmip6_change_factors_annual.csv").stat().st_size > 0
    # WF3: one ready scenario collection, read back by the production reader.
    marker = _one(project.glob("scenarios/_engine/collections/*/collection.json"))
    assert read_collection_v2(marker)["status"] == "ready"
    # WF4: a ready simulation and one published, readable indicator set.
    # The rapid config chooses a family; a fresh root gets its first number.
    experiment = project / "experiments" / "experiment_rapid_01"
    assert read_simulation_v2(experiment)["status"] == "ready"
    indicators = _one(experiment.glob("_engine/indicator_sets/*/indicators.json"))
    manifest = read_indicator_set(experiment, indicators)
    tables = [experiment / item["file"]["path"] for item in manifest["tables"]]
    assert tables and all(path.stat().st_size > 0 for path in tables)

    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (project / "config/_engine/invocations").glob("*.json")
    ]
    parent = max(
        (record for record in records if record["workflow"] is None),
        key=lambda record: record["started_at_utc"],
    )
    diagnostics = project / parent["execution_directory"]
    assert (diagnostics / "summary.md").is_file()
    transcript = (diagnostics / "console.log").read_text(encoding="utf-8")
    assert "run_workflows" in transcript and "\x1b" not in transcript
    for number in range(5):
        assert (diagnostics / f"wf{number}.log").is_file()
        assert (diagnostics / f"wf{number}-benchmarks.md").is_file()
        assert (diagnostics / f"wf{number}-benchmarks.tsv").is_file()
    assert (diagnostics / "wf4-indicators.log").is_file()
    assert not (project / "benchmarks").exists()


@pytest.mark.integration
def test_all_five_workflows_run_end_to_end_from_a_fresh_root(tmp_path):
    _skip_without_prerequisites()
    project = tmp_path / "project"
    config = _fresh_config_set(tmp_path, project)
    _run_and_assert_all_five(project, config, tmp_path / "run_workflows.log")


@pytest.mark.integration
def test_all_five_workflows_run_from_local_inputs(tmp_path):
    bundle_setting = os.environ.get("BLUEEARTH_CST_E2E_BUNDLE")
    if not bundle_setting:
        pytest.fail("set BLUEEARTH_CST_E2E_BUNDLE to a sealed local input bundle")
    bundle = Path(bundle_setting)
    try:
        verify_bundle(bundle)
    except (OSError, ValueError) as exc:
        pytest.fail(f"local E2E bundle is invalid: {exc}")
    if shutil.which("julia") is None or not _weathergenr_available():
        pytest.fail("local E2E needs julia and weathergenr")
    project = tmp_path / "project"
    raw_root = project / "data/climate/projections/cmip6/raw"
    raw_root.mkdir(parents=True)
    for name in RAW_NAMES:
        shutil.copyfile(bundle / "cmip6/raw" / name, raw_root / name)
    config = _fresh_config_set(tmp_path, project, bundle / "hydromt")
    _run_and_assert_all_five(
        project, config, tmp_path / "run_workflows_local.log", local_inputs=True
    )
