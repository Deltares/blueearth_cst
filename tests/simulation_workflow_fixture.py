"""Parse successor rule declarations with collection validation isolated."""

import uuid
from pathlib import Path

import yaml

from tests.snakemake_state import prepare_workdir


def parse_simulation_workflow(
    tmp_path, monkeypatch, *, config=None, dry_run=False, include_unperturbed=True
):
    """Supply an already-validated collection to test scheduling declarations.

    Physical artifact validation is exercised by the collection/response suites;
    this fixture deliberately substitutes only that external boundary.
    """
    import snakemake.api as api

    from blueearth_cst.experiment import simulation_runner
    from blueearth_cst.experiment.content_identity import atomic_record

    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    collection = tmp_path / "collection"
    collection.mkdir(exist_ok=True)
    atomic_record(
        collection / "collection_intent.json",
        {"scenario_spec": {"simulation_window": {"start": 2046, "end": 2054}}},
    )
    selection = {
        "manifest_path": (collection / "collection.json").as_posix(),
        "resolution_mode": "explicit-manifest",
    }
    manifest = {
        "series": [
            {
                "run_id": run,
                "file": {
                    "path": f"scenarios/fixture/series/{run}.nc",
                    "sha256": "0" * 64,
                },
            }
            for run in ("run_01", "run_02", "run_03")
        ]
    }
    if not include_unperturbed:
        from blueearth_cst.shared.workflow_config_snapshot import file_reference

        lookup = collection / "scenario_run_lookup.csv"
        lookup.write_text(
            "run_id,st_id\nrun_01,\nrun_02,1\nrun_03,2\n", encoding="utf-8"
        )
        manifest["scenario_run_lookup"] = file_reference(
            lookup, "record_directory", collection
        )
    monkeypatch.setattr(
        simulation_runner,
        "resolve_selected_collection",
        lambda *args: (selection, manifest),
    )
    monkeypatch.setenv("CST_SIMULATION_OPERATION", "simulate-and-indicators")
    monkeypatch.setenv("CST_SIMULATION_INVOCATION_ID", uuid.uuid4().hex)
    settings = tmp_path / "simulation.yml"
    settings.write_text(
        "experiment_name: fixture\n"
        + (
            "operation: simulate-and-indicators\n"
            if include_unperturbed
            else "operations: [simulations]\ninclude_unperturbed: false\n"
        ),
        encoding="utf-8",
    )
    document = config or {
        "schema_version": 2,
        "model": {"outvars": ["river discharge"]},
    }
    document["project"] = {"project_dir": project.as_posix()}
    document["workflows"] = {
        "simulate_system": {"enabled": True, "config_path": settings.name}
    }
    path = tmp_path / "project.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with api.SnakemakeApi() as sa:
        wf_api = sa.workflow(
            resource_settings=api.ResourceSettings(cores=1),
            config_settings=api.ConfigSettings(configfiles=[path]),
            storage_settings=api.StorageSettings(),
            workflow_settings=api.WorkflowSettings(),
            snakefile=repo / "simulate_system.smk",
            workdir=prepare_workdir(tmp_path / "snakemake"),
        )
        workflow = wf_api._workflow
        workflow.include(workflow.main_snakefile, overwrite_default_target=True)
        if dry_run:
            # Only external leaves are substituted. Snakemake builds the real
            # simulation DAG and defers its uncompleted snapshot checkpoint.
            Path(selection["manifest_path"]).touch()
            for leaf in (".outputs_configured", "wflow_sbm.toml"):
                path = project / "models/hydrology/wflow" / leaf
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for item in manifest["series"]:
                path = project / item["file"]["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            wf_api.dag(
                dag_settings=api.DAGSettings(targets={"simulations_only"})
            ).execute_workflow(executor="dryrun")
        return workflow
