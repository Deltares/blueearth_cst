"""Historical publications retain executed provenance and survive mutable runs."""

import json
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.model.historical_runs import (
    completion_inputs,
    completion_path,
    execution_inputs,
    historical_run_name,
    read_historical_run,
    retain_historical_run,
    run_directory,
    write_completion,
    write_forcing_record,
)
from blueearth_cst.shared.provenance import collection_sha256, file_sha256


@pytest.fixture
def historical_model(tmp_path):
    """Use the native output contract and a two-cell model map."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "run_default").mkdir()
    xr.Dataset({"gauges": (("y", "x"), [[1010, 1020]])}).to_netcdf(
        model / "staticmaps.nc"
    )
    xr.Dataset({"precip": ("time", [1.0, 2.0])}).to_netcdf(model / "forcing.nc")
    toml_path = model / "wflow_sbm.toml"
    toml_path.write_text(
        'dir_output="run_default"\n[time]\ncalendar="standard"\ntimestepsecs=86400\n'
        'starttime="2000-01-01"\nendtime="2000-01-03"\n'
        '[model]\ncold_start__flag=true\nriver_routing="kinematic_wave"\n'
        '[input]\npath_static="staticmaps.nc"\npath_forcing="forcing.nc"\n'
        '[output.csv]\npath="output.csv"\n[[output.csv.column]]\n'
        'header="Q"\nmap="gauges"\nparameter="river_water__volume_flow_rate"\n',
        encoding="utf-8",
    )
    registry = tmp_path / "location_registry.csv"
    registry.write_text(
        "wflow_id,subbasin_id,is_primary,station_name\n1010,101,True,Upstream\n1020,102,True,Outlet\n"
    )
    source = tmp_path / "project.yml"
    source.write_text("project: fixture\n")
    config = {"climate": {"selected": "era5"}}
    forcing_record = model / "forcing.json"
    write_forcing_record(toml_path, forcing_record, "era5", None)
    return dict(
        project_dir=tmp_path / "project",
        toml_path=toml_path,
        registry_path=registry,
        forcing_record_path=forcing_record,
        config=config,
        config_path=source,
        workflow_config_path=None,
        repo_root=Path(__file__).resolve().parents[1],
    )


def write_response(model, value=1.2345678901234567):
    """Write the configured native CSV shape (mapped header plus integer ID)."""
    path = model["toml_path"].parent / "run_default/output.csv"
    pd.DataFrame(
        {"Q_1010": [value, 2.0], "Q_1020": [3.0, 4.0]},
        index=pd.date_range("2000-01-02", periods=2),
    ).to_csv(path, index_label="time", float_format="%.17g")


def test_two_runs_preserve_full_precision_and_read_without_model(historical_model):
    model = historical_model
    with retain_historical_run(name="a", **model):
        write_response(model)
    root = run_directory(model["project_dir"], "a")
    before = {p.name: file_sha256(p) for p in root.iterdir()}
    with retain_historical_run(name="b", **model):
        write_response(model, 9.0)
    model["toml_path"].unlink()
    record, frame, stations = read_historical_run(model["project_dir"], "a")
    assert frame.iloc[0, 0] == 1.2345678901234567
    assert record["recipe"]["precip_source"] == "era5"
    assert stations.wflow_id.tolist() == ["1010", "1020"]
    assert before == {p.name: file_sha256(p) for p in root.iterdir()}


def test_collision_preserves_publication(historical_model):
    model = historical_model
    with retain_historical_run(name="a", **model):
        write_response(model)
    with pytest.raises(FileExistsError, match="already exists"):
        with retain_historical_run(name="a", **model):
            pytest.fail("collision admitted execution")
    read_historical_run(model["project_dir"], "a")


@pytest.mark.parametrize("damage", ["archive", "payload", "receipt", "completion"])
def test_workflow_completion_validates_retention(historical_model, damage):
    model = historical_model
    project = model["project_dir"]
    assert completion_inputs(project, "a") == []
    with retain_historical_run(name="a", **model):
        write_response(model)
    write_completion(project, "a")
    assert completion_inputs(project, "a") == []
    root = run_directory(project, "a")
    if damage == "archive":
        root.rename(root.with_name("saved_archive"))
    elif damage == "payload":
        (root / "discharge.csv").write_text("damaged")
    elif damage == "completion":
        (root / "complete.json").unlink()
    else:
        completion_path(project, "a").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        completion_inputs(project, "a")


@pytest.mark.workflow_contract
def test_wf1_declares_named_completion_without_parse_time_collision(
    historical_model, tmp_path
):
    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.conftest import parse_workflow, write_config

    model = historical_model
    with retain_historical_run(name="a", **model):
        write_response(model)
    write_completion(model["project_dir"], "a")
    config = load_composed_config(
        Path(__file__).with_name("project_config_fixture.yml")
    )
    config["project"]["project_dir"] = model["project_dir"].as_posix()
    config["workflows"]["build_model"]["historical_run"] = {"name": "a"}
    workflow = parse_workflow("build_model.smk", write_config(tmp_path, config))
    rule = workflow.get_rule("run_historical_simulation")
    receipt = completion_path(model["project_dir"], "a")
    assert receipt in map(Path, rule.output)
    assert receipt in map(Path, workflow.get_rule("all").input)
    assert not any(
        Path(path).is_relative_to(run_directory(model["project_dir"], "a"))
        for path in rule.output
    )


@pytest.mark.workflow_contract
def test_named_retention_dag_noop_new_label_collision_and_unrelated_target(
    historical_model, isolated_snakemake
):
    """Exercise actual engine scheduling/cleanup with the native retention context."""
    import subprocess
    import sys

    model = historical_model
    work = isolated_snakemake
    write_response(model)
    native = model["toml_path"].parent / "run_default/output.csv"
    fixture = work / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in model.items()
                },
                "native": native.as_posix(),
                "response": native.read_text(),
            }
        )
    )
    snakefile = work / "Snakefile"
    snakefile.write_text(
        "import json\nfrom pathlib import Path\n"
        "from blueearth_cst.model.historical_runs import completion_inputs, completion_path, retain_historical_run, write_completion\n"
        f"fixture=json.loads(Path({fixture.as_posix()!r}).read_text())\n"
        'arguments=fixture["arguments"]\n'
        'for key in ("toml_path", "registry_path", "forcing_record_path", "config_path", "repo_root"):\n'
        "    arguments[key]=Path(arguments[key])\n"
        'project=arguments["project_dir"]\nname=config["name"]\n'
        "rule all:\n    input: str(completion_path(project, name))\n"
        "rule simulate:\n"
        "    input: guard=lambda wc: completion_inputs(project, name)\n"
        '    output: receipt=str(completion_path(project, name)), csv=fixture["native"]\n'
        "    run:\n"
        "        with retain_historical_run(name=name, **arguments):\n"
        '            Path(output.csv).write_text(fixture["response"])\n'
        "        write_completion(project, name)\n"
        'rule figure:\n    output: "figure.txt"\n'
        '    run:\n        Path(output[0]).write_text("unrelated figure")\n'
    )

    def invoke(name="a", *options):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "snakemake",
                "--snakefile",
                str(snakefile),
                "--directory",
                str(work),
                "--cores",
                "1",
                *options,
                "--config",
                f"name={name}",
            ],
            capture_output=True,
            text=True,
        )

    dry = invoke("a", "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert "simulate" in dry.stdout + dry.stderr
    first = invoke()
    assert first.returncode == 0, first.stdout + first.stderr
    root = run_directory(model["project_dir"], "a")
    hashes = {p.name: file_sha256(p) for p in root.iterdir()}
    for options in (("--dry-run",), ()):
        repeated = invoke("a", *options)
        assert repeated.returncode == 0, repeated.stderr
        assert "Nothing to be done" in repeated.stdout + repeated.stderr
    fresh = invoke("b", "--dry-run")
    assert fresh.returncode == 0 and "simulate" in fresh.stdout + fresh.stderr
    fresh = invoke("b")
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert completion_path(model["project_dir"], "b").is_file()
    collision = invoke("a", "--forcerun", "simulate")
    assert collision.returncode != 0
    assert "already exists" in collision.stdout + collision.stderr
    assert hashes == {p.name: file_sha256(p) for p in root.iterdir()}
    # Engine cleanup may remove its receipt; the archive is never an engine output.
    missing_receipt = invoke("a")
    assert missing_receipt.returncode != 0
    assert "already exists" in missing_receipt.stdout + missing_receipt.stderr
    unrelated = invoke("a", "figure")
    assert unrelated.returncode == 0, unrelated.stdout + unrelated.stderr
    assert hashes == {p.name: file_sha256(p) for p in root.iterdir()}


@pytest.mark.parametrize("failure", ["interruption", "input_drift", "station_mismatch"])
def test_failed_execution_never_publishes(historical_model, failure):
    model = historical_model
    with pytest.raises((ValueError, RuntimeError)):
        with retain_historical_run(name="a", **model):
            write_response(model)
            if failure == "interruption":
                raise RuntimeError("interrupted engine")
            if failure == "input_drift":
                model["toml_path"].write_text("changed")
            else:
                output = model["toml_path"].parent / "run_default/output.csv"
                output.write_text(output.read_text().replace("Q_1010", "Q_9999"))
    assert not run_directory(model["project_dir"], "a").exists()
    with pytest.raises(FileNotFoundError):
        read_historical_run(model["project_dir"], "a")


def test_forcing_drift_before_execution_refuses(historical_model):
    model = historical_model
    (model["toml_path"].parent / "forcing.nc").write_bytes(b"edited")
    with pytest.raises(ValueError, match="prepared forcing"):
        with retain_historical_run(name="a", **model):
            pytest.fail("wrong forcing admitted")


def test_model_identity_excludes_forcing_but_includes_physics(historical_model):
    model = historical_model
    toml = model["toml_path"]
    original = toml.read_text()
    _, identity = execution_inputs(toml, model["registry_path"])
    (toml.parent / "forcing.nc").write_bytes(b"different forcing")
    _, changed = execution_inputs(toml, model["registry_path"])
    assert identity["model_sha256"] == changed["model_sha256"]
    assert identity["forcing_sha256"] != changed["forcing_sha256"]
    toml.write_text(
        original.replace(
            'river_routing="kinematic_wave"', 'river_routing="local_inertial"'
        )
    )
    assert (
        execution_inputs(toml, model["registry_path"])[1]["model_sha256"]
        != identity["model_sha256"]
    )


def test_damaged_payload_refuses(historical_model):
    model = historical_model
    with retain_historical_run(name="a", **model):
        write_response(model)
    (run_directory(model["project_dir"], "a") / "discharge.csv").write_text("damaged")
    with pytest.raises(ValueError, match="damaged"):
        read_historical_run(model["project_dir"], "a")


@pytest.mark.parametrize(
    "binding",
    [
        'name="rain"\nscale=1.0\noffset=0.0',
        'name="precip"\nscale=2.0\noffset=0.0',
        'name="precip"\nscale=1.0\noffset=3.0',
    ],
)
def test_forcing_variable_scale_and_offset_affect_model_identity(
    historical_model, binding
):
    model = historical_model
    toml = model["toml_path"]
    original = toml.read_text()
    section = "\n[input.forcing.atmosphere_water__precipitation_volume_flux]\n"
    toml.write_text(original + section + 'name="precip"\nscale=1.0\noffset=0.0\n')
    baseline = execution_inputs(toml, model["registry_path"])[1]["model_sha256"]
    toml.write_text(original + section + binding + "\n")
    assert execution_inputs(toml, model["registry_path"])[1]["model_sha256"] != baseline


def test_legacy_identity_recovers_forcing_mapping_without_archive_changes(
    historical_model,
):
    model = historical_model
    toml = model["toml_path"]
    toml.write_text(
        toml.read_text() + '\n[input.forcing]\natmosphere_air__temperature="temp"\n'
    )
    with retain_historical_run(name="old", **model):
        write_response(model)
    root = run_directory(model["project_dir"], "old")
    record_path = root / "run.json"
    record = json.loads(record_path.read_text())
    expected = record["model_sha256"]
    record["model"]["input"].pop("forcing")
    record["model_sha256"] = collection_sha256(record["model"])
    record_path.write_text(json.dumps(record))
    marker = root / "complete.json"
    complete = json.loads(marker.read_text())
    complete["payloads"]["run.json"] = file_sha256(record_path)
    marker.write_text(json.dumps(complete))
    before = {p.name: file_sha256(p) for p in root.iterdir()}
    toml.unlink()
    restored, _, _ = read_historical_run(model["project_dir"], "old")
    assert restored["model_sha256"] == expected
    assert restored["archived_model_sha256"] == record["model_sha256"]
    assert restored["forcing_mapping"] == {"atmosphere_air__temperature": "temp"}
    assert before == {p.name: file_sha256(p) for p in root.iterdir()}


def test_cold_state_pointer_is_unused_but_warm_state_is_required(historical_model):
    model = historical_model
    path = model["toml_path"]
    original = path.read_text() + '\n[state]\npath_input="missing.nc"\n'
    path.write_text(original)
    execution_inputs(path, model["registry_path"])
    path.write_text(original.replace("cold_start__flag=true", "cold_start__flag=false"))
    with pytest.raises(FileNotFoundError, match="missing executed input"):
        execution_inputs(path, model["registry_path"])


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"name": "../x"},
        {"name": "con"},
        {"name": "A"},
        {"name": "a", "extra": True},
    ],
)
def test_invalid_names(settings):
    with pytest.raises(ValueError):
        historical_run_name(settings)
