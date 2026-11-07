"""Historical publications retain executed provenance and survive mutable runs."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.model.historical_runs import (
    execution_inputs,
    historical_run_name,
    read_historical_run,
    retain_historical_run,
    run_directory,
    write_forcing_record,
)
from blueearth_cst.shared.provenance import file_sha256


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
