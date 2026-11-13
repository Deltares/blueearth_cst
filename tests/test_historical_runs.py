"""Recipe runs are read in place within their model (R16, t017 ruling D18)."""

import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.model.historical_runs import (
    _check_inputs,
    execution_inputs,
    read_recipe_run,
    warn_retired_historical_run,
    write_forcing_record,
)
from blueearth_cst.shared.named_forcings import HistoricalLayout


@pytest.fixture
def historical_model(tmp_path):
    """One named recipe in the layout 4 shape, with published response tables."""
    from blueearth_cst.experiment.wflow_response_reader import NativeRunArtifacts
    from blueearth_cst.shared.response_tables import publish_response_tables

    model = tmp_path / "model"
    layout = HistoricalLayout("era5")
    (model / layout.run_dir).mkdir(parents=True)
    (model / layout.config_dir).mkdir(parents=True)
    (model / layout.forcing).parent.mkdir(parents=True)
    xr.Dataset({"gauges": (("y", "x"), [[1010, 1020]])}).to_netcdf(
        model / "staticmaps.nc"
    )
    xr.Dataset({"precip": ("time", [1.0, 2.0])}).to_netcdf(model / layout.forcing)
    toml_path = model / layout.toml
    toml_path.write_text(
        f'dir_output="{layout.run_dir}"\n[time]\ncalendar="standard"\n'
        'timestepsecs=86400\nstarttime="2000-01-01"\nendtime="2000-01-03"\n'
        '[model]\ncold_start__flag=true\nriver_routing="kinematic_wave"\n'
        f'[input]\npath_static="staticmaps.nc"\npath_forcing="{layout.forcing}"\n'
        '[output.csv]\npath="output.csv"\n[[output.csv.column]]\n'
        'header="Q"\nmap="gauges"\nparameter="river_water__volume_flow_rate"\n',
        encoding="utf-8",
    )
    native = model / layout.discharge_csv
    pd.DataFrame(
        {"Q_1010": [1.25, 2.0], "Q_1020": [3.0, 4.0]},
        index=pd.date_range("2000-01-02", periods=2),
    ).to_csv(native, index_label="time")
    publish_response_tables(
        NativeRunArtifacts(native, toml_path), model / layout.run_dir
    )
    registry = tmp_path / "location_registry.csv"
    registry.write_text(
        "wflow_id,subbasin_id,is_primary,station_name\n"
        "1010,101,True,Upstream\n1020,102,True,Outlet\n"
    )
    write_forcing_record(
        toml_path, model / layout.config_dir / "historical_forcing.json", "era5", None
    )
    return model, toml_path, registry


def test_a_recipe_run_is_read_in_place(historical_model):
    model, _, registry = historical_model
    record, frame, stations, files = read_recipe_run(model, "era5", registry)
    assert record["recipe"]["precip_source"] == "era5"
    assert record["stations"] == ["1010", "1020"] == sorted(frame.columns)
    assert stations.wflow_id.tolist() == ["1010", "1020"]
    assert record["units"] == "m3 s-1"
    _check_inputs(files)


def test_a_run_changed_during_use_is_refused(historical_model):
    model, _, registry = historical_model
    *_, files = read_recipe_run(model, "era5", registry)
    record = model / HistoricalLayout("era5").response_record
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="changed during"):
        _check_inputs(files)


def test_forcing_that_differs_from_its_record_is_refused(historical_model):
    model, _, registry = historical_model
    (model / HistoricalLayout("era5").forcing).write_bytes(b"other forcing")
    with pytest.raises(ValueError, match="forcing differs from its record"):
        read_recipe_run(model, "era5", registry)


def test_model_identity_excludes_forcing_but_includes_physics(historical_model):
    model, toml, registry = historical_model
    original = toml.read_text()
    _, identity = execution_inputs(toml, registry)
    (model / HistoricalLayout("era5").forcing).write_bytes(b"different forcing")
    _, changed = execution_inputs(toml, registry)
    assert identity["model_sha256"] == changed["model_sha256"]
    assert identity["forcing_sha256"] != changed["forcing_sha256"]
    toml.write_text(
        original.replace(
            'river_routing="kinematic_wave"', 'river_routing="local_inertial"'
        )
    )
    assert (
        execution_inputs(toml, registry)[1]["model_sha256"] != identity["model_sha256"]
    )


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
    _, toml, registry = historical_model
    original = toml.read_text()
    section = "\n[input.forcing.atmosphere_water__precipitation_volume_flux]\n"
    toml.write_text(original + section + 'name="precip"\nscale=1.0\noffset=0.0\n')
    baseline = execution_inputs(toml, registry)[1]["model_sha256"]
    toml.write_text(original + section + binding + "\n")
    assert execution_inputs(toml, registry)[1]["model_sha256"] != baseline


def test_cold_state_pointer_is_unused_but_warm_state_is_required(historical_model):
    _, path, registry = historical_model
    original = path.read_text() + '\n[state]\npath_input="missing.nc"\n'
    path.write_text(original)
    execution_inputs(path, registry)
    path.write_text(original.replace("cold_start__flag=true", "cold_start__flag=false"))
    with pytest.raises(FileNotFoundError, match="missing executed input"):
        execution_inputs(path, registry)


def test_the_retired_historical_run_setting_only_warns():
    warn_retired_historical_run(None)
    with pytest.warns(FutureWarning, match="no longer used"):
        warn_retired_historical_run({"name": "era5_reference"})
