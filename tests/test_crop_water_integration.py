"""Crop-water publication over a synthetic retained WF4 simulation."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment import indicator_sets
from blueearth_cst.experiment.content_identity import read_canonical_json
from blueearth_cst.experiment.indicator_plan import (
    build_indicator_plan,
    current_indicator_request,
    publish_indicator_set,
    read_indicator_set,
)
from blueearth_cst.experiment.indicator_sets import SetSource
from blueearth_cst.indicators.series import ResponseSeries
from blueearth_cst.indicators.sets.crop_water.calculations import prepare_baseline_qcrit
from tests._v2_experiment import (
    FIXTURE_ENVIRONMENT,
    RUN_IDS,
    Responses,
    build_v2_experiment,
)

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


def _responses(locations=("9",), *, without_pet=False):
    """Retained responses at ``locations``; Qcrit is prepared at location 9 only.

    ``without_pet`` omits PET, as a simulation that never requested it.
    """
    drop = {"pet"} if without_pet else set()
    from datetime import timedelta

    time = pd.date_range("2046-10-01", "2047-10-31", freq="D")
    rows = {}
    for run, scale in zip(RUN_IDS, (1.0, 2.0), strict=True):
        rows[run] = (
            pd.DataFrame(
                {
                    f"{header}_{location}": np.full(len(time), value)
                    for location in locations
                    for header, value in (
                        ("Q", scale),
                        ("p", 2.0),
                        ("pet", 3.0),
                        ("aet", 1.0),
                        ("gwr", 2.0),
                    )
                    if header not in drop
                },
                index=time,
            )
            .to_csv(index_label="time")
            .encode()
        )
    names = {
        "Q": "river_water__volume_flow_rate",
        "p": "atmosphere_water__precipitation_volume_flux",
        "pet": "land_surface_water__potential_evaporation_volume_flux",
        "aet": "land_surface__evapotranspiration_volume_flux",
        "gwr": "soil_water_saturated_zone_top__net_recharge_volume_flux",
    }
    toml = (
        '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
        'starttime="2046-09-30T00:00:00"\nendtime="2047-10-31T00:00:00"\n'
        + "".join(
            f'[[output.csv.column]]\nheader="{header}"\nmap="subcatchment"\n'
            f'reducer="mean"\nparameter="{parameter}"\n'
            for header, parameter in names.items()
            if header not in drop
        )
    )
    temporal = {
        "source_calendar": "proleptic_gregorian",
        "prepared_forcing_calendar": "proleptic_gregorian",
        "response_calendar": "standard",
        "operations": ["clip_to_configured_window", "refresh_toml_endpoints"],
        "prepared_start": "2046-09-30T00:00:00",
        "prepared_end": str(time[-1]),
        "response_start": str(time[0]),
        "response_end": str(time[-1]),
        "time_label": "interval_end",
    }
    variables = [
        {
            "variable": variable,
            "locations": list(locations),
            "units": units,
            "calendar": "standard",
            "timestep": "P1D",
            "time_label": "interval_end",
            "start": str(time[0]),
            "end": str(time[-1]),
            "missing_value": "NaN",
        }
        for variable, units in (
            ("q", "m3 s-1"),
            ("precip", "mm dt-1"),
            ("pet", "mm dt-1"),
            ("aet", "mm dt-1"),
            ("gwr", "mm dt-1"),
        )
        if variable not in drop
    ]
    baseline = ResponseSeries(
        run_id="historical",
        variable="q",
        location_id="9",
        location_ordinal=0,
        time=time,
        calendar="standard",
        timestep=timedelta(days=1),
        time_label="interval_end",
        units="m3 s-1",
        values=np.full(len(time), 2.0),
        missing=np.zeros(len(time), dtype=bool),
    )
    thresholds = prepare_baseline_qcrit(
        [baseline], source_id="historical-fixture", years=[2047]
    )
    return Responses(rows, toml, temporal, variables), thresholds


@pytest.fixture
def crop_water_experiment(tmp_path, monkeypatch):
    from blueearth_cst.experiment import indicator_plan

    responses, _ = _responses()
    experiment = build_v2_experiment(tmp_path, responses)
    historical = tmp_path / "historical-model"
    (historical / "run_default").mkdir(parents=True)
    pd.DataFrame({"Q_9": 2.0}, index=pd.date_range("1999-01-01", "1999-12-31")).to_csv(
        historical / "run_default/output.csv", index_label="time"
    )
    (historical / "wflow_sbm.toml").write_text(
        '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
        'starttime="1998-12-31T00:00:00"\nendtime="1999-12-31T00:00:00"\n'
        '[input]\npath_forcing="forcing/inmaps_historical.nc"\n'
        '[[output.csv.column]]\nheader="Q"\nparameter="river_water__volume_flow_rate"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    source = SetSource(
        "crop-water",
        tmp_path / "crop-water.yml",
        False,
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "crop-water",
                "title": "Crop-water",
                "description": "Synthetic complete set",
                "selection": {
                    "registry": {
                        "set": "crop-water",
                        "overrides": {
                            "summer_low_flow_days": {
                                "season": "summer",
                                "thresholds": {
                                    "baseline_quantile": {
                                        "model_root": historical.as_posix(),
                                        "source_id": "historical-fixture",
                                        "years": [1999],
                                        "reference_period": "annual",
                                    }
                                },
                            }
                        },
                    }
                },
            }
        ),
    )
    source.path.write_text(source.text, encoding="utf-8")
    workflow = experiment.config.parent / "simulate_system.yml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment.root.name,
                "operations": ["indicators"],
                "indicator_sets": ["crop-water"],
                "indicator_set_definitions": {
                    "crop-water": {"config_path": source.path.name}
                },
            }
        ),
        encoding="utf-8",
    )
    return experiment, source


def _run_indicators(experiment, *extra):
    """Invoke WF4 against the configured synthetic retained responses."""
    environment = os.environ.copy()
    environment.pop("SNAKEMAKE_WORKFLOW_PROFILE", None)
    return subprocess.run(
        [
            sys.executable,
            "scripts/simulate_system.py",
            "--config",
            str(experiment.config),
            "--cores",
            "1",
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=1200,
    )


@pytest.mark.skipif(
    os.name != "nt", reason="run_project_child refuses project writers off Windows"
)
@pytest.mark.parametrize("inline", [False, True])
def test_crop_water_cli_publication(crop_water_experiment, inline):
    experiment, source = crop_water_experiment
    if inline:
        workflow = experiment.config.parent / "simulate_system.yml"
        settings = yaml.safe_load(workflow.read_text())
        settings.pop("indicator_set_definitions")
        overrides = yaml.safe_load(source.text)["selection"]["registry"]["overrides"]
        settings["indicator_sets"] = [
            {
                "set": "crop-water",
                **overrides,
                "winter_longest_dry_spell": {"enabled": False},
                "summer_longest_dry_spell": {"dry_day_mm": 2.0},
            }
        ]
        workflow.write_text(yaml.safe_dump(settings), encoding="utf-8")
    run = _run_indicators(experiment)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (
        experiment.root / "results/indicator_sets/crop-water/indicators.csv"
    ).is_file()
    dictionary = pd.read_csv(
        experiment.root / "results/indicator_sets/crop-water/indicators.csv"
    )
    assert len(dictionary) == (9 if inline else 10)
    assert set(dictionary.water_year_start) == {"Oct (recharge); fixed seasons"}
    assert dictionary.block_assignment.str.contains("1 Nov-31 May").all()


def test_crop_water_retained_response_publication(crop_water_experiment):
    experiment, source = crop_water_experiment
    run = _run_indicators(experiment, "--dry-run")
    assert run.returncode == 0, run.stdout + run.stderr
    resolved = indicator_sets.resolve_set(source)
    assert len(resolved.indicators) == 10
    request = current_indicator_request(
        experiment.root,
        indicator_sets.calculation_records(resolved.indicators),
        "YE-SEP",
    )
    plan = build_indicator_plan(experiment.root, request)
    manifest = publish_indicator_set(experiment.root, plan)
    marker = plan["targets"]["manifest"]
    assert read_indicator_set(experiment.root, marker) == manifest
    assert {row["name"] for row in manifest["declarations"]} == {
        indicator.indicator for indicator in resolved.indicators
    }
    assert set(plan["request"]["response_variables"]) == {
        "q",
        "precip",
        "pet",
        "aet",
        "gwr",
    }
    low_flow = next(
        row for row in manifest["declarations"] if row["name"] == "summer_low_flow_days"
    )
    assert low_flow["calculation_parameters"]["thresholds"]["9"][
        "source_id"
    ].startswith("historical-fixture;")
    diagnostics = read_canonical_json(
        Path(marker).parent / manifest["diagnostics"]["path"]
    )
    member = next(
        item
        for item in diagnostics["results"]
        if item["indicator"] == "summer_climatic_deficit" and item["unit"] == RUN_IDS[0]
    )
    periods = next(
        table for table in member["tables"] if table["name"] == "period_values"
    )
    assert any(
        row["period"] == 2047 and row["status"] == "valid" for row in periods["rows"]
    )

    changed_text = source.text.replace("historical-fixture", "different-history")
    changed = indicator_sets.resolve_set(
        SetSource("crop-water", source.path, False, changed_text)
    )
    changed_request = current_indicator_request(
        experiment.root,
        indicator_sets.calculation_records(changed.indicators),
        "YE-SEP",
    )
    assert (
        build_indicator_plan(experiment.root, changed_request)["indicator_set_id"]
        != plan["indicator_set_id"]
    )


def test_secondary_pet_bytes_change_calculation_identity(tmp_path, monkeypatch):
    from blueearth_cst.experiment import indicator_plan

    responses, _ = _responses()
    first = build_v2_experiment(tmp_path / "first", responses)
    altered_csv = dict(responses.csv)
    old = altered_csv[RUN_IDS[0]]
    altered_csv[RUN_IDS[0]] = old.replace(b"2.0,3.0,1.0,2.0", b"2.0,4.0,1.0,2.0", 1)
    assert altered_csv[RUN_IDS[0]] != old
    second = build_v2_experiment(
        tmp_path / "second",
        Responses(altered_csv, responses.toml, responses.temporal, responses.variables),
    )
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    source = indicator_sets.resolve_set(
        SetSource(
            "crop-water",
            tmp_path / "crop-water.yml",
            False,
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "id": "crop-water",
                    "title": "Crop-water",
                    "description": "Secondary-input identity probe",
                    "selection": {"registry": "crop-water"},
                }
            ),
        )
    )
    records = indicator_sets.calculation_records(source.indicators)
    plans = [
        build_indicator_plan(
            experiment.root,
            current_indicator_request(experiment.root, records, "YE-SEP"),
        )
        for experiment in (first, second)
    ]
    assert (
        plans[0]["identity_projection"]["response_identity_sha256"]
        != plans[1]["identity_projection"]["response_identity_sha256"]
    )
    assert plans[0]["indicator_set_id"] != plans[1]["indicator_set_id"]
