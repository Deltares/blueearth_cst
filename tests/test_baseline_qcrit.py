"""WF4 prepares historical thresholds and seals their numerical provenance."""

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment.baseline_qcrit import (
    HistoricalQcritUnavailable,
    resolve_thresholds,
)
from blueearth_cst.experiment.indicator_sets import (
    SetSource,
    calculation_records,
    resolve_set,
)


@pytest.fixture
def baseline(tmp_path):
    model = tmp_path / "model"
    (model / "run_default").mkdir(parents=True)
    time = pd.date_range("1999-01-01", "1999-12-31")
    pd.DataFrame({"Q_9": np.arange(365)}, index=time).to_csv(
        model / "run_default/output.csv", index_label="time"
    )
    (model / "wflow_sbm.toml").write_text(
        '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
        'starttime="1998-12-31T00:00:00"\nendtime="1999-12-31T00:00:00"\n'
        '[input]\npath_forcing="forcing/inmaps_historical.nc"\n'
        '[[output.csv.column]]\nheader="Q"\nparameter="river_water__volume_flow_rate"\n',
        encoding="utf-8",
    )
    parameters = {
        "season": "summer",
        "thresholds": {
            "baseline_quantile": {
                "model_root": "model",
                "source_id": "historical-era5",
                "years": [1999],
                "reference_period": "annual",
            }
        },
    }
    return tmp_path / "crop-water.yml", parameters


def source(path, parameters):
    return SetSource(
        "crop-water",
        path,
        False,
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "crop-water",
                "title": "Case",
                "description": "Historical Qcrit",
                "selection": {
                    "registry": {
                        "set": "crop-water",
                        "overrides": {"summer_low_flow_days": parameters},
                    }
                },
            }
        ),
    )


def test_inline_q90_binds_project_model_and_inherits_summer(baseline):
    from blueearth_cst.experiment.indicator_sets import parse_selection, selected_sets

    path, _ = baseline
    model = path.parent / "models/hydrology/wflow"
    model.parent.mkdir(parents=True)
    (path.parent / "model").rename(model)
    settings = {
        "indicator_sets": [
            {
                "set": "crop-water",
                "summer_low_flow_days": {"thresholds": {"baseline_quantile": {}}},
            }
        ]
    }
    selection = parse_selection(settings, path)
    kwargs = dict(
        simulation_path=True,
        retained_variables=lambda: [],
        legacy_context="simulation_rule",
    )
    project = {"project": {"project_dir": str(path.parent)}}
    resolved = selected_sets(project, settings, selection, **kwargs)
    low_flow = next(
        i for i in resolved[0].indicators if i.indicator == "summer_low_flow_days"
    )
    assert low_flow.parameters["season"] == "summer"
    threshold = low_flow.parameters["thresholds"]["9"]
    # May 1 (index 120) through October 31 (303): p=.10 -> 138.3.
    assert threshold["value"] == pytest.approx(138.3)
    assert threshold["years"] == [1999]
    assert threshold["year_selection"] == "all"
    assert threshold["source_id"].startswith(
        "wf1-historical-model-discharge;csv-sha256="
    )
    (model / "run_default/output.csv").unlink()
    pending = selected_sets(
        project, settings, selection, prepare_baseline=False, **kwargs
    )
    with pytest.raises(ValueError, match="pending"):
        calculation_records(pending[0].indicators)
    with pytest.raises(HistoricalQcritUnavailable, match="historical discharge CSV"):
        selected_sets(project, settings, selection, **kwargs)


def test_owned_set_resolution_prepares_fixed_thresholds_and_changes_identity(baseline):
    path, parameters = baseline
    resolved = resolve_set(source(path, parameters))
    records = calculation_records(resolved.indicators)
    item = next(i for i in resolved.indicators if i.indicator == "summer_low_flow_days")
    threshold = item.declaration.calculation_parameters["thresholds"]["9"]
    assert threshold["value"] == pytest.approx(36.4)
    assert "csv-sha256=" in threshold["source_id"]
    assert "toml-sha256=" in threshold["source_id"]
    parameters["thresholds"]["baseline_quantile"]["quantile"] = 0.8
    assert (
        calculation_records(resolve_set(source(path, parameters)).indicators) != records
    )
    parameters["thresholds"]["baseline_quantile"]["quantile"] = 0.9
    parameters["thresholds"]["baseline_quantile"]["reference_period"] = "summer"
    assert (
        calculation_records(resolve_set(source(path, parameters)).indicators) != records
    )
    parameters["thresholds"]["baseline_quantile"]["reference_period"] = "annual"
    csv_path = path.parent / "model/run_default/output.csv"
    csv_path.write_text(csv_path.read_text().replace(",0", ",0.5", 1), encoding="utf-8")
    assert (
        calculation_records(resolve_set(source(path, parameters)).indicators) != records
    )


@pytest.mark.parametrize(
    "change", ["future", "observations", "missing", "unknown", "bad_years"]
)
def test_baseline_rule_refuses_incompatible_inputs(baseline, change):
    path, parameters = baseline
    rule = parameters["thresholds"]["baseline_quantile"]
    if change in ("future", "observations"):
        toml_path = path.parent / "model/wflow_sbm.toml"
        toml_path.write_text(
            toml_path.read_text().replace("inmaps_historical.nc", f"{change}.nc")
        )
    elif change == "missing":
        rule["model_root"] = "absent"
    elif change == "unknown":
        rule["percentile"] = 90
    else:
        rule["years"] = "1999"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_thresholds(parameters, path)


@pytest.mark.parametrize("years", [None, "all", [1999]])
def test_discovered_and_explicit_years_seal_selection(baseline, years):
    path, parameters = baseline
    rule = parameters["thresholds"]["baseline_quantile"]
    if years is None:
        del rule["years"]
    else:
        rule["years"] = years
    record = resolve_thresholds(parameters, path)["thresholds"]["9"]
    assert record["years"] == record["discovered_years"] == [1999]
    assert record["year_selection"] == (
        "explicit" if isinstance(years, list) else "all"
    )
    assert record["value"] == pytest.approx(36.4)


def test_static_resolution_validates_without_historical_io_or_identity(
    baseline, monkeypatch
):
    from blueearth_cst.experiment import baseline_qcrit

    path, parameters = baseline
    monkeypatch.setattr(
        baseline_qcrit, "file_sha256", lambda _: pytest.fail("historical read")
    )
    resolved = resolve_set(source(path, parameters), prepare_baseline=False)
    assert any(item.pending_baseline for item in resolved.indicators)
    with pytest.raises(ValueError, match="identity is unavailable"):
        calculation_records(resolved.indicators)
    parameters["thresholds"]["baseline_quantile"]["quantile"] = 2
    with pytest.raises(ValueError, match="invalid Qcrit"):
        resolve_set(source(path, parameters), prepare_baseline=False)


@pytest.mark.parametrize("missing", ["run_default/output.csv", "wflow_sbm.toml"])
@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
@pytest.mark.parametrize("inline", [False, True])
def test_direct_wf4_reports_missing_historical_prerequisite(
    baseline, missing, extra, capsys, inline
):
    from scripts.simulate_system import main

    path, parameters = baseline
    path.write_text(source(path, parameters).text, encoding="utf-8")
    (path.parent / "model" / missing).unlink()
    workflow = path.parent / "wf4.yml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "operations": ["simulations", "indicators"],
                "indicator_sets": ["crop-water"],
                "indicator_set_definitions": {"crop-water": {"config_path": path.name}},
                "indicator_selection": {
                    "crop-water": {"include": ["summer_low_flow_days"]}
                },
            }
        ),
        encoding="utf-8",
    )
    project = path.parent / "project.yml"
    project.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": str(path.parent / "run")},
                "model": {"outvars": ["river discharge"]},
                "workflows": {
                    "simulate_system": {"enabled": True, "config_path": workflow.name}
                },
            }
        ),
        encoding="utf-8",
    )
    if inline:
        model = path.parent / "run/models/hydrology/wflow"
        model.parent.mkdir(parents=True)
        (path.parent / "model").rename(model)
        settings = yaml.safe_load(workflow.read_text())
        settings.pop("indicator_set_definitions")
        settings["indicator_sets"] = [
            {
                "set": "crop-water",
                "summer_low_flow_days": {"thresholds": {"baseline_quantile": {}}},
            }
        ]
        workflow.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--config", str(project), *extra])
    assert error.value.code == 2
    message = capsys.readouterr().err.replace("\\", "/")
    assert (
        missing in message
        and "WF1" in message
        and "historical simulation before WF4" in message
    )
    assert "FileNotFoundError" not in message


def test_corrupt_existing_toml_is_not_a_pending_prerequisite(baseline):
    path, parameters = baseline
    (path.parent / "model/run_default/output.csv").unlink()
    (path.parent / "model/wflow_sbm.toml").write_text("[broken", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        resolve_thresholds(parameters, path)
    assert not isinstance(error.value, HistoricalQcritUnavailable)


def test_all_year_spellings_share_identity_but_explicit_intent_differs(baseline):
    path, parameters = baseline
    rule = parameters["thresholds"]["baseline_quantile"]
    explicit = calculation_records(resolve_set(source(path, parameters)).indicators)
    del rule["years"]
    omitted = calculation_records(resolve_set(source(path, parameters)).indicators)
    rule["years"] = "all"
    assert (
        calculation_records(resolve_set(source(path, parameters)).indicators) == omitted
    )
    assert explicit != omitted
