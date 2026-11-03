"""Air temperature survives native reading and uses state reductions end to end."""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.experiment.indicator_registry import declarations, reduce_run
from blueearth_cst.experiment.wflow_response_reader import (
    NativeRunArtifacts,
    ResponseRequest,
    open_responses,
)
from blueearth_cst.indicators import loader
from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS
from blueearth_cst.shared.func_plot_signature import _basavg_statistics
from blueearth_cst.shared.indicator_tables import indicator_tables
from blueearth_cst.shared.wflow_outputs import CODES, PLOT_META
from tests._v2_experiment import (
    FIXTURE_ENVIRONMENT,
    Responses,
    build_v2_experiment,
    default_responses,
)


def test_temperature_native_binding_default_indicator_and_plot_aggregation(tmp_path):
    assert WFLOW_VARS["temperature"] == "atmosphere_air__temperature"
    assert CODES["temperature"] == "temp"
    assert "temp" in indicator_tables(["temperature"])
    time = pd.date_range("2001-01-01", "2001-12-31")
    values = np.full(len(time), -5.0)
    values[time.month >= 7] = 15.0
    csv_path = tmp_path / "output.csv"
    pd.DataFrame({"temp_9": values}, index=time).to_csv(csv_path, index_label="time")
    toml_path = tmp_path / "model.toml"
    toml_path.write_text(
        '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
        'starttime="2000-12-31T00:00:00"\nendtime="2001-12-31T00:00:00"\n'
        '[[output.csv.column]]\nheader="temp"\nmap="subcatchment"\nreducer="mean"\n'
        'parameter="atmosphere_air__temperature"\n',
        encoding="utf-8",
    )
    responses = open_responses(
        "run", NativeRunArtifacts(csv_path, toml_path), ResponseRequest(("temp",))
    )
    assert responses[0].units == "degC"
    np.testing.assert_array_equal(responses[0].values, values)
    (declaration,) = declarations(["temp"])
    assert declaration.name == "temp_annual_mean"
    expected = (-5 * 181 + 15 * 184) / 365
    result = reduce_run(declaration, responses, anchor="YE-DEC")
    assert result["9"] == pytest.approx(expected)
    local = next(
        d for d in loader.declarations("default") if d.name == declaration.name
    )
    local_result = loader.evaluate(local, responses, anchor="YE-DEC")
    loader.check(local, responses, local_result, anchor="YE-DEC")
    assert local_result.values["9"] == pytest.approx(expected)
    assert PLOT_META["temp"]["resample"] == "mean"
    assert PLOT_META["temp"]["annual_units"] == "°C"
    stats = _basavg_statistics(pd.Series(values, index=time), PLOT_META["temp"]["kind"])
    assert stats[0].loc[2001] == pytest.approx(expected)
    assert stats[1].loc[1, "mean"] == -5
    # A temperature output must refuse a spatial sum as well as wrong physical parameters.
    toml_path.write_text(
        toml_path.read_text().replace('reducer="mean"', 'reducer="sum"')
    )
    with pytest.raises(ValueError, match="subcatchment mean"):
        open_responses(
            "run", NativeRunArtifacts(csv_path, toml_path), ResponseRequest(("temp",))
        )


def test_temperature_retained_inventory_planning_and_publication(tmp_path, monkeypatch):
    from blueearth_cst.experiment import indicator_plan
    from blueearth_cst.experiment.content_identity import identity_segment
    from blueearth_cst.experiment.indicator_sets import default_calculations

    original = default_responses()
    native = Responses(
        csv={
            run: data.replace(b"gwr_9", b"temp_9") for run, data in original.csv.items()
        },
        toml=original.toml.replace(
            'header="gwr"', 'header="temp"\nmap="subcatchment"\nreducer="mean"'
        ).replace(
            "soil_water_saturated_zone_top__net_recharge_volume_flux",
            "atmosphere_air__temperature",
        ),
        temporal=original.temporal,
        variables=[{**original.variables[0], "variable": "temp", "units": "degC"}],
    )
    experiment = build_v2_experiment(tmp_path, native)
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    request = indicator_plan.current_indicator_request(
        experiment.root, default_calculations(["temp"]), "YE-DEC"
    )
    plan = indicator_plan.build_indicator_plan(experiment.root, request)
    manifest = indicator_plan.publish_indicator_set(experiment.root, plan)
    engine = (
        experiment.root
        / "_engine/indicator_sets"
        / identity_segment(plan["indicator_set_id"], "indicator_set_id")
    )
    assert (
        indicator_plan.read_indicator_set(experiment.root, engine / "indicators.json")
        == manifest
    )
    for scope in ("run", "group"):
        table = pd.read_csv(engine / "data" / f"temp_{scope}_indicators.csv")
        assert (table.value == 4.5).all()


def test_temperature_default_refuses_depth_sums():
    with pytest.raises(ValueError, match="statistic"):
        loader.resolve(
            "default", "annual_summary/1", {"variable": "temp", "statistic": "sum"}
        )
