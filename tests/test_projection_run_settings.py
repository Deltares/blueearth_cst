"""Bounded console lists and scenario-independent ensemble counts."""

from blueearth_cst.projections.resolution import RESOLVED, Combination
from blueearth_cst.projections.run_settings import projection_run_settings
from blueearth_cst.projections.variable_spec import parse


def test_settings_count_members_once_and_describe_selected_methods():
    models = ["A/GFDL", "B/INM", "C/MPI"]
    scenarios = ["ssp245", "ssp585"]
    combinations = [
        Combination(model, scenario, f"r{member}", RESOLVED)
        for model, count in zip(models, [1, 1, 3], strict=True)
        for scenario in scenarios
        for member in range(1, count + 1)
    ]
    settings = projection_run_settings(
        models,
        scenarios,
        combinations,
        parse(
            {
                "precip": {
                    "source": "precip",
                    "canonical": "rate",
                    "units": "mm/day",
                    "change": "relative",
                },
                "temp": {
                    "source": "temp",
                    "canonical": "state",
                    "units": "degC",
                    "change": "absolute",
                },
                "pet": {"method": "debruin"},
                "tdew": {"method": "specific_humidity"},
                "wind": {"method": "sfcwind"},
            }
        ),
    )
    assert settings["climate models"] == "GFDL, INM, MPI (3 total)"
    assert settings["ensemble members"] == "GFDL: 1; INM: 1; MPI: 3 (5 total)"
    assert settings["variables"] == "precip, temp, pet, tdew, wind"
    assert settings["PET method"] == "de Bruin"
    assert settings["wind statistic"] == "Monthly mean wind speed at 10 m"


def test_large_ensemble_reports_omissions_and_scenario_variation():
    models = [f"I/M{i}" for i in range(20)]
    combinations = [Combination(model, "ssp245", "r1", RESOLVED) for model in models]
    settings = projection_run_settings(
        models,
        ["ssp245", "ssp585"],
        combinations,
        parse({"wind": {"method": "components"}}),
    )
    assert settings["climate models"] == "M0, M1, M2 (+17 more) (20 total)"
    assert settings["ensemble members"] == (
        "20 model–member pairs; 1 member per model; varies by scenario"
    )
    assert settings["wind statistic"] == "Magnitude of monthly mean wind vector at 10 m"
    assert "PET method" not in settings
