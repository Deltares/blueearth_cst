"""Independent arithmetic and admission checks for monthly delta collections."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from pyproj import CRS

from blueearth_cst.experiment.delta_scenarios import (
    apply_factors,
    delta_groups,
    generate_collection,
    monthly_factors,
    scenario_records,
    validate_precipitation_reference,
)
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.experiment.simulation_runs import selected_run_ids


def monthly():
    """Two complete years with known month-dependent values."""
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    return xr.Dataset(
        {
            "temp": ("time", np.tile(np.arange(12.0), 2), {"units": "degC"}),
            "precip": ("time", np.tile(np.arange(1.0, 13.0), 2), {"units": "mm/day"}),
        },
        coords={"time": time},
    )


def test_factors_and_leap_day_application_preserve_grid_and_masks():
    reference = monthly()
    future = reference.copy(deep=True)
    future.temp.data += 2.5
    future.precip.data *= 1.25
    offsets, ratios = monthly_factors(
        reference, future, {"start": 2000, "end": 2001}, {"start": 2000, "end": 2001}
    )
    np.testing.assert_array_equal(offsets, np.full(12, 2.5))
    np.testing.assert_array_equal(ratios, np.full(12, 1.25))
    hist = historical()
    result = apply_factors(hist, offsets, ratios)
    np.testing.assert_array_equal(result.temp.values, hist.temp.values + 2.5)
    np.testing.assert_array_equal(result.precip.values, hist.precip.values * 1.25)
    xr.testing.assert_identical(result.press_msl, hist.press_msl)
    xr.testing.assert_identical(result.time, hist.time)
    assert np.datetime64("2000-02-29") in result.time.values
    xr.testing.assert_identical(apply_factors(hist, np.zeros(12), np.ones(12)), hist)


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "zero", "negative", "nonfinite", "units", "spatial"],
)
def test_invalid_projection_inputs_fail(problem):
    reference, future = monthly(), monthly()
    if problem == "missing":
        future = future.isel(time=slice(1, None))
    elif problem == "duplicate":
        future = future.assign_coords(
            time=list(future.time.values[:-1]) + [future.time.values[-2]]
        )
    elif problem == "zero":
        reference.precip.data[:] = 0
    elif problem == "negative":
        future.precip.data[0] = -1
    elif problem == "nonfinite":
        future.temp.data[0] = np.nan
    elif problem == "units":
        future.temp.attrs["units"] = "K"
    else:
        future = future.expand_dims(station=[1, 2])
    with pytest.raises(ValueError):
        monthly_factors(
            reference,
            future,
            {"start": 2000, "end": 2001},
            {"start": 2000, "end": 2001},
        )


@pytest.mark.parametrize("units", ["mm/day", "mm/month"])
@pytest.mark.parametrize("daily_rate", [0.099, 0.1, 0.101])
def test_dry_month_admission_uses_daily_units_and_strict_boundary(units, daily_rate):
    reference = monthly()
    reference.precip.data[:] = daily_rate
    if units == "mm/month":
        reference.precip.data *= reference.time.dt.days_in_month.values
    reference.precip.attrs["units"] = units
    policy = {"min_denominator": 0.1, "units": "mm/day"}
    if daily_rate < 0.1:
        with pytest.raises(ValueError, match="months 01, 02.*refused, not clipped"):
            validate_precipitation_reference(
                reference, {"start": 2000, "end": 2001}, policy
            )
    else:
        validate_precipitation_reference(
            reference, {"start": 2000, "end": 2001}, policy
        )


def test_dry_month_admission_refuses_unknown_projection_units():
    reference = monthly()
    reference.precip.attrs["units"] = "unverified"
    with pytest.raises(ValueError, match="mm/day or mm/month"):
        validate_precipitation_reference(
            reference,
            {"start": 2000, "end": 2001},
            {"min_denominator": 0.1, "units": "mm/day"},
        )


def historical():
    """Small georeferenced daily input containing a leap day and missing cell."""
    times = pd.date_range("2000-01-01", "2000-12-31")
    data = np.full((2, len(times), 2), 10, dtype="float32")
    data[0, :, 0] = np.nan
    ds = xr.Dataset(
        {
            name: (("y", "time", "x"), data.copy())
            for name in ("temp", "temp_min", "temp_max", "precip", "press_msl")
        },
        coords={"time": times, "x": [0.0, 1.0], "y": [1.0, 0.0]},
    )
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    return ds


def settings():
    """Explicit synthetic ensemble: two members of A and one of B."""
    return {
        "schema_version": "delta-change/1",
        "historical_path": "historical.nc",
        "source_like": "era5",
        "reader": {
            "metadata": {
                "cst_unit_interpretation": {
                    "revision": "fixture/1",
                    "evidence": "analytical test fixture",
                    "variables": [
                        [name, unit]
                        for name, unit in (
                            ("temp", "degC"),
                            ("temp_min", "degC"),
                            ("temp_max", "degC"),
                            ("precip", "mm/day"),
                            ("press_msl", "hPa"),
                        )
                    ],
                }
            }
        },
        "reference_period": {"start": 2000, "end": 2001},
        "future_period": {"start": 2000, "end": 2001},
        "simulation_window": {"start": 2000, "end": 2000},
        "unit_id_capacity": 1000,
        "members": [
            {
                "model": model,
                "scenario": "ssp245",
                "member": member,
                "reference_path": "reference.nc",
                "future_path": "future.nc",
            }
            for model, member in (("A", "r1"), ("A", "r2"), ("B", "r1"))
        ],
    }


def test_ensemble_groups_separate_models_scenarios_and_controls():
    config = settings()
    config["members"].append({**config["members"][0], "scenario": "ssp585"})
    rows = scenario_records(config)
    groups, references = delta_groups(rows)
    assert references == ("run_0001",)
    assert sorted(groups.values()) == [
        ("run_0001",),
        ("run_0002",),
        ("run_0003", "run_0004"),
        ("run_0005",),
        ("run_0006",),
    ]
    from blueearth_cst.experiment.indicator_groups import group_metadata

    fields, metadata = group_metadata(
        {key: f"group_{i:04d}" for i, key in enumerate(sorted(groups), 1)}, delta=True
    )
    assert fields == ["group_id", "scenario_key", "is_reference"]
    assert [row["group_id"] for row in metadata if row["is_reference"] == "true"] == [
        "group_0001"
    ]


def test_collection_roundtrip_reuse_selection_and_corruption(tmp_path):
    historical().to_netcdf(tmp_path / "historical.nc")
    monthly().to_netcdf(tmp_path / "reference.nc")
    future = monthly()
    future.temp.data += 3
    future.to_netcdf(tmp_path / "future.nc")
    config = tmp_path / "delta.yml"
    config.write_text(yaml.safe_dump(settings()), encoding="utf-8")
    project = tmp_path / "project"
    manifest = generate_collection(config, project)
    marker = read_collection_v2(manifest)
    intent = json.loads(manifest.with_name("collection_intent.json").read_text())
    assert intent["schema_version"] == "delta-scenario-intent/2"
    assert intent["factor_policy"] == {
        "precip": {"min_denominator": 0.1, "units": "mm/day"}
    }
    from blueearth_cst.experiment.simulation_runner import (
        verify_selected_collection_source,
    )

    project_config = {
        "project": {"project_dir": str(project)},
        "climate": {"selected": "era5"},
    }
    selection = {"manifest_path": str(manifest)}
    assert (
        verify_selected_collection_source(project_config, selection, marker) == "era5"
    )
    project_config["climate"]["selected"] = "chirps"
    with pytest.raises(ValueError, match="climate source differs"):
        verify_selected_collection_source(project_config, selection, marker)
    assert len(marker["series"]) == 5
    assert selected_run_ids(marker, manifest, include=False) == [
        "run_0003",
        "run_0004",
        "run_0005",
    ]
    before = manifest.read_bytes()
    assert generate_collection(config, project) == manifest
    assert manifest.read_bytes() == before
    with (
        xr.open_dataset(project / marker["series"][0]["file"]["path"]) as control,
        xr.open_dataset(project / marker["series"][1]["file"]["path"]) as neutral,
    ):
        xr.testing.assert_identical(control, neutral)
    (project / marker["series"][2]["file"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes differ"):
        read_collection_v2(manifest)
    with pytest.raises(ValueError, match="bytes differ"):
        generate_collection(config, project)


def test_duplicate_member_identity_is_rejected():
    from blueearth_cst.experiment.delta_scenarios import validate_settings

    config = settings()
    config["members"].append(deepcopy(config["members"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_settings(config)


def test_generation_refuses_dry_reference_before_creating_collection(tmp_path):
    historical().to_netcdf(tmp_path / "historical.nc")
    reference = monthly()
    reference.precip.data[[0, 12]] = 0.01
    reference.to_netcdf(tmp_path / "reference.nc")
    monthly().to_netcdf(tmp_path / "future.nc")
    config = tmp_path / "delta.yml"
    config.write_text(yaml.safe_dump(settings()), encoding="utf-8")
    with pytest.raises(ValueError, match="months 01; relative factors are refused"):
        generate_collection(config, tmp_path / "project")
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize(
    ("variable", "units"),
    [
        ("temp", "K"),
        ("temp_min", "K"),
        ("temp_max", "K"),
        ("precip", "m/day"),
        ("press_msl", "Pa"),
        ("kin", "J/m2"),
        ("kout", "J/m2"),
    ],
)
def test_historical_units_cannot_be_relabelled_by_wf4(variable, units):
    from blueearth_cst.experiment.delta_scenarios import validate_settings

    config = settings()
    binding = config["reader"]["metadata"]["cst_unit_interpretation"]
    binding["variables"] = [
        pair for pair in binding["variables"] if pair[0] != variable
    ] + [[variable, units]]
    with pytest.raises(ValueError, match="historical effective units"):
        validate_settings(config)
