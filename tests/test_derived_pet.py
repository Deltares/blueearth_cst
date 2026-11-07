"""Calculated PET config, native HydroMT transport and independent numerics."""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.projections.derived_pet import (
    INPUT_CONTRACT,
    admit_monthly_sources,
    calculate_pet,
    register_pet_preprocess,
)
from blueearth_cst.projections.fetch_gcm_raw import (
    clip_to_bbox,
    fetch_raw_slice,
    read_variable_slices,
)
from blueearth_cst.projections.get_stats_climate_proj import get_stats_clim_projections
from blueearth_cst.projections.variable_spec import parse, source_names


@pytest.mark.parametrize(
    "method,expected",
    [
        ("makkink", ["kin", "press", "temp"]),
        ("debruin", ["kin", "kout", "press", "temp"]),
    ],
)
def test_calculated_spec(method, expected):
    spec = parse({"pet": {"method": method}})
    assert source_names(spec) == expected
    assert spec["pet"].min_denominator == 0.1
    assert spec["pet"].units == "mm/day"
    assert source_names(parse({"temp": None, "pet": {"method": method}})) == expected


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"method": None},
        {"method": True},
        {"method": "typo"},
        {"method": "makkink", "source": "pet"},
        {"method": "makkink", "units": "mm/day"},
        {"method": "makkink", "change": "absolute", "min_denominator": 0.1},
        *(
            {"method": "makkink", "min_denominator": value}
            for value in [None, True, 0, -1, float("inf"), float("nan"), "0.1"]
        ),
    ],
)
def test_invalid_calculated_spec(body):
    with pytest.raises(ValueError, match="variables.pet"):
        parse({"pet": body})


def test_supplied_pet_preserved():
    spec = parse(
        {
            "pet": {
                "source": "evap",
                "canonical": "rate",
                "units": "mm/day",
                "change": "absolute",
            }
        }
    )
    assert spec["pet"].method is None
    assert source_names(spec) == ["evap"]


def native(variable, values, *, units=None, calendar="proleptic_gregorian"):
    """Amon native fields, with metadata checked before catalog conversion."""
    units = (
        units or {"tas": "K", "ps": "Pa", "rsds": "W m-2", "rsdt": "W m-2"}[variable]
    )
    data = np.broadcast_to(np.asarray(values), (2, 2, 2)).copy()
    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), data)},
        coords={
            "time": pd.date_range("2000-01-01", periods=2, freq="MS"),
            "lat": [0.0, 60.0],
            "lon": [0.0, 1.0],
        },
        attrs={
            "table_id": "Amon",
            "institution_id": "TEST",
            "source_id": "MODEL",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "grid_label": "gn",
        },
    )
    ds[variable].attrs.update(units=units, cell_methods="area: time: mean")
    ds.time.encoding["calendar"] = calendar
    return ds


def oracle(temp, press, kin, kout=None):
    """Independent scalar arithmetic, without calling production/HydroMT code."""
    vapour = 6.112 * math.exp(17.67 * temp / (temp + 243.5))
    slope = vapour * 17.269 * 243.5 / (temp + 243.5) ** 2
    latent = 2502000 - 2250 * temp
    fraction = slope / (slope + 1005 * press / (0.622 * latent))
    energy = (
        0.65 * fraction * kin
        if kout is None
        else (
            fraction * (0.77 * kin - 110 * kin / (kout + 0.00001)) + 20 if kout else 0
        )
    )
    return max(0, energy * 86400 / latent)


@pytest.fixture
def transported(tmp_path):
    """Use real HydroMT raster driver, generated adapter and source admission."""
    import geopandas as gpd
    import hydromt  # noqa: F401 -- register raster accessor
    from shapely.geometry import box

    for variable, value in {
        "tas": 293.15,
        "ps": 101325.0,
        "rsds": 200.0,
        "rsdt": 400.0,
    }.items():
        native(variable, value).to_netcdf(tmp_path / f"{variable}.nc")
    with open("config/catalogs/cmip6_data.yml", encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream)
    entry = next(value for key, value in defaults.items() if key != "meta")
    spec = {
        "uri": str(tmp_path / "{variable}.nc"),
        "data_type": "RasterDataset",
        "driver": {
            "name": "raster_xarray",
            "options": {"preprocess": register_pet_preprocess()},
        },
        "data_adapter": entry["data_adapter"],
        "metadata": {"crs": 4326},
        "placeholders": {"member": ["r1i1p1f1"]},
    }
    catalog = tmp_path / "catalog.yml"
    catalog.write_text(
        yaml.safe_dump({"cmip6_TEST/MODEL_historical_{member}": spec}), encoding="utf-8"
    )
    region = tmp_path / "region.geojson"
    gpd.GeoDataFrame(geometry=[box(-1, -1, 1, 61)], crs=4326).to_file(
        region, driver="GeoJSON"
    )
    raw = tmp_path / "raw.nc"
    fetch_raw_slice(
        region_path=region,
        raw_nc_out=raw,
        catalog_path=catalog,
        catalog_entry="cmip6_TEST/MODEL_historical_{member}",
        member="r1i1p1f1",
        variables=["temp", "press", "kin", "kout"],
        variable_units={
            "temp": "degC",
            "press": "hPa",
            "kin": "W m-2",
            "kout": "W m-2",
        },
        buffer=0,
        acquisition_window=("2000-01-01", "2000-02-29"),
        components={"pet_input_contract": INPUT_CONTRACT},
    )
    with xr.open_dataset(raw) as dataset:
        return dataset.load()


@pytest.mark.parametrize(
    "method,expected", [("makkink", 3.1081310), ("debruin", 3.0702580)]
)
def test_actual_adapter_transport_and_scalar_oracle(transported, method, expected):
    assert transported.temp.values == pytest.approx(np.full((2, 2, 2), 20.0))
    assert transported.press.values == pytest.approx(np.full((2, 2, 2), 1013.25))
    assert transported.press.attrs["cst_source_units"] == "Pa"
    assert transported.press.attrs["cst_monthly_admission"] == INPUT_CONTRACT
    calculated = calculate_pet(transported, method)
    independent = oracle(20, 1013.25, 200, 400 if method == "debruin" else None)
    assert independent == pytest.approx(expected, abs=1e-5)
    assert calculated.values == pytest.approx(np.full((2, 2, 2), independent), abs=1e-5)


def test_unequal_area_derive_before_average(transported):
    data = transported.copy(deep=True)
    data.temp.loc[{data.raster.y_dim: 60.0}] = -5
    spec = parse({"pet": {"method": "makkink"}})
    result = get_stats_clim_projections(
        data, "cmip6", "TEST/MODEL", "historical", "r1i1p1f1", variable_spec=spec
    )
    # Midpoint latitude edges: [-30,30,90]; areas are 1 and 0.5.
    expected = (oracle(20, 1013.25, 200) + 0.5 * oracle(-5, 1013.25, 200)) / 1.5
    wrong = oracle((20 + 0.5 * -5) / 1.5, 1013.25, 200)
    assert wrong != pytest.approx(expected, abs=1e-5)
    assert result.pet.values.ravel() == pytest.approx([expected, expected], abs=1e-5)
    assert set(result.data_vars) == {"pet"}


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_entire_selected_bbox_is_required(transported, value):
    data = transported.copy(deep=True)
    data.press.values[:, 1, :] = value
    with pytest.raises(ValueError, match="missing/nonfinite"):
        calculate_pet(data, "makkink")


def test_polar_night_and_valid_zero(transported):
    transported.kout.values[:] = 0
    transported.kin.values[:] = 0
    assert (calculate_pet(transported, "debruin") == 0).all()
    assert (calculate_pet(transported, "makkink") == 0).all()


@pytest.mark.parametrize("failure", ["units", "table", "interval", "calendar"])
def test_native_admission_refusals(failure):
    data = native("ps", 101325)
    if failure == "units":
        data.ps.attrs["units"] = "hPa"
    elif failure == "table":
        data.attrs["table_id"] = "day"
    elif failure == "interval":
        data.ps.attrs["cell_methods"] = "time: point"
    else:
        data.time.encoding.clear()
    with pytest.raises(ValueError, match="PET"):
        admit_monthly_sources(data)


def test_pet_admission_preserves_unrelated_supplied_variable_semantics():
    data = native("tas", 0.005).rename({"tas": "huss"})
    data.huss.attrs.update(units="kg kg-1", cell_methods="time: point")
    admitted = admit_monthly_sources(data)
    assert admitted.huss.attrs["units"] == "kg kg-1"
    assert admitted.huss.attrs["cell_methods"] == "time: point"
    assert admitted.huss.attrs["cst_source_calendar"] == "proleptic_gregorian"


def scalar_series(start, scenario, value, calendar):
    """Complete independent period fixture, with no production aggregation."""
    return xr.Dataset(
        {"pet": ("time", np.full(60, value), {"units": "mm/day"})},
        coords={"time": pd.date_range(f"{start}-01-01", periods=60, freq="MS")},
        attrs={"cst_calendar": calendar},
    ).expand_dims(
        model=["TEST/MODEL"],
        member=["r1i1p1f1"],
        scenario=[scenario],
        clim_project=["cmip6"],
    )


@pytest.mark.parametrize(
    "calendar,days", [("360_day", 360), ("noleap", 365), ("proleptic_gregorian", 365.4)]
)
@pytest.mark.parametrize(
    "reference,flagged", [(0.099, True), (0.1, False), (2.0, False)]
)
def test_calendar_integrals_daily_threshold_and_table_oracle(
    calendar, days, reference, flagged
):
    from blueearth_cst.projections.change_factor_table import tidy_rows
    from blueearth_cst.projections.get_change_climate_proj import (
        get_change_annual_clim_proj,
    )
    from blueearth_cst.projections.projection_analysis import analysis_change_table

    hist = scalar_series(2000, "historical", reference, calendar)
    future = scalar_series(2040, "ssp245", reference * 2, calendar)
    table = analysis_change_table(
        hist,
        future,
        identity={"model": "TEST/MODEL", "member": "r1i1p1f1", "scenario": "ssp245"},
        window_id="2040-2044",
        reference_window=(2000, 2004),
        future_window=(2040, 2044),
        variable_spec=parse({"pet": {"method": "makkink"}}),
        min_reference={"pet": 0.1},
    )
    annual = table.loc[table.period.eq("annual") & table.statistic.eq("mean")]
    absolute = annual.loc[annual.quantity.eq("change")].iloc[0]
    relative = annual.loc[annual.quantity.eq("change_pct")].iloc[0]
    assert absolute.value == pytest.approx(reference * days)
    assert absolute.units == "mm"
    assert absolute.mean_change_eligible
    assert pd.isna(relative.value) if flagged else relative.value == pytest.approx(100)
    assert relative.mean_change_eligible == (not flagged)
    seasons = table.loc[
        table.period.eq("MAM")
        & table.statistic.eq("mean")
        & table.quantity.eq("change")
    ]
    assert seasons.iloc[0].value == pytest.approx(
        reference * (90 if calendar == "360_day" else 92)
    )
    monthly = table.loc[
        table.period.eq("Feb")
        & table.statistic.eq("mean")
        & table.quantity.eq("change")
    ]
    assert monthly.iloc[0].value == pytest.approx(reference)
    spec = parse({"pet": {"method": "makkink"}})
    aligned = get_change_annual_clim_proj(
        hist,
        future,
        stats=["mean"],
        calendar=calendar,
        variable_spec=spec,
        min_reference={"pet": 0.1},
    )
    summary = tidy_rows(aligned, variable_spec=spec)[0]
    assert summary["status"] == ("reference_below_threshold" if flagged else "ok")
    assert summary["absolute_value"] - summary["reference_value"] == pytest.approx(
        reference
    )


def test_raw_identity_excludes_method_and_reporting_but_not_source_contract():
    from blueearth_cst.projections.series_identity import raw_digest, series_digest

    first = {
        "variable_spec": ["temp", "press", "kin", "kout"],
        "reducer_module_hash": "code",
        "derived_pet": {"method": "makkink"},
        "pet_input_contract": INPUT_CONTRACT,
    }
    changed = {**first, "derived_pet": {"method": "debruin"}}
    assert raw_digest(first, "region") == raw_digest(changed, "region")
    assert series_digest(first, "region") != series_digest(changed, "region")
    changed = {**first, "pet_input_contract": "changed admission"}
    assert raw_digest(first, "region") != raw_digest(changed, "region")


def test_output_selection_invalidates_series_even_when_pet_inputs_are_unchanged():
    from blueearth_cst.projections.series_identity import raw_digest, series_digest
    from blueearth_cst.projections.variable_spec import derived_pet_identity

    base = parse({"pet": {"method": "makkink"}})
    with_temperature = parse({"pet": {"method": "makkink"}, "temp": None})
    threshold_only = parse({"pet": {"method": "makkink", "min_denominator": 0.2}})
    assert source_names(base) == source_names(with_temperature)
    first = {
        "variable_spec": source_names(base),
        "derived_pet": derived_pet_identity(base),
    }
    changed = {
        "variable_spec": source_names(with_temperature),
        "derived_pet": derived_pet_identity(with_temperature),
    }
    assert raw_digest(first, "region") == raw_digest(changed, "region")
    assert series_digest(first, "region") != series_digest(changed, "region")
    assert derived_pet_identity(threshold_only) == derived_pet_identity(base)


def test_supplied_pet_keeps_legacy_analysis_product_boundary():
    from blueearth_cst.projections.get_change_climate_proj import (
        get_change_annual_clim_proj,
    )
    from blueearth_cst.projections.projection_analysis import (
        analysis_change_table,
        product_availability,
    )
    from blueearth_cst.projections.projection_trajectories import member_anomalies

    spec = parse(
        {
            "pet": {
                "source": "pet",
                "canonical": "rate",
                "units": "mm/day",
                "change": "absolute",
            }
        }
    )
    hist = scalar_series(2000, "historical", 2, "360_day")
    future = scalar_series(2040, "ssp245", 3, "360_day")
    identity = {"model": "TEST/MODEL", "member": "r1i1p1f1", "scenario": "ssp245"}
    table = analysis_change_table(
        hist,
        future,
        identity=identity,
        window_id="2040-2044",
        reference_window=(2000, 2004),
        future_window=(2040, 2044),
        variable_spec=spec,
    )
    assert table.status.eq("unavailable").all()
    assert set(table.quantity) == {"change"}
    assert member_anomalies(
        hist, future, identity=identity, reference_window=(2000, 2004)
    ).empty
    assert not product_availability({}).filename.str.contains("_pet_").any()
    annual = get_change_annual_clim_proj(
        hist, future, stats=["mean"], calendar="360_day", variable_spec=spec
    )
    # Existing supplied-variable annual levels remain canonical rates.
    assert annual.pet__level.item() == pytest.approx(3)
    assert annual.pet.item() == pytest.approx(360)


def test_pet_relative_std_is_flagged_without_losing_absolute_difference():
    from blueearth_cst.projections.projection_analysis import analysis_change_table

    table = analysis_change_table(
        scalar_series(2000, "historical", 2, "360_day"),
        scalar_series(2040, "ssp245", 3, "360_day"),
        identity={"model": "TEST/MODEL", "member": "r1i1p1f1", "scenario": "ssp245"},
        window_id="2040-2044",
        reference_window=(2000, 2004),
        future_window=(2040, 2044),
        variable_spec=parse({"pet": {"method": "makkink"}}),
        min_reference={"pet": 0.1},
    )
    row = table.loc[
        table.period.eq("annual")
        & table.statistic.eq("std")
        & table.quantity.eq("change_pct")
    ].iloc[0]
    assert row.status == "reference_below_threshold"
    assert np.isnan(row.value)
    absolute = table.loc[
        table.period.eq("annual")
        & table.statistic.eq("std")
        & table.quantity.eq("change")
    ].iloc[0]
    assert np.isfinite(absolute.value)


def test_dependency_holes_and_calendar_mismatch_refuse(transported):
    with pytest.raises(ValueError, match="consecutive"):
        calculate_pet(
            transported.assign_coords(
                time=pd.to_datetime(["2000-01-01", "2000-03-01"])
            ),
            "makkink",
        )
    transported.press.attrs["cst_source_calendar"] = "360_day"
    with pytest.raises(ValueError, match="calendars"):
        calculate_pet(transported, "makkink")


def test_shared_pin_is_applied_to_noncertified_dependency(tmp_path):
    folder = tmp_path / "ps/gn"
    folder.mkdir(parents=True)
    native("ps", 101325.0).to_netcdf(folder / "v1.nc")
    spec = {
        "uri": str(tmp_path / "{variable}/*/*"),
        "data_type": "RasterDataset",
        "driver": {
            "name": "raster_xarray",
            "options": {"preprocess": register_pet_preprocess()},
        },
        "data_adapter": {"rename": {"ps": "press"}, "unit_mult": {"press": 0.01}},
        "metadata": {"crs": 4326},
        "placeholders": {"member": ["r1i1p1f1"]},
    }
    args = (spec, "cmip6_TEST/MODEL_historical_{member}", "r1i1p1f1", ["press"])
    kwargs = dict(
        bbox=[-1, -1, 1, 61],
        buffer=0,
        window=("2000-01-01", "2000-02-29"),
        admit_pet=True,
    )
    data, _ = read_variable_slices(*args, chosen={"tas": "gn/v1.nc"}, **kwargs)
    assert (
        str(data.press.attrs["cst_selected_source_path"])
        .replace("\\", "/")
        .endswith("/ps/gn/v1.nc")
    )
    np.testing.assert_allclose(data.press.values, 1013.25)
    with pytest.raises(ValueError, match="no indexed pin.*ps"):
        read_variable_slices(
            *args, chosen={"tas": "gn/v1.nc", "rsds": "gn/v2.nc"}, **kwargs
        )


def test_out_of_bbox_missing_cells_are_excluded(transported):
    extended = transported.reindex(lon=[0.0, 1.0, 2.0])
    selected = clip_to_bbox(extended, [-0.1, -1, 1.1, 61], 0, "lat", "lon")
    assert selected.sizes["lon"] == 2
    assert np.isfinite(calculate_pet(selected, "makkink")).all()
