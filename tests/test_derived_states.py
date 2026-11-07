"""Independent dew-point/wind controls and the real native HydroMT admission seam."""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.projections.derived_states import (
    admit_state_sources as admit_monthly_sources,
)
from blueearth_cst.projections.derived_states import calculate_state
from blueearth_cst.projections.derived_states import (
    register_state_preprocess as register_pet_preprocess,
)
from blueearth_cst.projections.fetch_gcm_raw import read_variable_slices
from blueearth_cst.projections.get_stats_climate_proj import get_stats_clim_projections
from blueearth_cst.projections.variable_spec import (
    derived_state_identity,
    parse,
    source_names,
    source_units,
)

NATIVE = {
    "huss": ("sh", "1"),
    "ps": ("press", "Pa"),
    "sfcWind": ("wind10", "m s-1"),
    "uas": ("wind10_u", "m s-1"),
    "vas": ("wind10_v", "m s-1"),
}


def native(name, value):
    """Match CMIP6 Amon variable metadata; fixtures do not qualify live stores."""
    ds = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.broadcast_to(value, (2, 2, 2)).copy(),
                {"units": NATIVE[name][1], "cell_methods": "area: time: mean"},
            )
        },
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
    if name in {"sfcWind", "uas", "vas"}:
        ds = ds.assign_coords(
            height=xr.DataArray(
                10.0, attrs={"units": "m", "standard_name": "height", "positive": "up"}
            )
        )
    ds.time.encoding["calendar"] = "proleptic_gregorian"
    return ds


def admitted(fields):
    """Admitted arrays for isolated arithmetic tests; transport tested separately."""
    arrays = []
    for name, value in fields.items():
        target = NATIVE[name][0]
        ds = admit_monthly_sources(native(name, value)).rename({name: target})
        if name == "ps":
            attrs = dict(ds[target].attrs)
            ds[target] = ds[target] * 0.01
            ds[target].attrs = {**attrs, "units": "hPa"}
        elif name in {"sfcWind", "uas", "vas"}:
            ds[target].attrs["units"] = "m/s"
        arrays.append(ds)
    return xr.merge(arrays, join="exact", compat="no_conflicts")


@pytest.mark.parametrize(
    "name,method,inputs",
    [
        ("tdew", "specific_humidity", ["press", "sh"]),
        ("wind", "sfcwind", ["wind10"]),
        ("wind", "components", ["wind10_u", "wind10_v"]),
    ],
)
def test_spec(name, method, inputs):
    spec = parse({name: {"method": method}})
    assert source_names(spec) == inputs
    assert spec[name].canonical == "state"
    assert spec[name].change == "absolute"
    assert set(source_units(spec)) == set(inputs)


@pytest.mark.parametrize("name", ["tdew", "wind"])
@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"method": None},
        {"method": "unknown"},
        {"method": "components", "change": "relative"},
    ],
)
def test_invalid_spec(name, body):
    with pytest.raises(ValueError, match=f"variables.{name}"):
        parse({name: body})


@pytest.mark.parametrize("name,units", [("tdew", "degC"), ("wind", "m/s")])
def test_supplied_forms_preserved(name, units):
    spec = parse(
        {
            name: dict(
                source="supplied", canonical="state", units=units, change="absolute"
            )
        }
    )
    assert spec[name].method is None
    assert source_names(spec) == ["supplied"]
    assert source_units(spec) == {"supplied": units}
    assert derived_state_identity(spec) is None


@pytest.mark.parametrize("dewpoint", [-30.0, -5.0, 0.0, 20.0])
def test_independent_forward_vapour_relation(dewpoint):
    # NWS forward relation, not the production inverse; includes liquid water below zero.
    e = 6.11 * 10 ** (7.5 * dewpoint / (237.3 + dewpoint))
    pressure = 987.0
    q = 0.622 * e / (pressure - (1 - 0.622) * e)
    result = calculate_state(
        admitted({"huss": q, "ps": pressure * 100}), "tdew", "specific_humidity"
    )
    np.testing.assert_allclose(result, dewpoint, atol=1e-10)


def test_components_calm_and_cancellation_before_area_weighting():
    data = admitted({"uas": [[3.0, -3.0], [3.0, -3.0]], "vas": 4.0})
    result = get_stats_clim_projections(
        data,
        "cmip6",
        "TEST/MODEL",
        "historical",
        "r1i1p1f1",
        parse({"wind": {"method": "components"}}),
    )
    np.testing.assert_allclose(result.wind, 5.0, atol=1e-6)
    assert result.wind.attrs["cst_height_m"] == 10
    assert "monthly mean wind vector" in result.wind.attrs["cst_statistic"]
    np.testing.assert_array_equal(
        calculate_state(admitted({"uas": 0.0, "vas": 0.0}), "wind", "components"), 0.0
    )


def test_dewpoint_before_area_weighting():
    data = admitted({"huss": [[0.001, 0.02], [0.001, 0.02]], "ps": 100000.0})
    result = get_stats_clim_projections(
        data,
        "cmip6",
        "TEST/MODEL",
        "historical",
        "r1i1p1f1",
        parse({"tdew": {"method": "specific_humidity"}}),
    )

    def inverse(q):
        e = q * 1000 / (0.622 + 0.378 * q)
        return 237.3 * math.log10(e / 6.11) / (7.5 - math.log10(e / 6.11))

    expected = (inverse(0.001) + inverse(0.02)) / 2
    np.testing.assert_allclose(result.tdew, expected, atol=1e-6)
    assert abs(expected - inverse(0.0105)) > 5


@pytest.mark.parametrize(
    "name,value,variable,method",
    [
        ("huss", 0.0, "tdew", "specific_humidity"),
        ("huss", 1.0, "tdew", "specific_humidity"),
        ("ps", 0.0, "tdew", "specific_humidity"),
        ("huss", np.nan, "tdew", "specific_humidity"),
        ("sfcWind", -1.0, "wind", "sfcwind"),
        ("uas", np.inf, "wind", "components"),
    ],
)
def test_invalid_values_fail(name, value, variable, method):
    fields = (
        {"huss": 0.005, "ps": 100000.0}
        if variable == "tdew"
        else {"sfcWind": 2.0}
        if method == "sfcwind"
        else {"uas": 3.0, "vas": 4.0}
    )
    fields[name] = value
    with pytest.raises(ValueError):
        calculate_state(admitted(fields), variable, method, source="TEST/MODEL")


@pytest.mark.parametrize("height,units", [(None, "m"), (2.0, "m"), (10.0, "km")])
def test_native_height_before_harmonisation(height, units):
    ds = native("sfcWind", 2.0)
    ds = (
        ds.drop_vars("height")
        if height is None
        else ds.assign_coords(height=xr.DataArray(height, attrs={"units": units}))
    )
    with pytest.raises(ValueError, match="10 m height"):
        admit_monthly_sources(ds)


@pytest.mark.parametrize("name", ["huss", "sfcWind", "uas", "vas"])
def test_wrong_native_units_refuse(name):
    ds = native(name, 0.005 if name == "huss" else 2.0)
    ds[name].attrs["units"] = "unqualified"
    with pytest.raises(ValueError, match=f"source {name}: units"):
        admit_monthly_sources(ds)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "nonfinite_cell",
        "canonical_units",
        "monthly_hole",
        "calendar",
        "admission",
    ],
)
def test_required_support_and_contract_guards(fault):
    ds = admitted({"huss": 0.005, "ps": 100000.0})
    if fault == "missing":
        ds = ds.drop_vars("sh")
    elif fault == "nonfinite_cell":
        ds.sh.values[0, 0, 0] = np.nan
    elif fault == "canonical_units":
        ds.press.attrs["units"] = "Pa"
    elif fault == "monthly_hole":
        ds = ds.assign_coords(time=pd.to_datetime(["2000-01-01", "2000-03-01"]))
    elif fault == "calendar":
        ds.sh.attrs["cst_source_calendar"] = "360_day"
    else:
        ds.sh.attrs.pop("cst_monthly_admission")
    with pytest.raises(ValueError, match="tdew/specific_humidity"):
        calculate_state(ds, "tdew", "specific_humidity", source="TEST/MODEL")


@pytest.mark.parametrize(
    "fault", ["grid", "calendar", "member", "units", "coordinates"]
)
def test_real_transport_rejects_incompatible_sources(tmp_path, fault):
    _transport(tmp_path, fault=fault)


def _transport(tmp_path, fault=None, method="components"):
    """Exercise actual generated adapter and installed HydroMT raster driver."""
    catalog = yaml.safe_load(open("config/catalogs/cmip6_data.yml", encoding="utf-8"))
    adapter = next(v for k, v in catalog.items() if k != "meta")["data_adapter"]
    fields = {
        "huss": 0.005,
        "ps": 100000.0,
        **({"uas": 3.0, "vas": 4.0} if method == "components" else {"sfcWind": 5.0}),
    }
    for name, value in fields.items():
        ds = native(name, value)
        if name == "huss":
            if fault == "grid":
                ds.attrs["grid_label"] = "gr"
            if fault == "calendar":
                ds.time.encoding["calendar"] = "noleap"
            if fault == "member":
                ds.attrs["variant_label"] = "r2i1p1f1"
            if fault == "units":
                ds[name].attrs["units"] = "%"
            if fault == "coordinates":
                ds = ds.assign_coords(lon=[0.0, 2.0])
        ds.to_netcdf(tmp_path / f"{name}.nc")
    entry = dict(
        uri=str(tmp_path / "{variable}.nc"),
        data_type="RasterDataset",
        driver=dict(
            name="raster_xarray", options=dict(preprocess=register_pet_preprocess())
        ),
        data_adapter=adapter,
        metadata=dict(crs=4326),
        placeholders={"member": ["r1i1p1f1"]},
    )
    spec = parse({"tdew": {"method": "specific_humidity"}, "wind": {"method": method}})

    def read():
        return read_variable_slices(
            entry,
            "cmip6_TEST/MODEL_historical_{member}",
            "r1i1p1f1",
            source_names(spec),
            {},
            [-1.0, -1.0, 3.0, 61.0],
            0,
            ("2000-01-01", "2000-02-29"),
            admit_pet=True,
        )

    if fault:
        with pytest.raises(ValueError):
            read()
    else:
        ds, _ = read()
        for name, units in source_units(spec).items():
            ds[name].attrs["units"] = units
        return get_stats_clim_projections(
            ds, "cmip6", "TEST/MODEL", "historical", "r1i1p1f1", spec
        )


@pytest.mark.parametrize("method", ["sfcwind", "components"])
def test_real_transport_without_pet(tmp_path, method):
    result = _transport(tmp_path, method=method)
    assert set(result.data_vars) == {"tdew", "wind"}
    np.testing.assert_allclose(result.wind, 5.0, atol=1e-6)
    assert result.wind.attrs["cst_derived_method"] == method


def test_state_identity_separates_raw_from_selected_outputs():
    from blueearth_cst.projections.series_identity import raw_digest, series_digest

    base = parse({"tdew": {"method": "specific_humidity"}})
    supplied_pressure = {
        "source": "press",
        "canonical": "state",
        "units": "hPa",
        "change": "absolute",
    }
    expanded = parse(
        {"tdew": {"method": "specific_humidity"}, "pressure": supplied_pressure}
    )
    assert source_names(base) == source_names(expanded)
    first = {
        "variables": source_names(base),
        "derived_states": derived_state_identity(base),
    }
    second = {
        "variables": source_names(expanded),
        "derived_states": derived_state_identity(expanded),
    }
    assert raw_digest(first, "region") == raw_digest(second, "region")
    assert series_digest(first, "region") != series_digest(second, "region")
    assert derived_state_identity(
        parse({"tdew": {"method": "specific_humidity", "change": "absolute"}})
    ) == derived_state_identity(base)


@pytest.mark.parametrize(
    "calendar,days",
    [
        ("360_day", [30] * 12),
        ("noleap", [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
        ("proleptic_gregorian", [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
    ],
)
@pytest.mark.parametrize(
    "variable,method,units",
    [("tdew", "specific_humidity", "degC"), ("wind", "components", "m/s")],
)
def test_duration_weighted_states_not_totals(calendar, days, variable, method, units):
    from blueearth_cst.projections.get_change_climate_proj import (
        get_change_annual_clim_proj,
    )
    from blueearth_cst.projections.projection_analysis import period_values

    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    values = np.arange(12, dtype=float)
    ds = xr.Dataset(
        {variable: ("time", values, {"units": units})},
        coords={"time": time},
        attrs={"cst_calendar": calendar},
    )
    expected = np.dot(values, days) / sum(days)
    assert period_values(ds, variable)["annual"]["level"].item() == pytest.approx(
        expected
    )
    future = ds.assign_coords(time=pd.date_range("2040-01-01", periods=12, freq="MS"))
    future[variable] = future[variable] + 2
    result = get_change_annual_clim_proj(
        ds.expand_dims(scenario=["historical"]),
        future.expand_dims(scenario=["ssp245"]),
        stats=["mean"],
        calendar=calendar,
        variable_spec=parse({variable: {"method": method}}),
    )
    assert result[variable].item() == pytest.approx(2)
    assert result[f"{variable}__reference"].item() == pytest.approx(expected)
