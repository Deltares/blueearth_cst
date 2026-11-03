"""Tests for derive_change_factors.py — the composition record (design §5.7).

The change arithmetic is not retested here: step 4d imports it unchanged from
``get_change_climate_proj`` / ``get_change_climate_proj_summary``, which keep their
own suites. What is new in 4d, and therefore what is pinned here, is the
composition record — one row per **requested** combination, with the resolved-only
columns populated only when the point actually resolved.
"""

import csv
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import derive_change_factors as derive_module
from blueearth_cst.projections import resolution
from blueearth_cst.projections import series_identity as si
from blueearth_cst.projections.change_factor_table import tidy_rows
from blueearth_cst.projections.derive_change_factors import (
    COMPOSITION_CSV_COLUMNS,
    COMPOSITION_FIELDS,
    WindowUnavailable,
    _flagged_months,
    _window_bounds,
    composition_rows,
    derive_available_point,
    derive_point_datasets,
    write_composition,
)
from blueearth_cst.projections.get_change_climate_proj import (
    get_change_monthly_clim_proj,
    hydrological_year_bounds,
)


@pytest.mark.parametrize(
    "start_month, reference, level, n_years",
    [("Oct", 10.0, 20.0, 2), ("Jan", 50 / 3, 140 / 3, 3)],
)
def test_monthly_factors_use_the_reported_complete_water_years(
    tmp_path, start_month, reference, level, n_years
):
    """Partial October years carry outliers that January must retain."""
    paths, components = [], []
    for scenario, first_year, value, edge in [
        ("historical", 2000, 10.0, 30.0),
        ("ssp245", 2004, 20.0, 100.0),
    ]:
        time = pd.date_range(f"{first_year}-01-01", periods=36, freq="MS")
        values = np.full(36, value)
        values[:9] = edge
        values[-3:] = edge
        identity = {"scenario": scenario}
        ds = xr.Dataset({"precip": ("time", values)}, coords={"time": time})
        ds = ds.expand_dims(
            scenario=[scenario], member=["r1"], model=["model"], clim_project=["cmip6"]
        )
        ds.attrs.update(
            cst_calendar="noleap",
            cst_schema_version=si.SCHEMA_VERSION,
            cst_series_digest=si.series_digest(identity, "region"),
        )
        path = tmp_path / f"{scenario}.nc"
        ds.to_netcdf(path)
        paths.append(path)
        components.append(identity)
    _, monthly, facts = derive_point_datasets(
        series_path_hist=paths[0],
        series_path=paths[1],
        time_tuple_hist=("2000", "2002"),
        time_tuple_fut=("2004", "2006"),
        name_horizon="near",
        name_model="model",
        name_scenario="ssp245",
        region_fp="region",
        digest_components_hist=components[0],
        digest_components_fut=components[1],
        stats=["mean", "q_90"],
        water_year_start=start_month,
    )
    assert facts[2] == n_years
    assert f"q_90[n={n_years}]" in monthly.stats
    np.testing.assert_allclose(monthly.precip__reference.sel(stats="mean"), reference)
    np.testing.assert_allclose(monthly.precip__level.sel(stats="mean"), level)
    np.testing.assert_allclose(
        monthly.precip.sel(stats="mean"), (level - reference) / reference * 100
    )
    if start_month == "Jan":
        with xr.open_dataset(paths[0]) as hist, xr.open_dataset(paths[1]) as future:
            legacy = (
                get_change_monthly_clim_proj(hist, future, stats=["mean", "q_90"])
                .assign_coords(horizon="near")
                .expand_dims("horizon")
            )
            assert monthly.to_netcdf(engine="scipy") == legacy.to_netcdf(engine="scipy")


@pytest.mark.parametrize("future_end", [2039, 2060])
def test_independent_gcm_windows_and_missing_horizon_diagnostic(tmp_path, future_end):
    paths, components = [], []
    for scenario, start, end, value in [
        ("historical", 1985, 2004, 10.0),
        ("ssp245", 2015, future_end, 12.0),
    ]:
        dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
        ds = xr.Dataset(
            {"precip": ("time", np.full(len(dates), value))}, coords={"time": dates}
        ).expand_dims(
            scenario=[scenario], member=["r1"], model=["model"], clim_project=["cmip6"]
        )
        identity = {"scenario": scenario}
        ds.attrs.update(
            cst_calendar="noleap",
            cst_schema_version=si.SCHEMA_VERSION,
            cst_series_digest=si.series_digest(identity, "region"),
        )
        path = tmp_path / f"{scenario}.nc"
        ds.to_netcdf(path)
        paths.append(path)
        components.append(identity)
    kwargs = dict(
        series_path_hist=paths[0],
        series_path=paths[1],
        time_tuple_hist=("1985", "2004"),
        time_tuple_fut=("2040", "2060"),
        name_horizon="near",
        name_model="model",
        name_scenario="ssp245",
        region_fp="region",
        digest_components_hist=components[0],
        digest_components_fut=components[1],
        stats=["mean"],
    )
    if future_end == 2039:
        with pytest.raises(WindowUnavailable) as caught:
            derive_point_datasets(**kwargs)
        message = str(caught.value)
        assert "model ssp245 horizon 'near'" in message
        assert "2040..2060" in message and "2015-01-01..2039-12-01" in message
        assert str(paths[1]) in message
        exclusions = []
        cache, reference_cache = {}, {}
        point = {"model": "model", "scenario": "ssp245", "member": "r1"}
        assert (
            derive_available_point(
                point=point,
                unavailable_windows=exclusions,
                series_cache=cache,
                reference_cache=reference_cache,
                **kwargs,
            )
            is None
        )
        assert exclusions[0]["requested_window"] == "2040 / 2060"
        assert exclusions[0]["available_end"] == "2039-12-01"
        assert not reference_cache
        # The failed horizon cannot poison the reference cache or exclude another.
        kwargs["time_tuple_fut"] = ("2020", "2024")
        kwargs["name_horizon"] = "early"
        result = derive_available_point(
            point=point,
            unavailable_windows=exclusions,
            series_cache=cache,
            reference_cache=reference_cache,
            **kwargs,
        )
        np.testing.assert_allclose(result[0].precip.sel(stats="mean"), 20.0)
        assert len(exclusions) == 1
        # Calendar defects still fail even when the requested horizon is absent.
        kwargs["time_tuple_fut"] = ("2040", "2060")
        for (path, _), dataset in cache.items():
            if path == str(paths[1].resolve()):
                dataset.attrs["cst_calendar"] = "360_day"
        with pytest.raises(ValueError, match="different calendars"):
            derive_available_point(
                point=point,
                unavailable_windows=exclusions,
                series_cache=cache,
                reference_cache=reference_cache,
                **kwargs,
            )
        assert len(exclusions) == 1
    else:
        annual, monthly, facts = derive_point_datasets(**kwargs)
        assert facts[2] == 20
        np.testing.assert_allclose(annual.precip.sel(stats="mean"), 20.0)
        np.testing.assert_allclose(monthly.precip.sel(stats="mean"), 20.0)


def test_empty_hydrological_time_axis_has_a_clear_error():
    dataset = xr.Dataset(coords={"time": pd.DatetimeIndex([])})
    with pytest.raises(ValueError, match="empty time axis"):
        hydrological_year_bounds(dataset)


def test_no_complete_year_is_unavailable_but_invalid_start_month_is_hard():
    dataset = xr.Dataset(
        coords={"time": pd.date_range("2020-02-01", periods=5, freq="MS")}
    )
    with pytest.raises(
        WindowUnavailable, match="no complete hydrological year"
    ) as caught:
        _window_bounds(
            dataset, dataset, ("2020", "2020"), "future.nc", "horizon", "Jan"
        )
    assert caught.value.record["available_start"] == "2020-02-01"
    with pytest.raises(ValueError) as caught:
        _window_bounds(
            dataset, dataset, ("2020", "2020"), "future.nc", "horizon", "BAD"
        )
    assert not isinstance(caught.value, WindowUnavailable)


@pytest.mark.parametrize("include_usable", [False, True])
def test_stage_b_writes_auditable_products_with_unavailable_horizons(
    tmp_path, include_usable, capsys
):
    """Execute the actual Snakemake script using checked local series and geometry."""
    region_path = tmp_path / "region.geojson"
    region_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fingerprint = si.region_fingerprint(region_path)
    models = [("E3SM-Project/E3SM-1-1", 2024, 2.0)]
    if include_usable:
        models.extend([("TEST/Usable", 2060, 2.0), ("LONG/UsableHigher", 2060, 10.0)])
    points, combinations, declared = [], [], []
    for model, future_end, height in models:
        combo = _combo(model, "ssp245", "r1i1p1f1", "resolved")
        combinations.append(combo)
        point = dict(combo, model=model, tier="certified")
        for scenario, start, end, value in [
            ("historical", 1985, 2004, 10.0),
            ("ssp245", 2020, future_end, 12.0),
        ]:
            identity = {"scenario": scenario, "model": model}
            dates = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
            dataset = xr.Dataset(
                {"precip": ("time", np.full(len(dates), value))},
                coords={"time": dates, "height": height},
            ).expand_dims(
                scenario=[scenario],
                model=[model],
                member=["r1i1p1f1"],
                clim_project=["cmip6"],
            )
            dataset.attrs.update(
                cst_calendar="noleap",
                cst_schema_version=si.SCHEMA_VERSION,
                cst_series_digest=si.series_digest(identity, fingerprint),
                cst_time_first=str(dates[0].date()),
                cst_time_last=str(dates[-1].date()),
            )
            key = f"{model.replace('/', '_')}_{scenario}"
            path = tmp_path / f"{key}.nc"
            dataset.to_netcdf(path)
            declared.append(str(path))
            if scenario == "historical":
                point.update(
                    series_path_hist=str(path),
                    reference_series_key=key,
                    digest_components_hist=identity,
                )
            else:
                point.update(
                    series_path=str(path),
                    series_key=key,
                    digest_components_fut=identity,
                )
        points.append(point)
    excluded = resolution.resolve(
        {},
        clim_project="cmip6",
        models=["THU/CIESM"],
        scenarios=["ssp245"],
        members=["r1i1p1f1"],
    )[0]
    combinations.append(
        {
            **_combo(
                excluded.dataset, excluded.scenario, excluded.member, excluded.status
            ),
            "detail": excluded.detail,
        }
    )
    output = SimpleNamespace(
        **{
            name: str(tmp_path / filename)
            for name, filename in {
                "change_factors_annual": "annual.csv",
                "change_factors_monthly": "monthly.csv",
                "composition_csv": "composition.csv",
                "provenance_json": "provenance.json",
                "report_md": "report.md",
            }.items()
        }
    )
    sm = SimpleNamespace(
        input=SimpleNamespace(region_path=str(region_path), series_nc=declared),
        output=output,
        log=[str(tmp_path / "derive.log")],
        params=SimpleNamespace(
            clim_project_dir=str(tmp_path / "cmip6"),
            horizons={"near": [2040, 2060]},
            points=points,
            combinations=combinations,
            time_horizon_hist=[1985, 2004],
            reference_record={
                "reference_window_requested": "1985 / 2004",
                "reference_window_effective": "1985 / 2004",
                "reference_window_years": 20,
                "reference_window_clipped": False,
            },
            variable_spec={
                "precip": ["precip", "precip", "rate", "mm/day", "relative"]
            },
            stats=["mean"],
            water_year_start="Jan",
            min_reference={"precip": 0.1},
            catalog_crawled_on="2026-10-13",
            effective_config_sha256="synthetic",
            max_flagged_months=2,
        ),
    )
    runpy.run_path(derive_module.__file__, init_globals={"snakemake": sm})
    console = capsys.readouterr().out
    assert "- change - WARNING - Skipped E3SM-Project/E3SM-1-1" in console
    assert "2020-01-01–2024-12-01" in console
    for name, expected in [
        ("change_factors_annual", 1),
        ("change_factors_monthly", 12),
    ]:
        with open(getattr(output, name), encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            assert reader.fieldnames
        assert len(rows) == (2 * expected if include_usable else 0)
        assert {row["model"] for row in rows} == (
            {"Usable", "UsableHigher"} if include_usable else set()
        )
        for row in rows:
            assert float(row["relative_value"]) == pytest.approx(20.0)
    with open(output.composition_csv, encoding="utf-8", newline="") as fh:
        composition = {row["model"]: row for row in csv.DictReader(fh)}
    assert composition["E3SM-1-1"]["status"] == "unavailable"
    assert composition["CIESM"]["status"] == "source_quality_excluded"
    assert "all variables" in composition["CIESM"]["reason"]
    if include_usable:
        assert composition["Usable"]["status"] == "resolved"
        assert composition["UsableHigher"]["status"] == "resolved"
    document = json.loads(Path(output.provenance_json).read_text(encoding="utf-8"))
    assert document["composition"]["usable"] == 2 * int(include_usable)
    assert (
        document["composition"]["unresolved_by_status"]["source_quality_excluded"] == 1
    )
    assert document["source_quality_exclusions"] == [
        {
            "dataset": "THU/CIESM",
            "scenario": "ssp245",
            "member": "r1i1p1f1",
            "reason": excluded.detail,
        }
    ]
    assert len(document["unavailable_windows"]) == 1
    assert document["unavailable_windows"][0]["available_end"] == "2024-12-01"
    report = Path(output.report_md).read_text(encoding="utf-8")
    assert "E3SM-1-1" in report and "2024-12-01" in report
    assert "source_quality_excluded" in report and "CIESM" in report


def test_flagged_months_uses_tidy_model_identity_and_counts_distinct_months():
    coords = {
        "model": ["INM/INM-CM4-8"],
        "scenario": ["ssp245"],
        "member": ["r1i1p1f1"],
        "horizon": ["far"],
        "stats": ["mean", "median", "std"],
    }
    shape = tuple(len(values) for values in coords.values())
    ds = xr.Dataset(
        {
            "precip": (tuple(coords), np.zeros(shape)),
            "precip__flagged": (tuple(coords), np.ones(shape, dtype=bool)),
        },
        coords=coords,
    )
    monthly_rows = tidy_rows(ds, month=1) + tidy_rows(ds, month=2)
    assert all("dataset" not in row for row in monthly_rows)
    expected = [
        {
            "dataset": "INM/INM-CM4-8",
            "scenario": "ssp245",
            "member": "r1i1p1f1",
            "horizon": "far",
            "variable": "precip",
            "n_flagged_months": 2,
            "exceeds_max": False,
        }
    ]
    assert _flagged_months(monthly_rows, _rows(), 2) == expected
    assert _flagged_months(monthly_rows, _rows(), 1)[0]["exceeds_max"] is True
    ds["precip__flagged"][:] = False
    assert _flagged_months(tidy_rows(ds, month=3), _rows(), 2) == []


def _combo(dataset, scenario, member, status, detail=""):
    institution, _, source_id = dataset.partition("/")
    return {
        "point_key": f"{dataset.replace('/', '_')}_{scenario}_{member}",
        "dataset": dataset,
        "institution": institution,
        "source_id": source_id or dataset,
        "scenario": scenario,
        "member": member,
        "status": status,
        "detail": detail,
        "catalog_entry": f"cmip6_{dataset}_{scenario}_{{member}}",
    }


RESOLVED_FACTS = {
    "INM_INM-CM4-8_ssp245_r1i1p1f1": {
        "series_key": "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1",
        "reference_series_key": "cmip6_INM_INM-CM4-8_historical_r1i1p1f1",
        "tier": "certified",
        "reference_window_effective": "1970-01-01 / 2009-12-01",
        "n_hyd_years_reference": 40,
    }
}


def _rows():
    combinations = [
        _combo("INM/INM-CM4-8", "ssp245", "r1i1p1f1", "resolved"),
        _combo(
            "SNU/SAM0-UNICON",
            "ssp245",
            "r1i1p1f1",
            "scenario_not_published",
            "no entry cmip6_SNU/SAM0-UNICON_ssp245_{member}",
        ),
    ]
    return composition_rows(
        combinations,
        RESOLVED_FACTS,
        catalog_crawled_on="2026-07-29",
        window_nominal="1970 / 2010",
    )


def test_one_row_per_requested_combination_not_per_resolved_one():
    """The skips are the point: a record of only resolved rows is not auditable."""
    rows = _rows()
    assert len(rows) == 2
    assert [r["status"] for r in rows] == ["resolved", "scenario_not_published"]


@pytest.mark.parametrize("has_usable_horizon", [False, True])
def test_composition_separates_publication_from_window_availability(has_usable_horizon):
    rows = composition_rows(
        [_combo("INM/INM-CM4-8", "ssp245", "r1i1p1f1", "resolved")],
        RESOLVED_FACTS if has_usable_horizon else {},
        catalog_crawled_on="2026-07-29",
        window_nominal="1970 / 2010",
        unavailable_windows=[
            {
                "dataset": "INM/INM-CM4-8",
                "scenario": "ssp245",
                "member": "r1i1p1f1",
                "horizon": "far",
                "window_kind": "horizon",
                "reason": "requested window has no overlap",
            }
        ],
    )
    assert rows[0]["status"] == ("partial" if has_usable_horizon else "unavailable")
    assert "far (horizon)" in rows[0]["reason"]
    assert bool(rows[0]["n_hyd_years_reference"]) == has_usable_horizon


def test_resolved_row_carries_the_series_keys_tier_and_both_windows():
    resolved = _rows()[0]
    assert resolved["series_key"] == "cmip6_INM_INM-CM4-8_ssp245_r1i1p1f1"
    assert resolved["reference_series_key"] == "cmip6_INM_INM-CM4-8_historical_r1i1p1f1"
    assert resolved["tier"] == "certified"
    # nominal is what the config asked for; effective is what the hydrological-year
    # windowing actually used. They differ, and the record must show both.
    assert resolved["reference_window_nominal"] == "1970 / 2010"
    assert resolved["reference_window_effective"] == "1970-01-01 / 2009-12-01"
    assert resolved["n_hyd_years_reference"] == 40


def test_skip_row_states_why_and_leaves_resolved_only_columns_empty():
    """A skip must be legible without cross-referencing, and must not fake data."""
    skip = _rows()[1]
    assert skip["status"] == "scenario_not_published"
    assert "SNU/SAM0-UNICON" in skip["reason"]
    # It still names the entry consulted and the snapshot -- that is what makes the
    # skip checkable against the store later.
    assert skip["catalog_entry"].startswith("cmip6_SNU/SAM0-UNICON")
    assert skip["catalog_crawled_on"] == "2026-07-29"
    for column in (
        "series_key",
        "reference_series_key",
        "tier",
        "reference_window_nominal",
        "reference_window_effective",
        "n_hyd_years_reference",
    ):
        assert skip[column] == "", f"{column} must stay empty on a non-resolved row"


def test_written_csv_has_the_design_columns_in_order(tmp_path):
    out = tmp_path / "summary" / "composition.csv"
    write_composition(str(out), _rows())

    with out.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)

    assert header == [name for name, _ in COMPOSITION_CSV_COLUMNS]
    assert len(body) == 2
    # Created its parent: stage B writes into summary/ which need not exist yet.
    assert out.parent.is_dir()


# --- S8-05: the FILE is leaner than the in-memory record ----------------------


def test_the_csv_is_a_projection_of_the_wider_record(tmp_path):
    """`provenance.py` builds its institution roll-up from the full rows, so the
    record keeps fields the CSV drops. Projected on write, not at construction."""
    for field in ("institution", "source_id", "dataset", "catalog_crawled_on"):
        assert field in COMPOSITION_FIELDS
    written = {name for name, _ in COMPOSITION_CSV_COLUMNS}
    for dropped in (
        "institution",
        "dataset",
        "catalog_crawled_on",
        "reference_window_nominal",
        "reference_window_effective",
    ):
        assert dropped not in written


def test_the_csv_reports_the_source_id_under_the_name_model(tmp_path):
    out = tmp_path / "composition.csv"
    write_composition(str(out), _rows())
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["model"] == "INM-CM4-8"
    assert rows[0]["n_reference_years"] == "40"


def test_a_skip_row_stays_empty_in_the_projected_csv(tmp_path):
    out = tmp_path / "composition.csv"
    write_composition(str(out), _rows())
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    skip = rows[1]
    assert skip["status"] == "scenario_not_published"
    for column in ("series_key", "reference_series_key", "tier", "n_reference_years"):
        assert skip[column] == ""
