"""Synthetic suite render and recoverable-caption contract falsifiers."""

import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import projection_analysis as analysis
from blueearth_cst.projections import projection_figures as contract
from blueearth_cst.projections import projection_trajectories as trajectories
from blueearth_cst.projections.plot_proj_timeseries import figure_counts, render_suite
from blueearth_cst.projections.projection_captions import SUITE_NOTE, caption
from blueearth_cst.projections.projection_suite_figures import render_product


@pytest.fixture(scope="module")
def _suite_template():
    def series(start, scenario, offset):
        time = pd.date_range(f"{start}-01-01", periods=360, freq="MS")
        wave = np.repeat(np.sin(np.arange(30)), 12)
        return xr.Dataset(
            {
                "temp": ("time", 10 + offset + wave, {"units": "degC"}),
                "precip": ("time", 2 + offset / 10 + wave / 10, {"units": "mm/day"}),
            },
            coords={"time": time},
            attrs={"cst_calendar": "360_day"},
        ).expand_dims(
            model=["TEST/A"], scenario=[scenario], member=["r1"], clim_project=["cmip6"]
        )

    hist, future = series(1985, "historical", 0), series(2036, "ssp245", 2)
    identity = {"model": "TEST/A", "scenario": "ssp245", "member": "r1"}
    change = analysis.analysis_change_table(
        hist,
        future,
        identity=identity,
        window_id="2036-2065",
        reference_window=(1985, 2014),
        future_window=(2036, 2065),
    )
    changes = pd.concat(
        [
            change.assign(model=model, member=member)
            for model in ("TEST/A", "TEST/B")
            for member in ("r1", "r2")
        ],
        ignore_index=True,
    )
    models = analysis.model_changes(changes, statistic=None)
    ratios = analysis.member_null_bands(analysis.member_ratios(changes))
    anomalies = trajectories.member_anomalies(
        hist, future, identity=identity, reference_window=(1985, 2014)
    )
    tables = {
        "coverage": pd.DataFrame(
            [
                dict(
                    model=model,
                    source_id=model.split("/")[-1],
                    institution="TEST",
                    family="TEST",
                    family_source="listed",
                    country="",
                    reference_series_key=f"{model}/{member}",
                    scenario="ssp245",
                    member=member,
                    available=True,
                )
                for model in ("TEST/A", "TEST/B")
                for member in ("r1", "r2")
            ]
        ),
        "model_changes": models,
        "change_summary": analysis.ensemble_summary(models),
        "sign_agreement": analysis.sign_agreement(models),
        "climatology": analysis.climatology(changes),
        "climatology_summary": analysis.climatology_summary(
            analysis.climatology(changes)
        ),
        "variability_member_ratios": ratios,
        "variability_model_ratios": analysis.model_ratios(ratios, include_null=False),
        "variability_summary": analysis.ratio_summary(ratios),
        "member_anomalies": anomalies,
        "trajectory_range": trajectories.trajectory_range(
            trajectories.model_trajectories(anomalies)
        ),
    }
    tables["seasonal_values"] = analysis.seasonal_values(tables["climatology"], changes)
    tables["seasonal_values_summary"] = analysis.ensemble_summary(
        tables["seasonal_values"]
    )
    tables["availability"] = analysis.product_availability(
        {
            "coverage": tables["coverage"],
            "trajectories": anomalies,
            "models": models,
            "climatology": tables["climatology"],
            "member_ratios": ratios,
            "model_ratios": tables["variability_model_ratios"],
        }
    )
    record = {
        "models": [{"model": "TEST/A"}, {"model": "TEST/B"}],
        "reference": [1985, 2014],
        "windows": [
            {
                "window_id": "2036-2065",
                "used": [2036, 2065],
                "reference": [1985, 2014],
                "configured": {
                    "mid": {"reference": [1990, 2010], "future": [2046, 2054]}
                },
                "horizons": ["mid"],
                "requested": [2036, 2065],
                "shifted": False,
                "joined": False,
            }
        ],
    }
    logs = tables["variability_model_ratios"].copy()
    logs["value"] = logs.log_value
    summary = analysis.ensemble_summary(logs)
    for column in ("median", "min", "max", "q25", "q75"):
        summary[column] = np.exp(summary[column])
    record["variability_model_summary"] = summary.to_dict("records")
    return tables, record


@pytest.fixture
def suite(_suite_template):
    """Reuse analysis, but keep each test's table and record mutations isolated."""
    tables, record = _suite_template
    return {name: table.copy(deep=True) for name, table in tables.items()}, deepcopy(
        record
    )


def test_seasonal_grid_explains_unavailable_seasons(suite):
    tables, record = suite
    models = tables["model_changes"].copy()
    models.loc[models.period.isin(["DJF", "MAM", "JJA", "SON"]), "value"] = np.nan
    tables = {
        **tables,
        "model_changes": models,
        "change_summary": analysis.ensemble_summary(models),
    }
    fig = render_product(
        "cmip6_temp_annual_seasonal_change_box_basin_avg.png",
        tables,
        record,
        window_id="2036-2065",
    )
    assert len(fig.axes) == 4  # one panel per season, no annual panel
    for ax in fig.axes:
        assert [text.get_text() for text in ax.texts] == [
            "No eligible\nmodel estimates"
        ]
    plt.close(fig)


def test_seasonal_grid_drops_sparse_scenarios_beside_full_ones():
    from blueearth_cst.projections.projection_suite_figures import shown_scenarios

    rows = [
        dict(model=f"M/{i}", scenario=s, value=1.0)
        for s, n in (("ssp245", 12), ("ssp460", 4))
        for i in range(n)
    ]
    assert shown_scenarios(pd.DataFrame(rows)) == (["ssp245"], {"ssp460": 4})
    small = pd.DataFrame([r for r in rows if r["model"] in ("M/0", "M/1")])
    assert shown_scenarios(small) == (["ssp245", "ssp460"], {})


def test_each_artist_renders_with_units_and_without_cloud_labels(suite):
    tables, record = suite
    for row in tables["availability"].to_dict("records"):
        if not row["available"]:
            continue
        fig = render_product(
            row["filename"], tables, record, window_id=row["window_id"]
        )
        fig.canvas.draw()
        assert all(not ax.get_title(loc="center") for ax in fig.axes)
        if (
            row["filename"]
            == "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png"
        ):
            legend = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
            assert legend.x0 >= 0 and legend.x1 <= fig.bbox.width
        if row["filename"] == "cmip6_precip_temp_annual_change_cloud_basin_avg.png":
            assert all(not ax.texts for ax in fig.axes)
            assert fig.axes[0].get_box_aspect() == 1
        plt.close(fig)


@pytest.mark.parametrize("annotation", ["none", "caption", "title"])
def test_manifest_bijection_unavailable_and_annotation_mode(
    suite, tmp_path, annotation, capsys
):
    tables, record = suite
    # Keep one real available product to exercise selected-mode export.
    tables["availability"].loc[
        tables["availability"].filename
        != "cmip6_temp_annual_seasonal_change_box_basin_avg.png",
        "available",
    ] = False
    tables["availability"].loc[~tables["availability"].available, "reason"] = (
        "synthetic unavailable product"
    )
    plots = tmp_path / "plots"
    manifest, skipped = render_suite(tables, record, plots, annotation=annotation)
    assert len(manifest) == 1 and len(skipped) == 19
    console = capsys.readouterr().out
    assert "projection_plots - Rendering windows:" in console
    assert "; reference: " in console
    assert "Wrote 1 figure; 19 skipped; reasons → figure_captions.md" in console
    assert set(manifest) == {
        "period_2036_2065/cmip6_temp_annual_seasonal_change_box_basin_avg.png"
    }
    assert (plots / next(iter(manifest))).is_file()
    assert not (plots / "captioned").exists()
    assert (
        json.loads((tmp_path / "_engine/figure_captions.json").read_text()) == manifest
    )
    titles = json.loads(
        (tmp_path / "_engine/figure_titles.json").read_text(encoding="utf-8")
    )["figures"]
    assert set(titles) == set(manifest)
    assert all(item["title"] for item in titles.values())
    assert "2036–2065 relative to" in next(iter(titles.values()))["title"]
    assert (plots / "figure_titles.md").is_file()
    assert not (plots / "_engine").exists()
    assert "Not produced" in (plots / "figure_captions.md").read_text()
    stale = plots / "full_period/stale.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="owned PNGs differ"):
        render_suite(tables, record, plots, annotation=annotation)


def test_captions_and_distinct_folder_contract(suite):
    _, record = suite
    text = caption(
        "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
        record,
        "2036-2065",
        {"n_excluded": 3},
    )
    for phrase in (
        "2036–2065",
        "1985–2014",
        "precipitation (standard deviation, b)",
        "3 degenerate ratios are excluded.",
    ):
        assert phrase in text
    assert "zero future" not in text  # exclusion counts appear only when nonzero
    assert "not a confidence interval" in SUITE_NOTE
    horizons = {"far": [2071, 2100], "later": [2081, 2100]}
    assert contract.suite_folders(horizons) == [
        "full_period",
        "period_2071_2100",
        "period_2081_2100",
    ]
    assert sum(len(p["filenames"]) for p in contract.SUITE_PRODUCTS.values()) == 20


@pytest.mark.parametrize(
    "filename, units",
    [
        ("seasonal-cycle-pet.png", "mm/day"),
        ("change-pet-pct.png", "%"),
        ("change-pet.png", "mm, period integrals"),
    ],
)
def test_pet_window_captions_stand_alone(suite, filename, units):
    from blueearth_cst.projections.variable_spec import parse

    _, record = suite
    record["variable_spec"] = parse({"pet": {"method": "debruin"}})
    window = record["windows"][0]
    window.update(
        reference=[1971, 2000],
        used=[2010, 2039],
        requested=[2005, 2034],
        shifted=True,
        joined=True,
    )
    window.pop("horizon_windows", None)
    text = caption(
        contract.FIGURE_FILENAMES[filename],
        record,
        window["window_id"],
        {"n_models": 7, "by_scenario": {"ssp245": 5, "ssp585": 7}},
    )
    for phrase in (
        "1971–2000",
        "2010–2039",
        "Based on 7 models.",
        "Number of models: SSP2-4.5, 5; SSP5-8.5, 7.",
        "shifted from [2005, 2034] to [2010, 2039]",
        "Historical and scenario simulations are concatenated",
        "HydroMT debruin",
        "monthly-input approximation",
        "buffered bounding box",
        units,
    ):
        assert phrase in text
    assert "1985–2014" not in text  # use this window's actual reference


def test_pet_trajectory_caption_has_reference_shading_and_window_notes(suite):
    from blueearth_cst.projections.variable_spec import parse

    _, record = suite
    record["variable_spec"] = parse({"pet": {"method": "makkink"}})
    record["windows"] = [
        {
            "window_id": "2010-2039",
            "used": [2010, 2039],
            "horizon_windows": {
                "near": {
                    "requested": [2005, 2034],
                    "used": [2010, 2039],
                    "shifted": True,
                    "joined": True,
                }
            },
        },
        {"window_id": "2071-2100", "used": [2071, 2100]},
    ]
    text = caption(
        "cmip6_pet_annual_anomaly_ts_basin_avg.png",
        record,
        counts={"n_models": 3, "by_scenario": {"historical": 3, "ssp245": 2}},
    )
    for phrase in (
        "1985–2014 mean annual total",
        "grey shading marks the future analysis windows (2010–2039; 2071–2100)",
        "historical/scenario transition (2014/2015)",
        "Based on 3 models.",
        "Number of models: Historical, 3; SSP2-4.5, 2.",
        "window (near) was shifted",
        "concatenated to complete the window (near)",
        "HydroMT makkink",
        "mm/year",
    ):
        assert phrase in text


def test_pet_opt_out_retires_previous_horizon_without_deleting_other_files(
    suite, tmp_path
):
    from blueearth_cst.projections.variable_spec import parse

    tables, record = suite
    original_availability = tables["availability"].copy()
    for name, table in list(tables.items()):
        if "variable" in table:
            pet = table.loc[table.variable.eq("temp")].assign(variable="pet")
            tables[name] = pd.concat([table, pet], ignore_index=True)
    record["variable_spec"] = parse(
        {"pet": {"method": "makkink", "change": "absolute"}}
    )
    tables["availability"]["available"] = False
    old_folder = "period_2036_2065"
    pet_rows = [
        dict(
            filename=contract.FIGURE_FILENAMES[key],
            folder="full_period" if key.startswith("trajectory-") else old_folder,
            window_id="" if key.startswith("trajectory-") else "2036-2065",
            available=key == "seasonal-cycle-pet.png",
            reason="synthetic unavailable product",
        )
        for key in ("trajectory-pet.png", "seasonal-cycle-pet.png", "change-pet.png")
    ]
    tables["availability"] = pd.concat(
        [tables["availability"], pd.DataFrame(pet_rows)], ignore_index=True
    )
    plots = tmp_path / "plots"
    manifest, _ = render_suite(tables, record, plots)
    pet_path = plots / old_folder / contract.FIGURE_FILENAMES["seasonal-cycle-pet.png"]
    assert set(manifest) == {f"{old_folder}/{pet_path.name}"}
    assert pet_path.is_file()  # exercise real publication before opt-out

    preserved = [
        plots
        / old_folder
        / contract.FIGURE_FILENAMES["seasonal-cycle-temperature.png"],
        plots / old_folder / "unrelated.png",
        plots
        / old_folder
        / contract.FIGURE_FILENAMES["change-pet.png"],  # unmanaged PET
        tmp_path / "outside" / pet_path.name,
        tmp_path / "raw" / "source.nc",
    ]
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve these bytes")
    manifest_path = tmp_path / "_engine/figure_captions.json"
    manifest[
        f"../outside/{pet_path.name}"
    ] = {}  # untrusted manifest must not escape root
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record.pop("variable_spec")
    record["windows"][0].update(window_id="2071-2100", used=[2071, 2100])
    tables["availability"] = original_availability
    tables["availability"]["available"] = False
    period = tables["availability"].folder.eq(old_folder)
    tables["availability"].loc[period, "folder"] = "period_2071_2100"
    tables["availability"].loc[period, "window_id"] = "2071-2100"
    refreshed, _ = render_suite(tables, record, plots)
    assert not pet_path.exists()
    assert refreshed == json.loads(manifest_path.read_text()) == {}
    assert all(path.read_bytes() == b"preserve these bytes" for path in preserved)


def test_unplotted_statistics_do_not_enable_mean_figures(suite):
    tables, _ = suite
    models = tables["model_changes"].loc[tables["model_changes"].statistic != "mean"]
    result = analysis.product_availability({"models": models}, ["2036-2065"])
    mean_products = result.loc[
        result.filename.isin(
            [
                "cmip6_temp_annual_seasonal_change_box_basin_avg.png",
                "cmip6_precip_annual_seasonal_change_pct_box_basin_avg.png",
                "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png",
                "cmip6_precip_temp_annual_change_cloud_basin_avg.png",
            ]
        )
    ]
    assert not mean_products.available.any()


def test_driver_csv_roundtrip_normalizes_full_period_window(
    suite, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from blueearth_cst.projections import plot_proj_timeseries as driver

    tables, record = suite
    files = []
    for name, table in tables.items():
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(str(path))
    record_path = tmp_path / "analysis_record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    plots = tmp_path / "plots"
    captured = {}
    monkeypatch.setattr(
        driver,
        "render_suite",
        lambda tables, *args, **kwargs: captured.update(tables=tables, kwargs=kwargs),
    )
    driver.run(
        SimpleNamespace(
            input=SimpleNamespace(tables=files, record=str(record_path)),
            output=SimpleNamespace(
                captions_md=str(plots / "figure_captions.md"),
                captions_json=str(tmp_path / "_engine/figure_captions.json"),
                folders=[str(plots / "full_period"), str(plots / "period_2036_2065")],
            ),
            params=SimpleNamespace(annotation="title"),
        )
    )
    available = captured["tables"]["availability"]
    assert captured["kwargs"]["annotation"] == "title"
    assert captured["kwargs"]["captions_json"] == str(
        tmp_path / "_engine/figure_captions.json"
    )
    assert available.loc[available.folder == "full_period", "window_id"].eq("").all()
    assert available.available.dtype == bool


def test_precipitation_cycle_percentage_and_two_row_annual_content(suite):
    tables, record = suite
    cycle = render_product(
        "cmip6_precip_monthly_clim_basin_avg.png", tables, record, window_id="2036-2065"
    )
    change = (
        tables["change_summary"]
        .loc[
            (tables["change_summary"].variable == "precip")
            & (tables["change_summary"].statistic == "mean")
            & (tables["change_summary"].quantity == "change_pct")
        ]
        .set_index("period")
    )
    assert len(cycle.axes) == 1
    assert cycle.axes[0].get_ylabel() == "mm d⁻¹"
    monthly = render_product(
        "cmip6_precip_monthly_change_pct_box_basin_avg.png",
        tables,
        record,
        window_id="2036-2065",
    )
    medians = [
        line.get_ydata()[0]
        for line in monthly.axes[0].lines
        if len(line.get_xdata()) == 2 and np.isclose(np.ptp(line.get_xdata()), 0.6)
    ]
    np.testing.assert_allclose(medians, change.loc[list(analysis.MONTHS), "median"])
    plt.close(cycle)
    plt.close(monthly)
    annual = render_product(
        "cmip6_precip_annual_overview_basin_avg.png", tables, record
    )
    assert len(annual.axes) == 2
    assert "mm yr" in annual.axes[0].get_ylabel()
    assert "1985–2014 (%)" in annual.axes[1].get_ylabel()
    assert all(
        any(p.get_gid() == "window:2036-2065" for p in ax.patches) for ax in annual.axes
    )
    assert all(
        any(line.get_gid() == "scenario-handover" for line in ax.lines)
        for ax in annual.axes
    )
    assert any(line.get_gid() == "historical-median" for line in annual.axes[0].lines)
    plt.close(annual)


def test_cloud_pairs_derived_models_independent_of_member_labels(suite):
    tables, record = suite
    models = tables["model_changes"].copy()
    models.loc[models.model == "TEST/A", "members"] = "r1"
    models.loc[models.model == "TEST/B", "members"] = "r9 r10"
    unpaired = models.loc[
        (models.model == "TEST/A") & (models.variable == "temp")
    ].assign(model="TEST/unpaired")
    tables["model_changes"] = pd.concat([models, unpaired], ignore_index=True)
    fig = render_product(
        "cmip6_precip_temp_annual_change_cloud_basin_avg.png",
        tables,
        record,
        window_id="2036-2065",
    )
    assert len(fig.axes[0].collections[0].get_offsets()) == 2
    assert not fig.axes[0].texts
    plt.close(fig)


def test_historical_only_trajectory_preserves_historical_data(suite):
    tables, record = suite
    for name in ("member_anomalies", "trajectory_range"):
        tables[name] = tables[name].loc[tables[name].year < 2014]
    fig = render_product("cmip6_temp_annual_anomaly_ts_basin_avg.png", tables, record)
    assert any(np.isfinite(line.get_ydata()).any() for line in fig.axes[0].lines)
    plt.close(fig)


def test_caption_counts_exclude_unplotted_variability_and_monthly_strips():
    rows = pd.DataFrame(
        [
            dict(
                model="TEST/A",
                scenario="ssp245",
                window_id="2036-2065",
                period="Jan",
                variable="temp",
                statistic="std_detrended",
                quantity="change",
                value=2.0,
                ratio=2.0,
                variability_eligible=True,
            ),
            dict(
                model="TEST/B",
                scenario="ssp245",
                window_id="2036-2065",
                period="annual",
                variable="precip",
                statistic="std_detrended",
                quantity="change",
                value=2.0,
                ratio=2.0,
                variability_eligible=True,
            ),
        ]
    )
    tables = {
        "variability_member_ratios": rows,
        "variability_model_ratios": rows,
        "model_changes": rows.assign(statistic="mean"),
    }
    assert (
        figure_counts(
            "cmip6_temp_annual_seasonal_change_box_basin_avg.png", tables, "2036-2065"
        )["n_models"]
        == 0
    )
    assert (
        figure_counts(
            "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
            tables,
            "2036-2065",
        )["n_models"]
        == 1
    )
    tables["variability_member_ratios"] = rows.assign(period="annual")
    assert (
        figure_counts(
            "cmip6_precip_temp_annual_member_ratio_box_basin_avg.png",
            tables,
            "2036-2065",
        )["n_models"]
        == 2
    )


@pytest.mark.parametrize(
    "period,variable,quantity",
    [
        ("Jan", "temp", "change"),
        ("annual", "kin", "change"),
        ("annual", "temp", "change_pct"),
    ],
)
def test_agreement_requires_a_plotted_cell(period, variable, quantity):
    rows = pd.DataFrame(
        [
            dict(
                model=model,
                scenario="ssp245",
                window_id="2036-2065",
                period=period,
                variable=variable,
                quantity=quantity,
                statistic="mean",
                value=2.0,
            )
            for model in ("TEST/A", "TEST/B")
        ]
    )
    available = analysis.product_availability({"models": rows}, ["2036-2065"])
    assert not available.set_index("filename").loc[
        "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png", "available"
    ]
    assert (
        figure_counts(
            "cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png",
            {"model_changes": rows},
            "2036-2065",
        )["n_models"]
        == 0
    )


def test_mostly_dry_periods_are_shaded_and_captioned():
    from blueearth_cst.projections.projection_suite_figures import dry_shading

    def summary(kept, excluded):
        return pd.DataFrame(
            [
                dict(
                    window_id="2036-2065",
                    variable="precip",
                    statistic="mean",
                    quantity="change_pct",
                    period=period,
                    scenario="ssp245",
                    n_models=kept if period == "JJA" else 10,
                    n_models_excluded=excluded if period == "JJA" else 0,
                )
                for period in ("annual", "DJF", "MAM", "JJA", "SON")
            ]
        )

    pct = "cmip6_precip_annual_seasonal_change_pct_box_basin_avg.png"
    record = {"reference": [1985, 2014], "windows": []}
    assert dry_shading(pct, {"change_summary": summary(5, 5)}, record, "2036-2065")
    assert not dry_shading(pct, {"change_summary": summary(6, 4)}, record, "2036-2065")
    temperature = "cmip6_temp_annual_seasonal_change_box_basin_avg.png"
    assert not dry_shading(
        temperature, {"change_summary": summary(1, 9)}, record, "2036-2065"
    )
    window = {"window_id": "2036-2065", "used": [2036, 2065], "reference": [1985, 2014]}
    record["windows"] = [window]
    text = caption(pct, record, "2036-2065", {"n_models": 10, "mostly_dry": True})
    assert "Hatched panels mark seasons" in text
    assert "Gray" not in caption(pct, record, "2036-2065", {"n_models": 10})


def test_variability_band_scales_reference_noise_and_takes_model_medians():
    base = dict(
        scenario="ssp245",
        window_id="2036-2065",
        reference_start=1985,
        reference_end=2014,
        future_start=2036,
        future_end=2065,
        period="annual",
        variable="precip",
        statistic="mean",
        n_reference=20,
        n_future=21,
        mean_change_eligible=True,
    )
    rows = []
    for model, sd, mean in (("A/a", 0.1, 1.0), ("B/b", 0.2, 1.0), ("C/c", 0.3, 1.0)):
        for quantity, units in (("change", "mm/day"), ("change_pct", "%")):
            rows.append(
                dict(
                    base,
                    model=model,
                    quantity=quantity,
                    units=units,
                    null_reference_sd=sd,
                    null_reference_mean=mean,
                )
            )
    band = analysis.variability_band(pd.DataFrame(rows)).set_index("quantity")
    expected = 1.6448536 * 0.2 * np.sqrt(1 / 20 + 1 / 21)  # median model, sd 0.2
    assert band.loc["change", "noise_halfwidth"] == pytest.approx(expected)
    assert band.loc["change_pct", "noise_halfwidth"] == pytest.approx(expected * 100)
