"""Canonical WF0 declaration, missing-data and table round-trip contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.climate_analysis.diagnostic_outputs import (
    TABLES,
    diagnostic_comparison_outputs,
    diagnostic_source_outputs,
)
from blueearth_cst.climate_analysis.diagnostic_render import load_result, render
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings
from blueearth_cst.climate_analysis.diagnostic_tables import compute, write_tables


def test_short_record_acf_confidence_limits_fit_inside_axes():
    import matplotlib.pyplot as plt

    from blueearth_cst.climate_analysis.diagnostic_figures import View, acf

    view = View(
        sources=["chirps"],
        labels={"chirps": "CHIRPS"},
        styles={"chirps": ("#bf6a3c", "o")},
        results={
            "chirps": {
                "acf": pd.DataFrame(
                    {"r": [0.0, np.nan], "pairs": [10, 0]}, index=[1, 2]
                )
            }
        },
        years=np.array([2000]),
        m0=1,
        reference=(2000, 2000),
        wet_mm=1,
        agreement=pd.DataFrame(),
        common_years=np.array([2000]),
    )
    fig = acf(view)
    low, high = fig.axes[0].get_ylim()
    assert low <= -1.96 / np.sqrt(10) and high >= 1.96 / np.sqrt(10)
    plt.close(fig)


@pytest.mark.parametrize("stale", [False, True])
def test_render_checks_inputs_before_export_then_uses_loaded_snapshot(
    tmp_path, monkeypatch, stale
):
    from blueearth_cst.climate_analysis import diagnostic_render as dr

    settings = parse_settings(None, {"start": 2000, "end": 2005}, ["chirps"])
    daily = pd.DataFrame(
        {"precip": np.nan, "temp": np.nan},
        index=pd.date_range("2000-01-01", "2005-12-31"),
    )
    _, tables = compute(daily, settings, 1, (2000, 2005))
    paths = diagnostic_source_outputs(tmp_path, "chirps", settings)
    write_tables(paths, {"chirps": tables}, settings, 1, (2000, 2005), {})
    paths["figures"] = {
        key: value
        for key, value in paths["figures"].items()
        if key in {"coverage", "anomaly"}
    }
    checks = []

    def load_once(*args, **kwargs):
        assert not checks, "Rendering must not repeat the input check for each PNG"
        checks.append(True)
        if stale:
            raise ValueError("Historical reference is stale")
        return {}

    monkeypatch.setattr(dr, "load_reference", load_once)
    params = dict(
        source_specs=[], selected="chirps", m0=1, completeness={}, shared_inputs={}
    )
    arguments = dict(map_reference="reference.json", map_reference_params=params)
    if stale:
        with pytest.raises(ValueError, match="reference is stale"):
            render(paths, settings, 1, ["chirps"], [tmp_path], **arguments)
        assert not list(tmp_path.rglob("*.png"))
    else:
        records = render(paths, settings, 1, ["chirps"], [tmp_path], **arguments)
        assert len(records) == 2
        assert all(Path(filename).is_file() for filename in paths["figures"].values())
    assert len(checks) == 1


def test_source_summary_preserves_extracted_metadata_and_readiness(tmp_path):
    import xarray as xr

    from blueearth_cst.climate_analysis.diagnostic_source_summary import (
        summarize_sources,
        write_comparison_table,
    )

    stores = {}
    for source, step in (("era5", 0.25), ("chirps", 0.05)):
        ds = xr.Dataset(
            {"precip": (("time", "latitude", "longitude"), np.ones((2, 2, 2)))},
            coords={
                "time": pd.date_range("2001-02-03", periods=2),
                "latitude": [0, step],
                "longitude": [10, 10 + step],
            },
            attrs={"paper_ref": "Source paper", "notes": "Extraction note"},
        )
        stores[source] = tmp_path / f"{source}.nc"
        ds.to_netcdf(stores[source])
    table = summarize_sources(stores)
    assert list(table.columns) == [
        "source",
        "temporal_resolution",
        "time_window",
        "spatial_resolution",
        "reference",
        "remarks",
        "wg1_status",
        "wg1_diffs",
    ]
    assert table.source.tolist() == ["era5", "chirps"]
    assert table.temporal_resolution.tolist() == ["daily", "daily"]
    assert table.time_window.tolist() == ["2001-02-03 → 2001-02-04"] * 2
    assert table.spatial_resolution.tolist() == ["0.25°", "0.05°"]
    assert table.reference.tolist() == ["Source paper"] * 2
    assert table.remarks.tolist() == [
        "Extraction note",
        "Extraction note; precipitation only; no ERA5 companion fields; promotion requires re-extraction",
    ]
    assert table.wg1_status.tolist() == ["not ready"] * 2
    assert "expected data variable 'temp' absent" in table.wg1_diffs.iloc[1]
    csv, md = write_comparison_table(table, tmp_path / "comparison")
    pd.testing.assert_frame_equal(pd.read_csv(csv), table)
    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Gridded climate datasets compared\n\n")
    assert "## WG-1 readiness of extracted stores" in text
    assert (
        "not a check of record length, missing values or scientific suitability" in text
    )
    assert "precipitation only; no ERA5 companion fields" in text
    with pytest.raises(ValueError, match="two sources"):
        summarize_sources({"era5": stores["era5"]})


def test_canonical_inventory_and_one_source_refusal():
    settings = parse_settings(None, {"start": 2000, "end": 2016}, ["era5", "chirps"])
    era5 = diagnostic_source_outputs("era5", "era5", settings)
    chirps = diagnostic_source_outputs("chirps", "chirps", settings)
    comparison = diagnostic_comparison_outputs(
        "comparison", ["era5", "chirps"], settings
    )
    for name, paths in (("era5", era5), ("chirps", chirps), ("comparison", comparison)):
        assert Path(paths["root"]) == Path(name)
        assert all(
            Path(p).parent == Path(name) / "plots" for p in paths["figures"].values()
        )
        assert Path(paths["compute"]["coverage"]).parent == Path(name) / "tables"
        assert Path(paths["compute"]["metadata"]).parent == Path(name) / "_engine"
        assert (
            Path(paths["render"]["captions_md"])
            == Path(name) / "plots" / "figure_captions.md"
        )
    assert comparison["source_summary"] == {
        "dataset_comparison_csv": str(Path("comparison/dataset_comparison.csv")),
        "dataset_comparison_md": str(Path("comparison/dataset_comparison.md")),
    }
    assert [len(x["figures"]) for x in (era5, chirps, comparison)] == [21, 17, 17]
    assert all("percentile" not in x["figures"] for x in (era5, chirps, comparison))
    assert all(
        p.endswith(".png")
        for x in (era5, chirps, comparison)
        for p in x["figures"].values()
    )
    assert "plots/_engine/figure_captions.json" in era5["render"][
        "captions_json"
    ].replace("\\", "/")
    with pytest.raises(ValueError, match="two sources"):
        diagnostic_comparison_outputs("comparison", ["era5"], settings)
    with pytest.raises(ValueError, match="registered"):
        parse_settings(
            {"subbasin_figures": True}, {"start": 2000, "end": 2016}, ["era5"]
        )


@pytest.mark.parametrize("kind", ["zero", "constant", "missing", "no_overlap"])
def test_unavailable_records_keep_table_headers_and_round_trip(tmp_path, kind, capsys):
    idx = pd.date_range("2000-01-01", "2005-12-31")
    p = np.zeros(len(idx)) if kind == "zero" else np.ones(len(idx))
    if kind == "missing":
        p[:] = np.nan
    daily = pd.DataFrame({"precip": p, "temp": np.nan}, index=idx)
    settings = parse_settings(None, {"start": 2000, "end": 2005}, ["chirps"])
    period = (1990, 1995) if kind == "no_overlap" else (2000, 2005)
    _, tables = compute(daily, settings, 1, period)
    assert set(tables) == set(TABLES)
    assert all(len(t.columns) for t in tables.values())
    paths = diagnostic_source_outputs(tmp_path, "chirps", settings)
    write_tables(paths, {"chirps": tables}, settings, 1, period, {})
    assert all(Path(paths["compute"][k]).is_file() for k in TABLES)
    loaded = load_result(paths["root"], "chirps")
    assert loaded["tclim"] is None
    if kind in {"missing", "no_overlap"}:
        paths["figures"].pop("map-precip")
        records = render(paths, settings, 1, ["chirps"], [paths["root"]])
        assert len(records) == 16
        titles = json.loads(
            (tmp_path / "plots/_engine/figure_titles.json").read_text(encoding="utf-8")
        )["figures"]
        assert set(titles) == set(records)
        assert all("CHIRPS" in entry["title"] for entry in titles.values())
        assert (tmp_path / "plots/figure_titles.md").is_file()
        assert not any("monthly_percentile" in path for path in records)
        assert all((Path(paths["root"]) / p).is_file() for p in records)
        series = next(r for p, r in records.items() if "annual_trend_ts" in p)
        assert series["availability"].startswith("Unavailable")
        assert "no usable data or reference fit" in series["caption"]
        output = capsys.readouterr().out
        assert (
            "Wrote 16 diagnostic figures for chirps → plots/_engine/figure_captions.json"
        ) in output
        assert "no native temperature" in output


def test_opt_in_subbasin_tables_pngs_and_caption_inventory(tmp_path):
    import json
    from types import SimpleNamespace

    from blueearth_cst.climate_analysis.diagnostic_subbasins import render_subbasins
    from blueearth_cst.climate_analysis.diagnostic_tables import write_json

    settings = parse_settings({}, {"start": 2000, "end": 2016}, ["era5", "chirps"])
    settings["subbasin_figures"] = True
    folders = []
    idx = pd.date_range("2000-01-01", "2016-12-31")
    rng = np.random.default_rng(3)
    for source in ("era5", "chirps"):
        folder = tmp_path / source / "tables" / "subbasins"
        folders.append(str(folder))
        sub = folder / "subbasin_101"
        sub.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "precip": rng.gamma(0.6, 8, len(idx)),
                "temp": rng.normal(25, 2, len(idx)) if source == "era5" else np.nan,
            },
            index=idx,
        )
        frame.index.name = "date"
        frame.to_csv(sub / "daily_basin.csv")
        _, tables = compute(frame, settings, 1, (2000, 2016))
        for name in (
            "precip_climatology",
            "temp_climatology",
            "annual_indices",
            "trends",
        ):
            tables[name].to_csv(sub / f"{name}.csv", index=False)
    for sources in (["era5"], ["era5", "chirps"]):
        root = tmp_path / ("era5" if len(sources) == 1 else "comparison")
        write_json(
            root / "plots" / "_engine" / "figure_captions.json",
            {"schema_version": 1, "figures": {}},
        )
        output = SimpleNamespace(
            subbasin_figures=str(root / "plots" / "subbasins"),
            subbasin_tables=str(root / "tables" / "subbasins"),
        )
        sm = SimpleNamespace(
            params=SimpleNamespace(
                subbasin_table_dirs=folders[: len(sources)], m0=1, annotation="caption"
            ),
            output=output,
        )
        render_subbasins(sm, settings, sources)
        assert (root / "plots" / "figure_captions.md").is_file()
        assert not (root / "figure_captions.md").exists()
        records = json.loads(
            (root / "plots" / "_engine" / "figure_captions.json").read_text()
        )["figures"]
        assert len(records) == (3 if len(sources) == 1 else 2)
        assert all(
            (root / path).is_file() and r["captioned_path"] is None
            for path, r in records.items()
        )
        assert (
            root / "tables" / "subbasins" / "subbasin_101" / "annual_indices.csv"
        ).is_file()


def test_exact_basin_mask_and_unresolved_temperature_units(tmp_path):
    import xarray as xr

    from blueearth_cst.climate_analysis.diagnostic_tables import daily_series

    ds = xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), np.ones((3, 2, 2))),
            "temp": (("time", "latitude", "longitude"), np.full((3, 2, 2), 300.0)),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=3),
            "latitude": [0.0, 1.0],
            "longitude": [10.0, 11.0],
        },
    )
    ds.temp.attrs["units"] = "K"
    nc = tmp_path / "climate.nc"
    ds.to_netcdf(nc)
    cells = tmp_path / "basin_cells.csv"
    pd.DataFrame({"latitude": [0.0], "longitude": [10.0]}).to_csv(cells, index=False)
    frame, lineage = daily_series(nc, cells, "era5")
    assert frame.precip.eq(1.0).all() and frame.temp.isna().all()
    assert "era5: temperature diagnostics unavailable" in lineage["warnings"][0]
    pd.DataFrame({"latitude": [0.0, 3.0], "longitude": [10.0, 10.0]}).to_csv(
        cells, index=False
    )
    with pytest.raises(ValueError, match="every declared cell"):
        daily_series(nc, cells, "era5")
