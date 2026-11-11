"""Independent native-cell annual anomaly and retained display contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis.annual_spatial_anomalies import (
    anomaly_paths,
    file_hash,
    load_retained,
    parse_settings,
    prepare,
)
from blueearth_cst.climate_analysis.annual_spatial_render import (
    prepare_display,
    render_pages,
    retained_enabled,
    retire_pages,
)
from blueearth_cst.climate_analysis.diagnostic_outputs import diagnostic_source_outputs
from blueearth_cst.climate_analysis.diagnostic_tables import write_json


def settings(**changes):
    return parse_settings(
        {
            "reference_period": {"start": 2000, "end": 2001},
            "comparison_period": {"start": 1999, "end": 2006},
            "annual_spatial_anomalies": True,
            **changes,
        },
        {"start": 2000, "end": 2006},
        ["era5", "chirps"],
    )


@pytest.mark.parametrize("multiple_catalogs", [False, True])
def test_run_hashes_complete_catalog_paths(tmp_path, monkeypatch, multiple_catalogs):
    from types import SimpleNamespace

    from blueearth_cst.climate_analysis import annual_spatial_anomalies as module

    paths = [
        tmp_path / name
        for name in ("climate.nc", "cells.csv", "catalog.yml", "other.yml")
    ]
    for path in paths:
        path.write_text("input", encoding="utf-8")
    catalogs = [str(path) for path in paths[2:]] if multiple_catalogs else str(paths[2])
    sm = SimpleNamespace(
        input=SimpleNamespace(climate_nc=str(paths[0]), basin_cells=str(paths[1])),
        params=SimpleNamespace(source="era5", data_sources=catalogs),
    )
    hashed = []
    original_hash = module.file_hash

    def track_hash(path):
        hashed.append(str(path))
        return original_hash(path)

    def stop_after_hashing(*args, **kwargs):
        assert hashed == [str(path) for path in paths[: 4 if multiple_catalogs else 3]]
        raise RuntimeError("catalog hashing completed")

    monkeypatch.setattr(module, "file_hash", track_hash)
    monkeypatch.setattr(module, "daily_series", stop_after_hashing)
    with pytest.raises(RuntimeError, match="catalog hashing completed"):
        module.run(sm)


def raw(m0=1):
    index = pd.date_range("2000-01-01", "2006-12-31")
    # Explicit independent start-year offsets, a heterogeneous 2x2 native grid.
    year = index.year - (index.month < m0).astype(int)
    p = (year - 1998).to_numpy()[:, None, None] * np.array([[1.0, 2.0], [3.0, 4.0]])
    t = (year - 2000).to_numpy()[:, None, None] + np.array([[10.0, 20.0], [30.0, 40.0]])
    return xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), p, {"units": "mm/day"}),
            "temp": (("time", "latitude", "longitude"), t, {"units": "degC"}),
        },
        coords={"time": index, "latitude": [1.0, 0.0], "longitude": [10.0, 11.0]},
    )


def retain(root, ds, cfg, source="era5", m0=1):
    fields, record = prepare(ds, cfg, m0, source)
    paths = anomaly_paths(root)
    from pathlib import Path

    Path(paths["fields"]).parent.mkdir(parents=True, exist_ok=True)
    fields.to_netcdf(paths["fields"])
    record["fields_sha256"] = file_hash(paths["fields"])
    write_json(paths["record"], record)
    return fields, record


@pytest.mark.parametrize("m0,days0,days1", [(1, 366, 365), (10, 365, 365)])
def test_cellwise_mean_reference_leap_water_year_and_fixed_support(m0, days0, days1):
    ds = raw(m0)
    fields, record = prepare(ds, settings(), m0, "era5")
    factors = np.array([[1, 2], [3, 4]])
    baseline = ((2 * days0 + 3 * days1) / 2) * factors
    np.testing.assert_allclose(fields.precip_reference_mean, baseline)
    np.testing.assert_allclose(fields.temp_reference_mean, [[10.5, 20.5], [30.5, 40.5]])
    np.testing.assert_allclose(
        fields.precip_anomaly.sel(reporting_year=2002), 4 * 365 * factors - baseline
    )
    assert fields.precip_anomaly.sel(reporting_year=1999).isnull().all()
    assert record["pages"] == [list(range(1999, 2005)), [2005, 2006]]
    assert record["year_dates"]["2000"][0] == f"2000-{m0:02d}-01"
    # Losing one reference cell-year invalidates every anomaly at that cell.
    ds.precip.loc[
        {"time": slice("2001-10-01", "2001-11-01"), "latitude": 1.0, "longitude": 10.0}
    ] = np.nan
    fields, _ = prepare(ds, settings(), m0, "era5")
    assert fields.precip_reference_count.sel(latitude=1.0, longitude=10.0) == 1
    assert fields.precip_anomaly.sel(latitude=1.0, longitude=10.0).isnull().all()
    assert fields.precip_anomaly.sel(latitude=0.0, longitude=11.0).notnull().any()


def test_absent_dates_tolerated_infill_strict_missing_and_temperature_ownership():
    ds = raw().drop_sel(time=pd.Timestamp("2000-02-29"))
    fields, _ = prepare(ds, settings(), 1, "era5")
    assert (
        fields.precip_annual.sel(reporting_year=2000, latitude=1.0, longitude=10.0)
        == 732
    )
    strict = settings(max_missing_days_month=0, max_missing_days_year=0)
    fields, _ = prepare(ds, strict, 1, "era5")
    assert fields.precip_reference_count.max() == 1
    assert fields.precip_anomaly.isnull().all()
    fields, _ = prepare(ds, settings(), 1, "chirps")
    assert not any(name.startswith("temp") for name in fields)
    fields, _ = prepare(ds, settings(), 1, "era5", temperature_qualified=False)
    assert fields.temp_anomaly.isnull().all()


def test_partial_reference_exclusion_one_year_and_no_reference():
    cfg = settings(reference_period={"start": 1999, "end": 2000})
    with pytest.warns(UserWarning, match="incomplete/out-of-span.*1999"):
        _, record = prepare(raw(), cfg, 1, "era5")
    assert record["reference_eligible"] == [2000]
    assert record["single_reference_year"]
    with pytest.raises(ValueError, match="no full reporting year"):
        prepare(raw().isel(time=slice(0, 20)), settings(), 1, "era5")


@pytest.mark.parametrize("value", [1, "true", {}, None])
def test_opt_in_requires_boolean(value):
    with pytest.raises(ValueError, match="annual_spatial_anomalies.*boolean"):
        settings(annual_spatial_anomalies=value)


def test_display_scales_native_singletons_and_out_of_frame_extremes(tmp_path):
    cfg = settings()
    ds = raw().isel(latitude=[0], longitude=[0])
    for var in ds:
        ds[var].attrs.update(latitude_resolution=0.25, longitude_resolution=0.5)
    fields, _ = retain(tmp_path / "one", ds, cfg)
    ds2 = raw().isel(latitude=[0]).assign_coords(longitude=[10.0, 20.0])
    for var in ds2:
        ds2[var].attrs.update(latitude_resolution=0.25)
    ds2.precip.values[:, :, 1] *= 1.0e6  # unseen buffer extreme
    retain(tmp_path / "two", ds2, cfg, "chirps")
    reference = {
        "frame": {
            "extent": [9.75, 10.25, 0.875, 1.125],
            "longitude_ticks": [10.0],
            "latitude_ticks": [1.0],
            "aspect": 1.0,
        }
    }
    loaded, scales = prepare_display(
        [tmp_path / "one", tmp_path / "two"], reference, {}, cfg
    )
    expected = float(
        np.abs(fields.precip_anomaly.sel(reporting_year=list(range(1999, 2007)))).max()
    )
    assert scales["precip"]["maximum_absolute"] == expected
    assert len(loaded) == 2
    # Stored footprint must survive singleton latitude AND longitude.
    from blueearth_cst.shared.map_reference import field_edges

    x, y = field_edges(fields.precip_anomaly)
    np.testing.assert_allclose(x, [9.75, 10.25])
    np.testing.assert_allclose(y, [0.875, 1.125])
    assert "temp" not in loaded["chirps"][1]["variables"]


def test_retained_rerender_all_pages_annotations_and_managed_cleanup(
    tmp_path, monkeypatch
):
    cfg = settings()
    root = tmp_path / "source"
    retain(root, raw(), cfg)
    reference = {
        "frame": {
            "extent": [9.5, 11.5, -0.5, 1.5],
            "longitude_ticks": [10.0, 11.0],
            "latitude_ticks": [0.0, 1.0],
            "aspect": 1.0,
        }
    }
    paths = diagnostic_source_outputs(root, "era5", cfg)
    saved = []

    def save(path, *, fig, **kwargs):
        from pathlib import Path

        fig.canvas.draw()
        assert kwargs["annotation"] == "title"
        assert "reference" in kwargs["title"]
        assert "relative to the 2000–2001 mean" in kwargs["caption"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"managed")
        saved.append(path)

    monkeypatch.setattr(
        "blueearth_cst.climate_analysis.annual_spatial_render.save_figure", save
    )
    records = render_pages(paths, cfg, [root], reference, {}, "title")
    assert all(
        "calendar month's observed mean" in record["methods_detail"]
        for record in records.values()
    )
    assert len(saved) == 4
    assert all(Path(path).parent == root / "plots" for path in saved)
    assert {Path(path).name for path in saved} == {
        f"era5_{variable}_annual_anomaly_map_basin_page_{page:02d}.png"
        for variable in ("precip", "temp")
        for page in (1, 2)
    }
    assert [r["years"] for r in records.values()] == [list(range(1999, 2005))] * 2 + [
        [2005, 2006]
    ] * 2
    write_json(root / "plots/_engine/figure_captions.json", {"figures": records})
    assert retained_enabled(root)
    unrelated = root / "plots/annual_anomalies/keep.png"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated")
    retained_hash = file_hash(anomaly_paths(root)["fields"])
    keep = {key: value for key, value in records.items() if "page_01" in key}
    retire_pages(root, records, keep, "era5")
    assert len(list(root.rglob("*annual_anomaly_map*.png"))) == 2
    retire_pages(root, keep, {}, "era5")
    assert unrelated.read_bytes() == b"unrelated"
    assert retained_hash == file_hash(anomaly_paths(root)["fields"])
    fields, _ = load_retained(root)
    assert fields.precip_reference_valid.all()
    with open(anomaly_paths(root)["fields"], "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        load_retained(root)


def test_all_zero_and_all_unavailable_scales(tmp_path):
    cfg = settings()
    ds = raw()
    ds.precip[:] = 0
    ds.temp[:] = np.nan
    retain(tmp_path, ds, cfg)
    reference = {
        "frame": {
            "extent": [9.5, 11.5, -0.5, 1.5],
            "longitude_ticks": [],
            "latitude_ticks": [],
            "aspect": 1.0,
        }
    }
    _, scales = prepare_display([tmp_path], reference, {}, cfg)
    assert scales["precip"]["all_zero"]
    assert scales["precip"]["limits"] == [-1, 1]
    assert scales["temp"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "relative",
    [
        "plots/annual_anomalies/page_001/era5_precip_annual_anomaly_map_basin.png",
        "plots/annual_anomalies/page_001/era5_precip_annual_anomaly_map_basin_ext.png",
        "plots/era5_precip_annual_anomaly_map_basin.png",
        "plots/era5_precip_annual_anomaly_map_basin_page_01.png",
    ],
)
def test_cleanup_retires_current_and_legacy_managed_pages(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"managed")
    unrelated = target.parent / "keep.png"
    unrelated.write_bytes(b"unrelated")
    retire_pages(
        tmp_path, {relative: {"encoding": "annual_spatial_anomalies"}}, {}, "era5"
    )
    assert not target.exists()
    assert unrelated.read_bytes() == b"unrelated"


@pytest.mark.parametrize("year_count", [5, 6, 7])
def test_anomaly_figures_are_flat_and_suffix_only_when_split(tmp_path, year_count):
    cfg = settings(comparison_period={"start": 2000, "end": 1999 + year_count})
    paths = diagnostic_source_outputs(tmp_path, "chirps", cfg)
    figures = {
        key: Path(value)
        for key, value in paths["figures"].items()
        if key.startswith("annual-map-")
    }
    expected = (
        {"chirps_precip_annual_anomaly_map_basin.png"}
        if year_count <= 6
        else {
            "chirps_precip_annual_anomaly_map_basin_page_01.png",
            "chirps_precip_annual_anomaly_map_basin_page_02.png",
        }
    )
    assert {path.name for path in figures.values()} == expected
    assert all(path.parent == tmp_path / "plots" for path in figures.values())
    assert set(figures.values()) <= {Path(path) for path in paths["render"].values()}


def test_flat_layout_retires_nested_managed_png_and_preserves_current(tmp_path):
    old = "plots/annual_anomalies/page_001/chirps_precip_annual_anomaly_map_basin.png"
    current = "plots/chirps_precip_annual_anomaly_map_basin.png"
    for name in (old, current):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    record = {"encoding": "annual_spatial_anomalies"}
    retire_pages(tmp_path, {old: record, current: record}, {current: record}, "chirps")
    assert not (tmp_path / old).exists()
    assert (tmp_path / current).read_bytes() == current.encode()


@pytest.mark.parametrize("last_year", [2004, 2006])
def test_retained_figure_inventory_matches_flat_workflow_outputs(
    tmp_path, monkeypatch, last_year
):
    import json
    from types import SimpleNamespace

    from blueearth_cst.climate_analysis import annual_spatial_anomalies as module

    climate = tmp_path / "climate.nc"
    raw().to_netcdf(climate)
    cells = tmp_path / "cells.csv"
    catalog = tmp_path / "catalog.yml"
    cells.write_text("cells", encoding="utf-8")
    catalog.write_text("catalog", encoding="utf-8")
    cfg = settings(comparison_period={"start": 2000, "end": last_year})
    root = tmp_path / "source"
    paths = anomaly_paths(root)
    monkeypatch.setattr(
        module, "daily_series", lambda *a, **kw: (pd.DataFrame({"temp": [25.0]}), {})
    )
    module.run(
        SimpleNamespace(
            input=SimpleNamespace(climate_nc=climate, basin_cells=cells),
            output=SimpleNamespace(**paths),
            params=SimpleNamespace(
                source="chirps",
                data_sources=str(catalog),
                settings=cfg,
                m0=1,
                code_digest={},
                environment={},
            ),
        )
    )
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    declared = diagnostic_source_outputs(root, "chirps", cfg)["figures"]
    assert [page["figures"]["precip"] for page in record["figure_pages"]] == [
        Path(value).relative_to(root).as_posix()
        for key, value in declared.items()
        if key.startswith("annual-map-")
    ]


def test_cleanup_never_traverses_internal_junction(tmp_path, monkeypatch):
    from pathlib import Path

    relative = (
        "plots/annual_anomalies/page_001/era5_precip_annual_anomaly_map_basin_ext.png"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserved")
    monkeypatch.setattr(
        Path, "is_junction", lambda self: self == target.parent, raising=False
    )
    retire_pages(
        tmp_path, {relative: {"encoding": "annual_spatial_anomalies"}}, {}, "era5"
    )
    assert target.read_bytes() == b"preserved"


def test_out_of_span_and_partial_years_have_accurate_availability(
    tmp_path, monkeypatch
):
    cfg = settings(
        reference_period={"start": 1998, "end": 2000},
        comparison_period={"start": 1998, "end": 2000},
    )
    with pytest.warns(UserWarning, match=r"incomplete/out-of-span.*1998, 1999"):
        retain(tmp_path, raw(10), cfg, m0=10)
    monkeypatch.setattr(
        "blueearth_cst.climate_analysis.annual_spatial_render.save_figure",
        lambda *args, **kwargs: None,
    )
    reference = {
        "frame": {
            "extent": [9.5, 11.5, -0.5, 1.5],
            "longitude_ticks": [],
            "latitude_ticks": [],
            "aspect": 1.0,
        }
    }
    records = render_pages(
        diagnostic_source_outputs(tmp_path, "era5", cfg),
        cfg,
        [tmp_path],
        reference,
        {},
        "none",
    )
    for record in records.values():
        assert ", page " not in record["title"]
        assert record["unavailable_reasons"][:2] == ["Incomplete/out-of-span year"] * 2
        assert record["unavailable_reasons"][2] == ""
        assert (
            "excluded incomplete/out-of-span years: [1998, 1999]"
            in record["methods_detail"]
        )
