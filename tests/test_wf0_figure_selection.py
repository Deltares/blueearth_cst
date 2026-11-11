"""Historical figure selection keeps unique outputs and numerical products."""

import json
from pathlib import Path

import pytest

from blueearth_cst.climate_analysis.diagnostic_outputs import (
    diagnostic_comparison_outputs,
    diagnostic_source_outputs,
    parse_figure_settings,
)
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings


@pytest.mark.parametrize("source", ["era5", "chirps"])
def test_single_source_keeps_all_basin_figures(source):
    settings = parse_settings(None, {"start": 2000, "end": 2016}, [source])
    ordinary = diagnostic_source_outputs("store", source, settings)
    selected = diagnostic_source_outputs("store", source, settings, [source])
    assert selected == ordinary


def test_multi_source_omits_duplicates_and_preserves_unique_temperature_and_maps():
    sources = ["era5", "chirps"]
    settings = parse_settings(None, {"start": 2000, "end": 2016}, sources)
    comparison = diagnostic_comparison_outputs("comparison", sources, settings)
    era5 = diagnostic_source_outputs("era5", "era5", settings, sources)
    chirps = diagnostic_source_outputs("chirps", "chirps", settings, sources)
    assert set(era5["figures"]) == {
        "temperature",
        "map-precip",
        "map-temp",
        "map-pet",
    }
    assert set(chirps["figures"]) == {"map-precip"}
    assert "seasonal" in comparison["figures"]
    assert "temperature" not in comparison["figures"]
    assert "pt" in comparison["figures"]
    for source, paths in (("era5", era5), ("chirps", chirps)):
        assert (
            paths["compute"]
            == diagnostic_source_outputs(source, source, settings)["compute"]
        )
        assert {
            path for key, path in paths["render"].items() if key.startswith("figure_")
        } == set(paths["figures"].values())


def test_unique_source_temperature_relationship_survives_other_comparison_binding():
    sources = ["era5", "other_temperature_source"]
    settings = parse_settings(None, {"start": 2000, "end": 2016}, sources)
    paths = diagnostic_source_outputs("other", sources[1], settings, sources)
    assert "pt" in paths["figures"]
    assert "temperature" not in paths["figures"]


def test_source_opt_in_restores_full_inventory_without_changing_compute_paths():
    sources = ["era5", "chirps"]
    settings = parse_settings(None, {"start": 2000, "end": 2016}, sources)
    full = diagnostic_source_outputs("store", "era5", settings)
    opted_in = diagnostic_source_outputs(
        "store", "era5", settings | {"source_figures": True}, sources
    )
    assert opted_in == full


@pytest.mark.parametrize(
    "key", ["source_figures", "location_figures", "subbasin_figures"]
)
@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_figure_switches_require_booleans(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a boolean"):
        parse_figure_settings({key: value})


def test_retained_rerender_preserves_selected_forms(tmp_path, monkeypatch):
    from blueearth_cst.climate_analysis import diagnostic_rerender as retained

    store = tmp_path / "store"
    (store / "_engine").mkdir(parents=True)
    settings = parse_settings(None, {"start": 2000, "end": 2016}, ["era5", "chirps"])
    (store / "_engine/diagnostics.json").write_text(
        json.dumps({"settings": settings, "water_year_start": 1}), encoding="utf-8"
    )
    (store / "plots/_engine").mkdir(parents=True)
    (store / "plots/_engine/figure_captions.json").write_text(
        json.dumps({"figures": {}, "selected_forms": ["temperature", "map-precip"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(retained, "load_retained_reference", lambda _: {})
    monkeypatch.setattr(retained, "source_stores", lambda *args: {"era5": store})
    calls = []
    monkeypatch.setattr(retained, "render", lambda *args, **kwargs: calls.append(args))
    retained.rerender_sources(tmp_path, tmp_path / "preview")
    assert set(calls[0][0]["figures"]) == {"temperature", "map-precip"}


def test_selected_source_inventory_renders_unique_products_and_preserves_old_png(
    tmp_path,
):
    import pandas as pd
    from PIL import Image

    from blueearth_cst.climate_analysis.diagnostic_render import render
    from blueearth_cst.climate_analysis.diagnostic_tables import compute, write_tables
    from blueearth_cst.shared import map_reference as mr
    from tests.test_map_reference import _fields

    sources = ["era5", "chirps"]
    settings = parse_settings(None, {"start": 2000, "end": 2005}, sources)
    full = diagnostic_source_outputs(tmp_path, "era5", settings)
    paths = diagnostic_source_outputs(tmp_path, "era5", settings, sources)
    daily = pd.DataFrame(
        {"precip": 3.0, "temp": 25.0},
        index=pd.date_range("2000-01-01", "2005-12-31"),
    )
    _, tables = compute(daily, settings, 1, (2000, 2005))
    write_tables(paths, {"era5": tables}, settings, 1, (2000, 2005), {})

    fields = _fields()
    for variable, value, units in (
        ("temp", 25.0, "degC"),
        ("pet", 800.0, "mm/year"),
    ):
        fields[variable] = fields.precip.copy(deep=True)
        fields[variable].values[0, 0] = value
        fields[variable].attrs["units"] = units
        fields[f"{variable}_valid_years"] = fields.precip_valid_years.copy()
        fields[f"{variable}_unavailable"] = fields.precip_unavailable.copy()
    identity = {"source": "era5", "role": "temperature-carrier"}
    entry = mr.seal_fields(fields, identity)
    fields.to_netcdf(paths["compute"]["maps"])
    reference = {
        "reference_id": "synthetic-reference",
        "identity": {
            "source_preparations": {
                "era5": {**entry, "preparation_identity": identity}
            },
            "completeness": {
                "max_missing_days_month": 3,
                "max_missing_days_year": 15,
            },
        },
        "frame": mr.frame_record(fields.precip, "era5"),
        "variables": {
            variable: mr.scale_record(variable, value, value)
            for variable, value in (("precip", 365.0), ("temp", 25.0), ("pet", 800.0))
        },
    }
    old = Path(full["figures"]["seasonal"])
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"previously-generated-disabled-figure")
    records = render(
        paths, settings, 1, ["era5"], [tmp_path], reference_record=reference
    )

    assert set(paths["figures"]) == {"temperature", "map-precip", "map-temp", "map-pet"}
    assert len(records) == 4
    for key, filename in paths["figures"].items():
        with Image.open(filename) as image:
            image.verify()
        assert Path(filename).stat().st_size > 1000
        record = records[Path(filename).relative_to(tmp_path).as_posix()]
        assert record["title"] and record["caption"]
        if key.startswith("map-"):
            assert record["map_reference"]["reference_id"] == "synthetic-reference"
            assert record["map_reference"]["finite_cells"] == 1
    assert old.read_bytes() == b"previously-generated-disabled-figure"
    assert not Path(full["figures"]["series"]).exists()
    manifest = json.loads(Path(paths["render"]["captions_json"]).read_text())
    assert set(manifest["figures"]) == set(records)
    assert set(manifest["selected_forms"]) == set(paths["figures"])


@pytest.mark.workflow_contract
@pytest.mark.parametrize("enabled", [False, True])
def test_workflow_figure_switches_preserve_computation(tmp_path, enabled):
    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.conftest import parse_workflow, write_config

    cfg = load_composed_config(Path(__file__).with_name("project_config_fixture.yml"))
    cfg["climate"]["sources"] = ["era5", "chirps"]
    cfg["workflows"]["analyze_climate"].update(
        source_figures=enabled, location_figures=enabled
    )
    workflow = parse_workflow("analyze_climate.smk", write_config(tmp_path, cfg))
    source = workflow.get_rule("plot_climate_diagnostics_era5")
    assert ("figure_seasonal" in source.output.keys()) is enabled
    assert "figure_temperature" in source.output.keys()
    assert "figure_map-precip" in source.output.keys()
    assert (
        "source_figures"
        not in workflow.get_rule("compute_climate_diagnostics_era5").params.settings
    )
    requested = set(workflow.get_rule("all").input)
    locations = workflow.get_rule("prepare_location_comparisons")
    assert set(locations.output) <= requested
    figures = workflow.get_rule("plot_location_comparisons")
    assert bool(set(figures.output) & requested) is enabled
    assert set(workflow.get_rule("plot_climate_comparison").output) <= requested
