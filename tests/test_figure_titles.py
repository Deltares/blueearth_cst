"""Title metadata is independent of figure bytes and caption rendering."""

import json

import pytest

from blueearth_cst.climate_analysis.diagnostic_captions import TITLES as WF0_TITLES
from blueearth_cst.climate_analysis.diagnostic_captions import title as wf0_title
from blueearth_cst.climate_analysis.diagnostic_outputs import FORMS
from blueearth_cst.plotting.figure_titles import (
    figure_engine_dir,
    figure_inventory_paths,
    write_figure_titles,
)
from blueearth_cst.projections.projection_captions import TITLES as WF2_TITLES
from blueearth_cst.projections.projection_figures import FIGURE_FILENAMES
from blueearth_cst.shared.func_plot_signature import _basavg_caption
from blueearth_cst.shared.plot_spatial_maps import FIGURE_TITLES, SPATIAL_MAP_FIGURES


def test_all_declared_diagnostics_and_projection_products_have_titles():
    assert set(FORMS) <= set(WF0_TITLES)
    assert set(FIGURE_FILENAMES) == set(WF2_TITLES)
    assert {figure.stem for figure in SPATIAL_MAP_FIGURES} == set(FIGURE_TITLES)


@pytest.mark.parametrize("name", WF0_TITLES)
def test_wf0_titles_use_plot_dataset_and_parenthesized_window(name):
    assert wf0_title(
        name, ["ERA5"], (1971, 2020), "basin average, equal-weight cells"
    ) == (f"{WF0_TITLES[name]} — ERA5 (1971–2020)")


def test_wf0_comparison_titles_keep_dataset_names_and_actual_window():
    assert wf0_title("seasonal", ["ERA5", "CHIRPS"], (1981, 2020), "subbasin_101") == (
        "Monthly precipitation climatology — ERA5, CHIRPS (1981–2020)"
    )


def test_title_inventory_shares_keys_and_preserves_images_and_other_families(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    image = plots / "station.png"
    image.write_bytes(b"retained image")
    write_figure_titles(plots, {"station.png": {"title": "Discharge at station A"}})
    write_figure_titles(plots, {"map.png": {"title": "Basin elevation"}}, merge=True)
    path = tmp_path / "_engine/plots/figure_titles.json"
    assert set(json.loads(path.read_text(encoding="utf-8"))["figures"]) == {
        "station.png",
        "map.png",
    }
    write_figure_titles(plots, {"station.png": {"title": "Updated discharge title"}})
    assert json.loads(path.read_text(encoding="utf-8"))["figures"] == {
        "station.png": {"title": "Updated discharge title"}
    }
    assert "Updated discharge title" in (plots / "figure_titles.md").read_text(
        encoding="utf-8"
    )
    assert image.read_bytes() == b"retained image"


def test_basin_summary_titles_keep_variable_and_subbasin_identity(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    _basavg_caption(
        plots,
        "wflow_aet_annual_ts_subbasin_101.png",
        {"legend": "Actual evapotranspiration (mm)", "kind": "amount"},
        "annual_ts",
        "Subbasin 101",
        {
            "period": ["2000-01-01", "2002-12-31"],
            "partial_years": [],
            "incomplete_years": [],
        },
        {101: 5},
        False,
        {"n": 3, "slope": float("nan")},
    )
    titles = json.loads(
        (tmp_path / "_engine/plots/figure_titles.json").read_text(encoding="utf-8")
    )["figures"]
    assert titles["wflow_aet_annual_ts_subbasin_101.png"]["title"] == (
        "Simulated actual evapotranspiration: annual series — Subbasin 101; 2000–2002"
    )


def test_figure_inventories_live_under_the_units_engine(tmp_path):
    unit = tmp_path / "evaluation"
    assert figure_engine_dir(unit / "plots") == unit / "_engine/plots"
    assert figure_engine_dir(unit / "plots/stations") == unit / "_engine/plots/stations"
    assert figure_inventory_paths(unit / "plots/stations") == [
        "figure_titles.md",
        "../../_engine/plots/stations/figure_titles.json",
        "figure_captions.md",
        "../../_engine/plots/stations/figure_captions.json",
    ]
    with pytest.raises(ValueError, match="plots/"):
        figure_engine_dir(unit / "figures")
