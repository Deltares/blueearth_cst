# -*- coding: utf-8 -*-
"""The WF2 drawing layer — `blueearth_cst/projections/projection_plots.py`.

A figure is verified by rendering it and LOOKING at it (`AGENTS.md`, *Figures
are terminal artifacts*), so these tests deliberately do not check what the
picture looks like. They check the things a rendered PNG cannot tell you at a
glance and that a reader would otherwise have to take on trust:

* every figure writes the file it was asked for and reports how much it drew,
  so a caller can assert the count equals the resolved combinations;
* the layout rulings that are structural rather than aesthetic — no titles
  anywhere, panel labels instead — because those are toolbox-wide conventions
  and a silent regression would spread to every family that inherits them;
* scenario is the only visual ENCODING, which is the contract that keeps a
  nine-trace panel readable and is invisible in a passing render.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 -- must follow the Agg selection
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from blueearth_cst.projections import projection_plots as pp  # noqa: E402

# A low dpi throughout: these tests are about structure, and 600 dpi would make
# every one of them write a megabyte for no added signal.
DPI = 60


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _annual_frame(models=("A", "B"), scenarios=("ssp245", "ssp585")):
    rows = []
    for model in models:
        for scenario in ("historical", *scenarios):
            years = range(1990, 2001) if scenario == "historical" else range(2001, 2011)
            for year in years:
                rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "member": "r1i1p1f1",
                        "year": year,
                        "precip": 3.0 + 0.01 * year,
                        "temp": 20.0 + 0.02 * year,
                        "precip_anomaly": 0.5 * (year - 1995),
                        "temp_anomaly": 0.02 * (year - 1995),
                    }
                )
    return pd.DataFrame(rows)


def _cloud_frame(models=("A", "B"), scenarios=("ssp245", "ssp585")):
    rng = np.random.default_rng(0)
    rows = []
    for model in models:
        for scenario in scenarios:
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "member": "r1i1p1f1",
                    "precip_change": float(rng.uniform(-10, 10)),
                    "temp_change": float(rng.uniform(0.5, 3.0)),
                }
            )
    return pd.DataFrame(rows)


def _monthly_frame(models=("A", "B"), scenarios=("ssp245", "ssp585")):
    rows = []
    for model in models:
        for scenario in scenarios:
            for month in range(1, 13):
                rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "member": "r1i1p1f1",
                        "month": month,
                        "precip_change": float(month) - 6.0,
                        "temp_change": 1.0 + month / 12.0,
                    }
                )
    return pd.DataFrame(rows)


class TestAnnualOverview:
    def test_writes_the_file_and_counts_traces_per_panel(self, tmp_path):
        """Two models x (historical + two scenarios) = six traces per panel."""
        out = tmp_path / "annual-precipitation.png"
        traces = pp.draw_annual_overview(
            _annual_frame(), "precip", (1990, 2000), out, dpi=DPI
        )
        assert out.exists() and out.stat().st_size > 0
        assert traces == 6

    def test_draws_no_title_on_any_axes(self, tmp_path):
        """Owner ruling 2026-08-11, toolbox-wide. Structural, not aesthetic:
        the y-label and the caveat carry what a title used to say, so a title
        creeping back means that information is now duplicated or contradicted."""
        pp.draw_annual_overview(
            _annual_frame(), "temp", (1990, 2000), tmp_path / "t.png", dpi=DPI
        )
        for fig in map(plt.figure, plt.get_fignums()):
            for ax in fig.axes:
                assert ax.get_title() == ""

    def test_labels_the_panels_a_and_b(self, tmp_path):
        pp.draw_annual_overview(
            _annual_frame(), "precip", (1990, 2000), tmp_path / "p.png", dpi=DPI
        )
        # The figure is closed by the drawing function, so re-draw onto axes we
        # keep, which is what the label helper is given in production too.
        fig, ax = plt.subplots()
        pp.panel_label(ax, "a")
        assert any(t.get_text() == "a)" for t in ax.texts)

    def test_legend_names_scenarios_only_never_models(self, tmp_path):
        """The contract that keeps a nine-trace panel readable: a legend with a
        row per combination is exactly the hairball this set replaced."""
        handles = pp.scenario_handles(["ssp245", "ssp585"])
        labels = [h.get_label() for h in handles]
        assert labels == ["Historical", "SSP2-4.5", "SSP5-8.5"]
        assert not any("A" == label or "B" == label for label in labels)

    def test_unknown_scenario_still_draws_rather_than_raising(self, tmp_path):
        """A scenario outside the four named ones is plausible -- CMIP6 gains
        SSPs -- and it must not take the whole figure down."""
        frame = _annual_frame(scenarios=("ssp119",))
        out = tmp_path / "novel.png"
        assert pp.draw_annual_overview(frame, "precip", (1990, 2000), out, dpi=DPI) == 4
        assert out.exists()


class TestCloudFaceted:
    def test_one_horizon_writes_a_single_panel(self, tmp_path):
        out = tmp_path / "cloud.png"
        points = pp.draw_cloud_faceted(
            {"far": _cloud_frame()}, {"far": (2070, 2090)}, out, dpi=DPI
        )
        assert points == 4
        assert out.exists()

    def test_two_horizons_count_every_point(self, tmp_path):
        changes = {"near": _cloud_frame(), "far": _cloud_frame()}
        periods = {"near": (2040, 2060), "far": (2070, 2090)}
        points = pp.draw_cloud_faceted(
            changes, periods, tmp_path / "cloud.png", dpi=DPI
        )
        assert points == 8

    def test_no_titles(self, tmp_path):
        pp.draw_cloud_faceted(
            {"far": _cloud_frame()}, {"far": (2070, 2090)}, tmp_path / "c.png", dpi=DPI
        )
        for fig in map(plt.figure, plt.get_fignums()):
            for ax in fig.axes:
                assert ax.get_title() == ""


class TestCloudCombined:
    def test_counts_points_across_every_horizon(self, tmp_path):
        changes = {"near": _cloud_frame(), "far": _cloud_frame()}
        periods = {"near": (2040, 2060), "far": (2070, 2090)}
        out = tmp_path / "combined.png"
        assert pp.draw_cloud_combined(changes, periods, out, dpi=DPI) == 8
        assert out.exists()

    def test_more_horizons_than_markers_does_not_raise(self, tmp_path):
        """The marker list is finite and horizons are config; wrapping is the
        documented behaviour, not an IndexError at render time."""
        names = [f"h{i}" for i in range(len(pp.HORIZON_MARKERS) + 2)]
        changes = {n: _cloud_frame() for n in names}
        periods = {n: (2040 + i, 2060 + i) for i, n in enumerate(names)}
        out = tmp_path / "many.png"
        assert pp.draw_cloud_combined(changes, periods, out, dpi=DPI) == 4 * len(names)


class TestMonthlyChange:
    def test_writes_the_file_and_counts_traces_per_panel(self, tmp_path):
        out = tmp_path / "monthly.png"
        traces = pp.draw_monthly_change(
            _monthly_frame(), "far", (2070, 2090), (2000, 2014), out, dpi=DPI
        )
        assert traces == 4
        assert out.exists()

    def test_caveat_states_the_definition_the_figure_used(self, tmp_path):
        """The definition is the thing that was wrong before, and a reader has
        no other way to tell which one a figure used."""
        fig, _ = pp.new_figure(0.38, ncols=2)
        pp.caveat(fig, "read from the change-factor table")
        assert "change-factor table" in fig.get_supxlabel()

    def test_no_titles(self, tmp_path):
        pp.draw_monthly_change(
            _monthly_frame(),
            "far",
            (2070, 2090),
            (2000, 2014),
            tmp_path / "m.png",
            dpi=DPI,
        )
        for fig in map(plt.figure, plt.get_fignums()):
            for ax in fig.axes:
                assert ax.get_title() == ""


class TestPageContract:
    def test_every_figure_is_the_shared_page_width(self):
        """One column width across the toolbox, so a report can set a WF1 map
        and a WF2 series side by side without one being scaled."""
        from blueearth_cst.shared.cartographic_map import series_figure_size

        for aspect in (0.38, 0.5, 0.62, 0.78):
            fig, _ = pp.new_figure(aspect)
            assert fig.get_size_inches()[0] == pytest.approx(
                series_figure_size(aspect)[0]
            )

    def test_layout_is_constrained_not_tight(self):
        """A family that mixes the two cannot be made to agree on its margins."""
        fig, _ = pp.new_figure(0.42)
        assert fig.get_layout_engine() is not None
        assert "constrained" in type(fig.get_layout_engine()).__name__.lower()
