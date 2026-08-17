# -*- coding: utf-8 -*-
"""The statistics heatmap — `blueearth_cst/shared/statistics_heatmap.py`.

What a rendered PNG cannot tell you, and what the upstream original got wrong,
is where these tests are aimed:

* the frame is drawn in the ORIENTATION it arrives in (the original transposed
  internally, so a caller who had already oriented the table got it rotated);
* an inverted row inverts the COLOUR and keeps the annotation's true value —
  otherwise the figure and the CSV it renders disagree about a number;
* the colour limits are symmetric about zero, so a diverging map's midpoint is
  the meaningful zero rather than the middle of the data's range;
* a zero reference yields no percentage rather than `inf` or a substituted 0.0,
  which would draw "no change" over a cell where nothing was computed.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from blueearth_cst.shared import statistics_heatmap as sh  # noqa: E402

DPI = 60


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def drawn(monkeypatch):
    """Capture the axes `statistics_heatmap` drew on.

    The function closes its own figure, which is right — a producer that leaks
    figures exhausts matplotlib's limit over a long run — and it also makes the
    result uninspectable from outside. Rather than adding a return value or a
    `close=False` flag that exists only for tests, this spies on `subplots` and
    suspends `close` for the duration. The autouse fixture above tears down
    after monkeypatch has restored the real `close`, so nothing leaks.
    """
    captured = {}
    real_subplots = sh.plt.subplots

    def spy(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["fig"], captured["ax"] = fig, ax
        return fig, ax

    monkeypatch.setattr(sh.plt, "subplots", spy)
    monkeypatch.setattr(sh.plt, "close", lambda *a, **k: None)
    return captured


def _values():
    return pd.DataFrame(
        {"ssp245": [10.0, -4.0, 0.5], "ssp585": [25.0, -12.0, 1.5]},
        index=["q_annual_mean", "q_mean_annual_min", "q_baseflow_index"],
    )


class TestFlattenColumns:
    def test_leaves_a_single_level_axis_alone(self):
        frame = _values()
        assert list(sh.flatten_columns(frame).columns) == ["ssp245", "ssp585"]

    def test_joins_a_multiindex_into_one_label_per_column(self):
        frame = _values()
        frame.columns = pd.MultiIndex.from_tuples(
            [("ssp245", "far"), ("ssp585", "far")]
        )
        out = sh.flatten_columns(frame, separator="|")
        assert list(out.columns) == ["ssp245|far", "ssp585|far"]

    def test_does_not_mutate_the_caller_s_frame(self):
        frame = _values()
        frame.columns = pd.MultiIndex.from_tuples(
            [("ssp245", "far"), ("ssp585", "far")]
        )
        sh.flatten_columns(frame)
        assert isinstance(frame.columns, pd.MultiIndex)


class TestRelativeChange:
    def test_percent_against_the_reference_column(self):
        frame = pd.DataFrame(
            {"Historical": [100.0, 4.0], "ssp585": [150.0, 2.0]}, index=["a", "b"]
        )
        out = sh.relative_change(frame, "Historical")
        assert out.loc["a", "ssp585"] == pytest.approx(50.0)
        assert out.loc["b", "ssp585"] == pytest.approx(-50.0)

    def test_the_reference_column_is_kept_at_zero(self):
        """Dropping it would leave the figure without the baseline the
        percentages are measured from."""
        frame = pd.DataFrame({"Historical": [100.0], "ssp585": [150.0]}, index=["a"])
        out = sh.relative_change(frame, "Historical")
        assert out.loc["a", "Historical"] == pytest.approx(0.0)

    def test_a_zero_reference_yields_nan_not_inf_or_zero(self):
        """The original replaced both with 0.0, which draws 'no change' over a
        cell where nothing was computed."""
        frame = pd.DataFrame({"Historical": [0.0], "ssp585": [5.0]}, index=["a"])
        out = sh.relative_change(frame, "Historical")
        assert pd.isna(out.loc["a", "ssp585"])
        assert not np.isinf(out.to_numpy(dtype=float)).any()

    def test_raises_on_a_missing_reference_column(self):
        with pytest.raises(KeyError, match="reference column"):
            sh.relative_change(_values(), "Historical")


class TestAbsoluteWithRelativeAnnotations:
    def test_pairs_the_absolute_value_with_its_percentage(self):
        absolute = pd.DataFrame({"ssp585": [4.13]}, index=["q"])
        relative = pd.DataFrame({"ssp585": [-6.44]}, index=["q"])
        out = sh.absolute_with_relative_annotations(absolute, relative)
        assert out.loc["q", "ssp585"] == "4.1 (-6.4%)"

    def test_small_values_keep_more_decimals(self):
        """One decimal is right on a discharge of 4.1 and wrong on a baseflow
        index of 0.003; one setting cannot serve both in one table."""
        absolute = pd.DataFrame({"ssp585": [0.0034]}, index=["bfi"])
        relative = pd.DataFrame({"ssp585": [12.0]}, index=["bfi"])
        out = sh.absolute_with_relative_annotations(absolute, relative)
        assert out.loc["bfi", "ssp585"].startswith("0.003")

    def test_a_nan_change_shows_the_absolute_value_alone(self):
        absolute = pd.DataFrame({"ssp585": [5.0]}, index=["q"])
        relative = pd.DataFrame({"ssp585": [np.nan]}, index=["q"])
        out = sh.absolute_with_relative_annotations(absolute, relative)
        assert out.loc["q", "ssp585"] == "5.0"


class TestStatisticsHeatmap:
    def test_writes_the_file_and_reports_the_cell_count(self, tmp_path):
        out = tmp_path / "heatmap.png"
        assert sh.statistics_heatmap(_values(), out, dpi=DPI) == 6
        assert out.exists() and out.stat().st_size > 0

    def test_rows_stay_on_the_y_axis(self, tmp_path, drawn):
        """The original transposed internally. A caller who has already
        oriented the table must not have it rotated underneath them."""
        sh.statistics_heatmap(_values(), tmp_path / "h.png", dpi=DPI)
        ax = drawn["ax"]
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            "q_annual_mean",
            "q_mean_annual_min",
            "q_baseflow_index",
        ]

    def test_default_colour_limits_are_symmetric_about_zero(self, tmp_path, drawn):
        """A diverging map centred on the data's midpoint rather than on the
        meaningful zero mis-reads by construction."""
        sh.statistics_heatmap(_values(), tmp_path / "h.png", dpi=DPI)
        mesh = drawn["ax"].collections[0]
        low, high = mesh.get_clim()
        assert low == pytest.approx(-high)
        assert high == pytest.approx(25.0)

    def test_explicit_limits_win(self, tmp_path, drawn):
        sh.statistics_heatmap(
            _values(), tmp_path / "h.png", vmin=-100, vmax=100, dpi=DPI
        )
        assert drawn["ax"].collections[0].get_clim() == (-100.0, 100.0)

    def test_an_inverted_row_is_marked_and_keeps_its_true_annotation(
        self, tmp_path, drawn
    ):
        """The `*` says the colour was flipped; the number must still be the
        one in the CSV, or the figure and the table disagree."""
        sh.statistics_heatmap(
            _values(),
            tmp_path / "h.png",
            invert_rows=["q_mean_annual_min"],
            dpi=DPI,
        )
        ax = drawn["ax"]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "q_mean_annual_min *" in labels
        assert "-4.0" in [t.get_text() for t in ax.texts]
        assert "4.0" not in [t.get_text() for t in ax.texts]

    def test_an_unknown_invert_row_is_ignored_rather_than_raising(self, tmp_path):
        """Indicator sets are config-driven; a caller's standing list of
        'up is bad' names must not take the figure down on a table that
        happens not to contain one of them."""
        assert (
            sh.statistics_heatmap(
                _values(), tmp_path / "h.png", invert_rows=["not_a_row"], dpi=DPI
            )
            == 6
        )

    def test_custom_annotations_are_used_verbatim(self, tmp_path, drawn):
        values = _values()
        annotations = values.map(lambda v: f"<{v:.0f}>")
        sh.statistics_heatmap(
            values, tmp_path / "h.png", annotations=annotations, dpi=DPI
        )
        assert "<10>" in [t.get_text() for t in drawn["ax"].texts]

    def test_mismatched_annotations_raise(self, tmp_path):
        annotations = pd.DataFrame({"ssp245": ["a"]}, index=["q_annual_mean"])
        with pytest.raises(ValueError, match="do not match values"):
            sh.statistics_heatmap(
                _values(), tmp_path / "h.png", annotations=annotations, dpi=DPI
            )

    def test_an_empty_table_raises_rather_than_drawing_a_blank(self, tmp_path):
        with pytest.raises(ValueError, match="no rows"):
            sh.statistics_heatmap(pd.DataFrame(), tmp_path / "h.png", dpi=DPI)

    def test_bold_keyword_emboldens_only_the_matching_column(self, tmp_path, drawn):
        sh.statistics_heatmap(
            _values(), tmp_path / "h.png", bold_columns_containing="585", dpi=DPI
        )
        texts = drawn["ax"].texts
        bold = {t.get_text() for t in texts if t.get_fontweight() == "bold"}
        assert bold == {"25.0", "-12.0", "1.5"}

    def test_multiindex_columns_are_flattened_for_drawing(self, tmp_path, drawn):
        values = _values()
        values.columns = pd.MultiIndex.from_tuples(
            [("ssp245", "far"), ("ssp585", "far")]
        )
        sh.statistics_heatmap(values, tmp_path / "h.png", column_separator="|", dpi=DPI)
        labels = [t.get_text() for t in drawn["ax"].get_xticklabels()]
        assert labels == ["ssp245|far", "ssp585|far"]

    def test_does_not_leave_seaborn_styling_on_the_global_rc(self, tmp_path):
        """`sns.set_theme` rewrites the global rc, so any figure drawn later in
        the same process inherits seaborn's styling instead of this toolbox's
        page contract. The original called it; this must not."""
        before = dict(plt.rcParams)
        sh.statistics_heatmap(_values(), tmp_path / "h.png", dpi=DPI)
        changed = {k for k in before if plt.rcParams[k] != before[k]}
        assert not changed, f"global rc was mutated: {sorted(changed)}"
