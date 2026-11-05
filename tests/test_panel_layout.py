"""Physical panel geometry and shared planner fixtures."""

import csv
from pathlib import Path

import matplotlib as mpl
import pytest

from blueearth_cst.plotting.panel_layout import arrange_panels, choose_grid


def test_shared_grid_fixture():
    path = Path(__file__).parent / "fixtures" / "panel_layout.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for case in csv.DictReader(stream):
            args = (
                int(case["panel_count"]),
                [(3, float(case["width_3_mm"])), (2, float(case["width_2_mm"]))],
                float(case["max_width_mm"]),
            )
            if int(case["expected_columns"]) == 0:
                with pytest.raises(ValueError, match="do not fit"):
                    choose_grid(*args)
            else:
                assert choose_grid(*args) == (
                    int(case["expected_columns"]),
                    int(case["expected_rows"]),
                )
    assert choose_grid(1, [(1, 90)], 180) == (1, 1)
    assert choose_grid(2, [(3, 250), (2, 170)], 180) == (2, 1)


@pytest.mark.parametrize(
    "count,max_width,columns,rows",
    [
        (1, 180, 1, 1),
        (2, 180, 2, 1),
        (3, 180, 2, 2),
        (4, 180, 2, 2),
        (9, 270, 3, 3),
    ],
)
def test_measured_physical_panels(count, max_width, columns, rows):
    def draw(ax):
        ax.plot([0, 1], [0, 1], label="Observed")
        ax.set(xlabel="Observed daily mean", ylabel="Simulated daily mean")
        ax.set_title("a) Precipitation (mm day⁻¹)", loc="left")
        ax.legend(fontsize=6)

    with mpl.rc_context({"font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7}):
        fig, axes, result = arrange_panels([draw] * count, max_width_mm=max_width)
        try:
            assert (result.columns, result.rows) == (columns, rows)
            assert result.canvas_width_mm <= max_width + 1e-6
            for i, ax in enumerate(axes):
                bbox = ax.get_window_extent()
                assert bbox.width * 25.4 / fig.dpi == pytest.approx(70, abs=0.001)
                assert bbox.height * 25.4 / fig.dpi == pytest.approx(70, abs=0.001)
                assert ax.xaxis.label.get_fontsize() == 7
                assert list(ax.lines[0].get_ydata()) == [0, 1]
                assert result.cells[i] == divmod(i, columns)
            if count == 1:
                assert axes[0].get_title(loc="left") == ""
        finally:
            fig.clear()


def test_no_fit_and_invalid_inputs():
    with pytest.raises(ValueError, match="do not fit"):
        arrange_panels([lambda ax: None] * 2, max_width_mm=100)
    with pytest.raises(ValueError, match="at least one"):
        arrange_panels([])
    with pytest.raises(ValueError, match="distinct integers"):
        arrange_panels([lambda ax: None], preferred_columns=[1])
