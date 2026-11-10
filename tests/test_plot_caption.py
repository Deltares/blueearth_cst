"""Caption measurement preserves exported pixels without painting twice."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from PIL import Image

from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_preset import CST_PRESET


@pytest.mark.parametrize("annotation", ["caption", "title"])
@pytest.mark.parametrize("kind", ["line", "cropped_map"])
def test_caption_measurement_preserves_pixels_with_fewer_painted_paths(
    tmp_path, monkeypatch, kind, annotation
):
    painted = []
    draw_path = RendererAgg.draw_path

    def record_paint(renderer, *args, **kwargs):
        painted.append(1)
        return draw_path(renderer, *args, **kwargs)

    monkeypatch.setattr(RendererAgg, "draw_path", record_paint)
    counts, outputs = [], []
    for legacy in (True, False):
        painted.clear()
        with monkeypatch.context() as patch, use_preset(CST_PRESET):
            if legacy:
                patch.setattr(
                    Figure, "draw_without_rendering", lambda fig: fig.canvas.draw()
                )
            fig, ax = plt.subplots(figsize=(2.4, 1.8), layout="constrained")
            if kind == "line":
                ax.plot([0, 0.5, 1], [0, 0.75, 0.9], label="Basin")
                ax.legend(loc="lower right")
                ax.set(xlabel="PET/P", ylabel="Q/P")
                fig.legend(*ax.get_legend_handles_labels(), loc="upper center")
            else:
                field = ax.imshow(
                    [[0, 1], [2, 3]], extent=(9, 11, -1, 1), aspect="equal"
                )
                fig.colorbar(field, ax=ax)
                ax.set(xlabel="Longitude", ylabel="Latitude")
            output = tmp_path / f"{legacy}.png"
            try:
                export_figure(
                    fig,
                    output,
                    annotation=annotation,
                    title="Basin comparison",
                    caption="Retained basin values with a wrapped scientific caption.",
                    dpi=600,
                    facecolor="white",
                    metadata={"Description": "caption parity"},
                    **(
                        {"bbox_inches": "tight", "pad_inches": 2 / 25.4}
                        if kind == "cropped_map"
                        else {}
                    ),
                )
            finally:
                plt.close(fig)
            counts.append(len(painted))
            outputs.append(output)
    with Image.open(outputs[0]) as old, Image.open(outputs[1]) as new:
        assert old.mode == new.mode
        assert old.info == new.info
        np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    assert counts[1] < counts[0]


@pytest.mark.parametrize("extension", ["svg", "pdf"])
def test_caption_measurement_preserves_vector_bytes(tmp_path, monkeypatch, extension):
    metadata = (
        {"Date": None}
        if extension == "svg"
        else {"CreationDate": None, "ModDate": None}
    )
    outputs = []
    for legacy in (True, False):
        with (
            monkeypatch.context() as patch,
            use_preset(CST_PRESET),
            plt.rc_context({"svg.hashsalt": "caption-measurement-parity"}),
        ):
            if legacy:
                patch.setattr(
                    Figure, "draw_without_rendering", lambda fig: fig.canvas.draw()
                )
            fig, ax = plt.subplots(figsize=(2.4, 1.8), layout="constrained")
            ax.plot([0, 0.5, 1], [0, 0.75, 0.9], label="Basin")
            ax.legend(loc="lower right")
            ax.set(xlabel="PET/P", ylabel="Q/P")
            output = tmp_path / f"{legacy}.{extension}"
            try:
                export_figure(
                    fig,
                    output,
                    annotation="caption",
                    caption="Basin comparison with a retained legend and caption.",
                    dpi=600,
                    metadata=metadata,
                )
            finally:
                plt.close(fig)
            outputs.append(output)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
