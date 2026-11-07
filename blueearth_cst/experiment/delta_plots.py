"""Member-resolved monthly and spatial diagnostics of retained delta factors."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from blueearth_cst.experiment.content_identity import read_canonical_json
from blueearth_cst.experiment.delta_scenarios import read_delta_collection
from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference


def plot_delta_collection(manifest: Path, output: Path) -> list[Path]:
    """Render every member without pooling across models or implying probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.export import export_figure
    from blueearth_cst.plotting.fundamentals import MONTH_ABBR, new_fig
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared.plot_colors import (
        CLIMATE_COLORMAPS,
        PRECIP_ANOMALY_COLORMAP,
    )
    from blueearth_cst.shared.plot_preset import CST_PRESET

    manifest = manifest.resolve(strict=True)
    marker = read_delta_collection(manifest)
    project = manifest.parents[4]
    intent = read_canonical_json(
        resolve_file_reference(marker["intent"], {"project_root": project})
    )
    data = project / marker["data_root"]
    output = output.resolve()
    if output.is_relative_to(project / "scenarios/_engine"):
        raise ValueError("plots must be outside immutable collection storage")
    settings = intent["settings"]
    with xr.open_dataset(data / settings["historical_path"]) as historical:
        active = historical.precip.notnull().any("time")
        if settings.get("factor_layout") == "grid":
            active = active.transpose("latitude", "longitude")
        active = active.values
        coordinates = {
            key: historical[key].values
            for key in ("latitude", "longitude")
            if key in historical.coords
        }
    groups = defaultdict(list)
    for index, member in enumerate(settings["members"]):
        groups[(member["model"], member["scenario"])].append(index)
    paths, captions = [], []
    variables = [
        ("temp_offset_degC", "Temp.", "°C", 0),
        ("precip_ratio", "Precip.", "%", 1),
    ]
    if intent["factors"] and "pet_ratio" in intent["factors"][0]:
        variables.append(("pet_ratio", "PET", "%", 1))
    output.mkdir(parents=True, exist_ok=True)
    with use_preset(CST_PRESET):
        for group, indices in enumerate(groups.values(), 1):
            fig, axes = new_fig(85, ncols=len(variables), width_mm=190)
            labels = []
            for column, (key, label, unit, neutral) in enumerate(variables):
                ax = axes[0, column]
                for number, index in enumerate(indices):
                    member = settings["members"][index]
                    values = np.asarray(intent["factors"][index][key])
                    if neutral:
                        values = 100 * (values - 1)
                    if values.ndim == 3:
                        values = np.where(active[None, :, :], values, np.nan)
                        low, high = (
                            np.nanmin(values, axis=(1, 2)),
                            np.nanmax(values, axis=(1, 2)),
                        )
                        values = np.nanmean(values, axis=(1, 2))
                    (line,) = ax.plot(
                        range(1, 13),
                        values,
                        marker=("o", "s", "^")[number % 3],
                        linestyle=("-", "--", ":")[number % 3],
                        label=member["member"],
                    )
                    if settings.get("factor_layout") == "grid":
                        ax.fill_between(
                            range(1, 13), low, high, color=line.get_color(), alpha=0.12
                        )
                    if column == 0:
                        labels.append(
                            f"{member['model']} / {member['scenario']} / {member['member']}"
                        )
                ax.axhline(0, color="0.5", linewidth=0.5)
                ax.set_xticks(range(1, 13), MONTH_ABBR, rotation=90)
                ax.set_ylabel(
                    f"Relative to {settings['reference_period']['start']}–{settings['reference_period']['end']} ({unit})"
                )
                ax.set_title(f"{chr(97 + column)}) {label}", loc="left")
                ax.legend(loc="best", frameon=False)
            path = output / f"comparison_{group:04d}_delta_monthly.png"
            export_figure(fig, path, dpi=200)
            plt.close(fig)
            paths.append(path)
            captions.append(
                f"{path.name}: {', '.join(labels)}. Future {settings['future_period']}. "
                "Grid lines are unweighted active-cell means; shaded ranges are spatial minima/maxima, "
                "not confidence intervals. Scalar lines show basin factors. Members are not pooled."
            )
        if settings.get("factor_layout") == "grid":
            for index, member in enumerate(settings["members"]):
                for key, label, unit, neutral in variables:
                    values = np.asarray(intent["factors"][index][key])
                    if neutral:
                        values = 100 * (values - 1)
                    values = np.where(active[None, :, :], values, np.nan)
                    limit = max(
                        max(
                            float(
                                np.nanmax(
                                    np.abs(
                                        (np.asarray(item[key]) - 1) * 100
                                        if neutral
                                        else np.asarray(item[key])
                                    )
                                )
                            )
                            for item in intent["factors"]
                        ),
                        1e-6,
                    )
                    fig, axes = new_fig(105, nrows=3, ncols=4, width_mm=190)
                    for month, ax in enumerate(axes.flat):
                        mesh = ax.pcolormesh(
                            coordinates["longitude"],
                            coordinates["latitude"],
                            values[month],
                            shading="nearest",
                            cmap=PRECIP_ANOMALY_COLORMAP
                            if key == "precip_ratio"
                            else CLIMATE_COLORMAPS["temp_crossing_freezing"],
                            vmin=-limit,
                            vmax=limit,
                        )
                        ax.set_title(MONTH_ABBR[month], loc="left")
                        ax.set_aspect("equal")
                    fig.colorbar(
                        mesh,
                        ax=list(axes.flat),
                        label=f"{label} change ({unit})",
                        shrink=0.8,
                    )
                    path = (
                        output
                        / f"run_{index + 3:0{intent['run_group_id_width']}d}_{key.lower()}_monthly_grid.png"
                    )
                    export_figure(fig, path, dpi=200)
                    plt.close(fig)
                    paths.append(path)
                    captions.append(
                        f"{path.name}: {member['model']} / {member['scenario']} / {member['member']}. "
                        "Monthly factors on the historical forcing grid; common scale across all months. "
                        "Coordinates are longitude/latitude; blank cells are outside active historical support."
                    )
    (output / "figure_captions.md").write_text(
        "\n\n".join(captions) + "\n", encoding="utf-8"
    )
    return paths
