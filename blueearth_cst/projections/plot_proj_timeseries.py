"""Render WF2 suite products from durable analysis tables.

NOTE: Snakemake ``script:`` module; inputs, outputs and settings come from
``snakemake.input/output/params``. No standalone CLI or future-import preamble.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.projections.projection_analysis import MONTHS, PERIODS
from blueearth_cst.projections.projection_captions import SUITE_NOTE, caption, title
from blueearth_cst.projections.projection_figures import (
    SUITE_PRODUCTS,
    figure_key,
    period_folder,
)
from blueearth_cst.projections.projection_suite_figures import (
    GRID_PRODUCTS,
    dry_shading,
    grid_support,
    render_product,
)
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.snake_utils import (
    listed,
    log_row,
    plural,
    save_figure,
    tee_to_log,
)


def figure_counts(filename, tables, window_id=""):
    """Count plotted support, rather than claiming the full resolved ensemble."""
    filename = figure_key(filename)
    name = "model_changes"
    if filename.startswith(("trajectory-", "annual-")):
        name = "member_anomalies"
    elif filename.startswith("values-"):
        name = "seasonal_values"
    elif filename == "variability-member-ratios.png":
        name = "variability_member_ratios"
    elif filename == "availability.png":
        name = "coverage"
    elif filename.startswith("seasonal-cycle-"):
        name = "climatology"
    data = tables[name]
    if window_id and "window_id" in data:
        data = data.loc[data.window_id == window_id]
    if "temperature" in filename or "precipitation" in filename:
        data = data.loc[
            data.variable == ("temp" if "temperature" in filename else "precip")
        ]
    if filename.startswith("change-") and "cloud" not in filename:
        data = data.loc[
            (data.statistic == "mean")
            & (data.quantity == ("change_pct" if "-pct" in filename else "change"))
            & data.period.isin(PERIODS[:5])
        ]
    elif filename.startswith("monthly-change-"):
        data = data.loc[
            (data.statistic == "mean")
            & data.period.isin(MONTHS)
            & (
                data.quantity
                == ("change_pct" if "precipitation" in filename else "change")
            )
        ]
    elif "cloud" in filename:
        data = data.loc[(data.statistic == "mean") & (data.period == "annual")]
        keys = ["model", "scenario", "window_id"]
        temp = data.loc[
            (data.variable == "temp")
            & (data.quantity == "change")
            & np.isfinite(data.value),
            keys,
        ]
        precip = data.loc[
            (data.variable == "precip")
            & (data.quantity == "change_pct")
            & np.isfinite(data.value),
            keys,
        ]
        data = temp.merge(precip, on=keys).drop_duplicates()
    elif filename == "sign-agreement.png":
        data = data.loc[
            (data.statistic == "mean")
            & data.period.isin(PERIODS[:5])
            & (
                (data.variable.eq("temp") & data.quantity.eq("change"))
                | (
                    data.variable.eq("precip")
                    & data.quantity.isin(("change", "change_pct"))
                )
            )
        ]
    elif filename.startswith("values-"):
        data = data.loc[
            (data.statistic == GRID_PRODUCTS[filename][3])
            & (data.scenario != "historical")
        ]
    elif filename == "variability-member-ratios.png":
        data = data.loc[
            data.period.eq("annual")
            & (
                (data.variable.eq("temp") & data.statistic.eq("std_detrended"))
                | (
                    data.variable.eq("precip")
                    & data.statistic.isin(["std_detrended", "cv_detrended"])
                )
            )
        ]
    column = (
        "anomaly"
        if filename.startswith("trajectory-")
        else "value"
        if filename.startswith("annual-")
        else "ratio"
        if name == "variability_member_ratios"
        else "value"
        if "value" in data
        else None
    )
    used = data.loc[np.isfinite(data[column].astype(float))] if column else data
    if "available" in used:
        used = used.loc[used.available]
    if "variability_eligible" in used:
        used = used.loc[used.variability_eligible & used.ratio.gt(0)]
    if "reference" in used and "future" in used:
        used = used.loc[np.isfinite(used.reference) & np.isfinite(used.future)]
    return {
        "n_models": used.model.nunique(),
        "by_scenario": used.groupby("scenario").model.nunique().to_dict(),
        "n_excluded": int((~data.variability_eligible).sum())
        if "variability_eligible" in data
        else int(data.n_excluded.sum())
        if "n_excluded" in data
        else 0,
        "n_zero_future": int(data.status.eq("zero_future_variability").sum())
        if "status" in data
        else int(data.n_zero_future.sum())
        if "n_zero_future" in data
        else 0,
    }


def render_suite(
    tables,
    record,
    plots_root,
    *,
    annotation="none",
    scenarios=None,
    captions_json=None,
):
    """Render selected annotations and preserve independent figure metadata."""
    root = Path(plots_root)
    references = {tuple(w["reference"]) for w in record["windows"]}
    if len(references) == 1:
        reference = next(iter(references))
        window_description = listed([w["window_id"] for w in record["windows"]])
        window_description += f"; reference: {reference[0]}–{reference[1]}"
    else:
        window_description = listed(
            [
                f"{w['window_id']} (reference {w['reference'][0]}–{w['reference'][1]})"
                for w in record["windows"]
            ]
        )
    log_row(
        "Rendering windows: " + window_description,
        module="projection_plots",
    )
    folders = [
        "full_period",
        *dict.fromkeys(period_folder(w["window_id"]) for w in record["windows"]),
    ]
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)
    available = tables["availability"]
    expected = {
        (folder, filename)
        for product in SUITE_PRODUCTS.values()
        for folder in folders
        if (folder == "full_period") == (product["folder_kind"] == "full_period")
        for filename in product["filenames"]
    }
    actual = set(zip(available.folder, available.filename))
    if actual != expected or available.duplicated(["folder", "filename"]).any():
        raise ValueError(
            f"availability product contract differs: missing={expected - actual}, extra={actual - expected}"
        )
    manifest, skipped = {}, []
    for row in available.to_dict("records"):
        relative = f"{row['folder']}/{row['filename']}"
        if not isinstance(row["available"], bool):
            raise ValueError("availability flags must be booleans")
        if not row["available"]:
            skipped.append({"path": relative, "reason": row["reason"]})
            continue
        window_id = row.get("window_id") or ""
        counts = figure_counts(row["filename"], tables, window_id)
        counts["mostly_dry"] = dry_shading(row["filename"], tables, record, window_id)
        counts.update(grid_support(row["filename"], tables, window_id, scenarios))
        if figure_key(row["filename"]) == "availability.png":
            coverage = tables["coverage"]
            loaded = coverage.loc[coverage.available.astype(bool), "model"].unique()
            counts["n_unretrieved"] = int(coverage.model.nunique() - len(loaded))
        counts["variability_band"] = bool(
            figure_key(row["filename"]).startswith(("change-", "monthly-change-"))
            and "cloud" not in row["filename"]
            and "noise_halfwidth" in tables["change_summary"]
        )
        text = caption(row["filename"], record, window_id, counts)
        with use_preset(CST_PRESET):
            fig = render_product(
                row["filename"],
                tables,
                record,
                window_id=window_id,
                scenarios=scenarios,
            )
            try:
                save_figure(
                    root / relative,
                    module="projection_plots",
                    fig=fig,
                    annotation=annotation,
                    title=title(row["filename"], record, window_id),
                    caption=text,
                    dpi=600,
                    facecolor="white",
                )
            finally:
                plt.close(fig)
        manifest[relative] = {
            "title": title(row["filename"], record, window_id),
            "caption": text,
            "window_id": window_id,
            "counts": counts,
            "horizons": [
                h
                for w in record["windows"]
                if w["window_id"] == window_id
                for h in w.get("horizons", [])
            ],
        }
    pngs = {
        str(path.relative_to(root)).replace("\\", "/")
        for folder in folders
        for path in (root / folder).rglob("*.png")
    }
    if pngs != set(manifest):
        raise RuntimeError(
            f"owned PNGs differ from caption manifest: extra={pngs - set(manifest)}, missing={set(manifest) - pngs}"
        )
    manifest_path = (
        Path(captions_json)
        if captions_json is not None
        else root.parent / "_engine" / "figure_captions.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figure_titles(
        root, manifest, manifest_path=manifest_path.with_name("figure_titles.json")
    )
    markdown = [
        "# Projection figure captions",
        "",
        "Manifest keys identify the rendered figures.",
        "",
        SUITE_NOTE,
        "",
    ]
    for relative, item in manifest.items():
        markdown.extend([f"## {relative}", "", item["caption"], ""])
    markdown.extend(["## Not produced", ""])
    markdown.extend(f"- {item['path']}: {item['reason']}" for item in skipped)
    markdown.extend(
        [
            "",
            "## Retired paths",
            "",
            "Previous full-period/, period-<start>-<end>/, overview/ and windows/ paths are retired. "
            "Existing outputs are retained; use the canonical snake_case paths for new renders.",
            "",
        ]
    )
    (root / "figure_captions.md").write_text("\n".join(markdown), encoding="utf-8")
    log_row(
        f"Wrote {plural(len(manifest), 'figure')}; {len(skipped)} skipped; reasons → figure_captions.md",
        module="projection_plots",
    )
    return manifest, skipped


def run(sm):
    tables = {Path(path).stem: pd.read_csv(path) for path in sm.input.tables}
    record = json.loads(Path(sm.input.record).read_text(encoding="utf-8"))
    root = Path(sm.output.captions_md).parent
    expected_folders = {
        root / "full_period",
        *(root / period_folder(w["window_id"]) for w in record["windows"]),
    }
    if {Path(path) for path in sm.output.folders} != expected_folders:
        raise ValueError("declared suite folders differ from analysis record")
    tables["availability"]["window_id"] = tables["availability"].window_id.fillna("")
    render_suite(
        tables,
        record,
        root,
        annotation=getattr(sm.params, "annotation", "none"),
        scenarios=getattr(sm.params, "scenarios", None),
        captions_json=sm.output.captions_json,
    )


if "snakemake" in globals():
    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
