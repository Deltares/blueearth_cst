"""Optional canonical subbasin tables, PNGs and recoverable captions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blueearth_cst.climate_analysis import diagnostic_figures as artists
from blueearth_cst.climate_analysis.diagnostic_captions import caption, title
from blueearth_cst.climate_analysis.diagnostic_tables import compute, write_json
from blueearth_cst.climate_analysis.figure_naming import figure_filename
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.plot_preset import CST_PRESET


def render_subbasins(sm, settings, sources):
    """Source render reuses tables; comparisons recompute matched-period tables."""
    folders = list(sm.params.subbasin_table_dirs)
    ids = sorted(
        set.intersection(
            *[
                {p.name for p in Path(folder).glob("subbasin_*") if p.is_dir()}
                for folder in folders
            ]
        )
    )
    comparison = len(sources) > 1
    dataset = "comparison" if comparison else sources[0]
    target = Path(sm.output.subbasin_figures)
    target.mkdir(parents=True, exist_ok=True)
    root = target.parent.parent
    captions_path = root / "plots" / "_engine" / "figure_captions.json"
    inventory = json.loads(captions_path.read_text(encoding="utf-8"))
    declared_ids = set.union(
        *[
            {p.name for p in Path(folder).glob("subbasin_*") if p.is_dir()}
            for folder in folders
        ]
    )
    inventory["unavailable_subbasins"] = {
        id_: "Not all sources have intersecting cells"
        for id_ in declared_ids - set(ids)
    }
    m0 = int(sm.params.m0)
    table_root = Path(sm.output.subbasin_tables) if comparison else Path(folders[0])
    if comparison:
        table_root.mkdir(parents=True, exist_ok=True)
    for id_ in ids:
        results = {}
        if comparison:
            period = tuple(settings["comparison_period"][k] for k in ("start", "end"))
            output_tables = table_root / id_
            output_tables.mkdir(parents=True, exist_ok=True)
            tables = []
            for s, folder in zip(sources, folders):
                frame = pd.read_csv(
                    Path(folder) / id_ / "daily_basin.csv",
                    index_col="date",
                    parse_dates=["date"],
                )
                result, table = compute(frame, settings, m0, period)
                results[s] = result
                tables.append((s, table))
            for name in (
                "precip_climatology",
                "temp_climatology",
                "annual_indices",
                "trends",
            ):
                pd.concat(
                    [t[name].assign(source=s) for s, t in tables], ignore_index=True
                ).to_csv(output_tables / f"{name}.csv", index=False)
        else:
            folder = Path(folders[0]) / id_

            def read(name):
                return pd.read_csv(folder / f"{name}.csv")

            annual = read("annual_indices").set_index("reporting_year")
            period = (
                (int(annual.index.min()), int(annual.index.max()))
                if len(annual)
                else tuple(settings["reference_period"][k] for k in ("start", "end"))
            )
            temp = read("temp_climatology").set_index("calendar_month")
            results[sources[0]] = {
                "clim": read("precip_climatology").set_index("calendar_month"),
                "tclim": temp if len(temp) else None,
                "annual": annual,
                "trends": read("trends").set_index("indicator").to_dict(orient="index"),
            }
        view = artists.View(
            sources,
            {
                s: "CHIRPS" if s in {"chirps", "chirps_global"} else s.upper()
                for s in sources
            },
            artists.source_styles(sources),
            results,
            np.arange(period[0], period[1] + 1),
            m0,
            tuple(settings["reference_period"][k] for k in ("start", "end")),
            settings["wet_day_threshold_mm"],
            pd.DataFrame(),
            np.array([]),
            support=f"{id_}, equal-weight intersecting native cells",
            max_missing_month=settings["max_missing_days_month"],
            max_missing_year=settings["max_missing_days_year"],
        )
        for form, variable, context in (
            ("seasonal", "precip", "monthly_clim_band"),
            ("series", "precip", "annual_trend_ts"),
            ("temperature", "temp", "monthly_clim_band"),
        ):
            carriers = [s for s, r in results.items() if r["tclim"] is not None]
            if variable == "temp" and len(carriers) < (2 if comparison else 1):
                continue
            with use_preset(CST_PRESET):
                fig = artists.CHARTS[form](view)
                name = figure_filename(dataset, variable, context, f"{id_}_avg")
                text = (
                    caption(form, view)
                    + " Method implementation was exercised on Ntoum real data; this run is not scientifically qualified. Grid resolution can make neighboring subbasin series identical."
                )
                figure_title = title(
                    form,
                    [
                        view.labels[s]
                        for s in (carriers if variable == "temp" else sources)
                    ],
                    period,
                    view.support,
                )
                export_figure(
                    fig,
                    target / name,
                    dpi=600,
                    annotation=getattr(sm.params, "annotation", "none"),
                    title=figure_title,
                    caption=text,
                )
                plt.close(fig)
            inventory["figures"][f"plots/subbasins/{name}"] = {
                "title": title(
                    form,
                    [
                        view.labels[s]
                        for s in (carriers if variable == "temp" else sources)
                    ],
                    period,
                    view.support,
                ),
                "caption": text,
                "sources": carriers if variable == "temp" else sources,
                "period": period,
                "reference": settings["reference_period"],
                "support": view.support,
                "availability": "available"
                if any(results[s]["annual"].total.notna().any() for s in sources)
                else "unavailable: no valid annual data",
                "captioned_path": None,
                "caveats": [
                    "Not scientifically qualified",
                    "Source grid may not resolve subbasins",
                ],
            }
    inventory["produced_paths"] = list(inventory["figures"]) + [
        "plots/_engine/figure_titles.json",
        "plots/figure_titles.md",
        "plots/_engine/figure_captions.json",
        "plots/figure_captions.md",
    ]
    write_json(captions_path, inventory)
    write_figure_titles(root / "plots", inventory["figures"])
    if comparison and inventory["unavailable_subbasins"]:
        from blueearth_cst.shared.snake_utils import log_row, plural

        log_row(
            f"Subbasin comparison omitted for {plural(len(inventory['unavailable_subbasins']), 'subbasin')}: "
            "not all sources have intersecting cells; details → plots/_engine/figure_captions.json",
            module="compare",
        )
    (root / "plots" / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{record['caption']}"
            for key, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )
