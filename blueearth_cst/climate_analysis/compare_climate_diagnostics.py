"""Rule 0.05: matched-period comparison, recomputed from retained daily series."""

import json
from pathlib import Path

import pandas as pd

from blueearth_cst.climate_analysis import diagnostics as dg
from blueearth_cst.climate_analysis.diagnostic_outputs import (
    diagnostic_comparison_outputs,
)
from blueearth_cst.climate_analysis.diagnostic_render import render
from blueearth_cst.climate_analysis.diagnostic_source_summary import (
    summarize_sources,
    write_comparison_table,
)
from blueearth_cst.climate_analysis.diagnostic_tables import compute, write_tables
from blueearth_cst.shared.snake_utils import log_row, plural


def run(sm):
    settings = dict(sm.params.settings)
    sources = list(sm.params.sources)
    paths = diagnostic_comparison_outputs(sm.params.comparison_dir, sources, settings)
    write_comparison_table(
        summarize_sources(dict(zip(sources, sm.input.source_stores, strict=True))),
        sm.params.comparison_dir,
    )
    m0 = int(sm.params.m0)
    period = tuple(settings["comparison_period"][k] for k in ("start", "end"))
    log_row(
        f"Comparing {plural(len(sources), 'source')} over reporting years "
        f"{period[0]}–{period[1]}; reporting year starts in month {m0}",
        module="compare",
    )
    daily = {
        s: pd.read_csv(path, index_col="date", parse_dates=["date"])
        for s, path in zip(sources, sm.input.daily)
    }
    binding = settings["pt_temperature_source"]
    temperature = daily[binding].temp if binding in daily else None
    tables = {
        s: compute(
            daily[s],
            settings,
            m0,
            period,
            temperature=temperature,
            temperature_source=binding,
        )[1]
        for s in sources
    }
    for table in tables.values():
        monthly = table["monthly_values"]
        table["monthly_values"] = monthly[monthly.reporting_year.between(*period)]
    from itertools import combinations

    import numpy as np

    for left, right in combinations(sources, 2):
        a = tables[left]["annual_indices"].set_index("reporting_year")
        b = tables[right]["annual_indices"].set_index("reporting_year")
        rows = []
        for indicator in ("total", "wet_days", "sdii", "Rx1day", "Rx5day"):
            difference = (a[indicator] - b[indicator]).dropna()
            for method, fn in (("pettitt", dg.pettitt), ("snht", dg.snht)):
                good = len(difference) >= 3 and difference.std() > 0
                values = (
                    fn(difference)
                    if good
                    else {
                        "n": len(difference),
                        "stat": np.nan,
                        "p": np.nan,
                        "break_after": np.nan,
                    }
                )
                rows.append(
                    values
                    | {
                        "series_kind": "difference",
                        "source_pair": f"{left}-{right}",
                        "indicator": indicator,
                        "method": method,
                        "seed": 0,
                        "n_sim": 20000 if method == "snht" else 0,
                        "reason": ""
                        if good
                        else "insufficient or constant paired samples",
                    }
                )
        tables[left]["homogeneity"] = pd.concat(
            [tables[left]["homogeneity"], pd.DataFrame(rows)], ignore_index=True
        )
    lineage = {
        s: json.loads(Path(p).read_text(encoding="utf-8"))["lineage"]
        for s, p in zip(sources, sm.input.metadata)
    }
    tol = dg.Completeness(
        settings["max_missing_days_month"], settings["max_missing_days_year"]
    )
    agreement, common = dg.agreement(
        {s: f.precip[dg.in_years(f.index, m0, period)] for s, f in daily.items()},
        m0,
        settings["wet_day_threshold_mm"],
        tol,
    )
    agreement = (
        agreement.rename(
            columns={
                "total": "mean_total",
                "wet_days": "mean_wet_days",
                "sdii": "mean_sdii",
            }
        )
        .rename_axis("source")
        .reset_index()
    )
    agreement["n_common_valid_years"] = len(common)
    log_row(
        (
            f"Source agreement uses {plural(len(common), 'common valid reporting year')}; "
            "period and counts → tables/agreement.md"
            if len(common)
            else "Numerical source agreement unavailable: no common valid reporting years; "
            "details → tables/agreement.md"
        ),
        module="compare",
    )
    log_row(
        "Source agreement measures consistency between datasets; it does not establish accuracy",
        module="compare",
    )
    Path(paths["compute"]["agreement"]).parent.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(paths["compute"]["agreement"], index=False)
    lines = [
        "# Source agreement",
        "",
        f"Common valid reporting years: {', '.join(map(str, common)) or 'none; numerical agreement unavailable'}",
        "",
        "Source agreement is not accuracy. Ntoum real-data exercised; not scientifically qualified.",
        "",
        agreement.to_string(),
    ]
    Path(paths["compute"]["agreement_md"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_tables(paths, tables, settings, m0, period, lineage)
    render(
        paths,
        settings,
        m0,
        sources,
        list(sm.params.pooled_roots),
        annotation=getattr(sm.params, "annotation", "none"),
    )
    if settings["subbasin_figures"]:
        from blueearth_cst.climate_analysis.diagnostic_subbasins import render_subbasins

        render_subbasins(sm, settings, sources)


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
