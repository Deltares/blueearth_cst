"""`report.md` — the run, told in the order a reader needs it (design §5.9, step 7).

The report **reads** the durable records this workflow already writes —
`provenance.json`, `composition.csv`, the change-factor tables — and recomputes
nothing. That is not a style preference: a value recorded in two places has
disagreed five times in this milestone (the calendar, `n_years` twice, the
effective window twice), and every one was caught only because something compared
them. A report that derived its own disclaimer would be the sixth chance.

The disclaimer block is the part with a specification (§5.9). It must carry:

* requested vs effective reference window, and whether it was clipped;
* the alignment result against ``shared.historical_window``;
* the effective window length and any short-window warning;
* the spatial weighting scheme **and its approximation label**;
* the dry-month rule and its threshold;
* the catalog snapshot date;
* the count of requested-but-unresolved combinations, by status.

**Absence is stated, never implied.** On a basin with no clipped window, no short
window, no flagged months and no unresolved combinations, a disclaimer that
renders nothing is indistinguishable from one that is broken. Each line therefore
prints its negative — "no months flagged (threshold 0.1 mm/day)" — which is the
same lesson 6b's N6 records: on this fixture "nothing to report" is the correct
output *and* what a dead code path emits.
"""

from __future__ import annotations

#: The approximation D10's weighting carries, named so the reader can judge it.
WEIGHTING_APPROXIMATION = (
    "cell edges are adjacent-centre midpoints; true edges are unavailable because "
    "the catalog drops the bounds variables"
)


def _fmt_window(record):
    requested = record.get("reference_window_requested", "?")
    clipped = record.get("reference_window_clipped", False)
    if clipped:
        return f"requested {requested}, **clipped** to the historical experiment"
    return f"requested {requested}, used in full (no clip)"


def disclaimer_block(provenance, thresholds=None, max_flagged_months=None) -> list[str]:
    """The §5.9 disclaimer, one line per condition, negatives included."""
    reference = dict(provenance.get("reference_window", {}))
    sources = provenance.get("sources", [])
    composition = dict(provenance.get("composition", {}))
    flagged = provenance.get("flagged_months", [])

    effective = next(
        (
            s.get("reference_window_effective")
            for s in sources
            if s.get("reference_window_effective")
        ),
        "?",
    )
    n_years = next(
        (
            s.get("n_hyd_years_reference")
            for s in sources
            if s.get("n_hyd_years_reference") not in (None, "")
        ),
        "?",
    )

    lines = ["## Disclaimers", ""]
    lines.append(f"- **Reference window** — {_fmt_window(reference)}.")
    lines.append(
        f"  Effective window `{effective}`, {n_years} complete hydrological years."
    )

    try:
        short = int(n_years) < 20
    except (TypeError, ValueError):
        short = False
    lines.append(
        f"- **Window length** — {n_years} years"
        + (
            ", **below the 20-year floor**; statistics from it are correspondingly "
            "uncertain."
            if short
            else "; at or above the 20-year floor."
        )
    )

    scheme = provenance.get("weighting_scheme", "?")
    lines.append(
        f"- **Spatial weighting** — `{scheme}`. Approximation: "
        f"{WEIGHTING_APPROXIMATION}."
    )

    if thresholds:
        pairs = ", ".join(f"`{k}` < {v}" for k, v in sorted(dict(thresholds).items()))
        if flagged:
            over = [f for f in flagged if f.get("exceeds_max")]
            lines.append(
                f"- **Dry-month rule** — {len(flagged)} combination(s) have flagged "
                f"months ({pairs}); a flagged month reports no relative change and "
                f"keeps its absolute change. {len(over)} exceed "
                f"{max_flagged_months} flagged months."
            )
        else:
            lines.append(
                f"- **Dry-month rule** — no months flagged ({pairs}). Relative "
                "changes are defined for every month reported."
            )
    else:
        lines.append("- **Dry-month rule** — not applicable: no relative variables.")

    lines.append(
        f"- **Catalog snapshot** — {provenance.get('catalog_crawled_on', '?')}."
    )

    unresolved = dict(composition.get("unresolved_by_status", {}))
    if unresolved:
        detail = ", ".join(f"{n} {status}" for status, n in sorted(unresolved.items()))
        lines.append(
            f"- **Composition** — {composition.get('resolved', '?')} of "
            f"{composition.get('requested', '?')} requested combinations resolved; "
            f"{detail}. See `summary/composition.csv`."
        )
    else:
        lines.append(
            f"- **Composition** — all {composition.get('requested', '?')} requested "
            "combinations resolved; none skipped."
        )
    unavailable = provenance.get("unavailable_windows", [])
    if unavailable:
        lines.append(
            f"- **Unavailable windows** — {len(unavailable)} model/scenario/member/horizon "
            "combinations excluded from aligned change factors. Reasons and actual "
            "coverage are listed below and retained in `summary/provenance.json`."
        )
    lines.append("")
    return lines


def build(provenance, *, thresholds=None, max_flagged_months=None) -> str:
    """The whole report."""
    composition = dict(provenance.get("composition", {}))
    lines = [
        f"# Climate projections — {provenance.get('clim_project', '?')}",
        "",
        f"{composition.get('usable', composition.get('resolved', '?'))} usable data points from "
        f"{composition.get('models', '?')} model(s). Each "
        "`(model, scenario, member)` is one data point. The aligned change-factor "
        "CSVs are not averaged across models, members or scenarios; the analysis "
        "suite provides separately documented ensemble summaries.",
        "",
    ]
    lines += disclaimer_block(provenance, thresholds, max_flagged_months)

    exclusions = provenance.get("source_quality_exclusions", [])
    if exclusions:
        lines += ["## Source-quality exclusions", ""]
        for row in exclusions:
            lines.append(
                f"- **{row['dataset']} / {row['scenario']} / {row['member']}** — "
                f"{row['reason']}."
            )
        lines.append("")

    unavailable = provenance.get("unavailable_windows", [])
    if unavailable:
        lines += ["## Unavailable windows", ""]
        if composition.get("usable", composition.get("resolved")) == 0:
            lines += [
                "No aligned change factors could be derived for the requested windows.",
                "",
            ]
        for row in unavailable:
            identity = " / ".join(
                [
                    str(row.get("dataset", row.get("model", "?"))),
                    *(
                        str(row.get(key, "?"))
                        for key in ("scenario", "member", "horizon")
                    ),
                ]
            )
            lines.append(
                f"- **{identity}** — {row.get('window_kind', 'window')} requested "
                f"{row.get('requested_window', '?')}; available "
                f"{row.get('available_start', '?')} to {row.get('available_end', '?')}. "
                f"{row.get('reason', 'No usable coverage.')}"
            )
        lines.append("")

    lines += ["## Figures", ""]
    lines.append(
        "See [figure captions](plots/figure_captions.md) for the produced suite, "
        "analysis windows, methods, and products not produced with their reasons."
    )
    lines.append("")

    # S8-04/05/06: every result artifact now lives under summary/, and the tables
    # carry two values per row -- the future level and the change relative to the
    # baseline. Naming both here matters: `relative_value` mixes units across rows
    # by design (a difference for temperature, a percent for precipitation), and
    # `relative_units` is what says which.
    proj = provenance.get("clim_project", "?")
    lines += [
        "## Tables",
        "",
        f"- `summary/{proj}_change_factors_annual.csv` — one row per "
        "(model, scenario, member, horizon, variable, statistic). "
        "`absolute_value` is the future level in `units`; `relative_value` is the "
        "change against the reference window, in `relative_units` — a difference "
        "for an absolute variable, a percent for a relative one.",
        f"- `summary/{proj}_change_factors_monthly.csv` — the same, per calendar "
        "month.",
        "- `summary/composition.csv` — every **requested** combination and how it "
        "resolved.",
        "- `summary/provenance.json` — sources, digests, windows and settings for "
        "this run.",
        "",
        "Aligned precipitation levels use **mm/day**; relative changes use **%**. "
        "Suite captions state each figure's units: precipitation cycles and absolute "
        "changes use rates, annual overviews use annual integrals (mm/year), and "
        "anomalies use %. Precipitation CV uses period integrals and its ratio is dimensionless.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write(path, text) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
