"""Which file holds each WF0 diagnostic table (rule R15, t017 ruling D14).

``diagnostic_tables.compute`` still returns one frame per logical table
(``annual_indices``, ``coverage``, ...). Tables that share a key are written as
one file, parallel variables as a ``variable`` column, supporting checks under
``tables/supporting/`` and figure-only extracts under ``_engine/figure_data/``.
`combine` builds a file from its logical frames; `split` recovers one logical
frame from that file, so readers keep working with the logical frames.

The same files serve a dataset store and the source comparison.
"""

from __future__ import annotations

import pandas as pd

#: Columns every written frame carries (added by ``write_tables``).
TAGS = ("source", "period_start", "period_end")

#: Physical file key -> (folder relative to the unit root, logical tables).
FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "annual": ("tables", ("annual_indices", "valid_years", "rainfall_timing")),
    "monthly": ("tables", ("monthly_values", "coverage")),
    "climatology": ("tables", ("precip_climatology", "temp_climatology")),
    "spi": ("tables", ("spi",)),
    "drought_events": ("tables", ("drought_events",)),
    "trends": ("tables", ("trends",)),
    "homogeneity": ("tables/supporting", ("homogeneity",)),
    "spi_fit_checks": ("tables/supporting", ("spi_fit_checks",)),
    "dry_spells": ("tables/supporting", ("dry_spells",)),
    "dry_spell_exceedance": ("_engine/figure_data", ("dry_spell_exceedance",)),
    "anomaly_acf": ("_engine/figure_data", ("anomaly_acf",)),
    "trend_sensitivity": ("_engine/figure_data", ("trend_sensitivity",)),
    "pt_anomalies": ("_engine/figure_data", ("pt_anomalies",)),
}

#: Logical table -> physical file key.
FILE_OF = {table: key for key, (_, tables) in FILES.items() for table in tables}

_VALID_YEARS = ("full_year", "valid_precip", "valid_temp", "missing_days", "reason")
_COVERAGE = ("fraction", "missing_days", "class")
_CLIMATOLOGY = {"precip_climatology": "precip", "temp_climatology": "temp"}


def relative_path(key: str) -> str:
    """The file of ``key`` relative to its unit root."""
    folder, _ = FILES[key]
    return f"{folder}/{key}.csv"


def _timing_column(fraction: float) -> str:
    return f"day_{round(float(fraction) * 100)}"


def _merge(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    shared = (set(left) & set(right)) - set(keys)
    if shared:
        raise ValueError(f"tables sharing a file repeat columns: {sorted(shared)}")
    return left.merge(right, on=keys, how="outer", sort=True)


def combine(key: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One file's frame from its source-tagged logical frames."""
    if key == "annual":
        keys = [*TAGS, "reporting_year"]
        timing = frames["rainfall_timing"].pivot(
            index=keys, columns="fraction", values="day"
        )
        timing.columns = [_timing_column(value) for value in timing.columns]
        out = _merge(frames["annual_indices"], frames["valid_years"], keys)
        return _merge(out, timing.reset_index(), keys)
    if key == "monthly":
        keys = [*TAGS, "reporting_year", "calendar_month"]
        return _merge(frames["monthly_values"], frames["coverage"], keys)
    if key == "climatology":
        return pd.concat(
            [
                frames[table].assign(variable=variable)
                for table, variable in _CLIMATOLOGY.items()
            ],
            ignore_index=True,
        )
    (table,) = FILES[key][1]
    return frames[table]


def _as_bool(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore boolean columns that a merged CSV read back as objects."""
    for column in frame.columns:
        values = set(frame[column].dropna().unique())
        if frame[column].dtype == object and values and values <= {True, False}:
            frame[column] = frame[column].astype(bool)
    return frame


def split(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Recover logical ``table`` from the frame of the file holding it."""
    key = FILE_OF[table]
    if key == "annual":
        timing = [column for column in frame if column.startswith("day_")]
        if table == "rainfall_timing":
            # Timing has one row per valid precipitation year
            # (`accumulation_timing`), which is what `valid_precip` marks.
            valid = frame.loc[frame.valid_precip.astype(object).eq(True)]
            long = valid.melt(
                id_vars=[*TAGS, "reporting_year"],
                value_vars=timing,
                var_name="fraction",
                value_name="day",
            )
            long["fraction"] = long.fraction.str.removeprefix("day_").astype(int) / 100
            long["valid"] = long.day.notna()
            return long.reset_index(drop=True)
        if table == "valid_years":
            out = frame.loc[
                frame.full_year.notna(), [*TAGS, "reporting_year", *_VALID_YEARS]
            ]
            return _as_bool(out.reset_index(drop=True))
        # Annual indices have one row per full reporting year (`full_years`),
        # which is exactly what `full_year` marks.
        own = [c for c in frame if c not in (*_VALID_YEARS, *timing)]
        full = frame.full_year.astype(object).eq(True)
        return frame.loc[full, own].reset_index(drop=True)
    if key == "monthly":
        if table == "coverage":
            out = frame.loc[
                frame["class"].notna(),
                [*TAGS, "reporting_year", "calendar_month", *_COVERAGE],
            ]
            return out.reset_index(drop=True)
        out = frame.loc[frame.date.notna(), [c for c in frame if c not in _COVERAGE]]
        return _as_bool(out.reset_index(drop=True))
    if key == "climatology":
        out = frame.loc[frame.variable == _CLIMATOLOGY[table]].drop(columns="variable")
        return out.reset_index(drop=True)
    return frame
