"""Parity with eflowrisk R defaults, using recorded independent reference values."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.indicators import ResponseSeries, check, declarations, evaluate


def oracle_input() -> pd.Series:
    """Five years with leap days, ties, zeros, missing months and a rejected year."""
    time = pd.date_range("2000-01-01", "2004-12-31")
    day = np.arange(len(time))
    values = ((day * 17 % 97) + (day // 365) * 3).astype(float)
    values[day % 111 < 9] = 0
    values[(time.month == 2) & (time.day <= 7)] = np.nan
    values[(time.year == 2002) & (time.month <= 3)] = np.nan
    values[day % 211 == 0] = np.nan
    return pd.Series(values, index=time, name="q")


def test_all_indicators_match_recorded_r_oracle():
    daily = oracle_input()
    series = [
        ResponseSeries(
            "r1",
            "q",
            "gauge",
            0,
            daily.index,
            "standard",
            timedelta(days=1),
            "interval_end",
            "m3 s-1",
            daily.to_numpy(),
            daily.isna().to_numpy(),
        )
    ]
    reference = pd.read_csv(Path(__file__).parent / "data/eflow-erfa-r-oracle.csv")
    indicators = {item.name: item for item in declarations("eflow-erfa")}
    assert len(reference) == 210
    for anchor, rows in reference.groupby("anchor"):
        prepared = {}
        assert set(rows.indicator) == set(indicators)
        for row in rows.itertuples():
            indicator = indicators[row.indicator]
            result = evaluate(indicator, series, anchor=anchor, _prepared=prepared)
            np.testing.assert_allclose(
                result.values["gauge"],
                row.value,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{anchor}/{row.indicator}",
            )
            assert check(indicator, series, result, anchor=anchor)["status"] == "passed"
