"""Consolidated WF0 tables split back to the per-table frames (`table_files`, R15)."""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.climate_analysis.diagnostic_outputs import TABLES
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings
from blueearth_cst.climate_analysis.diagnostic_tables import compute
from blueearth_cst.climate_analysis.table_files import FILE_OF, FILES, combine, split


def _tables(gaps: bool):
    idx = pd.date_range("2000-01-01", "2006-12-31")
    rng = np.random.default_rng(1)
    precip = rng.gamma(0.6, 6.0, len(idx))
    temp = 20 + 5 * np.sin(np.arange(len(idx)) / 58.0)
    if gaps:
        precip[200:300] = np.nan
        temp[900:1300] = np.nan
    daily = pd.DataFrame({"precip": precip, "temp": temp}, index=idx)
    settings = parse_settings(None, {"start": 2000, "end": 2006}, ["era5"])
    return compute(daily, settings, 10, (2000, 2006))[1]


@pytest.mark.parametrize("gaps", [False, True])
def test_every_table_round_trips_through_its_file(tmp_path, gaps):
    tables = _tables(gaps)
    assert set(FILE_OF) == set(TABLES)
    tagged = {
        name: frame.assign(source="era5", period_start=2000, period_end=2006)
        for name, frame in tables.items()
    }
    for key, (_, members) in FILES.items():
        path = tmp_path / f"{key}.csv"
        combine(key, {name: tagged[name] for name in members}).to_csv(path, index=False)
        for name in members:
            legacy = tmp_path / f"legacy_{name}.csv"
            tagged[name].to_csv(legacy, index=False)
            expected = pd.read_csv(legacy)
            got = split(name, pd.read_csv(path))[list(expected.columns)]
            sort = [c for c in expected.columns if expected[c].notna().all()][:4]
            pd.testing.assert_frame_equal(
                got.sort_values(sort).reset_index(drop=True),
                expected.sort_values(sort).reset_index(drop=True),
                check_dtype=False,
                check_like=False,
            )
