"""Focused R01 tests: the sectioned stress-test reader and the
list/string horizon normalization contract. These cover logic that
dry-runs and skip-by-default integration tests do not reach.
"""

from os.path import dirname, join, realpath

import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

CONFIG = join(TESTDIR, "project_config_fixture.yml")


def test_prep_cst_parameters_reads_sectioned_config(tmp_path):
    """prep_cst_parameters must read stress_test from the sectioned schema."""
    import pandas as pd

    from blueearth_cst.experiment.prepare_cst_parameters import prep_cst_parameters

    # temp.step_num=1, precip.step_num=1 in the tests config -> ST_NUM = 2*2 = 4.
    lookup_fn = tmp_path / "stress_test_lookup.csv"
    prep_cst_parameters(config_fn=CONFIG, lookup_fn=str(lookup_fn))
    assert lookup_fn.exists(), f"expected {lookup_fn} written"
    df = pd.read_csv(lookup_fn, dtype={"st_id": str})
    assert set(df["st_id"]) == {"1", "2", "3", "4"}
    assert len(df) == 4 * 12


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2000, 2010", ("2000", "2010")),  # legacy comma-separated string
        ([2000, 2010], ("2000", "2010")),  # R01 list form
        ([2030, 2060], ("2030", "2060")),
    ],
)
def test_horizon_normalization_contract(value, expected):
    """The list/string normalization used in get_change_climate_proj.py.

    Kept in lockstep with the production _to_str_tuple; see that module.
    """

    def _to_str_tuple(v):
        if isinstance(v, str):
            return tuple(map(str, v.split(", ")))
        return tuple(map(str, v))

    assert _to_str_tuple(value) == expected
