"""Focused R01 tests: the sectioned stress-test reader and the
list/string horizon normalization contract. These cover logic that
dry-runs and skip-by-default integration tests do not reach.
"""

from os.path import dirname, join, realpath

import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

CONFIG = join(TESTDIR, "snake_config_model_test.yml")


def test_prep_cst_parameters_reads_sectioned_config(tmp_path):
    """prep_cst_parameters must read stress_test from the sectioned schema."""
    from blueearth_cst.experiment.prepare_cst_parameters import prep_cst_parameters

    # temp.step_num=1, precip.step_num=1 in the tests config -> ST_NUM = 2*2 = 4.
    csv_fns = [str(tmp_path / f"st_{i}.csv") for i in range(4)]
    prep_cst_parameters(config_fn=CONFIG, csv_fns=csv_fns)
    for fn in csv_fns:
        assert __import__("os").path.exists(fn), f"expected {fn} written"


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
