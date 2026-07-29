"""Unit tests for the shared ``stress_test_grid`` helper (R5 §3).

Pure arithmetic + strict validation; no heavy deps, no ``sys.modules``
pollution risk. Pins the strict contract both call sites (the Snakefile and
``prepare_cst_parameters.py``) now share.
"""


import pytest

from blueearth_cst.shared.snake_utils import stress_test_grid


def test_seed_config_grid():
    """The seed config (temp 1, precip 2) yields (2, 3, 6)."""
    cfg = {"temp": {"step_num": 1}, "precip": {"step_num": 2}}
    assert stress_test_grid(cfg) == (2, 3, 6)


def test_zero_step_num_is_single_point_axis():
    """step_num 0 is a valid degenerate axis (one point → count 1)."""
    cfg = {"temp": {"step_num": 0}, "precip": {"step_num": 0}}
    assert stress_test_grid(cfg) == (1, 1, 1)


@pytest.mark.parametrize(
    "cfg",
    [
        {"precip": {"step_num": 2}},  # missing temp axis section
        {"temp": {"step_num": 1}},  # missing precip axis section
        {"temp": {}, "precip": {"step_num": 2}},  # missing temp.step_num
        {"temp": {"step_num": 1}, "precip": {}},  # missing precip.step_num
    ],
)
def test_missing_step_num_raises_keyerror(cfg):
    """A missing axis section or step_num raises KeyError (no silent default)."""
    with pytest.raises(KeyError):
        stress_test_grid(cfg)


@pytest.mark.parametrize("bad", ["2", 1.5, True, False, None])
def test_non_integer_step_num_raises_valueerror(bad):
    """A non-integer step_num (incl. bool) raises ValueError."""
    cfg = {"temp": {"step_num": bad}, "precip": {"step_num": 2}}
    with pytest.raises(ValueError):
        stress_test_grid(cfg)


def test_negative_step_num_raises_valueerror():
    """A negative step_num raises ValueError."""
    cfg = {"temp": {"step_num": -1}, "precip": {"step_num": 2}}
    with pytest.raises(ValueError):
        stress_test_grid(cfg)
