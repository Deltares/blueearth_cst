"""Unit tests for prepare_cst_parameters.prep_cst_parameters (R5 §8).

Drives the CSV generation on a synthetic in-memory config written to a
tmp_path YAML, with csv_fns=None so the function auto-names cst_{i+1}.csv in
the config's directory. Uses only pandas/numpy/yaml — the function is already
import-clean (guarded), no heavy-dep stub, no sys.modules pollution risk.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment.prepare_cst_parameters import prep_cst_parameters


def _twelve(v):
    return [float(v)] * 12


def _write_cfg(tmp_path, *, temp_step=1, precip_step=2, var_min=1.0, var_max=1.0):
    """Write a synthetic snake config and return its path (str)."""
    cfg = {
        "workflows": {
            "climate_experiment": {
                "stress_test": {
                    "temp": {
                        "step_num": temp_step,
                        "mean": {"min": _twelve(0.0), "max": _twelve(3.0)},
                    },
                    "precip": {
                        "step_num": precip_step,
                        "mean": {"min": _twelve(0.7), "max": _twelve(1.3)},
                        "variance": {"min": _twelve(var_min), "max": _twelve(var_max)},
                    },
                }
            }
        }
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _read_cst_csvs(tmp_path):
    paths = sorted(
        glob.glob(str(tmp_path / "cst_*.csv")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    return [pd.read_csv(p) for p in paths]


def test_seed_like_grid_shape_and_endpoints(tmp_path):
    """temp step 1 x precip step 2 -> 6 CSVs, correct columns + linspace ends."""
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)
    prep_cst_parameters(cfg_path, csv_fns=None)

    dfs = _read_cst_csvs(tmp_path)
    assert len(dfs) == 6  # (1+1) * (2+1)

    for df in dfs:
        assert list(df.columns) == ["month", "temp_mean", "precip_mean", "precip_variance"]
        assert len(df) == 12  # one row per month

    # temp mean spans [0, 3]; precip mean spans [0.7, 1.3] across the grid.
    temp_means = np.concatenate([df["temp_mean"].values for df in dfs])
    precip_means = np.concatenate([df["precip_mean"].values for df in dfs])
    assert temp_means.min() == pytest.approx(0.0)
    assert temp_means.max() == pytest.approx(3.0)
    assert precip_means.min() == pytest.approx(0.7, abs=1e-6)
    assert precip_means.max() == pytest.approx(1.3, abs=1e-6)


def _precip_variance_grid_max(tmp_path):
    dfs = _read_cst_csvs(tmp_path)
    return max(df["precip_variance"].max() for df in dfs)


def test_precip_variance_grid_uses_max_endpoint(tmp_path):
    """The precip_variance grid spans up to variance.max (t260720a, fixed).

    Regression guard for the max-reads-min bug: prepare_cst_parameters once read
    variance['min'] into the max endpoint, collapsing a non-degenerate range
    (min=1.0, max=1.5) to [1.0, 1.0]. With the fix the grid max is variance.max.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=1, var_min=1.0, var_max=1.5)
    prep_cst_parameters(cfg_path, csv_fns=None)
    assert _precip_variance_grid_max(tmp_path) == pytest.approx(1.5)
