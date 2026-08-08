"""Unit tests for prepare_cst_parameters.prep_cst_parameters (R5 §8).

Drives the CSV generation on a synthetic in-memory config written to a
tmp_path YAML, with csv_fns=None so the function auto-names st_{i+1}.csv in
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
        glob.glob(str(tmp_path / "st_*.csv")),
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


# ---------------------------------------------------------------------------
# R11 P2 — zero-padded member ids (C27) and the design table (C23-C26)
# ---------------------------------------------------------------------------
#
# The tracked test config has ST_NUM = 6, so its width is 1 and NOTHING pads.
# That is correct per C27 -- st_1..st_6 already sort correctly -- but it means
# the fixture and the baseline cannot exercise padding at all. These cases carry
# that load: every padded assertion below uses a grid of ten or more.


def test_ids_are_unpadded_below_ten(tmp_path):
    """C27: width comes from the COUNT, so a small grid is not padded.

    Padding a 6-member grid would move every filename on every existing project
    to fix an ordering problem it does not have.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)  # 2 * 3 = 6
    prep_cst_parameters(cfg_path, csv_fns=None)

    names = sorted(os.path.basename(p) for p in glob.glob(str(tmp_path / "st_*.csv")))
    assert names == [f"st_{m}.csv" for m in range(1, 7)]


def test_ids_pad_once_the_count_reaches_ten(tmp_path):
    """C27: 12 members -> width 2, and LEXICAL order now matches RUN order.

    The falsifier is the sort: unpadded, `sorted()` yields st_1, st_10, st_11,
    st_12, st_2 ... which is the whole reason this change exists.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)  # 3 * 4 = 12
    prep_cst_parameters(cfg_path, csv_fns=None)

    names = sorted(os.path.basename(p) for p in glob.glob(str(tmp_path / "st_*.csv")))
    assert names == [f"st_{m:02d}.csv" for m in range(1, 13)]
    # lexical == numeric, which is the property, not the padding itself
    assert names == sorted(names, key=lambda n: int(n[3:5]))


def test_design_table_carries_one_row_per_member_plus_the_baseline(tmp_path):
    """C23: a row per design point AND a row for st_0 with every change zero."""
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)  # 12 members
    design = tmp_path / "stress_test_design.csv"
    prep_cst_parameters(cfg_path, csv_fns=None, design_fn=str(design))

    df = pd.read_csv(design, dtype={"st_id": str})
    assert list(df.columns) == [
        "st_id", "temp_change", "precip_change", "precip_variance_change",
    ]
    assert len(df) == 13  # 12 members + the baseline
    baseline = df.iloc[0]
    assert baseline["st_id"] == "00"
    assert (baseline[["temp_change", "precip_change",
                      "precip_variance_change"]] == 0.0).all()


def test_design_ids_are_the_filenames(tmp_path):
    """C26/C27: the id in the table IS the token in the filename.

    C28 will assert a results row against the design table's row for its
    st_id, so a table whose ids do not match the files on disk makes that check
    unjoinable.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)
    design = tmp_path / "stress_test_design.csv"
    prep_cst_parameters(cfg_path, csv_fns=None, design_fn=str(design))

    df = pd.read_csv(design, dtype={"st_id": str})
    on_disk = {os.path.basename(p)[3:-4]
               for p in glob.glob(str(tmp_path / "st_*.csv"))}
    assert set(df["st_id"]) - {"00"} == on_disk


def test_design_values_use_the_indicator_tables_own_reduction(tmp_path):
    """The design table's axes must equal what the results tables will carry.

    Both go through `annual_perturbation`, so this pins that they agree rather
    than merely look similar -- C28's consistency check is only meaningful if a
    disagreement here is impossible by construction.
    """
    from blueearth_cst.experiment.export_wflow_results import annual_perturbation

    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)
    design = tmp_path / "stress_test_design.csv"
    prep_cst_parameters(cfg_path, csv_fns=None, design_fn=str(design))

    df = pd.read_csv(design, dtype={"st_id": str}).set_index("st_id")
    for path in glob.glob(str(tmp_path / "st_*.csv")):
        member = pd.read_csv(path)
        st_id = os.path.basename(path)[3:-4]
        assert df.loc[st_id, "temp_change"] == pytest.approx(
            annual_perturbation(member, "temp_mean")
        )
        assert df.loc[st_id, "precip_change"] == pytest.approx(
            annual_perturbation(member, "precip_mean")
        )


def test_a_third_stress_axis_refuses_naming_c28(tmp_path):
    """C28's second obligation: a new dimension REFUSES, never silently drops.

    A third axis that merely went unrecorded would leave a design table
    describing a different experiment than the one that ran -- the exact failure
    a denormalised copy exists to prevent.
    """
    import yaml as _yaml

    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)
    cfg = _yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["workflows"]["climate_experiment"]["stress_test"]["wind"] = {
        "step_num": 1, "mean": {"min": _twelve(0.0), "max": _twelve(1.0)},
    }
    open(cfg_path, "w", encoding="utf-8").write(_yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="C28"):
        prep_cst_parameters(cfg_path, csv_fns=None)
