"""Unit cases for the wf3 reduction's path derivations (R07 B5/B6).

B5 moved the realization index out of the wflow run CSV's file name and into
its run directory (``hydrology_runs/rlz_<n>/output/cst_<m>.csv``). The old
``output_rlz_<n>_cst_<m>.csv`` name was split on ``_`` at a fixed position --
a derivation that raises ``IndexError`` on the new name, and that is computed
on every row even when ``aggr_rlz=True`` leaves the value unused. That makes it
the one place in the move where a path rename breaks logic rather than a
pointer, so it gets direct coverage.
"""

import sys
from os.path import dirname, join, realpath

import pytest


from blueearth_cst.experiment.export_wflow_results import (  # noqa: E402
    analyze_wflow_results,
    realization_from_run_csv,
)


@pytest.mark.parametrize("rlz", [1, 2, 11])
def test_realization_index_comes_from_the_run_directory(tmp_path, rlz):
    csv = tmp_path / "experiments" / "e" / "hydrology_runs" / f"rlz_{rlz}" / \
        "output" / "cst_3.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert realization_from_run_csv(csv) == rlz
    assert realization_from_run_csv(str(csv)) == rlz


def test_realization_index_survives_a_cst_only_file_name(tmp_path):
    """The regression this guards: the file name alone no longer carries the
    index, so any name-splitting derivation must be gone."""
    csv = tmp_path / "rlz_7" / "output" / "cst_0.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert csv.name.count("_") == 1  # 'cst_0.csv' -- no rlz component left
    assert realization_from_run_csv(csv) == 7


def test_realization_index_raises_naming_the_offending_path(tmp_path):
    csv = tmp_path / "model_runs" / "output" / "cst_1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    with pytest.raises(ValueError, match="rlz_<n>"):
        realization_from_run_csv(csv)


def test_incomplete_stress_test_grid_fails_loudly(tmp_path):
    """B6 declares cst_1..cst_ST_NUM as a real input. If the declared set and
    ``st_num`` ever disagree, that must name the mismatch, not KeyError deep in
    the reduction loop on whichever row happens to reach the missing index."""
    run_csv = tmp_path / "rlz_1" / "output" / "cst_1.csv"
    run_csv.parent.mkdir(parents=True)
    run_csv.write_text("time,Q_1\n2000-01-01,1.0\n")
    with pytest.raises(ValueError, match=r"do not cover 1\.\.3"):
        analyze_wflow_results(
            csv_fns=[str(run_csv)],
            st_csv_fns=[str(tmp_path / "cst_1.csv"), str(tmp_path / "cst_2.csv")],
            indicators_dir=str(tmp_path),
            st_num=3,
        )
