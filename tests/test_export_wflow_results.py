"""Unit cases for the wf3 reduction's path derivations (R07 B5/B6) and its
perturbation-axis collapse ([R9-3]).

B5 moved the realization index out of the wflow run CSV's file name and into
its run directory (``hydrology_runs/rlz_<n>/output/cst_<m>.csv``). The old
``output_rlz_<n>_cst_<m>.csv`` name was split on ``_`` at a fixed position --
a derivation that raises ``IndexError`` on the new name, and that is computed
on every row even when ``aggr_rlz=True`` leaves the value unused. That makes it
the one place in the move where a path rename breaks logic rather than a
pointer, so it gets direct coverage.

The [R9-3] cases carry a load the rest of the ladder cannot: every tracked
config uses a FLAT monthly perturbation vector, so the baseline manifest and
every fixture run are blind to the collapse by construction -- a seasonal
vector is the only input that can distinguish reading January from taking the
annual mean, and nothing on disk supplies one. Same lesson as R9-4: a check
that cannot fail on the fixture has to be written against a synthetic case.
"""


import numpy as np
import pandas as pd
import pytest


from blueearth_cst.experiment.export_wflow_results import (  # noqa: E402
    analyze_wflow_results,
    annual_perturbation,
    realization_from_run_csv,
)


def _cst_df(temp, precip):
    """A ``cst_<m>.csv`` as ``prepare_cst_parameters`` writes it: 12 rows, month 1..12."""
    return pd.DataFrame(
        {
            "month": np.arange(1, 13),
            "temp_mean": np.asarray(temp, dtype="float32"),
            "precip_mean": np.asarray(precip, dtype="float32"),
            "precip_variance": np.ones(12, dtype="float32"),
        }
    )


@pytest.mark.parametrize("rlz", [1, 2, 11])
def test_realization_index_comes_from_the_file_name(tmp_path, rlz):
    """R9 P2 put the index back in the stem, so that is where it is read from.

    It has moved twice: R07 B5 took it out of the filename into a `rlz_<n>/`
    run directory, and R9 dissolves that level. These cases follow the index,
    not the era.
    """
    csv = (
        tmp_path / "experiments" / "e" / "hydrology" / "wflow" / "output"
        / f"rlz_{rlz}_cst_3.csv"
    )
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert realization_from_run_csv(csv) == rlz
    assert realization_from_run_csv(str(csv)) == rlz


def test_the_cst_index_is_never_mistaken_for_the_realization(tmp_path):
    """The failure mode a `split("_")[-1]` derivation would produce SILENTLY.

    The stem now carries two indices. Splitting on "_" and taking the last
    field returns the CST member number, which is a plausible integer -- so
    every result row would be mislabelled with no error anywhere. Pinned with a
    case where the two indices differ and the wrong one is the tempting one.
    """
    csv = tmp_path / "output" / "rlz_2_cst_9.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert realization_from_run_csv(csv) == 2
    assert csv.stem.split("_")[-1] == "9"  # what the naive derivation returns


def test_the_directory_no_longer_carries_the_index(tmp_path):
    """The R7 shape must not keep working by accident.

    A `rlz_<n>/` directory with a `cst_<m>.csv` inside is the OLD layout. If it
    still resolved, a half-migrated tree would produce results silently instead
    of failing, which is exactly what P2 must not allow.
    """
    csv = tmp_path / "hydrology_runs" / "rlz_7" / "output" / "cst_0.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    with pytest.raises(ValueError, match="rlz_<n>_cst_<m>"):
        realization_from_run_csv(csv)


def test_realization_index_raises_naming_the_offending_path(tmp_path):
    csv = tmp_path / "model_runs" / "output" / "cst_1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    with pytest.raises(ValueError, match="rlz_<n>_cst_<m>"):
        realization_from_run_csv(csv)


def test_a_flat_vector_collapses_to_exactly_its_own_value():
    """The no-op case, asserted on IDENTITY rather than closeness.

    Every tracked config is flat, so this is the path the baseline manifest and
    every fixture run take. A weighted mean of twelve identical float32 values
    can land a ULP away from them, and the axis is written unrounded into a
    str-dtype frame -- so an inexact answer here is a baseline byte diff that
    means nothing. `is` on the float value, not `pytest.approx`.
    """
    df = _cst_df([3.0] * 12, [1.3] * 12)
    assert annual_perturbation(df, "temp_mean") == float(np.float32(3.0))
    precip = annual_perturbation(df, "precip_mean")
    assert precip == float(np.float32(1.3))
    # And the axis value the reduction actually writes, through the same
    # arithmetic as the caller.
    assert precip * 100 - 100 == float(np.float32(1.3)) * 100 - 100


def test_a_seasonal_vector_is_not_indexed_by_january():
    """The defect itself: JJA-only warming read as 0.0 degC because January is 0.

    Weights are the noleap month lengths, so a June-July-August delta covers
    30+31+31 = 92 of 365 days.
    """
    jja = [0.0] * 5 + [3.0] * 3 + [0.0] * 4
    df = _cst_df(jja, [1.0] * 12)
    expected = 3.0 * (30 + 31 + 31) / 365
    assert annual_perturbation(df, "temp_mean") == pytest.approx(expected, rel=1e-6)
    # What the pre-2026-08-07 read returned, and what must never come back.
    assert df["temp_mean"].iloc[0] == 0.0


def test_the_collapse_weights_by_month_length_not_by_count():
    """February and January must not carry the same weight.

    A plain arithmetic mean would return 3.0 * 1/12 for either month; the
    weighted mean separates them, which is what makes the temperature axis
    identical to WF2's duration-weighted annual change factor.
    """
    january = _cst_df([3.0] + [0.0] * 11, [1.0] * 12)
    february = _cst_df([0.0, 3.0] + [0.0] * 10, [1.0] * 12)
    assert annual_perturbation(january, "temp_mean") == pytest.approx(3.0 * 31 / 365)
    assert annual_perturbation(february, "temp_mean") == pytest.approx(3.0 * 28 / 365)
    assert annual_perturbation(january, "temp_mean") != annual_perturbation(
        february, "temp_mean"
    )


def test_the_axis_stays_evenly_spaced_across_the_grid():
    """The property that keeps the response surface RECTILINEAR.

    ``prepare_cst_parameters`` interpolates each member linearly between the
    min and max vectors, so the axis value must stay affine in the step index
    or the surface's rows stop being a regular grid. True for any weighted
    mean; it would fail for a max or a wet-season-only reduction, which is why
    it is pinned rather than assumed.
    """
    lo, hi = np.zeros(12), np.array([1.0, 2.0, 3.0] * 4)
    axis = [
        annual_perturbation(_cst_df(lo + (hi - lo) * f, [1.0] * 12), "temp_mean")
        for f in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    steps = np.diff(axis)
    assert steps == pytest.approx(np.full(4, steps[0]))
    assert steps[0] > 0


def test_row_order_does_not_change_the_answer():
    """Weights align to the 'month' column, not to file order."""
    df = _cst_df([0.0] * 5 + [3.0] * 3 + [0.0] * 4, [1.0] * 12)
    shuffled = df.iloc[::-1].reset_index(drop=True)
    assert annual_perturbation(shuffled, "temp_mean") == pytest.approx(
        annual_perturbation(df, "temp_mean")
    )


def test_a_partial_year_is_refused_by_name():
    df = _cst_df([1.0] * 12, [1.0] * 12).iloc[:6]
    with pytest.raises(ValueError, match="6 rows, expected one per month"):
        annual_perturbation(df, "temp_mean", "cst_4.csv")


def test_a_broken_month_column_is_refused_by_name():
    df = _cst_df([1.0] * 12, [1.0] * 12)
    df.loc[3, "month"] = 3  # month 4 duplicated as a second March
    with pytest.raises(ValueError, match="twelve calendar months"):
        annual_perturbation(df, "temp_mean", "cst_4.csv")


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
            results_dir=str(tmp_path),
            st_num=3,
        )
