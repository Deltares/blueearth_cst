"""Unit cases for the wf3 reduction's path derivations (R07 B5/B6) and its
long-format output shape (R11 CR-2).

B5 moved the realization index out of the wflow run CSV's file name and into
its run directory (``hydrology_runs/rlz_<n>/output/cst_<m>.csv``). The older
``output_rlz_<n>_cst_<m>.csv`` name was split on ``_`` at a fixed position --
a derivation that raises ``IndexError`` on the new name, and that is computed
on every row even when ``aggr_rlz=True`` leaves the value unused. That makes it
the one place in the move where a path rename breaks logic rather than a
pointer, so it gets direct coverage. Both of those names carry the pre-R11-P2
``cst_`` member token, deliberately: they name eras, and that is what a tree
from either one actually holds. Everything the current reduction reads or
raises about says ``st_``.

The perturbation-axis collapse that used to be tested here moved to
``tests/test_surface_axes.py`` with the derivation itself: the axis is no longer
a column this writer computes, it is derived at reporting time from
``stress_test_lookup.csv``. Its cases carry a load the rest of the ladder cannot
and are worth finding under their new name -- every tracked config uses a FLAT
monthly perturbation vector, so the baseline manifest and every fixture run are
blind to the collapse by construction, and a seasonal vector is the only input
that can tell reading January apart from taking the annual mean.
"""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.experiment.export_wflow_results import (  # noqa: E402
    VALUE_SIGNIFICANT_DIGITS,
    MissingOutputColumnError,
    _format_value,
    analyze_wflow_results,
    member_from_run_csv,
    subcatchment_columns,
)


@pytest.mark.parametrize("rlz", [1, 2, 11])
def test_realization_index_comes_from_the_file_name(tmp_path, rlz):
    """R9 P2 put the index back in the stem, so that is where it is read from.

    It has moved twice: R07 B5 took it out of the filename into a `rlz_<n>/`
    run directory, and R9 dissolves that level. These cases follow the index,
    not the era.
    """
    csv = (
        tmp_path
        / "experiments"
        / "e"
        / "hydrology"
        / "wflow"
        / "output"
        / f"rlz_{rlz}_st_3.csv"
    )
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert member_from_run_csv(csv) == (rlz, 3)
    assert member_from_run_csv(str(csv)) == (rlz, 3)


def test_the_cst_index_is_never_mistaken_for_the_realization(tmp_path):
    """The failure mode a `split("_")[-1]` derivation would produce SILENTLY.

    The stem now carries two indices. Splitting on "_" and taking the last
    field returns the CST member number, which is a plausible integer -- so
    every result row would be mislabelled with no error anywhere. Pinned with a
    case where the two indices differ and the wrong one is the tempting one.
    """
    csv = tmp_path / "output" / "rlz_2_st_9.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    assert member_from_run_csv(csv) == (2, 9)
    assert csv.stem.split("_")[-1] == "9"  # what the naive derivation returns


def test_the_directory_no_longer_carries_the_index(tmp_path):
    """The R7 shape must not keep working by accident.

    A `rlz_<n>/` directory with a `cst_<m>.csv` inside is the OLD layout — the
    old token included, since that is what a stale tree actually holds. If it
    still resolved, a half-migrated tree would produce results silently instead
    of failing, which is exactly what P2 must not allow.
    """
    csv = tmp_path / "hydrology_runs" / "rlz_7" / "output" / "cst_0.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    with pytest.raises(ValueError, match="rlz_<n>_st_<m>"):
        member_from_run_csv(csv)


def test_realization_index_raises_naming_the_offending_path(tmp_path):
    csv = tmp_path / "model_runs" / "output" / "st_1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("time,Q_1\n")
    with pytest.raises(ValueError, match="rlz_<n>_st_<m>"):
        member_from_run_csv(csv)


def test_incomplete_run_coverage_fails_loudly(tmp_path):
    """D22: the check verifies what actually RAN, not what was declared.

    Its predecessor compared the DECLARED parameter files against ``st_num``,
    which this rule no longer takes as an input at all. The replacement is the
    stronger statement and the one that survives those files going away — and it
    matters more than a tidier error message: a partial member set produces a
    response surface silently missing grid cells, or a biased one if the absent
    members sit at one end of the grid, which is far harder to notice than a
    KeyError.
    """
    run_csv = tmp_path / "rlz_1_st_1.csv"
    run_csv.write_text("time,Q_1\n2000-01-01,1.0\n")
    with pytest.raises(ValueError, match=r"do not cover \[0, 1, 2, 3\]"):
        analyze_wflow_results(
            csv_fns=[str(run_csv)],
            results_dir=str(tmp_path),
            st_num=3,
            indicator_tokens=["q"],
            table_paths={"q": str(tmp_path / "q_indicators.csv")},
        )


def test_run_coverage_honours_st_start(tmp_path):
    """``ST_START`` is 1 without ``run_historical``, so a hardcoded 0 would make
    the check fail on every config that drops the baseline — a guard that fires
    on a legitimate configuration gets deleted rather than fixed."""
    for member in (1, 2):
        (tmp_path / f"rlz_1_st_{member}.csv").write_text("time,Q_1\n2000-01-01,1.0\n")
    with pytest.raises(ValueError, match=r"do not cover \[0, 1, 2\]"):
        analyze_wflow_results(
            csv_fns=sorted(str(p) for p in tmp_path.glob("rlz_*.csv")),
            results_dir=str(tmp_path),
            st_num=2,
            indicator_tokens=["q"],
            table_paths={"q": str(tmp_path / "q_indicators.csv")},
            st_start=0,
        )
    # The same runs are COMPLETE coverage once the baseline is not expected.
    analyze_wflow_results(
        csv_fns=sorted(str(p) for p in tmp_path.glob("rlz_*.csv")),
        results_dir=str(tmp_path),
        st_num=2,
        indicator_tokens=["q"],
        table_paths={"q": str(tmp_path / "q_indicators.csv")},
        st_start=1,
    )


# --- the long shape (R11 CR-2) ------------------------------------------------


def _run_csv(path, seed, offset, basavg=True):
    """A wflow run CSV: two gauges, optionally two subcatchments' worth of AET.

    **The header is the one wflow actually emits**, which this fixture got wrong
    from R11 until 2026-08-11: it wrote ``actual evapotranspiration_basavg``, a
    spelling 8bd51de retired in favour of ``<code>_<subcatchment>`` (``aet_101``,
    ``gwr_101``). The reducer's matcher was never updated, so the fixture and the
    code agreed with each other and with nothing else — every test here stayed
    green while a real run wrote ``aet_indicators.csv`` and
    ``recharge_indicators.csv`` (``gwr_indicators.csv`` since the 2026-08-11
    token rename) empty. A fixture that invents its producer's
    output format cannot fail when the producer changes it, so this one is now a
    literal copy of a real run's header.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=365 * 4, freq="D")
    data = {
        "Q_101": rng.gamma(2, 5, len(idx)) + offset,
        "Q_202": rng.gamma(2, 3, len(idx)) + offset,
    }
    if basavg:
        data["aet_101"] = rng.gamma(2, 1, len(idx))
        data["aet_202"] = rng.gamma(2, 1, len(idx))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=idx).rename_axis("time").to_csv(path)


def _reduce(tmp_path, tokens=("q",), rlz=2, st=2, basavg=True):
    """Run the reduction over a small synthetic sweep and return the tables.

    No parameter artifact is staged: since D22 the reducer reads none. The id
    width comes from ``index_width(st_num)`` and the axis is derived downstream
    from the lookup, so a fixture that wrote member files would be staging inputs
    nothing opens.
    """
    seed = 0
    for member in range(0, st + 1):
        for r in range(1, rlz + 1):
            _run_csv(tmp_path / f"rlz_{r}_st_{member}.csv", seed, member, basavg)
            seed += 1
    paths = {t: str(tmp_path / f"{t}_indicators.csv") for t in tokens}
    analyze_wflow_results(
        csv_fns=sorted(str(p) for p in tmp_path.glob("rlz_*.csv")),
        results_dir=str(tmp_path),
        st_num=st,
        indicator_tokens=list(tokens),
        table_paths=paths,
    )
    return {t: pd.read_csv(p) for t, p in paths.items()}


def test_the_header_is_five_columns_and_does_not_grow_with_gauges(tmp_path):
    """The point of the long shape: two gauges and twenty give the same header.

    FIVE since the axis columns were removed, and the header is now fixed
    against the stress-dimension count as well as the gauge count. `st_id` was
    ruled ALONGSIDE those columns "at this stage", with an explicit revisit when
    a third dimension arrives; the revisit happened, and the answer was that the
    axis is a derivation over the lookup rather than a column here.
    """
    q = _reduce(tmp_path)["q"]
    assert list(q.columns) == [
        "metric",
        "location",
        "st_id",
        "rlz_id",
        "value",
    ]
    assert set(q.location.astype(str)) == {"101", "202"}


def test_st_id_is_the_padded_member_token(tmp_path):
    """C27/C28: the id WRITTEN is the token in the member filename.

    Asserted against the file's TEXT, not a parsed frame, and that distinction
    is the point: `pd.read_csv` with no dtype infers `st_id` as int64, so `01`
    comes back as `1` and the padding appears to vanish. The bytes on disk are
    the contract -- a consumer joining results to the design table must read the
    column as a string, which both tables' own readers do.
    """
    _reduce(tmp_path, st=12)
    text = (tmp_path / "q_indicators.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    # Locate the column by NAME in the header rather than hard-coding its index:
    # this read is positional by necessity (the padding only exists in the bytes),
    # and a fixed index silently reads the neighbouring column after a reorder --
    # which is exactly what the 2026-08-11 reorder did to the literal `[1]` here.
    st_col = lines[0].split(",").index("st_id")
    written = {line.split(",")[st_col] for line in lines[1:]}
    assert written == {f"{m:02d}" for m in range(0, 13)}


def test_st_id_survives_a_dtype_aware_round_trip(tmp_path):
    """The join C28 exists to make possible, done the way a consumer must.

    It matters more since the axis columns went: the join to the lookup is now
    the ONLY way to place a row on the surface, and under the `st_0`-absent
    encoding a mis-keyed join presents as "every row is the baseline" rather
    than as an empty result.
    """
    _reduce(tmp_path, st=12)
    q = pd.read_csv(tmp_path / "q_indicators.csv", dtype={"st_id": str})
    assert set(q["st_id"]) == {f"{m:02d}" for m in range(0, 13)}


# C28's SECOND obligation -- the reducer refusing a design table whose axis this
# header cannot express -- retired with the axis columns: the header expresses no
# axis, so a third perturbation parameter no longer needs a results column. The
# contract barrier stands and is tested where it now lives alone,
# `tests/test_prepare_cst_parameters.py::test_a_third_stress_axis_refuses_naming_c28`.


def test_the_unperturbed_baseline_is_present_as_its_own_member(tmp_path):
    """[R9-5], ruled 2026-08-07. It is also what Q5 needs: the class-C month is
    picked from the baseline, and it cannot be picked from a record not there.

    Identified by `st_id` rather than by "the row whose axes are both zero" —
    the axes are gone, and that identification was never sound anyway: an
    identity member's row carried the same two zeros while denoting a
    differently-processed climate.
    """
    q = _reduce(tmp_path)["q"]
    baseline = q[q.st_id == 0]
    assert not baseline.empty
    assert baseline.metric.nunique() == q.metric.nunique()


def test_class_a_is_per_realization_while_b_and_c_are_pooled(tmp_path):
    q = _reduce(tmp_path, rlz=2)["q"]
    per_rlz = q[q.metric == "q_annual_mean"].rlz_id
    assert set(per_rlz) == {1, 2}
    for pooled_metric in (
        "q_return_level_10yr_max",
        "q_return_level_2yr_7day_min",
        "q_wettest_month_mean",
        "q_driest_month_mean",
    ):
        assert set(q[q.metric == pooled_metric].rlz_id) == {0}


def test_values_are_not_rounded(tmp_path):
    """`.round(2)` was an accidental drift buffer; dropping it is why the
    baseline comparator moves to a tolerance rather than a byte hash."""
    q = _reduce(tmp_path)["q"]
    assert not np.allclose(q.value, q.value.round(2))


def test_the_class_c_month_is_the_same_for_every_member(tmp_path):
    """Q5: the month is FIXED from st_0, so the surface shows how flow in a
    given month responds rather than conflating that with the month moving."""
    q = _reduce(tmp_path, st=2)["q"]
    wet = q[(q.metric == "q_wettest_month_mean") & (q.location == 101)]
    # One value per member; if the month were re-picked per member the values
    # would come from different months, which this cannot detect directly --
    # what it CAN pin is that every member produced exactly one such row.
    assert len(wet) == q.st_id.nunique()


def test_a_non_discharge_variable_gets_its_own_table_per_subcatchment(tmp_path):
    """The table carries the finest grain the run offers, on BOTH axes.

    Per realization because these metrics are linear in years (ruling b1), and per
    SUBCATCHMENT because the model declares them with `map = "subcatchment"` -- so
    a run emits one column per subcatchment and no whole-basin column exists to
    read. Q11 is why the reducer does not manufacture one by area-weighting these:
    whether subcatchments nest or tile decides whether that mean is valid at all.
    `BASIN_LOCATION` stays reserved for a genuine basin-scalar column, which is a
    WF1/TOML change to produce.
    """
    tables = _reduce(tmp_path, tokens=("q", "aet"))
    aet = tables["aet"]
    assert list(aet.metric.unique()) == ["aet_annual_total"]
    assert set(aet.location.astype(str)) == {"101", "202"}
    assert set(aet.rlz_id) == {1, 2}


def test_the_column_code_is_matched_not_the_indicator_token(tmp_path):
    """`precip` is emitted as `p_<id>`, and three of five tokens differ likewise.

    The regression this pins: a matcher keyed on the indicator token finds `aet`
    and `gwr` and silently finds nothing for `precip`, `snow` or `overland_flow`
    -- so the variables a naive fix is tested against are the ones it happens to
    work for. The 2026-08-11 `recharge` -> `gwr` rename made that trap WIDER, not
    narrower: two of five tokens now coincide with their code, so this test keeps
    its assertions on tokens that do not.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=365, freq="D")
    columns = pd.DataFrame(
        {"p_101": rng.gamma(2, 1, len(idx)), "qof_101": rng.gamma(2, 1, len(idx))},
        index=idx,
    ).columns
    assert subcatchment_columns(columns, "precip") == {"p_101": "101"}
    assert subcatchment_columns(columns, "overland_flow") == {"qof_101": "101"}
    # `p_` does not over-claim `qof_101`, and a requested variable the run never
    # emitted returns nothing rather than borrowing another variable's columns.
    assert subcatchment_columns(columns, "aet") == {}


def test_a_variable_the_run_never_emitted_is_refused_by_name(tmp_path):
    """An empty table is indistinguishable from "never requested", so it raises.

    This reverses the pre-2026-08-11 behaviour, which wrote the header-only table
    and deferred the mismatch to `check_model_reference`. That deferral does not
    hold: that rule compares the live model's digest against the one the
    experiment recorded, so it fires when the model CHANGES and is silent about a
    `wflow_outvars` entry the model never emitted a column for. Nothing else was
    watching, which is how the `_basavg` rename emptied two of three configured
    tables with every rule green.
    """
    with pytest.raises(MissingOutputColumnError, match="swe_<subcatchment>"):
        _reduce(tmp_path, tokens=("q", "snow"), basavg=False)


# --- written number format (t2608090806) ------------------------------------
#
# The tables are a deliverable and an interchange surface, so how a value is
# SPELLED on disk is a contract, not a display choice. Two properties, tested
# separately because they can regress independently.


def test_values_are_written_without_scientific_notation(tmp_path):
    """Excel does not open `6.3476255e-05` cleanly, and low flows reach ~1e-5.

    Asserted on the file's TEXT rather than a parsed frame, for the same reason
    `test_st_id_is_the_padded_member_token` is: `pd.read_csv` would turn the
    bytes back into floats and hide exactly the property under test.
    """
    _reduce(tmp_path, st=3)
    text = (tmp_path / "q_indicators.csv").read_text(encoding="utf-8")
    values = [line.split(",")[-1] for line in text.splitlines()[1:]]
    assert values, "no rows written"
    assert not [v for v in values if "e" in v.lower()]


def test_the_value_column_is_the_only_column_reformatted(tmp_path):
    """The key columns' bytes must survive the value formatter untouched.

    Its original witnesses were `temp_change`/`precip_change`, which a
    `to_csv(float_format=...)` would have rewritten `0.0` -> `0`. Those columns
    are gone, and every remaining key column is non-numeric — so the claim is
    re-witnessed rather than deleted: `st_id` is the join key to the lookup and
    carries ZERO PADDING that a broadcast reformat would strip, which is the
    same defect class in the column that now matters most.
    """
    _reduce(tmp_path, st=12)
    text = (tmp_path / "q_indicators.csv").read_text(encoding="utf-8")
    header, *body = text.splitlines()
    cols = header.split(",")
    keys = [i for i, c in enumerate(cols) if c != "value"]
    written = [line.split(",") for line in body]
    assert {row[cols.index("st_id")] for row in written} == {
        f"{m:02d}" for m in range(0, 13)
    }
    assert not [row[i] for row in written for i in keys if "." in row[i]]


def test_missing_values_render_as_an_empty_field_not_the_string_nan():
    """`.map()` bypasses pandas' `na_rep` path, so this is ours to handle."""
    assert _format_value(float("nan")) == ""


def test_the_cap_is_significant_digits_not_decimal_places():
    """The distinction the whole change rests on. Decimal-place rounding is
    scale-destroying -- `round(2)` sends 0.0007395697 to 0.0 -- which is what
    check_baseline.py:161 calls the accidental drift buffer P1 removed. A
    significant-digit cap keeps the same RELATIVE precision at every scale."""
    assert _format_value(0.0007395697) == "0.0007396"
    assert _format_value(115.48856) == "115.5"
    assert _format_value(6.3476255e-05) == "0.00006348"


def test_the_cap_cannot_trip_the_baseline_gate():
    """Pins "this rounding is invisible to the comparator" to the comparator's
    OWN constants, applied to the REAL reference table.

    Without this the safety argument lives only in a docstring, and a later
    tolerance tightening would silently invalidate it instead of failing here.
    Tolerances are derived exactly as `check_baseline` derives them: ATOL from
    each group's own mean magnitude, RTOL relative.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "dev" / "scripts"))
    import check_baseline as cb  # noqa: E402

    ref = pd.read_csv(
        root / "dev" / "baseline" / "indicator_ref" / "74ed83c06b2e7e6c.csv"
    )
    assert not ref.empty

    worst = 0.0
    for _, group in ref.groupby(list(cb.INDICATOR_GROUP_COLUMNS)):
        values = group[cb.INDICATOR_VALUE_COLUMN].astype(float)
        atol = cb.INDICATOR_ATOL_FRAC * float(values.abs().mean())
        for v in values:
            formatted = float(_format_value(v))
            allowed = max(atol, cb.INDICATOR_RTOL * abs(v))
            assert abs(formatted - v) <= allowed, (v, formatted, allowed)
            if v:
                worst = max(worst, abs(formatted - v) / abs(v))

    # Worst case for an N-significant-digit cap is 5e-N relative: half a unit in
    # the last kept digit, worst when the leading digit is 1. Assert the real
    # table honours that bound, then that the bound sits well inside the
    # comparator's RTOL -- so tightening RTOL fails HERE, at the argument, rather
    # than leaving a docstring that is quietly no longer true.
    theoretical = 5 * 10**-VALUE_SIGNIFICANT_DIGITS
    assert worst <= theoretical, worst
    assert theoretical < cb.INDICATOR_RTOL / 10
