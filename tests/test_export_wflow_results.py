"""Unit cases for the wf3 reduction's path derivations (R07 B5/B6) and its
perturbation-axis collapse ([R9-3]).

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
    VALUE_SIGNIFICANT_DIGITS,
    MissingOutputColumnError,
    _format_value,
    analyze_wflow_results,
    annual_perturbation,
    member_from_run_csv,
    subcatchment_columns,
)


def _cst_df(temp, precip):
    """A ``st_<m>.csv`` as ``prepare_cst_parameters`` writes it: 12 rows, month 1..12."""
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
        annual_perturbation(df, "temp_mean", "st_4.csv")


def test_a_broken_month_column_is_refused_by_name():
    df = _cst_df([1.0] * 12, [1.0] * 12)
    df.loc[3, "month"] = 3  # month 4 duplicated as a second March
    with pytest.raises(ValueError, match="twelve calendar months"):
        annual_perturbation(df, "temp_mean", "st_4.csv")


def test_incomplete_stress_test_grid_fails_loudly(tmp_path):
    """B6 declares st_1..st_ST_NUM as a real input. If the declared set and
    ``st_num`` ever disagree, that must name the mismatch, not KeyError deep in
    the reduction loop on whichever row happens to reach the missing index."""
    run_csv = tmp_path / "rlz_1" / "output" / "st_1.csv"
    run_csv.parent.mkdir(parents=True)
    run_csv.write_text("time,Q_1\n2000-01-01,1.0\n")
    with pytest.raises(ValueError, match=r"do not cover 1\.\.3"):
        analyze_wflow_results(
            csv_fns=[str(run_csv)],
            st_csv_fns=[str(tmp_path / "st_1.csv"), str(tmp_path / "st_2.csv")],
            design_path=_write_design(tmp_path, 3),
            results_dir=str(tmp_path),
            st_num=3,
            indicator_tokens=["q"],
            table_paths={"q": str(tmp_path / "q_indicators.csv")},
        )


# --- the long shape (R11 CR-2) ------------------------------------------------


def _write_design(tmp_path, st_num, extra_axis=False):
    """A stress_test_design.csv matching what `_reduce` builds, as 3.09 writes it."""
    width = len(str(st_num))
    rows = [
        {
            "st_id": f"{0:0{width}d}",
            "temp_change": 0.0,
            "precip_change": 0.0,
            "precip_variance_change": 0.0,
        }
    ]
    for member in range(1, st_num + 1):
        rows.append(
            {
                "st_id": f"{member:0{width}d}",
                "temp_change": 0.5 * member,
                "precip_change": (1.0 + 0.1 * member) * 100 - 100,
                "precip_variance_change": 0.0,
            }
        )
    if extra_axis:
        for row in rows:
            row["wind_change"] = 0.0
    path = tmp_path / "stress_test_design.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


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


def _reduce(tmp_path, tokens=("q",), rlz=2, st=2, basavg=True, extra_axis=False):
    """Run the reduction over a small synthetic sweep and return the tables."""
    seed = 0
    for member in range(0, st + 1):
        for r in range(1, rlz + 1):
            _run_csv(tmp_path / f"rlz_{r}_st_{member}.csv", seed, member, basavg)
            seed += 1
    for member in range(1, st + 1):
        _cst_df(0.5 * member, 1.0 + 0.1 * member).to_csv(
            tmp_path / f"st_{member}.csv", index=False
        )
    paths = {t: str(tmp_path / f"{t}_indicators.csv") for t in tokens}
    analyze_wflow_results(
        csv_fns=sorted(str(p) for p in tmp_path.glob("rlz_*.csv")),
        st_csv_fns=sorted(str(p) for p in tmp_path.glob("st_*.csv")),
        design_path=_write_design(tmp_path, st, extra_axis=extra_axis),
        results_dir=str(tmp_path),
        st_num=st,
        indicator_tokens=list(tokens),
        table_paths=paths,
    )
    return {t: pd.read_csv(p) for t, p in paths.items()}


def test_the_header_is_seven_columns_and_does_not_grow_with_gauges(tmp_path):
    """The point of the long shape: two gauges and twenty give the same header.

    Seven since C28 added `st_id`. That column is the one thing in this header
    which is coupled to the stress-dimension COUNT rather than being fixed --
    ruled "at this stage" -- so the writer refuses a third axis rather than
    letting the shape drift a column at a time.
    """
    q = _reduce(tmp_path)["q"]
    assert list(q.columns) == [
        "metric",
        "location",
        "st_id",
        "rlz_id",
        "temp_change",
        "precip_change",
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
    """The join C28 exists to make possible, done the way a consumer must."""
    _reduce(tmp_path, st=12)
    q = pd.read_csv(tmp_path / "q_indicators.csv", dtype={"st_id": str})
    design = pd.read_csv(tmp_path / "stress_test_design.csv", dtype={"st_id": str})
    assert set(q["st_id"]) <= set(design["st_id"])
    assert set(q["st_id"]) == {f"{m:02d}" for m in range(0, 13)}


def test_a_third_stress_axis_refuses_naming_c28(tmp_path):
    """C28's second obligation, on the RESULTS side.

    A design table carrying an axis this header cannot express must stop the
    run and say so. Silently dropping it would leave results that describe a
    different experiment than the one that ran, and CR-2's fixed-shape property
    would degrade one column at a time with nothing noticing.
    """
    with pytest.raises(ValueError, match="C28"):
        _reduce(tmp_path, extra_axis=True)


def test_the_unperturbed_baseline_is_a_row_at_the_origin(tmp_path):
    """[R9-5], ruled 2026-08-07. It is also what Q5 needs: the class-C month is
    picked from the baseline, and it cannot be picked from a record not there."""
    q = _reduce(tmp_path)["q"]
    origin = q[(q.temp_change == 0) & (q.precip_change == 0)]
    assert not origin.empty
    assert origin.metric.nunique() == q.metric.nunique()


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
    assert len(wet) == q.temp_change.nunique()


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
    """`to_csv(float_format=...)` would also rewrite temp_change/precip_change
    (`0.0` -> `0`). Those are join keys for consumers and row-alignment keys for
    the baseline comparator, so their bytes must survive untouched."""
    _reduce(tmp_path, st=3)
    text = (tmp_path / "q_indicators.csv").read_text(encoding="utf-8")
    header, *body = text.splitlines()
    cols = header.split(",")
    axes = [cols.index("temp_change"), cols.index("precip_change")]
    written = {body_line.split(",")[i] for body_line in body for i in axes}
    assert all("." in v for v in written), written


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
