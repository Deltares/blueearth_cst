"""Optional local P0/lab parity; never claims fresh-run or figure-byte parity.

Set WF2_PARITY_ANALYSIS_DIR and WF2_PARITY_LAB_DIR to compare retained snapshots
or a fresh run. Missing local inputs skip with their exact paths. Seasonal
precipitation changes use rates by owner ruling; lab totals are excluded there.
Family votes, provisional families, null bands and weighted pooled quantiles
have independent falsifiers and are deliberately outside lab parity.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WINDOW = "2036-2065"
CELL = ["scenario", "period", "variable", "statistic", "quantity"]
SEASONS = {"DJF", "MAM", "JJA", "SON"}
# Explicit schema additions: retained arrays/companions support OLS, CV and
# exact-coordinate null simulation; eligibility/status metadata supports partial
# ensembles. These are not unexplained numerical parity exemptions.
ADDITIONS = {
    "window_id",
    "horizons",
    "family",
    "family_source",
    "null_lo",
    "null_hi",
    "outside_null",
    "n_eligible",
    "n_excluded",
    "n_zero_future",
    "null_n_valid",
    "null_n_invalid",
    "null_status",
    "null_parameters",
    "null_reference_mean",
    "null_reference_sd",
    "mean_change_eligible",
    "variability_eligible",
    "reference_mean",
    "future_mean",
    "reference_sd",
    "future_sd",
    "reference_years",
    "future_years",
    "reference_values",
    "future_values",
    "reference_integrals",
    "future_integrals",
    "annual_rate",
    "reference_rate",
    "reference_start",
    "reference_end",
    "future_start",
    "future_end",
    "anomaly_units",
    "reason",
    "status",
    "quantity",
    "units",
    "n_reference",
    "n_future",
    "n_models_excluded",
    "requested_known",
    "country",
}
# Lab-only presentation fields, configured/aligned-window identity and retired
# smoothed fan outputs. No smoothing enters the expected trajectory range.
LAB_ONLY = {
    "horizon",
    "window_kind",
    "spread",
    "smoothed",
    "fraction_label",
    "n_within_band",
    "n_outside_band",
    "n_agree_outside_band",
    "band_note",
}


@pytest.fixture(scope="module")
def parity_inputs():
    actual = Path(
        os.environ.get(
            "WF2_PARITY_ANALYSIS_DIR",
            "C:/data/cmip6/gabon-ntoum-dev-ensemble/data/climate/projections/cmip6/analysis",
        )
    )
    lab = Path(
        os.environ.get(
            "WF2_PARITY_LAB_DIR",
            str(
                Path.home()
                / "workspace/wf0-lab/outputs/wf2/gabon-ntoum-dev-ensemble/results"
            ),
        )
    )
    if not actual.is_dir() or not lab.is_dir():
        pytest.skip(f"Local P0/lab parity unavailable: analysis={actual}; lab={lab}")

    def read(root, name):
        path = root / f"{name}.csv"
        assert path.is_file(), f"Incomplete parity input: {path}"
        return pd.read_csv(path, keep_default_na=False, na_values=[""])

    return lambda name: read(actual, name), lambda name: read(lab, name)


def _scope(table):
    if "window_kind" in table:
        table = table.loc[table.window_kind.eq("analysis") & table.horizon.eq("mid")]
    if "window_id" in table:
        table = table.loc[table.window_id.eq(WINDOW)]
    return table.copy()


def _rate_exception(table):
    if {"period", "variable", "quantity"} <= set(table):
        return table.loc[
            ~(
                table.variable.eq("precip")
                & table.quantity.eq("change_pct")
                & table.period.isin(SEASONS)
            )
        ]
    return table


def _schema(actual, expected, keys, fields, *, actual_only=(), lab_only=()):
    mapped = set(keys) | set(fields)
    assert not (set(actual) - mapped - ADDITIONS - set(actual_only)), (
        f"Unmapped producer columns: {set(actual) - mapped - ADDITIONS - set(actual_only)}"
    )
    assert not (set(expected) - mapped - LAB_ONLY - set(lab_only)), (
        f"Unmapped lab columns: {set(expected) - mapped - LAB_ONLY - set(lab_only)}"
    )
    assert mapped <= set(actual), f"Missing producer fields: {mapped - set(actual)}"
    assert mapped <= set(expected), f"Missing lab fields: {mapped - set(expected)}"


def _compare(actual, expected, keys, fields, label):
    assert not actual.duplicated(keys).any(), f"{label}: duplicate producer keys"
    assert not expected.duplicated(keys).any(), f"{label}: duplicate lab keys"
    merged = actual[keys + fields].merge(
        expected[keys + fields],
        on=keys,
        how="outer",
        suffixes=("_actual", "_lab"),
        validate="one_to_one",
        indicator=True,
    )
    assert merged._merge.eq("both").all(), (
        f"{label}: unmatched keys\n{merged.loc[merged._merge.ne('both'), keys + ['_merge']].head(10)}"
    )
    for field in fields:
        a, b = merged[f"{field}_actual"], merged[f"{field}_lab"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            valid = np.isclose(
                a.to_numpy(float),
                b.to_numpy(float),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            valid = a.eq(b) | (a.isna() & b.isna())
        assert np.all(valid), (
            f"{label}.{field}: {np.count_nonzero(~valid)} mismatches\n{merged.loc[~valid, keys + [field + '_actual', field + '_lab']].head(10)}"
        )


def test_changes(parity_inputs):
    actual, lab = parity_inputs
    got, want = _scope(actual("changes")), _scope(lab("changes"))
    keys = ["model", "scenario", "member", "period", "variable", "statistic"]
    fields = [
        "reference_value",
        "future_value",
        "n_reference",
        "n_future",
        "n_reference_excluded",
        "n_future_excluded",
        "status",
        "reason",
        "institution",
        "source_id",
        "units",
    ]
    _schema(
        got,
        want,
        keys,
        fields,
        actual_only={"value"},
        lab_only={
            "change",
            "change_pct",
            "reference_start",
            "reference_end",
            "future_start",
            "future_end",
        },
    )
    for quantity in ("change", "change_pct"):
        a = got.loc[got.quantity.eq(quantity)].copy()
        b = (
            want.loc[want.variable.eq("precip")].copy()
            if quantity == "change_pct"
            else want.copy()
        )
        b["quantity"] = quantity
        # Seasonal relative lab changes use totals; suite uses rates. Absolute
        # changes remain rate-based in both. Lab units describe the level, so
        # its separate percentage column maps to the producer's '%' unit.
        a, b = _rate_exception(a), _rate_exception(b)
        b["value"] = b[quantity]
        if quantity == "change_pct":
            b["units"] = "%"
        _compare(a, b, keys, [*fields, "value"], quantity)


@pytest.mark.parametrize(
    "product,reference",
    [
        ("model_changes", "c1_model_changes"),
        ("change_summary", "c1_summary"),
        ("climatology", "c2_climatology"),
        ("climatology_summary", "c2_climatology_summary"),
        ("monthly_change_summary", "c2_monthly_change"),
        ("variability_member_ratios", "c4_member_ratios"),
        ("variability_model_ratios", "c4_model_ratios"),
        ("sign_agreement", "c5_agreement"),
    ],
)
def test_reductions(parity_inputs, product, reference):
    actual, lab = parity_inputs
    got, want = _scope(actual(product)), _scope(lab(reference))
    if product == "sign_agreement":
        want = want.rename(columns={"n_models": "n", "n_agree": "k"})
    keys = [c for c in ["model", *CELL, "member", "side"] if c in want]
    fields = [c for c in want if c not in keys and c not in LAB_ONLY]
    if product == "variability_member_ratios":
        keys = ["model", "scenario", "member", "period", "variable", "statistic"]
    excluded = (
        {"k_fam", "n_fam", "small_change"} if product == "sign_agreement" else set()
    )
    _schema(got, want, keys, fields, actual_only=excluded)
    if "statistic" in want:
        extra = set(got.statistic) - set(want.statistic)
        assert extra <= {"std", "std_detrended", "cv_detrended"}, (
            f"Undocumented added statistics: {extra}"
        )
        print(f"{product}: separately tested additional statistics {sorted(extra)}")
        got = got.loc[got.statistic.isin(want.statistic.unique())]
    _compare(_rate_exception(got), _rate_exception(want), keys, fields, product)


def test_unsmoothed_trajectories(parity_inputs):
    actual, lab = parity_inputs
    got, want = (
        actual("member_anomalies"),
        lab("c3_member_anomalies").rename(columns={"reference": "reference_value"}),
    )
    # Lab units describe anomaly (%) for precip; producer distinguishes absolute
    # annual-integral units and anomaly_units explicitly.
    want = want.rename(columns={"units": "anomaly_units"})
    keys = ["model", "scenario", "member", "variable", "year"]
    fields = [
        "institution",
        "source_id",
        "value",
        "reference_value",
        "anomaly",
        "anomaly_units",
    ]
    _schema(got, want, keys, fields)
    _compare(got, want, keys, fields, "member_anomalies")
    trajectories = lab("c3_model_trajectories")
    _schema(
        trajectories,
        trajectories,
        ["model", "scenario", "variable", "year"],
        ["institution", "source_id", "units", "anomaly", "n_members"],
        actual_only={"smoothed"},
    )
    group = trajectories.groupby(["scenario", "variable", "year"]).anomaly
    expected = group.agg(
        mean="mean", median="median", min="min", max="max", n_models="count"
    ).reset_index()
    got = actual("trajectory_range")
    _schema(
        got,
        expected,
        ["scenario", "variable", "year"],
        ["mean", "median", "min", "max", "n_models"],
    )
    _compare(
        got,
        expected,
        ["scenario", "variable", "year"],
        ["mean", "median", "min", "max", "n_models"],
        "unsmoothed trajectory_range",
    )


def test_coverage_and_product_support(parity_inputs):
    actual, lab = parity_inputs
    got, want = actual("coverage"), lab("c0_coverage")
    keys = ["model", "scenario", "member"]
    fields = [c for c in want if c not in keys]
    _schema(got, want, keys, fields)
    _compare(got, want, keys, fields, "coverage")
    families = lab("c0_availability")
    assert set(families) == {
        "Model",
        "Family",
        "institution",
        "country",
        "historical",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    }
    listed = got[["source_id", "institution", "family", "country"]].drop_duplicates()
    _compare(
        listed.rename(columns={"source_id": "Model", "family": "Family"}),
        families,
        ["Model", "institution"],
        ["Family", "country"],
        "listed families",
    )
    matrix = lab("c0_coverage_matrix")
    assert set(matrix) == {
        "model",
        "institution",
        "source_id",
        "scenario",
        "n_requested",
        "n_resolved",
        "state",
        "members",
        "n_reference_years_min",
        "reason",
    }
    counts = (
        got.loc[got.available]
        .groupby(["model", "scenario"])
        .member.nunique()
        .rename("n_resolved")
        .reset_index()
    )
    _compare(
        counts,
        matrix.loc[matrix.n_resolved.gt(0)],
        ["model", "scenario"],
        ["n_resolved"],
        "resolved member counts",
    )
    availability = actual("availability")
    assert set(availability) == {
        "filename",
        "folder",
        "window_id",
        "available",
        "reason",
    }
    # Product availability is a new long predicate table, not the lab's wide
    # source-availability matrix. P0 must expose all eligible product families;
    # only the combined cloud needs a second distinct window and is unavailable.
    assert availability.available.eq(
        ~availability.filename.eq(
            "cmip6_precip_temp_annual_multi_period_change_cloud_basin_avg.png"
        )
    ).all()
