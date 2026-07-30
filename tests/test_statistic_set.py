"""Statistic-set tests for step 5d (design §5.6 "Statistics").

J4 and J5 from ``dev/working/2026-07-30_wf2-5d-falsifier.md`` are **not visible to
the tree diff**, because shipped configs do not opt into tail quantiles. They live
here so their absence from the gate is not mistaken for absence of the feature.
"""

import pytest

from blueearth_cst.projections.get_change_climate_proj import (
    DEFAULT_STATS,
    quantile_label,
)


# --- J1: the default set is exactly three -------------------------------------


def test_J1_default_statistic_set_is_mean_median_std():
    """`var` and the four tail quantiles are dropped, not hidden."""
    assert DEFAULT_STATS == ("mean", "median", "std")
    for dropped in ("var", "q_90", "q_75", "q_10", "q_25"):
        assert dropped not in DEFAULT_STATS


# --- J5: an emitted quantile carries its effective sample size ----------------


@pytest.mark.parametrize("stat", ["q_90", "q_75", "q_10", "q_25"])
def test_J5_quantiles_are_labelled_with_the_sample_size(stat):
    """"The second-highest of 20" should be self-evident at the point of use.

    The label rides on the `stats` coordinate, so it reaches the CSV and the
    report without either needing to know about sample sizes.
    """
    assert quantile_label(stat, 20) == f"{stat}[n=20]"


@pytest.mark.parametrize("stat", ["mean", "median", "std", "var"])
def test_J5_non_quantile_statistics_are_not_labelled(stat):
    """`mean[n=20]` would be noise: a mean is not a claim 20 samples undermines."""
    assert quantile_label(stat, 20) == stat


def test_J5_label_is_omitted_when_the_sample_size_is_unknown():
    """A label asserting `n=0` would be worse than no label."""
    assert quantile_label("q_90", 0) == "q_90"
    assert quantile_label("q_90", None) == "q_90"


def test_J5_label_carries_the_window_actually_used_not_a_nominal_one():
    """Different windows must produce different labels, or the label says nothing."""
    assert quantile_label("q_90", 20) != quantile_label("q_90", 40)
    assert quantile_label("q_90", 40) == "q_90[n=40]"


# --- J4: opting in must still work --------------------------------------------


def test_J4_quantiles_remain_reachable_by_explicit_request():
    """Opt-in means the capability survives; 5d narrows the DEFAULT, not the API.

    Exercised at the signature level here — the end-to-end path is
    config `stats:` -> Snakefile param -> derive_one_point -> aggregation, and the
    aggregation is covered by tests/test_get_change_climate_proj.py, which passes
    an explicit `stats=` list.
    """
    import inspect

    from blueearth_cst.projections.get_change_climate_proj import (
        get_change_annual_clim_proj,
    )

    sig = inspect.signature(get_change_annual_clim_proj)
    assert "stats" in sig.parameters
    # None, not a hard-coded list: the default is resolved in-body so a caller
    # passing nothing gets the v2.0 set rather than a frozen historical one.
    assert sig.parameters["stats"].default is None
