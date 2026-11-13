"""Neutral indicator semantics, explicit references and operational refusals."""

import json
import pathlib
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.experiment.indicator_groups import indicator_groups
from blueearth_cst.experiment.indicator_registry import (
    DeclaredBundle,
    InsufficientReturnLevelBlocks,
    InvalidReturnLevelFit,
    ReturnLevelProbabilityNotCovered,
    declarations,
    indicator_unit_index,
    reduce_bundle,
    reduce_group_mean,
    reduce_run,
    resolve_month_reference,
)
from blueearth_cst.experiment.response_series import ResponseSeries
from blueearth_cst.experiment.scenario_rows import stochastic_rows


def frozen_candidate():
    """Import the frozen GF15 candidate as an independent value oracle.

    The production adapter must never be its own oracle, and the frozen module
    is the implementation the accepted 8x qualification was actually run
    against. A missing control fails rather than skips.
    """
    import importlib.util
    import sys

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "tests/data/gev-lmoments"
        / "candidate_adapter.py"
    )
    if not path.is_file():
        raise AssertionError(f"frozen parity control is missing: {path}")
    name = "gf15_frozen_candidate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def responses():
    times = pd.date_range("2046-01-02", "2054-12-31")
    rng = np.random.default_rng(12)
    result = []
    for run in ("run_01", "run_08"):
        for ordinal, location in enumerate(("9", "2")):
            # Native-first gauge peaks in June; lexically first peaks in January.
            values = (
                1
                + rng.random(len(times))
                + np.where(times.month == (6 if ordinal == 0 else 1), 10, 0)
            )
            result.append(
                ResponseSeries(
                    run,
                    "q",
                    location,
                    ordinal,
                    times,
                    "standard",
                    timedelta(days=1),
                    "interval_end",
                    "m3 s-1",
                    values,
                    np.zeros(len(times), dtype=bool),
                )
            )
    return tuple(result)


def test_registry_grains_vocabulary_and_empty_design_bundle():
    registry = declarations(("q", "gwr"))
    assert len(registry) == 12
    assert all(
        item.implementation_revision and item.value_validity for item in registry
    )
    class_c = [item for item in registry if item.reference]
    assert {item.grain for item in class_c} == {"run"}
    assert {item.reference_location for item in class_c} == {"class_c_first_gauge"}
    rows = stochastic_rows(2, 6, unit_id_capacity=14)
    groups, reference, realizations = indicator_groups(
        rows, n_realizations=2, st_num=6, unit_id_capacity=14
    )
    assert groups[""] == reference == ("run_01", "run_08")
    assert realizations["run_08"] == 2


def test_logical_unit_index_retains_empty_group_and_refuses_capacity():
    from blueearth_cst.experiment.scenario_rows import UnitNamespaceCapacityError

    runs = ("run_01", "run_02", "run_03")
    index = indicator_unit_index(
        runs,
        ("run_01", "run_03"),
        (DeclaredBundle("same_design_point", "", ("run_03", "run_01")),),
        unit_id_capacity=10,
    )
    assert [(item.unit_id, item.grain, item.member_run_id) for item in index] == [
        ("run_01", "run", "run_01"),
        ("run_03", "run", "run_03"),
        ("group_01", "bundle", "run_01"),
        ("group_01", "bundle", "run_03"),
    ]
    with pytest.raises(UnitNamespaceCapacityError, match="minimum replacement=4"):
        indicator_unit_index(
            ("run_1", "run_2", "run_3", "run_4"),
            ("run_1", "run_3"),
            (DeclaredBundle("same_design_point", "", ("run_1", "run_3")),),
            unit_id_capacity=3,
        )
    with pytest.raises(ValueError, match="unique evaluated"):
        indicator_unit_index(
            runs,
            ("run_01", "run_03"),
            (DeclaredBundle("same_design_point", "", ("run_01", "run_02")),),
            unit_id_capacity=10,
        )


@pytest.mark.parametrize("statistic", ["mean", "wetmonth_mean", "drymonth_mean"])
def test_group_means_weight_realizations_equally_and_propagate_nan(statistic):
    """Two and four January days have run means 2 and 8, hence group mean 5.

    The plausible pooled-day answer is 6, so this independent oracle detects
    sample-length weighting. The same fixed January month applies to Class C.
    """
    from blueearth_cst.experiment.indicator_registry import MonthReference

    indicator = next(m for m in declarations(("q",)) if m.statistic == statistic)
    members = []
    for run, count, value in (("run_01", 2, 2.0), ("run_02", 4, 8.0)):
        times = pd.date_range("2050-01-02", periods=count)
        members.append(
            ResponseSeries(
                run,
                "q",
                "9",
                0,
                times,
                "standard",
                timedelta(days=1),
                "interval_end",
                "m3 s-1",
                np.full(count, value),
                np.zeros(count, dtype=bool),
            )
        )
    reference = MonthReference(
        ("run_01",), "9", 1, "wet" if statistic == "wetmonth_mean" else "dry", ()
    )
    kwargs = {"reference": reference} if indicator.reference else {}
    result = reduce_group_mean(
        indicator,
        members,
        expected_run_ids=("run_01", "run_02"),
        anchor="YS-JAN",
        **kwargs,
    )
    assert result["9"] == 5.0
    assert (2 * 2 + 4 * 8) / 6 != result["9"]
    singleton = reduce_group_mean(
        indicator, members[:1], expected_run_ids=("run_01",), anchor="YS-JAN", **kwargs
    )
    assert singleton["9"] == 2.0
    invalid = replace(
        members[1], values=np.full(4, np.nan), missing=np.ones(4, dtype=bool)
    )
    assert np.isnan(
        reduce_group_mean(
            indicator,
            [members[0], invalid],
            expected_run_ids=("run_01", "run_02"),
            anchor="YS-JAN",
            **kwargs,
        )["9"]
    )
    with pytest.raises(ValueError, match="membership"):
        reduce_group_mean(
            indicator, members, expected_run_ids=("run_01",), anchor="YS-JAN", **kwargs
        )
    with pytest.raises(ValueError, match="unique"):
        reduce_group_mean(
            indicator,
            members,
            expected_run_ids=("run_01", "run_01"),
            anchor="YS-JAN",
            **kwargs,
        )
    if indicator.reference:
        with pytest.raises(ValueError, match="reference"):
            reduce_group_mean(
                indicator,
                members,
                expected_run_ids=("run_01", "run_02"),
                anchor="YS-JAN",
            )


def test_reference_uses_explicit_native_ordinal_and_is_order_invariant():
    series = responses()
    ref = resolve_month_reference(
        series, expected_run_ids=("run_01", "run_08"), which="wet"
    )
    reverse = resolve_month_reference(
        tuple(reversed(series)), expected_run_ids=("run_08", "run_01"), which="wet"
    )
    assert ref == reverse
    assert (ref.reference_location_id, ref.month) == ("9", 6)
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "wetmonth_mean"
    )
    values = reduce_run(indicator, series[:2], anchor="YE-DEC", reference=ref)
    assert values["9"] > values["2"]
    with pytest.raises(ValueError, match="reference"):
        reduce_run(indicator, series[:2], anchor="YE-DEC")
    incompatible = tuple(
        replace(item, location_ordinal=1 - item.location_ordinal)
        if item.run_id == "run_08"
        else item
        for item in series
    )
    with pytest.raises(ValueError, match="same first"):
        resolve_month_reference(
            incompatible, expected_run_ids=("run_01", "run_08"), which="wet"
        )


@pytest.mark.parametrize(
    "statistic,period,mode",
    [("return_level_max", 10, "max"), ("return_level_7day_min", 2, "min")],
)
def test_bundle_preserves_estimator_and_records_per_member_blocks(
    statistic, period, mode
):
    series = responses()
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == statistic
    )
    actual, evidence = reduce_bundle(
        indicator, series, expected_run_ids=("run_01", "run_08"), anchor="YE-DEC"
    )
    frames = [
        pd.DataFrame(
            {item.location_id: item.values for item in series if item.run_id == run},
            index=series[0].time,
        )
        for run in ("run_01", "run_08")
    ]
    blocks = pd.concat(
        [
            frame.resample("YE-DEC").max()
            if mode == "max"
            else frame.rolling(7).mean().resample("YE-DEC").min()
            for frame in frames
        ],
        ignore_index=True,
    )
    # The block extraction is unchanged (D1); the ESTIMATOR is not. The oracle is
    # the frozen candidate the 8x qualification was run against, applied to the
    # same pooled per-member blocks -- never the production adapter itself.
    probability = 1.0 - 1.0 / period if mode == "max" else 1.0 / period
    for location in actual.index:
        sample = blocks[location].to_numpy(dtype="float64")
        sample = sample[np.isfinite(sample)]
        reference = frozen_candidate().fit_case(sample, (probability,))
        assert reference["accepted"], location
        assert actual[location] == reference["quantiles"][0]["estimate"], location

    assert {item.total for item in evidence} == {18}
    assert {item.required for item in evidence} == {10}
    assert all(
        [entry[1] for entry in item.member_counts] == [9, 9] for item in evidence
    )
    # Retained fit evidence travels with each location (D3).
    for item in evidence:
        assert item.member_count == 2
        assert item.extraction_policy == (
            "member_local_annual_max/1"
            if mode == "max"
            else "member_local_rolling7_annual_min/1"
        )
        assert item.missingness_policy == "drop_nonfinite_blocks/1"
        assert item.partial_block_policy == "retain_partial_period_blocks/1"
        assert item.fit["estimator_id"] == "gf15-lmoments-c/1"
        assert item.fit["status"] == "accepted"
        assert item.fit["input"]["count"] == item.total
        assert len(item.parameters) == 3
        json.dumps(item.fit, allow_nan=False)

    with pytest.raises(InsufficientReturnLevelBlocks, match="actual=9"):
        reduce_bundle(
            indicator, series[:2], expected_run_ids=("run_01",), anchor="YE-DEC"
        )


def test_constant_sample_reaches_the_adapter_as_a_structured_refusal():
    """D1 removed the np.ptp guard, so the refusal now carries a FitResult."""
    series = responses()
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    constant = tuple(replace(item, values=np.ones(len(item.time))) for item in series)
    with pytest.raises(InvalidReturnLevelFit, match="invalid_range") as caught:
        reduce_bundle(
            indicator, constant, expected_run_ids=("run_01", "run_08"), anchor="YE-DEC"
        )
    result = caught.value.fit_result
    assert result is not None, "the refusal must carry its structured record"
    assert result.status == "refused"
    assert result.refusal_reasons == ("invalid_range",)
    assert "estimator refused" in str(caught.value)
    # The predecessor's unstructured constant guard must not survive.
    assert "constant sample" not in str(caught.value)


def test_publication_path_marks_refused_locations_unavailable_and_fits_the_rest():
    """One refused location never removes the group's other fitted locations."""
    from blueearth_cst.experiment import return_level_validation
    from blueearth_cst.indicators import loader

    series = responses()
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    mixed = tuple(
        replace(item, values=np.ones(len(item.time)))
        if item.location_id == "2"
        else item
        for item in series
    )
    validation = return_level_validation.build_declaration()
    context = {
        "anchor": "YE-DEC",
        "expected_run_ids": ("run_01", "run_08"),
        "validation": validation,
    }
    result = loader.evaluate(indicator, mixed, **context)
    assert result.statuses == {"2": "unavailable", "9": "available"}
    assert np.isnan(result.values["2"])
    assert (
        result.metadata["refusals"]["2"]["reason"] == "estimator refused: invalid_range"
    )
    fitted, _ = reduce_bundle(
        indicator,
        [item for item in series if item.location_id == "9"],
        expected_run_ids=("run_01", "run_08"),
        anchor="YE-DEC",
    )
    assert result.values["9"] == fitted["9"]
    assert [item.location_id for item in result.metadata["evidence"]] == ["9"]
    # The independent checker rechecks the fitted location and skips the refusal.
    outcome = loader.check(indicator, mixed, result, group=True, **context)
    assert (outcome["status"], outcome["numerical_checks"]) == ("passed", 1)


@pytest.mark.parametrize("field,value", [("units", "mm"), ("time_label", "instant")])
def test_indicator_refuses_incompatible_response_metadata(field, value):
    indicator = declarations(("q",))[0]
    series = [replace(item, **{field: value}) for item in responses()[:2]]
    with pytest.raises(ValueError, match="requirement"):
        reduce_run(indicator, series, anchor="YE-DEC")


def test_reference_refuses_unevaluated_members_and_preserves_month_tie():
    series = responses()
    with pytest.raises(ValueError, match="not evaluated"):
        resolve_month_reference(series, expected_run_ids=("run_01",), which="wet")
    tied = tuple(
        replace(item, values=(item.time.day == 2).astype(float)) for item in series
    )
    reference = resolve_month_reference(
        tied, expected_run_ids=("run_01", "run_08"), which="wet"
    )
    assert reference.month == 1


@pytest.mark.parametrize(
    "parameters, reason",
    [
        ({"c": float("nan"), "loc": 0.0, "scale": 1.0}, "invalid_parameters"),
        ({"c": 0.0, "loc": 0.0, "scale": -1.0}, "invalid_parameters"),
        ({"c": -1.5, "loc": 0.0, "scale": 1.0}, "invalid_parameters"),
    ],
    ids=["nan_shape", "negative_scale", "shape_at_or_below_minus_one"],
)
def test_invalid_estimator_result_refuses_without_fallback(
    monkeypatch, parameters, reason
):
    """A refused fit must abort the location; there is no fallback value."""
    from lmoments3 import distr

    monkeypatch.setattr(distr.gev, "lmom_fit", lambda **kwargs: dict(parameters))
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    with pytest.raises(InvalidReturnLevelFit, match=reason) as caught:
        reduce_bundle(
            indicator,
            responses(),
            expected_run_ids=("run_01", "run_08"),
            anchor="YE-DEC",
        )
    assert caught.value.fit_result.refusal_reasons == (reason,)


def test_unexpected_estimator_fault_is_not_converted_to_a_refusal(monkeypatch):
    """D3 removed the adapter-wide except Exception; faults keep their identity."""
    from lmoments3 import distr

    def faulty(**kwargs):
        raise RuntimeError("unexpected estimator failure")

    monkeypatch.setattr(distr.gev, "lmom_fit", faulty)
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    with pytest.raises(RuntimeError, match="unexpected estimator failure"):
        reduce_bundle(
            indicator,
            responses(),
            expected_run_ids=("run_01", "run_08"),
            anchor="YE-DEC",
        )


def test_uncovered_probability_is_a_fault_before_any_fit(monkeypatch):
    """D4: an off-domain probability aborts before the library is invoked."""
    from lmoments3 import distr

    calls = []
    real = distr.gev.lmom_fit

    def counting(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(distr.gev, "lmom_fit", counting)
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    # T=4 asks for p=0.75, which the bounded declaration does not cover.
    uncovered = replace(indicator, return_period=4)
    with pytest.raises(ReturnLevelProbabilityNotCovered, match="0.75"):
        reduce_bundle(
            uncovered,
            responses(),
            expected_run_ids=("run_01", "run_08"),
            anchor="YE-DEC",
        )
    assert calls == [], "the estimator was invoked despite an uncovered probability"


def test_covered_probabilities_are_the_declared_ones():
    """The two production probabilities must remain inside the declaration."""
    from blueearth_cst.experiment import return_level_validation as rlv

    declared = rlv.build_declaration()["tested_domain"]["probabilities"]
    assert declared == [0.9, 0.5]
    for indicator in declarations(("q",)):
        if indicator.grain != "bundle":
            continue
        probability = (
            1.0 - 1.0 / indicator.return_period
            if indicator.statistic == "return_level_max"
            else 1.0 / indicator.return_period
        )
        assert probability in declared, indicator.name


def test_shape_coverage_reports_an_in_domain_fit():
    """A fitted shape inside the assessed range is recorded as covered."""
    from blueearth_cst.experiment.indicator_registry import _shape_coverage

    block = _shape_coverage(0.05, [-0.2, 0.0, 0.2])
    assert block == {
        "tested_shapes_c": [-0.2, 0.0, 0.2],
        "tested_range_c": [-0.2, 0.2],
        "fitted_c": 0.05,
        "within_tested_range": True,
        "excess_beyond_tested_range": 0.0,
    }


def test_shape_coverage_measures_the_distance_outside():
    """Outside the range, the distance is recorded, not just the flag.

    A bare boolean invites treating a fitted 0.21 and a fitted 0.91 as the same
    finding. At the block counts this reduction produces they are not: the first
    is consistent with a true shape inside the domain, the second is not.
    """
    from blueearth_cst.experiment.indicator_registry import _shape_coverage

    near = _shape_coverage(0.21, [-0.2, 0.0, 0.2])
    far = _shape_coverage(-0.909, [-0.2, 0.0, 0.2])
    assert near["within_tested_range"] is False
    assert near["excess_beyond_tested_range"] == pytest.approx(0.01)
    assert far["within_tested_range"] is False
    assert far["excess_beyond_tested_range"] == pytest.approx(0.709)


def test_shape_coverage_is_unavailable_rather_than_covered_without_a_domain():
    """No declared shapes means null -- 'unavailable', never a silent pass."""
    from blueearth_cst.experiment.indicator_registry import _shape_coverage

    assert _shape_coverage(0.05, None) is None


def test_declared_shapes_never_raises_on_a_domainless_declaration():
    """Shape coverage is reported, never enforced, so it cannot abort a run.

    Contrast `_declared_probabilities`, which raises: a probability gates the
    request before any fit, while a shape is a property of the result.
    """
    from blueearth_cst.experiment.indicator_registry import _declared_shapes

    assert _declared_shapes({"tested_domain": {}}) is None
    assert _declared_shapes({"tested_domain": {"shapes_c": []}}) is None
    assert _declared_shapes({}) is None


def test_reduce_bundle_publishes_an_out_of_domain_shape(monkeypatch):
    """An out-of-domain fitted shape reduces and publishes; nothing refuses.

    This is the real pin for option B against a later change to option D, and it
    goes through `reduce_bundle` on purpose. A unit test of `_shape_coverage`
    alone cannot catch the regression that matters -- someone adding
    `if not coverage["within_tested_range"]: raise InvalidReturnLevelFit(...)`
    at the call site would leave such a test green while aborting every indicator
    set, since `InvalidReturnLevelFit` is caught nowhere in this package.

    The declaration is narrowed rather than the data widened: shrinking the
    tested range to a point puts every real fit outside it, which exercises the
    same branch without inventing a pathological sample.
    """
    from blueearth_cst.experiment import indicator_registry as mr

    monkeypatch.setattr(mr, "_declared_shapes", lambda validation: [0.0])
    indicator = next(
        item for item in declarations(("q",)) if item.statistic == "return_level_max"
    )
    values, evidence = reduce_bundle(
        indicator, responses(), expected_run_ids=("run_01", "run_08"), anchor="YE-DEC"
    )

    assert len(values) == 2, "the bundle still published every location"
    assert evidence, "evidence was recorded rather than the reduction aborting"
    for item in evidence:
        coverage = item.shape_coverage
        assert coverage["tested_range_c"] == [0.0, 0.0]
        assert coverage["within_tested_range"] is False
        assert coverage["excess_beyond_tested_range"] > 0.0


def test_shape_coverage_never_refuses_a_fit():
    """The helper itself has no refusal path, at the most extreme observed shape."""
    from blueearth_cst.experiment import return_level_validation as rlv
    from blueearth_cst.experiment.indicator_registry import (
        _declared_shapes,
        _shape_coverage,
    )

    declared = _declared_shapes(rlv.build_declaration())
    assert declared == [-0.2, 0.0, 0.2]
    # The most extreme shape seen in the retained production comparison.
    block = _shape_coverage(-0.909, declared)
    assert block["within_tested_range"] is False
    assert block["fitted_c"] == -0.909
