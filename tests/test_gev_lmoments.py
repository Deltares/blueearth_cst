"""Typed GEV L-moment adapter against independent retained controls (D2, D3).

Production code is never its own oracle here. Three independent sources are used:

* the FROZEN candidate `candidate_adapter.py` the 8x qualification was actually
  run against -- the pinned implementation defines parity, so the production
  adapter must reproduce it bit for bit;
* the retained attempt-2 `numerical-controls.json`, whose branch and quantile
  references were computed at 80-120 decimal digits by a separate oracle and
  carry the expected float64 bit patterns;
* deliberate mutants, which must fail.

Every control is tracked repository evidence. A missing one FAILS this module
rather than skipping it: a control that quietly vanishes is how a test starts
asserting a dead contract while still reporting green.
"""

import hashlib
import importlib.util
import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from blueearth_cst.experiment import gev_lmoments as gev

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/data/gev-lmoments"
CONTROLS = FIXTURES / "numerical-controls.json"
CANDIDATE = FIXTURES / "candidate_adapter.py"

# The readiness sample, carried over verbatim from the frozen control suite.
SAMPLE = np.array([0.0, 0.2, 0.2, 0.5, 0.75, 1.0, 1.4, 2.0])
PROBABILITIES = (0.9, 0.5)


def _load_frozen_candidate():
    """Import the frozen candidate as a module, without putting it on sys.path."""
    if not CANDIDATE.is_file():
        raise AssertionError(f"frozen parity control is missing: {CANDIDATE}")
    spec = importlib.util.spec_from_file_location("gf15_frozen_candidate", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _controls() -> dict:
    if not CONTROLS.is_file():
        raise AssertionError(f"retained numerical controls are missing: {CONTROLS}")
    return json.loads(CONTROLS.read_text(encoding="utf-8"))


def _bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


@pytest.fixture(scope="module")
def frozen():
    return _load_frozen_candidate()


@pytest.fixture(scope="module")
def controls():
    return _controls()


def test_controls_are_present_and_tracked():
    """Guard the guards: absent controls must fail, never skip."""
    import subprocess

    for path in (CONTROLS, CANDIDATE):
        assert path.is_file(), path
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path.relative_to(REPO).as_posix()],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0, f"control is not tracked: {path}"


def test_interface_takes_only_a_sample_and_probabilities():
    """No indicator, run id, truth, RNG or configuration may enter the adapter."""
    import inspect

    assert list(inspect.signature(gev.fit_case).parameters) == [
        "sample",
        "probabilities",
    ]


# --------------------------------------------------------------------------
# Parity with the frozen candidate the qualification was run against
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    [
        SAMPLE,
        SAMPLE + 100.0,
        SAMPLE * 1000.0,
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5, 0.5, 0.5, 0.5, 1.0, 2.0]),
        np.array([-5.0, -1.0, 0.0, 0.0, 3.0, 9.0, 12.0]),
        np.linspace(0.0, 1.0, 30),
    ],
    ids=["readiness", "translated", "scaled", "minimal", "ties", "negatives", "dense"],
)
def test_accepted_estimates_match_the_frozen_candidate_bit_for_bit(frozen, sample):
    produced = gev.fit_case(sample, PROBABILITIES)
    expected = frozen.fit_case(sample, PROBABILITIES)
    assert produced.accepted == expected["accepted"]
    if not produced.accepted:
        return
    for row, reference in zip(produced.quantiles, expected["quantiles"], strict=True):
        assert row.p == reference["p"]
        assert _bits(row.estimate) == _bits(reference["estimate"])
        assert _bits(row.normalized) == _bits(reference["normalized"])
    for name in ("c", "loc", "scale"):
        assert _bits(produced.normalized_parameters[name]) == _bits(
            expected["normalized_parameters"][name]
        )
        assert _bits(produced.physical_parameters[name]) == _bits(
            expected["physical_parameters"][name]
        )
    assert produced.moments["t3"] == expected["moments"]["t3"]
    assert produced.normalization["a"] == expected["normalization"]["a"]
    assert produced.normalization["s"] == expected["normalization"]["s"]


def test_diagnostic_counts_match_the_frozen_candidate(frozen):
    produced = gev.fit_case(SAMPLE, PROBABILITIES)
    expected = frozen.fit_case(SAMPLE, PROBABILITIES)
    for coordinate in ("normalized", "physical"):
        reference = expected[f"{coordinate}_diagnostics"]
        actual = getattr(produced, f"{coordinate}_diagnostics")
        assert actual["sample_outside_support"] == reference["sample_outside_support"]
        assert (
            actual["nonfinite_log_density_count"]
            == reference["nonfinite_log_density_count"]
        )


def test_sample_digest_matches_the_frozen_candidate(frozen):
    produced = gev.fit_case(SAMPLE, PROBABILITIES)
    expected = frozen.fit_case(SAMPLE, PROBABILITIES)
    assert produced.input["sample_sha256"] == expected["input"]["sample_sha256"]
    assert (
        produced.input["sample_sha256"]
        == hashlib.sha256(np.asarray(SAMPLE, dtype=np.float64).tobytes()).hexdigest()
    )


# --------------------------------------------------------------------------
# Independent high-precision controls
# --------------------------------------------------------------------------


def test_branch_controls_reproduce_the_pinned_inversion(controls):
    """All 19 retained branch references, including the Gumbel snap."""
    branches = controls["branches"]
    assert len(branches) == 19
    for control in branches:
        produced = gev._invert(list(control["rounded_moments"]))
        assert "exception_refusal" not in produced, control["label"]
        for name in ("c", "loc", "scale"):
            assert _bits(produced[name]) == _bits(control["fitted"][name]), (
                f"{control['label']}: {name}"
            )


# Windows reproduces the retained float64 bit patterns. Linux source-qualified
# parity (t2609151346) found three one-ULP differences among 44 quantile
# controls, all inside the retained 1e-10 normalized tolerance. Keep the exact
# Windows assertion and check Linux against that fixed tolerance separately.
_WINDOWS_BIT_CONTROL = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows bit-pattern control; Linux uses the retained fixed tolerance",
)
_LINUX_TOLERANCE_CONTROL = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux fixed-tolerance control; Windows retains bit-exact controls",
)
_RETAINED_QUANTILE_TOLERANCE = 1e-10
_QUANTILE_MUTANTS = [
    pytest.param(lambda c, loc, s: (-c, loc, s), id="wrong_shape_sign"),
    pytest.param(lambda c, loc, s: (c, loc, -s), id="negated_scale"),
    pytest.param(lambda c, loc, s: (c, loc + 1.0, s), id="shifted_location"),
    pytest.param(lambda c, loc, s: (c, loc, s * 1.000001), id="perturbed_scale"),
]


@_WINDOWS_BIT_CONTROL
def test_quantile_controls_reproduce_the_retained_bit_patterns(controls):
    """All 44 retained quantile references, checked as float64 bit patterns."""
    quantiles = controls["quantiles"]
    assert len(quantiles) == 44
    for control in quantiles:
        normalized = control["normalized"]
        produced = gev._quantile(normalized["parameters"], normalized["p"])
        assert _bits(produced) == normalized["output_bits"], control["label"]
        assert produced == normalized["output"], control["label"]


@_WINDOWS_BIT_CONTROL
@pytest.mark.parametrize("mutate", _QUANTILE_MUTANTS)
def test_quantile_controls_discriminate_against_mutants(controls, mutate):
    """Each mutation must break at least one exact Windows reference."""
    failures = 0
    for control in controls["quantiles"]:
        normalized = control["normalized"]
        parameters = normalized["parameters"]
        c, loc, scale = mutate(parameters["c"], parameters["loc"], parameters["scale"])
        mutant = {"c": c, "loc": loc, "scale": scale}
        if _bits(gev._quantile(mutant, normalized["p"])) != normalized["output_bits"]:
            failures += 1
    assert failures > 0, "the mutation changed no control output"


@_LINUX_TOLERANCE_CONTROL
def test_linux_quantile_controls_meet_retained_tolerance(controls):
    """All Linux quantiles must meet the frozen readiness tolerance."""
    quantiles = controls["quantiles"]
    assert len(quantiles) == 44
    for control in quantiles:
        normalized = control["normalized"]
        reference = normalized["output"]
        produced = gev._quantile(normalized["parameters"], normalized["p"])
        tolerance = _RETAINED_QUANTILE_TOLERANCE * max(1.0, abs(reference))
        assert np.isfinite(produced) and abs(produced - reference) <= tolerance, (
            control["label"]
        )


@_LINUX_TOLERANCE_CONTROL
@pytest.mark.parametrize("mutate", _QUANTILE_MUTANTS)
def test_linux_quantile_controls_discriminate_against_mutants(controls, mutate):
    """Each mutation must breach a retained tolerance, beyond Linux libm noise."""
    failures = 0
    for control in controls["quantiles"]:
        normalized = control["normalized"]
        parameters = normalized["parameters"]
        c, loc, scale = mutate(parameters["c"], parameters["loc"], parameters["scale"])
        mutant = {"c": c, "loc": loc, "scale": scale}
        reference = normalized["output"]
        tolerance = _RETAINED_QUANTILE_TOLERANCE * max(1.0, abs(reference))
        produced = gev._quantile(mutant, normalized["p"])
        failures += not np.isfinite(produced) or abs(produced - reference) > tolerance
    assert failures > 0, "the mutation breached no retained tolerance"


def test_c_zero_branch_uses_the_gumbel_form():
    """The c == 0 branch is a separate formula, not a limit of the other one."""
    parameters = {"c": 0.0, "loc": 2.0, "scale": 3.0}
    for p in (0.9, 0.5, 0.01, 0.99):
        expected = 2.0 + 3.0 * -np.log(-np.log(np.float64(p)))
        assert _bits(gev._quantile(parameters, p)) == _bits(expected)


# --------------------------------------------------------------------------
# Refusals, faults and the typed record (D3)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample, reason",
    [
        ([1.0, 2.0, 3.0], "invalid_sample"),
        ([[1.0, 2.0], [3.0, 4.0]], "invalid_sample"),
        ([1.0, 2.0, 3.0, float("nan")], "invalid_sample"),
        ([1.0, 2.0, 3.0, float("inf")], "invalid_sample"),
        ([2.0, 2.0, 2.0, 2.0], "invalid_range"),
    ],
)
def test_domain_refusals_are_structured(sample, reason):
    result = gev.fit_case(sample, PROBABILITIES)
    assert result.status == "refused"
    assert result.refusal_reasons == (reason,)
    assert not result.accepted
    with pytest.raises(ValueError, match="no usable value"):
        result.value(0.9)


def test_constant_sample_reaches_the_adapter_and_refuses_structurally():
    """D1 removed the predecessor's np.ptp guard for exactly this."""
    result = gev.fit_case([7.0, 7.0, 7.0, 7.0, 7.0], PROBABILITIES)
    assert result.status == "refused"
    assert result.refusal_reasons == ("invalid_range",)
    assert result.input is not None, "the refusal must still identify its input"
    assert result.normalization is None, "normalization was never completed"
    assert result.record()["status"] == "refused"


def test_empty_probability_collection_refuses():
    result = gev.fit_case(SAMPLE, [])
    assert result.refusal_reasons == ("invalid_probability_collection",)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_out_of_range_probability_refuses_the_whole_fit(probability):
    result = gev.fit_case(SAMPLE, [0.9, probability])
    assert result.status == "refused"
    assert "quantile_conjunction" in result.refusal_reasons
    rows = {row.refusal_reason for row in result.quantiles}
    assert "invalid_probability" in rows
    # The valid sibling loses its acceptance with the conjunction.
    assert "shared_baseline_revocation" in rows
    assert all(not row.accepted for row in result.quantiles)


def test_acceptance_requires_every_requested_quantile():
    accepted = gev.fit_case(SAMPLE, PROBABILITIES)
    assert accepted.status == "accepted"
    assert all(row.accepted for row in accepted.quantiles)
    assert accepted.value(0.9) == accepted.quantiles[0].estimate


def test_numeric_conversion_errors_propagate_rather_than_refusing():
    """D2: conversion errors are faults; only post-conversion content refuses."""
    with pytest.raises((ValueError, TypeError)):
        gev.fit_case(["bad", 1.0, 2.0, 3.0], PROBABILITIES)


def test_source_guard_refuses_to_fit_unverified_source(monkeypatch):
    monkeypatch.setitem(gev.SOURCE_SHA256, "distr.py", "0" * 64)
    with pytest.raises(gev.EstimatorSourceMismatch, match="qualified source"):
        gev.fit_case(SAMPLE, PROBABILITIES)


def test_source_guard_checks_both_pinned_files(monkeypatch):
    monkeypatch.setitem(gev.SOURCE_SHA256, "__init__.py", "0" * 64)
    with pytest.raises(gev.EstimatorSourceMismatch, match="__init__.py"):
        gev.fit_case(SAMPLE, PROBABILITIES)


def test_actual_pinned_raising_statement_is_classified():
    """Exercise the REAL line-1307 statement, not a stand-in for it."""
    from lmoments3 import distr

    try:
        distr.gev._lmom_fit([0.0, 0.0, 0.0])
    except ValueError as error:
        classified = gev.exception_refusal(error)
        assert classified is not None, "the pinned statement was not recognised"
        assert classified["line"] == 1307
        assert classified["type"] == "ValueError"
        assert classified["message"] == "L-Moments Invalid"
        assert classified["raising_module"] == "lmoments3.distr"
        assert classified["raising_function"] == "GenextremeGen._lmom_fit"
        assert classified["file_sha256"] == gev.SOURCE_SHA256["distr.py"]
    else:
        pytest.fail("the actual pinned raising statement was not exercised")


def test_second_pinned_statement_is_classified_from_a_real_frame():
    """Synthetic classifier coverage for line 1342; no natural nonconvergence."""
    import types

    from lmoments3 import distr

    try:
        distr.gev._lmom_fit([0.0, 0.0, 0.0])
    except ValueError as error:
        trace = error.__traceback__
        while trace.tb_next:
            trace = trace.tb_next
        synthetic = Exception("Iteration has not converged")
        synthetic.__traceback__ = types.TracebackType(
            None, trace.tb_frame, trace.tb_lasti, 1342
        )
        assert gev.exception_refusal(synthetic)["line"] == 1342
        # The same frame at any other line is not allowlisted.
        wrong_line = Exception("Iteration has not converged")
        wrong_line.__traceback__ = types.TracebackType(
            None, trace.tb_frame, trace.tb_lasti, 1341
        )
        assert gev.exception_refusal(wrong_line) is None
    else:
        pytest.fail("the actual pinned raising statement was not exercised")


def test_allowlisted_exception_becomes_a_structured_refusal(monkeypatch):
    """The classified exception must surface as a refusal, not a fault."""
    from lmoments3 import distr

    def raise_pinned(*args, **kwargs):
        # Call the real pinned statement so the traceback frame is genuine.
        distr.gev._lmom_fit([0.0, 0.0, 0.0])

    monkeypatch.setattr(distr.gev, "lmom_fit", raise_pinned)
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    assert result.status == "refused"
    assert result.refusal_reasons == ("allowlisted_library_exception",)
    assert result.exception_refusal["line"] == 1307
    assert result.stages == ("input", "normalize", "moments", "parameters")
    assert result.normalized_parameters is None
    json.dumps(result.record(), allow_nan=False)


def test_allowlisted_refusal_also_retains_its_warnings(monkeypatch):
    """This path shares the in-block construction that dropped warnings.

    Asserted by execution rather than by inspection, because it was inspection
    alone that originally left it uncovered.
    """
    from lmoments3 import distr

    def pinned(**kwargs):
        warnings.warn("probe before the pinned statement", RuntimeWarning, stacklevel=1)
        distr.gev._lmom_fit([0, 0, 0])

    monkeypatch.setattr(distr.gev, "lmom_fit", pinned)
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    assert result.refusal_reasons == ("allowlisted_library_exception",)
    assert any(
        "probe before the pinned statement" in item["message"]
        for item in result.warnings
    )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("L-Moments Invalid"),
        Exception("Iteration has not converged"),
        RuntimeError("something else entirely"),
    ],
    ids=["right_message_wrong_frame", "right_message_wrong_frame_2", "arbitrary"],
)
def test_foreign_frames_with_identical_messages_are_faults(error):
    """An identical message raised anywhere else must not become a refusal."""
    try:
        raise error
    except Exception as raised:
        assert gev.exception_refusal(raised) is None


def test_exception_subclass_is_not_allowlisted():
    class Subclass(ValueError):
        pass

    try:
        raise Subclass("L-Moments Invalid")
    except Exception as raised:
        assert gev.exception_refusal(raised) is None


def test_unexpected_library_fault_propagates_with_warning_notes(monkeypatch):
    import lmoments3

    def faulty(*args, **kwargs):
        warnings.warn("overflow encountered", RuntimeWarning, stacklevel=1)
        raise RuntimeError("unexpected library failure")

    monkeypatch.setattr(lmoments3, "lmom_ratios", faulty)
    with pytest.raises(RuntimeError, match="unexpected library failure") as caught:
        gev.fit_case(SAMPLE, PROBABILITIES)
    notes = getattr(caught.value, "__notes__", [])
    assert any("overflow encountered" in note for note in notes)


def test_lmom_ratios_value_error_is_a_fault_not_a_refusal(monkeypatch):
    """D3: every lmom_ratios exception, ValueError included, is a fault."""
    import lmoments3

    def faulty(*args, **kwargs):
        raise ValueError("L-Moments Invalid")

    monkeypatch.setattr(lmoments3, "lmom_ratios", faulty)
    with pytest.raises(ValueError, match="L-Moments Invalid"):
        gev.fit_case(SAMPLE, PROBABILITIES)


def test_inversion_is_called_exactly_once_with_no_retry(monkeypatch):
    from lmoments3 import distr

    calls = []
    real = distr.gev.lmom_fit

    def counting(*args, **kwargs):
        calls.append(kwargs.get("lmom_ratios"))
        return real(*args, **kwargs)

    monkeypatch.setattr(distr.gev, "lmom_fit", counting)
    gev.fit_case(SAMPLE, PROBABILITIES)
    assert len(calls) == 1


def test_warnings_are_retained_on_an_accepted_fit(monkeypatch):
    import lmoments3

    real = lmoments3.lmom_ratios

    def warning_ratios(*args, **kwargs):
        warnings.warn(
            "overflow encountered in scalar power", RuntimeWarning, stacklevel=1
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(lmoments3, "lmom_ratios", warning_ratios)
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    assert result.status == "accepted", "a warning alone must never refuse"
    assert any("overflow" in item["message"] for item in result.warnings)


def test_overflow_warnings_survive_an_early_refusal(frozen):
    """Ported from the retained control this suite originally failed to carry.

    `test_readiness.py::test_overflow_warning_retention_discriminates` exists for
    exactly this regression, and its absence here let a real one ship: a refusal
    constructed inside the warning context recorded `warnings: []`, which is not
    a missing record but a false one -- D3 reserves emptiness for "none occurred"
    and null for "unavailable". The overflow that EXPLAINS the refusal is what
    was being erased, in the attempt logs a failed comparison is diagnosed from.
    """
    sample = [-1e308, -1e307, 1e307, 1e308]
    result = gev.fit_case(sample, [0.5])
    assert result.refusal_reasons == ("invalid_range",)
    assert result.warnings, "warnings were discarded on the early-refusal path"
    assert any("overflow" in item["message"] for item in result.warnings)
    assert [dict(item) for item in result.warnings] == frozen.fit_case(sample, [0.5])[
        "warnings"
    ]
    assert result.record()["warnings"] == [dict(item) for item in result.warnings]


@pytest.mark.parametrize(
    "sample, reason",
    [
        ([1.0, 2.0, 3.0, 4.0], None),
        ([5.0, 5.0, 5.0, 5.0], "invalid_range"),
    ],
    ids=["accepted", "invalid_range"],
)
def test_warning_retention_is_symmetric_across_outcomes(monkeypatch, sample, reason):
    """A warning must be retained whichever branch constructs the record.

    The original defect was asymmetric -- retained on the accepted and
    conjunction paths, dropped on every early refusal -- so a consumer could not
    tell "no warnings occurred" from "warnings were discarded".
    """
    import lmoments3

    real = lmoments3.lmom_ratios

    def warning_ratios(*args, **kwargs):
        warnings.warn("synthetic retention probe", RuntimeWarning, stacklevel=1)
        return real(*args, **kwargs)

    monkeypatch.setattr(lmoments3, "lmom_ratios", warning_ratios)
    result = gev.fit_case(sample, PROBABILITIES)
    assert tuple(result.refusal_reasons) == ((reason,) if reason else ())
    if reason == "invalid_range":
        # Refused before the moment stage, so the probe never fires; the point
        # is that the field is genuinely empty, not emptied.
        assert result.warnings == ()
    else:
        assert any(
            "synthetic retention probe" in item["message"] for item in result.warnings
        )


def test_diagnostics_never_veto_an_otherwise_valid_fit():
    """Support violations and endpoint rounding are diagnostic only."""
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    assert result.status == "accepted"
    assert result.normalized_diagnostics is not None
    assert result.physical_diagnostics is not None
    assert result.normalized_diagnostics["sample_outside_support"] >= 0


def test_record_uses_the_finite_number_dialect():
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    record = result.record()
    assert record["schema_version"] == "gev-fit/1"
    assert record["estimator_id"] == "gf15-lmoments-c/1"
    assert record["source"]["sha256"] == gev.SOURCE_SHA256
    assert record["input"]["count"] == len(SAMPLE)
    assert set(record["input"]) == {"count", "sample_sha256", "probabilities"}, (
        "the record identifies its input by digest, never by raw values"
    )
    # Summary counts only: no per-sample log densities are retained.
    assert set(record["normalized_diagnostics"]) == {
        "sample_outside_support",
        "nonfinite_log_density_count",
    }
    assert set(record["physical_diagnostics"]) == set(record["normalized_diagnostics"])
    assert all(
        isinstance(value, int) for value in record["normalized_diagnostics"].values()
    )
    # The whole record must be JSON-serializable with no nonfinite numbers.
    json.dumps(record, allow_nan=False)


def test_record_serializes_nonfinite_numbers_as_strings():
    assert gev._number(float("nan")) == "nan"
    assert gev._number(float("inf")) == "inf"
    assert gev._number(float("-inf")) == "-inf"
    assert gev._number(None) is None
    assert gev._number(0.0) == 0.0


def test_refused_record_is_serializable_and_names_its_stages():
    result = gev.fit_case([2.0, 2.0, 2.0, 2.0], PROBABILITIES)
    record = result.record()
    json.dumps(record, allow_nan=False)
    assert record["stages"] == ["input", "normalize"]
    assert record["refusal_reasons"] == ["invalid_range"]
    assert record["normalized_parameters"] is None
    assert record["normalized_diagnostics"] is None


def test_stages_stop_where_the_refusal_happened():
    """Checks after an early refusal must not be invented."""
    assert gev.fit_case([1.0, 2.0], PROBABILITIES).record()["stages"] == ["input"]
    accepted = gev.fit_case(SAMPLE, PROBABILITIES)
    assert accepted.stages == (
        "input",
        "normalize",
        "moments",
        "parameters",
        "map",
        "quantile",
        "diagnostics",
    )


def test_result_is_immutable():
    result = gev.fit_case(SAMPLE, PROBABILITIES)
    with pytest.raises(Exception):
        result.status = "accepted"
    with pytest.raises(Exception):
        result.quantiles[0].estimate = 1.0


def test_repeated_fits_are_deterministic():
    first = gev.fit_case(SAMPLE, PROBABILITIES)
    second = gev.fit_case(SAMPLE, PROBABILITIES)
    assert first.record() == second.record()


def test_module_imports_without_the_estimator_dependency():
    """D6: the static import in indicator_registry must not demand lmoments3."""
    import ast

    tree = ast.parse(
        (REPO / "blueearth_cst/experiment/gev_lmoments.py").read_text("utf-8")
    )
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "lmoments3" not in top_level
    assert "scipy" not in top_level
