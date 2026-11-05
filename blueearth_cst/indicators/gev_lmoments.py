"""Range-normalized sample L-moment GEV estimator (candidate C), typed.

Accepted design D2-D3. This is the production form of the frozen candidate
`dev/records/milestones/r12/implementation/evidence/gf15-lmoments-readiness/candidate_adapter.py`,
which is what the 8x qualification was run against: the pinned implementation
defines parity, so the arithmetic here is a faithful port and not a rewrite.

`fit_case(sample, probabilities)` is the whole interface. No indicator name, run id,
location, truth value, generating parameter, ratio, RNG or configuration enters
it -- the reducer owns that context and the estimator must not be able to see it.

What this module deliberately does NOT do: no fallback, alternate estimator,
retry, shrinkage, shape bound beyond ``c > -1``, sampling rule or fit interval.
A refused result carries no usable indicator value; it is never a blank or a
silently substituted number.

`lmoments3` and `scipy` are imported lazily, inside the call. Importing this
module -- which `indicator_registry` does statically, so the adapter source enters
the repository code inventory -- must not require the estimator dependency, or
reading a retained pre-C indicator set would start demanding it (D6).
"""

from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "gev-fit/1"
ESTIMATOR_ID = "gf15-lmoments-c/1"

#: D7's pinned dependency identity. There is no configurable bypass: a
#: same-version repack with different source is unqualified and must not fit.
DEPENDENCY_VERSION = "1.0.8"
SOURCE_SHA256 = {
    "__init__.py": "1d2c066a2ec4925bb838b7680d63ae8ae87c5234aa3c986d6ebe81500d75d6e2",
    "distr.py": "f8859d11f10c3206e3e5009c1b238a328c6c9215b1cc7436800160fa929bb2ee",
}

#: The two exact raising statements in the verified `_lmom_fit` that become
#: refusals. Anything else from that call -- a subclass, the same message from
#: another line, an arbitrary error -- is a fault and propagates.
_ALLOWED_EXCEPTIONS = {
    (ValueError, "L-Moments Invalid"): 1307,
    (Exception, "Iteration has not converged"): 1342,
}

#: Minimum sample size the estimator itself will accept. Operational screening
#: is a separate, stricter gate the reducer applies before calling.
MINIMUM_SAMPLE = 4


class EstimatorSourceMismatch(RuntimeError):
    """The installed estimator source is not the qualified one (D3, D7).

    Raised before any fit. This is a fault, never a refusal: a result produced by
    unverified source would be attributed to a benchmark that did not assess it.
    """


@dataclass(frozen=True)
class Quantile:
    """One requested probability and what the fit produced for it."""

    p: float | None
    normalized: float | None = None
    estimate: float | None = None
    diagnostic_valid: bool = False
    accepted: bool = False
    refusal_reason: str | None = None


@dataclass(frozen=True)
class FitResult:
    """Immutable accepted/refused record of one scalar fit.

    `status` is the discriminant. A refused result provides no usable value, and
    acceptance requires every requested quantile to be valid -- there is no
    partial acceptance in which some probabilities are usable and others are not.
    """

    status: str
    refusal_reasons: tuple[str, ...] = ()
    exception_refusal: dict[str, Any] | None = None
    source: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    moments: dict[str, Any] | None = None
    normalized_parameters: dict[str, float] | None = None
    physical_parameters: dict[str, float] | None = None
    quantiles: tuple[Quantile, ...] = ()
    warnings: tuple[dict[str, str], ...] = ()
    normalized_diagnostics: dict[str, int] | None = None
    physical_diagnostics: dict[str, int] | None = None
    stages: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def value(self, probability: float) -> float:
        """Return the accepted physical estimate for one requested probability."""
        if not self.accepted:
            raise ValueError("a refused fit has no usable value")
        for row in self.quantiles:
            if row.p == probability:
                return row.estimate
        raise KeyError(f"probability {probability!r} was not requested of this fit")

    def record(self) -> dict[str, Any]:
        """Serialize in D3's finite-number dialect, for evidence and refusal logs.

        Nonfinite numbers become the strings `nan`, `inf`, `-inf`; `null` means a
        stage was never reached, and never zero. Summary moments, parameters and
        counts are retained -- raw samples and per-sample log densities are not.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "estimator_id": ESTIMATOR_ID,
            "source": self.source,
            "status": self.status,
            "refusal_reasons": list(self.refusal_reasons),
            "exception_refusal": self.exception_refusal,
            "stages": list(self.stages),
            "input": self.input,
            "normalization": self.normalization,
            "moments": self.moments,
            "normalized_parameters": self.normalized_parameters,
            "physical_parameters": self.physical_parameters,
            "quantiles": [
                {
                    "p": _number(row.p),
                    "normalized": _number(row.normalized),
                    "estimate": _number(row.estimate),
                    "diagnostic_valid": bool(row.diagnostic_valid),
                    "accepted": bool(row.accepted),
                    "refusal_reason": row.refusal_reason,
                }
                for row in self.quantiles
            ],
            "warnings": [dict(item) for item in self.warnings],
            "normalized_diagnostics": self.normalized_diagnostics,
            "physical_diagnostics": self.physical_diagnostics,
        }


def _number(value: float | None) -> float | str | None:
    """Keep nonfinite numbers legible without coercing them to a finite value."""
    if value is None:
        return None
    value = float(value)
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return value


@contextmanager
def _record_warnings(collected: list[dict[str, str]]):
    """Retain captured warnings on success, refusal and fault paths alike.

    Recording is EAGER, on each warning as it is raised, rather than swept up on
    exit. A refusal is constructed inside this block, so collecting only in a
    `finally` hands that refusal an empty list and records it as "no warnings"
    -- affirmatively false, and exactly where an overflow explaining the refusal
    would have been. `catch_warnings` restores the previous hook on exit.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")

        def record(message, category, filename, lineno, file=None, line=None):
            collected.append(
                {
                    "type": type(message).__name__
                    if isinstance(message, Warning)
                    else category.__name__,
                    "message": str(message),
                }
            )

        warnings.showwarning = record
        yield


def _verify_source() -> dict[str, Any]:
    """Check the installed estimator against D7 before it can be invoked."""
    import lmoments3
    from lmoments3 import distr

    observed = {}
    for name, module in (("__init__.py", lmoments3), ("distr.py", distr)):
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[name] = digest
        if digest != SOURCE_SHA256[name]:
            raise EstimatorSourceMismatch(
                f"installed lmoments3 {name} has sha256 {digest}, "
                f"but the qualified source is {SOURCE_SHA256[name]}"
            )
    return {
        "package": "lmoments3",
        "version": DEPENDENCY_VERSION,
        "sha256": observed,
    }


def observed_source() -> dict[str, str]:
    """Return the installed estimator's verified source digests.

    Verification is not optional here either: callers binding this into an
    identity must be recording what D7 accepts, not merely what is installed.
    """
    return dict(_verify_source()["sha256"])


def exception_refusal(error: Exception) -> dict[str, Any] | None:
    """Classify only the two exact pinned raising statements; otherwise decline.

    Every element is checked -- resolved file, code-object identity with the
    loaded method, module, function name, line, exception type, message and the
    D7 digest -- so a subclass, or an identical message raised anywhere else, is
    a fault rather than a refusal.
    """
    from lmoments3 import distr

    line = _ALLOWED_EXCEPTIONS.get((type(error), str(error)))
    if line is None or error.__traceback__ is None:
        return None
    trace = error.__traceback__
    while trace.tb_next is not None:
        trace = trace.tb_next
    frame = trace.tb_frame
    source = Path(frame.f_code.co_filename).resolve()
    installed = Path(distr.__file__).resolve()
    if (
        source != installed
        or frame.f_code is not distr.GenextremeGen._lmom_fit.__code__
        or frame.f_code.co_name != "_lmom_fit"
        or frame.f_globals.get("__name__") != "lmoments3.distr"
        or trace.tb_lineno != line
        or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256["distr.py"]
    ):
        return None
    return {
        "type": type(error).__name__,
        "class_module": type(error).__module__,
        "message": str(error),
        "raising_module": "lmoments3.distr",
        "raising_function": "GenextremeGen._lmom_fit",
        "line": line,
        "file_sha256": SOURCE_SHA256["distr.py"],
    }


def _invert(moments: list[float]) -> dict[str, Any]:
    """Call the pinned inversion exactly once; unknown errors propagate."""
    from lmoments3 import distr

    try:
        parameters = distr.gev.lmom_fit(lmom_ratios=moments)
    except Exception as error:
        refusal = exception_refusal(error)
        if refusal is None:
            raise
        return {"exception_refusal": refusal}
    return {name: float(parameters[name]) for name in ("c", "loc", "scale")}


def _quantile(parameters: dict[str, float], probability: float) -> float:
    """The specified float64 inverse, in SciPy's shape sign convention."""
    c, location, scale = (
        np.float64(parameters[name]) for name in ("c", "loc", "scale")
    )
    p = np.float64(probability)
    value = np.log(-np.log(p))
    standard = -value if c == 0 else -np.expm1(c * value) / c
    return float(location + scale * standard)


def _diagnostics(values, parameters: dict[str, float]) -> dict[str, int]:
    """Count support violations and nonfinite densities. Diagnostic only.

    These never veto acceptance -- including physical endpoint rounding, which
    is why the counts are reported rather than acted on. An exception raised
    here is a fault, so nothing is caught.
    """
    from scipy.stats import genextreme

    logs = genextreme.logpdf(values, **parameters)
    lower, upper = genextreme.support(**parameters)
    return {
        "sample_outside_support": int(
            np.count_nonzero((values < lower) | (values > upper))
        ),
        "nonfinite_log_density_count": int(np.count_nonzero(~np.isfinite(logs))),
    }


def fit_case(sample, probabilities) -> FitResult:
    """Fit the supplied finite float64 sample at the requested probabilities.

    Numeric conversion errors propagate; after conversion, invalid shape or
    content is a typed refusal. The estimator source is verified before any
    library call, and the inversion is invoked exactly once with no retry.
    """
    x = np.asarray(sample, dtype=np.float64)
    requested = np.asarray(probabilities, dtype=np.float64)
    source = _verify_source()

    collected: list[dict[str, str]] = []
    stages: list[str] = []
    reasons: list[str] = []

    def refuse(reason: str, **fields) -> FitResult:
        reasons.append(reason)
        return FitResult(
            status="refused",
            refusal_reasons=tuple(reasons),
            source=source,
            warnings=tuple(collected),
            stages=tuple(stages),
            **fields,
        )

    stages.append("input")
    if x.ndim != 1 or x.size < MINIMUM_SAMPLE or not np.isfinite(x).all():
        return refuse("invalid_sample")
    if requested.ndim != 1 or not requested.size:
        return refuse("invalid_probability_collection")
    record_input = {
        "count": int(x.size),
        "sample_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "probabilities": requested.tolist(),
    }

    try:
        with _record_warnings(collected):
            stages.append("normalize")
            a = np.min(x)
            s = np.max(x) - a
            if not np.isfinite(s) or s <= 0:
                return refuse("invalid_range", input=record_input)
            y = (x - a) / s
            normalization = {"a": float(a), "s": float(s)}
            if not np.isfinite(y).all():
                return refuse(
                    "nonfinite_normalization",
                    input=record_input,
                    normalization=normalization,
                )

            stages.append("moments")
            # No lmom_ratios exception is an allowed statistical refusal.
            import lmoments3

            moments = list(map(float, lmoments3.lmom_ratios(y, nmom=3)))
            l1, l2, t3 = moments
            record_moments = {
                "normalized": [l1, l2, t3 * l2],
                "physical": [float(a + s * l1), float(s * l2), float(s * t3 * l2)],
                "t3": t3,
            }
            partial = {
                "input": record_input,
                "normalization": normalization,
                "moments": record_moments,
            }
            if not np.isfinite(moments).all() or l2 <= 0 or abs(t3) >= 1:
                return refuse("invalid_moments", **partial)

            stages.append("parameters")
            inverted = _invert(moments)
            if "exception_refusal" in inverted:
                reasons.append("allowlisted_library_exception")
                return FitResult(
                    status="refused",
                    refusal_reasons=tuple(reasons),
                    exception_refusal=inverted["exception_refusal"],
                    source=source,
                    warnings=tuple(collected),
                    stages=tuple(stages),
                    **partial,
                )
            partial["normalized_parameters"] = inverted
            if (
                not np.isfinite(list(inverted.values())).all()
                or inverted["c"] <= -1
                or inverted["scale"] <= 0
            ):
                return refuse("invalid_parameters", **partial)

            stages.append("map")
            physical = {
                "c": inverted["c"],
                "loc": float(a + s * inverted["loc"]),
                "scale": float(s * inverted["scale"]),
            }
            partial["physical_parameters"] = physical
            if not np.isfinite(list(physical.values())).all() or physical["scale"] <= 0:
                return refuse("invalid_physical_parameters", **partial)

            stages.append("quantile")
            rows: list[Quantile] = []
            for probability in requested:
                p = float(probability)
                if not np.isfinite(p) or not 0 < p < 1:
                    rows.append(
                        Quantile(
                            p=p if np.isfinite(p) else None,
                            refusal_reason="invalid_probability",
                        )
                    )
                    continue
                normalized = _quantile(inverted, p)
                estimate = float(a + s * normalized)
                valid = bool(np.isfinite([normalized, estimate]).all())
                rows.append(
                    Quantile(
                        p=p,
                        normalized=normalized,
                        estimate=estimate,
                        diagnostic_valid=valid,
                        refusal_reason=None if valid else "nonfinite_quantile",
                    )
                )

            accepted = all(row.diagnostic_valid for row in rows)
            rows = [
                Quantile(
                    p=row.p,
                    normalized=row.normalized,
                    estimate=row.estimate,
                    diagnostic_valid=row.diagnostic_valid,
                    accepted=accepted,
                    refusal_reason=(
                        "shared_baseline_revocation"
                        if not accepted and row.diagnostic_valid
                        else row.refusal_reason
                    ),
                )
                for row in rows
            ]
            if not accepted:
                reasons.append("quantile_conjunction")

            stages.append("diagnostics")
            normalized_diagnostics = _diagnostics(y, inverted)
            physical_diagnostics = _diagnostics(x, physical)
    except EstimatorSourceMismatch:
        raise
    except Exception as error:
        # A fault keeps its traceback. Warnings ride along as notes so a refusal
        # log is never the only place they survive.
        for item in collected:
            error.add_note(f"warning during fit: {item['type']}: {item['message']}")
        raise

    return FitResult(
        status="accepted" if accepted else "refused",
        refusal_reasons=tuple(reasons),
        source=source,
        input=record_input,
        normalization=normalization,
        moments=record_moments,
        normalized_parameters=inverted,
        physical_parameters=physical,
        quantiles=tuple(rows),
        warnings=tuple(collected),
        normalized_diagnostics=normalized_diagnostics,
        physical_diagnostics=physical_diagnostics,
        stages=tuple(stages),
    )
