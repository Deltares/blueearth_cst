"""Range-normalized unbiased sample L-moment GEV; isolated GF15 candidate.

Only fit_case(sample, probabilities) is the evaluator-facing interface.
No truth labels, RNG, retries, alternate estimators or empirical fallback.
"""

import hashlib
import warnings
from contextlib import contextmanager
from pathlib import Path

import lmoments3
import numpy as np
from lmoments3 import distr
from scipy.stats import genextreme

DISTR_SHA256 = "f8859d11f10c3206e3e5009c1b238a328c6c9215b1cc7436800160fa929bb2ee"


@contextmanager
def _record_warnings(record: dict):
    """Retain captured warnings on success, early refusal and fault paths alike."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            record["warnings"] = [
                {"type": type(item.message).__name__, "message": str(item.message)}
                for item in captured
            ]


def exception_refusal(error: Exception) -> dict | None:
    """Classify only the two exact pinned raising statements; otherwise decline."""
    allowed = {
        (ValueError, "L-Moments Invalid"): 1307,
        (Exception, "Iteration has not converged"): 1342,
    }
    line = allowed.get((type(error), str(error)))
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
        or hashlib.sha256(source.read_bytes()).hexdigest() != DISTR_SHA256
    ):
        return None
    return {
        "type": type(error).__name__,
        "class_module": type(error).__module__,
        "message": str(error),
        "raising_module": "lmoments3.distr",
        "raising_function": "GenextremeGen._lmom_fit",
        "line": line,
        "file_sha256": DISTR_SHA256,
    }


def _parameters(moments: list[float]) -> dict:
    """Call exactly the pinned inversion once; unknown errors propagate."""
    if hashlib.sha256(Path(distr.__file__).read_bytes()).hexdigest() != DISTR_SHA256:
        raise RuntimeError("installed estimator source differs from pinned wheel")
    try:
        parameters = distr.gev.lmom_fit(lmom_ratios=moments)
    except Exception as error:
        refusal = exception_refusal(error)
        if refusal is None:
            raise
        return {"exception_refusal": refusal}
    return {name: float(parameters[name]) for name in ("c", "loc", "scale")}


def _quantile(parameters: dict, probability: float) -> float:
    """Compute the specified NumPy float64 inverse, with SciPy's shape sign."""
    c, location, scale = (
        np.float64(parameters[name]) for name in ("c", "loc", "scale")
    )
    p = np.float64(probability)
    value = np.log(-np.log(p))
    standard = -value if c == 0 else -np.expm1(c * value) / c
    return float(location + scale * standard)


def _refuse(record: dict, reason: str) -> dict:
    """Keep explicit domain refusals separate from exception-derived refusals."""
    record["refusal_reasons"].append(reason)
    record["accepted"] = False
    return record


def fit_case(sample, probabilities) -> dict:
    """Fit the supplied finite float64 sample and requested probabilities only."""
    x = np.asarray(sample, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    record = {
        "accepted": False,
        "refusal_reasons": [],
        "warnings": [],
        "quantiles": [],
        "source_sha256": DISTR_SHA256,
    }
    if x.ndim != 1 or x.size < 4 or not np.isfinite(x).all():
        return _refuse(record, "invalid_sample")
    if probabilities.ndim != 1 or not probabilities.size:
        return _refuse(record, "invalid_probability_collection")
    record["input"] = {
        "count": int(x.size),
        "sample_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "probabilities": probabilities.tolist(),
    }
    with _record_warnings(record):
        a = np.min(x)
        s = np.max(x) - a
        if not np.isfinite(s) or s <= 0:
            return _refuse(record, "invalid_range")
        y = (x - a) / s
        record["normalization"] = {"a": float(a), "s": float(s), "y": y.tolist()}
        if not np.isfinite(y).all():
            return _refuse(record, "nonfinite_normalization")
        # No lmom_ratios exception is an allowed statistical refusal.
        moments = list(map(float, lmoments3.lmom_ratios(y, nmom=3)))
        l1, l2, t3 = moments
        record["moments"] = {
            "normalized": [l1, l2, t3 * l2],
            "physical": [float(a + s * l1), float(s * l2), float(s * t3 * l2)],
            "t3": t3,
        }
        if not np.isfinite(moments).all() or l2 <= 0 or abs(t3) >= 1:
            return _refuse(record, "invalid_moments")
        parameters = _parameters(moments)
        if "exception_refusal" in parameters:
            record.update(parameters)
            return _refuse(record, "allowlisted_library_exception")
        record["normalized_parameters"] = parameters
        if (
            not np.isfinite(list(parameters.values())).all()
            or parameters["c"] <= -1
            or parameters["scale"] <= 0
        ):
            return _refuse(record, "invalid_parameters")
        physical = {
            "c": parameters["c"],
            "loc": float(a + s * parameters["loc"]),
            "scale": float(s * parameters["scale"]),
        }
        record["physical_parameters"] = physical
        if not np.isfinite(list(physical.values())).all() or physical["scale"] <= 0:
            return _refuse(record, "invalid_physical_parameters")
        for probability in probabilities:
            p = float(probability)
            row = {"p": p if np.isfinite(p) else None, "diagnostic_valid": False}
            if not np.isfinite(p) or not 0 < p < 1:
                row["refusal_reason"] = "invalid_probability"
            else:
                normalized = _quantile(parameters, p)
                quantile = float(a + s * normalized)
                row.update(normalized=normalized, estimate=quantile)
                row["diagnostic_valid"] = bool(
                    np.isfinite([normalized, quantile]).all()
                )
                if not row["diagnostic_valid"]:
                    row["refusal_reason"] = "nonfinite_quantile"
            record["quantiles"].append(row)
        record["accepted"] = all(row["diagnostic_valid"] for row in record["quantiles"])
        for row in record["quantiles"]:
            row["accepted"] = record["accepted"]
            if not record["accepted"] and row["diagnostic_valid"]:
                row["refusal_reason"] = "shared_baseline_revocation"
        if not record["accepted"]:
            record["refusal_reasons"].append("quantile_conjunction")
        for coordinate, values, params in (
            ("normalized", y, parameters),
            ("physical", x, physical),
        ):
            logs = genextreme.logpdf(values, **params)
            lower, upper = genextreme.support(**params)
            record[f"{coordinate}_diagnostics"] = {
                "sample_outside_support": int(
                    np.count_nonzero((values < lower) | (values > upper))
                ),
                "nonfinite_log_density_count": int(
                    np.count_nonzero(~np.isfinite(logs))
                ),
                "log_density": [float(v) if np.isfinite(v) else str(v) for v in logs],
            }
    return record
