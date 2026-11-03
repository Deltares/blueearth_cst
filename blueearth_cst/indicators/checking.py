"""Independent numerical primitives used only by set-owned checkers."""

import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.indicators.contract import EvaluationContext
from blueearth_cst.indicators.loader import load, local_declaration
from blueearth_cst.indicators.validation import validate_result


def member_expectation(indicator, series, context, scalar, *, group=False) -> pd.Series:
    """Check each member separately, then explicitly propagate undefined means."""
    runs = (
        context.expected_run_ids if group else tuple({item.run_id for item in series})
    )
    by_run = []
    for run in runs:
        members = [item for item in series if item.run_id == run]
        by_run.append(
            {
                item.location_id: scalar(pd.Series(item.values, index=item.time))
                for item in members
            }
        )
    if not by_run or any(set(row) != set(by_run[0]) for row in by_run):
        raise ValueError("check requires matching nonempty member locations")
    return pd.Series(
        {
            location: float(np.mean([row[location] for row in by_run]))
            for location in by_run[0]
        }
    )


def return_level_expected(indicator, series, context) -> pd.Series:
    """Recompute normalized L-moments and SciPy inverse independently of production."""
    import lmoments3
    from lmoments3 import distr
    from scipy.stats import genextreme

    probability = (
        1 - 1 / indicator.return_period
        if indicator.statistic == "return_level_max"
        else 1 / indicator.return_period
    )
    declared = (
        (context.validation or {}).get("tested_domain", {}).get("probabilities", [])
    )
    if probability not in declared:
        raise ValueError(
            f"probability {probability} outside qualified domain {declared}"
        )
    result = {}
    for location in sorted({item.location_id for item in series}):
        blocks = []
        for item in series:
            if item.location_id != location:
                continue
            daily = pd.Series(item.values, index=item.time)
            if indicator.statistic != "return_level_max":
                daily = daily.rolling(7).mean()
            annual = daily.resample(context.anchor)
            values = (
                annual.max()
                if indicator.statistic == "return_level_max"
                else annual.min()
            ).to_numpy()
            blocks.extend(values[np.isfinite(values)])
        sample = np.array(blocks)
        if len(sample) < max(
            math.ceil(indicator.blocks_per_return_period * indicator.return_period), 10
        ):
            raise ValueError("check has insufficient usable blocks")
        minimum, width = sample.min(), np.ptp(sample)
        ratios = lmoments3.lmom_ratios((sample - minimum) / width, nmom=3)
        fitted = distr.gev.lmom_fit(lmom_ratios=ratios)
        result[location] = float(
            minimum + width * genextreme.ppf(probability, **fitted)
        )
    return pd.Series(result)


def check(
    indicator,
    series,
    result,
    *,
    anchor,
    reference=None,
    expected_run_ids=None,
    validation=None,
    group=False,
) -> dict:
    """Independently recheck through the selected set, recording current checker identity."""
    indicator = local_declaration(indicator)
    module = importlib.import_module(
        f"{load(indicator.calculation_set).__package__}.checks"
    )
    context = EvaluationContext(
        anchor,
        reference,
        tuple(expected_run_ids or sorted({item.run_id for item in series})),
        validation,
    )
    try:
        validate_result(result)
        expected = module.expected(indicator, series, context, group=group)
        actual = result.values.reindex(expected.index).to_numpy(dtype=float)
        if set(result.values.index) != set(expected.index) or not np.allclose(
            actual,
            expected.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"numerical mismatch: expected {expected.to_dict()}, observed {result.values.to_dict()}"
            )
        if hasattr(module, "check_result"):
            module.check_result(indicator, series, result, context, group=group)
    except Exception as error:
        error.args = (
            f"{indicator.calculation_set}/{indicator.name}/location={list(result.values.index)}: {error}",
            *error.args[1:],
        )
        raise
    return {
        "status": "passed",
        "numerical_checks": len(expected),
        "checker_sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest(),
    }
