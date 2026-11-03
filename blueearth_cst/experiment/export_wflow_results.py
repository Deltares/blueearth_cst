# -*- coding: utf-8 -*-
"""The publication formatter for indicator-set values.

This module held the predecessor WF3 reduction (``analyze_wflow_results``,
``analyze_response_runs``) until 2026-09-25 (t2609151037). Nothing reached it
after R12: rule 4.08 reduces through ``indicator_plan`` and ``reduce_bundle``, and
its return-level helper still fitted the retired MLE estimator, up to 3x apart
from the L-moment one on the same statistic. It was removed rather than kept as
a trap for whoever revived it.

What remains is the one function ``indicator_plan`` imports. It stays at this path
because ``indicator_plan.py`` is in the indicator-plan code inventory: moving the
import would change every indicator-set id for no change in output.
"""

import numpy as np
import pandas as pd

#: Significant digits kept in the written ``value`` column. Not decimal places —
#: the difference is what makes this safe; see ``_format_value``.
VALUE_SIGNIFICANT_DIGITS = 4


def _format_value(value: float) -> str:
    """Render one indicator value as PLAIN DECIMAL text, 4 significant digits.

    Two separate requirements, both about how this file reads *outside* the
    pipeline. The tables are a deliverable and an interchange surface, so the
    bytes are a contract rather than a display choice.

    **No scientific notation.** Low-flow values reach ~1e-5, so pandas' default
    repr puts ``6.3476255e-05`` in the file, which Excel does not open cleanly.
    ``np.format_float_positional`` is what removes the exponent. Note that the
    obvious ``float_format="%.4g"`` does NOT: it still emits ``6.348e-05``.

    **Four SIGNIFICANT digits, not four decimal places.** This distinction is
    the whole reason a cap is safe to reintroduce here. The pre-R11
    ``.round(2)`` / ``.round(4)`` were decimal-place rounding, which is
    scale-destroying — ``round(2)`` turns ``0.0007395697`` into ``0.0`` — and
    ``dev/scripts/check_baseline.py`` records that as the "accidental drift
    buffer" P1 removed, the reason the comparator moved to a tolerance instead
    of a sha256. A significant-digit cap is scale-invariant: worst-case relative
    error is 5e-4 — half a unit in the last kept digit, worst when the leading
    digit is 1 — against that comparator's own ``INDICATOR_RTOL`` of 1e-2, a 20x
    margin, so it cannot mask a difference the baseline gate would have caught.
    ``tests/test_export_wflow_results.py`` pins that claim to the tolerance
    constants and to the real reference table, so tightening either fails loudly
    here rather than silently invalidating the argument. Measured worst case on
    the current reference table is 4.6e-4.

    Missing values render as the empty field pandas would have written via
    ``na_rep``. Stated explicitly because ``.map()`` bypasses that path and
    would otherwise put the literal string ``nan`` in the file.
    """
    if pd.isna(value):
        return ""
    return np.format_float_positional(
        value,
        precision=VALUE_SIGNIFICANT_DIGITS,
        unique=False,
        fractional=False,
        trim="-",
    )
