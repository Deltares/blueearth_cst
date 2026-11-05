"""The indicator-set value formatter (`export_wflow_results._format_value`).

The predecessor reduction these tests once covered was removed (t2609151037);
its perturbation-axis cases live in `tests/test_surface_axes.py`.
"""

import pandas as pd

from blueearth_cst.experiment.export_wflow_results import (
    VALUE_SIGNIFICANT_DIGITS,
    _format_value,
)


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
    # Preserve the recorded reference bytes; only translate its legacy header.
    ref = ref.rename(columns={"metric": "indicator"})
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
