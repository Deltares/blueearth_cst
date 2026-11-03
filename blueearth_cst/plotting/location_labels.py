"""Location labels shared by climate and spatial map renderers."""

import cmath
import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import numpy as np

# pandas' missing-value sentinels, recognised without importing pandas.
_PANDAS_MISSING = {"NAType", "NaTType"}


def _missing(value: Any) -> bool:
    """None, NaN, NaT or a pandas NA scalar, as ``pandas.isna`` treats scalars."""
    if value is None or type(value).__name__ in _PANDAS_MISSING:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, complex):
        return cmath.isnan(value)
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, (np.floating, np.complexfloating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


#: Characters a long label may break after; a space is dropped at the break.
_BREAKS = (" ", "_", "-")


def wrap_label(text: str, width: int) -> str:
    """Break a label longer than ``width`` characters onto two lines.

    Breaks at the separator closest to the middle, keeping ``_``/``-`` at the end
    of the first line so the name stays recoverable. A label without a separator
    stays on one line rather than being split mid-word.
    """
    if len(text) <= width:
        return text
    middle = len(text) / 2
    positions = [i for i, char in enumerate(text[1:-1], 1) if char in _BREAKS]
    if not positions:
        return text
    cut = min(positions, key=lambda i: abs(i + 0.5 - middle))
    first = text[:cut] if text[cut] == " " else text[: cut + 1]
    return f"{first}\n{text[cut + 1 :]}"


def location_label(properties: Mapping[str, Any], column="station_name") -> str:
    """Prefer the selected label, falling back to the Wflow ID for station names."""
    columns = (
        (column, "wflow_id", "location_code") if column == "station_name" else (column,)
    )
    for key in columns:
        value = properties.get(key)
        if not _missing(value):
            text = str(value).strip()
            if text:
                return text
    return ""
