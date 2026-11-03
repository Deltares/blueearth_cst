"""Generic closed-parameter and scalar diagnostic validation."""

from collections.abc import Mapping
from numbers import Number

import numpy as np
import pandas as pd

from blueearth_cst.indicators.contract import IndicatorResult


def validate_parameters(function, parameters, specifications: Mapping) -> dict:
    """Validate a callable's one local parameter declaration, without defaults."""
    if function not in specifications:
        raise ValueError(
            f"unknown calculation {getattr(function, '__name__', function)!r}"
        )
    rules = specifications[function]
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    if set(parameters) != set(rules):
        raise ValueError(
            f"invalid parameter fields {sorted(parameters)}; required keys: {sorted(rules)}"
        )
    for key, rule in rules.items():
        value = parameters[key]
        if isinstance(value, bool):
            raise ValueError(f"{key}: booleans are not accepted")
        if isinstance(rule, (tuple, list, set, frozenset)):
            if value not in rule:
                raise ValueError(f"{key}: got {value!r}; accepted values: {rule!r}")
        elif not rule(value):
            raise ValueError(f"{key}: got {value!r}; {rule.__doc__ or 'invalid value'}")
    return dict(sorted(parameters.items()))


def validate_result(result: IndicatorResult) -> None:
    """Reject ambiguous result shapes and nested diagnostic cells."""
    if not isinstance(result, IndicatorResult) or not isinstance(
        result.values, pd.Series
    ):
        raise ValueError(
            "calculation must return IndicatorResult with location-labelled values"
        )
    if (
        not result.values.index.is_unique
        or np.isinf(result.values.to_numpy(dtype=float)).any()
    ):
        raise ValueError("result values require unique locations and no infinity")
    names = set()
    for table in result.diagnostics:
        if (
            not table.name
            or table.name in names
            or not isinstance(table.data, pd.DataFrame)
        ):
            raise ValueError(
                "diagnostic tables require unique nonempty names and DataFrames"
            )
        names.add(table.name)
        if not table.data.columns.is_unique or any(
            not isinstance(c, str) for c in table.data.columns
        ):
            raise ValueError(f"diagnostic {table.name}: columns must be unique strings")
        for column in table.data:
            for cell in table.data[column]:
                if isinstance(cell, complex) or (
                    cell is not None
                    and not isinstance(cell, (str, Number, bool, np.bool_))
                ):
                    raise ValueError(
                        f"diagnostic {table.name}/{column}: scalar cells required; got {type(cell).__name__}"
                    )
                if (
                    isinstance(cell, Number)
                    and not isinstance(cell, (complex, bool))
                    and np.isinf(cell)
                ):
                    raise ValueError(
                        f"diagnostic {table.name}/{column}: infinity refused"
                    )
