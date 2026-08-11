"""Resolve the names hydromt_wflow gave the user's gauges.

``output_locations`` is a PATH in our config, but by the time the model exists
the gauges live under names hydromt_wflow chose, and those names are not the
filename. ``setup_gauges`` normalizes the basename
(``hydromt_wflow/wflow_base.py``)::

    basename = os.path.basename(gauges_fn).split(".")[0].replace("_", "-")

so ``output_locations.csv`` becomes ``output-locations`` in every derived name:
the staticgeoms layer ``gauges_output-locations``, the wflow TOML's ``map``, and
the parsed output columns ``Q_gauges_output-locations`` /
``P_gauges_output-locations``.

Three readers derived those names from the FILENAME instead, missed the
underscore-to-hyphen step, and — because each lookup was a membership test used
as a guard (``if name in geoms:``) — dropped the user's gauges in SILENCE.
Observed 2026-08-01 on a real basin: gauges absent from ``basin_area.png``, no
gauge hydrographs, no signature plots, and an EMPTY ``performance_metrics.csv``,
while ``output.csv`` carried all four stations correctly because wflow reads the
TOML rather than guessing.

This module is the single place that answers "what are the gauges called?", and
it answers by ASKING THE MODEL rather than by re-deriving the name. Mirroring
``replace("_", "-")`` in three places would fix today's symptom and re-break the
day ``setup_gauges`` is called with an explicit ``basename=`` (it accepts one)
or upstream changes the rule. Discovery is indifferent to both.

The second half of the lesson is the silence, not the name. A configured gauges
file whose layer cannot be found is now a loud WARNING naming what was looked
for and what exists — never a skip. It is a warning rather than an error
because absence is legitimately possible: ``setup_gauges`` returns early, having
logged "Skipping method, as no data has been found", when no gauge falls inside
the model domain.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional

#: Staticgeoms layer for the model's own outlets. Never a user gauge layer, and
#: excluded from discovery so a config WITHOUT output_locations cannot silently
#: adopt the outlets as if the user had supplied them.
OUTLETS_LAYER = "outlets"

#: Prefix hydromt_wflow gives every gauge-derived staticgeoms layer.
GAUGES_PREFIX = "gauges_"

#: Lowest ``wflow_id`` a user gauge should carry
#: (``config/templates/README.md``). These ids become wflow output
#: columns (``Q_101``) and burned-in values in the derived
#: ``subcatchment_<name>`` map, sharing a namespace with the model's own outlet
#: subcatchment ids (large, from the hydrography) and with the positional
#: ``wflow_1``/``wflow_2`` labels the evaluation figures generate for outlets.
#: A small id makes ``Q_1`` ambiguous with the first positional outlet on sight.
#:
#: A CONVENTION, not a constraint: lower ids work, so an existing dataset keeps
#: running. Only the silence was the problem — a user who renumbers the template
#: and not their data has no way to notice.
#:
#: **MOOT since ADR 0003 §12, and deliberately not re-tuned.** Generated ids are
#: now `basin_id*1000 + local_subbasin_number*10 + m`, so the smallest possible
#: one is 1010 — an order of magnitude above this floor, which no generated id
#: can trip. A user-PINNED id below the floor dies earlier and louder at
#: `assign_location_ids`' mismatch check, which compares the pinned value
#: against the resolved hierarchy by name. Raising the threshold to match the
#: new scheme would look like maintenance and buy nothing; the advisory is kept
#: only because a project carrying pre-§12 ids can still reach it.
MIN_GAUGE_ID = 100


def warn_if_low_gauge_ids(ids, source) -> list:
    """Warn about gauge ids below :data:`MIN_GAUGE_ID`; return the offenders.

    Parameters
    ----------
    ids : iterable
        The ``wflow_id`` values. Non-integer entries are ignored rather than
        raising — this is an advisory read of user data, not a schema check.
    source : str | Path
        The file they came from, named in the warning so it is actionable.
    """
    low = []
    for value in ids:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number < MIN_GAUGE_ID:
            low.append(number)
    if low:
        warnings.warn(
            f"{source} carries wflow_id values below {MIN_GAUGE_ID}: "
            f"{sorted(low)}. The convention is to start at {MIN_GAUGE_ID} "
            f"(100, 101, 102, ...) so a gauge id cannot be read as one of the "
            f"positional wflow_N outlet labels — a column named Q_1 is "
            f"ambiguous, Q_101 is not. Nothing rejects these ids; renumbering "
            f"means changing output_locations AND the timeseries column "
            f"headers together, then rebuilding from rule 1.05, because "
            f"setup_gauges writes the ids into the model.",
            stacklevel=2,
        )
    return low


def is_unset(gauges_fn) -> bool:
    """Is ``output_locations`` "not provided"?

    Two spellings, both in the wild: YAML ``null`` (Python ``None``) and the
    legacy unquoted ``None`` sentinel, which ``yaml.safe_load`` parses to the
    STRING ``"None"``. R07 O-08 is what happens when a reader knows only the
    first: it derives ``gauges_None`` and drops the gauges without saying so.
    """
    return gauges_fn is None or str(gauges_fn) == "None"


def hydromt_basename(gauges_fn) -> Optional[str]:
    """The basename hydromt_wflow WOULD derive, or None when unset.

    Kept as the first candidate during discovery so the common case resolves
    exactly, and so a model carrying several gauge layers picks the configured
    one rather than whichever sorts first. Mirrors ``wflow_base.py``'s rule
    verbatim, including the underscore-to-hyphen step.
    """
    if is_unset(gauges_fn):
        return None
    return os.path.basename(str(gauges_fn)).split(".")[0].replace("_", "-")


def _resolve(candidates: Iterable[str], gauges_fn, kind: str, prefix: str) -> Optional[str]:
    """Shared resolution: exact match first, then sole-candidate discovery."""
    if is_unset(gauges_fn):
        return None
    available = [name for name in candidates if name.startswith(prefix)]
    expected = f"{prefix}{hydromt_basename(gauges_fn)}"
    if expected in available:
        return expected
    if len(available) == 1:
        return available[0]
    warnings.warn(
        f"output_locations is set to {gauges_fn!r} but its {kind} could not be "
        f"resolved: looked for {expected!r}, found "
        f"{available or 'no gauge entries at all'}. The gauges will be missing "
        f"from this output. If no gauge falls inside the model domain this is "
        f"expected (hydromt logs 'Skipping method, as no data has been found' "
        f"during rule 1.05); otherwise the model was built without them.",
        stacklevel=3,
    )
    return None


def gauges_layer_name(geoms, gauges_fn) -> Optional[str]:
    """The staticgeoms layer holding the user's gauges, or None.

    Parameters
    ----------
    geoms : mapping
        ``mod.geoms.data`` — layer name to GeoDataFrame.
    gauges_fn : str | Path | None
        The config's ``output_locations``, in any of its spellings.
    """
    return _resolve(geoms, gauges_fn, "staticgeoms layer", GAUGES_PREFIX)


def gauges_variable_name(results, gauges_fn, variable: str = "Q") -> Optional[str]:
    """The ``output.csv`` variable holding the gauges' timeseries, or None.

    Parameters
    ----------
    results : mapping
        ``mod.output_csv.data`` — variable name to DataArray.
    gauges_fn : str | Path | None
        The config's ``output_locations``.
    variable : str
        The wflow output header, ``Q`` (discharge) or ``P`` (precipitation);
        rule 1.05 configures both for gauges.
    """
    return _resolve(results, gauges_fn, f"{variable} variable", f"{variable}_{GAUGES_PREFIX}")
