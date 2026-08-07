# -*- coding: utf-8 -*-
"""Which indicator tables a WF3 experiment emits, derived from ``wflow_outvars``.

CR-2 splits the response-surface results into **one table per output variable**
instead of the two fixed tables (``q_indicators.csv`` + ``basin_indicators.csv``)
that preceded it. The set is therefore config-dependent, and four places need to
agree on it before any of them can run:

- ``Snakefile_climate_experiment`` — ``WF3_TARGETS`` and rule 3.16's ``output:``,
  at DAG-construction time;
- ``blueearth_cst/experiment/export_wflow_results.py`` — what it writes, and the
  ``variable`` half of each composite ``metric``;
- ``blueearth_cst/shared/interchange_contracts.py`` — HM-7's per-table checks;
- ``dev/scripts/check_baseline.py`` and ``semantic_tree_diff.py`` — the target
  list and the path map.

Runtime discovery is not an option for the first of those. The pre-CR-2 writer
found its variables by reading the wflow run CSV's own columns
(``[x for x in sim.columns if "basavg" in x]``), which cannot work when Snakemake
needs the output paths *before* any rule has produced a CSV to inspect.

**The tokens are a third spelling, and that is the accepted cost.** Alongside the
CSDMS names (``river_water__volume_flow_rate``) and the Tier 2 display labels,
these short tokens exist because a composite metric built from snake-cased
semantic names would read ``actual_evapotranspiration_annual_total`` — which
undercuts the readability that motivated composing the name at all. CR-2 places
this mapping in the seam contract for that reason: it is a contract, not an
implementation detail.

The rule for minting future tokens, so they are not chosen ad hoc: **where the
repo already has a canonical short name, use it; only mint where none exists; and
disambiguate against names already in use.** Its three consequences are why
``precip`` is not ``p`` (``naming.md`` §6 tier 2 declares ``precip`` canonical, and
``p`` would be a seventh spelling), why ``aet`` is not ``et`` (``pet`` is already
canonical here and one letter apart in the same file is a misreading waiting to
happen), and why ``snow`` is not ``swe`` (the CSDMS name is
``snowpack_liquid_water__depth`` — snowpack *liquid water*, not total water
equivalent, so minting ``swe`` would assert a physical claim upstream does not
make, which ``AGENTS.md`` puts out of scope).
"""

from __future__ import annotations

#: Semantic name (as it appears in ``workflows.model_creation.wflow_outvars``)
#: → short token used in filenames and in the composite ``metric``.
#:
#: Authoritative source for the semantic names: the ``WFLOW_VARS`` map in
#: ``dev/reference/workflows/model_creation.md``. **Six entries, not five** —
#: ``precipitation`` is one of them, emitted at registry locations with header
#: ``P`` when a ``location_registry`` is configured.
VARIABLE_TOKENS = {
    "river discharge": "q",
    "precipitation": "precip",
    "actual evapotranspiration": "aet",
    "groundwater recharge": "recharge",
    "overland flow": "overland_flow",
    "snow": "snow",
}

#: Filename suffix shared by every indicator table. ``q_indicators.csv`` keeps the
#: name it was given in R9, which is why the pattern is token-first.
_TABLE_SUFFIX = "_indicators.csv"


class UnknownOutputVariableError(ValueError):
    """``wflow_outvars`` names a variable with no token, so no table can be named.

    Raised rather than skipped. A silently ignored entry would produce a run whose
    results are missing a variable the config asked for, with nothing in the tree
    saying so — and the absence would look identical to "that variable was never
    requested".
    """


def variable_token(outvar: str) -> str:
    """Short token for one ``wflow_outvars`` entry."""
    try:
        return VARIABLE_TOKENS[outvar]
    except KeyError:
        known = ", ".join(sorted(VARIABLE_TOKENS))
        raise UnknownOutputVariableError(
            f"wflow_outvars names {outvar!r}, which has no indicator-table token. "
            f"Known variables: {known}. Add the variable to VARIABLE_TOKENS in "
            f"blueearth_cst/shared/indicator_tables.py AND to the seam contract "
            f"dev/reference/contracts/hydrological-model-seam.md, which is where "
            f"this mapping is published."
        ) from None


def indicator_table_filename(outvar: str) -> str:
    """``'river discharge'`` → ``'q_indicators.csv'``."""
    return f"{variable_token(outvar)}{_TABLE_SUFFIX}"


def indicator_tables(wflow_outvars) -> dict[str, str]:
    """Map each configured output variable's TOKEN to its table filename.

    Keyed by token rather than by semantic name because the token is what every
    consumer downstream carries: it is in the filename, in the composite
    ``metric``, and in the Snakemake output key. Keying by the semantic name would
    make ``"river discharge"`` a dict key with a space in it, which then has to be
    translated at every use site.

    Order follows ``wflow_outvars`` so the derived output set is stable for a
    given config; duplicates collapse, since two identical entries would name one
    table.
    """
    tables: dict[str, str] = {}
    for outvar in wflow_outvars or []:
        token = variable_token(outvar)
        tables[token] = f"{token}{_TABLE_SUFFIX}"
    return tables
