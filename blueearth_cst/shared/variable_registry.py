"""One record per climate variable (R14 `C-57`, design D-10.6).

Before this, a variable was described in two places that did not know about each
other: ``CLIMATE_VARS`` in ``climate_analysis/climate_figures.py`` said how to
PLOT it, and every project's ``variables:`` block said what it MEANS. The second
had no defaults at all — `C-57` called the short form "registry-resolved" and
there was no registry — so every config restated the same four fields for the
same two variables, and a typo in any of them was a valid config.

**This module is in ``shared/`` because variable metadata already crosses
workflows.** ``CLIMATE_VARS`` is read by WF0's figures and by WF1's
``plot_map_forcing``; the projections spec is read by WF2. A home under
``projections/`` would have been wrong on the day it landed.

**Two attribute groups, deliberately not merged.**

* :class:`Presentation` — how a reader sees the variable. Axis label, the unit
  the FIGURE is drawn in, and how the variable aggregates in time.
* :class:`ProjectionSpec` — what WF2 needs to difference it: the post-rename
  source name, the canonical quantity, the units that quantity is stored in, and
  whether a change is relative or absolute.

They overlap on nothing, and the one field that looks shared is the trap:

    Presentation.unit    for `precip` is  "mm"       — a yearly TOTAL
    ProjectionSpec.units for `precip` is  "mm/day"   — a monthly mean RATE

Both are right for their reader. Collapsing them because two spellings of one
word sit next to each other would mislabel every figure axis or every change
factor, depending which way the collapse went, and nothing downstream would
raise. ``tests/test_variable_registry.py`` pins the INEQUALITY for that reason.

**``projections`` is nullable, and that is information rather than an omission.**
``pet`` is plotted and is not a CMIP6 variable this toolbox differences, so it
has no projection spec — and a variable in the registry with no spec gets a
different refusal from a variable that is not in the registry at all. One says
"declare it long-form", the other says "add it to the registry", and each is
the true remedy for its case.

The registry is toolbox VOCABULARY, not a project surface: the long form of
``variables:`` stays valid, so a project can always declare a variable the
registry has never heard of without editing code.
"""

from __future__ import annotations

from typing import NamedTuple

#: What a stored monthly series IS. A ``rate`` accumulates over its interval; a
#: ``state`` is an instantaneous quantity averaged over it.
CANONICAL_KINDS = frozenset({"rate", "state"})

#: How a change is expressed. Read by stage B; **nothing infers this from a
#: name**, which is the whole reason the spec exists.
CHANGE_KINDS = frozenset({"relative", "absolute"})


class Presentation(NamedTuple):
    """How a variable is drawn.

    ``how`` is not cosmetic: a summed temperature is meaningless and a meaned
    rainfall understates by roughly 365x.
    """

    label: str
    unit: str
    how: str
    style: str


class ProjectionSpec(NamedTuple):
    """What WF2 needs to compute a change factor for a variable.

    ``source`` names the **post-rename** variable: the catalog's
    ``data_adapter.rename`` maps ``pr -> precip`` and ``tas -> temp`` before the
    reducer sees the data.
    """

    source: str
    canonical: str
    units: str
    change: str
    #: The near-zero threshold below which a reference value makes a RELATIVE
    #: change meaningless (`C-64`). ``None`` for an absolute variable, which has
    #: no denominator to be small. **Appended, and it has to stay last**:
    #: `analyze_projections.smk` flattens the spec to a plain list to cross the
    #: Snakemake params boundary and `derive_change_factors` rebuilds it with
    #: ``VariableSpec(*fields)``, so a field inserted earlier would silently
    #: shift every value after it.
    min_denominator: float | None = None


class Variable(NamedTuple):
    """One climate variable, in both vocabularies.

    ``projections`` is ``None`` for a variable this toolbox plots but does not
    difference.
    """

    presentation: Presentation
    projections: ProjectionSpec | None


#: The registry. **Insertion order is load-bearing** — ``plot_map_forcing``
#: selects the canonical climate set with ``list(CLIMATE_VARS)``. Add a new
#: variable at the END unless the figure order is meant to change with it.
VARIABLES: dict[str, Variable] = {
    "precip": Variable(
        presentation=Presentation(
            label="precipitation", unit="mm", how="sum", style="precip"
        ),
        projections=ProjectionSpec(
            source="precip",
            canonical="rate",
            units="mm/day",
            change="relative",
            # A2, closing OQ-9: 0.1 mm/day. A month drier than this is a month
            # where "30% more rain" describes nothing a reader should act on.
            # Deliberately conservative and revisable by measurement without a
            # design change. It was `dry_month.DEFAULT_MIN_REFERENCE` until
            # `C-64`; the value is unchanged, only its home.
            min_denominator=0.1,
        ),
    ),
    "temp": Variable(
        presentation=Presentation(
            label="air temperature", unit="$\\degree$C", how="mean", style="temp"
        ),
        projections=ProjectionSpec(
            source="temp",
            canonical="state",
            units="degC",
            change="absolute",
            # Absolute: there is no denominator, so no threshold. Left at the
            # default rather than set to a number that would never be read.
        ),
    ),
    "pet": Variable(
        presentation=Presentation(
            label="potential evaporation", unit="mm", how="sum", style="pet"
        ),
        # Plotted, never differenced: no shipped config declares `pet` under
        # `variables:`. Naming the absence rather than inventing a spec is what
        # lets the short form refuse it with the remedy that is actually true.
        projections=None,
    ),
}

#: The figures' view of the registry, byte-identical to the table it replaces
#: and in the same order. Kept as a NAME rather than rewriting its 22 call
#: sites: `C-49` — getting this out of Python entirely — is a separate register
#: row and stays outside R14 P1c by owner ruling. Deriving it here means that
#: row, when it comes, acts on one file instead of reconciling two.
CLIMATE_VARS: dict[str, dict[str, str]] = {
    name: dict(variable.presentation._asdict()) for name, variable in VARIABLES.items()
}


def projection_defaults(name: str) -> ProjectionSpec | None:
    """The registry's projection spec for ``name``, or ``None``.

    ``None`` is returned both for an unknown variable and for a known variable
    with no spec; :func:`blueearth_cst.projections.variable_spec.parse` tells
    them apart with :data:`VARIABLES` because the two need different remedies.
    """
    entry = VARIABLES.get(name)
    return entry.projections if entry else None
