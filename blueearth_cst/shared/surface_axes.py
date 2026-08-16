# -*- coding: utf-8 -*-
"""Derive a response-surface axis from the stress-test lookup.

**This module is the REFERENCE IMPLEMENTATION of HM-7, not its definition.** No
rule in this repository calls it: WF3 ends at the reduction, and the consumers
that actually draw a surface are out-of-repo (CST-API, the frontend,
``csthelpers``). They re-implement from
``dev/reference/contracts/hydrological-model-seam.md``, as does this
repository's own ``docs/notebooks/Climate Stress Test.ipynb``, which is
deliberately written against the contract text rather than against this module.
So when the two disagree, the contract wins and this file is the defect.

**Why the axis is derived rather than stored.** It used to be two columns on
every indicator row, computed at reduction time as a month-length-weighted
annual mean of the member's twelve monthly perturbations. That misreports any
seasonal design -- +30% imposed in JJA reads as +7.6% -- and baking ONE collapse
into the results made every other axis unrecoverable from them. Given the
lookup, an axis is fully determined, so storing one is a cache of a derivation.

**Pure module.** No Snakemake import, and deliberately no
``shared/snake_utils`` import either: the join key's width is inferred from the
lookup rather than taken from ``index_width(ST_NUM)``, so nothing here needs to
be told the member count. A derivation that had to be told the member count
could be told the wrong one.
"""

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

#: Month lengths in the weather generator's ``noleap`` calendar -- a year is 365
#: days and February is always 28, so there is no leap branch to reach. The same
#: weights WF2 uses for its annual change factor, which is what lets the CMIP6
#: overlay be compared against these axes at all.
_MONTH_LENGTHS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

_ALL_MONTHS = tuple(range(1, 13))

_MONTH_INITIALS = "JFMAMJJASOND"
_MONTH_ABBREV = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

#: The lookup column each axis variable reads, and the name of the derived
#: column a consumer plots. The derived columns keep the RETIRED columns' names
#: on purpose: an existing call site keeps working and simply receives values
#: that are now correct for a seasonal design. What changed for a consumer is
#: that it must join the lookup to obtain them.
AXIS_COLUMN = {"temp": "temp_change", "precip": "precip_change"}

#: The unit each variable's levels are formatted in. ``temp_change`` is additive
#: degC; ``precip_change`` is percent.
_AXIS_UNIT = {"temp": " °C", "precip": "%"}

#: Closed vocabularies. A third variable is refused rather than admitted:
#: ``precip_variance`` is a lookup column but not a grid dimension -- its levels
#: are indexed by the PRECIP step, so an axis over it would be a relabelling of
#: the precip axis.
_VARIABLES = frozenset(AXIS_COLUMN)

#: ``mean`` alone, and extension is a ruling rather than a judgement call: a
#: statistic may be added ONLY if it is affine in the member's step index, with
#: the proof recorded in the design's revision log. A max, a quantile or a
#: variance is not, and admitting one breaks the evenly-spaced guarantee that
#: makes the surface a regular grid.
_STATISTICS = frozenset({"mean"})

_SURFACE_KEYS = frozenset({"id", "x", "y"})
_AXIS_KEYS = frozenset({"variable", "months", "statistic"})

_SURFACE_ID = re.compile(r"^[a-z0-9_]+$")

#: The rectilinearity tolerance (D17). A TRUE tolerance, unlike the exact-zero
#: threshold month classification uses, and for the opposite reason: the levels
#: compared here are DIFFERENT values reached by different arithmetic, so
#: float64 noise at the 1e-16 scale is expected and must not fail -- while a
#: non-affine statistic breaks the spacing by orders of magnitude. 1e-9 sits
#: between the two by seven decades in each direction.
RECTILINEARITY_RTOL = 1e-9

#: The cap on clause groups in a caption, PER GROUP rather than over their sum.
#: A combined cap would let a busy "also vary" clause swallow the held-month
#: clause, which is the more informative of the two -- a held month is a decision
#: the user made about a month the axis does not show.
_CLAUSE_CAP = 3


class SurfaceDeclarationError(ValueError):
    """A malformed ``reporting.surfaces`` declaration."""


class DuplicateAxisVariableError(SurfaceDeclarationError):
    """Both axes of one surface declare the same variable.

    Refused because the rest of this design CANNOT REPRESENT it, not because it
    is unwise: ``SurfaceJoin.axes`` is keyed by variable so one axis would
    overwrite the other, and both axes name their derived column through
    ``AXIS_COLUMN``, so both would target one column. An implementation handed
    such a declaration must either discard an axis or return an object that
    violates its own API. Orientation reversal stays legal.
    """


class LookupKeyWidthError(ValueError):
    """A lookup whose ``st_id`` values are not all one width.

    WG-2 pins one width per table, and the whole join-key inference rests on
    that being true -- so a table mixing widths is malformed rather than merely
    awkward.
    """


class HeterogeneousAxisError(ValueError):
    """The declared months do not share one ``(min, max)`` range.

    A caption is a claim, and no caption can honestly describe a mean of unlike
    perturbations. Not a dead end: the message names the homogeneous subsets, so
    the user declares one and gets an honest axis.
    """


class HeldMonthInAxisError(ValueError):
    """The declared months include a month held constant across members.

    A held month contributes a constant to the mean and so reproduces exactly
    the annual misreport this contract removed.
    """


class NonRectilinearAxisError(ValueError):
    """The distinct axis levels are not evenly spaced.

    The postcondition the evenly-spaced guarantee never had. A closed statistic
    vocabulary is the static half of the same claim; this is the half that would
    catch an affineness argument that turns out to be wrong.
    """


class BaselinePartitionError(ValueError):
    """The indicator/lookup partition is not baseline-and-surface.

    Under an absence-means-baseline encoding a mis-keyed join does NOT surface
    as an empty result -- it surfaces as "every row is baseline", which is a
    shape the partition is designed to produce and therefore looks plausible.
    """


class SurfaceMemberMismatchError(ValueError):
    """A member the lookup declares is missing from the indicator tables.

    The failure this catches is worse than a mis-keyed join, which produces a
    visibly wrong shape: a SHORT join produces a plausible surface with holes in
    it, or a biased one if the missing members sit at one end of the grid.
    """


@dataclass(frozen=True)
class Axis:
    """A declared axis: ``{variable, months, statistic}``.

    ``months`` is ``None`` when undeclared, which is not the same as "all
    twelve": it defaults to the member-VARYING month set, derived from the
    lookup. That default is the single most consequential decision here -- it
    makes a seasonal design report the value it imposed without the user having
    to remember to declare anything, while leaving a uniform design's axis
    identical to the retired annual collapse.
    """

    variable: str
    months: Optional[tuple] = None
    statistic: str = "mean"


@dataclass(frozen=True)
class Surface:
    """One declared response surface: an id and two axes."""

    id: str
    x: Axis
    y: Axis


@dataclass(frozen=True)
class AxisResult:
    """What a derivation returns: FOUR things, not one.

    A consumer needs all four. ``degenerate`` decides whether the axis is a plot
    dimension or an annotation, and ``months`` is derived rather than declared in
    the default case -- so a caller that did not declare it cannot otherwise know
    which months were collapsed.
    """

    values: pd.Series
    caption: str
    degenerate: bool
    months: tuple
    variable: str


@dataclass(frozen=True)
class SurfaceJoin:
    """Indicator rows placed on a surface, with the baseline partitioned out."""

    surface_df: pd.DataFrame
    baseline_df: pd.DataFrame
    axes: Mapping[str, AxisResult] = field(default_factory=dict)
    key_width: int = 1


#: The surface a project gets when it declares none. Both axes take the derived
#: month set, so this is behaviour-preserving on a uniform design -- every
#: shipped config -- and automatically correct on a seasonal one.
#:
#: Its id is ``default`` rather than ``annual``, and that is not cosmetic: a
#: typed label drifts from the design it describes, which is the whole premise of
#: deriving captions. ``annual`` would be accurate only for a uniform design; a
#: JFM design's default surface would be identified as ``annual`` while captioned
#: "mean change over JFM". ``annual`` is reserved for a surface a user declares
#: explicitly with all twelve months -- which the subset rule then correctly
#: refuses on a seasonal design.
DEFAULT_SURFACE = Surface(
    id="default", x=Axis(variable="temp"), y=Axis(variable="precip")
)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_lookup(lookup_path: Union[str, Path]) -> pd.DataFrame:
    """Read ``stress_test_lookup.csv`` with ``st_id`` forced to TEXT.

    The dtype is not optional and not defensive: ``pd.read_csv`` with no dtype
    returns ``01`` as ``1``, and under the ``st_0``-absent encoding the resulting
    miss presents as "every row is the baseline" rather than as an empty result.
    """
    return pd.read_csv(lookup_path, dtype={"st_id": str})


def read_indicators(indicators_path: Union[str, Path]) -> pd.DataFrame:
    """Read an indicator table with ``st_id`` forced to TEXT.

    Sits beside :func:`read_lookup` so the library owns BOTH reads. A caller who
    loaded the frame some other way is still repaired by :func:`join_axes`, which
    re-pads both key columns before partitioning -- but owning the read is what
    makes the repair unnecessary in the common case.
    """
    return pd.read_csv(indicators_path, dtype={"st_id": str})


def key_width(lookup_df: pd.DataFrame) -> int:
    """The join key's width, INFERRED from the lookup rather than passed in.

    WG-2 pins every ``st_id`` in one lookup to a single width, so this reads a
    pinned property rather than guessing.
    """
    widths = {len(str(value)) for value in lookup_df["st_id"]}
    if len(widths) != 1:
        raise LookupKeyWidthError(
            f"the lookup mixes st_id widths {sorted(widths)}; WG-2 pins one "
            f"width for the whole table, and the join key is inferred from it"
        )
    return widths.pop()


# ---------------------------------------------------------------------------
# declaration parsing
# ---------------------------------------------------------------------------


def _parse_axis(raw, surface_id: str, position: str) -> Axis:
    if not isinstance(raw, Mapping):
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}: axis {position!r} must be a mapping of "
            f"{sorted(_AXIS_KEYS)}, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _AXIS_KEYS)
    if unknown:
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: unknown key(s) "
            f"{unknown}. The key set is closed ({sorted(_AXIS_KEYS)}) so a typo "
            f"inside a declaration is refused rather than ignored"
        )
    variable = raw.get("variable")
    if variable not in _VARIABLES:
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: variable "
            f"{variable!r} is not one of {sorted(_VARIABLES)}"
        )
    statistic = raw.get("statistic", "mean")
    if statistic not in _STATISTICS:
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: statistic "
            f"{statistic!r} is not one of {sorted(_STATISTICS)}. Only AFFINE "
            f"statistics may define an axis -- a max or a quantile breaks the "
            f"evenly-spaced guarantee and the surface stops being a regular grid"
        )
    months = raw.get("months")
    if months is not None:
        months = _validate_months(months, surface_id, position)
    return Axis(variable=variable, months=months, statistic=statistic)


def _validate_months(months, surface_id: str, position: str) -> tuple:
    if isinstance(months, (str, bytes)) or not isinstance(months, Sequence):
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: months must be a list "
            f"of integers 1..12, got {months!r}"
        )
    values = list(months)
    if not values:
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: months is empty"
        )
    if any(not isinstance(m, int) or isinstance(m, bool) for m in values):
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: months must be "
            f"integers, got {values!r}"
        )
    if any(m < 1 or m > 12 for m in values):
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: months {values!r} "
            f"fall outside 1..12"
        )
    if len(set(values)) != len(values):
        raise SurfaceDeclarationError(
            f"surface {surface_id!r}, axis {position!r}: months {values!r} "
            f"repeat a month"
        )
    return tuple(sorted(values))


def parse_surfaces(config: Mapping) -> list:
    """Parse and REFUSE the ``reporting.surfaces`` declaration, at parse time.

    Called from ``run_stress_test.smk`` beside the other parse-time refusals, so
    a malformed declaration fails ``--dry-run`` and ``pytest tests/test_cli.py``
    is its gate. Absent or empty yields :data:`DEFAULT_SURFACE`.

    The section is optional, and the trailing ``or {}`` spellings below are the
    point rather than noise: ``reporting:`` written with no body parses as
    ``None`` and would raise on subscript, while D10 promises that case resolves
    to the default.
    """
    reporting_cfg = (config or {}).get("reporting") or {}
    surfaces_cfg = reporting_cfg.get("surfaces") or []
    if not surfaces_cfg:
        return [DEFAULT_SURFACE]

    if not isinstance(surfaces_cfg, Sequence) or isinstance(surfaces_cfg, (str, bytes)):
        raise SurfaceDeclarationError(
            f"reporting.surfaces must be a list of surface declarations, got "
            f"{type(surfaces_cfg).__name__}"
        )

    surfaces = []
    seen_ids = set()
    for index, raw in enumerate(surfaces_cfg):
        if not isinstance(raw, Mapping):
            raise SurfaceDeclarationError(
                f"reporting.surfaces[{index}] must be a mapping, got "
                f"{type(raw).__name__}"
            )
        unknown = sorted(set(raw) - _SURFACE_KEYS)
        if unknown:
            raise SurfaceDeclarationError(
                f"reporting.surfaces[{index}]: unknown key(s) {unknown}. The "
                f"key set is closed ({sorted(_SURFACE_KEYS)})"
            )
        surface_id = raw.get("id")
        if not isinstance(surface_id, str) or not _SURFACE_ID.match(surface_id):
            raise SurfaceDeclarationError(
                f"reporting.surfaces[{index}]: id {surface_id!r} must match [a-z0-9_]+"
            )
        if surface_id in seen_ids:
            raise SurfaceDeclarationError(
                f"reporting.surfaces: id {surface_id!r} is declared twice"
            )
        seen_ids.add(surface_id)

        x = _parse_axis(raw.get("x"), surface_id, "x")
        y = _parse_axis(raw.get("y"), surface_id, "y")
        # The one constraint NO per-field validator can reach: both variables
        # are individually inside the closed enum, and only the PAIR is illegal.
        if {x.variable, y.variable} != _VARIABLES:
            raise DuplicateAxisVariableError(
                f"surface {surface_id!r} declares {x.variable!r} on both axes. "
                f"The two axes must name different variables "
                f"({sorted(_VARIABLES)}); orientation reversal is legal, "
                f"repetition is not representable -- one axis would overwrite "
                f"the other and both would target the same derived column"
            )
        surfaces.append(Surface(id=surface_id, x=x, y=y))
    return surfaces


def warn_on_heterogeneous_design(stress_test_cfg: Mapping) -> None:
    """Warn when the varying months do not share one ``(min, max)`` range.

    A WARNING and not a refusal, deliberately: such an experiment is legitimate
    and runnable -- the members exist and the response is real -- and only its
    scalar SUMMARY is dishonest. Refusing here would forbid a legal experiment to
    prevent a bad label. The refusal lives at the axis instead, where a caption
    is actually being claimed.
    """
    for variable in sorted(_VARIABLES):
        block = ((stress_test_cfg or {}).get(variable) or {}).get("mean") or {}
        lo, hi = block.get("min"), block.get("max")
        if lo is None or hi is None:
            continue
        ranges = {
            month: (float(lo[month - 1]), float(hi[month - 1]))
            for month in _ALL_MONTHS
            if month - 1 < len(lo) and month - 1 < len(hi)
        }
        varying = {m: r for m, r in ranges.items() if r[0] != r[1]}
        if len(set(varying.values())) > 1:
            warnings.warn(
                f"stress_test.{variable}.mean: the varying months carry "
                f"differing (min, max) ranges {sorted(set(varying.values()))}. "
                f"The experiment is legal and will run, but no single axis can "
                f"honestly summarise it -- declare `months:` over one "
                f"homogeneous subset to get an interpretable axis.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# classification and collapse
# ---------------------------------------------------------------------------


def month_classes(lookup_df: pd.DataFrame, variable: str) -> tuple:
    """``(varying_months, held_levels)`` for one variable.

    A month is **varying** iff ``max - min > 0`` EXACTLY over the surface
    members, with no tolerance. Exact zero is right rather than merely simpler:
    a held month's values are bit-identical across members by construction --
    one written text, read by every member -- so a tolerance would buy nothing
    and would create a band in which a month is neither varying nor held.
    """
    column = AXIS_COLUMN[variable]
    grouped = lookup_df.groupby("month")[column]
    spans = grouped.max() - grouped.min()
    varying = [int(m) for m in sorted(spans.index) if spans[m] > 0]
    held = {
        int(m): float(grouped.first()[m]) for m in sorted(spans.index) if spans[m] == 0
    }
    return varying, held


def _member_ranges(lookup_df: pd.DataFrame, variable: str, months) -> dict:
    column = AXIS_COLUMN[variable]
    subset = lookup_df[lookup_df["month"].isin(months)]
    grouped = subset.groupby("month")[column]
    return {
        int(m): (float(grouped.min()[m]), float(grouped.max()[m]))
        for m in sorted(grouped.min().index)
    }


def _collapse(values: Mapping) -> float:
    """The month-length weighted mean, with its exact-equality short-circuit.

    The short-circuit is NORMATIVE, not an optimization. Under the homogeneity
    constraint, equal values are the normal path for every admissible axis, and a
    weighted mean of twelve identical values does not generally return that
    value: measured, ``np.average`` over the noleap month lengths differs from
    the input in 49% of random percents and 48% of random degC values. Realistic
    grids hit it -- a 0.6-1.4 precip range at ``step_num: 3`` gives -13.33333 ->
    -13.333330000000002 -- and this repo treats that ulp as load-bearing.
    """
    distinct = set(values.values())
    if len(distinct) == 1:
        return float(next(iter(distinct)))
    total_weight = sum(_MONTH_LENGTHS[m] for m in values)
    return float(sum(_MONTH_LENGTHS[m] * v for m, v in values.items()) / total_weight)


def derive_axis(lookup_df: pd.DataFrame, axis: Axis) -> AxisResult:
    """Evaluate a declared axis against the lookup.

    The evaluation ORDER is normative, because steps 1 and 2 decide whether
    step 3's refusals apply at all -- without it, the degenerate case and the
    subset rule collide, and the default month set for a degenerate axis is the
    precise input the subset rule says must raise.
    """
    varying, held = month_classes(lookup_df, axis.variable)
    column = AXIS_COLUMN[axis.variable]

    # 2. Degenerate: nothing varies, so the axis has a single level. A
    #    LEGITIMATE design -- a temperature-only stress test is exactly this on
    #    its precip axis -- so neither of step 3's constraints applies, and an
    #    explicit `months:` is admitted.
    if not varying:
        months = axis.months if axis.months is not None else _ALL_MONTHS
        values = _member_values(lookup_df, column, months)
        caption = _degenerate_caption(axis.variable, held, months)
        return AxisResult(
            values=values,
            caption=caption,
            degenerate=True,
            months=tuple(months),
            variable=axis.variable,
        )

    # 3. Constrain M, then collapse.
    months = axis.months if axis.months is not None else tuple(varying)
    held_in_m = [m for m in months if m in held]
    if held_in_m:
        raise HeldMonthInAxisError(
            f"axis {axis.variable!r}: months {held_in_m} are HELD constant "
            f"across members, and a held month contributes a constant to the "
            f"mean -- reproducing exactly the annual misreport this derivation "
            f"exists to remove. The varying months are {varying}"
        )
    ranges = _member_ranges(lookup_df, axis.variable, months)
    if len(set(ranges.values())) > 1:
        subsets: dict = {}
        for month, span in _member_ranges(lookup_df, axis.variable, varying).items():
            subsets.setdefault(span, []).append(month)
        named = "; ".join(
            f"{sorted(ms)} at {span[0]} to {span[1]}" for span, ms in subsets.items()
        )
        raise HeterogeneousAxisError(
            f"axis {axis.variable!r}: the declared months {list(months)} do not "
            f"share one (min, max) range, so their mean is an average of unlike "
            f"perturbations that no caption can honestly describe. Declare one "
            f"of the homogeneous subsets instead -- {named}"
        )

    values = _member_values(lookup_df, column, months)
    _assert_rectilinear(values, axis.variable)
    caption = _caption(lookup_df, axis.variable, months, varying, held)
    return AxisResult(
        values=values,
        caption=caption,
        degenerate=False,
        months=tuple(months),
        variable=axis.variable,
    )


def _member_values(lookup_df: pd.DataFrame, column: str, months) -> pd.Series:
    """One collapsed value per member, indexed by the padded ``st_id`` TEXT."""
    subset = lookup_df[lookup_df["month"].isin(list(months))]
    out = {}
    for st_id, member in subset.groupby("st_id"):
        out[st_id] = _collapse(
            dict(zip(member["month"].astype(int), member[column].astype(float)))
        )
    return pd.Series(out).sort_index()


def _assert_rectilinear(values: pd.Series, variable: str) -> None:
    levels = sorted(set(values.tolist()))
    if len(levels) <= 2:
        return  # two or fewer distinct levels are trivially evenly spaced
    gaps = [b - a for a, b in zip(levels, levels[1:])]
    mean_gap = sum(gaps) / len(gaps)
    for gap in gaps:
        if not math.isclose(gap, mean_gap, rel_tol=RECTILINEARITY_RTOL):
            raise NonRectilinearAxisError(
                f"axis {variable!r}: the distinct levels {levels} are not evenly "
                f"spaced (gaps {gaps}, mean {mean_gap}). Members are "
                f"min + (j/n)(max - min) month by month, so an affine collapse "
                f"is affine in the step index -- an unevenly spaced axis means "
                f"the statistic is not affine and the surface is not a regular "
                f"grid"
            )


def axis_values(lookup_df: pd.DataFrame, axis: Axis) -> pd.Series:
    """The collapsed value per member. See :func:`derive_axis` for the rest."""
    return derive_axis(lookup_df, axis).values


def axis_caption(lookup_df: pd.DataFrame, axis: Axis) -> str:
    """The derived caption. See :func:`derive_axis` for the rest."""
    return derive_axis(lookup_df, axis).caption


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------


def _label_months(months) -> str:
    months = sorted(months)
    if len(months) == 12:
        return "the year"
    run = _circular_run(months)
    if run is not None and len(months) <= 3:
        return "".join(_MONTH_INITIALS[m - 1] for m in run)
    if run is not None:
        return f"{_MONTH_ABBREV[run[0] - 1]}–{_MONTH_ABBREV[run[-1] - 1]}"
    return ", ".join(_MONTH_ABBREV[m - 1] for m in months)


def _circular_run(months):
    """The months in circular order if they form a contiguous run, else None.

    Circular rather than linear so ``{12, 1, 2}`` renders ``DJF`` and
    ``{9, ..., 5}`` renders ``Sep-May`` -- which is what subsumes the
    meteorological seasons with no season table.
    """
    present = set(months)
    for start in months:
        run = [((start - 1 + offset) % 12) + 1 for offset in range(len(months))]
        if set(run) == present:
            return run
    return None


def _format_level(value: float, variable: str) -> str:
    return f"{value:+.3g}{_AXIS_UNIT[variable]}"


def _format_range(span, variable: str) -> str:
    lo, hi = span
    return f"{_format_level(lo, variable)} to {_format_level(hi, variable)}"


def _clauses(groups, phrasing: str, variable: str, catch_all: str) -> list:
    """One clause builder, used for BOTH the varying and the held month sets."""
    if not groups:
        return []
    if len(groups) > _CLAUSE_CAP:
        return [catch_all]
    out = []
    for key, months in groups:
        label = _label_months(months)
        if phrasing == "held":
            if key == 0:
                out.append(f"{label} unchanged")
            else:
                out.append(f"{label} held at {_format_level(key, variable)}")
        else:
            out.append(f"{label} also vary, {_format_range(key, variable)}")
    return out


def _group_by_key(pairs) -> list:
    grouped: dict = {}
    for month, key in pairs:
        grouped.setdefault(key, []).append(month)
    return [(key, sorted(months)) for key, months in grouped.items()]


def _caption(lookup_df, variable: str, months, varying, held) -> str:
    """The non-degenerate caption.

    The leading phrase names ``M`` and NOTHING ELSE. A design whose twelve months
    all vary but whose declared ``M`` is JFM is captioned *over JFM*, because JFM
    is what was collapsed and what the plotted number is -- labelling it "over
    the year" asserts a quantity that was not computed, and the same ``M``
    collapses the projection overlay, so the error would propagate from the label
    into the comparison.
    """
    declared = set(months)
    extra_varying = [m for m in varying if m not in declared]
    held_months = [m for m in sorted(held) if m not in declared]

    parts = [f"mean change over {_label_months(months)}"]

    if extra_varying:
        spans = _member_ranges(lookup_df, variable, extra_varying)
        parts += _clauses(
            _group_by_key([(m, spans[m]) for m in extra_varying]),
            "vary",
            variable,
            "remaining months also vary",
        )
    if held_months:
        parts += _clauses(
            _group_by_key([(m, held[m]) for m in held_months]),
            "held",
            variable,
            "remaining months held at declared monthly offsets",
        )
    return "; ".join(parts)


def _degenerate_caption(variable: str, held, months) -> str:
    """The degenerate caption, over the held levels WITHIN ``M``.

    ``in <label(M)>`` is appended when ``M`` is not all twelve, for the same
    reason the leading phrase names ``M``: a caption may not describe months the
    axis did not collapse.
    """
    levels_in_m = {m: held[m] for m in months if m in held}
    distinct = sorted(set(levels_in_m.values()))
    if distinct == [0.0]:
        text = "unchanged"
    elif len(distinct) == 1:
        text = f"held at {_format_level(distinct[0], variable)}"
    else:
        mean = _collapse(levels_in_m)
        text = (
            f"held at declared monthly offsets "
            f"(weighted mean {_format_level(mean, variable)})"
        )
    if len(set(months)) != 12:
        text = f"{text} in {_label_months(months)}"
    return text


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


def join_axes(
    indicators_df: pd.DataFrame, lookup_df: pd.DataFrame, surface: Surface
) -> SurfaceJoin:
    """Place indicator rows on a surface, partitioning the baseline out.

    Padding happens HERE and nowhere else. ``derive_axis`` reads one table, so its
    index is whatever the lookup holds; only this function sees a second frame
    with a second provenance. The obvious implementer's error is to pad
    defensively in both, which double-pads any consumer that composes them.
    """
    width = key_width(lookup_df)
    baseline_token = "0".zfill(width)

    lookup_df = lookup_df.copy()
    lookup_df["st_id"] = lookup_df["st_id"].astype(str).str.zfill(width)
    indicators_df = indicators_df.copy()
    indicators_df["st_id"] = indicators_df["st_id"].astype(str).str.zfill(width)

    indicator_ids = set(indicators_df["st_id"])
    lookup_ids = set(lookup_df["st_id"])

    # a. what the tables carry and the lookup does not is EXACTLY the baseline
    extra = indicator_ids - lookup_ids
    if extra != {baseline_token}:
        raise BaselinePartitionError(
            f"the indicator tables carry st_id(s) {sorted(extra)} absent from "
            f"the lookup; exactly {baseline_token!r} (the reserved unperturbed "
            f"baseline) was expected. Under the st_0-absent encoding a mis-keyed "
            f"join presents as 'every row is baseline' rather than as an empty "
            f"result, so this is asserted rather than assumed"
        )
    # b. set EQUALITY between the surface members and the lookup's members
    missing = lookup_ids - (indicator_ids - {baseline_token})
    if missing:
        raise SurfaceMemberMismatchError(
            f"{len(missing)} member(s) the lookup declares are missing from the "
            f"indicator tables: {sorted(missing)}. A short join returns a "
            f"plausible surface with holes in it -- or a biased one if the "
            f"missing members sit at one end of the grid -- rather than "
            f"reporting a mismatch"
        )

    baseline_df = indicators_df[indicators_df["st_id"] == baseline_token]
    surface_df = indicators_df[indicators_df["st_id"] != baseline_token].copy()
    # c. the degenerate residue check b cannot see: an EMPTY lookup satisfies b
    #    vacuously.
    if surface_df.empty:
        raise BaselinePartitionError(
            "the surface partition is empty: every indicator row is the "
            "baseline, which means either the lookup declares no members or the "
            "join key did not match"
        )

    axes = {}
    for axis in (surface.x, surface.y):
        result = derive_axis(lookup_df, axis)
        axes[axis.variable] = result
        surface_df[AXIS_COLUMN[axis.variable]] = surface_df["st_id"].map(result.values)

    return SurfaceJoin(
        surface_df=surface_df,
        baseline_df=baseline_df,
        axes=axes,
        key_width=width,
    )
