# -*- coding: utf-8 -*-
"""Unit cases for the response-surface axis derivation (HM-7).

**This is the only gate `shared/surface_axes.py` has.** No rule in the repo
calls it: the consumers that draw a surface are out-of-repo and re-implement
from the contract text. Under this repo's own ladder — "only the tests covering
the file you changed" — an unnamed test file is an ungated file, and this is the
largest new surface in the design.

The collapse cases inherited from `test_export_wflow_results.py` carry a load
the rest of the ladder cannot: every tracked config uses a FLAT monthly
perturbation vector, so the baseline manifest and every fixture run are blind to
the collapse by construction. A seasonal vector is the only input that can tell
reading January apart from taking the annual mean, and nothing on disk supplies
one.
"""

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.shared.surface_axes import (
    DEFAULT_SURFACE,
    Axis,
    BaselinePartitionError,
    DuplicateAxisVariableError,
    HeldMonthInAxisError,
    HeterogeneousAxisError,
    LookupKeyWidthError,
    NonRectilinearAxisError,
    Surface,
    SurfaceDeclarationError,
    SurfaceMemberMismatchError,
    axis_caption,
    axis_values,
    derive_axis,
    join_axes,
    key_width,
    month_classes,
    parse_surfaces,
)

_MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)


def _lookup(members, width=None):
    """A lookup frame from ``{st_id: {"temp": [...12], "precip": [...12]}}``."""
    width = width or len(str(len(members)))
    rows = []
    for index, (st_id, spec) in enumerate(sorted(members.items()), start=1):
        del index
        for month in range(1, 13):
            rows.append(
                {
                    "st_id": st_id,
                    "month": month,
                    "temp_change": float(spec["temp"][month - 1]),
                    "precip_change": float(spec["precip"][month - 1]),
                    "precip_variance_change": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _linspace_members(lo, hi, steps, months=range(1, 13), held=0.0, variable="precip"):
    """``steps`` members whose declared months walk lo -> hi and the rest sit at
    ``held`` — the shape ``np.linspace`` gives rule 3.09."""
    months = set(months)
    levels = np.linspace(lo, hi, steps)
    members = {}
    for index, level in enumerate(levels, start=1):
        series = [float(level) if m in months else held for m in range(1, 13)]
        flat = [0.0] * 12
        members[f"{index}"] = (
            {"precip": series, "temp": flat}
            if variable == "precip"
            else {"temp": series, "precip": flat}
        )
    return _lookup(members)


# ---------------------------------------------------------------------------
# V5 / V6 — the collapse, and what the default month set buys
# ---------------------------------------------------------------------------


def test_uniform_axis_matches_the_retired_annual_collapse():
    """V5: on a uniform design the derived axis EQUALS the old baked column.

    That equality is the whole reason the new default is safe to adopt: every
    shipped config is uniform, so nothing a user has already run moves.
    """
    df = _linspace_members(-30.0, 30.0, 3)
    values = axis_values(df, Axis(variable="precip"))
    assert list(values) == [-30.0, 0.0, 30.0]

    # ... and the retired collapse, computed here directly, agrees exactly.
    for st_id, value in values.items():
        monthly = df[df.st_id == st_id].sort_values("month")["precip_change"].to_numpy()
        expected = (
            float(monthly[0])
            if np.ptp(monthly) == 0
            else float(np.average(monthly, weights=_MONTH_LENGTHS))
        )
        assert value == expected


def test_seasonal_axis_reports_the_imposed_value():
    """V6: the defect this design exists to remove.

    A +30% JJA perturbation read +7.6% on the old axis — roughly a quarter of
    what was imposed, and a single-month perturbation compressed to about a
    twelfth. The derived axis reports what the member actually imposed.
    """
    df = _linspace_members(-30.0, 30.0, 3, months=(6, 7, 8), held=0.0)
    values = axis_values(df, Axis(variable="precip"))
    assert list(values) == [-30.0, 0.0, 30.0]

    # The annual collapse of the SAME members, which is what used to be stored.
    monthly = df[df.st_id == "3"].sort_values("month")["precip_change"].to_numpy()
    annual = float(np.average(monthly, weights=_MONTH_LENGTHS))
    assert annual == pytest.approx(7.6, abs=0.1)


def test_flat_vector_short_circuit_holds_in_percent_space():
    """V19: the exact-equality short-circuit is normative, not an optimization.

    A weighted mean of twelve identical values does not generally return that
    value. The unit change is what created the hazard: in MULTIPLIER space the
    flat-vector mean was exact for every one of 50,000 random draws, while in
    percent space it differs in about half of them. A 0.6-1.4 grid at
    ``step_num: 3`` is a realistic config that hits it.
    """
    # A level this repo's OWN writer produces, not an illustrative number: the
    # 0.5-1.6 grid at `step_num: 6`, converted exactly as rule 3.09 converts it.
    # Two of its seven levels are not fixed points of the weighted mean.
    level = float(str(np.float32(np.linspace(0.5, 1.6, 7)[1]))) * 100 - 100
    df = _lookup({"1": {"precip": [level] * 12, "temp": [0.0] * 12}})
    assert (
        float(np.average([level] * 12, weights=_MONTH_LENGTHS)) != level
    )  # the hazard
    assert axis_values(df, Axis(variable="precip"))["1"] == level


def test_month_length_weighting_not_a_plain_mean():
    """The weighting is retained for compatibility with WF2 and the overlay, so
    it has to be the month-LENGTH mean rather than the simpler count mean."""
    january = [30.0] + [0.0] * 11
    february = [0.0, 30.0] + [0.0] * 10
    df_jan = _lookup({"1": {"precip": january, "temp": [0.0] * 12}})
    df_feb = _lookup({"1": {"precip": february, "temp": [0.0] * 12}})
    jan = axis_values(df_jan, Axis(variable="precip", months=tuple(range(1, 13))))["1"]
    feb = axis_values(df_feb, Axis(variable="precip", months=tuple(range(1, 13))))["1"]
    assert jan == pytest.approx(30.0 * 31 / 365)
    assert feb == pytest.approx(30.0 * 28 / 365)
    assert jan != feb  # a count mean would make these equal


def test_month_classification_uses_exact_zero():
    """D11: bit-identical by construction, so no tolerance and no grey band."""
    df = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=-20.0)
    varying, held = month_classes(df, "precip")
    assert varying == [1, 2, 3]
    assert held == {m: -20.0 for m in range(4, 13)}


# ---------------------------------------------------------------------------
# V7 / V8 / V9 / V10 — the refusals, and the one case that must NOT refuse
# ---------------------------------------------------------------------------


def test_heterogeneous_axis_refused_and_names_the_subsets():
    """V7: no caption can honestly describe a mean of unlike perturbations.

    The refusal names the homogeneous subsets, which is what keeps it from being
    a dead end — the user declares one and gets an honest axis.
    """
    members = {}
    for index, (jfm, rest) in enumerate(
        [(-30.0, -10.0), (0.0, 0.0), (30.0, 10.0)], start=1
    ):
        series = [jfm if m <= 3 else rest for m in range(1, 13)]
        members[f"{index}"] = {"precip": series, "temp": [0.0] * 12}
    df = _lookup(members)

    with pytest.raises(HeterogeneousAxisError) as excinfo:
        axis_values(df, Axis(variable="precip"))
    assert "[1, 2, 3]" in str(excinfo.value)

    # ... and declaring one of the named subsets works.
    assert list(axis_values(df, Axis(variable="precip", months=(1, 2, 3)))) == [
        -30.0,
        0.0,
        30.0,
    ]


def test_held_month_in_the_declared_set_refused():
    """V8: a held month dilutes a varying one, which IS the annual misreport.

    ``months: [1..12]`` on a JFM-varying, Apr-Dec-held-at--20% design returns
    -15% for a member that imposed -30%.
    """
    df = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=-20.0)
    with pytest.raises(HeldMonthInAxisError) as excinfo:
        axis_values(df, Axis(variable="precip", months=tuple(range(1, 13))))
    assert "[1, 2, 3]" in str(excinfo.value)


def test_a_proper_subset_of_a_homogeneous_varying_set_is_admitted():
    """The subset rule constrains held months, not narrowness."""
    df = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=0.0)
    assert list(axis_values(df, Axis(variable="precip", months=(1, 2)))) == [
        -30.0,
        0.0,
        30.0,
    ]


def test_rectilinearity_postcondition():
    """V9: the check a closed statistic vocabulary cannot make.

    Its subject is an affineness argument that turns out to be WRONG, so the
    fixture is a hand-built lookup whose levels are not evenly spaced — which is
    what a non-affine collapse would produce.
    """
    members = {
        "1": {"precip": [-30.0] * 12, "temp": [0.0] * 12},
        "2": {"precip": [-25.0] * 12, "temp": [0.0] * 12},
        "3": {"precip": [30.0] * 12, "temp": [0.0] * 12},
    }
    with pytest.raises(NonRectilinearAxisError):
        axis_values(_lookup(members), Axis(variable="precip"))


def test_two_or_fewer_levels_pass_rectilinearity_trivially():
    df = _linspace_members(-30.0, 30.0, 2)
    assert len(set(axis_values(df, Axis(variable="precip")))) == 2


def test_degenerate_axis_admits_explicit_months():
    """V10: a temperature-only stress test is legal, and this is it on the
    precip axis. D27's ordering is what makes the explicit-``months`` case work:
    classification decides degeneracy BEFORE the subset rule is consulted, and
    without it the all-twelve default for this case is the precise input the
    subset rule says must raise.
    """
    df = _linspace_members(0.0, 3.0, 3, variable="temp")

    default = derive_axis(df, Axis(variable="precip"))
    assert default.degenerate is True
    assert set(default.values) == {0.0}
    assert default.caption == "unchanged"

    explicit = derive_axis(df, Axis(variable="precip", months=(1, 2, 3)))
    assert explicit.degenerate is True
    assert explicit.caption == "unchanged in JFM"

    # ... while the varying axis of the same design is an ordinary one.
    assert derive_axis(df, Axis(variable="temp")).degenerate is False


def test_a_degenerate_axis_uses_the_collapse_not_some_other_scalar():
    """D32: degeneracy bypasses the CONSTRAINTS, never the FORMULA.

    With ``M``'s months held at DIFFERENT offsets, "the constant" is ambiguous —
    the first month's level, an unweighted mean and a weighted mean all differ —
    so two conforming implementations could return different numbers.
    """
    series = [-20.0 if m <= 6 else -10.0 for m in range(1, 13)]
    df = _lookup({"1": {"precip": series, "temp": [0.0] * 12}})
    result = derive_axis(df, Axis(variable="precip"))
    expected = float(np.average(series, weights=_MONTH_LENGTHS))
    assert result.degenerate is True
    assert result.values["1"] == expected
    assert "weighted mean" in result.caption
    assert result.values["1"] != series[0]  # the tempting wrong answer


# ---------------------------------------------------------------------------
# V11 — all ten caption cases
# ---------------------------------------------------------------------------


def test_caption_case_1_uniform_whole_year():
    df = _linspace_members(-30.0, 30.0, 3)
    assert axis_caption(df, Axis(variable="precip")) == "mean change over the year"


def test_caption_explicit_subset_of_all_varying():
    """Case 1b, and the one v2 got wrong — captioning it "over the year" names a
    quantity that was not computed, and differs from the plotted one by 2x."""
    members = {}
    for index, (jfm, rest) in enumerate(
        [(-30.0, -10.0), (0.0, 0.0), (30.0, 10.0)], start=1
    ):
        members[f"{index}"] = {
            "precip": [jfm if m <= 3 else rest for m in range(1, 13)],
            "temp": [0.0] * 12,
        }
    caption = axis_caption(_lookup(members), Axis(variable="precip", months=(1, 2, 3)))
    assert caption == "mean change over JFM; Apr–Dec also vary, -10% to +10%"


def test_caption_case_2_rest_unchanged():
    df = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=0.0)
    assert (
        axis_caption(df, Axis(variable="precip"))
        == "mean change over JFM; Apr–Dec unchanged"
    )


def test_caption_case_3_rest_held_at_one_offset():
    df = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=-20.0)
    assert (
        axis_caption(df, Axis(variable="precip"))
        == "mean change over JFM; Apr–Dec held at -20%"
    )


def test_caption_case_3b_rest_held_at_several_offsets():
    members = {}
    for index, level in enumerate([-30.0, 0.0, 30.0], start=1):
        series = []
        for m in range(1, 13):
            if m <= 3:
                series.append(level)
            elif m <= 9:
                series.append(-20.0)
            else:
                series.append(-10.0)
        members[f"{index}"] = {"precip": series, "temp": [0.0] * 12}
    # `OND`, not `Oct–Dec`: the labelling rule renders a contiguous circular run
    # of THREE months as its initials. The design's illustrative caption in this
    # row spells it `Oct–Dec`, which contradicts the rule it states twice —
    # flagged for an owner ruling, rule followed here.
    assert axis_caption(_lookup(members), Axis(variable="precip")) == (
        "mean change over JFM; Apr–Sep held at -20%; OND held at -10%"
    )


def test_caption_case_3c_more_than_three_held_levels():
    """The cap is a LEGIBILITY rule: beyond three groups the honest statement is
    that the pattern is not summarisable and the reader should look at the
    lookup."""
    offsets = {4: -5.0, 5: -10.0, 6: -15.0, 7: -20.0}
    members = {}
    for index, level in enumerate([-30.0, 0.0, 30.0], start=1):
        series = [level if m <= 3 else offsets.get(m, -25.0 - m) for m in range(1, 13)]
        members[f"{index}"] = {"precip": series, "temp": [0.0] * 12}
    assert axis_caption(_lookup(members), Axis(variable="precip")) == (
        "mean change over JFM; remaining months held at declared monthly offsets"
    )


def test_caption_case_1c_both_kinds_outside_m():
    members = {}
    for index, (jfm, amj) in enumerate(
        [(-30.0, -10.0), (0.0, 0.0), (30.0, 10.0)], start=1
    ):
        series = []
        for m in range(1, 13):
            if m <= 3:
                series.append(jfm)
            elif m <= 6:
                series.append(amj)
            else:
                series.append(-20.0)
        members[f"{index}"] = {"precip": series, "temp": [0.0] * 12}
    # `AMJ` for the same reason as case 3b's `OND`.
    assert axis_caption(
        _lookup(members), Axis(variable="precip", months=(1, 2, 3))
    ) == ("mean change over JFM; AMJ also vary, -10% to +10%; Jul–Dec held at -20%")


def test_caption_case_4_degenerate_all_zero():
    df = _linspace_members(0.0, 3.0, 3, variable="temp")
    assert axis_caption(df, Axis(variable="precip")) == "unchanged"


def test_caption_case_4b_degenerate_one_non_zero_level():
    df = _lookup({"1": {"precip": [-20.0] * 12, "temp": [0.0] * 12}})
    assert axis_caption(df, Axis(variable="precip")) == "held at -20%"


def test_caption_case_4c_degenerate_several_levels():
    series = [-20.0 if m <= 6 else -10.0 for m in range(1, 13)]
    df = _lookup({"1": {"precip": series, "temp": [0.0] * 12}})
    caption = axis_caption(df, Axis(variable="precip"))
    assert caption.startswith("held at declared monthly offsets (weighted mean ")


def test_month_labels_are_circular_and_subsume_the_seasons():
    """DJF is a contiguous run in CIRCULAR order, which is what removes the need
    for a season table."""
    df = _linspace_members(-30.0, 30.0, 3, months=(12, 1, 2), held=0.0)
    assert axis_caption(df, Axis(variable="precip")).startswith("mean change over DJF;")

    df = _linspace_members(-30.0, 30.0, 3, months=(1, 3, 7), held=0.0)
    assert axis_caption(df, Axis(variable="precip")).startswith(
        "mean change over Jan, Mar, Jul;"
    )


def test_temperature_levels_carry_their_own_unit():
    df = _linspace_members(0.0, 3.0, 3, months=(1, 2, 3), held=1.0, variable="temp")
    assert axis_caption(df, Axis(variable="temp")) == (
        "mean change over JFM; Apr–Dec held at +1 °C"
    )


# ---------------------------------------------------------------------------
# V22 — the cross-axis rule no per-field validator can reach
# ---------------------------------------------------------------------------


def test_no_declaration_yields_the_default_surface():
    assert parse_surfaces({}) == [DEFAULT_SURFACE]
    assert parse_surfaces({"reporting": None}) == [DEFAULT_SURFACE]
    assert parse_surfaces({"reporting": {"surfaces": None}}) == [DEFAULT_SURFACE]
    assert parse_surfaces({"reporting": {"surfaces": []}}) == [DEFAULT_SURFACE]


def test_duplicate_axis_variable_refused():
    """V22: both values are individually inside the closed enum and only the
    PAIR is illegal, which is how a closed key set plus a closed value enum
    still admitted a declaration no conforming implementation could serve."""
    config = {
        "reporting": {
            "surfaces": [
                {"id": "bad", "x": {"variable": "temp"}, "y": {"variable": "temp"}}
            ]
        }
    }
    with pytest.raises(DuplicateAxisVariableError, match="bad"):
        parse_surfaces(config)


def test_orientation_reversal_admitted():
    """V22's twin, and the one that must NOT be refused: nothing constrains
    which variable takes which axis."""
    config = {
        "reporting": {
            "surfaces": [
                {
                    "id": "flipped",
                    "x": {"variable": "precip"},
                    "y": {"variable": "temp"},
                }
            ]
        }
    }
    (surface,) = parse_surfaces(config)
    assert surface.x.variable == "precip"
    assert surface.y.variable == "temp"


@pytest.mark.parametrize(
    "surface",
    [
        {"id": "x", "x": {"variable": "temp"}, "y": {"variable": "precip"}, "z": 1},
        {"id": "x", "x": {"variable": "wind"}, "y": {"variable": "precip"}},
        {"id": "x", "x": {"variable": "temp", "typo": 1}, "y": {"variable": "precip"}},
        {
            "id": "x",
            "x": {"variable": "temp", "statistic": "max"},
            "y": {"variable": "precip"},
        },
        {
            "id": "x",
            "x": {"variable": "temp", "months": [0, 1]},
            "y": {"variable": "precip"},
        },
        {
            "id": "x",
            "x": {"variable": "temp", "months": [1, 1]},
            "y": {"variable": "precip"},
        },
        {
            "id": "x",
            "x": {"variable": "temp", "months": []},
            "y": {"variable": "precip"},
        },
        {"id": "Bad Id", "x": {"variable": "temp"}, "y": {"variable": "precip"}},
    ],
)
def test_malformed_declarations_are_refused(surface):
    """R11 Q7's posture: a typo INSIDE a declaration is refused, not ignored.

    `statistic: max` is the substantive one — a non-affine statistic breaks the
    evenly-spaced guarantee, so the closed vocabulary is the static half of the
    same claim the rectilinearity postcondition checks dynamically.
    """
    with pytest.raises(SurfaceDeclarationError):
        parse_surfaces({"reporting": {"surfaces": [surface]}})


def test_duplicate_surface_ids_are_refused():
    config = {
        "reporting": {
            "surfaces": [
                {"id": "a", "x": {"variable": "temp"}, "y": {"variable": "precip"}},
                {"id": "a", "x": {"variable": "precip"}, "y": {"variable": "temp"}},
            ]
        }
    }
    with pytest.raises(SurfaceDeclarationError, match="twice"):
        parse_surfaces(config)


# ---------------------------------------------------------------------------
# V18 — the partition, and the INCOMPLETE case that looks plausible
# ---------------------------------------------------------------------------


def _indicators(st_ids, metrics=("q_annual_mean",)):
    rows = [
        {
            "metric": metric,
            "location": "101",
            "st_id": st_id,
            "rlz_id": 1,
            "value": 1.0,
        }
        for st_id in st_ids
        for metric in metrics
    ]
    return pd.DataFrame(rows)


def test_join_partitions_the_baseline_out():
    lookup = _linspace_members(-30.0, 30.0, 3)
    indicators = _indicators(["0", "1", "2", "3"])
    joined = join_axes(indicators, lookup, DEFAULT_SURFACE)

    assert set(joined.baseline_df["st_id"]) == {"0"}
    assert set(joined.surface_df["st_id"]) == {"1", "2", "3"}
    assert joined.key_width == 1
    assert set(joined.axes) == {"temp", "precip"}
    assert list(joined.surface_df["precip_change"]) == [-30.0, 0.0, 30.0]


def test_missing_lookup_member_refused():
    """V18's load-bearing half (D28 check b).

    Checks a and c BOTH pass here: every id the table carries is in the lookup,
    the only absent id is still the baseline, and the surface is non-empty. So
    without check b this returns a plausible-looking response surface that is
    silently missing a grid cell — a worse failure than a mis-keyed join, which
    at least produces a visibly wrong shape.
    """
    lookup = _linspace_members(-30.0, 30.0, 3)
    indicators = _indicators(["0", "1", "2"])  # member 3 never reduced
    with pytest.raises(SurfaceMemberMismatchError, match="'3'"):
        join_axes(indicators, lookup, DEFAULT_SURFACE)


def test_unknown_indicator_member_refused():
    lookup = _linspace_members(-30.0, 30.0, 3)
    indicators = _indicators(["0", "1", "2", "3", "9"])
    with pytest.raises(BaselinePartitionError, match="'9'"):
        join_axes(indicators, lookup, DEFAULT_SURFACE)


def test_a_mis_keyed_join_does_not_read_as_all_baseline():
    """The failure mode the absence-means-baseline encoding creates.

    An indicator table read with inferred dtypes loses the padding, so `01`
    becomes `1` and matches nothing — and the result is not an empty frame, it is
    "every row is the baseline", which is a shape the partition is DESIGNED to
    produce. `join_axes` re-pads both key columns before partitioning, so a
    caller who loaded the frame some other way is repaired rather than silently
    mis-partitioned.
    """
    lookup = _linspace_members(-30.0, 30.0, 12)
    lookup["st_id"] = lookup["st_id"].str.zfill(2)
    indicators = _indicators([str(m) for m in range(0, 13)])  # unpadded
    joined = join_axes(indicators, lookup, DEFAULT_SURFACE)
    assert joined.key_width == 2
    assert set(joined.baseline_df["st_id"]) == {"00"}
    assert len(joined.surface_df) == 12


def test_an_empty_surface_partition_is_refused():
    """Check c's residue: an EMPTY lookup satisfies check b vacuously."""
    lookup = _linspace_members(-30.0, 30.0, 3).iloc[0:0]
    with pytest.raises(LookupKeyWidthError):
        join_axes(_indicators(["0"]), lookup, DEFAULT_SURFACE)


def test_mixed_key_widths_are_refused():
    lookup = _linspace_members(-30.0, 30.0, 3)
    lookup.loc[lookup["st_id"] == "1", "st_id"] = "01"
    with pytest.raises(LookupKeyWidthError):
        key_width(lookup)


def test_padding_happens_only_in_the_join():
    """`derive_axis` reads ONE table, so its index is whatever the lookup holds.

    Said explicitly because the obvious implementer's error is to pad defensively
    in both places, which double-pads any consumer that composes them.
    """
    lookup = _linspace_members(-30.0, 30.0, 12)
    lookup["st_id"] = lookup["st_id"].str.zfill(2)
    assert list(derive_axis(lookup, Axis(variable="precip")).values.index) == [
        f"{m:02d}" for m in range(1, 13)
    ]


def test_a_declared_surface_names_the_frame_not_the_columns():
    """Two surfaces from one experiment differ in magnitude and label, never in
    the column a consumer plots."""
    lookup = _linspace_members(-30.0, 30.0, 3, months=(1, 2, 3), held=0.0)
    indicators = _indicators(["0", "1", "2", "3"])
    surface = Surface(
        id="jfm",
        x=Axis(variable="temp"),
        y=Axis(variable="precip", months=(1, 2, 3)),
    )
    joined = join_axes(indicators, lookup, surface)
    assert "precip_change" in joined.surface_df.columns
    assert "temp_change" in joined.surface_df.columns
