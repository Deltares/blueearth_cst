"""The WF0 figure filename grammar, pinned against the rule that defines it.

``dev/working/wf0-figure-filename-rule.md`` is the agreement; this module is
what keeps the code on it. The examples in its "Examples" section are asserted
VERBATIM below, so a drift in either direction shows up as a failure naming the
exact string that moved.

The vocabulary is deliberately CLOSED — an unknown ``plot_context`` or
``spatial_scope`` raises rather than producing a file. That is what makes the
Snakefile's declaration and the plotter's writes provably the same set: both go
through :func:`figure_filename`, so a typo cannot make one write
``era5_precip_annual_ts_basin_avg.png`` while the other declares
``..._basin_average.png``.
"""

from __future__ import annotations

import pytest

from blueearth_cst.climate_analysis.climate_figures import (
    MAP_EXTENT,
    source_climate_vars,
    source_figure_names,
)
from blueearth_cst.climate_analysis.compare_sources import comparison_figure_names
from blueearth_cst.climate_analysis.figure_naming import (
    COMPARISON_SCOPE,
    FIXED_SPATIAL_SCOPES,
    PLOT_CONTEXTS,
    figure_filename,
    map_spatial_scope,
    subbasin_scope,
)

#: Straight from the rule's "Examples" block.
DOCUMENTED_EXAMPLES = [
    "era5_precip_annual_ts_basin_avg.png",
    "era5_temp_annual_clim_map_basin_ext.png",
    "chirps_precip_monthly_box_basin_avg.png",
    "era5_precip_annual_ts_subbasin_1010_avg.png",
    "comparison_precip_annual_ts_basin_avg.png",
    "comparison_temp_monthly_box_basin_avg.png",
]


def test_the_documented_examples_are_producible():
    """Every example in the rule can be built from the vocabulary."""
    built = [
        figure_filename("era5", "precip", "annual_ts", "basin_avg"),
        figure_filename("era5", "temp", "annual_clim_map", "basin_ext"),
        figure_filename("chirps", "precip", "monthly_box", "basin_avg"),
        figure_filename("era5", "precip", "annual_ts", subbasin_scope("1010")),
        figure_filename(COMPARISON_SCOPE, "precip", "annual_ts", "basin_avg"),
        figure_filename(COMPARISON_SCOPE, "temp", "monthly_box", "basin_avg"),
    ]
    assert built == DOCUMENTED_EXAMPLES


def test_the_grammar_has_four_fields_in_order():
    name = figure_filename("era5", "precip", "monthly_box", "basin_avg")
    dataset_scope, rest = name.split("_", 1)
    assert dataset_scope == "era5"
    assert rest == "precip_monthly_box_basin_avg.png"


def test_variables_are_never_abbreviated():
    """`precip`, `temp`, `pet` in full — the rule says so explicitly."""
    for name in source_figure_names("era5"):
        assert any(f"_{var}_" in name for var in ("precip", "temp", "pet"))


@pytest.mark.parametrize("context", sorted(PLOT_CONTEXTS))
def test_every_declared_context_is_usable(context):
    assert figure_filename("era5", "precip", context, "basin_avg").endswith(".png")


@pytest.mark.parametrize("scope", sorted(FIXED_SPATIAL_SCOPES))
def test_every_declared_spatial_scope_is_usable(scope):
    assert figure_filename("era5", "precip", "annual_ts", scope).endswith(".png")


def test_an_unknown_context_is_refused():
    """The vocabulary is closed; a new token is added to it, not spelled ad hoc."""
    with pytest.raises(ValueError, match="unknown plot_context"):
        figure_filename("era5", "precip", "yearly_line", "basin_avg")


def test_an_unknown_spatial_scope_is_refused():
    with pytest.raises(ValueError, match="unknown spatial_scope"):
        figure_filename("era5", "precip", "annual_ts", "catchment_avg")


def test_an_id_scope_is_accepted_for_any_id():
    assert subbasin_scope(1010) == "subbasin_1010_avg"
    assert subbasin_scope("Upper Reach") == "subbasin_upper_reach_avg"
    assert figure_filename(
        "era5", "precip", "annual_ts", subbasin_scope("x1")
    ).endswith("subbasin_x1_avg.png")


def test_an_empty_id_is_refused():
    with pytest.raises(ValueError, match="id is empty"):
        subbasin_scope("  ")


# --- the map's scope tracks how the map is FRAMED -----------------------------


def test_map_scope_follows_the_extent_policy():
    """Derived, never hardcoded — MAP_EXTENT has already flipped once."""
    assert map_spatial_scope("basin") == "basin_ext"
    assert map_spatial_scope("raster") == "source_ext"


def test_the_source_map_is_named_for_its_current_framing():
    expected = map_spatial_scope(MAP_EXTENT["source"])
    maps = [n for n in source_figure_names("era5") if "annual_clim_map" in n]
    assert maps and all(n.endswith(f"{expected}.png") for n in maps)


# --- what each family declares ------------------------------------------------


def test_a_precipitation_only_source_is_named_for_precipitation_only():
    names = source_figure_names("chirps", variables=source_climate_vars("chirps"))
    assert all("_precip_" in name for name in names)
    assert all(name.startswith("chirps_") for name in names)


def test_the_two_families_use_different_plot_contexts_for_monthly():
    """A box plot and a line are different figures, so they are named apart."""
    per_source = [n for n in source_figure_names("era5") if "monthly" in n]
    comparison = [n for n in comparison_figure_names(["precip"]) if "monthly" in n]
    assert all("monthly_box" in n for n in per_source)
    assert all("monthly_clim_line" in n for n in comparison)


def test_the_declared_source_set_grows_with_the_spatial_scopes():
    basin = source_figure_names("era5", spatial_scopes=("basin_avg",))
    both = source_figure_names(
        "era5", spatial_scopes=("basin_avg", subbasin_scope("1010"))
    )
    # One extra file per AGGREGATED kind per variable; the map is drawn once.
    assert len(both) - len(basin) == len(source_climate_vars("era5")) * 2
    assert set(basin) < set(both)
