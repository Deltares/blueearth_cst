"""Tests for the Workflow 1 spatial-foundation config contract."""

from __future__ import annotations

import pytest

from blueearth_cst.spatial.config import (
    DEFAULT_GAUGE_SNAP_TOLERANCE_M,
    DEFAULT_MAX_SUBBASINS_PER_BASIN,
    DEFAULT_RIVER_UPAREA_KM2,
    parse_spatial_config,
    resolve_gauge_points_path,
)


def _basin_config(**updates):
    config = {
        "region": "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        "resolution": 0.00833,
    }
    config.update(updates)
    return config


def test_parse_spatial_config_applies_documented_defaults():
    """Minimal existing basin configs gain the neutral spatial defaults."""
    config = parse_spatial_config(_basin_config(), {})

    assert config.region == {"subbasin": [9.666, 0.4476], "uparea": 100}
    assert config.resolution == pytest.approx(0.00833)
    assert config.hydrography == "merit_hydro_ihu"
    assert config.basin_index == "merit_hydro_index"
    assert config.gauge_points_path is None
    assert config.max_subbasins_per_basin == DEFAULT_MAX_SUBBASINS_PER_BASIN
    assert config.gauge_snap_tolerance_m == DEFAULT_GAUGE_SNAP_TOLERANCE_M
    assert config.river_uparea_km2 == DEFAULT_RIVER_UPAREA_KM2
    assert config.sources.rivers == "rivers_lin2019_v1"
    assert config.sources.lulc == "vito"
    assert config.sources.lai == "modis_lai"
    assert config.sources.soil == "soilgrids"


def test_parse_spatial_config_accepts_explicit_basin_settings():
    """Project-specific ceilings, tolerances, and sources stay config-driven."""
    config = parse_spatial_config(
        _basin_config(
            region={"basin": [10, 20]},
            hydrography="custom_hydro",
            basin_index=None,
            gauge_points="C:/observations/gauges.csv",
            automatic_subbasins={"max_per_basin": 8},
            gauge_snap_tolerance_m=2500,
            river_uparea_km2=15,
            spatial_sources={
                "rivers": "custom_rivers",
                "lulc": "custom_lulc",
                "lai": "custom_lai",
                "soil": "custom_soil",
            },
        ),
        {},
    )

    assert config.region == {"basin": [10, 20]}
    assert config.hydrography == "custom_hydro"
    assert config.basin_index is None
    assert config.gauge_points_path == "C:/observations/gauges.csv"
    assert config.max_subbasins_per_basin == 8
    assert config.gauge_snap_tolerance_m == 2500
    assert config.river_uparea_km2 == 15
    assert config.sources.rivers == "custom_rivers"


@pytest.mark.parametrize("legacy_unset", [None, "None"])
def test_gauge_points_supports_legacy_unset_spellings(legacy_unset):
    """Legacy unset values do not become input paths."""
    assert (
        resolve_gauge_points_path(
            {"gauge_points": None}, {"output_locations": legacy_unset}
        )
        is None
    )


def test_legacy_output_locations_warns_and_remains_compatible():
    """One-release compatibility is explicit rather than a silent fallback."""
    with pytest.warns(FutureWarning, match="shared.basin.gauge_points"):
        path = resolve_gauge_points_path(
            {}, {"output_locations": "C:/observations/gauges.csv"}
        )

    assert path == "C:/observations/gauges.csv"


def test_matching_canonical_and_legacy_gauge_paths_are_allowed():
    """A staged config migration can carry both keys when they agree."""
    path = resolve_gauge_points_path(
        {"gauge_points": "C:/observations/gauges.csv"},
        {"output_locations": "C:\\observations\\gauges.csv"},
    )

    assert path == "C:/observations/gauges.csv"


def test_conflicting_gauge_paths_fail_loudly():
    """No precedence rule can silently select the wrong gauge file."""
    with pytest.raises(ValueError, match="Conflicting gauge-point paths"):
        resolve_gauge_points_path(
            {"gauge_points": "C:/observations/new.csv"},
            {"output_locations": "C:/observations/old.csv"},
        )


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"region": "not a mapping"}, ValueError, "region"),
        ({"region": {"bbox": [0, 0, 1, 1]}}, ValueError, "basin.*subbasin"),
        ({"resolution": 0}, ValueError, "resolution"),
        ({"automatic_subbasins": {"max_per_basin": 0}}, ValueError, "max_per_basin"),
        ({"automatic_subbasins": {"max_per_basin": 100}}, ValueError, "<= 99"),
        # ADR 0003 §11: the OLD key must be rejected BY NAME, not ignored.
        ({"automatic_subbasins": {"max_count": 20}}, ValueError, "max_per_basin"),
        ({"gauge_snap_tolerance_m": -1}, ValueError, "tolerance"),
        ({"gauge_points": 123}, TypeError, "path string"),
        ({"spatial_sources": {"soil": ""}}, TypeError, "soil"),
    ],
)
def test_invalid_spatial_config_fails_at_parse_time(updates, error, match):
    """Invalid strategy bounds and source names fail before the DAG executes."""
    with pytest.raises(error, match=match):
        parse_spatial_config(_basin_config(**updates), {})
