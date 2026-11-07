"""Public forcing resolution preserves scientific recipes without guessing."""

import pytest

from blueearth_cst.shared.forcing_recipe import (
    climate_forcing_summary,
    resolve_climate_forcing,
)


def test_legacy_null_selection_summary_reports_inference():
    rows = climate_forcing_summary(
        {"sources": ["chirps"], "selected": None}, required=True
    )
    assert rows[0] == "Precipitation chirps (inferred)"


@pytest.mark.parametrize("value", [None, "", 1, "bad/path"])
def test_invalid_temperature_source_names_public_path(value):
    with pytest.raises(ValueError, match=r"climate\.forcing\.temp_source"):
        resolve_climate_forcing(
            {"sources": ["era5"], "forcing": {"temp_source": value}}, required=True
        )


@pytest.mark.parametrize("source", ["era5", "chirps", "chirps_global"])
def test_public_legacy_inferred_equivalence(source):
    cases = [
        {"sources": [source]},
        {"sources": [source], "forcing": {"precip_source": source}},
        {"sources": [source], "selected": source},
        {
            "sources": [source],
            "forcing": {"precip_source": source, "temp_source": "era5"},
        },
    ]
    resolved = [
        resolve_climate_forcing(case, required=True, notices=False) for case in cases
    ]
    assert all(item == resolved[0] for item in resolved)
    assert len({recipe.canonical_bytes() for _, recipe in resolved}) == 1
    assert len({recipe.extraction_bytes() for _, recipe in resolved}) == 1


@pytest.mark.parametrize("forcing", [None, {}])
def test_comparison_does_not_infer(forcing):
    climate = {"sources": ["chirps"]}
    if forcing is not None:
        climate["forcing"] = forcing
    result, recipe = resolve_climate_forcing(climate, required=False)
    assert result["selected"] is None
    assert recipe is None


@pytest.mark.parametrize(
    "climate",
    [
        {"sources": ["era5", "chirps"]},
        {"sources": []},
        {"sources": ["era5", "era5"]},
        {"sources": ["era5", "typo"]},
        {"sources": ["era5"], "forcing": None},
        {"sources": ["era5"], "forcing": {"precip_source": None}},
        {"sources": ["era5"], "forcing": {"temp_souce": "era5"}},
        {"sources": ["era5"], "selected": "era5", "forcing": {"precip_source": "era5"}},
        {
            "sources": ["era5"],
            "forcing": {"temp_source": "era5", "temp_pet_source": "era5"},
        },
    ],
)
def test_invalid_public_values_refused(climate):
    with pytest.raises(ValueError):
        resolve_climate_forcing(climate, required=True, notices=False)
