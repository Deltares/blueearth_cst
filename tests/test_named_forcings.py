"""Named historical forcing recipes resolve to the legacy single-recipe form."""

from __future__ import annotations

import pytest

from blueearth_cst.shared import config_composition as cc
from blueearth_cst.shared.named_forcings import (
    HistoricalLayout,
    historical_forcing_ids,
    retained_run_names,
)
from tests.test_config_composition import write_split

CHIRPS = {
    "precip_source": "chirps",
    "temp_source": "era5",
    "orography_source": "era5_orography",
}


def _named(tmp_path, *, climate=None, build=None, simulate=None):
    return write_split(
        tmp_path,
        sections={
            "basin": {"region": "x"},
            "climate": climate
            or {
                "sources": ["era5", "chirps"],
                "forcings": {"era5": {"precip_source": "era5"}, "chirps": CHIRPS},
                "default_forcing": "chirps",
            },
        },
        bodies={
            "build_model": build or {},
            "generate_scenarios": {},
            "simulate_system": simulate or {},
        },
    )


@pytest.mark.parametrize("entry", ["build_model", "generate_scenarios"])
def test_default_recipe_becomes_the_legacy_internal_form(tmp_path, entry):
    composed = cc.load_composed_config(_named(tmp_path), entry=entry)
    climate = composed["climate"]
    assert climate["selected"] == "chirps"
    assert climate["forcing_id"] == "chirps"
    assert climate["default_forcing"] == "chirps"
    assert set(climate["forcings"]) == {"era5", "chirps"}
    assert climate["forcings"]["era5"] == {"selected": "era5"}
    # The resolved default equals what the single-recipe form produces.
    legacy = write_split(
        tmp_path / "legacy",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5", "chirps"], "forcing": CHIRPS},
        },
        bodies={"build_model": {}, "generate_scenarios": {}},
    )
    single = cc.load_composed_config(legacy, entry=entry)["climate"]
    assert {k: climate.get(k) for k in ("selected", "forcing")} == {
        k: single.get(k) for k in ("selected", "forcing")
    }


def test_simulation_selects_a_named_recipe(tmp_path):
    path = _named(tmp_path, simulate={"historical_forcing": "era5"})
    climate = cc.load_composed_config(path, entry="simulate_system")["climate"]
    assert (climate["selected"], climate["forcing_id"]) == ("era5", "era5")
    path = _named(tmp_path / "d", simulate={"historical_forcing": "default"})
    assert (
        cc.load_composed_config(path, entry="simulate_system")["climate"]["forcing_id"]
        == "chirps"
    )


@pytest.mark.parametrize(
    "climate,match",
    [
        ({"forcings": {"era5": {}}}, "declared together"),
        (
            {"forcings": {"era5": {}}, "default_forcing": "x"},
            "not one of climate.forcings",
        ),
        (
            {"forcings": {"default": {}}, "default_forcing": "default"},
            "reserved",
        ),
        (
            {"forcings": {"Era5": {}}, "default_forcing": "Era5"},
            "must match",
        ),
        (
            {
                "forcings": {"era5": {}},
                "default_forcing": "era5",
                "forcing": {"precip_source": "era5"},
            },
            "declare only one form",
        ),
        (
            {
                "forcings": {
                    "a": {"precip_source": "era5"},
                    "b": {"precip_source": "era5"},
                },
                "default_forcing": "a",
            },
            "repeats the recipe",
        ),
    ],
)
def test_ambiguous_or_malformed_named_forcings_fail(tmp_path, climate, match):
    path = _named(tmp_path, climate={"sources": ["era5", "chirps"], **climate})
    with pytest.raises(ValueError, match=match):
        cc.load_composed_config(path, entry="build_model")


@pytest.mark.parametrize(
    "requested,selected,forcing_id",
    [
        (None, "chirps", "chirps"),
        ("default", "chirps", "chirps"),
        ("era5", "era5", "era5"),
    ],
)
def test_wf4_loader_resolves_the_same_recipe_as_composition(
    tmp_path, requested, selected, forcing_id
):
    from blueearth_cst.experiment.simulation_runner import simulation_settings

    simulate = {"operation": "simulate-and-indicators"}
    if requested is not None:
        simulate["historical_forcing"] = requested
    path = _named(tmp_path, simulate=simulate)
    project, _ = simulation_settings(path)
    climate = project["climate"]
    assert (climate["selected"], climate["forcing_id"]) == (selected, forcing_id)
    assert climate["default_forcing"] == "chirps"
    composed = cc.load_composed_config(path, entry="simulate_system")["climate"]
    assert {k: climate.get(k) for k in ("selected", "forcing", "forcing_id")} == {
        k: composed.get(k) for k in ("selected", "forcing", "forcing_id")
    }
    # Indicators-only skips forcing validation but still binds the recipe ID.
    path = _named(
        tmp_path / "ind",
        simulate={**simulate, "operation": "indicators-only"},
    )
    assert simulation_settings(path)[0]["climate"]["forcing_id"] == forcing_id


def test_wf4_loader_keeps_legacy_climate_unchanged(tmp_path):
    from blueearth_cst.experiment.simulation_runner import simulation_settings

    path = write_split(
        tmp_path,
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5"], "forcing": {"precip_source": "era5"}},
        },
        bodies={"simulate_system": {"operation": "simulate-and-indicators"}},
    )
    climate = simulation_settings(path)[0]["climate"]
    assert climate["selected"] == "era5"
    assert not {"forcing_id", "forcings", "default_forcing"} & set(climate)


def test_unknown_simulation_selection_fails(tmp_path):
    path = _named(tmp_path, simulate={"historical_forcing": "eobs"})
    with pytest.raises(ValueError, match="unknown forcing 'eobs'"):
        cc.load_composed_config(path, entry="simulate_system")


def test_legacy_projects_only_offer_default(tmp_path):
    path = write_split(
        tmp_path,
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5"], "forcing": {"precip_source": "era5"}},
        },
        bodies={"simulate_system": {"historical_forcing": "era5"}},
    )
    with pytest.raises(ValueError, match="requires climate.forcings"):
        cc.load_composed_config(path, entry="simulate_system")
    path = write_split(
        tmp_path / "ok",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5"], "forcing": {"precip_source": "era5"}},
        },
        bodies={"build_model": {}},
    )
    composed = cc.load_composed_config(path, entry="build_model")
    assert not {"forcing_id", "forcings"} & set(composed["climate"])


def test_build_model_requests_and_retained_names(tmp_path):
    climate = {
        "sources": ["era5", "chirps"],
        "forcings": {"era5": {"precip_source": "era5"}, "chirps": CHIRPS},
        "default_forcing": "era5",
    }
    assert historical_forcing_ids(climate, {}) == ("era5",)
    assert historical_forcing_ids(
        climate, {"historical_forcings": ["chirps", "era5"]}
    ) == (
        "chirps",
        "era5",
    )
    with pytest.raises(ValueError, match="unknown forcing"):
        historical_forcing_ids(climate, {"historical_forcings": ["eobs"]})
    with pytest.raises(ValueError, match="requires climate.forcings"):
        historical_forcing_ids({"sources": ["era5"]}, {"historical_forcings": ["era5"]})
    assert retained_run_names("v2", ("era5", "chirps")) == {
        "era5": "v2_era5",
        "chirps": "v2_chirps",
    }
    assert retained_run_names("v2", None) == {None: "v2"}
    with pytest.raises(ValueError, match="historical_run.name is required"):
        retained_run_names(None, ("era5", "chirps"))
    path = _named(tmp_path, build={"historical_forcings": ["era5", "chirps"]})
    with pytest.raises(ValueError, match="historical_run.name is required"):
        cc.load_composed_config(path, entry="build_model")


def test_layouts_keep_legacy_paths_and_isolate_named_runs():
    legacy = HistoricalLayout(None)
    assert (legacy.forcing, legacy.toml, legacy.run_dir) == (
        "forcing/inmaps_historical.nc",
        "wflow_sbm.toml",
        "run_default",
    )
    named = HistoricalLayout("chirps")
    assert (named.forcing, named.toml, named.end_state) == (
        "forcing/chirps/inmaps_historical.nc",
        "wflow_sbm_chirps.toml",
        "run_chirps/outstate/outstates.nc",
    )


def test_runner_summary_names_the_active_recipe(tmp_path):
    import scripts.run_workflows as rw

    rows = rw._consumer_forcing_rows(str(_named(tmp_path)), "build_model")
    assert rows[0] == "> Historical forcing: chirps (default)"
    assert any("CHIRPS" in row for row in rows)
