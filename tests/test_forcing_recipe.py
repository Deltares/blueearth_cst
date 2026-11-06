"""Pure recipe admission and separation of extraction/preparation identities."""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from blueearth_cst.shared.forcing_recipe import (
    normalize_forcing_recipe,
    require_workflow_compatibility,
)


@pytest.mark.parametrize("source", ["era5", "chirps", "chirps_global"])
def test_omitted_and_explicit_defaults_share_identity(source):
    default = normalize_forcing_recipe(source)
    explicit = normalize_forcing_recipe(
        source,
        {
            "temp_pet_source": "era5",
            "orography_source": "era5_orography",
            "temp_correction": True,
            "press_correction": True,
            "pet": {"method": "debruin", "mode": "computed"},
        },
    )
    assert default == explicit == normalize_forcing_recipe(source, {})
    assert default.fingerprint == explicit.fingerprint
    assert default.extraction_fingerprint == explicit.extraction_fingerprint
    assert default.required_variables == (
        "precip",
        "temp",
        "temp_min",
        "temp_max",
        "press_msl",
        "kin",
        "kout",
    )


@pytest.mark.parametrize(
    "forcing,field",
    [
        ([], "mapping"),
        (False, "mapping"),
        ({"temperature_source": "era5"}, "unknown key"),
        ({1: True}, "unknown key"),
        ({"temp_correction": 1}, "temp_correction"),
        ({"temp_correction": "false"}, "temp_correction"),
        ({"press_correction": 0}, "press_correction"),
        ({"press_correction": None}, "press_correction"),
        ({"temp_pet_source": ""}, "temp_pet_source"),
        ({"temp_pet_source": 1}, "temp_pet_source"),
        ({"temp_pet_source": " era5"}, "temp_pet_source"),
        ({"orography_source": "x/y.nc"}, "orography_source"),
        ({"orography_source": "C:\\data\\dem.nc"}, "orography_source"),
        ({"orography_source": None}, "orography_source"),
        ({"pet": None}, "mapping"),
        ({"pet": []}, "mapping"),
        ({"pet": {"algorithm": "debruin"}}, "unknown key"),
        ({"pet": {"mode": "automatic"}}, "pet.mode"),
        ({"pet": {"mode": []}}, "pet.mode"),
        ({"pet": {"method": "penman-monteith_tdew"}}, "pet.method"),
        ({"pet": {"method": None}}, "pet.method"),
        ({"pet": {"source": "supplied"}}, "computed PET forbids"),
        ({"pet": {"mode": "direct"}}, "pet.source"),
        ({"pet": {"mode": "direct", "source": ""}}, "pet.source"),
        (
            {"pet": {"mode": "direct", "source": "pet", "method": None}},
            "direct PET forbids",
        ),
        (
            {"pet": {"mode": "direct", "source": "pet"}, "press_correction": False},
            "direct PET forbids",
        ),
        (
            {"pet": {"mode": "direct", "source": "pet"}, "press_correction": None},
            "direct PET forbids",
        ),
    ],
)
def test_invalid_scientific_settings_fail_instead_of_being_ignored(forcing, field):
    with pytest.raises(ValueError, match=field):
        normalize_forcing_recipe("era5", forcing)


@pytest.mark.parametrize(
    "source", ["eobs", "era5_daily_zarr", "new_precip", "", None, []]
)
def test_no_additional_precipitation_products(source):
    with pytest.raises(ValueError, match="climate.selected"):
        normalize_forcing_recipe(source)


def test_alternate_companion_requires_explicit_elevation_even_without_correction():
    with pytest.raises(ValueError, match="orography_source is required"):
        normalize_forcing_recipe(
            "chirps", {"temp_pet_source": "alternate", "temp_correction": False}
        )
    recipe = normalize_forcing_recipe(
        "chirps",
        {
            "temp_pet_source": "alternate",
            "orography_source": "alternate_dem",
            "pet": {"method": "makkink"},
        },
    )
    assert recipe.orography_source == "alternate_dem"
    assert recipe.companion_variables == (
        "temp",
        "temp_min",
        "temp_max",
        "press_msl",
        "kin",
    )
    assert "kout" not in recipe.required_variables


def test_direct_pet_reads_only_temperature_from_companion():
    recipe = normalize_forcing_recipe(
        "chirps", {"pet": {"mode": "direct", "source": "reference_pet"}}
    )
    assert recipe.pet_method is None
    assert recipe.press_correction is None
    assert recipe.companion_variables == ("temp",)
    assert recipe.required_variables == ("precip", "temp", "pet")
    assert recipe.extraction_dict()["pet_source"] == "reference_pet"


@pytest.mark.parametrize("source", ["era5", "chirps", "chirps_global"])
def test_pressure_only_changes_preparation_identity_not_generator_input(source):
    default = normalize_forcing_recipe(source)
    uncorrected = normalize_forcing_recipe(source, {"press_correction": False})
    assert default.fingerprint != uncorrected.fingerprint
    assert default.extraction_bytes() == uncorrected.extraction_bytes()
    assert default.extraction_fingerprint == uncorrected.extraction_fingerprint
    assert "press_correction" not in default.extraction_dict()


@pytest.mark.parametrize(
    "change", [{"temp_correction": False}, {"orography_source": "other_dem"}]
)
def test_same_source_extraction_does_not_gain_model_lapse_dependencies(change):
    raw = normalize_forcing_recipe("era5")
    changed = normalize_forcing_recipe("era5", change)
    assert raw.fingerprint != changed.fingerprint
    assert raw.extraction_bytes() == changed.extraction_bytes()
    assert "temp_correction" not in raw.extraction_dict()
    assert "orography_source" not in raw.extraction_dict()
    hybrid = normalize_forcing_recipe("chirps")
    changed_hybrid = normalize_forcing_recipe("chirps", change)
    assert hybrid.extraction_fingerprint != changed_hybrid.extraction_fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"temp_pet_source": "other", "orography_source": "other_dem"},
        {"pet": {"method": "makkink"}},
        {"pet": {"mode": "direct", "source": "pet"}},
    ],
)
def test_changed_extraction_inputs_invalidate_both_identities(change):
    default = normalize_forcing_recipe("era5")
    changed = normalize_forcing_recipe("era5", change)
    assert default.fingerprint != changed.fingerprint
    assert default.extraction_fingerprint != changed.extraction_fingerprint


def test_precipitation_and_supplied_pet_sources_are_identity_inputs():
    one = normalize_forcing_recipe(
        "chirps", {"pet": {"mode": "direct", "source": "pet_a"}}
    )
    other_pet = normalize_forcing_recipe(
        "chirps", {"pet": {"source": "pet_b", "mode": "direct"}}
    )
    other_precip = normalize_forcing_recipe(
        "chirps_global", {"pet": {"source": "pet_a", "mode": "direct"}}
    )
    assert len({r.fingerprint for r in (one, other_pet, other_precip)}) == 3
    assert len({r.extraction_fingerprint for r in (one, other_pet, other_precip)}) == 3


def test_canonical_serialization_is_stable_and_detached_from_mutable_inputs():
    settings = {"pet": {"method": "makkink"}, "temp_correction": False}
    recipe = normalize_forcing_recipe("era5", settings)
    reordered = normalize_forcing_recipe("era5", dict(reversed(list(settings.items()))))
    before = recipe.canonical_bytes()
    assert before == reordered.canonical_bytes()
    assert before.endswith(b"\n")
    assert json.loads(before)["schema_version"] == "forcing-recipe/1"
    assert recipe.fingerprint == hashlib.sha256(before).hexdigest()
    assert (
        recipe.extraction_fingerprint
        == hashlib.sha256(recipe.extraction_bytes()).hexdigest()
    )
    settings["pet"]["method"] = "debruin"
    recipe.to_dict()["pet"]["method"] = "debruin"
    recipe.extraction_dict()["required_variables"].clear()
    assert recipe.canonical_bytes() == before
    with pytest.raises(FrozenInstanceError):
        recipe.temp_correction = True


@pytest.mark.parametrize(
    "workflow",
    [
        "analyze_climate",
        "build_model",
        "analyze_projections",
        "generate_scenarios",
        "simulate_system",
    ],
)
def test_compatibility_guard_names_restriction_and_remedy(workflow):
    require_workflow_compatibility(normalize_forcing_recipe("era5"), workflow)
    direct = normalize_forcing_recipe(
        "era5", {"pet": {"mode": "direct", "source": "pet"}}
    )
    if workflow in {"generate_scenarios", "simulate_system"}:
        with pytest.raises(ValueError, match=f"{workflow}.*use computed PET"):
            require_workflow_compatibility(direct, workflow)
    else:
        require_workflow_compatibility(direct, workflow)
    with pytest.raises(ValueError, match="unknown workflow"):
        require_workflow_compatibility(direct, "simulate_systm")
