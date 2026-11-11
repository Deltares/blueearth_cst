"""Local plotting choices remain separate from numerical and shared settings."""

from copy import deepcopy

import pytest

from blueearth_cst.shared.workflow_plotting import (
    LEGACY_PLOTTING_KEYS,
    PLOTTING_DEFAULTS,
    normalize_workflow_plotting,
    workflow_plotting,
)


@pytest.mark.parametrize("owner", PLOTTING_DEFAULTS)
def test_defaults_and_explicit_overrides_are_workflow_local(owner):
    assert normalize_workflow_plotting(owner, {}) == {}
    assert workflow_plotting(owner, {}) == PLOTTING_DEFAULTS[owner]
    body = {
        "plotting": dict.fromkeys(PLOTTING_DEFAULTS[owner], False),
        "numerical": {"threshold": 42},
    }
    original = deepcopy(body)
    normalized = normalize_workflow_plotting(owner, body)
    assert not any(normalized["plotting"].values())
    assert normalized["numerical"] == original["numerical"]
    assert body == original


@pytest.mark.parametrize(
    "owner", ["analyze_projections", "generate_scenarios", "simulate_system"]
)
def test_other_workflows_do_not_acquire_empty_plotting_sections(owner):
    assert normalize_workflow_plotting(owner, {"key": "value"}) == {"key": "value"}


@pytest.mark.parametrize(
    "owner,path,key",
    [
        (owner, path, key)
        for owner, aliases in LEGACY_PLOTTING_KEYS.items()
        for path, key in aliases.items()
    ],
)
def test_legacy_aliases_warn_and_normalize_without_retaining_old_keys(owner, path, key):
    body = (
        {path[0]: {path[1]: True, "budyko": False}}
        if len(path) == 2
        else {path[0]: True}
    )
    with pytest.warns(DeprecationWarning, match=f"plotting.{key}"):
        normalized = normalize_workflow_plotting(owner, body)
    assert normalized["plotting"][key] is True
    if len(path) == 2:
        assert normalized["evaluation"] == {"budyko": False}
    else:
        assert path[0] not in normalized


@pytest.mark.parametrize(
    "owner,key",
    [(owner, key) for owner, defaults in PLOTTING_DEFAULTS.items() for key in defaults],
)
@pytest.mark.parametrize("value", [None, "false", 0, 1, [], {}])
def test_local_options_require_booleans(owner, key, value):
    with pytest.raises(ValueError, match=rf"{owner}.plotting.{key} must be a boolean"):
        normalize_workflow_plotting(owner, {"plotting": {key: value}})


@pytest.mark.parametrize("owner", PLOTTING_DEFAULTS)
@pytest.mark.parametrize("value", [None, False, "all", []])
def test_plotting_requires_a_mapping(owner, value):
    with pytest.raises(ValueError, match="plotting must be a mapping"):
        normalize_workflow_plotting(owner, {"plotting": value})


@pytest.mark.parametrize("owner", PLOTTING_DEFAULTS)
def test_unknown_options_and_project_annotation_are_rejected(owner):
    with pytest.raises(ValueError, match="Unknown.*plotting options"):
        normalize_workflow_plotting(owner, {"plotting": {"unexpected": True}})
    with pytest.raises(ValueError, match="annotation belongs in the project file"):
        normalize_workflow_plotting(owner, {"plotting": {"annotation": "title"}})


@pytest.mark.parametrize(
    "owner,path,key",
    [
        (owner, path, key)
        for owner, aliases in LEGACY_PLOTTING_KEYS.items()
        for path, key in aliases.items()
    ],
)
def test_conflicting_new_and_legacy_options_refuse(owner, path, key):
    body = {path[0]: {path[1]: True}} if len(path) == 2 else {path[0]: True}
    body["plotting"] = {key: False}
    with pytest.raises(ValueError, match="Conflicting.*plotting"):
        normalize_workflow_plotting(owner, body)


@pytest.mark.parametrize(
    "owner,body",
    [
        ("analyze_climate", {"subbasin_figures": True, "location_figures": False}),
        (
            "build_model",
            {"evaluation": {"subbasin_plots": True, "station_plots": False}},
        ),
    ],
)
def test_conflicting_combined_location_aliases_refuse(owner, body):
    with pytest.raises(ValueError, match="Conflicting.*plotting.location"):
        normalize_workflow_plotting(owner, body)


def test_equal_new_and_legacy_declarations_warn_without_conflict():
    with pytest.warns(DeprecationWarning, match="forcing_plots"):
        normalized = normalize_workflow_plotting(
            "build_model", {"plotting": {"forcing": True}, "forcing_plots": True}
        )
    assert normalized["plotting"]["forcing"] is True


def test_composition_keeps_global_annotation_and_local_selection_separate(tmp_path):
    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.test_config_composition import MINIMAL_T1_SECTIONS, write_split

    path = write_split(
        tmp_path,
        sections=MINIMAL_T1_SECTIONS | {"plotting": {"annotation": "title"}},
        bodies={
            "analyze_climate": {"plotting": {"source": True}},
            "build_model": {"plotting": {"spatial": False}},
            "analyze_projections": {},
        },
    )
    composed = load_composed_config(path)
    assert composed["plotting"] == {"annotation": "title"}
    assert composed["workflows"]["analyze_climate"]["plotting"] == {
        "comparison": True,
        "source": True,
        "location": False,
    }
    assert composed["workflows"]["build_model"]["plotting"] == {
        "spatial": False,
        "forcing": False,
        "location": False,
    }
    assert "plotting" not in composed["workflows"]["analyze_projections"]


def test_composition_migrates_legacy_selection_and_preserves_numerical_evaluation(
    tmp_path,
):
    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.test_config_composition import write_split

    path = write_split(
        tmp_path,
        bodies={
            "build_model": {
                "forcing_plots": True,
                "evaluation": {"station_plots": True, "budyko": False},
            }
        },
    )
    with pytest.warns(DeprecationWarning):
        model = load_composed_config(path)["workflows"]["build_model"]
    assert model["plotting"] == {"spatial": True, "forcing": True, "location": True}
    assert model["evaluation"] == {"budyko": False}
    assert "forcing_plots" not in model
