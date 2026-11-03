"""Named indicator sets: selection parsing, closed set schema and resolution."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.experiment import indicator_registry as registry
from blueearth_cst.experiment import indicator_sets as sets
from blueearth_cst.experiment.indicator_sets import IndicatorSetError
from blueearth_cst.indicators import loader
from blueearth_cst.shared.config_composition import normalize_indicator_settings
from blueearth_cst.shared.indicator_tables import VARIABLE_TOKENS


def test_old_scalar_settings_are_input_aliases_only():
    original = {"operation": "metrics-only", "metrics": ["gwr"]}
    assert normalize_indicator_settings(original) == {
        "operation": "indicators-only",
        "indicators": ["gwr"],
    }
    assert original == {"operation": "metrics-only", "metrics": ["gwr"]}
    with pytest.raises(ValueError, match="both old and new"):
        normalize_indicator_settings({"metrics": [], "indicators": []})


def _source(text, set_id="example_subset", bundled=False):
    return sets.SetSource(set_id, Path(f"{set_id}.yml"), bundled, textwrap.dedent(text))


def test_inline_sets_inherit_defaults_disable_inputs_and_snapshot(tmp_path):
    settings = {
        "indicator_sets": [
            {
                "set": "crop-water",
                "id": "crops",
                "summer_low_flow_days": {
                    "enabled": False,
                    "thresholds": {"baseline_quantile": {}},
                },
                "summer_longest_dry_spell": {"dry_day_mm": 2.0},
                "summer_et_limitation_days": {"ratio": 0.6},
            },
            {"set": "flow_regime", "q_mean_annual_90day_min": {"enabled": False}},
            {"set": "default"},
        ]
    }
    selection = sets.parse_selection(settings, tmp_path / "wf4.yml")
    snapshot = sets.write_selection_snapshot(selection, tmp_path / "selection.json")
    assert sets.read_selection_snapshot(snapshot) == selection
    resolved = sets.selected_sets(
        {"model": {"outvars": ["river discharge"]}},
        settings,
        selection,
        simulation_path=True,
        retained_variables=lambda: [],
        legacy_context="simulation_rule",
    )
    crops, flow, default = resolved
    parameters = {item.indicator: item.parameters for item in crops.indicators}
    assert parameters["summer_longest_dry_spell"] == {
        "season": "summer",
        "dry_day_mm": 2.0,
    }
    assert parameters["summer_et_limitation_days"] == {
        "season": "summer",
        "pet_min_mm": 1.0,
        "ratio": 0.6,
    }
    assert "summer_low_flow_days" not in parameters
    assert "q" not in sets.required_variables([crops])
    assert "q_mean_annual_90day_min" not in {i.indicator for i in flow.indicators}
    assert sets.calculation_records(default.indicators) == sets.default_calculations(
        ["q"]
    )


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"summer_low_flow_days": {"enabled": "false"}}, "boolean"),
        ({"typo": {}}, "unknown indicators"),
        (
            {"summer_longest_dry_spell": {"enabled": False, "dry_day_mm": -1}},
            "dry_day_mm",
        ),
        (
            {
                "summer_low_flow_days": {
                    "enabled": False,
                    "thresholds": {"baseline_quantile": {"years": None}},
                }
            },
            "years",
        ),
        ({"summer_longest_dry_spell": {"season": "winter"}}, "season must remain"),
        ({"summer_longest_dry_spell": {"typo": 1}}, "typo"),
    ],
)
def test_inline_rejects_malformed_disabled_entries_and_name_drift(
    tmp_path, patch, message
):
    selection = sets.parse_selection(
        {"indicator_sets": [{"set": "crop-water", **patch}]}, tmp_path / "wf4.yml"
    )
    with pytest.raises(ValueError, match=message):
        sets.resolve_set(selection.sets[0])


def test_flat_file_and_inline_have_same_calculation_identity(tmp_path):
    import yaml

    entry = {"set": "flow_regime", "q_mean_annual_90day_min": {"enabled": False}}
    inline = sets.parse_selection({"indicator_sets": [entry]}, tmp_path / "wf4.yml")
    file_source = sets.SetSource(
        "flow_regime",
        tmp_path / "flow.yml",
        False,
        yaml.safe_dump({"schema_version": 1, **entry}),
    )
    assert sets.calculation_records(
        sets.resolve_set(inline.sets[0]).indicators
    ) == sets.calculation_records(sets.resolve_set(file_source).indicators)


def test_inline_duplicate_set_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        sets.parse_selection(
            {"indicator_sets": ["crop-water", {"set": "crop-water"}]},
            tmp_path / "wf4.yml",
        )


def test_configured_default_keeps_response_driven_scope(tmp_path):
    settings = {
        "indicator_sets": [{"set": "default", "q_annual_mean": {"enabled": False}}]
    }
    selection = sets.parse_selection(settings, tmp_path / "wf4.yml")
    resolved = sets.selected_sets(
        {"model": {"outvars": ["river discharge"]}},
        settings,
        selection,
        simulation_path=True,
        retained_variables=lambda: [],
        legacy_context="simulation_rule",
    )
    assert sets.required_variables(resolved) == ["q"]
    assert "q_annual_mean" not in {i.indicator for i in resolved[0].indicators}


def test_inline_definition_conflicts(tmp_path):
    with pytest.raises(ValueError, match="also have"):
        sets.parse_selection(
            {
                "indicator_sets": [{"set": "crop-water"}],
                "indicator_set_definitions": {
                    "crop-water": {"config_path": "crop.yml"}
                },
            },
            tmp_path / "wf4.yml",
        )


def test_inline_default_rejects_boolean_schema_version(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        sets.parse_selection(
            {"indicator_sets": [{"set": "default", "schema_version": True}]},
            tmp_path / "wf4.yml",
        )


EXAMPLE = """
    schema_version: 1
    id: example_subset
    title: Example discharge subset
    description: Illustrative reuse, not an approved second scientific set.
    selection:
      indicators:
        - definition: q_annual_mean
        - definition: q_mean_annual_7day_min
    """

AUTHORED = """
    schema_version: 1
    id: example_subset
    title: t
    description: d
    selection:
      indicators:
        - id: high_flow
          title: High flow
          description: Alias of the annual p95.
          references: [Author 2020]
          calculation:
            method: annual_summary/1
            parameters: {variable: q, statistic: p95}
    """


def test_registry_covers_every_existing_definition_exactly():
    names = {
        item.name
        for token in VARIABLE_TOKENS.values()
        for item in registry.declarations([token])
    }
    optional = {item.name for item in registry.optional_declarations()}
    assert not names & optional
    assert set(registry.DEFINITIONS) == names | optional
    assert (len(names), len(optional)) == (18, 5)
    for definition, (method, parameters, _) in registry.DEFINITIONS.items():
        assert registry.validate_calculation(method, parameters)[0] == definition
        assert registry.declaration_for(definition).name == definition


def test_optional_definitions_stay_out_of_the_default_preset():
    path = sets.BUNDLED_ROOT / "default.yml"
    source = sets.SetSource("default", path, True, path.read_text(encoding="utf-8"))
    resolved = sets.resolve_set(source, ["q"])
    optional = {item.name for item in registry.optional_declarations()}
    assert not {item.indicator for item in resolved.indicators} & optional


def test_optional_reductions_match_hand_computed_values():
    time = pd.date_range("2001-01-02", periods=6, freq="D")
    frame = pd.DataFrame({"g": [1.0, 3.0, 2.0, 2.0, 5.0, 4.0]}, index=time)
    # One water-year block: rises 2 and 3, falls 1 and 1; the zero change is neither.
    assert registry._reduce_optional(frame, "rise_rate", "YE-SEP")["g"] == 2.5
    assert registry._reduce_optional(frame, "fall_rate", "YE-SEP")["g"] == 1.0
    assert registry.output_units(registry.declaration_for("q_mean_rise_rate")) == (
        "m3 s-1 d-1"
    )
    long = pd.DataFrame(
        {"g": np.arange(60, dtype=float)},
        index=pd.date_range("2001-01-01", periods=60, freq="D"),
    )
    # 30-day window over 0..59: means run 14.5 to 44.5 inside one block.
    assert registry._reduce_optional(long, "Q30day_max", "YE-SEP")["g"] == 44.5
    assert registry._reduce_optional(long, "Q30day_min", "YE-SEP")["g"] == 14.5
    assert np.isnan(registry._reduce_optional(long, "Q90day_min", "YE-SEP")["g"])


@pytest.mark.parametrize(
    ("method", "parameters"),
    [
        ("annual_summary/1", {"variable": "q", "statistic": "median"}),
        ("annual_summary/1", {"variable": "aet", "statistic": "mean"}),
        ("annual_summary/1", {"variable": "q"}),
        ("annual_rolling7_extreme/1", {"extreme": "min", "window": 7}),
        ("annual_min7_mean_ratio/1", {"variable": "q"}),
        ("unknown/1", {}),
    ],
)
def test_calculations_reject_unsupported_extra_or_missing_parameters(
    method, parameters
):
    with pytest.raises(ValueError):
        registry.validate_calculation(method, parameters)


def test_output_units_are_declared_not_inferred():
    units = {
        name: registry.output_units(registry.declaration_for(name))
        for name in registry.DEFINITIONS
    }
    assert units["q_baseflow_index"] == "1"
    assert units["precip_annual_total"] == units["aet_annual_total"] == "mm"
    assert units["pet_annual_total"] == "mm"
    assert units["snow_annual_max"] == "mm"
    assert units["overland_flow_annual_mean"] == units["q_annual_mean"] == "m3 s-1"


def test_default_set_resolves_legacy_pet_total_for_indicator_planning():
    legacy = registry.declaration_for("pet_annual_total")
    local = loader.local_declaration(legacy)
    assert (local.calculation_set, local.calculation_method) == (
        "default",
        "annual_summary/1",
    )
    assert local.required_responses.variable == "pet"


def test_default_preset_delegates_to_registry_declarations():
    path = sets.BUNDLED_ROOT / "default.yml"
    source = sets.SetSource("default", path, True, path.read_text(encoding="utf-8"))
    resolved = sets.resolve_set(source, ["q", "aet"])
    assert resolved.declarations == registry.declarations(["q", "aet"])
    assert all(item.equivalent_definition == "" for item in resolved.indicators)
    with pytest.raises(IndicatorSetError, match="no response variables"):
        sets.resolve_set(source, [])


def test_erfa_bundled_selection_resolves_all_seventy_indicators():
    selection = sets.parse_selection(
        {"operations": ["indicators"], "indicator_sets": ["eflow-erfa"]},
        "simulate_system.yml",
    )
    resolved = sets.resolve_set(selection.sets[0])
    assert resolved.set_id == "eflow-erfa"
    assert len(resolved.indicators) == 70
    assert len({item.indicator for item in resolved.indicators}) == 70
    assert {item.calculation_set for item in resolved.indicators} == {"eflow-erfa"}
    assert {
        item.declaration.required_responses.variable for item in resolved.indicators
    } == {"q"}


def test_authored_erfa_alias_keeps_metric_units_and_description():
    text = AUTHORED.replace(
        "method: annual_summary/1",
        "set: eflow-erfa\n            method: erfa_annual_statistic/1",
    ).replace(
        "{variable: q, statistic: p95}",
        "{metric: wyday_amax1d, statistic: med}",
    )
    indicator = sets.resolve_set(_source(text)).indicators[0].declaration
    assert indicator.name == "high_flow"
    assert registry.output_units(indicator) == "d"
    assert "Day of water year" in registry.quantity(indicator)


@pytest.mark.parametrize("name", ["../eflow-erfa", "eflow/erfa", "-erfa", "CON", "nul"])
def test_hyphen_support_preserves_portable_path_refusals(name):
    with pytest.raises(IndicatorSetError, match="must match"):
        sets.validate_id(name, "indicator set")


def test_preset_is_reserved_for_the_bundled_default():
    text = EXAMPLE.replace(
        "  indicators:\n        - definition: q_annual_mean\n        - definition: q_mean_annual_7day_min",
        "  preset: current_default",
    )
    with pytest.raises(IndicatorSetError, match="only the bundled default"):
        sets.resolve_set(_source(text), ["q"])


def test_definition_selectors_reference_registered_declarations():
    resolved = sets.resolve_set(_source(EXAMPLE))
    assert [item.indicator for item in resolved.indicators] == [
        "q_annual_mean",
        "q_mean_annual_7day_min",
    ]
    assert resolved.declarations[0] == registry.declaration_for("q_annual_mean")
    assert resolved.indicators[0].authored_references == ()


def test_authored_entry_keeps_registered_calculation_under_its_own_indicator_key():
    item = sets.resolve_set(_source(AUTHORED)).indicators[0]
    registered = registry.declaration_for("q_mean_annual_p95")
    assert item.indicator == "high_flow"
    assert item.equivalent_definition == "q_mean_annual_p95"
    assert item.authored_references == ("Author 2020",)
    assert item.references == ()
    assert item.declaration.statistic == registered.statistic == "q95"
    assert item.declaration.required_responses == registered.required_responses


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("id: high_flow", "id: q_annual_mean", "equals a registered definition"),
        ("id: high_flow", "id: High", "must match"),
        ("references: [Author 2020]", "references: ['']", "nonempty strings"),
        ("statistic: p95}", "statistic: p95, units: mm}", "unsupported parameters"),
        ("          title: High flow\n", "", "exactly"),
    ],
)
def test_authored_entry_refusals(old, new, match):
    with pytest.raises(IndicatorSetError, match=match):
        sets.resolve_set(_source(AUTHORED.replace(old, new)))


def test_within_set_duplicate_calculation_contracts_are_refused():
    duplicate = AUTHORED.rstrip(" ") + "        - definition: q_mean_annual_p95\n"
    with pytest.raises(IndicatorSetError, match="share one calculation contract"):
        sets.resolve_set(_source(duplicate))
    repeated = EXAMPLE.rstrip(" ") + "        - definition: q_annual_mean\n"
    with pytest.raises(IndicatorSetError, match="repeats indicator"):
        sets.resolve_set(_source(repeated))


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("schema_version: 1", "schema_version: 2", "schema_version"),
        ("id: example_subset", "id: other", "differs from its selected name"),
        ("title: Example discharge subset", "title: ''", "title"),
        (
            "    - definition: q_annual_mean",
            "    - {definition: q_annual_mean, title: x}",
            "no overrides",
        ),
        (
            "    - definition: q_annual_mean",
            "    - definition: q_unknown",
            "unknown registered",
        ),
        (
            "description: Illustrative",
            "units: mm\n    description: Illustrative",
            "exactly",
        ),
        (
            "title: Example discharge subset",
            "title: a\n    title: b",
            "duplicate YAML key",
        ),
    ],
)
def test_closed_set_schema_refusals(old, new, match):
    with pytest.raises(IndicatorSetError, match=match):
        sets.resolve_set(_source(EXAMPLE.replace(old, new, 1)))


def test_parse_selection_defaults_and_custom_paths(tmp_path):
    custom = tmp_path / "indicator_sets" / "example_subset.yml"
    custom.parent.mkdir()
    custom.write_text(textwrap.dedent(EXAMPLE), encoding="utf-8")
    workflow = tmp_path / "wf4.yml"
    selection = sets.parse_selection(
        {
            "operations": ["indicators", "simulations"],
            "indicator_sets": ["default", "example_subset"],
            "indicator_set_definitions": {
                "example_subset": {"config_path": "indicator_sets/example_subset.yml"}
            },
        },
        workflow,
    )
    assert selection.operations == ("simulations", "indicators")
    assert [s.set_id for s in selection.sets] == ["default", "example_subset"]
    assert (
        selection.sets[0].bundled and selection.sets[0].path.parent == sets.BUNDLED_ROOT
    )
    assert selection.sets[1].path == custom.resolve()
    # The admitted bytes are captured; a later source edit cannot change them.
    custom.write_text("changed", encoding="utf-8")
    assert sets.resolve_set(selection.sets[1]).set_id == "example_subset"
    omitted = sets.parse_selection({}, workflow)
    assert omitted.operations == ("simulations", "indicators")
    assert [s.set_id for s in omitted.sets] == ["default"]


def test_simulations_only_validates_structure_without_opening_sets(tmp_path):
    selection = sets.parse_selection(
        {
            "operations": ["simulations"],
            "indicator_sets": [],
            "indicator_set_definitions": {"later": {"config_path": "missing.yml"}},
        },
        tmp_path / "wf4.yml",
    )
    assert selection.sets == () and not selection.indicators


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"operations": None}, "nonempty list"),
        ({"operations": "indicators"}, "nonempty list"),
        ({"operations": []}, "nonempty list"),
        ({"operations": ["indicators", "indicators"]}, "duplicate"),
        ({"operations": ["metrics"]}, "unknown entries"),
        ({"indicator_sets": None}, "nonempty list"),
        ({"indicator_sets": []}, "nonempty list"),
        ({"indicator_sets": ["default", "default"]}, "duplicate"),
        ({"indicator_sets": ["con"]}, "Windows reserved"),
        ({"indicator_sets": ["../x"]}, "must match"),
        ({"indicator_sets": ["nope"]}, "unknown indicator set"),
        ({"indicator_set_definitions": None}, "mapping"),
        (
            {"indicator_set_definitions": {"default": {"config_path": "x.yml"}}},
            "collides",
        ),
        ({"indicator_set_definitions": {"x": {"config_path": ""}}}, "exactly"),
        ({"indicator_set_definitions": {"x": {"config_path": "a", "y": 1}}}, "exactly"),
        (
            {"operations": ["indicators"], "operation": "indicators-only"},
            "legacy interface",
        ),
        ({"indicator_sets": ["default"], "indicators": ["q"]}, "legacy interface"),
    ],
)
def test_parse_selection_refusals(tmp_path, settings, match):
    with pytest.raises(IndicatorSetError, match=match):
        sets.parse_selection(settings, tmp_path / "wf4.yml")


def test_legacy_operation_maps_to_default_set():
    both = sets.legacy_selection({"operation": "simulate-and-indicators"})
    only = sets.legacy_selection({"operation": "indicators-only"})
    assert both.operations == ("simulations", "indicators") and both.legacy
    assert only.operations == ("indicators",)
    assert [s.set_id for s in only.sets] == ["default"]


def test_required_variables_is_the_union_across_sets():
    resolved = [
        sets.resolve_set(_source(EXAMPLE)),
        sets.resolve_set(
            sets.SetSource(
                "default",
                sets.BUNDLED_ROOT / "default.yml",
                True,
                (sets.BUNDLED_ROOT / "default.yml").read_text(encoding="utf-8"),
            ),
            ["aet", "q"],
        ),
    ]
    assert sets.required_variables(resolved) == ["q", "aet"]


def test_the_shipped_indicator_set_template_resolves():
    path = (
        Path(__file__).resolve().parents[1]
        / "config/templates/indicator_set.template.yml"
    )
    source = sets.SetSource(
        "example_subset", path, False, path.read_text(encoding="utf-8")
    )
    resolved = sets.resolve_set(source)
    assert [item.indicator for item in resolved.indicators] == [
        "q_annual_mean",
        "q_mean_annual_7day_min",
    ]


def test_legacy_declaration_batch_observes_revisions_once_and_refreshes(monkeypatch):
    records = sets.default_calculations(["q", "gwr"])
    calls = []

    def revision(set_id="default"):
        calls.append(set_id)
        return str(len(calls)) * 64

    monkeypatch.setattr(registry, "_revision", revision)
    first = sets.declarations_for_calculations(records)
    assert calls == ["default"]
    assert {item.implementation_revision for item in first} == {"1" * 64}
    fresh = sets.declarations_for_calculations(records)
    assert calls == ["default", "default"]
    assert {item.implementation_revision for item in fresh} == {"2" * 64}


def _crop_water(tmp_path, choice):
    settings = {
        "operations": ["simulations", "indicators"],
        "indicator_sets": ["crop-water"],
        "indicator_selection": {"crop-water": choice},
    }
    selection = sets.parse_selection(settings, tmp_path / "wf4.yml")
    resolved = sets.selected_sets(
        {"model": {}},
        settings,
        selection,
        simulation_path=True,
        retained_variables=lambda: [],
        legacy_context="simulation_rule",
    )
    return selection, resolved[0]


def test_indicator_selection_excludes_without_authoring_declarations(tmp_path):
    from blueearth_cst.experiment.simulation_runner import check_required_variables

    _, full = _crop_water(tmp_path, {"exclude": ["annual_recharge"]})
    selection, narrowed = _crop_water(tmp_path, {"exclude": ["summer_low_flow_days"]})
    assert selection.choices == (("crop-water", "exclude", ("summer_low_flow_days",)),)
    keys = [item.indicator for item in narrowed.indicators]
    assert len(keys) == 9 and "summer_low_flow_days" not in keys
    # A deselected indicator is outside the request: no Q is required.
    assert "q" not in sets.required_variables([narrowed])
    check_required_variables([narrowed], {"precip", "pet", "aet", "gwr"}, "outvars")
    assert sets.calculation_records(narrowed.indicators) != sets.calculation_records(
        full.indicators
    )


@pytest.mark.parametrize("available,kind", [({"q"}, "parameter"), (set(), "response")])
def test_preflight_distinguishes_parameters_from_response_variables(
    tmp_path, available, kind
):
    from blueearth_cst.experiment.simulation_runner import check_required_variables

    _, selected = _crop_water(tmp_path, {"include": ["summer_low_flow_days"]})
    with pytest.raises(ValueError) as error:
        check_required_variables([selected], available, "the configured model.outvars")
    message = str(error.value)
    if kind == "parameter":
        assert "missing_parameter:thresholds" in message
        assert "indicator parameter" in message
        assert "model.outvars" not in message
        assert "--new-experiment" not in message
    else:
        assert "needs response q" in message
        assert "model.outvars" in message
        assert "--new-experiment" in message


def test_indicator_selection_include_keeps_set_order(tmp_path):
    _, narrowed = _crop_water(
        tmp_path, {"include": ["annual_recharge", "winter_climatic_deficit"]}
    )
    assert [item.indicator for item in narrowed.indicators] == [
        "winter_climatic_deficit",
        "annual_recharge",
    ]


def test_indicator_selection_survives_the_invocation_snapshot(tmp_path):
    selection, _ = _crop_water(tmp_path, {"include": ["annual_recharge"]})
    path = sets.write_selection_snapshot(selection, tmp_path / "snapshot.json")
    assert sets.read_selection_snapshot(path) == selection


@pytest.mark.parametrize(
    ("choice", "match"),
    [
        (None, "exactly one of"),
        ({}, "exactly one of"),
        ({"include": ["annual_recharge"], "exclude": ["x"]}, "exactly one of"),
        ({"only": ["annual_recharge"]}, "exactly one of"),
        ({"exclude": []}, "nonempty list"),
        ({"exclude": "annual_recharge"}, "nonempty list"),
        ({"exclude": ["annual_recharge", "annual_recharge"]}, "duplicate"),
        ({"exclude": ["low_flow"]}, "absent from the set"),
        ({"include": ["nope"]}, "absent from the set"),
    ],
)
def test_indicator_selection_refusals(tmp_path, choice, match):
    with pytest.raises(IndicatorSetError, match=match):
        _crop_water(tmp_path, choice)


def test_indicator_selection_refuses_unselected_sets_and_empty_results(tmp_path):
    with pytest.raises(IndicatorSetError, match="must be a mapping"):
        sets.parse_selection({"indicator_selection": []}, tmp_path / "wf4.yml")
    with pytest.raises(IndicatorSetError, match="absent from indicator_sets"):
        sets.parse_selection(
            {"indicator_selection": {"crop-water": {"exclude": ["annual_recharge"]}}},
            tmp_path / "wf4.yml",
        )
    all_keys = [item.name for item in loader.declarations("crop-water")]
    with pytest.raises(IndicatorSetError, match="leaves no indicators"):
        _crop_water(tmp_path, {"exclude": all_keys})
