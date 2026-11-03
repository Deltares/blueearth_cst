"""Early attribution guards must work without WF3 or prepared model inputs."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment import reservation_request as request_module
from blueearth_cst.experiment.content_identity import content_sha256


@pytest.fixture
def case(tmp_path, monkeypatch):
    from blueearth_cst.experiment import collection_resolution

    project = tmp_path / "project"
    project.mkdir()
    marker = project / "collection.json"
    marker.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        collection_resolution,
        "resolve_explicit_collection_v2",
        lambda selector: (
            {"manifest_path": str(marker)},
            {},
        ),
    )
    catalog = tmp_path / "catalog.yml"
    yaml_document = {
        "elevation": {
            "data_type": "RasterDataset",
            "uri": str(tmp_path / "elevation.tif"),
            "driver": "rasterio",
            "data_adapter": {"unit_mult": {"elevtn": 1}},
        },
    }
    catalog.write_text(yaml.safe_dump(yaml_document), encoding="utf-8")
    wf4 = tmp_path / "simulate.yml"
    wf4.write_text(
        yaml.safe_dump(
            {
                "operation": "simulate-and-indicators",
                "scenario_collection": {"manifest_path": str(marker)},
            }
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build.yml"
    build.write_text(
        yaml.safe_dump({"simulation_window": {"start": 2000, "end": 2016}}),
        encoding="utf-8",
    )
    path = tmp_path / "project.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {"project_dir": str(project), "catalog": [str(catalog)]},
                "basin": {
                    "region": "{'subbasin': [9.666, 0.4476]}",
                    "resolution": 0.00833,
                },
                "climate": {"selected": "era5", "window": {"start": 2000, "end": 2016}},
                "workflows": {
                    "simulate_system": {"enabled": True, "config_path": wf4.name},
                    "build_model": {"enabled": False, "config_path": build.name},
                    "generate_scenarios": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    return path, catalog, wf4, build


def project_request(case, **kwargs):
    return request_module.capture_request(case[0], Path.cwd(), **kwargs)


def test_unperturbed_switch_is_frozen_but_default_identity_is_unchanged(case):
    from blueearth_cst.experiment.reservation import validate_request

    original, _ = project_request(case)
    settings = yaml.safe_load(case[2].read_bytes())
    settings["include_unperturbed"] = True
    case[2].write_text(yaml.safe_dump(settings), encoding="utf-8")
    explicit_default, _ = project_request(case)
    assert explicit_default == original
    settings["include_unperturbed"] = False
    case[2].write_text(yaml.safe_dump(settings), encoding="utf-8")
    omitted, _ = project_request(case)
    assert omitted["simulation"]["include_unperturbed"] is False
    assert content_sha256(omitted) != content_sha256(original)
    validate_request(omitted, case[0].parent / "project")


def test_retained_explicit_selection_does_not_capture_unused_producer_files(case):
    from blueearth_cst.experiment.reservation import (
        CapturedSource,
        validate_request,
        verify_captured_sources,
    )

    document = yaml.safe_load(case[0].read_bytes())
    document["basin"]["output_locations"] = str(case[0].parent / "missing.csv")
    case[0].write_text(yaml.safe_dump(document), encoding="utf-8")
    request, sources = project_request(case)
    assert request["generation"] is None
    assert request["simulation"]["model_request"] is None
    assert "output_locations" not in {role for role, _, _ in sources}
    validate_request(request, document["project"]["project_dir"])
    captured = [CapturedSource(role, path, data) for role, path, data in sources]
    verify_captured_sources(captured, request)


@pytest.mark.parametrize("change", ["uri", "unit", "reader"])
def test_explicit_collection_catalog_changes_guarded_without_generation(case, change):
    before, _ = project_request(case)
    document = yaml.safe_load(case[1].read_bytes())
    if change == "uri":
        document["elevation"]["uri"] = str(case[1].parent / "different.tif")
    elif change == "unit":
        document["elevation"]["data_adapter"]["unit_mult"]["elevtn"] = 1000
    else:
        document["elevation"]["driver"] = {
            "name": "rasterio",
            "options": {"preprocess": "round"},
        }
    case[1].write_text(yaml.safe_dump(document), encoding="utf-8")
    after, _ = project_request(case)
    assert before["generation"] is after["generation"] is None
    assert content_sha256(before) != content_sha256(after)


def test_catalog_comments_and_mapping_order_compatible_but_archived(case):
    before, original_sources = project_request(case)
    document = yaml.safe_load(case[1].read_bytes())
    document["elevation"] = dict(reversed(list(document["elevation"].items())))
    case[1].write_text(
        "# explanatory comment\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    after, changed_sources = project_request(case)
    assert before == after
    assert original_sources != changed_sources


def test_catalog_relative_anchor_is_scientific(case):
    document = yaml.safe_load(case[1].read_bytes())
    document["elevation"]["uri"] = "elevation.tif"
    case[1].write_text(yaml.safe_dump(document), encoding="utf-8")
    before, _ = project_request(case)
    other = case[1].parent / "another"
    other.mkdir()
    moved = other / case[1].name
    moved.write_bytes(case[1].read_bytes())
    project = yaml.safe_load(case[0].read_bytes())
    project["project"]["catalog"] = [str(moved)]
    case[0].write_text(yaml.safe_dump(project), encoding="utf-8")
    after, _ = project_request(case)
    assert before["input_catalogs"] != after["input_catalogs"]


def test_unchanged_catalog_bytes_still_resolve_current_roots(case):
    document = yaml.safe_load(case[1].read_bytes())
    preferred = case[1].parent / "preferred"
    fallback = case[1].parent / "fallback"
    fallback.mkdir()
    document["meta"] = {"roots": [str(preferred), str(fallback)]}
    document["elevation"]["uri"] = "elevation.tif"
    case[1].write_text(yaml.safe_dump(document), encoding="utf-8")
    before, _ = project_request(case)
    preferred.mkdir()
    after, _ = project_request(case)
    assert (
        before["input_catalogs"][0]["document"]
        == after["input_catalogs"][0]["document"]
    )
    assert before["input_catalogs"] != after["input_catalogs"]


@pytest.mark.parametrize("change", [None, "root", "document"])
def test_legacy_catalog_request_comparison_preserves_records(case, change):
    from blueearth_cst.experiment.reservation import compare_request, validate_request

    current, _ = project_request(case)
    retained = deepcopy(current)
    retained["schema_version"] = "experiment-request/1"
    for catalog in retained["input_catalogs"]:
        root = catalog.pop("root")
        catalog["resolved_sources"] = {"meta": {"root": root}, "elevation": {}}
    original = deepcopy(retained)
    validate_request(retained, case[0].parent / "project")
    if change == "root":
        current["input_catalogs"][0]["root"] += "/different"
    elif change == "document":
        current["input_catalogs"][0]["document"]["elevation"]["uri"] = "different.tif"
    if change is None:
        compare_request(retained, current)
    else:
        with pytest.raises(ValueError, match="input_catalogs"):
            compare_request(retained, current)
    assert retained == original


def test_model_recipe_guard_precedes_model_existence(case):
    before, captures = project_request(case, execute_model=True)
    assert before["simulation"]["model_request"] is not None
    assert {"build_template", "waterbodies_template"} <= {
        role for role, _, _ in captures
    }
    document = yaml.safe_load(case[3].read_bytes())
    document["simulation_window"]["end"] = 2015
    case[3].write_text(yaml.safe_dump(document), encoding="utf-8")
    after, _ = project_request(case, execute_model=True)
    assert before != after


@pytest.mark.parametrize("settings", [{}, {"simulation_window": None}])
def test_missing_model_window_is_refused_upfront(case, settings):
    case[3].write_text(
        "# simulation_window is commented out.\n" + yaml.safe_dump(settings),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="build_model.simulation_window.*is required"):
        project_request(case, execute_model=True)


def test_model_window_outside_climate_record_is_refused_upfront(case):
    case[3].write_text(
        yaml.safe_dump({"simulation_window": {"start": 1999, "end": 2016}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not inside climate.window"):
        project_request(case, execute_model=True)


def test_unclassified_simulation_field_refused(case):
    document = yaml.safe_load(case[2].read_bytes())
    document["new_scientific_parameter"] = 3
    case[2].write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unclassified simulate_system"):
        project_request(case)


def test_resources_and_indicators_do_not_change_scientific_request(case):
    before, _ = project_request(case)
    document = yaml.safe_load(case[2].read_bytes())
    document["compute"] = {"julia_threads": 4, "batch_size": 2}
    document["indicators"] = ["q"]
    case[2].write_text(yaml.safe_dump(document), encoding="utf-8")
    after, _ = project_request(case)
    assert before == after
    document["compute"]["future_setting"] = 1
    case[2].write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unclassified simulate_system.compute"):
        project_request(case)
