"""Named hydrology models: registration identity, listing and selection (t2609281102)."""

import json

import pytest

from blueearth_cst.shared.model_registry import (
    ModelChangedError,
    ModelSelectionError,
    build_record,
    copy_model_inputs,
    list_models,
    model_root,
    read_record,
    registration_changes,
    registration_entries,
    require_unchanged,
    resolve_model_name,
    validate_model_name,
    write_record,
)

_TOML = """\
dir_output = "historical/default/run"

[input]
path_static = "staticmaps.nc"
path_forcing = "historical/default/forcing/inmaps_historical.nc"

[state]
path_input = "historical/default/run/outstate/outstates.nc"
"""


def _model(project, name="wflow", toml=_TOML):
    root = project / "models" / "hydrology" / name
    (root / "staticgeoms").mkdir(parents=True)
    (root / "historical" / "default" / "forcing").mkdir(parents=True)
    (root / "wflow_sbm.toml").write_text(toml, encoding="utf-8")
    (root / "staticmaps.nc").write_bytes(b"STATIC")
    (root / "staticgeoms" / "region.geojson").write_text("{}", encoding="utf-8")
    (root / "historical" / "default" / "forcing" / "inmaps_historical.nc").write_bytes(
        b"F"
    )
    return root


def _register(root, name, origin="user_managed", parent="wflow"):
    record = build_record(
        root, name=name, origin=origin, parent=parent, notes="calibrated"
    )
    write_record(root, record)
    return record


@pytest.mark.parametrize("name", ["wflow", "wflow_calibrated", "w2"])
def test_valid_names(name):
    assert validate_model_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["Wflow", "wflow-v2", "2wflow", "", "comparisons", "wflow_staging_era5", None],
)
def test_invalid_names(name):
    with pytest.raises(ModelSelectionError):
        validate_model_name(name)


def test_identity_excludes_toolbox_runs_and_includes_geoms(tmp_path):
    root = _model(tmp_path)
    entries = registration_entries(root)
    assert set(entries) == {
        "staticgeoms/region.geojson",
        "staticmaps.nc",
        "wflow_sbm.toml",
    }


def test_record_round_trip_and_unchanged_after_recipe_runs(tmp_path):
    root = _model(tmp_path, "wflow_calibrated")
    record = _register(root, "wflow_calibrated")
    assert read_record(root) == json.loads(
        (root / "_engine/model.json").read_text("utf-8")
    )
    assert record["origin"] == "user_managed" and record["parent"] == "wflow"
    # Everything WF1 writes for a registered model's recipes.
    (root / "wflow_sbm_default.toml").write_text("x", encoding="utf-8")
    (root / "historical" / "default" / "run").mkdir(parents=True)
    (root / "historical" / "default" / "run" / "output.csv").write_text(
        "q", encoding="utf-8"
    )
    (root / "historical" / "default" / "forcing" / "inmaps_historical.nc").write_bytes(
        b"F2"
    )
    assert registration_changes(root) == []
    assert require_unchanged(root)["name"] == "wflow_calibrated"


def test_edited_model_is_refused_naming_the_file(tmp_path):
    root = _model(tmp_path, "wflow_calibrated")
    _register(root, "wflow_calibrated")
    (root / "staticmaps.nc").write_bytes(b"EDITED")
    (root / "staticgeoms" / "extra.geojson").write_text("{}", encoding="utf-8")
    assert registration_changes(root) == [
        "added: staticgeoms/extra.geojson",
        "changed: staticmaps.nc",
    ]
    with pytest.raises(ModelChangedError, match="changed: staticmaps.nc"):
        require_unchanged(root)


def test_built_model_is_not_held_to_its_record(tmp_path):
    root = _model(tmp_path)
    _register(root, "wflow", origin="built", parent=None)
    (root / "staticmaps.nc").write_bytes(b"REBUILT")
    assert require_unchanged(root)["origin"] == "built"


def test_record_requires_existing_static_maps(tmp_path):
    root = _model(tmp_path, "broken")
    (root / "staticmaps.nc").unlink()
    with pytest.raises(FileNotFoundError, match="path_static"):
        build_record(root, name="broken", origin="user_managed")


def test_listing_skips_non_models(tmp_path):
    _model(tmp_path)
    calibrated = _model(tmp_path, "wflow_calibrated")
    _register(calibrated, "wflow_calibrated")
    home = tmp_path / "models" / "hydrology"
    (home / "comparisons" / "a").mkdir(parents=True)
    (home / "wflow_staging_era5").mkdir()
    (home / "empty").mkdir()
    entries = list_models(tmp_path)
    assert [(e.name, e.origin) for e in entries] == [
        ("wflow", "built"),
        ("wflow_calibrated", "user_managed"),
    ]


def test_resolution(tmp_path):
    assert resolve_model_name(tmp_path, None) == "wflow"
    _model(tmp_path)
    assert resolve_model_name(tmp_path, None) == "wflow"
    _model(tmp_path, "wflow_calibrated")
    with pytest.raises(ModelSelectionError, match=r"wflow \(built\), wflow_calibrated"):
        resolve_model_name(tmp_path, None)
    assert resolve_model_name(tmp_path, "wflow_calibrated") == "wflow_calibrated"
    # A new name is WF1's to build; consumers report a missing model themselves.
    assert resolve_model_name(tmp_path, "wflow_v2") == "wflow_v2"
    with pytest.raises(ModelSelectionError, match="snake_case"):
        resolve_model_name(tmp_path, "Wflow-V2")


def test_copy_model_inputs_fills_missing_files_only(tmp_path):
    parent = _model(tmp_path)
    target = model_root(tmp_path, "wflow_calibrated")
    target.mkdir(parents=True)
    (target / "staticmaps.nc").write_bytes(b"CALIBRATED")
    copied = copy_model_inputs(parent, target)
    assert copied == ["staticgeoms/region.geojson", "wflow_sbm.toml"]
    assert (target / "staticmaps.nc").read_bytes() == b"CALIBRATED"
    assert not (target / "historical").exists()
    assert copy_model_inputs(parent, target) == []
