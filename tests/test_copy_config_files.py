"""R07 B9: the project config snapshot is routed by KIND, not to one dir.

This was a signature change rather than a rename -- `copy_config_files` derived
a single `output_dir` from the snake config's output path and wrote everything
beside it, so it could not serve four destinations (runs/, catalogs/,
templates/, generated/). These pin the new contract.
"""

import json
from pathlib import Path

import pytest
import yaml

from blueearth_cst.model.copy_config_files import copy_config_files  # noqa: E402
from blueearth_cst.shared.provenance import SHORT_DIGEST_CHARS  # noqa: E402


@pytest.fixture()
def sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    snake = src / "snake_config_model_creation.yml"
    snake.write_text("project:\n  project_dir: somewhere\n", encoding="utf-8")
    catalog = src / "deltares_data.yml"
    catalog.write_text("meta: {}\n", encoding="utf-8")
    template = src / "wflow_build_model.yml"
    template.write_text("steps: []\n", encoding="utf-8")
    return snake, catalog, template


def test_each_kind_lands_in_its_own_bin(tmp_path, sources):
    snake, catalog, template = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={
            str(catalog): str(cfg / "catalogs"),
            str(template): str(cfg / "templates"),
        },
    )
    assert (cfg / "runs" / "snake_config_model_creation.yml").is_file()
    assert (cfg / "catalogs" / "deltares_data.yml").is_file()
    assert (cfg / "templates" / "wflow_build_model.yml").is_file()
    # nothing leaks into the parent bin
    assert not (cfg / "deltares_data.yml").exists()
    assert not (cfg / "snake_config_model_creation.yml").exists()


def test_content_is_copied_verbatim(tmp_path, sources):
    """A snapshot that mutates content would break the drift guard, which
    compares digests of these files across workflows."""
    snake, catalog, _ = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={str(catalog): str(cfg / "catalogs")},
    )
    assert (cfg / "runs" / "snake_config_model_creation.yml").read_text(
        encoding="utf-8"
    ) == snake.read_text(encoding="utf-8")
    assert (cfg / "catalogs" / "deltares_data.yml").read_text(
        encoding="utf-8"
    ) == catalog.read_text(encoding="utf-8")


def test_missing_source_is_skipped_not_fatal(tmp_path, sources):
    """hydromt's predefined catalogs have no path on disk -- an absent entry
    must be skipped rather than crash the snapshot."""
    snake, catalog, _ = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={
            str(catalog): str(cfg / "catalogs"),
            "artifact_data": str(cfg / "catalogs"),  # predefined, no file
        },
    )
    assert (cfg / "catalogs" / "deltares_data.yml").is_file()
    assert not (cfg / "catalogs" / "artifact_data").exists()


def test_destination_dirs_are_created(tmp_path, sources):
    """Snakemake creates parents for DECLARED outputs; the catalog/template
    bins are not declared outputs, so the script must create them itself."""
    snake, catalog, _ = sources
    cfg = tmp_path / "deep" / "nested" / "config"
    assert not cfg.exists()
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={str(catalog): str(cfg / "catalogs")},
    )
    assert (cfg / "runs").is_dir() and (cfg / "catalogs").is_dir()


# --------------------------------------------------------------------------- #
# The observations bin (2026-08-01)
# --------------------------------------------------------------------------- #


def test_observations_land_in_their_own_bin(tmp_path, sources):
    """The two observation inputs are a fifth KIND, routed like the rest.

    They live outside the repo AND outside project_dir, referenced by absolute
    path (R07 O-01), so without this snapshot a finished project cannot say
    what it was evaluated against.
    """
    snake, _, _ = sources
    cfg = tmp_path / "project" / "config"
    locations = tmp_path / "src" / "output_locations.csv"
    locations.write_text("wflow_id,station_name,x,y\n", encoding="utf-8")
    series = tmp_path / "src" / "observations_timeseries.csv"
    series.write_text("time;101;102\n", encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
        other_config_files={
            str(locations): str(cfg / "observations"),
            str(series): str(cfg / "observations"),
        },
    )

    assert (cfg / "observations" / "output_locations.csv").is_file()
    assert (cfg / "observations" / "observations_timeseries.csv").is_file()
    # Routed, not duplicated into the other bins.
    assert not (cfg / "catalogs").exists()
    assert not (cfg / "templates").exists()


def test_the_snapshot_is_a_faithful_copy(tmp_path, sources):
    """A snapshot that silently truncated would be worse than none."""
    snake, _, _ = sources
    cfg = tmp_path / "project" / "config"
    series = tmp_path / "src" / "observations_timeseries.csv"
    body = "time;101;102\n2000-01-01T00:00:00;1.5;2.5\n2000-01-02T00:00:00;;3.0\n"
    series.write_text(body, encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake.yml",
        other_config_files={str(series): str(cfg / "observations")},
    )
    copied = (cfg / "observations" / "observations_timeseries.csv").read_text(
        encoding="utf-8"
    )
    assert copied == body


# --------------------------------------------------------------------------- #
# Immutable effective-config bundle
# --------------------------------------------------------------------------- #


def test_writes_effective_config_bundle_and_archives_references(tmp_path, sources):
    """The durable bundle records resolved settings and immutable file copies."""
    snake, catalog, template = sources
    cfg = tmp_path / "project" / "config"
    snapshot_dir = cfg / "runs" / "model_creation" / "bundle-digest"
    effective_config = {"project": {"project_dir": "somewhere"}, "values": [1, 2]}
    advanced_settings = {
        "constraints": {"min_historical_years": 16},
        "defaults": {"julia_threads": 4},
        "runtime": {"julia_version": "1.11.7"},
    }

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
        other_config_files={
            str(catalog): str(cfg / "catalogs"),
            str(template): str(cfg / "templates"),
            "artifact_data": str(cfg / "catalogs"),
        },
        snapshot_dir=snapshot_dir,
        effective_config=effective_config,
        advanced_settings=advanced_settings,
        workflow_name="model_creation",
    )

    assert (snapshot_dir / "source.yml").read_text(encoding="utf-8") == (
        snake.read_text(encoding="utf-8")
    )
    effective = yaml.safe_load(
        (snapshot_dir / "effective.yml").read_text(encoding="utf-8")
    )
    assert effective["project_config"] == effective_config
    assert effective["advanced_settings"] == advanced_settings
    assert len(effective["effective_config_sha256"]) == 64

    manifest = json.loads(
        (snapshot_dir / "referenced-files.json").read_text(encoding="utf-8")
    )
    assert manifest["workflow"] == "model_creation"
    assert manifest["effective_config_sha256"] == effective["effective_config_sha256"]
    assert manifest["source_config"]["sha256"]
    assert manifest["source_config"]["archived_path"] == "source.yml"

    by_source = {entry["source"]: entry for entry in manifest["referenced_files"]}
    catalog_entry = by_source[str(catalog)]
    assert catalog_entry["kind"] == "catalogs"
    assert catalog_entry["status"] == "archived"
    assert (snapshot_dir / catalog_entry["archived_path"]).read_text(
        encoding="utf-8"
    ) == catalog.read_text(encoding="utf-8")

    logical_entry = by_source["artifact_data"]
    assert logical_entry["status"] == "logical_identifier"
    assert logical_entry["archived_path"] is None
    assert logical_entry["sha256"] is None

    # An archived file is named by the SHARED naming length, not by a literal
    # 12 that could drift away from the directory it sits in.
    archived_name = Path(catalog_entry["archived_path"]).name
    assert archived_name == (
        f"{catalog_entry['sha256'][:SHORT_DIGEST_CHARS]}-{catalog.name}"
    )


def test_the_bundle_announces_itself(tmp_path, sources, capsys):
    """The bundle used to be written in silence, so it was found by accident."""
    snake, catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    snapshot_dir = cfg / "runs" / "model_creation" / "bundle-digest"

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
        other_config_files={str(catalog): str(cfg / "catalogs")},
        snapshot_dir=snapshot_dir,
        effective_config={"project": {"project_dir": "somewhere"}},
        advanced_settings={"constraints": {}, "defaults": {}, "runtime": {}},
        workflow_name="model_creation",
    )

    manifest = json.loads(
        (snapshot_dir / "referenced-files.json").read_text(encoding="utf-8")
    )
    row = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "Config snapshot bundle" in line
    ]
    assert len(row) == 1
    # Both forms: the path to go look at, and the full digest the manifest
    # records — the short directory name alone cannot be compared to another
    # bundle's identity.
    assert str(snapshot_dir) in row[0]
    assert manifest["snapshot_bundle_sha256"] in row[0]


def test_snapshot_bundle_is_deterministic(tmp_path, sources):
    """Writing the same snapshot twice produces byte-identical metadata."""
    snake, catalog, _ = sources
    cfg = tmp_path / "project" / "config"
    kwargs = {
        "config": snake,
        "config_out_path": cfg / "runs" / "snake.yml",
        "other_config_files": {str(catalog): str(cfg / "catalogs")},
        "effective_config": {"b": 2, "a": 1},
        "advanced_settings": {"defaults": {"threads": 4}},
        "workflow_name": "model_creation",
    }
    first = tmp_path / "first"
    second = tmp_path / "second"

    copy_config_files(snapshot_dir=first, **kwargs)
    copy_config_files(snapshot_dir=second, **kwargs)

    assert (first / "effective.yml").read_bytes() == (
        second / "effective.yml"
    ).read_bytes()
    assert (first / "referenced-files.json").read_bytes() == (
        second / "referenced-files.json"
    ).read_bytes()
