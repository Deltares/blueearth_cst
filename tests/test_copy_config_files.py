"""R07 B9: the project config snapshot is routed by KIND, not to one dir.

This was a signature change rather than a rename -- `copy_config_files` derived
a single `output_dir` from the snake config's output path and wrote everything
beside it, so it could not serve four destinations (runs/, catalogs/,
templates/, generated/). These pin the new contract.
"""


import pytest

from blueearth_cst.model.copy_config_files import copy_config_files  # noqa: E402


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
            "artifact_data": str(cfg / "catalogs"),   # predefined, no file
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
