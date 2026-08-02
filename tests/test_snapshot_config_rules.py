"""Static contracts for the three effective-configuration snapshot rules."""

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _rule_block(snakefile: Path, name: str) -> str:
    """Return one rule body from a Snakefile."""
    text = snakefile.read_text(encoding="utf-8")
    match = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert match, f"rule {name} not found in {snakefile.name}"
    return match.group(1)


@pytest.mark.parametrize(
    ("snakefile_name", "stable_output", "snapshot_scope"),
    [
        (
            "Snakefile_model_creation",
            "config/runs/snake_config_model_creation.yml",
            "config/runs/model_creation/",
        ),
        (
            "Snakefile_climate_projections",
            "config/runs/snake_config_climate_projections.yml",
            "config/runs/climate_projections/",
        ),
        (
            "Snakefile_climate_experiment",
            "config/snake_config_climate_experiment.yml",
            "config/runs/climate_experiment/",
        ),
    ],
)
def test_snapshot_rule_keeps_current_copy_and_adds_effective_bundle(
    snakefile_name, stable_output, snapshot_scope
):
    """Every workflow keeps its guard-compatible copy and adds a bundle."""
    snakefile = REPO / snakefile_name
    text = snakefile.read_text(encoding="utf-8")
    block = _rule_block(snakefile, "snapshot_config")

    assert "rule copy_config:" not in text
    assert stable_output in block
    assert "effective_config = config" in block
    assert "advanced_settings = ADVANCED_SETTINGS" in block
    assert "snapshot_bundle = directory(CONFIG_SNAPSHOT_DIR)" in block
    assert snapshot_scope in text


@pytest.mark.parametrize(
    "snakefile_name",
    [
        "Snakefile_model_creation",
        "Snakefile_climate_projections",
        "Snakefile_climate_experiment",
    ],
)
def test_snapshot_identity_includes_resolved_and_referenced_settings(snakefile_name):
    """The content-addressed path derives from all provenance inputs."""
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert "snapshot_bundle_digest(" in text
    assert "ADVANCED_SETTINGS" in text
    assert "config_path" in text
    assert "_config_snapshot_references" in text


@pytest.mark.parametrize(
    "snakefile_name",
    [
        "Snakefile_model_creation",
        "Snakefile_climate_projections",
        "Snakefile_climate_experiment",
    ],
)
def test_bundle_directory_is_named_with_the_short_digest(snakefile_name):
    """One naming length for the bundle dir and the files inside it.

    The directory used to carry the full 64-hex digest while its own archived
    files used 12 — unreadable on the outside, inconsistent on the inside. The
    length now lives in ``provenance.SHORT_DIGEST_CHARS``; this keeps any one
    workflow from drifting back to the raw digest.
    """
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert "short_digest(CONFIG_SNAPSHOT_DIGEST)" in text
    assert "{CONFIG_SNAPSHOT_DIGEST}" not in text
