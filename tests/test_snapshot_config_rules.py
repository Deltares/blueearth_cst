"""Static contracts for the three configuration snapshot rules.

The content-addressed bundle these once pinned was removed on 2026-08-13
(config-snapshot redesign): it had no readers, and its directory name was a
digest over the WHOLE config, so an edit to any other workflow's section minted
a fresh one. What each rule writes now is a current-only ``run_record.yml``.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SNAKEFILES = [
    "Snakefile_model_creation",
    "Snakefile_climate_projections",
    "Snakefile_climate_experiment",
]


def _rule_block(snakefile: Path, name: str) -> str:
    """Return one rule body from a Snakefile."""
    text = snakefile.read_text(encoding="utf-8")
    match = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert match, f"rule {name} not found in {snakefile.name}"
    return match.group(1)


@pytest.mark.parametrize(
    ("snakefile_name", "stable_output", "record_path"),
    [
        (
            "Snakefile_model_creation",
            "config/runs/snake_config_model_creation.yml",
            "config/runs/model_creation/run_record.yml",
        ),
        (
            "Snakefile_climate_projections",
            "config/runs/snake_config_climate_projections.yml",
            "config/runs/climate_projections/run_record.yml",
        ),
        (
            # WF3's record sits directly in the experiment's config bin, not
            # under a runs/ sub-bin: the experiment IS the partition (R2).
            "Snakefile_climate_experiment",
            "config/snake_config_climate_experiment.yml",
            "config/run_record.yml",
        ),
    ],
)
def test_snapshot_rule_keeps_current_copy_and_writes_a_run_record(
    snakefile_name, stable_output, record_path
):
    """Every workflow keeps its guard-compatible copy and adds a run record.

    The flat copy's path is load-bearing twice over -- three of them are
    baseline-fingerprinted, and the WF3 drift guard reads them -- so it is
    pinned here rather than left to the rule.
    """
    snakefile = REPO / snakefile_name
    text = snakefile.read_text(encoding="utf-8")
    block = _rule_block(snakefile, "snapshot_config")

    assert "rule copy_config:" not in text
    assert stable_output in block
    assert "effective_config = config" in block
    assert "advanced_settings = ADVANCED_SETTINGS" in block
    assert "run_record = RUN_RECORD" in block
    assert record_path in text


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_the_content_addressed_bundle_is_gone(snakefile_name):
    """No workflow may reintroduce the bundle under any of its old names.

    An absence needs its own test: nothing else fails when a digest-named
    directory quietly comes back, because it was write-only in the first place.
    """
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert "snapshot_bundle" not in text
    assert "CONFIG_SNAPSHOT_DIR" not in text
    assert "CONFIG_SNAPSHOT_DIGEST" not in text
    assert "snapshot_bundle_digest(" not in text


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_the_run_record_is_one_file_per_workflow(snakefile_name):
    """One record, replaced in place -- not a directory that accumulates.

    The bundle's defect was that every distinct config minted another
    directory nobody ever read. A record named after the workflow rather than
    after a digest is what keeps that from returning.
    """
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert "RUN_RECORD = " in text
    assert text.count("RUN_RECORD = ") == 1
    assert "run_record.yml" in text
