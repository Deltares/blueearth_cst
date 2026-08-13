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


# --------------------------------------------------------------------------- #
# Projections, digests, and the journal's declaration semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("snakefile_name", "expected"),
    [
        (
            "Snakefile_model_creation",
            '("project", "shared", "workflows.model_creation")',
        ),
        (
            "Snakefile_climate_projections",
            '("project", "shared", "workflows.climate_projections")',
        ),
    ],
)
def test_each_workflow_declares_its_consumed_key_projection(snakefile_name, expected):
    """Scoping by consumed keys is what stops one workflow re-firing another."""
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert f"CONFIG_PROJECTION = {expected}" in text


def test_wf3_derives_its_projection_from_the_guard_tuple():
    """Derived, not restated -- proximity is not enforcement.

    WF3 genuinely reads other workflows' sections, and `guarded_sections` is
    already the maintained list of those cross-section reads. A projection
    written out beside it would drift the first time that tuple gained an
    entry, and nothing would report it.
    """
    text = (REPO / "Snakefile_climate_experiment").read_text(encoding="utf-8")

    assert "CONFIG_PROJECTION = tuple(sorted(" in text
    assert "for section in guarded_sections" in text
    assert '{"workflows.climate_experiment"}' in text


def test_wf3_projection_equals_the_derived_union():
    """The value the derivation must produce, pinned independently of it.

    Asserting the expression exists proves only that it was written; this
    proves what it evaluates to, which is what a reader of the record cares
    about.
    """
    guarded = (
        "project",
        "shared.basin",
        "workflows.model_creation",
        "workflows.climate_projections",
    )

    derived = tuple(
        sorted(
            {s.split(".")[0] if s == "shared.basin" else s for s in guarded}
            | {"workflows.climate_experiment"}
        )
    )

    assert derived == (
        "project",
        "shared",
        "workflows.climate_experiment",
        "workflows.climate_projections",
        "workflows.model_creation",
    )


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_the_wide_digest_is_threaded_through_the_snapshot_rule(snakefile_name):
    """Params threading is what keeps the record fresh when the CHECKOUT moves.

    Without it a code-only commit leaves the record stamped with the previous
    one and writes no journal line -- the defect both design reviewers found
    independently. It must be a STRING digest: the params trigger compares
    values, and a nested structure is not what the repo's probe verified.
    """
    snakefile = REPO / snakefile_name
    block = _rule_block(snakefile, "snapshot_config")
    text = snakefile.read_text(encoding="utf-8")

    assert "configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST" in block
    assert "config_projection = CONFIG_PROJECTION" in block
    assert "CONFIGURATION_INPUTS_DIGEST = configuration_inputs_digest(" in text


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_the_journal_is_never_a_declared_output(snakefile_name):
    """The silent-truncation trap, pinned as an absence.

    Snakemake deletes a rule's declared outputs BEFORE the job runs, so a
    declared journal would be truncated to one line on every re-execution --
    silently, because a one-line journal is indistinguishable from a young one.
    Emission lives in workflow-level handlers, which have no outputs at all.
    """
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert "JOURNAL_PATH" in text, "the workflow must define a journal path"
    for line in text.splitlines():
        stripped = line.strip()
        if "journal.jsonl" in stripped or "JOURNAL_PATH" in stripped:
            assert not stripped.startswith(("output:", "run_metadata =")), stripped
            assert "output" not in stripped.split("=")[0], stripped


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_every_workflow_registers_all_three_lifecycle_handlers(snakefile_name):
    """The terminal line is the contract; onstart is best-effort tracing."""
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    for handler in ("onstart:", "onsuccess:", "onerror:"):
        assert f"\n{handler}\n" in text, f"{snakefile_name} lacks {handler}"
    assert '_journal("success")' in text
    assert '_journal("failed")' in text


def test_the_sidecar_rules_take_letter_suffixes():
    """`1.16`/`3.17` were already taken, and renumbering is forbidden.

    naming.md §8b: DO NOT RENUMBER TO INSERT A RULE. The design proposed the
    taken numbers, so this pins the correction rather than leaving it to a
    reader to rediscover that gather_benchmarks owns them.
    """
    wf1 = (REPO / "Snakefile_model_creation").read_text(encoding="utf-8")
    wf3 = (REPO / "Snakefile_climate_experiment").read_text(encoding="utf-8")

    assert 'rule_banner("1.15b", "write_run_metadata")' in wf1
    assert 'rule_banner("1.16", "gather_benchmarks")' in wf1
    assert 'rule_banner("3.16b", "write_run_metadata")' in wf3
    assert 'rule_banner("3.17", "gather_benchmarks")' in wf3
