"""Static contracts for WF0--WF2 captured run-record/2 archives."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SNAKEFILES = [
    "build_model.smk",
    "analyze_projections.smk",
]


def _rule_block(snakefile: Path, name: str) -> str:
    """Return one rule body from a Snakefile."""
    text = snakefile.read_text(encoding="utf-8")
    match = re.search(
        rf"^[ \t]*rule {name}:\n(.*?)(?=^[ \t]*(?:rule|checkpoint) |\Z)",
        text,
        re.S | re.M,
    )
    assert match, f"rule {name} not found in {snakefile.name}"
    return match.group(1)


@pytest.mark.parametrize(
    ("snakefile_name", "record_path"),
    [
        (
            "analyze_climate.smk",
            "config/analyze_climate/run_record.yml",
        ),
        (
            "build_model.smk",
            "config/build_model/run_record.yml",
        ),
        (
            "analyze_projections.smk",
            "config/analyze_projections/run_record.yml",
        ),
    ],
)
def test_launcher_owns_archive_publication(snakefile_name, record_path):
    snakefile = REPO / snakefile_name
    text = snakefile.read_text(encoding="utf-8")
    assert "rule snapshot_config:" not in text
    assert "require_capture(" in text
    assert f'RUN_RECORD = f"{{project_dir}}/{record_path}"' in text.splitlines()
    assert (
        'JOURNAL_PATH = f"{project_dir}/config/_engine/journal.jsonl"'
        in text.splitlines()
    )


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
            "build_model.smk",
            '("project", "basin", "climate", "model", "workflows.build_model")',
        ),
        (
            "analyze_projections.smk",
            '("project", "basin", "climate", "model", "workflows.analyze_projections")',
        ),
    ],
)
def test_each_workflow_declares_its_consumed_key_projection(snakefile_name, expected):
    """Scoping by consumed keys is what stops one workflow re-firing another."""
    text = (REPO / snakefile_name).read_text(encoding="utf-8")

    assert f"CONFIG_PROJECTION = {expected}" in text


def test_successors_use_typed_retained_records_instead_of_snapshot_guard():
    generation = (REPO / "generate_scenarios.smk").read_text(encoding="utf-8")
    simulation = (
        REPO / "blueearth_cst/experiment/rules/simulate_and_indicators.smk"
    ).read_text(encoding="utf-8")
    assert "generation_configuration" in generation
    assert "read_pinned_plan" in generation
    assert "freeze_simulation_v2" in simulation
    for text in (generation, simulation):
        assert "check_project_consistency" not in text
        assert "snapshot_config" not in text


@pytest.mark.parametrize("snakefile_name", SNAKEFILES)
def test_the_wide_digest_still_describes_consumed_settings(snakefile_name):
    text = (REPO / snakefile_name).read_text(encoding="utf-8")
    assert "configuration_inputs_digest(" in text
    assert "CAPTURE_DIGESTS" in text


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
        assert re.search(rf"^[ \t]*{handler}$", text, re.M), (
            f"{snakefile_name} lacks {handler}"
        )
    assert '_journal("success")' in text
    assert '_journal("failed")' in text


def test_the_sidecar_rule_is_numbered_in_sequence():
    """WF1's run-metadata sidecar sits between the evaluation and the gathers.

    It took the letter suffix `1.15b` when added, since dev-naming-conventions.md forbids
    renumbering to insert a rule; the 2026-09-24 rule-naming migration then
    renumbered WF1 deliberately, so the sequence is contiguous again.
    """
    wf1 = (REPO / "build_model.smk").read_text(encoding="utf-8")

    assert 'RULES.banner_only("1.16", "write_run_metadata"' in wf1
    assert 'RULES.banner_only("1.17", "gather_benchmarks"' in wf1
