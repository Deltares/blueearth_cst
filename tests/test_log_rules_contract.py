"""``LOG_RULES`` must name exactly the rules that write a log part.

``merge_logs`` discovers a rule's parts by listing the directory named after its
``LOG_RULES`` label, and is deliberately scoped to that list so a renamed rule's
orphan directory is never read. The consequence is that **both** directions of
disagreement fail silently:

* a rule whose label is **missing** from the list writes parts that are neither
  merged nor cleaned up — its section vanishes from the merged log while the
  files accumulate on disk forever;
* a label with **no producing rule** contributes an empty "no part from this
  run" section to every merged log, forever.

Neither raises. Both have happened: rules 1.01b, 3.01b and 2.03b were each added
without their label, and ``2.11_extract_climate_grid`` outlived the rule ADR 0003
§5 deleted (`dev/followups.md` [R10-8]). ``rule-naming-design.md`` treats this as
checklist discipline for the R10 sweep — the single edit most likely to break
exactly this surface. It is mechanically checkable, so it is checked here
instead.

The label is derived from each rule's own ``log:`` path rather than from its
name, which is what makes rule 3.10 fall out correctly: its identifiers are
``run_wflow_batch_<b>`` (one per batch, parse-time generated) while every batch
writes into the single ``3.10_run_wflow`` directory. Deriving from the path
records that divergence as the deliberate thing it is (P3-3).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

import pytest

SNAKEDIR = Path(__file__).resolve().parents[1]
CONFIG_FN = Path(__file__).resolve().parent / "snake_config_model_test.yml"

#: Every entry point, with the module-level list naming its log sections.
WORKFLOWS = [
    "Snakefile_model_creation",
    "Snakefile_climate_projections",
    "Snakefile_climate_experiment",
]

#: The directory every per-rule log part lives under, in all three workflows.
PARTS_DIR_NAME = "_parts"


#: The ``LOG_RULES = [ ... ]`` block, closing bracket anchored at column 0.
_LOG_RULES_BLOCK = re.compile(r"^LOG_RULES\s*=\s*(\[.*?^\])", re.MULTILINE | re.DOTALL)


def _declared_log_rules(snakefile: str) -> list[str]:
    """Read the ``LOG_RULES`` literal out of a Snakefile without executing it.

    A Snakefile is **not valid Python** — ``rule all:`` is Snakemake grammar, so
    ``ast.parse`` over the whole file raises ``SyntaxError``. The list itself is
    a plain literal of string constants in all three files, so the block is
    lifted out textually and only that is parsed. Comments inside the block are
    fine: ``literal_eval`` tokenizes them away.

    Reading the source rather than the executed workflow's globals also keeps
    this independent of where a given Snakemake version stashes a Snakefile's
    module namespace.
    """
    text = (SNAKEDIR / snakefile).read_text(encoding="utf-8")
    match = _LOG_RULES_BLOCK.search(text)
    assert match, f"{snakefile}: no module-level LOG_RULES list literal"
    value = ast.literal_eval(match.group(1))
    assert all(isinstance(item, str) for item in value), (
        f"{snakefile}: LOG_RULES must be a list of string literals"
    )
    return list(value)


def _parse_workflow(snakefile: str):
    """Parse a Snakefile in-process and return its ``Workflow``.

    Same entry point and the same private-accessor caveat as
    ``tests/test_climate_store_contract.py``: ``wf_api._workflow`` is private on
    the pinned Snakemake and there is no public accessor for the parsed
    workflow. Parsing only builds rules — no DAG, so no input file has to exist.
    """
    import snakemake.api as api

    with api.SnakemakeApi() as sa:
        wf_api = sa.workflow(
            resource_settings=api.ResourceSettings(cores=1),
            config_settings=api.ConfigSettings(configfiles=[CONFIG_FN]),
            storage_settings=api.StorageSettings(),
            workflow_settings=api.WorkflowSettings(),
            snakefile=SNAKEDIR / snakefile,
            workdir=SNAKEDIR,
        )
        workflow = wf_api._workflow
        workflow.include(workflow.main_snakefile, overwrite_default_target=True)
        return workflow


def _label_from_log_path(log_path: str) -> str:
    """Derive a rule's log label from one declared ``log:`` path.

    Two shapes exist, and the label is the path component directly under
    ``logs/_parts/`` in both:

    * ``logs/_parts/<label>.log``           — one part per run
    * ``logs/_parts/<label>/<wildcards>.log`` — one part per fanned-out job
    """
    parts = PurePosixPath(str(log_path).replace("\\", "/")).parts
    index = parts.index(PARTS_DIR_NAME)
    remainder = parts[index + 1:]
    if len(remainder) == 1:
        return PurePosixPath(remainder[0]).stem
    return remainder[0]


def _labels_with_producers(snakefile: str) -> set[str]:
    """Every log label some rule in this workflow actually writes."""
    workflow = _parse_workflow(snakefile)
    labels = set()
    for rule in workflow.rules:
        for log_path in getattr(rule, "log", []) or []:
            if PARTS_DIR_NAME in PurePosixPath(
                str(log_path).replace("\\", "/")
            ).parts:
                labels.add(_label_from_log_path(log_path))
    return labels


@pytest.mark.parametrize("snakefile", WORKFLOWS)
def test_every_declared_label_has_a_producing_rule(snakefile):
    """No label may outlive the rule that wrote it (the [R10-8] defect)."""
    declared = set(_declared_log_rules(snakefile))
    produced = _labels_with_producers(snakefile)
    orphaned = sorted(declared - produced)
    assert not orphaned, (
        f"{snakefile}: LOG_RULES names labels no rule writes: {orphaned}. "
        "Each contributes an empty 'no part from this run' section to every "
        "merged log. Delete the entry, or restore the rule."
    )


@pytest.mark.parametrize("snakefile", WORKFLOWS)
def test_every_logging_rule_is_declared(snakefile):
    """No rule may write parts merge_logs will never look for."""
    declared = set(_declared_log_rules(snakefile))
    produced = _labels_with_producers(snakefile)
    unlisted = sorted(produced - declared)
    assert not unlisted, (
        f"{snakefile}: rules write log parts under labels LOG_RULES omits: "
        f"{unlisted}. Their sections vanish from the merged log and their part "
        "files are never cleaned up. Add each label, in rule-number order."
    )


@pytest.mark.parametrize("snakefile", WORKFLOWS)
def test_declared_labels_are_unique(snakefile):
    """A repeated label would merge the same section twice."""
    declared = _declared_log_rules(snakefile)
    duplicates = sorted({label for label in declared if declared.count(label) > 1})
    assert not duplicates, f"{snakefile}: duplicate LOG_RULES entries: {duplicates}"


# NOT asserted: that LOG_RULES reads in rule-number order.
#
# It is the merge order, and `Snakefile_climate_projections` says "Order is by
# RULE NUMBER" — but its list opens `2.03b_delineate_region`, `2.01_...`,
# `2.02_...`, so the comment and the list disagree. Which one is wrong is a
# ruling, not a fact: 2.03b first is correct by EXECUTION order (the region is
# delineated before the first fetch) and wrong by rule number.
#
# Asserting either would encode a convention nobody has chosen, and the question
# dissolves once `[R10-5]` renumbers positionally — number, execution and sort
# order coincide from then on. Recorded there; add the assertion after.
