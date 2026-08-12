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
§5 deleted (`dev/followups-archive.md` [R10-8]). ``rule-naming-design.md`` treats this as
checklist discipline for the R10 sweep — the single edit most likely to break
exactly this surface. It is mechanically checkable, so it is checked here
instead.

The label is derived from each rule's own ``log:`` path rather than from its
name, which is what makes the batched Wflow rule fall out correctly: its
identifiers are ``run_wflow_batch_<b>`` (one per batch, parse-time generated)
while every batch writes into a single ``<W.NN>_run_wflow`` directory. Deriving
from the path records that divergence as the deliberate thing it is (P3-3).

**This module is the ONLY home for the property** (`dev/followups-archive.md`
``[R10-10]``). ``tests/test_model_reference.py`` used to assert a subset of it
by slicing the ``LOG_RULES`` text to the first ``]`` and matching an f-string
label form; both were wrong in ways that let a real failure sit unread, and two
modules asserting one property by different parsers is how they came to
disagree. Do not re-add a second parser — extend this one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

import pytest

SNAKEDIR = Path(__file__).resolve().parents[1]
CONFIG_FN = Path(__file__).resolve().parent / "snake_config_fixture.yml"

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

#: A label starts with its rule number -- `3.12_perturb_climate_realization`.
#: The same `<W.NN>` grammar `merge_logs._RULE_NUMBER` recognises.
_LABEL_PREFIX = re.compile(r"^\d+\.\d+[a-z]?_")


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

    Two shapes exist below the label, and the label itself is recognised by the
    ``<W.NN>_`` rule-number prefix every part carries:

    * ``.../_parts/<label>.log``             — one part per run
    * ``.../_parts/<label>/<wildcards>.log`` — one part per fanned-out job

    **Found by the prefix, not by position.** The label used to be read as the
    component directly under ``_parts/``, which broke the moment WF3's parts
    moved to ``logs/_parts/<experiment>/`` (2026-08-11): every WF3 rule reported
    the label ``experiment``, and the contract this module exists to enforce
    became one assertion about a directory name. Matching the numbering
    convention instead makes the derivation independent of how many scoping
    levels sit between ``_parts/`` and the label.
    """
    parts = PurePosixPath(str(log_path).replace("\\", "/")).parts
    index = parts.index(PARTS_DIR_NAME)
    for component in parts[index + 1 :]:
        # Strip only the `.log` extension. NOT `PurePosixPath.stem`, which reads
        # the rule number's own dot as a suffix and turns `2.04_fetch_gcm_slice`
        # into `2`.
        label = component[: -len(".log")] if component.endswith(".log") else component
        if _LABEL_PREFIX.match(label):
            return label
    raise AssertionError(
        f"no <W.NN>_<rule> component under {PARTS_DIR_NAME}/ in {log_path!r}"
    )


def _labels_with_producers(snakefile: str) -> set[str]:
    """Every log label some rule in this workflow actually writes."""
    workflow = _parse_workflow(snakefile)
    labels = set()
    for rule in workflow.rules:
        for log_path in getattr(rule, "log", []) or []:
            if PARTS_DIR_NAME in PurePosixPath(str(log_path).replace("\\", "/")).parts:
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


@pytest.mark.parametrize("snakefile", WORKFLOWS)
def test_declared_labels_read_in_rule_number_order(snakefile):
    """``LOG_RULES`` is the MERGE order, so it must read as the workflow does.

    Deferred until `[R10-5]` and added with it, which is the whole point of the
    delay. Before the renumber this file's own note recorded why it could not be
    asserted: `Snakefile_climate_projections` said "Order is by RULE NUMBER"
    while its list opened ``2.03b``, ``2.01``, ``2.02`` — correct by EXECUTION
    order, wrong by number. Which of the two the list should follow was a ruling
    nobody had made, and asserting either would have encoded a convention by
    accident. `W.NN` is positional now, so number order, dependency order and
    sort order coincide and the question dissolves.

    **Plain string sort is the right comparison here, and only because `NN` is
    zero-padded to two digits** — ``"1.09" < "1.10"`` holds lexically, which it
    would not for ``1.9`` / ``1.10``. A future rule inserted with a letter
    suffix also sorts correctly (``"1.09" < "1.09b" < "1.10"``), which is what
    keeps the documented insert-with-a-suffix escape hatch compatible with this
    assertion.

    What a failure means: the merged log's sections come out in an order that
    contradicts both the rule map and the benchmark table, so following one run
    means knowing the list rather than reading the file.
    """
    declared = _declared_log_rules(snakefile)
    assert declared == sorted(declared), (
        f"{snakefile}: LOG_RULES is not in rule-number order.\n"
        f"  got:      {declared}\n"
        f"  expected: {sorted(declared)}\n"
        "It is the merge order for merge_logs, so this is the order the merged "
        "log's sections come out in. Reorder the list wholesale rather than "
        "entry by entry."
    )


# NOT asserted: that the rules are DEFINED in that order in the Snakefile.
#
# They are not, and are not required to be. Module-level code is interleaved
# between rule blocks and depends on its position (`_basavg_pngs` is built just
# above the rule that declares it; the `_batches` loop generates its rules), so
# reordering the blocks to match the numbers would be a behaviour risk taken for
# cosmetics. `W.NN` is the rule's position in the workflow's logical order, not
# its offset in the file.
