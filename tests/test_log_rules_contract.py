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

import functools
import re
import tempfile
from pathlib import Path, PurePosixPath

import pytest

SNAKEDIR = Path(__file__).resolve().parents[1]
CONFIG_FN = Path(__file__).resolve().parent / "snake_config_fixture.yml"

#: Every entry point, with the module-level list naming its log sections.
WORKFLOWS = [
    "analyze_climate.smk",
    "build_model.smk",
    "analyze_projections.smk",
    "run_stress_test.smk",
]

#: The directory every per-rule log part lives under, in all four workflows.
PARTS_DIR_NAME = "_parts"

#: (snakefile, config variant) pairs. A workflow whose RULE SET depends on its
#: config appears MORE THAN ONCE, because the contract is a property of a parsed
#: workflow rather than of a file. WF0 is the only one today: rule 0.06 and its
#: appended label both exist only for a multi-source run, and that pair is
#: precisely what the old source-text parser could not see.
CASES = [(snakefile, "default") for snakefile in WORKFLOWS] + [
    ("analyze_climate.smk", "two_source")
]

#: Readable ids, so a failure names the configuration and not just an index.
CASE_IDS = [f"{snakefile}-{variant}" for snakefile, variant in CASES]


#: A label starts with its rule number -- `3.12_perturb_climate_realization`.
#: The same `<W.NN>` grammar `merge_logs._RULE_NUMBER` recognises.
_LABEL_PREFIX = re.compile(r"^\d+\.\d+[a-z]?_")


def _declared_log_rules(workflow, snakefile: str) -> list[str]:
    """``LOG_RULES`` as the parsed workflow actually holds it.

    Read from ``workflow.globals`` rather than from the source text, and the
    difference is not stylistic. ``analyze_climate.smk`` APPENDS
    ``0.06_compare_climate_sources`` at parse time when a run has candidate
    sources, so the list literal is five entries while a multi-source run merges
    six. A source-text parser cannot see that by construction: it reads the
    literal, and the append is a statement.

    That blindness is what forced a SECOND parser into existence --
    ``test_compare_climate_sources`` matched the ``.append(...)`` call with its
    own regex, exactly the duplication this module's docstring forbids. Reading
    the executed globals lets one parser cover both halves, so that regex is
    gone and the rule stated above holds again.

    It also makes both sides of every comparison below describe the SAME
    configuration. Previously the declared side came from source text and the
    produced side from a parsed workflow, so under a config-dependent rule set
    they were answers to different questions.
    """
    value = workflow.globals.get("LOG_RULES")
    assert value is not None, f"{snakefile}: no module-level LOG_RULES"
    assert isinstance(value, list) and all(isinstance(i, str) for i in value), (
        f"{snakefile}: LOG_RULES must be a list of strings"
    )
    return list(value)


def _parse_workflow(snakefile: str, config_path: Path = CONFIG_FN):
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
            config_settings=api.ConfigSettings(configfiles=[config_path]),
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


def _labels_with_producers(workflow) -> set[str]:
    """Every log label some rule in this workflow actually writes."""
    labels = set()
    for rule in workflow.rules:
        for log_path in getattr(rule, "log", []) or []:
            if PARTS_DIR_NAME in PurePosixPath(str(log_path).replace("\\", "/")).parts:
                labels.add(_label_from_log_path(log_path))
    return labels


@functools.lru_cache(maxsize=None)
def _config_for(variant: str) -> Path:
    """The config a case is parsed under.

    ``two_source`` is the fixture config with one candidate source added, which
    is what brings rule 0.06 into existence. Built once and cached: it is read
    by every test in the module and nothing mutates it.
    """
    if variant == "default":
        return CONFIG_FN
    assert variant == "two_source", f"unknown config variant {variant!r}"
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["workflows"]["analyze_climate"]["candidate_sources"] = ["chirps"]
    path = Path(tempfile.mkdtemp(prefix="log_rules_")) / "snake_config_two_sources.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@functools.lru_cache(maxsize=None)
def _case(snakefile: str, variant: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """Both sides of the contract, from ONE parse of ONE configuration.

    Cached because parsing a Snakefile is the expensive part and four tests ask
    the same question of each case. It also guarantees the declared and produced
    sides can never come from different parses.
    """
    workflow = _parse_workflow(snakefile, _config_for(variant))
    declared = tuple(_declared_log_rules(workflow, snakefile))
    return declared, frozenset(_labels_with_producers(workflow))


@pytest.mark.parametrize(("snakefile", "variant"), CASES, ids=CASE_IDS)
def test_every_declared_label_has_a_producing_rule(snakefile, variant):
    """No label may outlive the rule that wrote it (the [R10-8] defect)."""
    declared, produced = _case(snakefile, variant)
    orphaned = sorted(set(declared) - produced)
    assert not orphaned, (
        f"{snakefile}: LOG_RULES names labels no rule writes: {orphaned}. "
        "Each contributes an empty 'no part from this run' section to every "
        "merged log. Delete the entry, or restore the rule."
    )


@pytest.mark.parametrize(("snakefile", "variant"), CASES, ids=CASE_IDS)
def test_every_logging_rule_is_declared(snakefile, variant):
    """No rule may write parts merge_logs will never look for."""
    declared, produced = _case(snakefile, variant)
    unlisted = sorted(produced - set(declared))
    assert not unlisted, (
        f"{snakefile}: rules write log parts under labels LOG_RULES omits: "
        f"{unlisted}. Their sections vanish from the merged log and their part "
        "files are never cleaned up. Add each label, in rule-number order."
    )


@pytest.mark.parametrize(("snakefile", "variant"), CASES, ids=CASE_IDS)
def test_declared_labels_are_unique(snakefile, variant):
    """A repeated label would merge the same section twice."""
    declared, _ = _case(snakefile, variant)
    duplicates = sorted({label for label in declared if declared.count(label) > 1})
    assert not duplicates, f"{snakefile}: duplicate LOG_RULES entries: {duplicates}"


@pytest.mark.parametrize(("snakefile", "variant"), CASES, ids=CASE_IDS)
def test_declared_labels_read_in_rule_number_order(snakefile, variant):
    """``LOG_RULES`` is the MERGE order, so it must read as the workflow does.

    Deferred until `[R10-5]` and added with it, which is the whole point of the
    delay. Before the renumber this file's own note recorded why it could not be
    asserted: `analyze_projections.smk` said "Order is by RULE NUMBER"
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
    declared, _ = _case(snakefile, variant)
    declared = list(declared)
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
