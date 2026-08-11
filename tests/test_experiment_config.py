"""experiment.yml: recorded per experiment, frozen once it has run.

The immutability falsifier needs BOTH directions. A test that only checks the
refusal would pass against a file frozen at CREATION — which is the behaviour
this feature explicitly rejects, and the brief's rollback condition. So the
writable-before case is asserted first and given equal weight.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blueearth_cst.experiment.write_experiment_config import (  # noqa: E402
    ExperimentConfigFrozenError,
    build_experiment_config,
    has_run_successfully,
    write_experiment_config,
)

_CFG = {"realizations_num": 2, "run_length": 20, "Tlow": 2, "Tpeak": 10}

_SNAKEFILE = Path(__file__).resolve().parents[1] / "Snakefile_climate_experiment"


def _exp(tmp_path, name="gabon_dry"):
    d = tmp_path / "experiments" / name
    (d / "config").mkdir(parents=True)
    return d


def _marker(tmp_path, name="gabon_dry"):
    """The merged-log path for an experiment, project-scoped and name-keyed.

    Spelled here the way the Snakefile spells it. The module no longer owns a
    ``RUN_MARKER`` constant, and it must not: a path this module composes for
    itself is a second spelling of a name the Snakefile owns, and when they
    drift the guard fails OPEN — ``has_run_successfully`` returns ``False``
    forever and nothing raises. That is why the wiring test below checks this
    literal against rule 3.18's own ``output:`` rather than trusting it.
    """
    return tmp_path / "logs" / f"wf3_climate_experiment_{name}.log"


def _mark_run(tmp_path, name="gabon_dry"):
    marker = _marker(tmp_path, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("merged log", encoding="utf-8")


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def test_the_document_is_the_id_plus_this_experiments_own_section():
    doc = build_experiment_config("gabon_dry", _CFG)
    assert doc["experiment_name"] == "gabon_dry"
    assert doc["climate_experiment"] == _CFG


def test_writing_produces_readable_yaml(tmp_path):
    exp = _exp(tmp_path)
    out = exp / "config" / "experiment.yml"
    written = write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == written


# ---------------------------------------------------------------------------
# Immutability — BOTH directions
# ---------------------------------------------------------------------------


def test_editable_before_the_first_successful_run(tmp_path):
    """The direction a creation-time freeze would break, asserted FIRST.

    Changing an experiment's parameters before it has produced anything is
    ordinary work. A feature that forbade this to make the other case easy
    would be worse than no feature.
    """
    exp = _exp(tmp_path)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    assert not has_run_successfully(_marker(tmp_path))

    changed = dict(_CFG, realizations_num=5)
    doc = write_experiment_config(
        _marker(tmp_path), out, "gabon_dry", changed
    )  # must not raise
    assert doc["climate_experiment"]["realizations_num"] == 5


def test_frozen_after_the_first_successful_run(tmp_path):
    exp = _exp(tmp_path)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    _mark_run(tmp_path)

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(
            _marker(tmp_path), out, "gabon_dry", dict(_CFG, Tpeak=25)
        )
    msg = str(excinfo.value)
    assert "Tpeak" in msg  # what changed
    assert "gabon_dry" in msg  # which experiment
    assert "new experiment" in msg.lower()  # what to do
    # ...and the recorded file is untouched, so the results still describe it.
    assert (
        yaml.safe_load(out.read_text(encoding="utf-8"))["climate_experiment"]["Tpeak"]
        == 10
    )


def test_an_unchanged_rewrite_after_a_run_is_allowed(tmp_path):
    """Snakemake may re-run this rule for reasons unrelated to the config.
    Failing on a no-op would make the guard fire on its own bookkeeping."""
    exp = _exp(tmp_path)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    _mark_run(tmp_path)
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)  # must not raise


def test_the_marker_is_a_completed_run_not_a_started_one(tmp_path):
    """The merged log is WF3's LAST rule and a `rule all` target, so a run that
    failed midway never produces it. Partial artifacts must not freeze."""
    exp = _exp(tmp_path)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)

    # a partial run: log PARTS exist, the merged log does not
    parts = tmp_path / "logs" / "_parts" / "gabon_dry"
    parts.mkdir(parents=True)
    (parts / "3.11_generate_weather_realizations.log").write_text("x", encoding="utf-8")
    (exp / "results").mkdir()
    (exp / "results" / "q_indicators.csv").write_text("a\n", encoding="utf-8")

    assert not has_run_successfully(_marker(tmp_path))
    write_experiment_config(
        _marker(tmp_path), out, "gabon_dry", dict(_CFG, Tlow=3)
    )  # allowed


def test_another_experiments_merged_log_does_not_freeze_this_one(tmp_path):
    """The marker is per EXPERIMENT, and after the move to a shared project
    `logs/` that is carried by the filename alone.

    A name-blind marker -- `logs/wf3_climate_experiment.log`, or a glob -- would
    make the first experiment to complete freeze every other experiment in the
    project, which is the failure the id in the filename exists to prevent.
    """
    exp = _exp(tmp_path, "gabon_wet")
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path, "gabon_wet"), out, "gabon_wet", _CFG)
    _mark_run(tmp_path, "gabon_dry")  # a DIFFERENT experiment completed

    assert not has_run_successfully(_marker(tmp_path, "gabon_wet"))
    write_experiment_config(
        _marker(tmp_path, "gabon_wet"), out, "gabon_wet", dict(_CFG, Tlow=3)
    )  # allowed


def test_no_recorded_file_means_nothing_to_freeze(tmp_path):
    """A run marker without a recorded config is not a frozen state -- there is
    nothing to compare against, and refusing would strand the experiment."""
    exp = _exp(tmp_path)
    _mark_run(tmp_path)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)  # must not raise
    assert out.is_file()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_the_rule_declares_the_file_and_reaches_rule_all():
    """The rule writes what it claims to, and something asks for it.

    The `LOG_RULES` half of this test is GONE, and the reason is worth keeping.
    It asserted `'"3.01e_write_experiment_config"' in text` -- a hardcoded label
    matched against the raw source, a THIRD parser for a property two other
    modules already checked -- and it broke on the [R10-5] renumber for exactly
    the reason [R10-10] predicts: parsers that each know the property a little
    differently drift apart, and the one that breaks first is whichever hardcoded
    the most. `tests/test_log_rules_contract.py` owns it now, derives the label
    from the rule's own `log:` path, and asserts both directions for all three
    workflows -- so this rule's registration is covered more strongly than it
    was here, and without a number to keep in step.
    """
    text = _SNAKEFILE.read_text(encoding="utf-8")
    start = text.index("rule write_experiment_config:")
    block = text[start : text.index("\nrule ", start + 1)]
    assert "config/experiment.yml" in block[block.index("output:") :]
    assert (
        "experiment_config"
        in text[text.index("WF3_TARGETS = {") : text.index("rule all:")]
    )


def _rule_block(text, name):
    start = text.index(f"rule {name}:")
    end = text.find("\nrule ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def test_the_freeze_marker_is_rule_3_18s_own_output():
    """The guard's marker and the merged log must be ONE path expression.

    This is the test that has to be discriminating, because the failure it
    guards is silent: if 3.07's `run_marker` and 3.18's `output:` drift apart,
    `has_run_successfully` returns False forever, the freeze never fires, and
    nothing raises — an experiment's configuration quietly becomes editable
    after it has produced results.

    So it compares the two SOURCE EXPRESSIONS rather than asserting each against
    a literal this file also owns. Checking both against a constant defined here
    would pass just as green with both of them wrong.
    """
    text = _SNAKEFILE.read_text(encoding="utf-8")
    writer = _rule_block(text, "write_experiment_config")
    marker = writer[writer.index("run_marker") :].split("=", 1)[1].split(",")[0].strip()
    gather = _rule_block(text, "gather_logs")
    declared = gather[gather.index("output:") :].splitlines()[1].strip().rstrip(",")
    assert marker == declared, (
        f"3.07 reads {marker}, 3.18 writes {declared} — the freeze guard would "
        "fail open"
    )


def test_the_merged_log_is_keyed_by_the_experiment():
    """One project's `logs/` holds every experiment's merged log, so the name
    must carry the id -- otherwise two experiments write one file and the freeze
    marker cannot tell them apart."""
    text = _SNAKEFILE.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("WORKFLOW_LOG_NAME"))
    assert "{experiment}" in line, line
