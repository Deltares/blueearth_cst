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

#: A resolved ``workflows.climate_experiment`` section, in miniature. Every key
#: is a LIVE one: the drift cases below change ``horizontime_climate`` and
#: ``run_historical``, which were ``Tpeak`` and ``Tlow`` until 2026-08-12. A
#: guard demonstrated on a key no config declares still passes -- the recorded
#: section is whatever it is handed -- but it stops being evidence that the
#: guard fires on a change anyone can actually make.
_CFG = {
    "realizations_num": 2,
    "run_length": 20,
    "horizontime_climate": 2050,
    "run_historical": False,
}

_SNAKEFILE = Path(__file__).resolve().parents[1] / "run_stress_test.smk"


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
            _marker(tmp_path), out, "gabon_dry", dict(_CFG, horizontime_climate=2085)
        )
    msg = str(excinfo.value)
    assert "horizontime_climate" in msg  # what changed
    assert "gabon_dry" in msg  # which experiment
    assert "new experiment" in msg.lower()  # what to do
    # ...and the recorded file is untouched, so the results still describe it.
    assert (
        yaml.safe_load(out.read_text(encoding="utf-8"))["climate_experiment"][
            "horizontime_climate"
        ]
        == 2050
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
        _marker(tmp_path), out, "gabon_dry", dict(_CFG, run_historical=True)
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
        _marker(tmp_path, "gabon_wet"),
        out,
        "gabon_wet",
        dict(_CFG, run_historical=True),
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


# --- toolbox-side retirements vs user edits (t2608072234) ---------------------
#
# The freeze compares a key UNION, so a key the toolbox retired reads as changed
# and every already-run experiment refused. That was ruled per-milestone twice --
# correctly for `aggregate_rlz` (its removal changed the table's grain, so the old
# rows really do mean something else) and wrongly for `Tpeak`/`Tlow` (value-
# preserving). Since 2026-08-12 the registry declares which, per key.


def _frozen_with(tmp_path, recorded_cfg, name="gabon_dry"):
    """An experiment that HAS RUN, with ``recorded_cfg`` frozen into its record."""
    exp = _exp(tmp_path, name)
    out = exp / "config" / "experiment.yml"
    write_experiment_config(_marker(tmp_path, name), out, name, recorded_cfg)
    _mark_run(tmp_path, name)
    return out


def test_a_value_preserving_retirement_does_not_freeze_the_experiment(tmp_path):
    """The false positive this fixes.

    `Tpeak`/`Tlow` shipped at exactly the values that became the constants, so
    the recorded results are bit-for-bit what they always were. Refusing here
    would strand a completed experiment over a change the user did not make and
    that changed nothing about their numbers.
    """
    out = _frozen_with(tmp_path, dict(_CFG, Tpeak=10, Tlow=2))

    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)  # allowed


def test_the_record_migrates_forward_and_drops_the_retired_keys(tmp_path):
    """Allowing the rewrite is only half of it -- the record must stop carrying
    keys the toolbox no longer has, or every future run re-derives the same
    exemption from a file that is drifting further from the config."""
    out = _frozen_with(tmp_path, dict(_CFG, Tpeak=10, Tlow=2))

    write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)

    recorded = yaml.safe_load(out.read_text(encoding="utf-8"))["climate_experiment"]
    assert "Tpeak" not in recorded and "Tlow" not in recorded
    assert recorded == _CFG


def test_a_grain_changing_retirement_still_freezes(tmp_path):
    """`aggregate_rlz` is the case the freeze is RIGHT about, and R11 ruled it so.

    Retiring it changed the table's grain, so an experiment that ran under the
    old shape cannot continue under the new one. Widening the exemption to every
    retired key would have silently un-refused exactly this.
    """
    out = _frozen_with(tmp_path, dict(_CFG, aggregate_rlz=True))

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    assert "aggregate_rlz" in str(excinfo.value)


def test_an_unregistered_disappearing_key_still_freezes(tmp_path):
    """The default, and the reason it is the default.

    A retirement nobody registered is indistinguishable here from a key that
    vanished for an unknown reason. Refusing makes the omission loud; exempting
    it would silently unfreeze every experiment in the project -- the same
    failure mode as the retired-key registry going unwritten.
    """
    out = _frozen_with(tmp_path, dict(_CFG, some_forgotten_key=7))

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    assert "some_forgotten_key" in str(excinfo.value)


def test_a_real_edit_alongside_a_transparent_retirement_still_freezes(tmp_path):
    """The exemption must not become cover for a genuine change that rides with it.

    Only the real change is named: reporting the retired key too would send the
    user looking for an edit they did not make.
    """
    out = _frozen_with(tmp_path, dict(_CFG, Tpeak=10))

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(
            _marker(tmp_path), out, "gabon_dry", dict(_CFG, horizontime_climate=2085)
        )
    message = str(excinfo.value)
    assert "horizontime_climate" in message
    assert "Tpeak" not in message


def test_a_retired_name_still_present_in_the_config_is_a_user_edit(tmp_path):
    """The exemption is for a key the config no longer DECLARES.

    A name that is in the registry but still present in the current section is a
    live setting whatever the registry says, so a difference in its value is the
    user's edit and freezes. (Reachable only in principle -- the DAG-time
    `refuse_retired_experiment_keys` stops such a config earlier -- but the guard
    must not depend on that other guard having run.)
    """
    out = _frozen_with(tmp_path, dict(_CFG, Tpeak=10))

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(
            _marker(tmp_path), out, "gabon_dry", dict(_CFG, Tpeak=25)
        )
    assert "Tpeak" in str(excinfo.value)


def test_an_unexplained_top_level_key_freezes(tmp_path):
    """An older record shape is not a retirement and must not be dropped
    silently: `build_experiment_config` emits exactly two top-level keys."""
    out = _frozen_with(tmp_path, _CFG)
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    doc["schema_from_the_future"] = 1
    out.write_text(yaml.safe_dump(doc), encoding="utf-8")

    with pytest.raises(ExperimentConfigFrozenError) as excinfo:
        write_experiment_config(_marker(tmp_path), out, "gabon_dry", _CFG)
    assert "schema_from_the_future" in str(excinfo.value)
