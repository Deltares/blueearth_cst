"""Model reference: writing it, and refusing to simulate when it drifts.

The guard's value is entirely in WHEN it runs. A check that fires after the
forcing is downscaled and the members have run is a post-mortem, not a guard —
so the ordering is asserted structurally here, by parsing the Snakefile, rather
than trusted to a comment.
"""

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blueearth_cst.experiment.check_model_reference import (  # noqa: E402
    ModelDriftError,
    check_model_reference,
    compare_reference,
)
from blueearth_cst.experiment.write_model_reference import (  # noqa: E402
    build_model_reference,
    write_model_reference,
)
from blueearth_cst.shared.model_digest import DIGEST_VERSION  # noqa: E402

_TOML = """\
[input]
path_static = "staticmaps.nc"
path_forcing = "forcing/inmaps_historical.nc"
"""

SNAKEFILE = Path(__file__).resolve().parents[1] / "run_stress_test.smk"


def _model(tmp_path):
    root = tmp_path / "models" / "hydrology" / "wflow"
    (root / "forcing").mkdir(parents=True)
    (root / "wflow_sbm.toml").write_text(_TOML, encoding="utf-8")
    (root / "staticmaps.nc").write_bytes(b"STATICMAPS")
    (root / "forcing" / "inmaps_historical.nc").write_bytes(b"FORCING")
    return root


# ---------------------------------------------------------------------------
# The reference document
# ---------------------------------------------------------------------------


def test_the_reference_stores_a_relative_posix_model_path(tmp_path):
    """Relative and POSIX, so the reference survives the project moving or
    being read on another platform -- the same reason the digest hashes no
    absolute path."""
    root = _model(tmp_path)
    doc = build_model_reference(root, tmp_path)
    assert doc["model_path"] == "models/hydrology/wflow"
    assert not os.path.isabs(doc["model_path"]) and "\\" not in doc["model_path"]


def test_the_reference_records_per_input_hashes_not_just_the_digest(tmp_path):
    """A bare digest can only say SOMETHING moved; the guard must name what."""
    doc = build_model_reference(_model(tmp_path), tmp_path)
    assert set(doc["inputs"]) == {
        "wflow_sbm.toml",
        "staticmaps.nc",
        "forcing/inmaps_historical.nc",
    }
    assert doc["digest_version"] == DIGEST_VERSION


def test_writing_produces_readable_yaml(tmp_path):
    root = _model(tmp_path)
    out = tmp_path / "experiments" / "e" / "config" / "model_reference.yml"
    written = write_model_reference(root, tmp_path, out)
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == written


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_an_unchanged_model_passes(tmp_path):
    root = _model(tmp_path)
    assert compare_reference(build_model_reference(root, tmp_path), root) == []


def test_a_changed_input_is_named(tmp_path):
    """Naming the input is the point: 'the forcing was replaced' and 'the
    parameters were re-derived' call for different responses."""
    root = _model(tmp_path)
    doc = build_model_reference(root, tmp_path)
    (root / "forcing" / "inmaps_historical.nc").write_bytes(b"REPLACED")
    diffs = compare_reference(doc, root)
    assert len(diffs) == 1
    assert "forcing/inmaps_historical.nc" in diffs[0] and "changed" in diffs[0]


def test_check_raises_and_the_message_says_what_to_do(tmp_path):
    root = _model(tmp_path)
    out = tmp_path / "model_reference.yml"
    write_model_reference(root, tmp_path, out)
    (root / "staticmaps.nc").write_bytes(b"REBUILT")

    with pytest.raises(ModelDriftError) as excinfo:
        check_model_reference(out, root, experiment="gabon_dry")
    msg = str(excinfo.value)
    assert "staticmaps.nc" in msg  # what changed
    assert "gabon_dry" in msg  # which experiment
    assert "new experiment" in msg.lower()  # what to do about it


def test_the_drift_message_reaches_the_rules_own_log_part(tmp_path):
    """[R10-13]: the rule fails naming a log file that must explain why.

    Snakemake reported `Error in rule check_model_reference ... log:
    .../3.06_check_model_reference.log (check log file(s) for error details)`
    and that file held the three header lines and nothing else -- the
    ModelDriftError went to Snakemake's own log instead. Wired together here
    because the general fix in `tee_to_log` is only worth anything if it
    actually reaches THIS raise, which happens inside the manager.
    """
    from blueearth_cst.shared.snake_utils import tee_to_log

    root = _model(tmp_path)
    out = tmp_path / "model_reference.yml"
    write_model_reference(root, tmp_path, out)
    (root / "staticmaps.nc").write_bytes(b"REBUILT")

    log = tmp_path / "logs" / "_parts" / "3.06_check_model_reference.log"
    with pytest.raises(ModelDriftError):
        with tee_to_log(log):
            check_model_reference(out, root, experiment="gabon_dry")

    text = log.read_text(encoding="utf-8")
    assert "staticmaps.nc" in text  # what changed -- the useful part
    assert "gabon_dry" in text  # which experiment
    assert "new experiment" in text.lower()  # what to do about it
    assert "ModelDriftError" in text


def test_a_newly_created_reference_passes_against_the_changed_model(tmp_path):
    """The other half of the end-to-end falsifier: drift blocks the OLD
    experiment, and a NEW one records the current model and proceeds."""
    root = _model(tmp_path)
    old = tmp_path / "old.yml"
    write_model_reference(root, tmp_path, old)
    (root / "staticmaps.nc").write_bytes(b"REBUILT")

    with pytest.raises(ModelDriftError):
        check_model_reference(old, root)

    new = tmp_path / "new.yml"
    write_model_reference(root, tmp_path, new)
    check_model_reference(new, root)  # must not raise


def test_a_digest_version_change_is_reported_as_incomparable(tmp_path):
    """Not as drift. Entries hashed under a different scheme are not
    comparable, and calling it drift sends an operator looking for a model
    change that never happened."""
    root = _model(tmp_path)
    doc = build_model_reference(root, tmp_path)
    doc["digest_version"] = DIGEST_VERSION + 1
    diffs = compare_reference(doc, root)
    assert len(diffs) == 1 and "not comparable" in diffs[0]


# ---------------------------------------------------------------------------
# Ordering — the property the guard IS
# ---------------------------------------------------------------------------


def _rule_block(name: str) -> str:
    text = SNAKEFILE.read_text(encoding="utf-8")
    start = text.index(f"rule {name}:")
    nxt = text.find("\nrule ", start + 1)
    return text[start : nxt if nxt != -1 else len(text)]


def test_the_guard_gates_the_first_rule_that_touches_the_model():
    """Structural, not a comment: rule 3.09 downscale_climate_realization is the
    first rule to use the model, and it must declare the guard's sentinel as an
    INPUT. Without this edge the guard could run after the simulation and every
    other test here would still pass."""
    downscale = _rule_block("downscale_climate_realization")
    inputs = downscale[downscale.index("input:") : downscale.index("output:")]
    assert ".model_reference_ok" in inputs, (
        "rule 3.09 does not declare the drift guard's sentinel as an input, so "
        "nothing orders the guard before simulation work"
    )


def test_the_guards_verdict_does_not_persist_between_invocations():
    """The edge alone is not the guard — this is what P4 missed.

    Rule 3.09 declaring the sentinel as an input orders the guard BEFORE the
    work, and the test above pins that. But a sentinel that survives the run
    satisfies that edge with a STALE VERDICT: the check passed once against
    model M1, the file remains, and a later invocation re-simulates against M2
    without the guard running at all.

    Demonstrated in isolation at the R9 landing gate, on a two-rule probe, so
    the result does not depend on WF3's scheduling quirks:

      * consumer up to date, model drifted -> nothing runs. Safe by INACTION:
        the guard does not fire, but neither does the consumer.
      * consumer must re-run, model drifted -> with a persisted sentinel the
        guard is NOT scheduled and the consumer runs anyway; with temp() the
        guard IS scheduled, first, and stops it.

    The second case is the hole. It needs 3.09 to genuinely re-run — a different
    `-c` (the batch split is core-derived), deleted temp intermediates, a retried
    failure, added realizations — so it is narrower than "every re-run", and
    saying otherwise would overstate it.

    Confirmed end to end in the real workflow: model perturbed, 3.09 forced to
    re-run at the tree's own core count, the run stopped at 1 of 34 steps with
    ModelDriftError and no member simulated. Detection was correct throughout;
    only the trigger was missing.

    `temp()` is the trigger: Snakemake deletes the sentinel once 3.09 has
    consumed it, so the next invocation finds it absent and re-evaluates against
    whatever the pointer-derived digest currently covers. That is also why the
    fix is not "drop ancient() from model_toml" — the digest reaches files this
    rule does not declare, and declaring them would duplicate what
    `model_digest` discovers through the TOML's pointers.
    """
    guard = _rule_block("check_model_reference")
    out = guard[guard.index("output:") : guard.index("log:")]
    assert "temp(" in out, (
        "the guard's sentinel is not temp(), so its verdict persists and rule "
        "3.09's edge can be satisfied by a check that ran against a different "
        "model — the R9 gate measured exactly this"
    )
    assert "touch(" in out, "the sentinel must still be a touch() marker"


def test_the_guard_reads_the_reference_and_the_writer_produces_it():
    """The two rules are a producer/consumer pair -- the class of bug this
    milestone hit three times. Asserted rather than assumed."""
    writer = _rule_block("write_model_reference")
    guard = _rule_block("check_model_reference")
    assert "model_reference.yml" in writer[writer.index("output:") :]
    assert (
        "model_reference.yml" in guard[guard.index("input:") : guard.index("output:")]
    )


def test_the_writer_declares_its_model_inputs_ancient():
    """Load-bearing, not incidental. If a rebuilt model re-triggered the writer,
    the reference would be rewritten to match, the comparison would always pass,
    and the guard would be decorative."""
    writer = _rule_block("write_model_reference")
    decl = writer[writer.index("input:") : writer.index("params:")]
    for line in decl.splitlines():
        if "basin_dir" in line:
            assert "ancient(" in line, f"model input not ancient(): {line.strip()}"


def test_rule_3_00b_sentinels_are_untouched():
    """The brief gates 3.00b's declared inputs and sentinel paths behind
    approval, because they carry the incremental-execution constraint. The new
    guard uses its OWN sentinel; this pins that 3.00b was not co-opted."""
    guard_00b = _rule_block("check_project_consistency")
    assert ".model_reference_ok" not in guard_00b
    assert ".project_consistency_ok" in guard_00b
    assert ".guard_ok" in guard_00b


# The declared-label-is-registered check that used to live here is GONE, folded
# into tests/test_log_rules_contract.py [R10-10]. It asserted a strict SUBSET of
# what `test_every_logging_rule_is_declared` asserts there, by a weaker parser,
# and the two disagreed for two independent reasons:
#
#  - its slicer ran `text.index("]", ...)`, so the FIRST `]` anywhere in the
#    block -- including one inside a comment -- ended the slice and every label
#    below it read as unregistered. Commit 129580f put a bracketed followups
#    reference in WF1's list and this test went red and stayed red;
#  - its label regex matched the f-string form `{LOG_PARTS_DIR}/<label>` only,
#    so WF2's and WF3's fan-out rules -- which build their `log:` path by
#    concatenation -- were invisible to it. That is the "14/3/14" its own
#    docstring recorded: it saw 3 of WF2's 6 labels.
#
# The contract module derives labels from the PARSED workflow's `rule.log`, so
# it has neither blind spot, and it asserts across all three Snakefiles at once
# rather than inside a loop where the first failure hides the rest.


def _script_modules():
    """Every module Snakemake EXECUTES via `script:`, resolved to a real path.

    `REGION.script` is a variable, not a literal, so it is resolved through
    `snake_utils.REGION_SCRIPT` rather than skipped -- it is the one splatted
    producer shared by all three workflows, so missing it would leave the widest
    blast radius uncovered.
    """
    from blueearth_cst.shared.snake_utils import REGION_SCRIPT

    repo = SNAKEFILE.parent
    found = {REGION_SCRIPT}
    for snakefile in sorted(repo.glob("Snakefile_*")):
        text = snakefile.read_text(encoding="utf-8")
        found |= set(re.findall(r'script:\s*"([^"]+\.py)"', text))
    return sorted(repo / rel for rel in found if (repo / rel).is_file())


def test_no_script_module_carries_a_future_import():
    """Snakemake PREPENDS a preamble to a `script:` module, so a
    `from __future__` import inside it is no longer at the top of the file and
    the job dies with SyntaxError before running a line of our code.

    Found at R9's landing gate by the first real run of P4's three new rules --
    3.01c, 3.01d and 3.01e all carried it, and all three failed. Nothing before
    that run could have caught it: the unit tests IMPORT these modules, where the
    future import is perfectly legal, and `--dry-run` never executes a `script:`
    body at all. So the defect was invisible to every rung of the ladder below a
    real run.

    No placement inside the file can fix it -- the preamble goes before the whole
    file -- so the import must simply be absent. On the pinned Python (3.12) it
    buys nothing: PEP 585 and PEP 604 annotations are native.

    SCOPE: this covers modules reached by a literal `script: "…"` plus the one
    variable-resolved `REGION.script`. A path assembled some other way would not
    be seen. It also does not check IMPORTED modules -- `model_digest.py` and
    five others keep their future import legitimately, because nothing prepends
    anything to a module you import.
    """
    offenders = [
        p
        for p in _script_modules()
        if re.search(r"^from __future__ import", p.read_text(encoding="utf-8"), re.M)
    ]
    assert not offenders, (
        "`from __future__` in a Snakemake `script:` module fails at RUN time with "
        "SyntaxError, and only a real run reveals it: "
        + ", ".join(p.name for p in offenders)
    )


def test_the_future_import_check_can_actually_fail(tmp_path):
    """Guard the guard. The test above asserts an ABSENCE, so a bug in how it
    finds script modules would make it pass over an empty set forever."""
    assert _script_modules(), "no script: modules discovered -- the check is vacuous"
    decoy = tmp_path / "decoy.py"
    decoy.write_text(
        '"""d."""\n\nfrom __future__ import annotations\n', encoding="utf-8"
    )
    assert re.search(
        r"^from __future__ import", decoy.read_text(encoding="utf-8"), re.M
    )
