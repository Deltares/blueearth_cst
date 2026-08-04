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

SNAKEFILE = Path(__file__).resolve().parents[1] / "Snakefile_climate_experiment"


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
        "wflow_sbm.toml", "staticmaps.nc", "forcing/inmaps_historical.nc"
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
    assert "staticmaps.nc" in msg          # what changed
    assert "gabon_dry" in msg              # which experiment
    assert "new experiment" in msg.lower()  # what to do about it


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
    return text[start: nxt if nxt != -1 else len(text)]


def test_the_guard_gates_the_first_rule_that_touches_the_model():
    """Structural, not a comment: rule 3.09 downscale_climate_realization is the
    first rule to use the model, and it must declare the guard's sentinel as an
    INPUT. Without this edge the guard could run after the simulation and every
    other test here would still pass."""
    downscale = _rule_block("downscale_climate_realization")
    inputs = downscale[downscale.index("input:"): downscale.index("output:")]
    assert ".model_reference_ok" in inputs, (
        "rule 3.09 does not declare the drift guard's sentinel as an input, so "
        "nothing orders the guard before simulation work"
    )


def test_the_guard_reads_the_reference_and_the_writer_produces_it():
    """The two rules are a producer/consumer pair -- the class of bug this
    milestone hit three times. Asserted rather than assumed."""
    writer = _rule_block("write_model_reference")
    guard = _rule_block("check_model_reference")
    assert "model_reference.yml" in writer[writer.index("output:"):]
    assert "model_reference.yml" in guard[guard.index("input:"): guard.index("output:")]


def test_the_writer_declares_its_model_inputs_ancient():
    """Load-bearing, not incidental. If a rebuilt model re-triggered the writer,
    the reference would be rewritten to match, the comparison would always pass,
    and the guard would be decorative."""
    writer = _rule_block("write_model_reference")
    decl = writer[writer.index("input:"): writer.index("params:")]
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


def test_every_declared_log_part_label_is_registered_in_log_rules():
    """The class behind P1's F7 and P4's two repeats, closed mechanically.

    A rule that declares a `log:` part under LOG_PARTS_DIR but is absent from
    LOG_RULES loses its section from the merged log and strands its part --
    silently, on every run. Three instances were fixed by hand across the three
    workflows; this is what stops a fourth.

    SCOPE, stated so the test is not read as stronger than it is: it matches the
    `f"{LOG_PARTS_DIR}/<label>..."` form, which is how every current declaration
    is written, and it counts 14/3/14 labels across the three Snakefiles today.
    A declaration built some other way -- a label assembled from a variable, say
    -- would not be seen, so this closes the common case rather than the whole
    class. Registered-but-undeclared is deliberately NOT an error: merge_logs
    simply finds no part for it, which is how the shared store rules are listed.
    """
    for snakefile in sorted(SNAKEFILE.parent.glob("Snakefile_*")):
        text = snakefile.read_text(encoding="utf-8")
        declared = set(re.findall(r'LOG_PARTS_DIR\}/([0-9]+\.[0-9]+[a-z]?_[a-z_]+)', text))
        block = text[text.index("LOG_RULES = ["): text.index("]", text.index("LOG_RULES = ["))]
        registered = set(re.findall(r'"([0-9]+\.[0-9]+[a-z]?_[a-z_]+)"', block))
        missing = declared - registered
        assert not missing, (
            f"{snakefile.name}: log part label(s) {sorted(missing)} declared but "
            f"absent from LOG_RULES -- merge_logs will drop the section silently"
        )
