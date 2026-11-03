"""Unit tests for the pointer-derived model digest (R9 P4 commit 1).

The fingerprint's purpose is to detect an ABSENCE of change, so a test that only
confirms "the digest changes when the model changes" is half the job. The
discriminating cases here are:

* **pointer discovery** — a file that a fixed list would never have known about
  enters the digest, and editing ITS CONTENT ALONE moves the digest. This is
  the property the design rejected a fixed triple over, and it is demonstrated
  by building a fixed-list digest alongside and showing it does NOT move;
* **the exclusions**, which must hold without a blocklist;
* **the absence marker**, which must make "optional input missing" a different
  state from "optional input present" rather than the same state as "no such
  key".
"""

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blueearth_cst.shared.model_digest import (  # noqa: E402
    ABSENT,
    compare_model_digest,
    model_digest,
    model_digest_entries,
    model_file_set,
)

_TOML = """\
dir_output = "run_default"

[input]
path_static = "staticmaps.nc"
path_forcing = "forcing/inmaps_historical.nc"

[state]
path_input = "instate/instates.nc"
"""


def _model(tmp_path, toml=_TOML, with_instate=True, extra=None):
    """A minimal model root: the TOML plus the files it points at."""
    root = tmp_path / "models" / "hydrology" / "wflow"
    (root / "forcing").mkdir(parents=True)
    (root / "staticgeoms").mkdir(parents=True)
    (root / "wflow_sbm.toml").write_text(toml, encoding="utf-8")
    (root / "staticmaps.nc").write_bytes(b"STATICMAPS")
    (root / "forcing" / "inmaps_historical.nc").write_bytes(b"FORCING")
    if with_instate:
        (root / "instate").mkdir()
        (root / "instate" / "instates.nc").write_bytes(b"WARMSTATE")
    # hydromt-side artifacts Wflow.jl never reads at run time
    (root / "hydromt.log").write_text("ts 1", encoding="utf-8")
    (root / "hydromt_data.yml").write_text("a: 1", encoding="utf-8")
    (root / "staticgeoms" / "basins.geojson").write_text("{}", encoding="utf-8")
    for rel, payload in (extra or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
    return root


# ---------------------------------------------------------------------------
# Pointer discovery — the property a fixed file list fails
# ---------------------------------------------------------------------------

_TOML_WITH_SIDE_FILE = (
    _TOML
    + """
[input.lateral.river.lake]
path = "lake_rating_curve.csv"
"""
)


def _fixed_list_digest(root):
    """A stand-in for the implementation the design REJECTED.

    TOML + staticmaps.nc + instates.nc, enumerated. Present so the
    pointer-discovery falsifier can be shown to FAIL against it, which the
    acceptance criteria require -- otherwise "discovery works" is asserted
    against nothing.
    """
    h = hashlib.sha256()
    for rel in ("wflow_sbm.toml", "staticmaps.nc", "instate/instates.nc"):
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<absent>")
    return h.hexdigest()


def test_a_new_toml_pointer_brings_its_file_into_the_digest(tmp_path):
    """A hydromt setup_* that writes a TOML-referenced side file adds an input."""
    root = _model(
        tmp_path,
        toml=_TOML_WITH_SIDE_FILE,
        extra={"lake_rating_curve.csv": b"h,q\n1,2\n"},
    )
    assert "lake_rating_curve.csv" in model_file_set(root)


def test_editing_a_discovered_file_alone_moves_the_digest(tmp_path):
    """THE falsifier. The TOML is untouched; only the pointed-at file changes.

    A fixed list cannot see this: it would catch the POINTER appearing, because
    the TOML is hashed, but not a later in-place edit of the file pointed at.
    """
    root = _model(
        tmp_path,
        toml=_TOML_WITH_SIDE_FILE,
        extra={"lake_rating_curve.csv": b"h,q\n1,2\n"},
    )
    before, fixed_before = model_digest(root), _fixed_list_digest(root)

    (root / "lake_rating_curve.csv").write_bytes(b"h,q\n1,999\n")  # TOML untouched
    after, fixed_after = model_digest(root), _fixed_list_digest(root)

    assert after != before, "a discovered input changed and the digest did not move"
    # ...and the rejected implementation is blind to it, which is the point.
    assert fixed_after == fixed_before, (
        "the fixed-list stand-in was supposed to MISS this change; if it now "
        "catches it, this falsifier no longer discriminates"
    )


# ---------------------------------------------------------------------------
# Exclusions — structural, not a blocklist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel", ["hydromt.log", "hydromt_data.yml", "staticgeoms/basins.geojson"]
)
def test_artifacts_wflow_never_reads_are_excluded(tmp_path, rel):
    """Excluded because nothing points at them -- no blocklist involved.

    That is deliberate: if a future TOML DID point at one, Wflow would read it,
    it would be a runtime input, and it should enter the digest. A hardcoded
    blocklist would be wrong in exactly that case.
    """
    root = _model(tmp_path)
    before = model_digest(root)
    (root / rel).write_text("mutated", encoding="utf-8")
    assert model_digest(root) == before
    assert rel not in model_file_set(root)


# ---------------------------------------------------------------------------
# The absence marker
# ---------------------------------------------------------------------------


def test_absent_optional_input_differs_from_present_one(tmp_path):
    """Presence and absence of the warm state are different model states."""
    with_state = model_digest(_model(tmp_path / "a", with_instate=True))
    without = model_digest(_model(tmp_path / "b", with_instate=False))
    assert with_state != without


def test_absence_is_marked_not_omitted(tmp_path):
    """The marker, not omission: "the key exists and its target is missing" is
    not the same model as "there is no such key", and the two must not collide."""
    root = _model(tmp_path, with_instate=False)
    entries = dict(model_digest_entries(root))
    assert entries["instate/instates.nc"] == ABSENT

    no_key_toml = _TOML.replace('path_input = "instate/instates.nc"\n', "")
    other = _model(tmp_path / "other", toml=no_key_toml, with_instate=False)
    assert "instate/instates.nc" not in model_file_set(other)
    assert model_digest(other) != model_digest(root)


# ---------------------------------------------------------------------------
# Containment, determinism, and reporting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escape", ["../outside.nc", "../../etc/passwd", "forcing/../../outside.nc"]
)
def test_a_pointer_escaping_the_model_root_is_an_error(tmp_path, escape):
    """Not a silently widened digest: the fingerprint claims to cover the model."""
    root = _model(tmp_path, toml=_TOML.replace("staticmaps.nc", escape))
    with pytest.raises(ValueError, match="outside the model root"):
        model_digest(root)


def test_the_digest_does_not_depend_on_where_the_model_lives(tmp_path):
    """Determinism across platforms and checkouts: no absolute path is hashed.

    Two identical models at different absolute locations must agree, or an
    experiment's reference would break when the project moved.
    """
    a = _model(tmp_path / "here")
    b = _model(tmp_path / "somewhere" / "else" / "deeper")
    assert model_digest(a) == model_digest(b)


def test_entries_are_sorted_by_relative_path(tmp_path):
    """Sorted, never filesystem order -- the other half of determinism."""
    rels = [rel for rel, _ in model_digest_entries(_model(tmp_path))]
    assert rels == sorted(rels)
    assert all(not os.path.isabs(r) and "\\" not in r for r in rels)


def test_compare_names_the_changed_input(tmp_path):
    """A bare digest comparison can only say SOMETHING moved; the guard has to
    say what, or the operator cannot act on it."""
    root = _model(tmp_path)
    recorded = model_digest_entries(root)
    assert compare_model_digest(root, recorded) == []

    (root / "forcing" / "inmaps_historical.nc").write_bytes(b"DIFFERENT")
    diffs = compare_model_digest(root, recorded)
    assert len(diffs) == 1
    assert "forcing/inmaps_historical.nc" in diffs[0] and "changed" in diffs[0]


def test_compare_distinguishes_appeared_from_changed(tmp_path):
    """The absence marker has to survive into the REPORT, not just the digest."""
    root = _model(tmp_path, with_instate=False)
    recorded = model_digest_entries(root)
    (root / "instate").mkdir()
    (root / "instate" / "instates.nc").write_bytes(b"WARMSTATE")
    diffs = compare_model_digest(root, recorded)
    assert len(diffs) == 1 and "appeared" in diffs[0]


def test_a_missing_toml_is_named(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="wflow_sbm.toml"):
        model_digest(empty)
