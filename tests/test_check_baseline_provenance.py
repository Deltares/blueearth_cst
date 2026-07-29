"""R7-21: the manifest records which branch/commit wrote it.

The baseline fixture (`project_dir`) is **untracked**, so it belongs to no
branch: every branch, worktree and session that runs a workflow writes into the
same tree. `check` therefore answers "does the tree match the manifest" for
whichever branch ran LAST, not for the branch you are on -- a green check can
mean someone else's code is consistent with your manifest.

That is not hypothetical. A `basin_area.png` produced on `feat/outputs-figures`
sat in the shared fixture for days and was read as the pre-R07 baseline
reference; only a byte-size mismatch at the R07 gate forced the question
(dev/followups.md R7-3). These tests pin the stamp that makes such a
misattribution visible instead of silent.
"""

import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402


def test_provenance_reports_branch_commit_and_dirtiness():
    prov = cb.git_provenance()
    assert prov is not None, "this repo is a git checkout; provenance must resolve"
    assert prov["commit"] and len(prov["commit"]) >= 7
    assert prov["branch"]
    assert isinstance(prov["dirty"], bool)


def test_provenance_is_best_effort_outside_a_repository(tmp_path):
    """It must never be the reason a baseline command fails."""
    assert cb.git_provenance(tmp_path) is None


def test_provenance_survives_a_missing_git(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise OSError("git not found")
    monkeypatch.setattr(subprocess, "run", _boom)
    assert cb.git_provenance(tmp_path) is None


def test_format_is_readable_and_flags_dirty():
    assert cb.format_provenance(None) == "(unrecorded)"
    s = cb.format_provenance(
        {"branch": "main", "commit": "0123456789abcdef", "dirty": False}
    )
    assert s == "main@0123456789ab"
    assert cb.format_provenance(
        {"branch": "x", "commit": "abcdef1234567", "dirty": True}
    ).endswith("+dirty")


def _manifest(tmp_path, recorded_by):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "version": cb.MANIFEST_VERSION,
        "project_dir": "test_case/test_local",
        "recorded_by": recorded_by,
        "targets": {},
    }), encoding="utf-8")
    return m


def _check(manifest, tmp_path):
    """Run `check` against an empty target set so only provenance output
    varies; a manifest with no targets trivially matches."""
    import argparse
    args = argparse.Namespace(
        manifest=manifest, project_dir=str(tmp_path / "proj"),
        workflow=None, tolerance=0.0,
    )
    (tmp_path / "proj").mkdir(exist_ok=True)
    return args


def test_check_warns_when_another_branch_recorded_the_manifest(tmp_path, capsys):
    """THE R7-3 SCENARIO, simulated: the manifest was written from a branch
    that is not the one being checked from."""
    m = _manifest(tmp_path, {
        "branch": "feat/outputs-figures",
        "commit": "e917a8e" + "0" * 33,
        "dirty": False,
    })
    rc = cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "feat/outputs-figures" in out
    assert "SHARED BY EVERY BRANCH" in out
    assert rc == 0, "provenance is advisory -- it must not change the verdict"


def test_same_branch_different_commit_is_a_softer_note(tmp_path, capsys):
    cur = cb.git_provenance()
    m = _manifest(tmp_path, {
        "branch": cur["branch"], "commit": "f" * 40, "dirty": False,
    })
    cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "Same branch, different commit" in out
    assert "SHARED BY EVERY BRANCH" not in out


def test_a_pre_stamp_manifest_says_so_rather_than_pretending(tmp_path, capsys):
    m = _manifest(tmp_path, None)
    rc = cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "predates provenance stamping" in out
    assert rc == 0


def test_matching_provenance_is_quiet(tmp_path, capsys):
    m = _manifest(tmp_path, cb.git_provenance())
    cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" not in out and "predates" not in out
