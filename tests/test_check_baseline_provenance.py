"""R7-21: the manifest records which branch/commit wrote it.

The baseline fixture (`project_dir`) is **untracked**, so it belongs to no
branch: every branch, worktree and session that runs a workflow writes into the
same tree. `check` therefore answers "does the tree match the manifest" for
whichever branch ran LAST, not for the branch you are on -- a green check can
mean someone else's code is consistent with your manifest.

That is not hypothetical. A `basin_area.png` produced on `feat/outputs-figures`
sat in the shared fixture for days and was read as the pre-R07 baseline
reference; only a byte-size mismatch at the R07 gate forced the question
(pre-board record 45e3a564 R7-3). These tests pin the stamp that makes such a
misattribution visible instead of silent.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402


def test_provenance_reports_branch_commit_and_dirtiness():
    prov = cb.git_provenance()
    assert prov is not None, "this repo is a git checkout; provenance must resolve"
    assert prov["commit"] and len(prov["commit"]) >= 7
    assert prov["branch"]
    assert isinstance(prov["dirty"], bool)


def test_provenance_is_best_effort_outside_a_repository(tmp_path, monkeypatch):
    """It must never be the reason a baseline command fails."""
    # Keep real git discovery, but prevent a scratch fixture from discovering
    # the hosting checkout above it. This directory itself has no repository.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
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
    m.write_text(
        json.dumps(
            {
                "version": cb.MANIFEST_VERSION,
                "project_dir": "test_case/test_local",
                "recorded_by": recorded_by,
                "targets": {},
            }
        ),
        encoding="utf-8",
    )
    return m


def _check(manifest, tmp_path):
    """Run `check` against an empty target set so only provenance output
    varies; a manifest with no targets trivially matches."""
    import argparse

    args = argparse.Namespace(
        manifest=manifest,
        project_dir=str(tmp_path / "proj"),
        workflow=None,
        tolerance=0.0,
    )
    (tmp_path / "proj").mkdir(exist_ok=True)
    return args


def test_check_warns_when_another_branch_recorded_the_manifest(
    tmp_path, capsys, monkeypatch
):
    """THE R7-3 SCENARIO, simulated: the manifest was written from a branch
    that is not the one being checked from."""
    monkeypatch.setattr(cb, "TARGETS", [])
    m = _manifest(
        tmp_path,
        {
            "branch": "feat/outputs-figures",
            "commit": "e917a8e" + "0" * 33,
            "dirty": False,
        },
    )
    rc = cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "feat/outputs-figures" in out
    assert "SHARED BY EVERY BRANCH" in out
    # These fixtures empty `TARGETS` so only the banner varies, so the verdict
    # is NOT CHECKED rather than a pass -- correct, and not caused by
    # provenance. The claim being pinned is that the warning does not turn a
    # run into a DIFFERENCE; `test_a_warning_does_not_flip_a_real_pass` below
    # carries the same claim against a fixture that actually compares something.
    assert rc == cb.EXIT_NOT_CHECKED
    assert rc != 1, "provenance is advisory -- it must not report a difference"


def test_same_branch_different_commit_is_a_softer_note(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cb, "TARGETS", [])
    cur = cb.git_provenance()
    m = _manifest(
        tmp_path,
        {
            "branch": cur["branch"],
            "commit": "f" * 40,
            "dirty": False,
        },
    )
    cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "Same branch, different commit" in out
    assert "SHARED BY EVERY BRANCH" not in out


def test_a_pre_stamp_manifest_says_so_rather_than_pretending(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(cb, "TARGETS", [])
    m = _manifest(tmp_path, None)
    rc = cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "predates provenance stamping" in out
    assert rc == cb.EXIT_NOT_CHECKED
    assert rc != 1, "an unstamped manifest is a note, not a difference"


def test_matching_provenance_is_quiet(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cb, "TARGETS", [])
    m = _manifest(tmp_path, cb.git_provenance())
    cb.cmd_check(_check(m, tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" not in out and "predates" not in out


# --- the flag must describe the checkout, not this command's own output -------


def test_provenance_is_captured_before_record_writes_anything(monkeypatch, tmp_path):
    """`record` writes the TRACKED sidecar reference tables while fingerprinting.

    Reading provenance afterwards made `dirty` describe this command's own
    output: R13's pass-1 re-record stamped `dirty: true` from a clean checkout
    detached at 9cbb72a, with a porcelain status containing only the
    `dev/baseline/indicator_ref/*.csv` that `record` had just written. Pass 2
    rewrote no sidecar and stamped `false` from a checkout in the same
    condition — the control case, unplanned.

    The ordering IS the fix, so the ordering is what this pins. A test on the
    flag's value alone would pass against the broken version on any checkout
    that happened to be dirty already.
    """
    calls = []

    monkeypatch.setattr(
        cb, "git_provenance", lambda *a, **k: calls.append("provenance") or {}
    )
    monkeypatch.setattr(
        cb,
        "compute_manifest",
        lambda *a, **k: (calls.append("fingerprint") or {}, []),
    )
    monkeypatch.setattr(
        cb,
        "record_discharge",
        lambda *a, **k: (calls.append("write discharge sidecar") or {}, []),
    )
    monkeypatch.setattr(
        cb,
        "record_indicator",
        lambda *a, **k: (calls.append("write indicator sidecar") or {}, []),
    )

    args = argparse.Namespace(
        workflow=None,
        manifest=tmp_path / "manifest.json",
        project_dir=str(tmp_path / "project"),
    )
    assert cb.cmd_record(args) == 0
    assert calls[0] == "provenance", calls


def test_the_recorded_flag_is_the_one_that_was_captured(monkeypatch, tmp_path):
    """Not a tautology: it pins that the captured value REACHES the payload,
    which a fix that captured early and then called `git_provenance()` again in
    the payload would fail."""
    stamp = {"branch": "b", "commit": "c" * 12, "dirty": False}
    monkeypatch.setattr(cb, "git_provenance", lambda *a, **k: stamp)
    monkeypatch.setattr(cb, "compute_manifest", lambda *a, **k: ({}, []))
    monkeypatch.setattr(cb, "record_discharge", lambda *a, **k: ({}, []))
    monkeypatch.setattr(cb, "record_indicator", lambda *a, **k: ({}, []))

    args = argparse.Namespace(
        workflow=None,
        manifest=tmp_path / "manifest.json",
        project_dir=str(tmp_path / "project"),
    )
    assert cb.cmd_record(args) == 0
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["recorded_by"] == stamp


def test_a_warning_does_not_flip_a_real_pass(tmp_path, capsys, monkeypatch):
    """The advisory claim, against a fixture that actually compares a target.

    The sibling tests empty `TARGETS` to isolate the banner, so since
    2026-09-17 they land on NOT CHECKED and can no longer witness "the warning
    does not change a PASS". This one can: one real target, recorded and
    matching, with a foreign branch stamped on the manifest.
    """
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    target = proj / "sample.yml"
    target.write_text("a: 1" + chr(10), encoding="utf-8")
    template = "{project_dir}/sample.yml"
    monkeypatch.setattr(cb, "TARGETS", [("build_model", "yaml", template)])
    # The manifest key must be byte-identical to what `resolve` produces, not
    # `str(target)` -- on Windows those differ by separator alone, and the check
    # would report the same file as both missing and unrecorded.
    key = cb.resolve(template, str(proj))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": cb.MANIFEST_VERSION,
                "project_dir": "test_case/test_local",
                "recorded_by": {
                    "branch": "feat/outputs-figures",
                    "commit": "e917a8e" + "0" * 33,
                    "dirty": False,
                },
                "targets": {key: cb.fingerprint_yaml(key)},
            }
        ),
        encoding="utf-8",
    )

    rc = cb.cmd_check(_check(manifest, tmp_path))

    out = capsys.readouterr().out
    assert "SHARED BY EVERY BRANCH" in out, "the advisory warning must still fire"
    assert rc == 0, "provenance is advisory -- it must not change the verdict"
    assert "OK -" in out


def test_run_record_fingerprint_ignores_provenance_and_checkout(tmp_path):
    # A new commit, invocation or settings section must not move the gate; the
    # same configuration recorded from another worktree must not either.
    record = {
        "loaded_config": {
            "project": {"catalog": str(cb.REPO_ROOT / "test_case" / "c.yml")}
        },
        "toolbox": {"commit": "a"},
        "advanced_settings": {"batching": {"threads": 1}},
    }
    first = tmp_path / "first.yml"
    first.write_text(yaml.safe_dump(record), encoding="utf-8")
    record["toolbox"]["commit"] = "b"
    record["advanced_settings"]["batching"]["threads"] = 4
    second = tmp_path / "second.yml"
    second.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert cb.fingerprint_run_record(first) == cb.fingerprint_run_record(second)
    assert "test_case/c.yml" in json.dumps(cb._repo_relative(record["loaded_config"]))
    record["loaded_config"]["climate"] = {"selected": "era5"}
    second.write_text(yaml.safe_dump(record), encoding="utf-8")
    assert cb.fingerprint_run_record(first) != cb.fingerprint_run_record(second)
