"""Exercise landing evidence through its public command line."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev/scripts/verify_landing.py"


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitignore").write_text(
        "evidence.json\ndelta.json\nreview.json\n*.log\n", encoding="utf-8"
    )
    for name in ("pixi.lock", "code.py", "guide.md", "input.txt"):
        (tmp_path / name).write_text("original\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def _cli(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=root, capture_output=True, text=True
    )


def _record(root, *extra, command=None):
    return _cli(
        root,
        "record",
        "--evidence",
        "evidence.json",
        "--gate",
        "suite",
        "--log",
        "gate.log",
        "--input",
        "input.txt",
        *extra,
        "--",
        *(command or [sys.executable, "-c", "print('passed')"]),
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "failed",
        "skipped",
        "schema",
        "log",
        "input",
        "environment",
        "full",
        "append",
    ],
)
def test_evidence_integrity(repo, fault):
    extra = ["--skipped"] if fault == "skipped" else []
    command = (
        [sys.executable, "-c", "raise SystemExit(2)"] if fault == "failed" else None
    )
    result = _record(repo, *extra, command=command)
    assert result.returncode == (1 if fault == "failed" else 0), result.stderr
    evidence = repo / "evidence.json"
    if fault in {"schema", "environment"}:
        data = json.loads(evidence.read_text())
        if fault == "schema":
            data["version"] = "invalid"
        else:
            data["environment"]["python_version"] = "other"
        evidence.write_text(json.dumps(data))
    if fault in {"log", "input"}:
        (repo / ("gate.log" if fault == "log" else "input.txt")).write_text("changed")
    if fault in {"full", "append"}:
        result = _cli(
            repo,
            "record",
            "--evidence",
            "evidence.json",
            "--gate",
            "test-full",
            "--log",
            "full.log",
            "--input",
            "input.txt",
            "--",
            sys.executable,
            "-c",
            "print(123)",
        )
        assert result.returncode == 0, result.stderr
    gate = (
        "absent" if fault == "missing" else "test-fast" if fault == "full" else "suite"
    )
    result = _cli(repo, "verify", "--evidence", "evidence.json", "--require-gate", gate)
    assert result.returncode == (0 if fault in {None, "full", "append"} else 1), (
        result.stderr
    )


@pytest.mark.parametrize(
    "change,allow,expected",
    [
        ("guide.md", True, 0),
        ("guide.md", False, 1),
        ("code.py", True, 1),
        ("dirty", False, 1),
    ],
)
def test_candidate_tree_boundaries(repo, change, allow, expected):
    assert _record(repo).returncode == 0
    name = "code.py" if change == "dirty" else change
    (repo / name).write_text("changed")
    if change != "dirty":
        _git(repo, "add", name)
        _git(repo, "commit", "-m", "change")
    args = ["--allow-path", name] if allow else []
    result = _cli(
        repo, "verify", "--evidence", "evidence.json", "--require-gate", "suite", *args
    )
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize("fault", ["dirty", "untracked", "during", "append-mismatch"])
def test_record_refuses_unstable_identity(repo, fault):
    if fault == "untracked":
        (repo / "untracked.py").write_text("pass")
    if fault == "dirty":
        (repo / "code.py").write_text("changed")
    if fault == "append-mismatch":
        assert _record(repo).returncode == 0
        (repo / "guide.md").write_text("changed")
        _git(repo, "add", "guide.md")
        _git(repo, "commit", "-m", "changed")
    command = (
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('code.py').write_text('changed')",
        ]
        if fault == "during"
        else None
    )
    assert _record(repo, command=command).returncode == 1


@pytest.mark.parametrize("change", ["modify", "add", "delete", "rename"])
def test_staged_evidence_survives_commit_without_rerunning(repo, change):
    original_head = _git(repo, "rev-parse", "HEAD").stdout
    if change == "modify":
        (repo / "code.py").write_text("changed\n")
    elif change == "add":
        (repo / "new.py").write_text("new\n")
    elif change == "delete":
        (repo / "code.py").unlink()
    else:
        (repo / "code.py").rename(repo / "renamed.py")
    _git(repo, "add", "-A")
    index = _git(repo, "ls-files", "--stage").stdout
    result = _record(repo, "--staged")
    assert result.returncode == 0, result.stderr
    assert _git(repo, "rev-parse", "HEAD").stdout == original_head
    assert _git(repo, "ls-files", "--stage").stdout == index
    log = (repo / "gate.log").read_bytes()
    _git(repo, "commit", "-m", "tested change")
    result = _cli(
        repo, "verify", "--evidence", "evidence.json", "--require-gate", "suite"
    )
    assert result.returncode == 0, result.stderr
    assert (repo / "gate.log").read_bytes() == log
    data = json.loads((repo / "evidence.json").read_text())
    assert data["version"] == 2
    assert (
        data["tree"] == _git(repo, "rev-parse", "HEAD^{tree}").stdout.decode().strip()
    )


@pytest.mark.parametrize(
    "fault", ["unstaged", "untracked", "during", "different-commit"]
)
def test_staged_evidence_refuses_untested_content(repo, fault):
    (repo / "code.py").write_text("tested\n")
    _git(repo, "add", "code.py")
    if fault == "unstaged":
        (repo / "code.py").write_text("not staged\n")
    elif fault == "untracked":
        (repo / "untracked.py").write_text("not staged\n")
    command = (
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('code.py').write_text('mutated')",
        ]
        if fault == "during"
        else None
    )
    result = _record(repo, "--staged", command=command)
    if fault != "different-commit":
        assert result.returncode == 1
        assert not (repo / "evidence.json").exists()
        return
    assert result.returncode == 0, result.stderr
    (repo / "code.py").write_text("untested change\n")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-m", "untested change")
    result = _cli(
        repo, "verify", "--evidence", "evidence.json", "--require-gate", "suite"
    )
    assert result.returncode == 1
    assert "unallowed paths" in result.stderr


def test_staged_base_and_followup_support_reviewed_delta(repo):
    (repo / "code.py").write_text("first change\n")
    _git(repo, "add", "code.py")
    assert _record(repo, "--staged").returncode == 0
    _git(repo, "commit", "-m", "first change")
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    (repo / "code.py").write_text("followup\n")
    _git(repo, "add", "code.py")
    result = _cli(
        repo,
        "record",
        "--staged",
        "--evidence",
        "delta.json",
        "--gate",
        "focused",
        "--log",
        "delta.log",
        "--input",
        "input.txt",
        "--",
        sys.executable,
        "-c",
        "print('focused pass')",
    )
    assert result.returncode == 0, result.stderr
    _git(repo, "commit", "-m", "followup")
    review = {
        "base_commit": base_commit,
        "candidate_tree": _git(repo, "rev-parse", "HEAD^{tree}")
        .stdout.decode()
        .strip(),
        "paths": {"code.py": "Followup covered by focused check."},
        "retained_gates": {"suite": "Other behavior unchanged."},
        "require_gates": ["focused"],
    }
    (repo / "review.json").write_text(json.dumps(review))
    result = _cli(
        repo,
        "verify",
        "--evidence",
        "evidence.json",
        "--delta-evidence",
        "delta.json",
        "--delta-review",
        "review.json",
        "--require-gate",
        "suite",
        "--require-gate",
        "focused",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        (None, "Retained earlier claims: suite"),
        ("unreviewed", "exactly every changed path"),
        ("stale-review", "does not match base and candidate"),
        ("reason", "explicit reasons"),
        ("missing", "Required gate did not pass"),
        ("failed", "Required gate did not pass"),
        ("skipped", "Required gate did not pass"),
        ("base-log", "Gate log changed"),
        ("delta-log", "Gate log changed"),
        ("environment", "Environment identity differs"),
        ("input-omitted", "same named inputs"),
        ("later-edit", "current tree"),
        ("overlap", "both retained and freshly required"),
        ("no-fresh-check", "fresh required gates"),
        ("no-review", "together"),
        ("allow-path", "cannot be combined"),
    ],
)
def test_reviewed_delta_reuse(repo, fault, diagnostic):
    assert _record(repo).returncode == 0
    base = json.loads((repo / "evidence.json").read_text())
    (repo / "code.py").write_text("console wording changed\n")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-m", "wording")
    result = _cli(
        repo,
        "record",
        "--evidence",
        "delta.json",
        "--gate",
        "console",
        "--log",
        "delta.log",
        "--input",
        "input.txt",
        *(["--skipped"] if fault == "skipped" else []),
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(1)" if fault == "failed" else "print('checked wording')",
    )
    assert result.returncode == (1 if fault == "failed" else 0), result.stderr
    delta = json.loads((repo / "delta.json").read_text())
    review = {
        "base_commit": base["commit"],
        "candidate_tree": delta["tree"],
        "paths": {
            "code.py": "Only human-readable wording changed; console check covers it."
        },
        "retained_gates": {
            "suite": "Runtime control flow and output contracts unchanged."
        },
        "require_gates": ["console"],
    }
    if fault == "unreviewed":
        review["paths"] = {"guide.md": "Wrong path"}
    if fault == "stale-review":
        review["candidate_tree"] = base["tree"]
    if fault == "reason":
        review["paths"]["code.py"] = " "
    if fault == "missing":
        review["require_gates"] = ["missing"]
    if fault == "overlap":
        review["require_gates"] = ["suite"]
    if fault == "no-fresh-check":
        review["require_gates"] = []
    if fault in {"base-log", "delta-log"}:
        (repo / ("gate.log" if fault == "base-log" else "delta.log")).write_text(
            "edited"
        )
    if fault == "environment":
        delta["environment"]["python_version"] = "different"
    if fault == "input-omitted":
        delta["inputs"] = {}
    (repo / "delta.json").write_text(json.dumps(delta))
    (repo / "review.json").write_text(json.dumps(review))
    if fault == "later-edit":
        (repo / "code.py").write_text("another change")
        _git(repo, "add", "code.py")
        _git(repo, "commit", "-m", "later")
    result = _cli(
        repo,
        "verify",
        "--evidence",
        "evidence.json",
        "--require-gate",
        "suite",
        "--delta-evidence",
        "delta.json",
        *([] if fault == "no-review" else ["--delta-review", "review.json"]),
        *(["--allow-path", "guide.md"] if fault == "allow-path" else []),
    )
    assert result.returncode == (0 if fault is None else 1), result.stderr
    assert diagnostic in result.stdout + result.stderr
