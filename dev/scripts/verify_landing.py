"""Record gate evidence and verify its reuse without executing another gate.

Run from the checkout root. Evidence and logs should live in ignored scratch.
A successful command is not proof that required checks ran: the gate owner must
inspect skips and mark wholly skipped commands with --skipped.
For a reviewed follow-up, verify --delta-evidence and --delta-review combine
unchanged claims from earlier evidence with passing checks on the new tree.
This attests selective reuse, not a new full-suite pass. See dev-validation.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


class EvidenceError(ValueError):
    """Evidence cannot establish the requested gate."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(*, staged: bool = False) -> dict:
    if staged:
        if _git("diff", "--name-only") or _git(
            "ls-files", "--others", "--exclude-standard"
        ):
            raise EvidenceError(
                "Stage all changes and remove or ignore untracked files before recording"
            )
    elif _git("status", "--porcelain"):
        raise EvidenceError("Checkout has dirty or nonignored untracked files")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "tree": _git("write-tree") if staged else _git("rev-parse", "HEAD^{tree}"),
        "environment": {
            "pixi_lock_sha256": _hash(Path("pixi.lock")),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _inputs(paths: list[str]) -> dict[str, str]:
    return {str(Path(path).resolve()): _hash(Path(path)) for path in paths}


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data.get("version") not in (1, 2)
    ):
        raise EvidenceError("Unsupported evidence schema")
    if data["version"] == 2 and data.get("snapshot") != "staged":
        raise EvidenceError("Unsupported snapshot kind")
    for field in ("commit", "tree"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise EvidenceError(f"Invalid {field}")
    for field in ("environment", "inputs"):
        values = data.get(field)
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in values.items()
        ):
            raise EvidenceError(f"Invalid {field}")
    if set(data["environment"]) != {
        "pixi_lock_sha256",
        "python_executable",
        "python_version",
        "platform",
    }:
        raise EvidenceError("Invalid environment fields")
    gates = data.get("gates")
    if not isinstance(gates, dict):
        raise EvidenceError("Invalid gates")
    for name, gate in gates.items():
        if not isinstance(name, str) or not name or not isinstance(gate, dict):
            raise EvidenceError("Invalid gate")
        if (
            type(gate.get("exit_code")) is not int
            or gate.get("status") not in ("passed", "failed", "skipped")
            or not isinstance(gate.get("command"), list)
            or not gate["command"]
            or any(not isinstance(arg, str) for arg in gate["command"])
            or not isinstance(gate.get("log"), str)
            or not isinstance(gate.get("log_sha256"), str)
        ):
            raise EvidenceError(f"Invalid gate {name}")
    return data


def _record(args: argparse.Namespace) -> int:
    identity = _identity(staged=args.staged)
    inputs = _inputs(args.input)
    evidence = Path(args.evidence).resolve()
    log = Path(args.log).resolve()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise EvidenceError("Record needs an explicit command after --")
    if not args.gate.strip():
        raise EvidenceError("Gate name must not be empty")
    if evidence == log or str(log) in inputs or str(evidence) in inputs:
        raise EvidenceError("Evidence, log and inputs must be distinct")
    data = {
        "version": 2 if args.staged else 1,
        **identity,
        "inputs": inputs,
        "gates": {},
    }
    if args.staged:
        data["snapshot"] = "staged"
    if evidence.exists():
        data = _load(evidence)
        if (data["version"] == 2) != args.staged:
            raise EvidenceError("Existing evidence uses a different snapshot kind")
        if any(
            data[key] != value for key, value in {**identity, "inputs": inputs}.items()
        ):
            raise EvidenceError("Existing evidence has a different identity")
    if log.exists():
        raise EvidenceError("Log already exists; use a fresh log path")
    evidence.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if _identity(staged=args.staged) != identity or _inputs(args.input) != inputs:
        raise EvidenceError("Checkout, environment or inputs changed during the gate")
    status = "failed" if result.returncode else "skipped" if args.skipped else "passed"
    data["gates"][args.gate] = {
        "command": command,
        "exit_code": result.returncode,
        "status": status,
        "log": str(log),
        "log_sha256": _hash(log),
    }
    evidence.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {args.gate}: {status}")
    return int(result.returncode != 0)


def _check_identity(data: dict, current: dict) -> None:
    """Check evidence provenance and its external dependencies."""
    if current["environment"] != data["environment"]:
        raise EvidenceError("Environment identity differs")
    if _inputs(list(data["inputs"])) != data["inputs"]:
        raise EvidenceError("Input identity differs")
    if data["version"] == 2:
        # commit is the pre-test HEAD; tree is the tested staged content.
        _git("rev-parse", "--verify", f"{data['commit']}^{{commit}}")
        if _git("cat-file", "-t", data["tree"]) != "tree":
            raise EvidenceError("Recorded snapshot is not a Git tree")
    elif _git("rev-parse", f"{data['commit']}^{{tree}}") != data["tree"]:
        raise EvidenceError("Recorded commit/tree disagree")


def _check_gate(data: dict, name: str) -> None:
    """Require a passing, intact log, allowing full to satisfy fast."""
    gate = data["gates"].get(name)
    if name == "test-fast" and (
        gate is None or gate["status"] != "passed" or gate["exit_code"] != 0
    ):
        gate = data["gates"].get("test-full")
    if gate is None or gate["status"] != "passed" or gate["exit_code"] != 0:
        raise EvidenceError(f"Required gate did not pass: {name}")
    if _hash(Path(gate["log"])) != gate["log_sha256"]:
        raise EvidenceError(f"Gate log changed: {name}")


def _verify_delta(args: argparse.Namespace, base: dict, current: dict) -> int:
    """Verify a pinned human review plus fresh evidence for the entire delta."""
    if args.allow_path:
        raise EvidenceError("Delta review cannot be combined with --allow-path")
    delta = _load(Path(args.delta_evidence))
    _check_identity(delta, current)
    if delta["tree"] != current["tree"]:
        raise EvidenceError("Delta evidence must cover the current tree")
    if delta["inputs"] != base["inputs"]:
        raise EvidenceError("Delta evidence must retain the same named inputs")
    review = json.loads(Path(args.delta_review).read_text(encoding="utf-8"))
    fields = {
        "base_commit",
        "candidate_tree",
        "paths",
        "retained_gates",
        "require_gates",
    }
    if not isinstance(review, dict) or set(review) != fields:
        raise EvidenceError("Invalid delta review fields")
    if (base["version"] == 1 and review["base_commit"] != base["commit"]) or review[
        "candidate_tree"
    ] != current["tree"]:
        raise EvidenceError("Delta review does not match base and candidate")
    if _git("rev-parse", f"{review['base_commit']}^{{tree}}") != base["tree"]:
        raise EvidenceError("Delta review base must commit the tested tree")
    _git("merge-base", "--is-ancestor", review["base_commit"], current["commit"])
    for field in ("paths", "retained_gates"):
        mapping = review[field]
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(reason, str)
                or not reason.strip()
                for key, reason in mapping.items()
            )
        ):
            raise EvidenceError(f"Delta review needs explicit reasons in {field}")
    required = review["require_gates"]
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) or not name.strip() for name in required)
    ):
        raise EvidenceError("Delta review needs fresh required gates")
    changed = set(
        _git("diff", "--name-only", "--no-renames", "-z", base["tree"], "HEAD")
        .rstrip("\0")
        .split("\0")
    )
    if changed != set(review["paths"]):
        raise EvidenceError("Delta review must classify exactly every changed path")
    retained = review["retained_gates"]
    if not set(retained) <= set(args.require_gate):
        raise EvidenceError("Retained gates must be requested by --require-gate")
    if set(retained) & set(required):
        raise EvidenceError("A gate cannot be both retained and freshly required")
    for name in required:
        _check_gate(delta, name)
    for name in args.require_gate:
        _check_gate(base if name in retained else delta, name)
    print(f"Retained earlier claims: {', '.join(sorted(retained))}")
    print(
        f"Fresh delta checks: {', '.join(sorted(set(required) | (set(args.require_gate) - set(retained))))}"
    )
    print("Reviewed delta evidence verified; no gates executed")
    return 0


def _verify(args: argparse.Namespace) -> int:
    data = _load(Path(args.evidence))
    current = _identity()
    _check_identity(data, current)
    if bool(args.delta_evidence) != bool(args.delta_review):
        raise EvidenceError("Use --delta-evidence and --delta-review together")
    if args.delta_evidence:
        return _verify_delta(args, data, current)
    allowed = set(args.allow_path)
    for path in allowed:
        if Path(path).suffix.lower() != ".md" or "*" in path or "?" in path:
            raise EvidenceError("Allowed changes must be exact Markdown paths")
    if current["tree"] != data["tree"]:
        changed = set(
            _git(
                "diff", "--name-only", "--no-renames", data["tree"], "HEAD"
            ).splitlines()
        )
        if not changed <= allowed:
            raise EvidenceError(
                f"Candidate changed unallowed paths: {sorted(changed - allowed)}"
            )
        # Deletion/rename loses evidence context even when its old path is allowed.
        if _git("diff", "--name-only", "--diff-filter=D", data["tree"], "HEAD"):
            raise EvidenceError("Documentation deletions are not reusable")
    for name in args.require_gate:
        _check_gate(data, name)
    print("Required evidence verified; no gates executed")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the record or verify command, reporting refusals with exit status 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    record = sub.add_parser("record")
    record.add_argument("--evidence", required=True)
    record.add_argument("--gate", required=True)
    record.add_argument("--log", required=True)
    record.add_argument("--input", action="append", default=[])
    record.add_argument("--skipped", action="store_true")
    record.add_argument(
        "--staged",
        action="store_true",
        help="Record the staged tree before commit; requires no unstaged or nonignored untracked files",
    )
    record.add_argument("command", nargs=argparse.REMAINDER)
    verify = sub.add_parser("verify")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--require-gate", action="append", required=True)
    verify.add_argument("--allow-path", action="append", default=[])
    verify.add_argument("--delta-evidence")
    verify.add_argument("--delta-review")
    args = parser.parse_args(argv)
    try:
        return _record(args) if args.operation == "record" else _verify(args)
    except (
        EvidenceError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"Evidence refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
