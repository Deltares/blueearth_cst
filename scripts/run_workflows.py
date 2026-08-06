"""Enabled-aware wrapper over the three CST Snakefiles (design §7).

Reads a full-orchestration `--configfile` YAML, checks each
`workflows.<name>.enabled` flag, and invokes `snakemake -s Snakefile_<name>
--configfile <cfg> ...` for exactly the enabled workflows, in the fixed order
model_creation -> climate_projections -> climate_experiment.

This is the evolution of the run_snake_test.cmd / run_snake_docker.sh runners --
a *runner over* the three Snakefiles, NOT a fourth Snakemake entry point. The
Snakefiles do not read `enabled:`; the flag governs this wrapper only.

Contract (pinned, design §7 (a)-(g)):

 (a) Full-orchestration configs only: a `workflows:` section with all three
     subsections each carrying an `enabled:` key. The single-workflow
     projections configs (no `workflows:` section) are direct `snakemake -s`
     inputs, not wrapper inputs.
 (b) A missing `workflows:` section or a missing `<name>.enabled` subkey is a
     HARD ERROR (nonzero exit, message naming the absent key) -- never a silent
     default to true.
 (c) Each `enabled:` value must PARSE to a real boolean (isinstance(v, bool) on
     the post-yaml.safe_load value). YAML 1.1 resolves unquoted
     true/false/yes/no/on/off to booleans, so all those spellings are accepted;
     quoted strings ("true"), integers (1/0), or any non-bool are REJECTED.
 (d) Enabled workflows run in fixed order; on the first nonzero snakemake exit
     the wrapper STOPS and returns that exit code (does not continue).
 (e) --cores N (default 3) is forwarded to every invocation; args after a `--`
     sentinel are appended verbatim to every invocation. --configfile is
     supplied by the wrapper.
 (f) Per-workflow flags are preserved from a hardcoded map matching the runners:
     --keep-going on climate_projections only.
 (g) Every valid wrapper invocation creates and atomically finalizes a unique
     `<project_dir>/config/runs/invocations/*.json` lifecycle manifest -- a
     SIBLING of the per-workflow `config/runs/<workflow>/<digest>/` bundles,
     because an invocation spans workflows. Its runner-side config digest
     covers the source YAML plus resolved advanced settings;
     passthrough `--config` overrides are recorded but intentionally excluded,
     because the Snakefile snapshot owns Snakemake's authoritative merged config.

Disabling a workflow neither deletes its prior outputs nor guarantees downstream
freshness: the wrapper invokes each Snakefile independently with no
prerequisite-freshness check -- identical to invoking a single Snakefile
directly today. A user who disables a prerequisite owns the staleness of what
downstream consumes.

Usage::

    python scripts/run_workflows.py --config config/workflows/snake_config_model_test.yml
    python scripts/run_workflows.py --config <cfg> --cores 4 -- --dry-run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Make the plain source tree importable when this file is executed directly as
# ``python scripts/run_workflows.py`` rather than imported by pytest.
_REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_PATH))

from blueearth_cst.shared.provenance import (  # noqa: E402
    effective_config_digest,
    file_sha256,
)
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS  # noqa: E402

# Fixed run order (model -> projections -> experiment). Each maps to its
# Snakefile and the per-workflow flags preserved verbatim from the runners
# (design §7(f)): --keep-going on climate_projections only.
WORKFLOW_ORDER = ("model_creation", "climate_projections", "climate_experiment")

SNAKEFILE = {
    "model_creation": "Snakefile_model_creation",
    "climate_projections": "Snakefile_climate_projections",
    "climate_experiment": "Snakefile_climate_experiment",
}

PER_WORKFLOW_FLAGS = {
    "model_creation": [],
    "climate_projections": ["--keep-going"],
    "climate_experiment": [],
}

# Repo root = parent of scripts/. Snakefiles and config paths are repo-root
# relative and the wrapper is invoked from repo root, mirroring the runners.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SENSITIVE_KEY_RE = re.compile(
    r"api[-_]?key|auth|credential|pass(?:word|wd)?|private[-_]?key|secret|token",
    re.IGNORECASE,
)


class ConfigError(Exception):
    """Raised for a config that violates the wrapper's contract (a)-(c)."""


def read_enabled_flags(config_path: str) -> dict[str, bool]:
    """Parse and validate the per-workflow enabled flags (contract (a)-(c)).

    Raises ConfigError on: no `workflows:` section, a missing `<name>` or
    `<name>.enabled` key, or an `enabled:` value that does not parse to a bool.
    """
    cfg = _read_config(config_path)
    return _enabled_flags(cfg, config_path)


def _read_config(config_path: str) -> Mapping[str, Any]:
    """Load a wrapper source config as a YAML mapping."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ConfigError(f"{config_path}: config is not a mapping")
    return config


def _enabled_flags(
    cfg: Mapping[str, Any], config_path: str
) -> dict[str, bool]:
    """Validate and return workflow enablement from an already loaded config."""

    if "workflows" not in cfg:
        raise ConfigError(
            f"{config_path}: no 'workflows:' section -- this is not a full "
            f"orchestration config. The single-workflow projections configs are "
            f"direct 'snakemake -s' inputs, not wrapper inputs."
        )
    workflows = cfg["workflows"]
    if not isinstance(workflows, dict):
        raise ConfigError(f"{config_path}: 'workflows:' is not a mapping")

    flags: dict[str, bool] = {}
    for name in WORKFLOW_ORDER:
        if name not in workflows or not isinstance(workflows[name], dict):
            raise ConfigError(
                f"{config_path}: missing 'workflows.{name}' section "
                f"(required for a full orchestration config)"
            )
        section = workflows[name]
        if "enabled" not in section:
            raise ConfigError(
                f"{config_path}: missing 'workflows.{name}.enabled' key"
            )
        value = section["enabled"]
        if not isinstance(value, bool):
            raise ConfigError(
                f"{config_path}: 'workflows.{name}.enabled' must parse to a "
                f"boolean (got {value!r} of type {type(value).__name__}); use an "
                f"unquoted true/false (yes/no/on/off also accepted), not a "
                f"quoted string or integer"
            )
        flags[name] = value
    return flags


def _project_dir(cfg: Mapping[str, Any], config_path: str) -> Path:
    """Resolve the configured project output root from a wrapper config."""
    project = cfg.get("project")
    if not isinstance(project, Mapping):
        raise ConfigError(f"{config_path}: missing or invalid 'project:' section")
    project_dir = project.get("project_dir")
    if not isinstance(project_dir, (str, os.PathLike)) or not os.fspath(
        project_dir
    ).strip():
        raise ConfigError(
            f"{config_path}: 'project.project_dir' must be a non-empty path"
        )
    path = Path(project_dir).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT_PATH / path
    return path.resolve()


def build_command(
    name: str, config_path: str, cores: int, extra: list[str]
) -> list[str]:
    """Assemble the snakemake argv for one workflow (contract (e)/(f))."""
    return [
        "snakemake",
        "all",
        "-c", str(cores),
        "-s", SNAKEFILE[name],
        "--configfile", config_path,
        *PER_WORKFLOW_FLAGS[name],
        *extra,
    ]


def run(config_path: str, cores: int, extra: list[str]) -> int:
    """Invoke each enabled workflow in fixed order; stop on first nonzero exit
    and return that code (contract (d)). Returns 0 if all enabled workflows
    succeed (or all are disabled)."""
    cfg = _read_config(config_path)
    flags = _enabled_flags(cfg, config_path)
    project_dir = _project_dir(cfg, config_path)
    manifest_path, manifest = _initialize_manifest(
        cfg=cfg,
        config_path=config_path,
        project_dir=project_dir,
        flags=flags,
        cores=cores,
        extra=extra,
    )
    _write_json_atomic(manifest_path, manifest)

    exit_code = 0
    try:
        for name in WORKFLOW_ORDER:
            workflow = manifest["workflows"][name]
            if not flags[name]:
                print(f"[run_workflows] skipping {name} (enabled: false)")
                continue
            cmd = build_command(name, config_path, cores, extra)
            workflow["command"] = sanitize_argv(cmd)
            workflow["status"] = "running"
            print(f"[run_workflows] {name}: {' '.join(sanitize_argv(cmd))}")
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            workflow["exit_code"] = result.returncode
            if result.returncode != 0:
                workflow["status"] = "failed"
                exit_code = result.returncode
                print(
                    f"[run_workflows] {name} exited {result.returncode}; stopping "
                    f"(later workflows not invoked)."
                )
                break
            workflow["status"] = "succeeded"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        for workflow in manifest["workflows"].values():
            if workflow["status"] == "running":
                workflow["status"] = "failed"
        _mark_pending_not_run(manifest)
        _finalize_manifest(manifest_path, manifest, exit_code=None)
        raise

    if exit_code != 0:
        manifest["status"] = "failed"
        _mark_pending_not_run(manifest)
    else:
        manifest["status"] = "succeeded"
    _finalize_manifest(manifest_path, manifest, exit_code=exit_code)
    return exit_code


def sanitize_argv(argv: list[str]) -> list[str]:
    """Redact values attached to credential-like flags or assignments."""
    sanitized: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-"):
            flag, separator, assignment = value.partition("=")
            if _is_sensitive_key(flag.lstrip("-")):
                if separator:
                    sanitized.append(f"{flag}=<redacted>")
                else:
                    sanitized.append(value)
                    redact_next = True
                continue
        key, separator, _ = value.partition("=")
        if separator and _is_sensitive_key(key):
            sanitized.append(f"{key}=<redacted>")
        else:
            sanitized.append(value)
    return sanitized


def _is_sensitive_key(key: str) -> bool:
    """Return whether a flag or assignment key may carry a secret."""
    return bool(_SENSITIVE_KEY_RE.search(key))


def _config_overrides(extra: list[str]) -> list[str]:
    """Extract sanitized Snakemake ``--config`` assignments for disclosure."""
    overrides: list[str] = []
    is_config = False
    for value in extra:
        if value == "--config":
            is_config = True
            continue
        if value.startswith("--config="):
            overrides.extend(sanitize_argv([value.removeprefix("--config=")]))
            is_config = False
            continue
        if value.startswith("-"):
            is_config = False
            continue
        if is_config:
            overrides.extend(sanitize_argv([value]))
    return overrides


def _initialize_manifest(
    *,
    cfg: Mapping[str, Any],
    config_path: str,
    project_dir: Path,
    flags: Mapping[str, bool],
    cores: int,
    extra: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Create an in-memory initial invocation record and its unique path."""
    started_at = _utc_now()
    # Under `config/runs/`, NOT a `provenance/` root of its own (R9 follow-up,
    # ruled 2026-08-05). The migration map's Finding 1 disqualified `logs/` for
    # the config snapshot because logs are what a user deletes to reclaim space
    # and their parts are merged-then-deleted by design, while the snapshot is
    # immutable and retained. This manifest is immutable and retained for the
    # same reasons, so the same reasoning places it here. `invocations/` is a
    # SIBLING of the `<workflow>/<digest>/` bundles rather than a fourth
    # workflow entry: an invocation spans workflows.
    runs_dir = project_dir / "config" / "runs" / "invocations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    filename_stamp = started_at.replace("-", "").replace(":", "")
    filename = f"{filename_stamp}-{uuid.uuid4().hex[:12]}.json"
    source_path = Path(config_path).expanduser().resolve()
    overrides = _config_overrides(extra)
    workflows = {
        name: {
            "enabled": flags[name],
            "status": "pending" if flags[name] else "disabled",
            "command": (
                sanitize_argv(build_command(name, config_path, cores, extra))
                if flags[name]
                else None
            ),
            "exit_code": None,
        }
        for name in WORKFLOW_ORDER
    }
    manifest = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "status": "running",
        "exit_code": None,
        "source_config": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "effective_config": {
            "sha256": effective_config_digest(cfg, ADVANCED_SETTINGS),
            "scope": "source_config_plus_resolved_advanced_settings",
            "includes_cli_config_overrides": False,
        },
        "snakemake_config_overrides": overrides,
        "argv": sanitize_argv(
            ["--config", config_path, "--cores", str(cores), "--", *extra]
        ),
        "extra_args": sanitize_argv(extra),
        "cores": cores,
        "dry_run": "--dry-run" in extra or "-n" in extra,
        "no_op": not any(flags.values()),
        "workflows": workflows,
        "git": _git_metadata(),
        "environment_files": _environment_file_hashes(),
        "runtime": _runtime_versions(),
    }
    return runs_dir / filename, manifest


def _mark_pending_not_run(manifest: dict[str, Any]) -> None:
    """Mark enabled workflows skipped after an earlier failure."""
    for workflow in manifest["workflows"].values():
        if workflow["status"] == "pending":
            workflow["status"] = "not_run"


def _finalize_manifest(
    path: Path, manifest: dict[str, Any], exit_code: int | None
) -> None:
    """Atomically replace an initial manifest with its terminal record."""
    manifest["ended_at_utc"] = _utc_now()
    manifest["exit_code"] = exit_code
    _write_json_atomic(path, manifest)


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write deterministic JSON via same-directory atomic replacement."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    """Return a millisecond UTC timestamp in ISO 8601 ``Z`` form."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _git_metadata() -> dict[str, str | bool | None]:
    """Return the checkout commit and tracked/untracked dirty state."""
    commit_result = _run_metadata_command(["git", "rev-parse", "HEAD"])
    status_result = _run_metadata_command(["git", "status", "--porcelain"])
    commit = None
    if commit_result is not None:
        commit = commit_result.strip() or None
    dirty = None if status_result is None else bool(status_result.strip())
    return {"commit": commit, "dirty": dirty}


def _run_metadata_command(command: list[str]) -> str | None:
    """Run a cheap metadata query without making provenance capture fragile."""
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return getattr(result, "stdout", "")


def _environment_file_hashes() -> dict[str, str]:
    """Hash tracked environment lock files when present."""
    hashes = {}
    for name in ("pixi.lock", "Manifest.toml"):
        path = _REPO_ROOT_PATH / name
        if path.is_file():
            hashes[name] = file_sha256(path)
    return hashes


def _runtime_versions() -> dict[str, str | None]:
    """Return runtime versions available without launching another process."""
    try:
        snakemake_version = importlib.metadata.version("snakemake")
    except importlib.metadata.PackageNotFoundError:
        snakemake_version = None
    return {"python": platform.python_version(), "snakemake": snakemake_version}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the enabled CST workflows in fixed order.",
    )
    ap.add_argument(
        "--config", required=True,
        help="path to a full-orchestration snake config (config/workflows/...)",
    )
    ap.add_argument(
        "--cores", type=int, default=3,
        help="cores forwarded to every snakemake invocation (default: 3)",
    )
    ap.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="args after `--` are appended verbatim to every invocation",
    )
    args = ap.parse_args(argv)

    # argparse.REMAINDER captures the leading `--` sentinel; strip it.
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    try:
        return run(args.config, args.cores, extra)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
