"""Plan and erase the toolbox's generated namespaces before a project rebuild.

The runner owns confirmation and the project execution lock. Source configuration,
referenced local inputs, logs/history, and paths outside these namespaces survive.
No deletion follows links, and the complete plan is checked again before erasure.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from blueearth_cst.shared.config_paths import TOOLBOX_ROOT
from blueearth_cst.shared.configuration_archive import WORKFLOWS

# These namespaces contain generated products, not user inputs. Keep their
# parents: data/, models/, and config/ also accept user-owned files.
OUTPUT_ROOTS = (
    "data/spatial",
    "data/climate",
    "models/hydrology/wflow",
    "scenarios",
    "experiments",
    *(f"config/{name}" for name in WORKFLOWS),
    "config/_engine/execution-configs",
)


@dataclass(frozen=True)
class Entry:
    """One checked filesystem entry; file metadata detects changed plans."""

    path: Path
    mode: int
    device: int
    inode: int
    size: int
    modified: int


@dataclass(frozen=True)
class RebuildPlan:
    """The deletion boundary and input observations shown before confirmation."""

    root: Path
    targets: tuple[Path, ...]
    entries: tuple[Entry, ...]
    sources: tuple[tuple[str, str], ...]


def _confined(root: Path, path: Path) -> None:
    """Check all ancestors even when the final path does not yet exist."""
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"rebuild target escapes the project: {path}")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError(f"rebuild refuses linked paths: {parent}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"rebuild target resolves outside the project: {path}")


def _entry(root: Path, path: Path) -> Entry:
    """Inspect a confined entry without following symlinks or junctions."""
    _confined(root, path)
    if path.name == ".git":
        raise ValueError(f"rebuild refuses repository metadata inside outputs: {path}")
    info = path.lstat()
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"rebuild refuses reparse points: {path}")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError(f"rebuild refuses special files: {path}")
    return Entry(
        path, info.st_mode, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    )


def _input_paths(
    value: Any, source: Path, key: str = "", roots: tuple[Path, ...] = ()
) -> dict[Path, bool]:
    """Conservatively find local references, including nested catalog inputs.

    Check both declaration-relative and legacy working-directory spellings.
    This is an erasure veto, not an alternative config or HydroMT resolver.
    """
    if isinstance(value, dict):
        metadata = value.get("meta")
        if isinstance(metadata, dict):
            roots = (
                *roots,
                *(
                    path
                    for name in ("root", "roots")
                    for path in _input_paths(metadata.get(name), source, name)
                ),
            )
        found = {}
        for name, child in value.items():
            if name not in {"project_dir", "out_dir", "output_dir"}:
                for path, read_yaml in _input_paths(
                    child, source, str(name), roots
                ).items():
                    found[path] = found.get(path, False) or read_yaml
        return found
    if isinstance(value, list):
        found = {}
        for child in value:
            for path, read_yaml in _input_paths(child, source, key, roots).items():
                found[path] = found.get(path, False) or read_yaml
        return found
    if not isinstance(value, str) or not value or "\n" in value:
        return {}
    if "://" in value:
        scheme, location = value.split("://", 1)
        if scheme not in {"file", "zip", "tar"}:
            return {}
        if "::" in location:
            raise ValueError(
                "rebuild cannot establish source ownership for compound local catalog URIs"
            )
        if scheme == "file":
            parsed = urlsplit(value)
            value = unquote(parsed.path)
            if parsed.netloc:
                value = f"//{parsed.netloc}{value}"
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", value):
                value = value[1:]
        else:
            value = location
    if value.lstrip().startswith("{"):
        try:
            region = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return _input_paths(region, source, roots=roots)
    read_yaml = key in {
        "config_path",
        "catalog",
        "build_config",
        "waterbodies_config",
        "weathergen_config",
    }
    if value.startswith("toolbox:"):
        return {(TOOLBOX_ROOT / value.removeprefix("toolbox:")).resolve(): read_yaml}
    if os.name != "nt" and PureWindowsPath(value).drive:
        return {}
    # A catalog's URI may name a glob or template instead of an existing file.
    # Protect its enclosing directory too, even if no member currently exists.
    prefix = re.split(r"[\*?{]", value, maxsplit=1)[0]
    if not prefix:
        return {}
    path = Path(prefix).expanduser()
    if not (
        "/" in prefix
        or "\\" in prefix
        or path.suffix
        or read_yaml
        or key in {"uri", "root", "roots", "path", "file", "basin", "subbasin", "geom"}
    ):
        return {}
    if prefix != value and not prefix.endswith(("/", "\\")):
        path = path.parent
    bases = (
        (source.parent, Path.cwd(), TOOLBOX_ROOT)
        if not path.is_absolute()
        else (Path(),)
    )
    if key == "uri" and not path.is_absolute():
        bases = (*bases, *roots)
    return {(base / path).resolve(): read_yaml for base in bases}


def _sources(
    config_path: Path, targets: tuple[Path, ...]
) -> tuple[tuple[str, str], ...]:
    """Refuse any source or referenced local dataset overlapping erased outputs."""
    pending = [config_path.resolve()]
    observed = {}
    while pending:
        source = pending.pop()
        if source in observed:
            continue
        for target in targets:
            if source == target or source.is_relative_to(target):
                raise ValueError(
                    f"rebuild would erase a source configuration or input: {source}"
                )
        payload = source.read_bytes()
        observed[source] = hashlib.sha256(payload).hexdigest()
        try:
            document = yaml.safe_load(payload)
        except yaml.YAMLError as error:
            raise ValueError(f"rebuild cannot inspect source YAML: {source}") from error
        for path, read_yaml in _input_paths(document, source).items():
            for target in targets:
                if (
                    path == target
                    or path.is_relative_to(target)
                    or target.is_relative_to(path)
                ):
                    raise ValueError(
                        f"rebuild input overlaps generated outputs: {path} (from {source})"
                    )
            if read_yaml and path.is_file():
                pending.append(path)
    return tuple(sorted((str(path), digest) for path, digest in observed.items()))


def plan_rebuild(project_root: Path, config_path: Path) -> RebuildPlan:
    """Inventory generated outputs and refuse unsafe boundaries without writing."""
    root = project_root.absolute()
    if root != root.resolve() or root.is_symlink() or root.is_junction():
        raise ValueError("rebuild requires an unlinked, resolved project root")
    if (
        root == root.parent
        or root == Path.home().resolve()
        or TOOLBOX_ROOT.is_relative_to(root)
    ):
        raise ValueError(f"rebuild refuses this project root: {root}")
    if (root / ".git").exists():
        raise ValueError("rebuild refuses a repository root")
    document = yaml.safe_load(config_path.read_bytes())
    declared = Path(document["project"]["project_dir"]).expanduser()
    base = (
        config_path.resolve().parent
        if document.get("schema_version") == 3
        else TOOLBOX_ROOT
    )
    declared = (base / declared).absolute()
    if declared.resolve() != root:
        raise ValueError("rebuild root differs from the source configuration")
    for parent in (declared, *declared.parents):
        if parent.is_symlink() or parent.is_junction():
            raise ValueError(f"rebuild refuses a linked project root: {parent}")
    candidates = [root / name for name in OUTPUT_ROOTS]
    for path in [
        *candidates,
        root / "config/_engine/archive-transactions",
        root / "config/_engine/invocations",
        root / "logs",
    ]:
        _confined(root, path)
    config = root / "config"
    if config.exists():
        _entry(root, config)
        for path in sorted(config.iterdir()):
            if re.fullmatch(
                r"\.({})\.[0-9a-f]{{32}}\.(staged|previous)".format(
                    "|".join(WORKFLOWS)
                ),
                path.name,
            ):
                candidates.append(path)
    candidates.extend(
        root / f"config/_engine/archive-transactions/{name}.json" for name in WORKFLOWS
    )
    # Artifact owners use hashed keys rather than workflow names. Their journals
    # must be reset along with the artifacts, even after an earlier partial reset
    # has already removed the target. Keep locks and unrelated archive journals.
    transactions = root / "config/_engine/archive-transactions"
    for directory in (transactions, transactions / "completed"):
        _confined(root, directory)
        for journal in sorted(directory.glob("*.json")):
            if journal in candidates:
                continue
            _entry(root, journal)
            transaction = json.loads(journal.read_text(encoding="utf-8"))
            target = root / transaction["target"]
            _confined(root, target)
            if any(
                target.resolve().is_relative_to(root / name)
                for name in ("scenarios", "experiments")
            ):
                candidates.append(journal)
    targets = tuple(
        path
        for path in candidates
        if path.exists() or path.is_symlink() or path.is_junction()
    )
    # Veto source overlap even for a currently missing product namespace.
    sources = _sources(
        config_path, tuple(root / name for name in OUTPUT_ROOTS) + targets
    )
    entries = []
    for target in targets:
        pending = [target]
        while pending:
            path = pending.pop()
            entry = _entry(root, path)
            entries.append(entry)
            if stat.S_ISDIR(entry.mode):
                pending.extend(sorted(path.iterdir(), reverse=True))
    return RebuildPlan(root, targets, tuple(entries), sources)


def erase_outputs(plan: RebuildPlan, config_path: Path) -> None:
    """Erase exactly a confirmed plan while the caller holds project ownership.

    A changed plan is refused before the first deletion. A subsequent filesystem
    failure stops the reset; no backup or rollback is promised.
    Windows read-only generated files are made writable before deletion.
    """
    if plan_rebuild(plan.root, config_path) != plan:
        raise ValueError(
            "project outputs or source configuration changed; preview and confirm rebuild again"
        )
    for expected in reversed(plan.entries):
        current = _entry(plan.root, expected.path)
        if (current.mode, current.device, current.inode) != (
            expected.mode,
            expected.device,
            expected.inode,
        ):
            raise ValueError(f"rebuild entry changed during erasure: {expected.path}")
        if stat.S_ISDIR(expected.mode):
            expected.path.rmdir()
        else:
            if (current.size, current.modified) != (expected.size, expected.modified):
                raise ValueError(
                    f"rebuild file changed during erasure: {expected.path}"
                )
            if os.name == "nt" and not current.mode & stat.S_IWRITE:
                # Immutable generation snapshots carry Windows' read-only bit.
                expected.path.chmod(current.mode | stat.S_IWRITE)
            expected.path.unlink()
