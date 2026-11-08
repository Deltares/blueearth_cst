"""Create a named external case trial from the maintained starter or another trial."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import yaml

TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLBOX_ROOT))

from blueearth_cst.shared.config_paths import (  # noqa: E402
    resolve_path,
    resolve_project_paths,
    resolve_workflow_paths,
)

_WORKFLOW_LABELS = {
    "analyze_climate": ("WF0", "Historical climate analysis"),
    "build_model": ("WF1", "Model building"),
    "analyze_projections": ("WF2", "Climate projection analysis"),
    "generate_scenarios": ("WF3", "Scenario generation"),
    "simulate_system": ("WF4", "System simulation"),
}


def _yaml_parameters(
    node: yaml.Node | None,
    path: tuple[str, ...] = (),
    seen: set[int] | None = None,
) -> Iterator[tuple[tuple[str, ...], yaml.ScalarNode, yaml.Node]]:
    """Walk physical mapping keys, including sequence items, without revisiting aliases."""
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode):
                child = (*path, key.value)
                yield child, key, value
                yield from _yaml_parameters(value, child, seen)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _yaml_parameters(item, (*path, "*"), seen)


def _template_comments(path: Path) -> dict[tuple[str, ...], str]:
    """Read parameter help from the maintained template, including optional examples."""
    text = path.read_text(encoding="utf-8")
    # Optional YAML uses #key / #  key; prose uses '# ' and stays commented.
    expanded = re.sub(
        r"^(\s*)#(?=(?:  )*(?:(?:[\w][\w -]*|'[^']+'|\"[^\"]+\"):|- ))",
        r"\1",
        text,
        flags=re.MULTILINE,
    )
    lines = expanded.splitlines()
    comments = {}
    for keys, key, _ in _yaml_parameters(yaml.compose(expanded)):
        row = key.start_mark.line
        if row:
            previous = lines[row - 1].strip()
            if previous.startswith("# ") and re.search(
                r"\b(Input|Options): ", previous
            ):
                if len(keys) >= 2 and keys[0] in {
                    "observations",
                    "indicator_set_definitions",
                    "indicator_selection",
                }:
                    keys = (keys[0], "*", *keys[2:])
                elif len(keys) >= 3 and keys[:2] == ("members", "overrides"):
                    keys = (*keys[:2], "*", *keys[3:])
                elif keys[:2] == ("variables", "custom_variable"):
                    keys = ("variables", "*", *keys[2:])
                comments[keys] = previous
    return comments


def _annotate_config(text: str, comments: dict[tuple[str, ...], str]) -> str:
    """Insert parameter help without reserializing YAML or losing existing comments."""
    lines = text.splitlines(keepends=True)
    additions: dict[int, list[str]] = {}
    for keys, key, value in _yaml_parameters(yaml.compose(text)):
        comment = comments.get(keys)
        if comment is None:
            # Custom mapping entries inherit help from a template wildcard.
            for candidate, help_text in comments.items():
                if len(candidate) == len(keys) and all(
                    expected == actual or expected == "*"
                    for expected, actual in zip(candidate, keys)
                ):
                    comment = help_text
                    break
        if comment is None:
            kind = value.tag.rsplit(":", 1)[-1]
            expected = {
                "map": "mapping",
                "seq": "list",
                "str": "text",
                "int": "integer",
                "float": "number",
                "bool": "true | false",
                "null": "null or a configured value",
            }.get(kind, "YAML value")
            comment = f"# Configure {key.value.replace('_', ' ')}. Input: {expected}."
        row = key.start_mark.line
        indent = len(lines[row]) - len(lines[row].lstrip())
        # Flow mappings share a line: label each parameter's help above that line.
        prefix = lines[row][: key.start_mark.column].strip()
        if prefix not in {"", "-"}:
            comment = f"# {key.value}: {comment[2:]}"
        existing = []
        previous = row - 1
        while previous >= 0 and lines[previous].lstrip().startswith("#"):
            existing.append(lines[previous].strip())
            previous -= 1
        if comment not in existing:
            additions.setdefault(row, []).append(" " * indent + comment + "\n")
    return "".join(
        "".join(additions.get(row, [])) + line for row, line in enumerate(lines)
    )


def _config_header(
    text: str, path: Path, project: str, created: str, owners: list[str]
) -> str:
    """Stamp a scaffold YAML without replacing its settings or explanatory comments."""
    border = "# " + "-" * 77
    # Only replace our exact header shape, never an arbitrary leading comment block.
    text = re.sub(
        rf"\A{re.escape(border)}\r?\n# [^\r\n]+\r?\n# Created: [^\r\n]+\r?\n"
        rf"# Project: [^\r\n]+\r?\n# Workflow: [^\r\n]+\r?\n"
        rf"{re.escape(border)}\r?\n\r?\n",
        "",
        text,
        count=1,
    )
    title = "Project configuration"
    workflow = "All workflows — shared settings and workflow enablement"
    if path.name != "project_config.yml":
        title = "Supporting configuration"
        workflow = "Supporting settings — no direct workflow assignment"
        if owners:
            title = "Shared workflow configuration"
            if len(owners) == 1:
                title = _WORKFLOW_LABELS.get(owners[0], ("", owners[0]))[1]
                title += " configuration"
            workflow = "; ".join(
                f"{code} — {label} ({name})" if code else name
                for name in owners
                for code, label in [_WORKFLOW_LABELS.get(name, ("", name))]
            )
    return (
        f"{border}\n# {title} — {path.name}\n# Created: {created}\n"
        f"# Project: {project}\n# Workflow: {workflow}\n{border}\n\n{text}"
    )


def _confined(path: Path, root: Path) -> Path:
    """Resolve a path and reject links leading outside the repository."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes cases root: {path}")
    return resolved


def _toolbox_version() -> str:
    """Describe the toolbox checkout from its release tags, or report unknown."""
    try:
        result = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--match",
                "v[0-9]*.[0-9]*.[0-9]*",
                "--dirty",
                "--always",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _name(value: str, label: str) -> None:
    """Reject unsafe, reserved and layout-owned case or trial names."""
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        "data",
        "outputs",
        "logs",
        "cache",
        "applications",
        "runs",
        "trials",
        "templates",
    } | {f"{prefix}{n}" for prefix in ("com", "lpt") for n in range(1, 10)}
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) or value in reserved:
        raise ValueError(
            f"{label} must be lowercase kebab-case and not a reserved filename"
        )


def _no_links(path: Path, root: Path) -> Path:
    """Reject every link component before resolving a confined path."""
    _confined(path, root)
    current = path
    while current != root:
        if current.is_symlink() or current.is_junction():
            raise ValueError(f"Path must not be a symbolic link or junction: {current}")
        current = current.parent
    return path.resolve()


def _mapping(path: Path) -> dict:
    """Read a required YAML mapping with an actionable error."""
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def _region(config: dict) -> dict:
    """Read the coordinate identity supported by the case scaffold."""
    region = config["basin"]["region"]
    if isinstance(region, str):
        region = ast.literal_eval(region)
    if not isinstance(region, dict):
        raise ValueError(
            "Case identity requires a basin.region subbasin coordinate pair"
        )
    point = region.get("subbasin")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(
            "Case identity requires a basin.region subbasin coordinate pair"
        )
    for number, low, high in zip(point, (-180, -90), (180, 90)):
        if isinstance(number, bool) or not isinstance(number, (float, int)):
            raise ValueError("Case coordinates must be finite numbers")
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError("Case coordinates are outside longitude/latitude bounds")
    return region


def _rewrite(text: str, desired: dict) -> str:
    """Change YAML scalar values without losing comments or alias safety."""
    edits = []
    seen = set()

    def visit(node: yaml.Node, value: object) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode) and isinstance(value, dict):
            for key, child in node.value:
                visit(child, value[key.value])
        elif isinstance(node, yaml.SequenceNode) and isinstance(value, list):
            for child, item in zip(node.value, value, strict=True):
                visit(child, item)
        elif isinstance(node, yaml.ScalarNode):
            current = yaml.safe_load(text[node.start_mark.index : node.end_mark.index])
            if current != value:
                edits.append(
                    (
                        node.start_mark.index,
                        node.end_mark.index,
                        json.dumps(value, ensure_ascii=False),
                    )
                )
        else:
            raise ValueError("Unsupported configuration shape during path copying")

    visit(yaml.compose(text), desired)
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    if yaml.safe_load(text) != desired:
        raise ValueError(
            "YAML aliases affect fields outside the trial paths; expand them first"
        )
    return text


def _inventory(previous: bytes | None, slug: str, trial: str, purpose: str) -> bytes:
    """Append one trial row while preserving all existing case prose and rows."""
    text = (
        previous.decode("utf-8") if previous is not None else f"# {slug}\n\n{purpose}\n"
    )
    newline = "\r\n" if "\r\n" in text else "\n"
    start, end = "<!-- cst-trials -->", "<!-- /cst-trials -->"
    purpose = " ".join(purpose.split()).replace("|", "&#124;")
    row = f"| [{trial}]({trial}/project_config.yml) | {purpose} | working |"
    if start in text or end in text:
        if (
            text.count(start) != 1
            or text.count(end) != 1
            or text.index(start) > text.index(end)
        ):
            raise ValueError(
                "README trial inventory markers are malformed; repair them first"
            )
        position = text.index(end)
        text = text[:position] + row + newline + text[position:]
    else:
        text += newline + newline.join(
            [
                "## Trials",
                "",
                start,
                "| Trial | Purpose | Status |",
                "|---|---|---|",
                row,
                end,
                "",
            ]
        )
    return text.encode("utf-8")


def _prepare(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[Path, bytes], dict[Path, bytes | None], list[str]]:
    """Build a confined, portable creation plan without writing any files."""
    _name(args.case_slug, "Case slug")
    _name(args.trial, "Trial")
    if args.from_trial:
        _name(args.from_trial, "Source trial")
    if not args.purpose.strip():
        raise ValueError("Purpose must be nonempty")
    root = Path(args.cases_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("--cases-root must name an existing absolute directory")
    root = root.resolve()
    case = _no_links(root / args.case_slug, root)
    destination = _no_links(case / args.trial, root)
    if destination.exists():
        raise ValueError(f"Trial already exists: {destination}")
    if (root / "applications" / args.case_slug).exists() or (
        root / "runs" / args.case_slug
    ).exists():
        raise ValueError(
            "Legacy split-layout case exists; copy and verify it before creating peer trials"
        )
    if case.exists() and not case.is_dir():
        raise ValueError(f"Case is not a directory: {case}")
    if (case / "project_config.yml").exists():
        raise ValueError(
            "Legacy flat case detected; migrate a copy to a named trial first"
        )
    template = TOOLBOX_ROOT / "config/templates/example-basin"
    source = case / args.from_trial if args.from_trial else template
    source_root = case if args.from_trial else TOOLBOX_ROOT.resolve()
    _no_links(source, source_root)
    config_file = _no_links(source / "project_config.yml", source_root)
    config = _mapping(config_file)
    if config.get("schema_version") != 3:
        raise ValueError(
            "Trial scaffolding requires schema_version: 3; migrate a copy of schema-2 configs first"
        )
    identity = config
    if case.exists() and not args.from_trial:
        identity_path = _no_links(case / "default/project_config.yml", case)
        if not identity_path.is_file():
            raise ValueError(
                "No default trial identity; select an existing trial with --from-trial"
            )
        identity = _mapping(identity_path)
    region = _region(identity)
    supplied = (args.longitude, args.latitude)
    if not case.exists() and None in supplied:
        raise ValueError("A new case requires --longitude and --latitude")
    if (args.longitude is None) != (args.latitude is None):
        raise ValueError("Supply both --longitude and --latitude, or omit both")
    if None not in supplied:
        candidate = {"basin": {"region": {"subbasin": list(supplied)}}}
        _region(candidate)
        if case.exists() and list(supplied) != list(region["subbasin"]):
            raise ValueError(
                "Coordinates conflict with existing case identity; choose a new case"
            )
        region = {**region, "subbasin": list(supplied)}
    source_outputs = Path(
        resolve_project_paths(config, config_file)["project"]["project_dir"]
    )
    config["project"]["project_dir"] = "outputs"
    if args.catalog is not None:
        if not args.catalog.strip():
            raise ValueError("Catalog must be nonempty")
        config["project"]["catalog"] = args.catalog
    if not args.from_trial:
        config["basin"]["region"] = str(
            {**_region(config), "subbasin": region["subbasin"]}
        )
        config["basin"]["output_locations"] = "../data/output_locations.csv"
    workflows = config.get("workflows")
    if not isinstance(workflows, dict) or not workflows:
        raise ValueError("Project requires workflow stanzas")
    files: dict[Path, bytes] = {}
    dependencies: list[str] = []
    previous: dict[Path, bytes | None] = {}
    documents = {config_file: (Path(args.trial) / "project_config.yml", config, None)}
    for name, stanza in workflows.items():
        if not isinstance(stanza, dict):
            raise ValueError(f"Invalid workflow stanza: {name}")
        if not args.from_trial and stanza.get("enabled") is not (
            name == "analyze_climate"
        ):
            raise ValueError("Starter must enable analyze_climate alone")
        raw = stanza.get("config_path")
        if raw:
            target = _no_links(source / raw, source_root)
            if not target.is_relative_to(source) or not target.is_file():
                raise ValueError(
                    f"Workflow config must be an existing file inside the source trial: {raw}"
                )
            relative = target.relative_to(source)
            documents[target] = (Path(args.trial) / relative, _mapping(target), name)
            stanza["config_path"] = relative.as_posix()

    def dependency(
        raw: str, resolved: str, owner: Path, target: Path, field: str
    ) -> str:
        if field.endswith("manifest_path"):
            raise ValueError(
                f"Retained-product dependency ({field}): {raw}. Remove the pinned product "
                "from a working source copy and generate fresh trial products."
            )
        if raw.startswith("toolbox:") and not Path(resolved).is_file():
            raise ValueError(f"Missing bundled input dependency ({field}): {raw}")
        if (
            raw.startswith("toolbox:")
            or "://" in raw
            or not Path(resolved).is_absolute()
        ):
            dependencies.append(f"External resource ({field}): {raw}")
            return raw
        path = Path(resolved)
        if not path.exists():
            raise ValueError(f"Missing input dependency ({field}): {path}")
        if path.is_relative_to(source_root):
            _no_links(path, source_root)
        else:
            if path.is_symlink() or path.is_junction():
                raise ValueError(f"Unsafe external dependency: {path}")
        if path.is_relative_to(source_outputs) or (
            path.is_relative_to(case) and "outputs" in path.relative_to(case).parts
        ):
            raise ValueError(
                f"Retained-product dependency ({field}): {path}. "
                "Remove the pinned product from a working source copy and generate fresh trial products."
            )
        if path.is_relative_to(source):
            relative = Path(args.trial) / path.relative_to(source)
            if field.endswith("catalog"):
                catalog = _mapping(path)
                metadata = catalog.get("meta") or {}
                if not isinstance(metadata, dict):
                    raise ValueError(f"Catalog meta must be a mapping: {path}")
                roots = metadata.get("roots", [])
                root = metadata.get("root")
                if root is not None:
                    roots = [root]
                if not isinstance(roots, list):
                    raise ValueError(f"Catalog meta.roots must be a list: {path}")
                local_uris = [
                    value.value
                    for _, key, value in _yaml_parameters(
                        yaml.compose(path.read_text(encoding="utf-8"))
                    )
                    if key.value in {"uri", "path"}
                    and isinstance(value, yaml.ScalarNode)
                ]
                relative_data = any(
                    isinstance(uri, str)
                    and "://" not in uri
                    and not Path(uri).is_absolute()
                    for uri in local_uris
                )
                absolute_roots = roots and all(
                    isinstance(root, str)
                    and (Path(root).is_absolute() or "://" in root)
                    for root in roots
                )
                case_locators = any(
                    isinstance(value, str)
                    and Path(value).is_absolute()
                    and Path(value).resolve().is_relative_to(case)
                    for value in [*roots, *local_uris]
                )
                if (relative_data and not absolute_roots) or case_locators:
                    raise ValueError(
                        f"Trial-local catalog has relative dataset dependencies or case-bound absolute paths: {path}. "
                        "Place the catalog and its datasets together in shared case data/ "
                        "and update the source reference before copying."
                    )
                dependencies.append(
                    f"Catalog datasets remain external; verify catalog roots and URIs: {relative.as_posix()}"
                )
            if any(
                part in {"outputs", "logs", ".snakemake", ".cache", "__pycache__"}
                for part in relative.parts[1:]
            ):
                raise ValueError(f"Generated dependency cannot be copied: {path}")
            if not path.is_file():
                raise ValueError(f"Supporting dependency must be a file: {path}")
            files[relative] = path.read_bytes()
            dependencies.append(f"Copy input ({field}): {relative.as_posix()}")
            return Path(os.path.relpath(case / relative, target.parent)).as_posix()
        if path.is_relative_to(case / "data"):
            dependencies.append(
                f"Shared input ({field}): {path.relative_to(case).as_posix()}"
            )
            return Path(os.path.relpath(path, target.parent)).as_posix()
        if path.is_relative_to(case):
            raise ValueError(
                f"Input belongs to another trial: {path}; move shared authored input to case data/"
            )
        dependencies.append(f"External input ({field}): {path.as_posix()}")
        return path.as_posix()

    def paths(
        raw: object, resolved: object, owner: Path, target: Path, key: str = ""
    ) -> object:
        if isinstance(raw, dict):
            return {
                name: paths(
                    value, resolved[name], owner, target, f"{key}.{name}".lstrip(".")
                )
                for name, value in raw.items()
            }
        if isinstance(raw, list):
            return [
                paths(value, resolved[i], owner, target, key)
                for i, value in enumerate(raw)
            ]
        if (
            isinstance(raw, str)
            and isinstance(resolved, str)
            and (
                raw != resolved
                or raw.startswith("toolbox:")
                or "://" in raw
                or Path(raw).is_absolute()
                or key.endswith("catalog")
            )
        ):
            return dependency(raw, resolved, owner, target, key)
        return raw

    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    for owner, (relative, document, workflow) in documents.items():
        text = owner.read_text(encoding="utf-8")
        if workflow is None:
            # The output root and newly shared outlet are destinations, not copied inputs.
            resolved = resolve_project_paths(document, owner)
            resolved["project"]["project_dir"] = "outputs"
            if args.catalog is not None:
                resolved["project"]["catalog"] = resolve_path(
                    args.catalog, destination / "project_config.yml", catalog=True
                )
            if not args.from_trial:
                resolved["basin"]["output_locations"] = document["basin"][
                    "output_locations"
                ]
        else:
            resolved = resolve_workflow_paths(document, owner, workflow, 3)
        desired = paths(document, resolved, owner, case / relative)
        rewritten = _rewrite(text, desired)
        if not args.from_trial:
            comments = _template_comments(owner)
            rewritten = _annotate_config(rewritten, comments)
        files[relative] = _config_header(
            rewritten, relative, args.case_slug, created, [workflow] if workflow else []
        ).encode("utf-8")
    outlet = case / "data/output_locations.csv"
    if not args.from_trial:
        _no_links(outlet, case)
        if case.exists():
            if not outlet.is_file():
                raise ValueError(
                    "Shared data/output_locations.csv is missing; use --from-trial or restore case data"
                )
            dependencies.append("Shared input: data/output_locations.csv")
        else:
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["station_name", "x", "y"])
            writer.writerow(["outlet", *region["subbasin"]])
            files[Path("data/output_locations.csv")] = stream.getvalue().encode("utf-8")
    version = _toolbox_version()
    origin = args.from_trial or "maintained starter"
    files[Path(args.trial) / "notes.md"] = (
        f"# {args.case_slug} / {args.trial}\n\nPurpose: {args.purpose.strip()}\n\n"
        f"Created: {created}\nSource: {origin}\nToolbox revision: {version}\n\n"
        "No execution history is inherited.\n\n"
        "## Run notes\n\n- Configuration revision:\n- Command:\n- Checks and outcome:\n"
        "- Limitations:\n- Evidence (relative paths):\n"
    ).encode("utf-8")
    files[Path(args.trial) / "INSTRUCTIONS.md"] = (
        f"# {args.case_slug} / {args.trial}\n\n"
        "Review the copied settings and external data access before running.\n"
        "The maintained starter enables historical climate analysis alone; copied trials preserve enablement.\n\n"
        "From the toolbox root, replace <cases-root> with your case location:\n\n"
        "```powershell\n"
        f'pixi run python scripts/run_workflows.py --config "<cases-root>/{args.case_slug}/{args.trial}/project_config.yml" --cores 3 -- --dry-run\n'
        "```\n\nRemove the final `-- --dry-run` to run enabled workflows.\n"
        "See `docs/guide/running.md` and `docs/guide/new-case.md` in the matching toolbox checkout.\n\n"
        "Paths resolve beside their declaring YAML; `toolbox:` selects bundled resources.\n"
        "This trial writes to `outputs/`. Input datasets remain external unless explicitly copied.\n\n"
        "Keep working, retained or complete status in the case README. Complete means the declared scope\n"
        "was reviewed, not that every workflow ran. Keep retained evidence unchanged for new experiments.\n\n"
        "Git ignores are not backups. An evidence archive preserves reviewed outputs and provenance;\n"
        "a rerunnable archive also needs input datasets, catalogs and the software environment.\n"
        "Moving YAML does not relocate runtime locators in historical outputs.\n\n"
        "## Input dependencies\n\n"
        + "\n".join(f"- {item}" for item in sorted(set(dependencies)))
        + "\n"
    ).encode("utf-8")
    for name in ("README.md", ".gitignore"):
        path = _no_links(case / name, root)
        previous[Path(name)] = path.read_bytes() if path.exists() else None
    files[Path("README.md")] = _inventory(
        previous[Path("README.md")], args.case_slug, args.trial, args.purpose.strip()
    )
    ignore = previous[Path(".gitignore")] or b""
    rule = f"/{args.trial}/outputs/"
    if rule not in ignore.decode("utf-8").splitlines():
        ignore += (b"\n" if ignore and not ignore.endswith(b"\n") else b"") + (
            rule + "\n"
        ).encode()
    files[Path(".gitignore")] = ignore
    return case, destination, files, previous, sorted(set(dependencies))


def _install(
    case: Path,
    destination: Path,
    files: dict[Path, bytes],
    previous: dict[Path, bytes | None],
) -> None:
    """Stage everything and roll back this creation if installation fails."""
    existing = case.exists()
    with tempfile.TemporaryDirectory(
        prefix=".create-case-", dir=case.parent
    ) as scratch:
        stage = Path(scratch) / "stage"
        for relative, content in files.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if not existing:
            case.mkdir()
            try:
                for child in stage.iterdir():
                    shutil.move(str(child), case / child.name)
            except OSError:
                shutil.rmtree(case)
                raise
            return
        lock = case / ".create-trial.lock"
        with lock.open("x", encoding="utf-8"):
            pass
        installed = False
        changed = []
        try:
            for relative, content in previous.items():
                path = case / relative
                actual = path.read_bytes() if path.exists() else None
                if actual != content:
                    raise ValueError(
                        f"Case record changed during creation; retry: {relative}"
                    )
            destination.mkdir()
            installed = True
            for child in (stage / destination.name).iterdir():
                shutil.move(str(child), destination / child.name)
            for relative in previous:
                backup = Path(scratch) / (relative.name + ".backup")
                if previous[relative] is not None:
                    backup.write_bytes(previous[relative])
                os.replace(stage / relative, case / relative)
                changed.append(relative)
        except (OSError, ValueError):
            for relative in reversed(changed):
                if previous[relative] is None:
                    (case / relative).unlink()
                else:
                    os.replace(
                        Path(scratch) / (relative.name + ".backup"), case / relative
                    )
            if installed:
                shutil.rmtree(destination)
            raise
        finally:
            lock.unlink()


def main(argv: list[str] | None = None) -> int:
    """Create a named case trial; return 2 for actionable input or I/O failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_slug")
    parser.add_argument("--cases-root", required=True)
    parser.add_argument(
        "--trial", default="default", help="New trial name (default: default)"
    )
    parser.add_argument(
        "--from-trial", help="Copy authored settings from a trial in this case"
    )
    for name in ("longitude", "latitude"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--purpose", required=True)
    parser.add_argument(
        "--catalog",
        help="Catalog path beside the destination YAML; toolbox: selects bundled resources",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files and dependencies without writing",
    )
    args = parser.parse_args(argv)
    try:
        case, destination, files, previous, dependencies = _prepare(args)
        if not args.dry_run:
            _install(case, destination, files, previous)
        print(f"{'Would create' if args.dry_run else 'Created'} {destination}")
        print(f"Source: {args.from_trial or 'maintained starter'}")
        print(f"Outputs: {destination / 'outputs'}")
        for relative in sorted(files):
            print(
                f"  {'Update' if relative in previous and previous[relative] is not None else 'Create'}: {relative.as_posix()}"
            )
        for item in dependencies:
            print(f"  {item}")
        print(
            "Next: review settings and data access, then preview from the toolbox root:"
        )
        print(
            f'pixi run python scripts/run_workflows.py --config "{destination.as_posix()}/project_config.yml" --cores 3 -- --dry-run'
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        yaml.YAMLError,
    ) as error:
        print(f"Cannot create trial: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
