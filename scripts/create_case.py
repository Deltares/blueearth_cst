"""Create an external case from the toolbox's maintained example-basin template."""

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
    if path != Path("project_config.yml"):
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


def _replace(text: str, keys: tuple[str, ...], value: object) -> str:
    """Replace one YAML scalar using parser offsets, preserving comments."""
    node = yaml.compose(text)
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Expected YAML mapping at {'.'.join(keys)}")
        matches = [child for name, child in node.value if name.value == key]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one YAML key: {'.'.join(keys)}")
        node = matches[0]
    if not isinstance(node, yaml.ScalarNode):
        raise ValueError(f"Expected scalar at {'.'.join(keys)}")
    replacement = json.dumps(value, ensure_ascii=False)
    return text[: node.start_mark.index] + replacement + text[node.end_mark.index :]


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
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _prepare(args: argparse.Namespace) -> tuple[Path, dict[Path, bytes]]:
    """Validate inputs and build all new file contents without filesystem writes."""
    slug = args.case_slug
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{n}" for prefix in ("com", "lpt") for n in range(1, 10)
    }
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in reserved:
        raise ValueError(
            "Case slug must be lowercase kebab-case and not a reserved filename"
        )
    for name, low, high in (("longitude", -180, 180), ("latitude", -90, 90)):
        number = getattr(args, name)
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} must be finite and between {low} and {high}")
    if not args.purpose.strip():
        raise ValueError("Purpose must be nonempty")
    root = Path(args.cases_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("--cases-root must name an existing absolute directory")
    root = root.resolve()
    for relative in ("applications", "runs", f"runs/{slug}"):
        candidate = root / relative
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError(
                f"Destination must not be a symbolic link or junction: {candidate}"
            )
    template = _confined(TOOLBOX_ROOT / "config/templates/example-basin", TOOLBOX_ROOT)
    applications = _confined(root / "applications", root)
    if not applications.is_dir():
        raise ValueError("Cases root requires applications/")
    if not template.is_dir():
        raise ValueError("Toolbox requires config/templates/example-basin/")
    destination = applications / slug
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Application already exists: {destination}")
    _confined(destination, root)
    outputs = _confined(root / "runs" / slug, root)
    files: dict[Path, bytes] = {}
    for source in template.rglob("*"):
        if source.is_symlink() or source.is_junction():
            raise ValueError(
                f"Template must not contain symbolic links or junctions: {source}"
            )
        if source.is_file():
            files[source.relative_to(template)] = source.read_bytes()
    config_path = Path("project_config.yml")
    text = files[config_path].decode("utf-8")
    config = yaml.safe_load(text)
    if not isinstance(config, dict) or config.get("schema_version") not in (2, 3):
        raise ValueError("Template requires project config schema_version: 2 or 3")
    workflows = config.get("workflows")
    if not isinstance(workflows, dict) or not workflows:
        raise ValueError("Template requires workflow stanzas")
    config_owners: dict[Path, list[str]] = {}
    for name, stanza in workflows.items():
        if not isinstance(stanza, dict) or stanza.get("enabled") is not (
            name == "analyze_climate"
        ):
            raise ValueError("Template must enable analyze_climate alone")
        if stanza.get("config_path"):
            if not isinstance(stanza["config_path"], str):
                raise ValueError(f"Workflow config_path must be a string: {name}")
            target = Path(stanza["config_path"])
            if target.is_absolute() or ".." in target.parts or target not in files:
                raise ValueError(f"Workflow config must be inside template: {target}")
            config_owners.setdefault(target, []).append(name)
    portable = config.get("schema_version") == 3
    output_locator = f"../../runs/{slug}" if portable else outputs.as_posix()
    locations_locator = (
        "data/output_locations.csv"
        if portable
        else (destination / "data/output_locations.csv").as_posix()
    )
    text = _replace(text, ("project", "project_dir"), output_locator)
    text = _replace(
        text,
        ("basin", "output_locations"),
        locations_locator,
    )
    region = config["basin"]["region"]
    if isinstance(region, str):
        try:
            region = ast.literal_eval(region)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                "Template basin.region must define a subbasin mapping"
            ) from error
    if not isinstance(region, dict) or "subbasin" not in region:
        raise ValueError("Template basin.region must define a subbasin mapping")
    region = {**region, "subbasin": [args.longitude, args.latitude]}
    text = _replace(text, ("basin", "region"), str(region))
    if args.catalog is not None:
        if not args.catalog.strip():
            raise ValueError("Catalog must be nonempty")
        text = _replace(text, ("project", "catalog"), args.catalog)
    expected = yaml.safe_load(files[config_path])
    expected["project"]["project_dir"] = output_locator
    expected["basin"]["output_locations"] = locations_locator
    expected["basin"]["region"] = str(region)
    if args.catalog is not None:
        expected["project"]["catalog"] = args.catalog
    if yaml.safe_load(text) != expected:
        raise ValueError("Template aliases affect fields outside the case identity")
    files[config_path] = text.encode("utf-8")
    templates = Path(__file__).resolve().parents[1] / "config/templates"
    created = datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")
    created = f"{created[:-6]} {created[-6:]}"
    for relative, content in list(files.items()):
        if relative.suffix not in {".yml", ".yaml"}:
            continue
        yaml_text = content.decode("utf-8")
        parsed = yaml.safe_load(yaml_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Template YAML must be a mapping: {relative}")
        observations = parsed.get("observations", {})
        if not isinstance(observations, dict):
            raise ValueError(f"observations must be a mapping: {relative}")
        for variable, raw in observations.items():
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise ValueError(f"Observation path must be a string: {variable}")
            source = Path(raw)
            source_base = (template / relative).parent if portable else template
            source = source if source.is_absolute() else source_base / source
            source = source.resolve()
            if not source.is_relative_to(template) or not source.is_file():
                raise ValueError(
                    f"Observation must be an existing template sidecar: {raw}"
                )
            sidecar = source.relative_to(template)
            yaml_text = _replace(
                yaml_text,
                ("observations", variable),
                Path(
                    os.path.relpath(
                        destination / sidecar, (destination / relative).parent
                    )
                ).as_posix()
                if portable
                else (destination / sidecar).as_posix(),
            )
        comment_sources = (
            [templates / "example-basin/project_config.yml"]
            if relative == config_path
            else [
                templates / "example-basin" / f"project_config_{owner}.yml"
                for owner in config_owners.get(relative, [])
                if owner in _WORKFLOW_LABELS
            ]
        )
        if "selection" in parsed and "id" in parsed:
            comment_sources.append(templates / "indicator_set.template.yml")
        comments = {}
        for source in comment_sources:
            comments.update(_template_comments(source))
        yaml_text = _annotate_config(yaml_text, comments)
        files[relative] = _config_header(
            yaml_text, relative, slug, created, config_owners.get(relative, [])
        ).encode("utf-8")
    outlet = Path("data/output_locations.csv")
    rows = list(csv.reader(io.StringIO(files[outlet].decode("utf-8"))))
    if len(rows) != 2 or rows[0] != ["station_name", "x", "y"] or len(rows[1]) != 3:
        raise ValueError(
            "Template outlet CSV must have station_name,x,y and one outlet row"
        )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(rows[0])
    writer.writerow(["outlet", args.longitude, args.latitude])
    files[outlet] = stream.getvalue().encode("utf-8")
    files.pop(Path("README.md"), None)
    toolbox_version = _toolbox_version()
    documentation = (TOOLBOX_ROOT / "docs/guide").as_posix()
    files[Path("INSTRUCTIONS.md")] = f"""# {slug}: quick reference

Run in PowerShell from the toolbox root. Use one running session per output directory.

Toolbox version at creation: `{toolbox_version}`.
Use the documentation from that checkout, available locally without a hosted manual:
[create a case](<{documentation}/new-case.md>),
[configure it](<{documentation}/configuration.md>),
[run or resume](<{documentation}/running.md>),
[choose indicators](<{documentation}/indicators.md>), and
[read results](<{documentation}/outputs.md>).
If the checkout moves, update these local links and use the recorded version to identify matching guidance.
Input datasets are external; software installation does not supply them.

## Run

The starter enables WF0 alone. Review its historical climate comparison and choose forcing before
disabling WF0 and enabling downstream stages. WF1 and WF3 independently supply WF4; WF2 is optional
projection context and never drives scenario generation. Review all copied settings for this basin.

```powershell
$config = "{destination.as_posix()}/project_config.yml"
$projectDir = "{outputs.as_posix()}"

# Preview; downstream stages need upstream products.
pixi run python scripts/run_workflows.py --config $config --cores 3 -- --dry-run

# Run enabled workflows.
pixi run python scripts/run_workflows.py --config $config --cores 3

# Allocate a new experiment after changing model or scenario settings.
pixi run python scripts/run_workflows.py --config $config --cores 3 --new-experiment
```

## Resume or unlock

```powershell
# Rerun jobs left incomplete by an interrupted run.
pixi run python scripts/run_workflows.py --config $config --cores 3 -- --rerun-incomplete

# Clear stale locks only after confirming no workflow is still running.
pixi run python scripts/run_workflows.py --config $config --cores 3 -- --unlock
```

## Rebuild for testing

This permanently erases generated outputs, including all experiments, without making a backup.
Source inputs and logs are preserved. Enable WF1 and WF3 when rebuilding WF4; indicators-only cannot rebuild.

```powershell
# Preview what will be erased; no files change and no workflows start.
pixi run python scripts/run_workflows.py --config $config --rebuild -- --dry-run

# Confirm at Continue? [y/N]:, then rerun enabled workflows.
pixi run python scripts/run_workflows.py --config $config --rebuild
```

For automation, add `--yes` before the `--` separator. See `docs/guide/running.md` in the toolbox
for the exact reset scope and input protection rules.

## Individual workflows

Enable the selected workflow in `project_config.yml`. WF4 needs WF1 and WF3 products.

```powershell
# WF0: preview historical climate analysis.
pixi run python scripts/run_workflow.py analyze_climate --config $config --project-dir $projectDir --cores 3 --dry-run

# WF1: build and run the historical model.
pixi run python scripts/run_workflow.py build_model --config $config --project-dir $projectDir --cores 3

# WF2: projection plausibility overlay.
pixi run python scripts/run_workflow.py analyze_projections --config $config --project-dir $projectDir --cores 3

# WF3: generate scenarios (enable only WF3 for a confined run).
pixi run python scripts/run_workflows.py --config $config --cores 3

# WF4: simulations and indicators.
pixi run python scripts/simulate_system.py --config $config --cores 3
```

## Recalculate indicators

In `project_config_simulate_system.yml`, set:

```yaml
operations: [indicators]
```

Keep the existing `experiment_name` and choose `indicator_sets` in that file.
The selected experiment must have a completed `_engine/simulation.json` record, even for a dry run.
Then run against compatible retained responses:

```powershell
pixi run python scripts/simulate_system.py --config $config --cores 3
```

Restore `operations: [simulations, indicators]` for full runs.

## Results

Paths within the configured output root:

| Output | Path |
|---|---|
| Historical climate | `data/climate/historical/` |
| Historical model | `models/hydrology/wflow/` |
| Projections | `data/climate/projections/` |
| Scenarios | `scenarios/` |
| Responses | `experiments/<name>/hydrology/wflow/output/` |
| Indicators | `experiments/<name>/results/indicator_sets/<set-id>/` |
| Logs | `logs/` |

## Configuration

Edit `project_config.yml` for shared settings and workflow enablement;
edit sibling `project_config_<workflow>.yml` files for workflow settings.
Review the catalog, climate forcing, and upstream-area threshold before running.

| Setting | Value |
|---|---|
| Project | {slug} |
| Created | {created} |
| Schema | {config["schema_version"]} |
| Coordinates (longitude, latitude) | {args.longitude}, {args.latitude} |
| Output root | `{output_locator}` |

Schema 3 resolves case paths beside their YAML; `toolbox:` selects bundled resources.
Schema 2 uses the run directory for relative paths. Retained run records: [notes.md](notes.md).
""".encode("utf-8")
    files[Path("notes.md")] = (
        f"# {slug} — run notes\n\nStatus: configured; not run.\n\n"
        f"Purpose: {args.purpose.strip()}\n\n"
        f"CST version at creation: {toolbox_version}\n\n"
        "No run history or validation is inherited from the template.\n\n"
        "## Retained run record\n\n"
        "- Date:\n- Cases revision and dirty state:\n"
        "- Toolbox revision and dirty state:\n- Configuration choices:\n"
        "- Command:\n- Outcome:\n- Checks actually performed:\n"
        "- Evidence location:\n- Follow-up:\n\n"
        "Register the case in `cst-applications.md` only after every workflow "
        "completes.\n"
    ).encode("utf-8")
    return destination, files


def main(argv: list[str] | None = None) -> int:
    """Validate and create a case; return 2 for actionable input or I/O failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_slug")
    parser.add_argument("--cases-root", required=True)
    for name in ("longitude", "latitude"):
        parser.add_argument(f"--{name}", required=True, type=float)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--catalog", help="Toolbox-relative or absolute data catalog")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and preview without writing"
    )
    args = parser.parse_args(argv)
    try:
        destination, files = _prepare(args)
        if not args.dry_run:
            with tempfile.TemporaryDirectory(
                prefix=".create-case-", dir=destination.parent
            ) as scratch:
                stage = Path(scratch) / "case"
                stage.mkdir()
                for relative, content in files.items():
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                # mkdir claims the destination without overwriting an existing case.
                destination.mkdir()
                try:
                    for child in stage.iterdir():
                        shutil.move(str(child), destination / child.name)
                except OSError:
                    shutil.rmtree(destination)
                    raise
        verb = "Would create" if args.dry_run else "Created"
        print(f"{verb} {destination}")
        print(f"Outputs: {Path(args.cases_root).resolve() / 'runs' / args.case_slug}")
        print("Next: review settings and preview WF0 from the toolbox root:")
        print(
            f'pixi run python scripts/run_workflows.py --config "{destination.as_posix()}/project_config.yml" --cores 3 -- --dry-run'
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"Cannot create case: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
