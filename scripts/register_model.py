"""List, register, compare and prune the hydrology models of a CST project.

A project keeps each Wflow model in its own folder ``models/hydrology/<name>/``.
WF1 builds the default ``wflow``. A model prepared outside the toolbox, for
example by manual calibration, is copied into a sibling folder and registered
here; WF1 then runs its historical recipes without changing it, and WF4 selects
it with ``model.name`` in the project configuration.

Usage::

    python scripts/register_model.py list --project-dir <project_dir>
    python scripts/register_model.py register --project-dir <project_dir> \\
        --name wflow_calibrated --parent wflow --copy-from-parent --notes "Manual calibration"
    python scripts/register_model.py compare --project-dir <project_dir> \
        --run wflow/default --run wflow_calibrated/default --name calibration_effect
    python scripts/register_model.py prune --project-dir <project_dir> --name wflow_old [--delete]

``register`` records the model's files in ``<model>/_engine/model.json``. With
``--copy-from-parent`` it first copies the parent's model inputs that the new
folder lacks (never its runs), so only the edited files need to be brought back.
A registered model cannot be registered again; register an edited model under a
new name. ``compare`` compares historical runs of two or more models (each
``<model>/<recipe>``, ``default`` for a single recipe) into
``models/hydrology/comparisons/<name>/``, with the same tables and figures as
a model's own recipe comparison. ``prune`` reports by default and deletes only with ``--delete``; it
refuses a model that an experiment uses.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from blueearth_cst.shared.model_registry import (  # noqa: E402
    MODELS_DIR,
    RECORD_PATH,
    ModelSelectionError,
    build_record,
    copy_model_inputs,
    list_models,
    model_root,
    read_record,
    registration_changes,
    write_record,
)


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _experiments_using(project_dir: Path, name: str) -> list[str]:
    """Experiments whose model reference points at model ``name``."""
    target = f"{MODELS_DIR}/{name}"
    users = []
    for reference in sorted(
        (project_dir / "experiments").glob("*/_engine/model_reference.yml")
    ):
        document = yaml.safe_load(reference.read_text(encoding="utf-8")) or {}
        if document.get("model_path") == target:
            users.append(reference.parents[1].name)
    return users


def cmd_list(args: argparse.Namespace) -> int:
    entries = list_models(args.project_dir)
    if not entries:
        print(f"No models in {args.project_dir / MODELS_DIR}.")
        return 0
    for entry in entries:
        record = entry.record or {}
        state = ""
        if entry.origin == "user_managed":
            state = (
                "changed since registration"
                if registration_changes(entry.root)
                else "unchanged"
            )
        parts = [
            entry.name,
            entry.origin,
            f"parent={record.get('parent')}" if record.get("parent") else "",
            state,
        ]
        print("  ".join(part for part in parts if part))
        if record.get("notes"):
            print(f"    {record['notes']}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    root = model_root(args.project_dir, args.name)
    if read_record(root) is not None:
        print(
            f"error: {root / RECORD_PATH} exists; register an edited model under a new name.",
            file=sys.stderr,
        )
        return 2
    if args.copy_from_parent:
        if args.parent is None:
            print("error: --copy-from-parent needs --parent", file=sys.stderr)
            return 2
        parent = model_root(args.project_dir, args.parent)
        if not parent.is_dir():
            print(
                f"error: parent model folder {parent} does not exist", file=sys.stderr
            )
            return 2
        for rel in copy_model_inputs(parent, root):
            print(f"copied from {args.parent}: {rel}")
    elif (
        args.parent is not None
        and not model_root(args.project_dir, args.parent).is_dir()
    ):
        print(
            f"error: parent model folder {model_root(args.project_dir, args.parent)} "
            "does not exist",
            file=sys.stderr,
        )
        return 2
    leftovers = [
        name
        for name in ("historical", "historical_comparison")
        if (root / name).exists()
    ]
    if leftovers:
        print(
            f"error: {root} already holds toolbox runs ({', '.join(leftovers)}); "
            "remove them before registering",
            file=sys.stderr,
        )
        return 2
    try:
        record = build_record(
            root,
            name=args.name,
            origin="user_managed",
            parent=args.parent,
            notes=args.notes,
        )
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    path = write_record(root, record)
    print(f"registered {args.name} ({len(record['inventory'])} files) -> {path}")
    print(
        f"Select it with `model: {{name: {args.name}}}` in the project configuration, "
        "then run build_model for its historical runs."
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from blueearth_cst.model.historical_comparison import compare_models
    from blueearth_cst.model.plot_historical_comparison import render

    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", args.name):
        print(
            f"error: comparison name {args.name!r} must be lowercase snake_case",
            file=sys.stderr,
        )
        return 2
    settings = {
        "project_dir": str(args.project_dir),
        "runs": list(args.run),
        "registry_path": str(args.project_dir / "data/spatial/location_registry.csv"),
        "output_dir": str(args.project_dir / MODELS_DIR / "comparisons" / args.name),
        "water_year_start": args.water_year_start,
    }
    if args.observations is not None:
        settings["observations_path"] = str(args.observations.resolve())
    try:
        output = compare_models(settings)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    render(output, annotation="none")
    print(f"compared {', '.join(args.run)} -> {output}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    root = model_root(args.project_dir, args.name)
    if not root.is_dir():
        print(f"error: no model folder {root}", file=sys.stderr)
        return 2
    users = _experiments_using(args.project_dir, args.name)
    print(f"{root}: {_size(root) / 1e6:.1f} MB")
    if users:
        print(f"used by experiments: {', '.join(users)}; not deleted")
        return 1
    if not args.delete:
        print("report only; pass --delete to remove this model and its historical runs")
        return 0
    shutil.rmtree(root)
    print(f"deleted {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument(
            "--project-dir", type=Path, required=True, help="the project's project_dir"
        )
        return sub

    command("list", "list the project's models")
    register = command("register", "register a model prepared outside the toolbox")
    register.add_argument(
        "--name", required=True, help="model folder name, e.g. wflow_calibrated"
    )
    register.add_argument(
        "--parent", help="the model this one was derived from, e.g. wflow"
    )
    register.add_argument(
        "--copy-from-parent",
        action="store_true",
        help="first copy the parent's model inputs the folder lacks",
    )
    register.add_argument("--notes", default="", help="free text stored with the model")
    compare = command("compare", "compare historical runs of two or more models")
    compare.add_argument(
        "--run",
        action="append",
        required=True,
        help="<model>/<recipe>, e.g. wflow_calibrated/default; repeat for each run",
    )
    compare.add_argument("--name", required=True, help="comparison folder name")
    compare.add_argument(
        "--observations", type=Path, help="observed discharge timeseries CSV"
    )
    compare.add_argument(
        "--water-year-start", default="Jan", help="first month of the reporting year"
    )
    prune = command("prune", "report, and with --delete remove, one model")
    prune.add_argument("--name", required=True)
    prune.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)
    args.project_dir = args.project_dir.resolve()
    try:
        return {
            "list": cmd_list,
            "register": cmd_register,
            "compare": cmd_compare,
            "prune": cmd_prune,
        }[args.command](args)
    except ModelSelectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
