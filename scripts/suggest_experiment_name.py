"""Write a suggested ``experiment_name`` into a config, only if absent.

R07 B8. Run once, deliberately, before the first climate-experiment run::

    python scripts/suggest_experiment_name.py test_case/snake_config_baseline.yml

Reads ``project.project_dir``, slugifies its basename, appends today's date,
validates the result through the same grammar the workflow enforces, and writes
it to ``workflows.climate_experiment.experiment_name``.

**An existing value is never overwritten** — the command exits nonzero naming
the value already present. The experiment name is the directory every wf3
artifact hangs off, so silently changing it would strand a completed
experiment's outputs under a name nothing points at any more.

The name is NEVER generated at run time. A runtime timestamp would make every
invocation target a fresh ``experiments/<id>/``: nothing would ever be up to
date, incremental reruns would be impossible, ``--dry-run`` would mislead, and
the baseline gate would have no fixed path to check.

The config is edited as TEXT, one line, not round-tripped through
``yaml.safe_dump``. A dump discards every comment in the file: the shipped
template carries ~110 of them, and this command is the first thing a new user
runs against their copy, so dumping would delete the annotations they had just
been handed — including the ones telling them to run this. PyYAML cannot
preserve comments and a round-tripping parser is not worth a dependency here,
so the write is a targeted insertion whose result is verified by reloading it.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blueearth_cst.experiment.allocate import (  # noqa: E402
    ExperimentCollisionError,
    allocate_experiment_name,
)
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    suggest_experiment_name,
    validate_experiment_name,
)

_KEY_RE = re.compile(r"^(\s*)experiment_name\s*:(.*)$")


def _is_skippable(line: str) -> bool:
    """A blank or comment line, which carries no indentation information."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _plan_edit(text: str) -> tuple[int, int, Callable[[str], list[str]]]:
    """Plan where ``experiment_name`` goes in the raw config text.

    Returns ``(index, n_replaced, render)``: replace ``n_replaced`` lines from
    ``index`` (0 to insert) with ``render(name)``. Deferring the name to a
    callable lets the plan be computed BEFORE the name is reserved, so a config
    this cannot edit leaves no orphaned ``experiments/<id>/`` behind.

    Raises ``ValueError`` only for flow style (``climate_experiment: {…}``),
    which cannot be edited a line at a time. A *missing* block is not an error:
    the ``yaml.safe_dump`` this replaced created one via ``setdefault``, and
    ``tests/test_experiment_allocation.py`` pins that, so an absent
    ``workflows:`` or ``climate_experiment:`` is appended instead.
    """
    lines = text.splitlines(keepends=True)
    nl = "\r\n" if "\r\n" in text else "\n"

    def _find(start: int, parent_indent: int, key: str) -> tuple[int, int] | None:
        """Index and indent of ``key``'s line, or None if the block has no such key."""
        for i in range(start, len(lines)):
            line = lines[i]
            if _is_skippable(line):
                continue
            indent = _indent_of(line)
            if indent <= parent_indent:
                break  # dedented out of the block without finding the key
            head, sep, tail = line.strip().partition(":")
            if sep and head.strip() == key:
                if tail.strip() and not tail.strip().startswith("#"):
                    raise ValueError(
                        f"{key!r} is written inline (flow style); this command "
                        "edits block-style YAML one line at a time"
                    )
                return i, indent
        return None

    def _block_end(start: int, parent_indent: int) -> int:
        """One past the block's last REAL line — trailing blanks and comments
        belong to whatever follows, so appending before them keeps them there."""
        last = start
        for i in range(start, len(lines)):
            if _is_skippable(lines[i]):
                continue
            if _indent_of(lines[i]) <= parent_indent:
                break
            last = i + 1
        return last

    found_wf = _find(0, -1, "workflows")
    if found_wf is None:
        # Append the whole path at EOF. A config with no workflows: section is
        # not runnable anyway, but the dump this replaced accepted one.
        pad = [] if not lines or lines[-1].endswith(("\n", "\r")) else [nl]
        return (
            len(lines),
            0,
            lambda name: (
                pad
                + [
                    f"workflows:{nl}",
                    f"  climate_experiment:{nl}",
                    f"    experiment_name: {name}{nl}",
                ]
            ),
        )
    wf_idx, wf_indent = found_wf

    found_ce = _find(wf_idx + 1, wf_indent, "climate_experiment")
    if found_ce is None:
        ci = " " * (wf_indent + 2)
        return (
            _block_end(wf_idx + 1, wf_indent),
            0,
            lambda name: [
                f"{ci}climate_experiment:{nl}",
                f"{ci}  experiment_name: {name}{nl}",
            ],
        )
    ce_idx, ce_indent = found_ce

    # Indentation comes from the block's own first real line, so the edit
    # matches whatever the file already uses rather than assuming two spaces.
    first_child_indent = None
    insert_at = None
    # A comment run naming the key marks where the config says the key belongs
    # — the template's own block ends "inserts the key just below". Honour it,
    # so the value does not land above the paragraph explaining it.
    after_comment = None
    run_end = run_names_key = None
    end = len(lines)
    for i in range(ce_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            run_end = i
            run_names_key = run_names_key or "experiment_name" in stripped
            continue
        if run_names_key and after_comment is None:
            after_comment = run_end + 1
        run_end = run_names_key = None
        if not stripped:
            continue
        if _indent_of(line) <= ce_indent:
            end = i
            break
        if first_child_indent is None:
            first_child_indent = _indent_of(line)
            insert_at = i
        # Only a DIRECT child is the key we mean; `experiment_name:` nested
        # under stress_test: would be a different key entirely.
        if _indent_of(line) == first_child_indent:
            m = _KEY_RE.match(line)
            if m:
                indent, trailing = m.group(1), m.group(2)
                # Keep any trailing comment on the line being filled in.
                comment = (
                    "  " + trailing[trailing.index("#") :].rstrip()
                    if "#" in trailing
                    else ""
                )
                eol = line[len(line.rstrip("\r\n")) :] or nl
                return (
                    i,
                    1,
                    lambda name: [f"{indent}experiment_name: {name}{comment}{eol}"],
                )
    if run_names_key and after_comment is None:
        after_comment = run_end + 1
    if first_child_indent is None:
        # No keys yet: the block is empty or comment-only.
        at, ind = (after_comment or end), " " * (ce_indent + 2)
    else:
        at, ind = (after_comment or insert_at), " " * first_child_indent
    return at, 0, lambda name: [f"{ind}experiment_name: {name}{nl}"]


def _write_experiment_name(path: Path, name: str) -> None:
    """Set ``experiment_name`` to ``name`` by editing the text of ``path``.

    Verifies by reloading: the edited text must parse to the original config
    with exactly this one key added. A text edit that produced anything else —
    invalid YAML, a key at the wrong depth, a clobbered neighbour — raises
    instead of writing, so the config is never left worse than it was found.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    idx, n_replaced, render = _plan_edit(text)
    lines[idx : idx + n_replaced] = render(name)
    new_text = "".join(lines)

    expected = yaml.safe_load(text) or {}
    expected.setdefault("workflows", {}).setdefault("climate_experiment", {})[
        "experiment_name"
    ] = name
    if yaml.safe_load(new_text) != expected:
        raise ValueError(
            f"the edit to {path} did not reload to the expected config; "
            "nothing was written. Set "
            f"workflows.climate_experiment.experiment_name: {name} by hand"
        )
    path.write_text(new_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="path to the orchestration config YAML")
    ap.add_argument(
        "--date",
        default=None,
        metavar="YYYYMMDD",
        help="date stamp to append (default: today). Explicit values keep the "
        "command reproducible in tests and scripted setups",
    )
    ap.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="use this experiment name instead of the generated suggestion. A "
        "name you choose is NEVER silently versioned: if it is already "
        "taken the command fails, where a generated one would become "
        "_v2. Validated by the same grammar the workflow enforces",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the suggestion and leave the config untouched. Reserves "
        "nothing, so the name it prints may be taken by the time you use "
        "it",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"error: no such config: {cfg_path}", file=sys.stderr)
        return 2
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    try:
        project_dir = doc["project"]["project_dir"]
    except (KeyError, TypeError):
        print("error: config has no project.project_dir", file=sys.stderr)
        return 2

    stamp = args.date or date.today().strftime("%Y%m%d")
    try:
        if args.name is not None:
            name = validate_experiment_name(args.name, project_dir)
        else:
            name = suggest_experiment_name(project_dir, stamp)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # --dry-run reports the suggestion even when a value is already set: the
    # point of inspecting first is to SEE what would be proposed. It still
    # writes nothing.
    if args.dry_run:
        print(name)
        return 0

    existing = ((doc.get("workflows") or {}).get("climate_experiment") or {}).get(
        "experiment_name"
    )
    if existing is not None:
        print(
            f"error: experiment_name is already set to {existing!r}; refusing "
            f"to overwrite (would have suggested {name!r}). Remove the key "
            f"first if you really want a new experiment directory.",
            file=sys.stderr,
        )
        return 1

    # Confirm the config is EDITABLE before reserving anything. Reservation is
    # a side effect on disk; failing after it would leave an experiments/<id>/
    # nothing points at, for a config we then could not write to anyway.
    try:
        _plan_edit(cfg_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(
            f"error: cannot edit {cfg_path}: {exc}. Nothing was reserved or "
            "written; add workflows.climate_experiment.experiment_name by hand",
            file=sys.stderr,
        )
        return 2

    # Reserve BEFORE writing the config: an atomic mkdir claims the name, so
    # two sessions creating experiments at the same moment cannot both believe
    # they own it. A user-supplied collision is an error; a generated one is
    # versioned to _v2, _v3 (R9 P4 commit 4).
    try:
        name = allocate_experiment_name(
            project_dir, name, user_supplied=args.name is not None
        )
    except ExperimentCollisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        _write_experiment_name(cfg_path, name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote workflows.climate_experiment.experiment_name: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
