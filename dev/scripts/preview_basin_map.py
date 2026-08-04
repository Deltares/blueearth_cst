# -*- coding: utf-8 -*-
"""Render the basin-area figure with any tunable overridden, without a WF1 run.

``blueearth_cst/shared/plot_map.py`` opens with a TUNABLE CONSTANTS block —
every size, weight, colour and position the figure uses. This script drives that
block from the command line against a model that already exists on disk, so a
value can be tried and LOOKED AT in seconds instead of edit-rerun-WF1.

    # what can I change, and what is it set to now?
    pixi run python dev/scripts/preview_basin_map.py --list

    # render once with two values changed, and open it
    pixi run python dev/scripts/preview_basin_map.py \
        --set FONT_SIZE_BASE=9 --set _PANEL_LEFT=1.10 --open

    # render one figure PER value, named after it, for side-by-side comparison
    pixi run python dev/scripts/preview_basin_map.py \
        --sweep _LAYOUT_RIGHT=0.70,0.78,0.86

    # two swept knobs render their full cross-product
    pixi run python dev/scripts/preview_basin_map.py \
        --sweep RIVER_WIDTH_MAX=0.8,1.2 --sweep MARKER_SIZE=12,18,28

Overrides are applied to the module's globals and restored afterwards, so
nothing is written back to the source. Renders land in ``--out-dir`` (a
gitignored scratch tree by default) and NEVER in a project's ``plots/``: a
hand-tuned preview must not be able to take the place of a run product that the
baseline fingerprints.

**When adding a tunable to plot_map.py**, derive anything assembled from it in a
function (see that module's DERIVED VALUES section). A module-level constant
built from other constants snapshots them at import, so this script would set
the input and silently change nothing.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import io
import itertools
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

# The tunable comments contain em dashes; a Windows console defaults to cp1252
# and would either mojibake them or raise mid-listing.
with contextlib.suppress(AttributeError, OSError):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared import plot_map  # noqa: E402

#: Names in ``plot_map`` that count as tunables: SHOUTING_CASE, optionally
#: private. Mixed-case imports (``LatitudeFormatter``, ``Line2D``) do not match,
#: and callables are filtered out separately.
_TUNABLE_PATTERN = re.compile(r"^_?[A-Z][A-Z0-9_]*$")

#: Project directories tried, in order, when neither ``--project-dir`` nor
#: ``$BASIN_MAP_PROJECT_DIR`` is given.
#:
#: ``basin_map_fixture`` is a five-subcatchment model with a gauge layer, kept
#: for exactly this purpose (its README says so, and says not to delete it): it
#: exercises the subcatchment divides, the dissolved outline, the gauge labels
#: and a legend long enough to size the side panel against. ``test_local`` is
#: the fallback and has one outlet and no gauges — the figure renders, but half
#: the tunables have nothing to act on, which is why it is not first.
#:
#: Both are resolved against the PRIMARY checkout as well as the current one:
#: ``test_case/`` exists only there, so a worktree finds neither otherwise.
_CANDIDATE_PROJECT_DIRS = (
    "test_case/basin_map_fixture",
    "test_case/test_local",
)

_FIGURE_STEM = "basin_area"


# --- the tunable block, read out of the source --------------------------------


def _tunables() -> dict:
    """Every overridable constant in ``plot_map``, name to current value."""
    return {
        name: value
        for name, value in vars(plot_map).items()
        if _TUNABLE_PATTERN.match(name) and not callable(value)
    }


def _assigned_names(node: ast.Assign) -> list:
    """The plain names an assignment binds, unpacking ``A, B = 1, 2``."""
    names = []
    for target in node.targets:
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        names.extend(part.id for part in elements if isinstance(part, ast.Name))
    return names


def _comments_by_line(source: str) -> dict:
    """Line number to comment text, tokenized.

    Tokenizing rather than splitting on ``#`` is what keeps ``COLOR_RIVER =
    "#2c6fad"`` from reading its own value as a comment.
    """
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return {
        token.start[0]: token.string.lstrip("#").lstrip(": ").strip()
        for token in tokens
        if token.type == tokenize.COMMENT
    }


def _tunable_docs() -> dict:
    """The ``#:`` comment on each tunable, as one flat line.

    Read from the source rather than duplicated here, so ``--list`` cannot drift
    from what the block actually says. Both spellings are picked up: the comment
    block ABOVE an assignment, and the trailing comment on its own line.
    """
    source = Path(plot_map.__file__).read_text(encoding="utf-8")
    comments = _comments_by_line(source)
    docs = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        # A trailing comment, else the comment block immediately above.
        parts = []
        if node.lineno in comments:
            parts = [comments[node.lineno]]
        else:
            # Standalone comment lines only — a trailing comment on the line
            # above belongs to the PREVIOUS assignment, not to this one.
            lines = source.splitlines()
            line = node.lineno - 1
            while line in comments and lines[line - 1].lstrip().startswith("#"):
                parts.insert(0, comments[line])
                line -= 1
        text = " ".join(part for part in parts if part)
        for name in _assigned_names(node):
            docs[name] = text
    return docs


def _coerce(name: str, text: str):
    """Parse a CLI value as a Python literal, guided by what the tunable holds.

    ``literal_eval`` covers every shape the block uses — floats, ints, ``None``,
    tuples such as ``(0, (4, 2))``, and colour tuples — while a bare ``#2c6fad``
    or ``upper right`` stays a string.

    The one place a bare ``literal_eval`` gets it wrong is a GREYSCALE COLOUR:
    matplotlib spells those as the string ``"0.45"``, and handing it the float
    ``0.45`` raises deep inside a collection's ``set_edgecolor``. So a numeric
    literal is kept as text whenever the tunable currently holds a string.
    ``None`` is not caught by that rule, because ``_LEGEND_TITLE=None`` (drop the
    title row) is a real thing to want.
    """
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if numeric and isinstance(getattr(plot_map, name), str):
        return text
    return value


def _split_values(name: str, spec: str) -> list:
    """The comma-separated values of a ``--sweep``, literal-aware.

    Splits on the parse tree rather than on commas, which is what keeps a tuple
    value intact: a naive ``spec.split(",")`` shreds ``(0, (4, 2)),(0, (1, 1))``
    into five fragments. Each element is handed back to ``_coerce`` as SOURCE
    TEXT so it gets the same string-vs-number treatment a ``--set`` would.
    """
    listed = f"[{spec}]"
    try:
        elements = ast.parse(listed, mode="eval").body.elts
    except (SyntaxError, AttributeError):
        fragments = [part.strip() for part in spec.split(",")]
    else:
        fragments = [ast.get_source_segment(listed, node) for node in elements]
    return [_coerce(name, fragment) for fragment in fragments]


def _parse_assignment(argument: str, flag: str) -> tuple:
    if "=" not in argument:
        raise SystemExit(f"{flag} expects NAME=VALUE, got {argument!r}")
    name, _, value = argument.partition("=")
    name = name.strip()
    known = _tunables()
    if name not in known:
        suggestion = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(suggestion)}?" if suggestion else ""
        raise SystemExit(
            f"{name!r} is not a tunable in plot_map.py.{hint} "
            f"Run --list to see all {len(known)} of them."
        )
    return name, value.strip()


# --- inputs -------------------------------------------------------------------


def _primary_checkout() -> Path:
    """The main working tree, which is where ``test_case/`` and ``.tmp/`` live.

    In a worktree the repo root holds neither, so resolving candidates against
    the cwd alone leaves the default dead exactly where the figure work happens.
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return REPO_ROOT
    return (REPO_ROOT / common).resolve().parent


def _warn_if_transient(path: Path) -> None:
    """Say so when the model being rendered lives in a tree that gets swept.

    Applied to the resolved path WHEREVER it came from, not just to the built-in
    candidates: pointing ``$BASIN_MAP_PROJECT_DIR`` at a scratch run is the easy
    thing to do, and when that tree is cleaned the resolution quietly falls
    through to a one-outlet basin. Rendering a weaker figure without saying so is
    how a tunable gets written off as having no effect.
    """
    if ".tmp" in path.parts:
        print(
            f"note: {path} is a transient scratch tree and may be swept. "
            "Point at test_case/basin_map_fixture, or another project you keep.",
            file=sys.stderr,
        )


def _resolve_project_dir(requested: str | None) -> Path:
    """``--project-dir``, then the env var, then the first candidate present."""
    explicit = requested or os.environ.get("BASIN_MAP_PROJECT_DIR")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "models" / "hydrology" / "wflow" / "staticmaps.nc").is_file():
            raise SystemExit(f"{path} does not hold a models/hydrology/wflow/staticmaps.nc")
        _warn_if_transient(path)
        return path
    roots = dict.fromkeys((_primary_checkout(), REPO_ROOT))  # ordered, deduped
    for root, candidate in itertools.product(roots, _CANDIDATE_PROJECT_DIRS):
        path = root / candidate
        if (path / "models" / "hydrology" / "wflow" / "staticmaps.nc").is_file():
            _warn_if_transient(path)
            return path.resolve()
    raise SystemExit(
        "No project directory found. Pass --project-dir <dir> (the folder "
        "holding models/hydrology/wflow/), or set $BASIN_MAP_PROJECT_DIR. Tried: "
        + ", ".join(
            str(root / c) for root, c in itertools.product(roots, _CANDIDATE_PROJECT_DIRS)
        )
    )


def _gauges_fn(project_dir: Path) -> str | None:
    """A value for ``output_locations`` that resolves to this model's gauges.

    The figure takes the config's filename and lets ``gauges_layer_name`` map it
    onto a staticgeoms layer; there is no config here, so the filename is
    reconstructed from the layer instead. hydromt_wflow's naming rule replaces
    underscores with hyphens, so inverting it round-trips exactly.
    """
    geoms_dir = project_dir / "models" / "hydrology" / "wflow" / "staticgeoms"
    layers = sorted(geoms_dir.glob("gauges_*.geojson"))
    if not layers:
        return None
    return layers[0].stem[len("gauges_") :].replace("-", "_")


# --- rendering ----------------------------------------------------------------


@contextlib.contextmanager
def _overridden(values: dict):
    """Set tunables for the duration of one render, then put them back."""
    previous = {name: getattr(plot_map, name) for name in values}
    for name, value in values.items():
        setattr(plot_map, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(plot_map, name, value)


def _slug(values: dict) -> str:
    """Filename suffix naming the overrides, so renders are self-identifying."""
    parts = [f"{name}={value!r}" for name, value in values.items()]
    return re.sub(r"[^A-Za-z0-9=.,_+-]", "", "__".join(parts).replace(" ", ""))


def render(project_dir: Path, out_dir: Path, values: dict, suffix: str) -> list:
    """Render one figure into ``out_dir``, returning the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with _overridden(values):
        plot_map.plot_basin_map(
            project_dir=str(project_dir),
            gauges_fn=_gauges_fn(project_dir),
            plot_dir=str(out_dir),
        )
    written = []
    for extension in (".png", ".pdf"):
        produced = out_dir / f"{_FIGURE_STEM}{extension}"
        if not produced.is_file():
            continue
        if suffix:
            final = out_dir / f"{_FIGURE_STEM}__{suffix}{extension}"
            produced.replace(final)
            produced = final
        written.append(produced)
    return written


def _open(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 — the whole point of --open
    else:
        subprocess.run(["xdg-open" if sys.platform != "darwin" else "open", str(path)])


# --- CLI ----------------------------------------------------------------------


def _print_tunables() -> None:
    docs = _tunable_docs()
    for name, value in _tunables().items():
        print(f"{name} = {value!r}")
        if docs.get(name):
            print(f"    {docs[name]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every tunable with its current value and its comment, then exit",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one tunable for every render; repeatable",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help="render one figure per value; repeatable (renders the cross-product)",
    )
    parser.add_argument(
        "--project-dir",
        help="the folder holding models/hydrology/wflow/ (default: $BASIN_MAP_PROJECT_DIR, "
        "then the first known test project present)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / ".tmp" / "basin_map_preview"),
        help="where the renders go (default: %(default)s)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the first render when it is done",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_tunables()
        return 0

    fixed = dict(
        (name, _coerce(name, value))
        for name, value in (_parse_assignment(item, "--set") for item in args.set)
    )
    swept = [
        (name, _split_values(name, value))
        for name, value in (_parse_assignment(item, "--sweep") for item in args.sweep)
    ]

    project_dir = _resolve_project_dir(args.project_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.is_relative_to(project_dir):
        raise SystemExit(
            f"--out-dir {out_dir} is inside the project directory. Previews must "
            "not land where a workflow writes its own figures."
        )

    # No sweep is the single-render case: one combination, and no name suffix.
    combinations = [
        dict(zip((name for name, _ in swept), choice))
        for choice in itertools.product(*(values for _, values in swept))
    ] or [{}]

    print(f"project : {project_dir}")
    print(f"output  : {out_dir}")
    if fixed:
        print(f"fixed   : {', '.join(f'{k}={v!r}' for k, v in fixed.items())}")
    print(f"renders : {len(combinations)}")

    written = []
    used = set()
    for index, combination in enumerate(combinations, start=1):
        values = {**fixed, **combination}
        # Sanitising for the filesystem can map two distinct values onto one
        # name; numbering the collision keeps a render from overwriting another.
        label = _slug(combination)
        if label in used:
            label = f"{label}__{index}"
        used.add(label)
        print(f"[{index}/{len(combinations)}] {label or 'defaults'}")
        written.extend(render(project_dir, out_dir, values, label))

    if args.open and written:
        _open(written[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
