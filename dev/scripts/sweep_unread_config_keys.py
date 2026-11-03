"""`C-37` — find a declared config key that no reader reads.

The third sweep, and the one R14 left behind: `D-11.4` put everything except
`C-37` and `C-83` in one bundle, and no phase brief picked it up afterwards.
It is also the only remaining answer to design question `P2` — R14 found its
misplaced keys by hand and zero by machine, and a config shape that does not
make this check feasible has not solved the problem with consequences.

**What it asks.** A user scaffolds from `config/templates/example-basin/` and
edits the copied configs. Every key there — live, or commented with an optional example
value — is a promise that setting it does something. A key no code reads is a
promise the toolbox does not keep, and it is invisible: the run succeeds and the
setting changes nothing.

**And its mirror** (t2609070015): a key a shipped seed under `test_case/` sets
that no template documents. The seeds are declared too, so their keys also need
a reader; a seed-only key additionally needs a template that names it.

**Why it was not `sweep_stale_spellings`** (retired with R14's campaign). That
sweep asked whether a spelling is DEAD, from the loader's retired-key table. This asks whether a LIVE key has a
consumer, and its two sides are built from different places entirely.

**Fail-closed** (`D-14.5`, and the reason this note exists at all). R14's own
register re-measure demonstrated the failure mode live: its ground-truth
extraction silently returned zero keys, and the pass then reported MORE problems
while knowing LESS, with nothing in its output saying so. A check like this one
fails open the same way and just as quietly — an empty reader side reports every
key unread, an empty declared side reports a clean bill. So BOTH sides carry a
floor, checked before anything is reported, and a floor breach replaces the
result rather than annotating it.

Usage::

    python dev/scripts/sweep_unread_config_keys.py          # exit 1 on defects
    python dev/scripts/sweep_unread_config_keys.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.scripts.sweep_common import (  # noqa: E402
    REPO_ROOT,
    Allowance,
    Floor,
    check_floors,
)

#: The declared side: the templates a user copies. `config/defaults/` is NOT
#: here — those are build configs a rule reads wholesale and hands to hydromt,
#: whose vocabulary is hydromt's rather than ours (AGENTS.md, Hard Constraints).
#: Nor is `config/templates/archive/`, which is outside the starter directory:
#: its README withholds support and says a stale key there
#: fails at parse time rather than being ignored.
#:
TEMPLATE_GLOB = "config/templates/example-basin/project_config*.yml"

#: The shipped seeds are declared too (t2609070015). A green run once certified
#: the templates only, while the seeds carried keys the templates did not name;
#: most of those were read through forms this check did not recognise
#: (`body["source"]`, `spells.get("dry")`), and two it could not see at all
#: because the templates comment them one level deeper (`#  build_config:`).
#: Only tracked seeds: `project_config_` is the prefix AGENTS.md keeps tracked,
#: and the untracked personal `test*.yml` configs are nobody's promise.
SEED_GLOB = "test_case/project_config_*.yml"

#: A YAML key line, live or commented with an optional example value, so a
#: commented key is a DECLARATION — arguably the more important half, since an
#: unread live key at least shows up in a filled-in config someone reads.
_KEY_LINE = re.compile(
    r"^(?P<indent> *)(?:# *)?(?P<key>[a-z_][a-z0-9_]*)\s*:"
    r"(?P<rest>\s*(?:#.*)?$|\s+\S.*$)"
)

#: "Format: observed_daily_discharge_template.csv." is a sentence, not a key.
#: Prose is what a naive key matcher reports, and one such line was the entire
#: output of this check's first draft.
_PROSE_REST = re.compile(r"^\s+[A-Z][a-z]+\s+[a-z]")

#: The reader side. An identifier that HOLDS a config mapping — NOT any
#: identifier, and the difference decides whether the check can fire at all.
#: Matching every subscript in the tree put 651 names on the reader side and
#: left exactly one key reportable, which is a green run that means nothing.
#: `t1`/`t2` are the loader's names for the two config tiers; `raw` is the
#: freshly-parsed document.
#: `compute`, `spells`, `body` and `entry` hold a config sub-mapping by a name
#: that says what it is rather than that it is config: the batch-sizing stanza,
#: the spell factors, one `variables:` entry, one `future_windows` entry.
#: `climate` is the project climate section passed to the shared map producer.
#: `thresholds` holds the crop-water historical quantile or fixed thresholds.
_CFG_NAME = r"(?:[a-z_]*(?:cfg|config|settings|stanza|section|params)[a-z_]*|t1|t2|raw|workflows|perturbations|compute|spells|body|entry|climate|thresholds)"

#: A subscript chain, and EVERY segment of it counts as a read. `min` reaches a
#: reader only through `stress_test_cfg["temp"]["mean"]["min"]`, and so do
#: `mean` and `temp` — a matcher that recorded the last segment alone reported
#: the intermediate keys unread, which is a false alarm on the deepest and most
#: error-prone part of the config.
#:
#: A segment subscripted by a VARIABLE is stepped over rather than ending the
#: chain: `axis_cfg[axis_name]["n_levels"]` reads `n_levels` past one, and
#: stopping there reported both stress-test axis keys unread.
_CHAIN_SEGMENT = re.compile(r"""\[\s*(?:["'](?P<key>[\w-]+)["']|[^]\n]{0,40})\s*\]""")
_CHAIN_HEAD = re.compile(rf"{_CFG_NAME}\s*(?=\[)")

_GET_CONFIG = re.compile(rf"""get_config\(\s*{_CFG_NAME}[^,]*,\s*["']([\w-]+)["']""")
_DOT_GET = re.compile(
    rf"""{_CFG_NAME}(?:\[[^]\n]{{0,40}}\])*\s*\.\s*get\(\s*["']([\w-]+)["']"""
)
#: A key handed to an accessor, not subscripted: `_source_name(sources_cfg,
#: "lai", ...)` in `spatial/config.py`, and batch sizing's local
#: `pick("julia_threads", ...)`, whose first argument is the `compute` key.
_ACCESSOR_CALL = re.compile(
    rf"""(?:\w+\(\s*{_CFG_NAME}\s*,|\bpick\()\s*["']([\w-]+)["']"""
)
#: R takes its config through `commandArgs`, then reads it as a list.
_R_DOLLAR = re.compile(rf"{_CFG_NAME}\$([A-Za-z_][\w.]*)")
_R_SUBSCRIPT = re.compile(rf"""{_CFG_NAME}\[\[\s*["']([\w-]+)["']\s*\]\]""")

ALLOWANCES = (
    Allowance(
        name="a key pair read by iterating a literal tuple",
        reason=(
            "`window` is validated and rendered as a PAIR, so its two leaves "
            "are read by `for key in ('start', 'end')` rather than by name -- "
            "`snake_utils.validate_historical_window` and `window_year_pair`. "
            "Naming them separately would let one bound be checked and the "
            "other not, which is the bug the loop shape prevents"
        ),
        line_patterns=(r"^(start|end)$",),
    ),
)

#: Measured on the live tree. The floors sit well under the counts so honest
#: churn does not trip them, and well over zero because zero is the number that
#: makes this check lie: an empty reader side reports every key unread, and an
#: empty declared side reports a clean bill.
FLOORS = (
    Floor(
        name="declared",
        minimum=30,
        count=79,
        measured="2026-09-25, seeds declared (t2609070015)",
        why=(
            "the starter configs are the config surface a user copies; a "
            "collapse means the glob or the key matcher stopped seeing them, "
            "and an empty declared side reports a clean bill while knowing "
            "nothing"
        ),
    ),
    Floor(
        name="readers",
        minimum=45,
        count=168,
        measured="2026-09-25, seeds declared (t2609070015)",
        why=(
            "an empty reader side reports EVERY declared key as unread, which "
            "is the shape R14's re-measure failed in -- more findings, less "
            "knowledge, nothing in the output saying so"
        ),
    ),
    Floor(
        name="surfaces",
        minimum=70,
        count=150,
        measured="2026-09-25, seeds declared (t2609070015)",
        why=(
            "the files the reader side is extracted from. Counted separately "
            "because a walk that returns nothing and a walk that returns files "
            "with no config reads are different failures with the same symptom"
        ),
    ),
)


def _yaml_keys(node, out: set) -> set:
    """Every mapping key in a parsed YAML document, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            out.add(str(key))
            _yaml_keys(value, out)
    elif isinstance(node, list):
        for value in node:
            _yaml_keys(value, out)
    return out


def declared_keys(root: Path = REPO_ROOT) -> dict[str, tuple[str, int, str]]:
    """``{key: (file, line, text)}`` for every key the templates or seeds declare.

    Templates are read LINE BY LINE, because their commented defaults are
    declarations YAML would discard. Seeds are real YAML with nothing commented
    out, so they are PARSED, which also sees flow-style keys
    (`{source: precip, canonical: rate}`) a line matcher cannot. A user-chosen
    name outside the key grammar (`river discharge`) is not a declared key.
    Templates first, so a key both declare is reported against the template.
    """
    import yaml

    out: dict[str, tuple[str, int, str]] = {}
    for path in sorted(root.glob(TEMPLATE_GLOB)):
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _KEY_LINE.match(line)
            if match is None or _PROSE_REST.match(match.group("rest")):
                continue
            out.setdefault(match.group("key"), (rel, lineno, line.strip()[:80]))
    grammar = re.compile(r"[a-z_][a-z0-9_]*")
    for path in sorted(root.glob(SEED_GLOB)):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for key in sorted(_yaml_keys(yaml.safe_load(text), set())):
            if key in out or not grammar.fullmatch(key):
                continue
            where = re.compile(rf"(?<![\w-]){re.escape(key)}\s*:")
            lineno = next(
                (i for i, line in enumerate(lines, 1) if where.search(line)), 0
            )
            text_line = lines[lineno - 1].strip()[:80] if lineno else key
            out[key] = (rel, lineno, text_line)
    return out


def reader_surfaces(root: Path = REPO_ROOT) -> list[Path]:
    """The run-path files a config key can be read on.

    The same invocation-model rule the D-9.6 scan follows: the four entry
    points, the shipped package and `scripts/`. `dev/scripts/` is never part of
    a run, so a key read only there is a key no run reads.
    """
    found = [
        root / f"{name}.smk"
        for name in (
            "analyze_climate",
            "build_model",
            "analyze_projections",
            "generate_scenarios",
            "simulate_system",
        )
    ]
    for package in ("blueearth_cst", "scripts"):
        for pattern in ("*.py", "*.R", "*.smk"):
            found += sorted((root / package).rglob(pattern))
    return [
        path for path in found if path.is_file() and "__pycache__" not in path.parts
    ]


def reader_keys(root: Path = REPO_ROOT) -> dict[str, str]:
    """``{key: where first seen}`` for every key some run-path file reads."""
    seen: dict[str, str] = {}
    for path in reader_surfaces(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        patterns = (
            (_R_DOLLAR, _R_SUBSCRIPT)
            if path.suffix == ".R"
            else (_GET_CONFIG, _DOT_GET, _ACCESSOR_CALL)
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                seen.setdefault(match.group(1), rel)
        if path.suffix == ".R":
            continue
        for head in _CHAIN_HEAD.finditer(text):
            tail = text[head.end() :]
            while True:
                segment = _CHAIN_SEGMENT.match(tail)
                if segment is None:
                    break
                if segment.group("key"):
                    seen.setdefault(segment.group("key"), rel)
                tail = tail[segment.end() :]
    return seen


def sweep(root: Path = REPO_ROOT):
    """Return ``(defects, allowed, observed)``.

    ``observed`` carries both ground truths and the surface count, so the
    floors are checked against what was actually extracted rather than against
    a side effect of it.
    """
    declared = declared_keys(root)
    readers = reader_keys(root)
    observed = {
        "declared": len(declared),
        "readers": len(readers),
        "surfaces": len(reader_surfaces(root)),
    }
    defects, allowed = [], []
    for key, (rel, lineno, text) in sorted(declared.items()):
        if key in readers:
            continue
        # The allowance is matched on the KEY, not on the template line: what
        # is being excused is the way a key is read, which the template cannot
        # show.
        for allowance in ALLOWANCES:
            if allowance.covers(root / rel, key):
                allowed.append((root / rel, lineno, text, allowance.name))
                break
        else:
            defects.append((root / rel, lineno, text, "UNREAD"))
    # The MIRROR: a key a shipped seed sets that no template documents, so a
    # user copying the templates never learns it exists. A template counts as
    # documenting a key it names anywhere, flow style included --
    # `- {start: 2046, end: 2054, name: mid}` documents `name`.
    template_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.glob(TEMPLATE_GLOB))
    )
    for key, (rel, lineno, text) in sorted(declared.items()):
        if rel.startswith("config/templates/"):
            continue
        if not re.search(rf"(?<![\w-]){re.escape(key)}\s*:", template_text):
            defects.append((root / rel, lineno, text, "UNDOCUMENTED"))
    return defects, allowed, observed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    defects, allowed, observed = sweep()
    blind = check_floors(FLOORS, observed)

    if args.json:
        print(
            json.dumps(
                {
                    "defects": [
                        {
                            "path": str(p.relative_to(REPO_ROOT)),
                            "line": n,
                            "text": t,
                            "kind": kind,
                        }
                        for p, n, t, kind in defects
                    ],
                    "allowed": len(allowed),
                    "observed": observed,
                    "blind": blind,
                },
                indent=1,
            )
        )
    elif blind:
        # Replaces the result rather than annotating it. A check that cannot
        # see one of its two sides has no opinion about the other, and printing
        # a defect count beside this warning invites reading the count.
        print("THIS CHECK IS BLIND and its result means nothing:\n")
        for message in blind:
            print(f"  {message}")
    else:
        print(
            f"{observed['declared']} declared key(s) against "
            f"{observed['readers']} read across {observed['surfaces']} surfaces."
        )
        print(f"{len(allowed)} declared-but-unmatched key(s) classified.")
        if defects:
            print(
                f"\n{len(defects)} key(s) UNREAD (declared, no reader) or "
                "UNDOCUMENTED (a seed sets it, no template names it):\n"
            )
            for path, lineno, text, kind in defects:
                print(f"  {kind}  {path.relative_to(REPO_ROOT)}:{lineno}")
                print(f"      {text[:110]}")
        else:
            print("Every declared key reaches a reader.")
    return 1 if (defects or blind) else 0


if __name__ == "__main__":
    raise SystemExit(main())
