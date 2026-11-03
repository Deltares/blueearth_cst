"""Find non-ASCII in strings that reach a CONSOLE.

Run it from the repo root: `python dev/scripts/scan_console_encoding.py`.

Narrowed to the call sites that print or raise: a figure label may hold any
character it likes, because it goes into a PNG. A console row may not -- a
Windows console defaults to cp1252 and raises `UnicodeEncodeError`.

Writes its findings to a file rather than stdout, because printing them is the
very thing that fails.
"""

from __future__ import annotations

import ast
import pathlib

EMITTERS = {"log_row", "warn_row", "defer_warning", "print"}
OUT = pathlib.Path(".tmp/console-encoding.txt")


def strings(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child


def scan(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        emitter = None
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in EMITTERS:
                emitter = name
            elif name and name.endswith("Error"):
                emitter = name
        elif isinstance(node, ast.Raise):
            emitter = "raise"
        if emitter is None:
            continue
        for const in strings(node):
            bad = sorted({c for c in const.value if ord(c) > 127})
            if bad:
                yield (
                    path,
                    const.lineno,
                    emitter,
                    "".join(f"U+{ord(c):04X}" for c in bad),
                    const.value[:70],
                )


rows = []
for p in sorted(pathlib.Path("blueearth_cst").rglob("*.py")):
    rows.extend(scan(p))
for p in sorted(pathlib.Path(".").glob("*.smk")):
    rows.extend(scan(p))

lines = [f"{len(rows)} console-bound strings with non-ASCII", ""]
for path, lineno, emitter, chars, text in rows:
    lines.append(f"{path}:{lineno}  via {emitter}  [{chars}]")
    lines.append(f"    {text!r}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"{len(rows)} findings -> {OUT}")
