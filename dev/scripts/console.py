"""Shared console-formatting helpers (colour, glyphs, path rendering).

Vendored from the `console-formatting` skill — keep this file in sync with
the upstream copy. Self-contained, no third-party deps.

Public API:
    USE_COLOR (bool, evaluated once at import)
    green / yellow / red / cyan / dim / bold      colour wrappers
    fmt_path(p)                                    cwd-relative > ~-prefixed > absolute
    relto(p, base)                                 p relative to base, posix slashes
    warn(msg)                                      prints `! <msg>` in yellow
    pad(text, width, color=None)                   left-pad plain text to width, then colour
    banner(label)                                  cyan-bold `━━ LABEL ━━`
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _enable_windows_ansi() -> bool:
    if platform.system() != "Windows":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VT
        return True
    except Exception:
        return False


def _color_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    return _enable_windows_ansi()


USE_COLOR = _color_supported()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(s):
    return _c(str(s), "32")


def yellow(s):
    return _c(str(s), "33")


def red(s):
    return _c(str(s), "31")


def cyan(s):
    return _c(str(s), "36")


def dim(s):
    return _c(str(s), "2")


def bold(s):
    return _c(str(s), "1")


def fmt_path(p) -> str:
    p = Path(p)
    s = str(p).replace("\\", "/")
    home = str(Path.home()).replace("\\", "/")
    if s.startswith(home):
        s = "~" + s[len(home) :]
    try:
        rel = os.path.relpath(p, Path.cwd()).replace("\\", "/")
    except ValueError:
        rel = None
    if rel and not rel.startswith("..") and len(rel) < len(s):
        return rel
    return s


def relto(p, base) -> str:
    try:
        return Path(os.path.relpath(p, base)).as_posix()
    except ValueError:
        return fmt_path(p)


def warn(msg: str) -> None:
    print(f"{yellow('!')} {msg}")


def pad(text: str, width: int, color=None) -> str:
    out = text.ljust(width)
    return color(out) if color else out


def banner(label: str) -> str:
    return cyan(bold(f"━━ {label.upper()} ━━"))
