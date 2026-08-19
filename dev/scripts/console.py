"""Shared console-formatting helpers (colour, glyphs, path rendering).

A drop-in module for any Python ≥ 3.9 CLI. Encodes a small, opinionated
look — modelled on `codex doctor` — so commands across a project (or across
sibling projects) read consistently: a `Name v… · platform` title line,
bold Title-Case group headings, glyph-keyed check rows over indented
key/value detail, full-width rules, forward-slashed shortest-form paths,
and a closing count tally.

No third-party dependencies. Auto-disables colour when stdout is not a
TTY, when `NO_COLOR=1` is set, or when the Windows console can't enable
ANSI; every wrapper degrades to plain text in that case.

Public API:
    __version__ (str — the skill release this copy was vendored from)
    USE_COLOR (bool, evaluated once at import)
    USE_ASCII (bool, evaluated once at import — glyphs degrade to ASCII)
    green / yellow / red / cyan / magenta / dim / bold   colour wrappers
    glyph(ch)                                      non-ASCII glyph → ASCII when USE_ASCII
    status_glyph(state)                            coloured ✓ ○ ↑ ⚠ ✗ (→ [ok] [--] [up] [!!] [XX])
    fmt_path(p)                                    cwd-relative > ~-prefixed > absolute
    relto(p, base)                                 p relative to base, posix slashes
    warn(msg)                                      prints `! <msg>` in yellow
    pad(text, width, color=None)                   left-pad plain text to width, then colour
    banner(label)                                  bold Title Case, no decoration (group / region heading)
    section_banner(label)                          bold Title Case, auto-cased (router phase heading)

Diagnostic / status report layout (modelled on `codex doctor`):
    report_title(name, version, platform)          `Name v… · platform` title line
    rule(width=None)                               full-width `─ … ─` zone separator
    check_row(state, name, summary)                `  <glyph> <name>  <summary>`
    detail_row(key, value)                         `      <key>  <value>` under a check row
    status_tally(counts, verdict=None)             `17 ok · 1 idle · 0 fail  ok`
    hint_footer(pairs)                             dim grid of `<flag>  <desc>` mode hints

Width convention:
    Content lines (wrapped lists, prose, fitted paths) target `content_width()`
    — `min(terminal, 92)` — so long lines wrap to multiple lines (or truncate)
    rather than sprawling on a wide terminal. Override the 92 cap with
    `CONSOLE_MAX_WIDTH`. Full-terminal layout (rules, tables) may still use
    `term_width()` directly; flowing *content* should go through `content_width()`.

Env overrides:
    Optional overrides use the `ENV_PREFIX` (default `CONSOLE_`): `CONSOLE_ASCII`
    forces ASCII glyphs, `CONSOLE_MAX_WIDTH` sets the content-width cap. Change
    `ENV_PREFIX` once to rebrand them all for a vendoring project. The standard
    `NO_COLOR` is honoured unprefixed.

Higher-level helpers (built on the primitives above):
    term_width(default=100)                        adaptive terminal width (full)
    content_width()                                min(terminal, 92) — wrap target for content
    wrap(text, indent="", …)                       fill prose to content_width(), hanging indent
    hang(head, tail, head_w=None)                  fixed head + tail wrapped under tail's start
    bullet(marker, text, indent=4)                 `<indent><glyph> <msg>` finding, hanging-wrapped
    name_list(label, names, …)                     ` · `-joined list, wrapped to content_width()
    fit_path(path_str, reserve=30)                 middle-ellipsis to fit content_width()
    show_path(p, reserve=30)                       fit_path(fmt_path(p)) — body-line path
    mode_badge(dry_run)                            `[dry-run]` / `[apply]` for the `done` line
    cli_error(action, reason, hint=None)           canonical error block to stderr
    examples_epilog(*lines)                        `examples:` argparse epilog (matches `options:`)
    next_hint(text)                                `→ next: …` styled hint to stdout
    progress_supported()                           True when ProgressBar can redraw in place
    ProgressBar(total, width=28)                   single-line bar; .draw / .tick / .clear
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import textwrap
from pathlib import Path

#: The `console-formatting` release this file is a copy of, matching that
#: skill's `SKILL.md` frontmatter `version:`. A vendored copy drifts silently —
#: it keeps working while missing helpers the skill's guidance already assumes —
#: so compare this against upstream before editing a consumer's console surface.
#: A copy with NO `__version__` at all predates the marker and is therefore
#: older than every version that carries one.
__version__ = "0.10.0"

# Prefix for this module's optional env-var overrides. Neutral by default so the
# module vendors cleanly into any project; a consumer can rebrand every override
# in one place by changing this (e.g. "MYAPP_"). The cross-tool standard
# `NO_COLOR` is honoured unprefixed and is unaffected.
ENV_PREFIX = "CONSOLE_"


def _env(name: str) -> str | None:
    """Read an optional override env var under `ENV_PREFIX` (e.g. `MAX_WIDTH`
    → `CONSOLE_MAX_WIDTH`)."""
    return os.environ.get(ENV_PREFIX + name)


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


#: Every non-ASCII character this module can emit. `_encoding_incapable()`
#: probes stdout against exactly this string, so a glyph added to the vocabulary
#: belongs here and in `_GLYPH_ASCII` in the SAME edit: a glyph the probe does
#: not carry is one whose fallback may never fire, on an encoding that happens
#: to accept the rest of the set.
_GLYPH_VOCABULARY = "✓○↑⚠✗▸▌·─→…█░"


def _encoding_incapable() -> bool:
    """True when stdout's encoding cannot represent `_GLYPH_VOCABULARY`."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        _GLYPH_VOCABULARY.encode(enc)
        return False
    except (UnicodeEncodeError, LookupError):
        return True


def _ascii_only() -> bool:
    """True when glyphs must degrade to ASCII: `CONSOLE_ASCII` is set, or stdout's
    encoding can't represent the non-ASCII glyph vocabulary (`✓ ○ ↑ ⚠ ✗ ▸ · ─ → …`).

    Independent of `USE_COLOR` — colour and glyph capability are separate axes.
    Evaluated once at import, like `USE_COLOR`.
    """
    return bool(_env("ASCII")) or _encoding_incapable()


def _survive_narrow_stdout() -> None:
    """Make an encoding-incapable stdout lossy rather than fatal.

    `USE_ASCII` degrades the glyphs this module emits, and that covers the
    module's own vocabulary only. A command's PROSE is not routed through
    `glyph()` — an em dash in a verdict, a `·` between counts, a curly quote in
    a hint — so on a stream whose encoding cannot carry it, `print()` raises
    `UnicodeEncodeError` and the run dies mid-report with the work already
    done. Measured under `PYTHONIOENCODING=ascii`: `rule()` renders `-`
    correctly, and the next line, `bold("OK — done")`, ends the process.

    The brain dispatcher hides this by reconfiguring stdout to UTF-8 before
    import, but that machinery does not travel with a vendored copy of this
    file — so a consumer that drops the module in as documented gets the
    degradation story without the stream it quietly assumes.

    `errors="replace"` rather than a forced UTF-8 encoding: the stream keeps
    its own encoding, so a capable console is untouched and a narrow one loses
    a character instead of the run. At import, and by the same licence
    `USE_COLOR` already takes when it calls `SetConsoleMode` on Windows — this
    module configures the console it is about to write to.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            # Not a reconfigurable text stream (a capture object, a replaced
            # stream, a raw buffer): nothing to harden here, and nothing that
            # would have raised either.
            pass


USE_ASCII = _ascii_only()
if _encoding_incapable():
    _survive_narrow_stdout()

# Non-ASCII glyph → ASCII fallback. The diff glyphs (`+ ~ = - ! x`) are already
# ASCII and need no entry; the status glyphs (`✓ ○ ↑ ⚠ ✗`) degrade separately
# via `status_glyph()`. `glyph()` applies this map when `USE_ASCII`.
_GLYPH_ASCII = {
    "▸": ">", "▌": "#", "·": "|", "─": "-",
    "→": "->", "✗": "x", "…": "...", "█": "#", "░": "-",
}


def glyph(ch: str) -> str:
    """Return the ASCII fallback for a decorative glyph when stdout can't render
    it (see `USE_ASCII`); otherwise the glyph unchanged.

    Wrap any non-ASCII glyph a command emits — `▸ ▌ · ─ → ✗` — so it
    degrades to its ASCII form (`> # | - -> x`) instead of becoming `?`
    on a non-UTF-8 console. Colour wrappers compose on top: `cyan(glyph("▸"))`.
    The status glyphs (`✓ ○ ↑ ⚠ ✗`) go through `status_glyph()` instead.
    Reads the module-level `USE_ASCII` at call time, so the value is current
    even if a caller forces it.
    """
    return _GLYPH_ASCII.get(ch, ch) if USE_ASCII else ch


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(s):   return _c(str(s), "32")
def yellow(s):  return _c(str(s), "33")
def red(s):     return _c(str(s), "31")
def cyan(s):    return _c(str(s), "36")
def magenta(s): return _c(str(s), "35")
def dim(s):     return _c(str(s), "2")
def bold(s):    return _c(str(s), "1")


# ---------------------------------------------------------------------------
# Status glyph vocabulary — the health / report states of a diagnostic or
# status command (`✓ ○ ↑ ⚠ ✗`), distinct from the diff / change glyphs
# (`+ ~ = -`) an *action* command uses. Each state resolves to a glyph, an
# ASCII fallback that carries the state in text (so it survives a non-UTF-8
# console and a screen reader), and a colour wrapper. `status_glyph()` picks
# the right form for the current console; `check_row()` renders a full row.
#
#   ok      ✓  [ok]  green   passing / healthy
#   idle    ○  [--]  dim     inactive / not running (a benign non-state)
#   update  ↑  [up]  cyan    an update or notice is available (also `notes`)
#   warn    ⚠  [!!]  yellow  degraded — needs attention, not broken
#   fail    ✗  [XX]  red     failed / broken / missing
#
# The ASCII forms are deliberately not in `_GLYPH_ASCII`: `✗` maps to a bare
# `x` there for the `cli_error` header, but as a *status* glyph it degrades to
# `[XX]`, so the two contexts must resolve independently.
_STATUS_GLYPH = {
    "ok":     ("✓", "[ok]", green),
    "idle":   ("○", "[--]", dim),
    "update": ("↑", "[up]", cyan),
    "notes":  ("↑", "[up]", cyan),
    "warn":   ("⚠", "[!!]", yellow),
    "fail":   ("✗", "[XX]", red),
}


def status_glyph(status: str) -> str:
    """Return the coloured status glyph for one of the health states
    (`ok idle update notes warn fail`), degraded to its bracketed ASCII form
    (`[ok] [--] [up] [!!] [XX]`) when `USE_ASCII`. The vocabulary for a status
    or diagnostic command; an action command reporting change uses the diff
    glyphs (`+ ~ = -`) instead."""
    ch, ascii_form, color = _STATUS_GLYPH[status]
    return color(ascii_form if USE_ASCII else ch)


def fmt_path(p) -> str:
    """Shortest readable form: cwd-relative, else ~-prefixed, else absolute. Forward slashes."""
    p = Path(p)
    s = str(p).replace("\\", "/")
    home = str(Path.home()).replace("\\", "/")
    if s.startswith(home):
        s = "~" + s[len(home):]
    try:
        rel = os.path.relpath(p, Path.cwd()).replace("\\", "/")
    except ValueError:
        rel = None
    if rel and not rel.startswith("..") and len(rel) < len(s):
        return rel
    return s


def relto(p, base) -> str:
    """Render p relative to base with forward slashes; falls back to fmt_path."""
    try:
        return Path(os.path.relpath(p, base)).as_posix()
    except ValueError:
        return fmt_path(p)


def warn(msg: str) -> None:
    print(f"{yellow('!')} {msg}")


def pad(text: str, width: int, color=None) -> str:
    """Left-pad plain text to `width` visible columns, then optionally colour.

    Always pad first, colour after — wrapping with a colour wrapper *then*
    using format-spec width counts ANSI bytes against the column, which
    breaks visible alignment when different rows use different colours.
    Pass the bare label / state cell here:

        print(f"  {pad('mode', 10, dim)} {pad('hifi', 8, green)}  {dim(note)}")
    """
    out = text.ljust(width)
    return color(out) if color else out


def _titlecase(label: str) -> str:
    """Capitalise the first letter of each word for a Title-Case heading,
    preserving all-caps tokens (`MCP`, `PATH`) and anything not starting with
    a letter (paths, `1/2`, `::`). Conservative on purpose — it upper-cases a
    leading lowercase letter and touches nothing else, so a constructed heading
    like `promote :: some-file.md` becomes `Promote :: some-file.md` without
    mangling the dynamic tail."""
    def cap(word: str) -> str:
        if not word or not word[0].isalpha() or word.isupper():
            return word
        return word[0].upper() + word[1:]
    return " ".join(cap(w) for w in label.split(" "))


def banner(label: str) -> str:
    """Render a group / region heading: bold Title Case, no decoration.

    Use for every region a command emits — `Summary`, `Settings`,
    `Needs Attention`, `Done`, body step headers. The header sits at
    column 0; all content underneath should be indented at least one
    level (2 spaces) so the reader can see nesting at a glance.

    Distinguishing features (no rule decoration):
      - **Title Case** label, as written by the caller — pass `"Summary"`,
        not `"summary"` (the helper does not transform the label, so a
        dynamic heading with a path in it is never mangled)
      - **bold** weight
      - blank line above (caller's job)
      - body indented relative to the header (caller's job)

    Sub-section labels (`inputs:`, `flags:`, `agents:`) and nested subject
    names sit one indent level deeper and stay lowercase, so they read as
    distinct from the bold col-0 Title-Case heading. For a higher-level
    phase marker in a router, use `section_banner()`.
    """
    return bold(label)


def section_banner(label: str) -> str:
    """Render a section / phase heading: bold Title Case, no decoration.

    A router or multi-phase command marks each phase with this above the
    regions belonging to it. Unlike `banner()`, it Title-Cases the label for
    you (`section_banner("external")` → **External**), because a phase marker
    is always a name safe to capitalise. Pair it with a full-width `rule()`
    above the phase when the run needs a hard visual break between phases;
    surrounding callers add the blank lines for breathing room.
    """
    return bold(_titlecase(label))


# ---------------------------------------------------------------------------
# Higher-level helpers — built on the primitives above. Each encodes one
# small CLI-output convention (mode badge, error block, examples region,
# next-step hint) so commands using this module read interchangeably.

def term_width(default: int = 100) -> int:
    """Adaptive terminal width with a sane fallback."""
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default


# Cap for wrapped *content* (name lists, inventories). Long lines stay readable
# on very wide terminals instead of running the full width before wrapping.
# Override with CONSOLE_MAX_WIDTH; narrower terminals still win.
try:
    _MAX_CONTENT_WIDTH = int(_env("MAX_WIDTH") or 92)
except ValueError:
    _MAX_CONTENT_WIDTH = 92


def content_width() -> int:
    """Wrap target for content lines: the terminal width, capped so lines wrap
    to multiple lines on wide terminals rather than sprawling."""
    return min(term_width(), _MAX_CONTENT_WIDTH)


def wrap(text: str, *, indent: str = "", subsequent_indent: str | None = None,
         width: int | None = None) -> str:
    """Fill flowing prose to `content_width()` (min terminal, 92), returning a
    joined multi-line string. The standard wrapper for prose-like CLI content —
    use it instead of `textwrap.fill`/`term_width()` so every command wraps at
    the same width. `subsequent_indent` defaults to `indent` (hanging indent);
    long unbreakable tokens (paths, slugs) are never split."""
    w = max(8, (width or content_width()) - len(indent))
    sub = indent if subsequent_indent is None else subsequent_indent
    lines = textwrap.wrap(text, width=w, break_long_words=False,
                          break_on_hyphens=False) or [""]
    return "\n".join((indent if i == 0 else sub) + ln for i, ln in enumerate(lines))


def hang(head: str, tail: str, *, head_w: int | None = None,
         width: int | None = None) -> str:
    """Render `head` immediately followed by `tail`, wrapping `tail` to
    `content_width()` with a hanging indent aligned under where `tail` begins,
    so a long tail (and any path it carries) never exceeds the shared cap. Use
    for `<fixed-head> <flowing-tail>` finding lines. `head_w` is the *visible*
    width of `head`; pass it when `head` carries ANSI colour (its byte length
    would otherwise overcount the indent). `tail` must be plain text — apply
    colour to `head`, not the wrapped body. Long unbreakable tokens (paths,
    slugs) are never split; a path longer than the line occupies its own line
    whole rather than being truncated."""
    hw = head_w if head_w is not None else len(head)
    pad_ = " " * hw
    wrapped = wrap(tail, indent=pad_, subsequent_indent=pad_, width=width)
    return head + wrapped[hw:]


def bullet(marker: str, text: str, *, indent: int = 4,
           width: int | None = None) -> str:
    """Render a `<indent-spaces><marker> <text>` finding line, wrapping `text`
    to `content_width()` with a hanging indent aligned under the text. The
    standard shape for a linter/audit finding: `marker` is the already-coloured
    one-column glyph (e.g. ``green('+')`` or a severity glyph), `text` is the
    plain message. Continuation lines align under the message, not the glyph, so
    multi-line findings stay scannable and never run past the cap."""
    lead = " " * indent
    return hang(f"{lead}{marker} ", text, head_w=indent + 2, width=width)


def name_list(label, names, *, indent="  ", label_w=None, sep=" · ",
              label_color=bold, name_color=None, width=None) -> None:
    """Print `label` then `names` joined by `sep`, wrapped to the terminal with
    a hanging indent aligned under the first name:

        label (62)   alpha · bravo · charlie · delta · echo · foxtrot ·
                     golf · hotel

    Names are wrapped on plain length and coloured after, so ANSI bytes never
    count against the column. `label_w` right-pads the label cell so sibling
    calls align their name columns; it defaults to the label's own width.
    `label_color` styles the label cell, `name_color` (a `str -> str` colour
    fn) styles every name uniformly. No-op on an empty `names`.
    """
    if not names:
        return
    width = width or content_width()
    if label_w is None:
        label_w = len(label)
    name_col = len(indent) + label_w + 2  # 2-space gap between label and names
    hang = " " * name_col
    out_lines: list[list[str]] = []
    line: list[str] = []
    cur = name_col
    for n in names:
        add = len(n) + (len(sep) if line else 0)
        if line and cur + add > width:
            out_lines.append(line)
            line, cur = [n], name_col + len(n)
        else:
            line.append(n)
            cur += add
    if line:
        out_lines.append(line)
    for i, ln in enumerate(out_lines):
        rendered = dim(sep).join(name_color(n) if name_color else n for n in ln)
        prefix = f"{indent}{label_color(label.ljust(label_w))}  " if i == 0 else hang
        print(prefix + rendered)


def fit_path(path_str: str, reserve: int = 30) -> str:
    """Middle-truncate a path so a body line still fits the terminal.

    `reserve` is how much horizontal space to keep free for the trailing
    context + actions on the right. The filename is always preserved. Bounded by
    `content_width()` so path-bearing lines respect the same cap as wrapped text.
    """
    max_w = max(24, content_width() - reserve)
    if len(path_str) <= max_w:
        return path_str
    ell = glyph("…")
    parts = path_str.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return ell + path_str[-(max_w - len(ell)):]
    name = parts[-1]
    head = parts[0]
    candidate = f"{head}/{ell}/{parts[-2]}/{name}"
    if len(candidate) <= max_w:
        return candidate
    candidate = f"{head}/{ell}/{name}"
    if len(candidate) <= max_w:
        return candidate
    return ell + path_str[-(max_w - len(ell)):]


def show_path(p, reserve: int = 30) -> str:
    """Render a Path for body lines: shortest readable form, width-fitted."""
    return fit_path(fmt_path(p), reserve=reserve)


def mode_badge(dry_run: bool) -> str:
    """Return the canonical mode badge for the `done` receipt line.

    `[dry-run]` in bold yellow when previewing; `[apply]` in bold green
    when the run will actually write. Place at the start of the `done`
    line so "am I about to write?" is scannable at a glance.
    """
    return bold(yellow("[dry-run]")) if dry_run else bold(green("[apply]"))


def cli_error(action: str, reason: str, hint: str | None = None) -> None:
    """Write a canonical action-oriented error block to stderr.

    Format:
      ✗ <action>
        reason  <one-line cause>
        → try:  <suggested next command>

    Use instead of bare `stderr.write("foo: bar\\n")`. The `✗` glyph is
    red; `→ try:` is cyan. Always include a hint when there's a sensible
    fix the user can paste; omit when there isn't.
    """
    sys.stderr.write(f"{red(glyph('✗'))} {bold(action)}\n")
    sys.stderr.write(f"  {dim('reason')}  {reason}\n")
    if hint:
        sys.stderr.write(f"  {cyan(glyph('→') + ' try:')}  {hint}\n")


def examples_epilog(*lines: str) -> str:
    """Build an `examples:` argparse epilog from a sequence of lines.

    Each `line` is one example invocation; this helper stacks them under a
    lowercase `examples:` header that matches argparse's own `options:` and
    `positional arguments:` section headers, so the whole `--help` page reads
    uniformly. Pair with
    `formatter_class=argparse.RawDescriptionHelpFormatter` so the header and
    lines render intact.

        parser = argparse.ArgumentParser(
            ...,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=examples_epilog(
                "tool --flag             short description",
                "tool subcmd --dry-run   preview only",
            ),
        )
    """
    body = "\n".join(f"  {line}" for line in lines)
    # `bold("examples:")` mirrors how the help formatter bolds argparse's own
    # section headers: bold lowercase in a terminal, plain `examples:` when
    # colour is off — matching `options:` in both modes.
    return bold("examples:") + "\n" + body + "\n"


def output_mode_from_args(args) -> str:
    """Resolve a single output-mode label from argparse Namespace.

    Returns one of: `"json"`, `"quiet"`, `"verbose"`, `"default"`. The mutually
    exclusive group is resolved here so each command can branch off a single
    label rather than the full flag matrix.
    """
    if getattr(args, "json", False):
        return "json"
    if getattr(args, "quiet", False):
        return "quiet"
    if getattr(args, "verbose", False):
        return "verbose"
    return "default"


def elapsed_str(seconds: float) -> str:
    """Format a wall-clock duration for the TOTAL line.

    Sub-second runs render in ms ("420ms"); seconds with one decimal up to
    a minute ("3.2s"); minutes-and-seconds beyond ("1m12s"). Always short
    enough to share a line with `mode_badge()` and a counts pill.
    """
    if seconds < 1.0:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds - minutes * 60)
    return f"{minutes}m{rem:02d}s"


def next_hint(text: str) -> None:
    """Print the canonical next-step hint line: `→ next: <text>`.

    Use after a successful command run to point the reader at the
    likely next command. A multi-command dispatcher can wire this
    automatically using a per-command "next" field; standalone scripts
    call it directly at the end of `main()`.
    """
    print()
    print(f"{cyan(glyph('→') + ' next:')} {dim(text)}")


# ---------------------------------------------------------------------------
# Diagnostic / status report layout — the grid a read-only health or status
# command prints: a title line, Title-Case group headings, one glyph-keyed
# check row per subject, indented key/value detail rows, a full-width rule,
# and a count tally. See `references/layout-command-shapes.md` → *Status commands*.

def report_title(name: str, version: str | None = None,
                 platform_str: str | None = None) -> str:
    """Render the top title line of a status / diagnostic report:
    `<Name> v<version> · <platform>`. The name (and version) are bold; the
    ` · ` separators are dim. Print it as the first line, followed by a blank
    line. Omit `version` / `platform_str` for a bare title."""
    head = f"{name} v{version}" if version else name
    line = bold(head)
    if platform_str:
        line = f"{line} {dim(glyph('·'))} {platform_str}"
    return line


def rule(width: int | None = None, char: str = "─") -> str:
    """Return a full-width horizontal rule (`─ … ─`), the separator between
    the report's major zones — a notices callout, the body, the tally footer.
    Spans `content_width()` by default (capped so it doesn't sprawl on a very
    wide terminal); pass `width` to size it explicitly. Degrades to `-`."""
    return glyph(char) * (width or content_width())


def check_row(status: str, name: str, summary: str = "", *, name_w: int = 13) -> str:
    """Render one check row: `  <glyph> <name>  <summary>`.

    Two-space indent, the coloured `status_glyph()` for one of
    `ok idle update warn fail`, the check `name` left-aligned in a fixed
    `name_w`-wide column, then a one-line `summary`. The name column keeps
    every row's summary aligned; a name at or past `name_w` gets a single
    trailing space so the summary never abuts it. Emit `detail_row()`s
    underneath for the expanded view."""
    field = name if len(name) < name_w else name + " "
    return f"  {status_glyph(status)} {field.ljust(name_w)}{summary}"


def detail_row(key: str, value: str = "", *, key_w: int = 25) -> str:
    """Render one detail row under a `check_row()`: `      <key>  <value>`.

    Six-space indent (nested under the col-2 check row), the `key` dim and
    left-aligned in a fixed `key_w`-wide column, then the `value`. A key at or
    past `key_w` overflows with a single trailing space. Shown in the detailed
    view; suppressed in a `--summary`-style compact view."""
    field = key if len(key) < key_w else key + " "
    return f"      {dim(field.ljust(key_w))}{value}"


def status_tally(counts: dict, verdict: str | None = None) -> str:
    """Render the count tally that closes a status report:
    `17 ok · 1 idle · 0 warn · 0 fail  ok`.

    `counts` maps state names to integers; the canonical order
    (`ok idle notes warn fail`) is rendered, defaulting missing states to 0
    and skipping any whose count is 0 *and* absent from `counts`. The trailing
    `verdict` word is coloured — green unless it names a failure — and defaults
    to `"failed"` when any `fail` count is non-zero, else `"ok"`."""
    order = ("ok", "idle", "notes", "warn", "fail")
    cells = [f"{counts.get(k, 0)} {k}" for k in order
             if k in counts or counts.get(k)]
    line = f" {dim(glyph('·'))} ".join(cells)
    if verdict is None:
        verdict = "failed" if counts.get("fail") else "ok"
    vcolor = red if verdict.lower().startswith("fail") else green
    return f"{line} {vcolor(verdict)}"


def hint_footer(pairs, *, cols: int = 2) -> str:
    """Render a dim grid of `<flag>  <description>` mode hints, `cols` per row —
    the footer a top-level command prints to advertise its other output modes
    (`--summary`, `--json`, `--verbose`). `pairs` is a sequence of
    `(flag, description)` tuples. Suppress it in `--quiet` and `--json`."""
    cells = [f"{flag} {desc}" for flag, desc in pairs]
    if not cells:
        return ""
    col_w = max(len(c) for c in cells) + 3
    lines = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        rendered = "".join(c.ljust(col_w) for c in row[:-1]) + row[-1]
        lines.append(dim(rendered))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Progress bar — single-line redraw via ANSI escapes. See
# `references/progress.md` for the full caller pattern (interleaving with
# error lines, --verbose interaction, non-TTY fallback).

def progress_supported() -> bool:
    """True when the bar can redraw in place.

    Tied to the same conditions as `USE_COLOR`: interactive TTY, no
    `NO_COLOR`, ANSI escapes enabled. Callers should branch on this
    before instantiating `ProgressBar`; in non-TTY contexts (piped
    output, log files, CI capture) the bar has nothing to redraw on
    and should be omitted.
    """
    return USE_COLOR


class ProgressBar:
    """Single-line progress bar that redraws in place via `\\r\\x1b[2K`.

    The bar occupies one terminal line; each call to `draw()` or
    `tick()` erases the line and re-renders. Interleave with per-item
    error lines by calling `clear()` first, printing the line, then
    `draw()`-ing again so the bar reappears below it.

    Minimal use::

        bar = ProgressBar(len(items))
        for item in items:
            bar.draw(item.name)        # show "working on X"
            do_work(item)
            bar.tick(item.name)        # advance counter, redraw
        bar.clear()

    With error interleaving::

        for item in items:
            bar.draw(item.name)
            try:
                do_work(item)
            except Exception as e:
                bar.clear()
                print(f"  {red('x')} {item.name}  {dim(str(e))}")
            bar.tick(item.name)
        bar.clear()

    The bar uses `cyan` for the rule, `bold` for the counter, and
    `dim` for the percentage and label so it sits visually below
    foregrounded content. `label` is truncated with a leading ellipsis
    to fit reasonable widths.
    """

    def __init__(self, total: int, width: int = 28) -> None:
        self.total = total
        self.width = width
        self.done = 0

    def _render(self, label: str) -> str:
        pct = self.done / self.total if self.total else 0.0
        filled = int(pct * self.width)
        bar = glyph("█") * filled + glyph("░") * (self.width - filled)
        if len(label) > 48:
            label = glyph("…") + label[-47:]
        return (
            f"  {cyan('[' + bar + ']')} "
            f"{bold(f'{self.done}/{self.total}')} "
            f"{dim(f'{pct * 100:>3.0f}%')}  {dim(label)}"
        )

    def draw(self, label: str = "") -> None:
        """Erase the current line and redraw the bar at its current count."""
        sys.stdout.write("\r\033[2K" + self._render(label))
        sys.stdout.flush()

    def tick(self, label: str = "") -> None:
        """Advance `done` by one and redraw."""
        self.done += 1
        self.draw(label)

    def clear(self) -> None:
        """Erase the bar's line. Call once at the end of the loop, and
        before each interleaved error/warning line."""
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
