"""Lift an inline ``<svg>`` figure out of an HTML page into a standalone file.

For diagrams hand-authored as inline SVG inside an Artifact page or any other
HTML: extracts one figure (or all of them), inlines the page's stylesheet with
its custom properties RESOLVED, and optionally rasterises to PNG.

**Why the style has to come along.** A hand-authored diagram is styled by CSS
classes on the page (``.d-box``, ``.d-edge``, …) whose colours are theme tokens
(``var(--accent)``). Cut the ``<svg>`` out on its own and none of that follows:
every ``<rect>`` falls back to the SVG default of solid black fill, and the
figure is a stack of black bars. So this resolves ``:root``'s token values into
literals, inlines the stylesheet inside the SVG, and paints an explicit ground —
which is what makes the file portable to Inkscape, Illustrator or a rasteriser.

A standalone file has no viewer to inherit a theme from, so it commits to one:
``--theme light`` (default) or ``--theme dark``, the latter resolved from the
page's ``:root[data-theme="dark"]`` block.

**PNG needs a browser, and that is deliberate.** The pixi env carries no SVG
rasteriser (no cairosvg, no rsvg-convert, no Inkscape) and adding one for a
figure export is not worth a dependency. Chrome or Edge is already installed on
any machine this repo is developed on, and ``--headless`` screenshots it.
Prefer the SVG for anything that gets typeset — it is vector and scales; the
PNG is for slides and issue threads.

Usage (from the repo root, inside pixi)::

    python dev/scripts/export_svg.py page.html                 # first figure -> .svg
    python dev/scripts/export_svg.py page.html --all --png     # every figure, + PNGs
    python dev/scripts/export_svg.py page.html -i 1 --theme dark --scale 3

Not part of a run: this is an authoring helper
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Where a Windows install puts a headless-capable browser. Checked in order,
#: after ``PATH`` — which is usually where it is NOT, since neither installer
#: adds itself.
BROWSER_CANDIDATES = (
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_browser() -> str | None:
    """A Chrome/Edge binary, from PATH or the usual install locations."""
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _strip_at_rules(css: str) -> str:
    """Drop ``@media``/``@supports`` blocks (one level of nesting).

    They cannot apply inside a standalone SVG — there is no viewer preference to
    match — and leaving them in means a dark-mode block silently overriding the
    theme this export just committed to.
    """
    return re.sub(r"@(?:media|supports)[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)


def token_map(css: str, *, dark: bool = False) -> dict[str, str]:
    """Custom properties declared on ``:root``, optionally with the dark overlay.

    The bare ``:root`` block is the base set. ``--theme dark`` layers
    ``:root[data-theme="dark"]`` on top, which is the block an explicit theme
    toggle uses and therefore the one written to be complete.
    """
    base: dict[str, str] = {}
    overlay: dict[str, str] = {}
    for match in re.finditer(r"(:root[^{]*)\{([^}]*)\}", css):
        selector, body = match.group(1).strip(), match.group(2)
        target = None
        if selector == ":root":
            target = base
        elif 'data-theme="dark"' in selector:
            target = overlay
        if target is None:
            continue
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body):
            target[name] = value.strip()

    tokens = dict(base)
    if dark:
        tokens.update(overlay)

    # A token may be defined in terms of another; a few passes settle it.
    for _ in range(5):
        changed = False
        for name, value in list(tokens.items()):
            resolved = _resolve_vars(value, tokens)
            if resolved != value:
                tokens[name] = resolved
                changed = True
        if not changed:
            break
    return tokens


def _resolve_vars(text: str, tokens: dict[str, str]) -> str:
    """Substitute ``var(--name)`` / ``var(--name, fallback)`` with its literal."""

    def sub(match: re.Match) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in tokens:
            return tokens[name]
        return fallback.strip() if fallback else match.group(0)

    return re.sub(r"var\(\s*(--[\w-]+)\s*(?:,([^()]*))?\)", sub, text)


def page_style(html: str, *, dark: bool) -> str:
    """The page's CSS, at-rules dropped and custom properties resolved."""
    blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.S)
    css = _strip_at_rules("\n".join(blocks))
    return _resolve_vars(css, token_map("\n".join(blocks), dark=dark))


def extract_svgs(html: str) -> list[str]:
    return re.findall(r"<svg\b.*?</svg>", html, flags=re.S)


def standalone_svg(svg: str, css: str, *, ground: str) -> tuple[str, int, int]:
    """One inline ``<svg>`` as a self-contained document. Returns (svg, w, h)."""
    viewbox = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not viewbox:
        raise ValueError('the <svg> has no `viewBox="0 0 W H"` to size it by')
    width, height = (round(float(v)) for v in viewbox.groups())

    svg = re.sub(
        r"^<svg\b[^>]*?(?=\s(?:viewBox|role|aria-label))",
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"',
        svg,
        count=1,
    )

    # The page style first, then the overrides this export needs: a real ground
    # (an artifact composites over the viewer's, a file has none) and an
    # intrinsic size that a page-level `width: 100%` must not take back.
    head = (
        f"<style>\n{css}\n"
        f"svg {{ width: {width}px; height: {height}px; max-width: none; }}\n"
        f"</style>\n"
        f'<rect width="{width}" height="{height}" fill="{ground}"/>\n'
    )
    if "<defs>" in svg:
        svg = svg.replace("<defs>", head + "<defs>", 1)
    else:
        svg = re.sub(r"(?<=>)", "\n" + head, svg, count=1)
    return svg, width, height


def rasterise(
    svg_path: Path, width: int, height: int, scale: int, browser: str
) -> Path:
    """Screenshot the SVG through a headless browser at ``scale``x."""
    wrapper = svg_path.with_name(f"{svg_path.stem}__wrap.html")
    wrapper.write_text(
        f'<body style="margin:0">'
        f'<img src="{svg_path.name}" '
        f'style="width:{width * scale}px;height:{height * scale}px">'
        f"</body>",
        encoding="utf-8",
    )
    png_path = svg_path.with_suffix(".png")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width * scale},{height * scale}",
            f"--screenshot={png_path}",
            wrapper.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    wrapper.unlink()
    return png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path, help="the HTML file holding the figure(s)")
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="which <svg> (default: 0, the first)"
    )
    parser.add_argument("--all", action="store_true", help="export every <svg> found")
    parser.add_argument(
        "--stem", help="output basename (default: the page's, suffixed when --all)"
    )
    parser.add_argument("--out-dir", type=Path, help="default: beside the page")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--png", action="store_true", help="also rasterise")
    parser.add_argument("--scale", type=int, default=2, help="PNG scale (default: 2)")
    args = parser.parse_args(argv)

    html = args.page.read_text(encoding="utf-8")
    svgs = extract_svgs(html)
    if not svgs:
        print(f"no <svg> element in {args.page}", file=sys.stderr)
        return 1

    indices = range(len(svgs)) if args.all else [args.index]
    if not args.all and not 0 <= args.index < len(svgs):
        print(
            f"--index {args.index} out of range: {args.page} has {len(svgs)} <svg>",
            file=sys.stderr,
        )
        return 1

    css = page_style(html, dark=args.theme == "dark")
    tokens = token_map(html, dark=args.theme == "dark")
    ground = tokens.get("--ground", "#0D1518" if args.theme == "dark" else "#FFFFFF")

    out_dir = args.out_dir or args.page.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.page.stem

    browser = None
    if args.png:
        browser = find_browser()
        if browser is None:
            print(
                "no Chrome/Edge found for --png; the .svg is still written",
                file=sys.stderr,
            )

    for position, index in enumerate(indices):
        svg, width, height = standalone_svg(svgs[index], css, ground=ground)
        name = f"{stem}_{index}" if (args.all and len(svgs) > 1) else stem
        svg_path = out_dir / f"{name}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        print(f"svg  {svg_path}  {width}x{height}  ({args.theme})")

        if browser is not None:
            png_path = rasterise(svg_path, width, height, args.scale, browser)
            size_kb = png_path.stat().st_size / 1024
            print(
                f"png  {png_path}  "
                f"{width * args.scale}x{height * args.scale}  {size_kb:.0f} KB"
            )
        del position

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
