"""Screenshot an HTML review page headlessly and report its console errors.

Drives the installed Chrome (or Edge with ``--channel msedge``) through
Playwright, so no browser download is needed. Use it to check a figure-
comparison page when the interactive browser tool is unavailable:

    pixi run python dev/scripts/capture_page.py http://127.0.0.1:<port>/comparison.html \
        --out .tmp/tasks/<id>/figure-review/captures

Writes one PNG per ``--width`` and exits non-zero when the page logs an error or
an image fails to load. ``--select SELECTOR=VALUE`` and ``--click SELECTOR``
(both repeatable, applied selects first, then clicks in order) put the page in a
state before capture, for example a figure with captions shown:

    ... --select "#choice=3" --click "#view-options summary" --click "#captions" \
        --click h1 --name chirps-captions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, nargs="+", default=[1440, 390])
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--channel", default="chrome")
    parser.add_argument(
        "--wait-ms", type=int, default=1500, help="settle time after load"
    )
    parser.add_argument(
        "--select", action="append", default=[], help="SELECTOR=VALUE to choose"
    )
    parser.add_argument(
        "--click", action="append", default=[], help="SELECTOR to click, in order"
    )
    parser.add_argument("--name", default="page", help="capture file-name prefix")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        for width in args.width:
            page = browser.new_page(viewport={"width": width, "height": args.height})
            # HTTP failures are reported by URL below; the browser's own
            # "Failed to load resource" console line names no URL.
            page.on(
                "console",
                lambda m: (
                    problems.append(f"console {m.type}: {m.text}")
                    if m.type == "error"
                    and not m.text.startswith("Failed to load resource")
                    else None
                ),
            )
            # The browser requests a favicon nobody declared; its 404 is noise.
            page.on(
                "response",
                lambda r: (
                    problems.append(f"HTTP {r.status}: {r.url}")
                    if r.status >= 400 and not r.url.endswith("/favicon.ico")
                    else None
                ),
            )
            page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
            page.on(
                "requestfailed", lambda r: problems.append(f"request failed: {r.url}")
            )
            page.goto(args.url, wait_until="networkidle")
            page.wait_for_timeout(args.wait_ms)
            if args.select or args.click:
                for choice in args.select:
                    selector, _, value = choice.partition("=")
                    page.select_option(selector, value)
                for selector in args.click:
                    page.click(selector)
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(args.wait_ms)
            broken = page.evaluate(
                "[...document.images].filter(i => i.src && !i.naturalWidth)"
                ".map(i => i.src)"
            )
            problems += [f"image not loaded: {src}" for src in broken]
            target = args.out / f"{args.name}-{width}px.png"
            page.screenshot(path=target, full_page=True)
            print(target)
            page.close()
        browser.close()
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
