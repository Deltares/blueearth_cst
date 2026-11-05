"""Assemble the bounded HTML/PDF pilot without editing or executing live sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree


def select_text(text: str, section: dict) -> str:
    """Select one fragment, refusing missing, ambiguous or reversed boundaries."""
    if "start" not in section:
        return text.strip()
    start, stop = section["start"], section["stop"]
    for marker in (start, stop):
        if text.count(marker) != 1:
            raise ValueError(
                f"Expected one boundary {marker!r}; found {text.count(marker)}"
            )
    left, right = text.index(start), text.index(stop)
    if right <= left:
        raise ValueError(f"Reversed section boundaries: {start!r}, {stop!r}")
    if section.get("include_stop"):
        right += len(stop)
    return text[left:right].strip()


def diagram_fallback(asset: str) -> tuple[str, str]:
    """Derive styled SVG and all five explanations from the interactive source."""
    svgs = re.findall(r"<svg\b.*?</svg>", asset, re.DOTALL)
    objects = re.findall(r"const details = \{(.*?)\n  \};", asset, re.DOTALL)
    styles = re.findall(r"<style>(.*?)</style>", asset, re.DOTALL)
    if len(svgs) != 1 or len(objects) != 1 or len(styles) != 1:
        raise ValueError("Expected one SVG, details object and stylesheet in diagram")
    details = json.loads("{" + re.sub(r"\b(wf\d+):", r'"\1":', objects[0]) + "}")
    svg = svgs[0].replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    # Standalone SVG renderers do not inherit the host HTML's CSS variables.
    variables = dict(re.findall(r"--([\w-]+):\s*([^;]+);", styles[0]))
    css = styles[0]
    for name, value in variables.items():
        css = css.replace(f"var(--{name})", value)
    svg = svg.replace(">", f"><style>{css}</style>", 1)
    root = ElementTree.fromstring(svg)
    node_ids = [n.attrib["data-id"] for n in root.iter() if "data-id" in n.attrib]
    if list(details) != node_ids or len(node_ids) != 5:
        raise ValueError("Diagram details must cover all five SVG workflows in order")
    legend = " ".join((n.text or "") for n in root.iter() if n.tag.endswith("desc"))
    explanations = "\n\n".join(
        f"### {label}: {title}\n\n{body}" for label, title, body in details.values()
    )
    return svg, f"{legend}\n\n{explanations}"


def rewrite_links(text: str, source: Path, repo: Path, online: str) -> str:
    """Keep included navigation local and label references outside the sample."""

    def replace(match: re.Match) -> str:
        label, target = match.groups()
        if target == "assets/shape-of-a-run.html":
            return match.group(0)
        if target == "outputs.md" and source == Path("docs/guide/running.md"):
            return f"[{label}](#reading-an-indicator-set)"
        if urlsplit(target).scheme or target.startswith("#"):
            return match.group(0)
        path, separator, fragment = target.partition("#")
        resolved = (repo / source.parent / path).resolve()
        if not resolved.is_relative_to(repo.resolve()) or not resolved.is_file():
            raise ValueError(f"Unresolved repository link in {source}: {target}")
        relative = resolved.relative_to(repo.resolve()).as_posix()
        return f"[{label} (repository/online reference)]({online}{relative}{separator}{fragment})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, text)


def stage_pilot(repo: Path, output: Path) -> dict[str, str]:
    """Refresh derived sources after all required inputs and boundaries pass.

    Args:
        repo: Checkout containing canonical inputs and the pilot manifest.
        output: Dedicated generated directory beneath this checkout's .tmp.

    Returns:
        Byte hashes of every consumed source, configuration and fixture input.
    """
    repo, output = repo.resolve(), output.resolve()
    if not output.is_relative_to(repo / ".tmp"):
        raise ValueError("Pilot output must be inside this checkout's .tmp directory")
    manifest_path = Path("docs/publishing-pilot/manifest.json")
    manifest = json.loads((repo / manifest_path).read_text(encoding="utf-8"))
    compatibility = manifest["compatibility"]
    old = Path(compatibility["fixture"]) / compatibility["old"]
    canonical = Path(compatibility["fixture"]) / compatibility["canonical"]
    paths = {
        manifest_path,
        Path("docs/publishing-pilot/format.yml"),
        Path(manifest["asset"]),
        old,
        canonical,
        *(Path(s["source"]) for s in manifest["sections"]),
    }
    inputs = {path: (repo / path).read_bytes() for path in sorted(paths)}
    hashes = {
        p.as_posix(): hashlib.sha256(data).hexdigest() for p, data in inputs.items()
    }
    fragments = []
    for section in manifest["sections"]:
        source = Path(section["source"])
        text = inputs[source].decode("utf-8").replace("\r\n", "\n")
        fragment = select_text(text, section)
        if section.get("heading"):
            fragment = section["heading"] + "\n\n" + fragment
        fragment = re.sub(
            r"^# Default indicators$",
            "## Default indicators",
            fragment,
            flags=re.MULTILINE,
        )
        fragment = rewrite_links(fragment, source, repo, manifest["repository_url"])
        heading, rest = fragment.split("\n", 1)
        fragment = f"{heading} {{#{section['anchor']}}}\n{rest}"
        fragments.append(fragment + f"\n\nSource: `{source.as_posix()}`.")
    svg, fallback = diagram_fallback(inputs[Path(manifest["asset"])].decode("utf-8"))
    annex = fragments[-1]
    iframe = re.search(r"<iframe.*?</iframe>", annex, re.DOTALL)
    if iframe is None:
        raise ValueError("Expected workflow iframe in annex")
    html_only = '::: {.content-visible when-format="html"}\n\n'
    annex = annex.replace(iframe.group(), html_only + iframe.group() + "\n\n:::")
    interactive_link = (
        "[Open the interactive diagram](assets/shape-of-a-run.html) "
        "if your Markdown editor does not show it inline."
    )
    if annex.count(interactive_link) != 1:
        raise ValueError("Expected interactive-diagram link in annex")
    annex = annex.replace(interactive_link, html_only + interactive_link + "\n\n:::")
    annex += (
        "\n\n![Workflow execution order; dashed arrows are not dependencies.]"
        "(assets/shape-of-a-run.svg)\n\n" + fallback
    )
    fragments[-1] = "```{=typst}\n#pagebreak()\n```\n\n# Presentation annex\n\n" + annex
    config = inputs[Path("docs/publishing-pilot/format.yml")].decode("utf-8")
    header = (
        "---\ntitle: "
        + json.dumps(manifest["title"], ensure_ascii=False)
        + "\nsubtitle: "
        + json.dumps(manifest["source_revision"], ensure_ascii=False)
        + "\n"
        + config
        + "\n---\n\n"
    )
    navigation = (
        "[Configuration](#configuration) · "
        "[Indicator tables](#reading-an-indicator-set) · "
        "[Default conventions](#default-indicators) · "
        "[Workflow orientation](#the-shape-of-a-run)\n\n"
    )
    wrapper = (
        inputs[old]
        .decode("utf-8")
        .replace("indicator-requests.md#", "indicator-requests.html#")
    )
    generated = {
        "index.qmd": header.replace(
            "\n---\n\n",
            "\nformat-links:\n  - html\n  - format: typst\n    text: PDF\n"
            "    icon: file-pdf\n---\n\n",
        )
        + navigation
        + "\n\n".join(fragments)
        + "\n",
        "technical-manual/indicator-requests.qmd": header.replace(
            "\n---\n\n", "\nformat-links: false\n---\n\n"
        )
        + fragments[1]
        + "\n",
        "toolbox-reference/workflow-simulate-system.qmd": wrapper.replace(
            "../../technical-manual/", "../technical-manual/"
        ),
        "assets/shape-of-a-run.svg": svg,
        "assets/shape-of-a-run.html": inputs[Path(manifest["asset"])].decode("utf-8"),
        "source-hashes.json": json.dumps(hashes, indent=2) + "\n",
    }
    for name, text in generated.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    return hashes


def render_pilot(output: Path) -> None:
    """Render assembled Markdown to HTML and Typst PDF without execution."""
    quarto = shutil.which("quarto")
    if quarto is None:
        raise FileNotFoundError(
            "Quarto is required; tools are not installed automatically"
        )
    # Keep publisher caches beside the isolated artifacts, including refresh copies.
    environment = os.environ | {
        "XDG_CACHE_HOME": str(output / "cache"),
        "XDG_DATA_HOME": str(output / "data"),
    }
    for source, format_name in (
        ("index.qmd", "html"),
        ("index.qmd", "typst"),
        ("technical-manual/indicator-requests.qmd", "html"),
        ("toolbox-reference/workflow-simulate-system.qmd", "html"),
    ):
        subprocess.run(
            [quarto, "render", source, "--to", format_name, "--no-execute"],
            cwd=output,
            env=environment,
            check=True,
        )


def main() -> None:
    """Build the pilot from an explicit checkout and generated output root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    stage_pilot(args.source_root, args.output)
    if not args.stage_only:
        render_pilot(args.output.resolve())


if __name__ == "__main__":
    main()
