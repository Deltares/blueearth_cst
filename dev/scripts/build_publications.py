"""Stage complete manuals and chapter routes from canonical sources, without execution.

CLI: python dev/scripts/build_publications.py --source-root CHECKOUT --output DIR
[--stage-only]. DIR must be beneath the executing checkout's .tmp directory;
CHECKOUT may be a different, read-only documentation lane. No tools are installed.
The schema-version-1 docs/publication/manifest.json owns chapter order and titles.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import yaml

try:
    from dev.scripts.build_docs_pilot import diagram_fallback
except ModuleNotFoundError:  # Direct execution from dev/scripts, as in the site hook.
    from build_docs_pilot import diagram_fallback


def _prose(text: str, transform: Callable[[str], str]) -> str:
    """Transform prose while preserving fenced examples and inline code exactly."""
    parts = re.split(
        r"(^ {0,3}(?:`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?:`{3,}|~{3,})[^\n]*(?:\n|$)|`+[^`\n]+`+)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return "".join(part if i % 2 else transform(part) for i, part in enumerate(parts))


def _links(text: str, replace: Callable[[str], str]) -> str:
    """Rewrite inline/reference Markdown links and literal HTML href/src values."""

    def prose(part: str) -> str:
        protected = [(m.start(), m.end()) for m in re.finditer(r"`+[^`\n]+`+", part)]

        def substitute(match: re.Match) -> str:
            if any(left <= match.start() < right for left, right in protected):
                return match[0]
            return match[1] + replace(match[2].strip("<>")) + match[3]

        patterns = (
            r"(!?\[[^\]]*\]\(\s*)(<[^>\n]+>|[^\s)]+(?:\([^\n)]*\)[^\s)]*)?)(\s*(?:[\"'][^\n]*?[\"']\s*)?\))",
            r"(^ {0,3}\[[^\]\n]+\]:\s*)(<[^>\n]+>|\S+)()",
            r"(\b(?:href|src)=[\"'])([^\"']+)([\"'])",
        )
        for pattern in patterns:
            protected = [
                (m.start(), m.end()) for m in re.finditer(r"`+[^`\n]+`+", part)
            ]
            part = re.sub(
                pattern,
                substitute,
                part,
                flags=re.MULTILINE,
            )
        return part

    parts = re.split(
        r"(^ {0,3}(?:`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?:`{3,}|~{3,})[^\n]*(?:\n|$))",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return "".join(p if i % 2 else prose(p) for i, p in enumerate(parts))


def _split_source(text: str) -> tuple[dict, str]:
    """Separate front matter from the complete source body."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end < 0:
            raise ValueError("Unterminated source front matter")
        return yaml.safe_load(text[4:end]) or {}, text[end + 5 :].lstrip("\n")
    return {}, text


def _identifier(heading: str) -> str:
    """Match Pandoc's auto identifiers for the supported Markdown headings."""
    heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"<[^>]*>", "", heading)
    heading = re.sub(r"[^\w\s.-]", "", heading.lower()).replace(" ", "-")
    heading = re.sub(r"^[^a-zA-Z\u0080-\uffff]*", "", heading)
    return heading or "section"


def _headings(body: str, namespace: str | None = None) -> tuple[str, dict[str, str]]:
    """Make heading IDs explicit and optionally namespace them for a volume."""
    seen: set[str] = set()
    anchors: dict[str, str] = {}

    def prose(part: str) -> str:
        def heading(match: re.Match) -> str:
            level, title = match.groups()
            explicit = re.search(r"\s*\{#([^}\s]+)[^}]*\}\s*$", title)
            if explicit:
                anchor = explicit[1]
                title = title[: explicit.start()]
            else:
                base = _identifier(title.rstrip("# "))
                anchor, index = base, 0
                while anchor in seen:
                    index += 1
                    anchor = f"{base}-{index}"
            if anchor in seen:
                raise ValueError(f"Duplicate explicit heading ID: {anchor}")
            seen.add(anchor)
            target = f"{namespace}--{anchor}" if namespace else anchor
            anchors[anchor] = target
            return f"{level} {title.rstrip('# ')} {{#{target}}}"

        return re.sub(
            r"^(#{1,6})[ \t]+([^\n]+?)[ \t]*$", heading, part, flags=re.MULTILINE
        )

    # Inline code is meaningful in headings; skip fences, but keep heading lines whole.
    parts = re.split(
        r"(^`{3,}[^\n]*\n.*?^`{3,}[^\n]*(?:\n|$))", body, flags=re.MULTILINE | re.DOTALL
    )
    return "".join(p if i % 2 else prose(p) for i, p in enumerate(parts)), anchors


def prepare_publications(source_root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Prepare all outputs in memory; fail before writes for invalid/missing inputs.

    Returns generated relative paths/bytes and SHA-256 hashes of consumed inputs.
    Chapter sources, complete appendices, wrapper routes and link targets are read
    only. Repository-only references are visibly labelled rather than included.
    """
    repo = source_root.resolve()
    inputs: dict[str, bytes] = {}

    def read(relative: str) -> bytes:
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise FileNotFoundError(
                f"Missing or unconfined publication source: {relative}"
            )
        key = path.relative_to(repo).as_posix()
        if key not in inputs:
            inputs[key] = path.read_bytes()
        return inputs[key]

    manifest = json.loads(read("docs/publication/manifest.json"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected publication schema_version 1")
    config = yaml.safe_load(read(manifest["format"]))
    appendix = yaml.safe_load(read(manifest["appendix_manifest"]))
    for key, expected in {
        "schema_version": 1,
        "path_base": "repository-root",
        "include_mode": "complete-source",
        "link_base": "source-directory",
    }.items():
        if appendix.get(key) != expected:
            raise ValueError(f"Unsupported appendix {key}: {appendix.get(key)!r}")
    publications = manifest["publications"]
    if not publications or len({p["id"] for p in publications}) != len(publications):
        raise ValueError("Publications must have unique, nonempty IDs")
    sources: dict[str, dict] = {}
    volumes: dict[str, list[str]] = {}
    for publication in publications:
        name = publication["id"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"Invalid publication ID: {name!r}")
        if not publication["chapters"]:
            raise ValueError(f"No chapters in {name}")
        chapters = [
            (p, Path(p).relative_to("docs").with_suffix(".qmd").as_posix(), None)
            for p in publication["chapters"]
        ]
        if publication["include_appendices"]:
            chapters += [
                (
                    s["path"],
                    f"technical-manual/appendices/{s['anchor']}.qmd",
                    s["anchor"],
                )
                for s in appendix["sources"]
            ]
        volumes[name] = []
        for path, route, anchor in chapters:
            if path in sources or route in {s["route"] for s in sources.values()}:
                raise ValueError(f"Repeated publication source/route: {path}")
            if (
                not route.startswith(("guide/", "technical-manual/"))
                or ".." in Path(route).parts
            ):
                raise ValueError(f"Unsupported chapter route: {route}")
            metadata, body = _split_source(
                read(path).decode("utf-8").replace("\r\n", "\n")
            )
            title = metadata.get("title")
            if not title:
                first = re.match(r"# ([^\n]+)\n", body)
                if first is None:
                    raise ValueError(f"Missing chapter title: {path}")
                title = first[1]
            namespace = anchor or route[:-4].replace("/", "-")
            standalone, ids = _headings(body)
            combined, volume_ids = _headings(body, namespace)
            if combined.startswith("# "):
                volume_ids[next(iter(volume_ids))] = namespace
            sources[path] = dict(
                route=route,
                volume=name,
                title=title,
                namespace=namespace,
                metadata=metadata,
                body=standalone,
                combined=combined,
                ids=ids,
                volume_ids=volume_ids,
            )
            volumes[name].append(path)

    asset = manifest["asset"]
    generated: dict[str, bytes] = {
        "publication-links.lua": Path(__file__)
        .with_name("publication_links.lua")
        .read_bytes(),
    }
    if Path(asset).suffix == ".html":
        svg, fallback = diagram_fallback(read(asset).decode("utf-8"))
        generated.update(
            {
                "guide/assets/shape-of-a-run.html": inputs[asset],
                "guide/assets/shape-of-a-run.svg": svg.encode("utf-8"),
                "manuals/assets/shape-of-a-run.svg": svg.encode("utf-8"),
            }
        )
    else:
        read(asset)
        fallback = ""

    def target_url(
        target: str, source: str, destination: str, volume: str | None
    ) -> str:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("/"):
            return target
        resolved = (repo / Path(source).parent / unquote(parsed.path)).resolve()
        relative = (
            resolved.relative_to(repo).as_posix()
            if resolved.is_relative_to(repo)
            else ""
        )
        if not parsed.path:
            relative = source
        included = sources.get(relative)
        if included:
            fragment = unquote(parsed.fragment)
            if (
                fragment
                and fragment not in included["ids"]
                and fragment != included["namespace"]
            ):
                raise ValueError(f"Missing anchor in {source}: {target}")
            if volume:
                anchor = included["volume_ids"].get(fragment, included["namespace"])
                prefix = (
                    "" if included["volume"] == volume else f"{included['volume']}.html"
                )
                return f"{prefix}#{anchor}"
            route = included["route"][:-4] + ".html"
            prefix = (
                ""
                if relative == source and not parsed.path
                else posixpath.relpath(route, posixpath.dirname(destination))
            )
            return prefix + (f"#{parsed.fragment}" if fragment else "")
        if relative.startswith(
            "docs/technical-manual/assets/"
        ) and resolved.suffix.lower() in {
            ".png",
            ".svg",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            data = read(relative)
            tail = relative.removeprefix("docs/technical-manual/assets/")
            prefix = (
                "manuals/assets/technical-manual"
                if volume
                else "technical-manual/assets"
            )
            route = f"{prefix}/{tail}"
            generated[route] = data
            return posixpath.relpath(route, posixpath.dirname(destination))
        if relative == asset and Path(asset).suffix == ".html":
            if volume:
                return "data:text/html;base64," + base64.b64encode(
                    inputs[asset]
                ).decode("ascii")
            route = "guide/assets/shape-of-a-run.html"
            return posixpath.relpath(route, posixpath.dirname(destination))
        if relative.startswith("docs/guide/assets/") and resolved.suffix.lower() in {
            ".png",
            ".svg",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            data = read(relative)
            tail = relative.removeprefix("docs/guide/")
            route = ("manuals/" if volume else "guide/") + tail
            generated[route] = data
            return posixpath.relpath(route, posixpath.dirname(destination))
        read(relative)
        # Site-owned wrappers/scientific pages stay local in the website only.
        if not volume and relative.startswith("docs/site/"):
            route = str(Path(relative[10:]).with_suffix(".html")).replace("\\", "/")
            return posixpath.relpath(route, posixpath.dirname(destination)) + (
                f"#{parsed.fragment}" if parsed.fragment else ""
            )
        return (
            manifest["repository_url"].rstrip("/")
            + "/"
            + quote(relative, safe="/")
            + (f"#{parsed.fragment}" if parsed.fragment else "")
        )

    def rewrite(body: str, source: str, destination: str, volume: str | None) -> str:
        text = _links(body, lambda t: target_url(t, source, destination, volume))
        # A visible qualifier keeps repository evidence distinct from offline contents.
        online = re.escape(manifest["repository_url"].rstrip("/"))
        return _prose(
            text,
            lambda p: re.sub(
                rf"(?<!!)\[([^]\n]+)\]\(({online}/[^)]+)\)",
                r"[\1 (repository/online reference)](\2)",
                p,
            ),
        )

    def presentation(body: str, destination: str) -> str:
        if "<iframe" not in body:
            return body
        visible = '::: {.content-visible when-format="html"}\n\n'
        body = re.sub(
            r"<iframe\b.*?</iframe>",
            lambda m: visible + m[0] + "\n\n:::",
            body,
            flags=re.DOTALL,
        )
        body = re.sub(
            r"^\[Open the interactive diagram\][^\n]+",
            lambda m: visible + m[0] + "\n\n:::",
            body,
            flags=re.MULTILINE,
        )
        image = "assets/shape-of-a-run.svg"
        return (
            body
            + f"\n\n![Workflow execution order; dashed arrows are not dependencies.]({image})\n\n"
            + fallback
            + "\n"
        )

    def document(metadata: dict, body: str) -> bytes:
        # Canonical per-page format: html must not override the volume formats.
        metadata = metadata | {"engine": "markdown", "execute": {"enabled": False}}
        return (
            "---\n"
            + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
            + "---\n\n"
            + body
            + "\n"
        ).encode("utf-8")

    for path, chapter in sources.items():
        route = chapter["route"]
        body = rewrite(chapter["body"], path, route, None)
        generated[route] = document(
            chapter["metadata"] | {"title": chapter["title"], "format": "html"},
            presentation(body, route),
        )
    for publication in publications:
        name = publication["id"]
        destination = f"manuals/{name}.qmd"
        parts = []
        for path in volumes[name]:
            chapter = sources[path]
            body = chapter["combined"]
            # README H1 is the chapter title; preserve it once, with its namespaced ID.
            if body.startswith("# "):
                body = re.sub(r"^# [^\n]+\n", "", body, count=1).lstrip("\n")
            body = rewrite(body, path, destination, name)
            parts.append(
                f"# {chapter['title']} {{#{chapter['namespace']}}}\n\n"
                + presentation(body, destination)
            )
        metadata = config | {
            "title": publication["title"],
            "format-links": [
                "html",
                {"format": "typst", "text": "PDF", "icon": "file-pdf"},
            ],
            "filters": ["../publication-links.lua"],
            "publication-siblings": list(volumes),
        }
        if manifest.get("source_revision"):
            metadata["subtitle"] = manifest["source_revision"]
        generated[destination] = document(metadata, "\n\n".join(parts))

    # Precompute exact rewrites for tracked site pages. The Lua filter consumes
    # these during a website render; staging never overwrites authored wrappers.
    site_links = {}
    site = repo / "docs/site"
    for folder in (
        site,
        site / "setup",
        site / "toolbox-reference",
        site / "scientific-approach",
    ):
        for file in sorted(folder.glob("*")):
            if not file.is_file() or file.suffix not in {".md", ".qmd"}:
                continue
            path = file.relative_to(repo).as_posix()
            destination = file.relative_to(site).with_suffix(".qmd").as_posix()
            text = read(path).decode("utf-8").replace("\r\n", "\n")
            replacements = {}

            def record(target: str) -> str:
                # Already-staged links need only their source extension changed.
                staged = posixpath.normpath(
                    posixpath.join(posixpath.dirname(destination), target.split("#")[0])
                )
                if staged in {
                    f"manuals/{name}.{ext}"
                    for name in volumes
                    for ext in ("html", "pdf")
                }:
                    new = target
                elif any(
                    staged == s["route"]
                    or staged
                    == str(Path(s["route"]).with_suffix(".md")).replace("\\", "/")
                    for s in sources.values()
                ):
                    new = str(Path(staged).with_suffix(".html")).replace("\\", "/")
                    new = posixpath.relpath(new, posixpath.dirname(destination)) + (
                        "#" + target.split("#", 1)[1] if "#" in target else ""
                    )
                elif (
                    target.startswith("#")
                    or urlsplit(target).scheme
                    or target.startswith("/")
                ):
                    new = target
                else:
                    new = target_url(target, path, destination, None)
                replacements[target] = new
                return new

            transformed = _links(text, record)
            site_links[file.relative_to(site).as_posix()] = replacements
            generated[destination] = transformed.encode("utf-8")
    generated["technical-manual/publication-links.json"] = (
        json.dumps(site_links, indent=2, sort_keys=True) + "\n"
    ).encode()
    hashes = {p: hashlib.sha256(data).hexdigest() for p, data in sorted(inputs.items())}
    generated["technical-manual/source-hashes.json"] = (
        json.dumps(hashes, indent=2) + "\n"
    ).encode()
    return generated, hashes


def stage_publications(source_root: Path, output: Path) -> dict[str, str]:
    """Refresh isolated derived sources; return hashes of every consumed input.

    Args:
        source_root: Explicit read-only checkout containing the publication manifest.
        output: Dedicated directory under the executing checkout's .tmp directory.
    """
    output = output.resolve()
    scratch = Path(__file__).resolve().parents[2] / ".tmp"
    if not output.is_relative_to(scratch) or output == scratch:
        raise ValueError(
            "Publication output must be beneath the executing checkout's .tmp"
        )
    generated, hashes = prepare_publications(source_root)
    check_stale_outputs(output, generated)
    for relative, data in generated.items():
        target = output / relative
        if not target.resolve().is_relative_to(output):
            raise ValueError(f"Unconfined generated target: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return hashes


def check_stale_outputs(output: Path, generated: dict[str, bytes]) -> None:
    """Refuse undeclared chapter/volume products without deleting existing files.

    The publication trees are generated-only. After manifest removals, the owner
    must explicitly remove named stale products before the next refresh/build.
    Assets and caches are not traversed for cleanup or inferred to be disposable.
    """
    allowed = set(generated)
    for name in generated:
        if name.endswith(".qmd"):
            allowed.add(str(Path(name).with_suffix(".html")).replace("\\", "/"))
            if name.startswith("manuals/"):
                allowed.add(str(Path(name).with_suffix(".pdf")).replace("\\", "/"))
    for folder in ("guide", "technical-manual", "manuals"):
        for path in (output / folder).rglob("*"):
            if path.suffix not in {".qmd", ".html", ".pdf"} or not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            if relative not in allowed:
                raise ValueError(
                    f"Stale generated publication file: {relative}; remove explicitly before rebuilding"
                )


def render_publications(output: Path) -> None:
    """Render staged volumes to HTML and Typst PDF, with execution disabled."""
    quarto = shutil.which("quarto")
    if quarto is None:
        raise FileNotFoundError(
            "Quarto is required; tools are not installed automatically"
        )
    env = os.environ | {
        "XDG_CACHE_HOME": str(output / "cache"),
        "XDG_DATA_HOME": str(output / "data"),
    }
    for source in sorted((output / "manuals").glob("*.qmd")):
        for format_name in ("html", "typst"):
            subprocess.run(
                [quarto, "render", source.name, "--to", format_name, "--no-execute"],
                cwd=source.parent,
                env=env,
                check=True,
            )


def main() -> None:
    """Build complete manuals from an explicit read-only source checkout."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    stage_publications(args.source_root, args.output)
    if not args.stage_only:
        render_publications(args.output.resolve())


if __name__ == "__main__":
    main()
