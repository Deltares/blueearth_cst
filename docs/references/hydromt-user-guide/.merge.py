#!/usr/bin/env python3
"""One-off: merge the per-page hydromt user-guide markdown into 6 section files.

Run from the repo root:

    python _raw/hydromt-user-guide/.merge.py

Reads the per-page `.md` files produced by the private-checkout-only helper
`dev/scripts/private/ingest_hydromt_user_guide.py`, strips each one's
frontmatter, and concatenates them into one file per upstream section under
`_raw/hydromt-user-guide/_merged/`. Each merged file gets a consolidated
frontmatter block listing every source URL, plus inline section dividers.

Idempotent: re-running overwrites every output file.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "_raw" / "hydromt-user-guide"
DST_ROOT = SRC_ROOT / "_merged"

# (slug, title, [relative source paths in upstream toctree order]).
SECTIONS = [
    (
        "overview", "Overview",
        [
            "intro.md",
            "overview/index.md",
            "overview/hydromt_intro.md",
            "overview/hydromt_cli.md",
            "overview/hydromt_python.md",
        ],
    ),
    (
        "models", "Working with models",
        [
            "models/model_overview.md",
            "models/model_build.md",
            "models/model_update.md",
            "models/model_workflow.md",
            "models/model_region.md",
            "models/model_components.md",
            "models/model_processes.md",
        ],
    ),
    (
        "data-catalog", "Data Catalog",
        [
            "data_catalog/data_overview.md",
            "data_catalog/data_existing_cat.md",
            "data_catalog/data_prepare_cat.md",
            "data_catalog/data_types.md",
            "data_catalog/data_conventions.md",
            "data_catalog/data_cloud_storage.md",
        ],
    ),
    (
        "supporting-functions", "Supporting functionalities",
        ["supporting_functions/methods_main.md"],
    ),
    (
        "migration-guide", "Migration guide",
        [
            "migration_guide/index.md",
            "migration_guide/data_catalog.md",
            "migration_guide/model_workflow.md",
            "migration_guide/python_updates.md",
        ],
    ),
    (
        "reference", "Terminology and FAQ",
        ["terminology.md", "faq.md"],
    ),
]

FM_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def parse_page(path: Path) -> tuple[dict, str]:
    """Return ({frontmatter}, body) for an ingested page file."""
    text = path.read_text(encoding="utf-8")
    m = FM_RE.match(text)
    if not m:
        return {}, text
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    body = text[m.end():]
    # Drop the leading `# Title` line — we'll synthesize a per-page H2 instead
    # so all pages live under the merged file's H1.
    body = re.sub(r"\A#\s+[^\n]+\n+", "", body)
    return fm, body.strip()


def merge_section(slug: str, title: str, rel_paths: list[str]) -> Path:
    pages: list[tuple[str, dict, str]] = []  # (rel, fm, body)
    for rel in rel_paths:
        src = SRC_ROOT / rel
        if not src.exists():
            sys.exit(f"missing input: {src}")
        fm, body = parse_page(src)
        pages.append((rel, fm, body))

    # Pull the shared upstream pin from the first page (every page has the same).
    first_fm = pages[0][1]
    sha = first_fm.get("upstream-commit", "")
    pulled = first_fm.get("pulled", "")

    fm_lines = [
        "---",
        f"title: HydroMT user guide — {title}",
        "ingest-source: hydromt-user-guide",
        f"upstream-repo: {first_fm.get('upstream-repo', 'Deltares/hydromt')}",
        f"upstream-commit: {sha}",
        f"pulled: {pulled}",
        f"section: {slug}",
        "doc-type: user-guide",
        "license: MIT",
        "sources:",
    ]
    for _, fm, _ in pages:
        fm_lines.append(f"  - {fm.get('source', '')}")
    fm_lines.append("---")
    fm_lines.append("")

    out_lines = ["\n".join(fm_lines)]
    out_lines.append(f"# HydroMT user guide — {title}\n")
    out_lines.append(
        f"_Merged from {len(pages)} upstream page(s) at commit `{sha[:8]}`, "
        f"pulled {pulled}. Page boundaries are preserved as `## ` headings._\n"
    )

    for rel, fm, body in pages:
        page_title = Path(rel).stem.replace("_", " ")
        src_url = fm.get("source", "")
        repo_url = fm.get("repo-source", "")
        out_lines.append("\n---\n")
        out_lines.append(f"## {page_title}\n")
        out_lines.append(
            f"> **Source:** [{rel}]({src_url}) · "
            f"[upstream `.rst`]({repo_url})\n"
        )
        out_lines.append(body)
        out_lines.append("")  # trailing newline

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    dst = DST_ROOT / f"{slug}.md"
    dst.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return dst


def main() -> None:
    if not SRC_ROOT.exists():
        sys.exit(f"ingested tree missing: {SRC_ROOT}")

    for slug, title, rels in SECTIONS:
        dst = merge_section(slug, title, rels)
        size_kb = dst.stat().st_size / 1024
        print(f"  {dst.relative_to(REPO_ROOT).as_posix()}  ({len(rels)} pages, {size_kb:.0f} KB)")

    total_in = sum(len(rels) for _, _, rels in SECTIONS)
    print(f"\nmerged {total_in} pages into {len(SECTIONS)} files under _raw/hydromt-user-guide/_merged/")


if __name__ == "__main__":
    main()
