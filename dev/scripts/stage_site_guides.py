"""Stage canonical Markdown guides for Quarto's site-local render tree.

Called by the Quarto pre-render hook for both render and preview. Canonical
links resolve from docs/guide; staged links resolve from docs/site/guide.
"""

from pathlib import Path
from shutil import copyfile

GUIDES = (
    "install",
    "quick-start",
    "new-case",
    "running",
    "outputs",
    "troubleshooting",
    "notebooks",
)


def stage_guides(docs_dir: Path) -> None:
    """Refresh the site copies and their self-contained diagram asset."""
    if (docs_dir / "publication/manifest.json").is_file():
        try:
            from dev.scripts.build_publications import (
                check_stale_outputs,
                prepare_publications,
            )
        except ModuleNotFoundError:
            from build_publications import check_stale_outputs, prepare_publications

        generated, _ = prepare_publications(docs_dir.parent)
        check_stale_outputs(docs_dir / "site", generated)
        # Authored setup/science/compatibility pages are rebased by the render
        # filter. Only canonical chapters and publisher metadata are staged here.
        for relative, data in generated.items():
            if not relative.startswith(("guide/", "technical-manual/")):
                continue
            target = docs_dir / "site" / relative
            if not target.resolve().is_relative_to((docs_dir / "site").resolve()):
                raise ValueError(f"Unconfined generated site target: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return
    source_dir = docs_dir / "guide"
    target_dir = docs_dir / "site" / "guide"
    # Read every required source before writing, so missing inputs fail the build.
    sources = {
        name: (source_dir / f"{name}.md").read_text(encoding="utf-8") for name in GUIDES
    }
    asset = Path("assets/shape-of-a-run.html")
    if not (source_dir / asset).is_file():
        raise FileNotFoundError(source_dir / asset)
    (target_dir / "assets").mkdir(parents=True, exist_ok=True)
    for name, content in sources.items():
        content = content.replace("](../site/", "](../")
        (target_dir / f"{name}.md").write_text(content, encoding="utf-8")
    copyfile(source_dir / asset, target_dir / asset)


if __name__ == "__main__":
    stage_guides(Path(__file__).resolve().parents[2] / "docs")
