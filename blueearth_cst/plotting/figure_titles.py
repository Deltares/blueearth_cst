"""External figure titles, independent of optional caption and title rendering."""

import json
import os
from collections.abc import Mapping
from pathlib import Path


def figure_engine_dir(plots_dir: str | Path) -> Path:
    """Machine inventory directory for a figure directory.

    ``<unit>/plots[/<family>]`` maps to ``<unit>/_engine/plots[/<family>]``, so each
    output unit keeps all machine files under its single root ``_engine/``.
    """
    path = Path(plots_dir)
    for ancestor in (path, *path.parents):
        if ancestor.name == "plots":
            return ancestor.parent / "_engine" / "plots" / path.relative_to(ancestor)
    raise ValueError(f"{path} is not inside a plots/ directory")


def figure_inventory_paths(plots_dir: str | Path) -> list[str]:
    """Readable and machine title/caption inventories, relative to ``plots_dir``."""
    engine = Path(os.path.relpath(figure_engine_dir(plots_dir), plots_dir))
    return [
        "figure_titles.md",
        (engine / "figure_titles.json").as_posix(),
        "figure_captions.md",
        (engine / "figure_captions.json").as_posix(),
    ]


def write_figure_titles(
    plots_root: str | Path,
    figures: Mapping[str, Mapping],
    *,
    manifest_path: str | Path | None = None,
    merge: bool = False,
) -> None:
    """Mirror caption keys into readable and machine-readable title inventories.

    ``manifest_path`` follows the owning caption inventory's engine location.
    Keys retain that inventory's path convention. ``merge`` is for producers
    sharing a plot directory. Older retained caption records are migrated using
    their introductory sentence until their producer supplies an explicit title.
    This function never modifies images or captions.
    """
    root = Path(plots_root)
    path = (
        Path(manifest_path)
        if manifest_path
        else figure_engine_dir(root) / "figure_titles.json"
    )
    titles = {}
    if merge and path.is_file():
        titles.update(json.loads(path.read_text(encoding="utf-8"))["figures"])
    for key, record in figures.items():
        title = record.get("title")
        if title is None:
            title = record["caption"].split(". ", 1)[0].rstrip(".")
        if not isinstance(title, str) or not title.strip() or "\n" in title:
            raise ValueError(f"Figure {key!r} requires a nonempty single-line title")
        titles[key] = {"title": title}
    path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "figures": titles}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "figure_titles.md").write_text(
        "# Figure titles\n\n"
        + "\n\n".join(
            f"## {key}\n\n{record['title']}" for key, record in titles.items()
        )
        + "\n",
        encoding="utf-8",
    )
