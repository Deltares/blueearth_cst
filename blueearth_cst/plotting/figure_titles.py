"""External figure titles, independent of optional caption and title rendering."""

import json
from collections.abc import Mapping
from pathlib import Path


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
    path = Path(manifest_path) if manifest_path else root / "_engine/figure_titles.json"
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
