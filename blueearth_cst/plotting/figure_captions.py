"""Caption inventories for figure producers sharing an output directory."""

import json
from collections.abc import Mapping
from pathlib import Path

from blueearth_cst.plotting.figure_titles import write_figure_titles


def write_figure_captions(
    plots_root: str | Path, figures: Mapping[str, Mapping]
) -> None:
    """Merge explicit title/caption records and write both independent inventories."""
    root = Path(plots_root)
    path = root / "_engine/figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "figures": {}}
    )
    inventory["figures"].update(figures)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{record['caption']}"
            for key, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(root, inventory["figures"], merge=True)
