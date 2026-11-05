"""Presentation files must not enter the weather provider's scientific identity."""

from pathlib import Path

from blueearth_cst.experiment.content_identity import repository_code_inventory
from blueearth_cst.experiment.generation_plan import (
    GENERATION_CODE_ENTRIES,
    GENERATION_INVENTORY_EXCLUDED,
)


def test_weather_presentation_edits_do_not_change_generation_inventory(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    inventory = repository_code_inventory(
        repo, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    paths = {entry["path"] for entry in inventory} | set(GENERATION_INVENTORY_EXCLUDED)
    presentation = (
        "config/plotting/style.yml",
        "config/plotting/colors.yml",
        "config/plotting/weathergenr.yml",
        "blueearth_cst/weathergen/plot_style.R",
        "blueearth_cst/plotting/panel_layout.py",
    )
    for relative in paths | set(presentation):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    before = repository_code_inventory(
        tmp_path, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    for relative in presentation:
        path = tmp_path / relative
        path.write_bytes(path.read_bytes() + b"\n# Changed presentation token.\n")
    after = repository_code_inventory(
        tmp_path, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    assert before == after
    assert "blueearth_cst/weathergen/generate_weather.R" in {
        entry["path"] for entry in before
    }
