"""The path-byte provider inventory must not include WF4-only helpers."""

import sys
from pathlib import Path

from blueearth_cst.experiment.content_identity import repository_code_inventory
from blueearth_cst.experiment.generation_plan import (
    GENERATION_CODE_ENTRIES,
    GENERATION_INVENTORY_EXCLUDED,
)
from blueearth_cst.shared import run_log_core, snake_utils

# t2609251004: the whole generation inventory, pinned. A module entering or
# leaving it re-keys every scenario collection, so either is a decision to make
# here in review, never drift from a new import.
GENERATION_INVENTORY = {
    "blueearth_cst/__init__.py",
    "blueearth_cst/experiment/__init__.py",
    "blueearth_cst/experiment/content_identity.py",
    "blueearth_cst/experiment/forcing_descriptor.py",
    "blueearth_cst/experiment/generation_plan.py",
    "blueearth_cst/experiment/generation_sources.py",
    "blueearth_cst/experiment/prepare_cst_parameters.py",
    "blueearth_cst/experiment/prepare_weathergen_config.py",
    "blueearth_cst/experiment/scenario_collection_v2.py",
    "blueearth_cst/experiment/scenario_provider.py",
    "blueearth_cst/experiment/scenario_rows.py",
    "blueearth_cst/shared/__init__.py",
    "blueearth_cst/shared/wf3_science.py",
    "blueearth_cst/weathergen/generate_weather.R",
    "blueearth_cst/weathergen/global.R",
    "blueearth_cst/weathergen/impose_climate_change.R",
    "blueearth_cst/weathergen/read_member_grid.R",
    "blueearth_cst/weathergen/member_helpers.R",
}


def _generation_paths(repo):
    inventory = repository_code_inventory(
        repo, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    return {entry["path"] for entry in inventory}


def test_generation_inventory_is_pinned():
    assert _generation_paths(Path(__file__).resolve().parents[1]) == (
        GENERATION_INVENTORY
    )


def test_generation_inventory_ignores_settings_schema_and_logging_edits(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    for relative in GENERATION_INVENTORY | set(GENERATION_INVENTORY_EXCLUDED):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / relative).read_bytes())
    before = repository_code_inventory(
        tmp_path, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    for relative in GENERATION_INVENTORY_EXCLUDED:
        path = tmp_path / relative
        path.write_bytes(path.read_bytes() + b"\n# non-generation edit\n")
    after = repository_code_inventory(
        tmp_path, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
    )
    assert after == before
    science = tmp_path / "blueearth_cst/shared/wf3_science.py"
    science.write_bytes(science.read_bytes() + b"\n# generation edit\n")
    assert (
        repository_code_inventory(
            tmp_path, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
        )
        != before
    )


def test_wf3_provider_helper_closure_excludes_mixed_snake_utils():
    paths = _generation_paths(Path(__file__).resolve().parents[1])
    assert "blueearth_cst/shared/wf3_science.py" in paths
    assert "blueearth_cst/shared/snake_utils.py" not in paths


def test_wf3_helper_inventory_ignores_wf4_adapter_edit(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / "blueearth_cst" / "shared"
    target = tmp_path / "blueearth_cst" / "shared"
    target.mkdir(parents=True)
    for name in ("wf3_science.py", "run_log_core.py", "snake_utils.py"):
        (target / name).write_bytes((source / name).read_bytes())
    roots = [
        "blueearth_cst/shared/wf3_science.py",
        "blueearth_cst/shared/run_log_core.py",
    ]
    before = repository_code_inventory(tmp_path, roots)
    (target / "snake_utils.py").write_bytes(b"# WF4-only edit\n")
    assert repository_code_inventory(tmp_path, roots) == before


def test_legacy_runner_injects_relay_but_wf3_runner_uses_neutral_core(
    tmp_path, monkeypatch
):
    class Relay:
        def __init__(self):
            self.seen = []

        def tick(self):
            return None

        def feed(self, raw, *, stream):
            self.seen.append(raw)
            return raw

        def close(self):
            return None

    relay = Relay()
    monkeypatch.setattr(snake_utils, "_wflow_frame_relay", lambda: relay)
    command = [sys.executable, "-c", "print('helper boundary probe')"]
    legacy = tmp_path / "legacy.log"
    neutral = tmp_path / "neutral.log"
    assert snake_utils.run_and_tee(command, legacy) == 0
    assert run_log_core.run_and_tee(command, neutral) == 0
    assert relay.seen
    assert "helper boundary probe" in legacy.read_text(encoding="utf-8")
    assert "helper boundary probe" in neutral.read_text(encoding="utf-8")
