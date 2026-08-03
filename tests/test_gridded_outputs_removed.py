"""S8-08(c): the gridded outputs, and both spellings of their config key, are gone.

`save_grids` was renamed to `save_gridded` at 5e (OQ-12); S8-08(c) removed the
feature outright. The reason the option was never worth its branch:
`raw/{series_key}.nc` already IS the basin slice on the source grid, and for an
`Amon` source the monthly resample between it and `grids/series/` is the identity
— so the gridded series would have been a near-copy of a file every run writes.
`grids/change/` was the only genuinely new artifact and no rule ever declared it.

This file replaces the old-key-raises test with a **removal** test. The asymmetry
it pins is deliberate: a `true` raises, because silently handing back `false`
behaviour is the exact failure the 5e rename existed to prevent; a `false` warns
and continues, because it asks for precisely what the workflow now always does.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueearth_cst.projections.gridded_outputs import (
    RemovedGriddedOutputsError,
    validate_removed_gridded_options,
)

REPO = Path(__file__).resolve().parents[1]


def _dry_run(config_path):
    return subprocess.run(
        [
            sys.executable, "-m", "snakemake", "all", "-n",
            "-s", str(REPO / "Snakefile_climate_projections"),
            "--configfile", str(config_path),
        ],
        cwd=REPO, capture_output=True, text=True,
    )


@pytest.fixture
def seed_config():
    return yaml.safe_load(
        (REPO / "config/workflows/snake_config_dev_fast.yml").read_text(encoding="utf-8")
    )


def _write(tmp_path, cfg, name="cfg.yml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.parametrize("key", ["save_grids", "save_gridded"])
def test_a_true_gridded_key_is_rejected(key):
    """Asking for a removed output must fail loudly, not be ignored."""
    with pytest.raises(RemovedGriddedOutputsError, match=rf"`{key}: true`"):
        validate_removed_gridded_options({key: True})


@pytest.mark.parametrize("key", ["save_grids", "save_gridded"])
def test_a_false_gridded_key_returns_a_warning(key):
    """A stale false key agrees with current behaviour and is only warned."""
    warnings = validate_removed_gridded_options({key: False})

    assert warnings == [
        f"WARNING climate_projections: `{key}` is obsolete and ignored "
        "(S8-08c); the gridded outputs were removed. Delete the key."
    ]


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_snakefile_rejects_a_removed_gridded_key(tmp_path, seed_config):
    """The WF2 Snakefile applies the directly tested removal policy."""
    cfg = dict(seed_config)
    projection_cfg = dict(cfg["workflows"]["climate_projections"])
    projection_cfg.pop("save_grids", None)
    projection_cfg["save_gridded"] = True
    cfg["workflows"] = dict(cfg["workflows"], climate_projections=projection_cfg)

    result = _dry_run(_write(tmp_path, cfg))
    combined = result.stdout + result.stderr
    assert result.returncode != 0, "a true gridded key must stop the run"
    assert "`save_gridded: true`" in combined
    assert "gridded outputs were removed" in combined


def test_no_shipped_config_carries_a_gridded_key():
    for cfg_path in sorted((REPO / "config/workflows").glob("*.yml")):
        text = cfg_path.read_text(encoding="utf-8")
        for key in ("save_grids:", "save_gridded:"):
            assert key not in text, f"{cfg_path.name} still carries the dead key"


def test_no_rule_declares_a_grids_path():
    text = (REPO / "Snakefile_climate_projections").read_text(encoding="utf-8")
    assert '"/grids/' not in text and "'/grids/" not in text
