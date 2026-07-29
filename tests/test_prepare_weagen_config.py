"""Unit tests for prepare_weagen_config helpers (R5 §8).

Targets the year math (compute_nr_years, generate-branch) and the stress-test
branch dict assembly (build_weagen_config). Both are import-clean after the R5
function extraction (commit 3) — no snakemake global, no heavy deps.
"""

import math
import os

import pytest
import yaml

from blueearth_cst.experiment.prepare_weagen_config import (
    build_weagen_config,
    compute_nr_years,
    read_yml,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
# The path rule 3.04 (Snakefile_climate_experiment:131) hands to
# prepare_weagen_config as ``default_config``. After the R06 config split this
# lives under config/templates/. This literal must track that Snakefile param.
DEFAULT_WEAGEN_CONFIG = os.path.join(
    REPO_ROOT, "config", "templates", "weathergen_config.yml"
)


@pytest.mark.parametrize(
    "middle_year, run_length, expected",
    [
        (2080, 20, math.ceil((2080 + 20 / 2) - 2010 + 2)),  # 82 (seed config)
        (2050, 30, math.ceil((2050 + 30 / 2) - 2010 + 2)),  # 57
        (2010, 0, math.ceil((2010 + 0) - 2010 + 2)),        # 2 (degenerate)
    ],
)
def test_compute_nr_years(middle_year, run_length, expected):
    """Year math spans 2010 -> horizon +/- run_length/2, +2 pad."""
    assert compute_nr_years(middle_year, run_length) == expected


def test_seed_year_math_value():
    """Pin the seed-config value explicitly (horizon 2080, run_length 20)."""
    assert compute_nr_years(2080, 20) == 82


def test_default_weagen_config_resolves_at_templates_path():
    """R06 config-split smoke (--dry-run-blind): the moved weathergen_config.yml
    must resolve at config/templates/. Snakefile_climate_experiment:131 passes
    this path as the ``default_config`` param; rule 3.04 reads it via read_yml.
    A green --dry-run / test_cli would NOT catch a broken move here because the
    param is not a declared input:."""
    assert os.path.isfile(DEFAULT_WEAGEN_CONFIG), (
        "config/templates/weathergen_config.yml missing — the R06 template move "
        "or the Snakefile_climate_experiment:131 default_config param is broken"
    )
    cfg = read_yml(DEFAULT_WEAGEN_CONFIG)
    assert "generateWeatherSeries" in cfg
    assert cfg["generateWeatherSeries"]["seed"] == 123


def test_build_weagen_config_generate_reads_moved_default(tmp_path):
    """Exercise the exact resolution path rule 3.04 uses: build_weagen_config's
    generate branch read_yml(default_config_path) against the moved template."""
    snake_cfg = {
        "workflows": {
            "climate_experiment": {"realizations_num": 2},
        }
    }
    snake_path = tmp_path / "snake.yml"
    snake_path.write_text(yaml.safe_dump(snake_cfg), encoding="utf-8")

    out = build_weagen_config(
        cftype="generate",
        snake_config_path=str(snake_path),
        output_path="out/",
        nc_file_prefix="rlz_1",
        default_config_path=DEFAULT_WEAGEN_CONFIG,
        middle_year=2080,
        sim_years=20,
    )
    # Seeded from the moved default template, then overridden by snake config.
    assert out["generateWeatherSeries"]["seed"] == 123
    assert out["generateWeatherSeries"]["realizations_num"] == 2
    assert out["generateWeatherSeries"]["sim.year.num"] == 82


def test_stress_test_branch_dict_shape(tmp_path):
    """The stress_test branch assembles the imposeClimateChanges block + copies
    the temp/precip sections verbatim from the snake config — no arithmetic."""
    snake_cfg = {
        "workflows": {
            "climate_experiment": {
                "stress_test": {
                    "temp": {"step_num": 1, "transient_change": True},
                    "precip": {"step_num": 2, "transient_change": False},
                }
            }
        }
    }
    snake_path = tmp_path / "snake.yml"
    snake_path.write_text(yaml.safe_dump(snake_cfg), encoding="utf-8")

    out = build_weagen_config(
        cftype="stress_test",
        snake_config_path=str(snake_path),
        output_path="out/",
        nc_file_prefix="rlz_1_cst",
        nc_file_suffix="3",
    )

    assert out["imposeClimateChanges"] == {
        "output.path": "out/",
        "nc.file.prefix": "rlz_1_cst",
        "nc.file.suffix": "3",
    }
    assert out["temp"] == {"step_num": 1, "transient_change": True}
    assert out["precip"] == {"step_num": 2, "transient_change": False}
