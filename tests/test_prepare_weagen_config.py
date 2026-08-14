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
# The path rule 3.04 (run_stress_test.smk:131) hands to
# prepare_weagen_config as ``default_config``. It lives under config/defaults/:
# the 2026-08-11 split moved the three rule-read configs out of
# config/templates/, which now holds only files you copy. This literal must
# track that Snakefile param.
DEFAULT_WEAGEN_CONFIG = os.path.join(
    REPO_ROOT, "config", "defaults", "weathergen_config.yml"
)


@pytest.mark.parametrize(
    "middle_year, run_length, expected",
    [
        (2080, 20, math.ceil((2080 + 20 / 2) - 2010 + 2)),  # 82 (seed config)
        (2050, 30, math.ceil((2050 + 30 / 2) - 2010 + 2)),  # 57
        (2010, 0, math.ceil((2010 + 0) - 2010 + 2)),  # 2 (degenerate)
    ],
)
def test_compute_nr_years(middle_year, run_length, expected):
    """Year math spans 2010 -> horizon +/- run_length/2, +2 pad."""
    assert compute_nr_years(middle_year, run_length) == expected


def test_seed_year_math_value():
    """Pin the seed-config value explicitly (horizon 2080, run_length 20)."""
    assert compute_nr_years(2080, 20) == 82


def test_default_weagen_config_resolves_at_defaults_path():
    """Config-split smoke (--dry-run-blind): weathergen_config.yml must resolve
    at config/defaults/. run_stress_test.smk:131 passes this path as
    the ``default_config`` param; rule 3.04 reads it via read_yml. A green
    --dry-run / test_cli would NOT catch a broken move here because the param is
    not a declared input:."""
    assert os.path.isfile(DEFAULT_WEAGEN_CONFIG), (
        "config/defaults/weathergen_config.yml missing — the 2026-08-11 split "
        "or the run_stress_test.smk:131 default_config param is broken"
    )
    cfg = read_yml(DEFAULT_WEAGEN_CONFIG)
    assert "generate_weather" in cfg
    assert cfg["generate_weather"]["warm_var"] == "precip"
    # The seed is NOT in the template: its default lives in
    # config/advanced_settings.yml (`defaults.seed`) and rule 3.10 injects
    # the resolved value. A `seed:` here would be a second default.
    assert "seed" not in cfg["generate_weather"]


def test_build_weagen_config_generate_reads_moved_default(tmp_path):
    """Exercise the exact resolution path rule 3.04 uses: build_weagen_config's
    generate branch read_yml(default_config_path) against the moved template."""
    out = build_weagen_config(**_generate_kwargs(tmp_path))
    # Seeded from the moved default template, then overridden by snake config.
    assert out["generate_weather"]["seed"] == 123  # injected, not templated
    assert out["generate_weather"]["n_realizations"] == 2
    assert out["generate_weather"]["n_years"] == 82


def _generate_kwargs(tmp_path, stress_test=None):
    """The call rule 3.04 makes, with a writable snake config beside it."""
    if stress_test is None:
        stress_test = {
            "temp": {"step_num": 1, "transient_change": True},
            "precip": {"step_num": 2, "transient_change": False},
        }
    snake_cfg = {
        "workflows": {
            "climate_experiment": {
                "realizations_num": 2,
                "stress_test": stress_test,
            }
        }
    }
    snake_path = tmp_path / "snake.yml"
    snake_path.write_text(yaml.safe_dump(snake_cfg), encoding="utf-8")
    return dict(
        snake_config_path=str(snake_path),
        output_path="out/",
        nc_file_prefix="rlz_1",
        default_config_path=DEFAULT_WEAGEN_CONFIG,
        middle_year=2080,
        sim_years=20,
        seed=123,
        water_year_start="Jan",
        dry_spell_factor=[1.0] * 12,
        wet_spell_factor=[1.0] * 12,
    )


def test_transient_flags_reach_the_one_shared_config(tmp_path):
    """C29: the flags impose_climate_change.R reads now live in THIS file.

    They used to arrive in a per-member weathergen_config_rlz_<n>_cst_<m>.yml
    that rule 3.05 wrote once per member. With 3.05 gone the shared config is
    their only carrier, so an omission here is a silent behaviour change in the
    perturbation step -- which is what `validate_wg3` now pins.
    """
    out = build_weagen_config(**_generate_kwargs(tmp_path))
    assert out["temp"] == {"transient_change": True}
    assert out["precip"] == {"transient_change": False}


def test_only_the_flags_are_copied_not_the_perturbation_ranges(tmp_path):
    """F6: the retired per-member file copied in the whole stress_test blocks.

    It carried `step_num` and the monthly min/max ranges, none of which the R
    read -- so anyone opening it to see what a run did read plausible
    perturbation ranges that had no part in it. The real values come from
    st_<m>.csv. Do not reintroduce them here.
    """
    out = build_weagen_config(
        **_generate_kwargs(
            tmp_path,
            stress_test={
                "temp": {
                    "step_num": 1,
                    "transient_change": True,
                    "mean": {"min": [0.0] * 12, "max": [3.0] * 12},
                },
                "precip": {"step_num": 2, "transient_change": True},
            },
        )
    )
    assert set(out["temp"]) == {"transient_change"}
    assert "mean" not in out["temp"]
    assert "step_num" not in out["temp"]


@pytest.mark.parametrize("variable", ["temp", "precip"])
def test_missing_transient_flag_refuses_and_names_the_key(tmp_path, variable):
    """No silent default: it decides whether a perturbation ramps or steps."""
    stress_test = {
        "temp": {"transient_change": True},
        "precip": {"transient_change": True},
    }
    del stress_test[variable]["transient_change"]
    with pytest.raises(ValueError, match=rf"stress_test\.{variable}\.transient_change"):
        build_weagen_config(**_generate_kwargs(tmp_path, stress_test=stress_test))


# ---------------------------------------------------------------------------
# F7 (regression) and C34 — R11 P2 commit 4
# ---------------------------------------------------------------------------


def test_f7_the_template_is_a_declared_input_of_rule_3_10():
    """F7: the weathergen template must be an `input:`, not a params-only read.

    It landed with CR-5 (`9260668`, 2026-08-05); this pins it so it cannot slide
    back. The failure it guards is silent: edit the template and rule 3.10 does
    NOT re-run, its generated config stays stale, and 3.11 keeps generating
    realizations from superseded settings -- propagating quietly precisely
    BECAUSE the generated config is properly declared, so every downstream
    timestamp stays consistent.

    Asserted against the rule's own `input:` block rather than the whole file,
    so a `params:` mention alone cannot satisfy it -- that is the exact state
    F7 described.
    """
    import re
    from pathlib import Path

    snakefile = (
        Path(__file__).resolve().parents[1] / "run_stress_test.smk"
    ).read_text(encoding="utf-8")
    rule = re.search(
        r"rule prepare_weathergen_config:.*?\n    output:", snakefile, re.S
    )
    assert rule, "rule prepare_weathergen_config not found"
    inputs = re.search(r"\n    input:(.*)", rule.group(0), re.S).group(1)
    assert "config/defaults/weathergen_config.yml" in inputs, (
        "F7 regression: the weathergen template is not declared as an input of "
        "rule 3.10, so editing it will not re-trigger the rule"
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("generate_weather", "save_plots", True),
        ("apply_climate_perturbations", "pet_method", "hargreaves"),
        ("apply_climate_perturbations", "qm_fit_method", "mme"),
        ("apply_climate_perturbations", "diagnostic", False),
        ("run_weather_generator", "eval_max_grids", 25),
        ("write_netcdf", "calendar", "noleap"),
        ("generate_weather", "dry_spell_factor", [1.0] * 12),
        ("generate_weather", "wet_spell_factor", [1.0] * 12),
    ],
)
def test_surfaced_arguments_reach_the_generated_config(tmp_path, section, key, value):
    """C34: a surfaced argument is worthless if it does not reach the R.

    `build_weagen_config` seeds from the template wholesale, so a key added
    there arrives automatically -- this pins that, because the alternative
    (an explicit copy list) is what would silently drop it.
    """
    out = build_weagen_config(**_generate_kwargs(tmp_path))
    assert out[section][key] == value


def test_c34_retired_the_dead_evaluate_keys(tmp_path):
    """The two keys weathergenr 1.2.0 reaches with NOTHING are gone.

    `evaluate.model` claimed to control plot emission and did not; a user who
    set it FALSE still got every plot. Leaving a dead key that reads as a live
    setting is worse than removing it.
    """
    out = build_weagen_config(**_generate_kwargs(tmp_path))
    gw = out["generate_weather"]
    assert "evaluate.model" not in gw
    assert "evaluate.grid.num" not in gw
    # The whole pre-1.2.0 spelling is gone, not just those two.
    assert "generateWeatherSeries" not in out
