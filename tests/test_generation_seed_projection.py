"""GF27: value dependencies select draws; execution and presentation do not."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment.generation_plan import generator_seed_projection


@pytest.fixture
def generator():
    path = Path(__file__).resolve().parents[1] / "config/defaults/weathergen_config.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("generate_weather", "parallel", True),
        ("generate_weather", "n_cores", 2),
        ("generate_weather", "verbose", False),
        ("generate_weather", "save_plots", False),
        ("generate_weather", "seed", 999),
        ("generate_weather", "out_dir", "different/path"),
        ("run_weather_generator", "eval_max_grids", 1),
        ("run_weather_generator", "log_messages", True),
        ("apply_climate_perturbations", "verbose", True),
        ("write_netcdf", "compression", 1),
        ("write_netcdf", "file_prefix", "elsewhere"),
        ("write_netcdf", "verbose", False),
    ],
)
def test_excluded_fields_leave_draw_projection_unchanged(
    generator, section, key, value
):
    changed = deepcopy(generator)
    changed[section][key] = value
    assert generator_seed_projection(changed) == generator_seed_projection(generator)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("generate_weather", "warm_signif", 0.9),
        ("generate_weather", "wet_q", 0.3),
        ("apply_climate_perturbations", "enforce_target_mean", False),
        ("write_netcdf", "calendar", "standard"),
        ("write_netcdf", "signif_digits", 4),
    ],
)
def test_value_dependencies_change_draw_projection(generator, section, key, value):
    changed = deepcopy(generator)
    changed[section][key] = value
    assert generator_seed_projection(changed) != generator_seed_projection(generator)


@pytest.mark.parametrize("section", [None, "generate_weather", "write_netcdf"])
def test_unclassified_fields_fail_closed(generator, section):
    if section is None:
        generator["future_function"] = {}
    else:
        generator[section]["future_argument"] = True
    with pytest.raises(ValueError, match="seed-dependency declaration"):
        generator_seed_projection(generator)


@pytest.mark.parametrize("value", [True, None, 0])
def test_diagnostic_return_shape_is_enforced(generator, value):
    generator["apply_climate_perturbations"]["diagnostic"] = value
    with pytest.raises(ValueError, match="return-shape"):
        generator_seed_projection(generator)
