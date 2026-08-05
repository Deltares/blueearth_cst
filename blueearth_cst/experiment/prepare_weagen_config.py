"""Assemble a weathergenr config YAML for a generate or stress-test run.

The config-assembly body was previously module-level code reading the
``snakemake`` global on import, which made it un-importable for unit tests.
R5 extracts it into named functions (``build_weagen_config`` /
``compute_nr_years``) above a nested ``__main__`` / ``globals()`` guard so the
year math is reachable without a live ``snakemake`` global. Behavior-neutral:
the same dict is assembled and written.
"""

import os
import math
import yaml


def read_yml(yml_path):
    """Read a yml file and return a dictionary."""
    with open(yml_path, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)
    return yml


def compute_nr_years(middle_year, wflow_run_length):
    """Number of weagen years to generate.

    Spans from the end of the historical period (2010) to the wflow run window
    around the horizon (``middle_year`` ± ``wflow_run_length``/2), plus a 2-year
    pad. The ``2010`` and ``+2`` literals are the historical-end anchor and pad.
    """
    return math.ceil((middle_year + wflow_run_length / 2) - 2010 + 2)


def _transient_flag(stress_test_cfg, variable):
    """Read ``stress_test.<variable>.transient_change``, refusing a silent default.

    Absent, this would decide whether a perturbation ramps or steps and nobody
    would know which they got. The house rule for a missing required key is to
    refuse and name it (``variable_spec.parse``), not to guess.
    """
    try:
        return stress_test_cfg[variable]["transient_change"]
    except (KeyError, TypeError):
        raise ValueError(
            f"workflows.climate_experiment.stress_test.{variable}.transient_change "
            "is required: it decides whether the perturbation ramps over the run "
            "or applies as a step, and the weather generator has no defensible "
            "default for it."
        ) from None


def build_weagen_config(
    snake_config_path,
    output_path,
    nc_file_prefix,
    default_config_path,
    middle_year,
    sim_years,
):
    """Assemble the ONE weathergenr config the experiment uses.

    Seeds from the default weagen template, then overrides the output path,
    historical start year, number of years (``compute_nr_years``), file prefix
    and realization count from the snake config. Adds the two
    ``transient_change`` flags that ``impose_climate_change.R`` reads.

    **C29 removed the second, per-member config** this function used to build.
    Rule 3.05 emitted one ``weathergen_config_rlz_<n>_cst_<m>.yml`` per member —
    RLZ_NUM x ST_NUM files, each with its own log and benchmark — and the only
    thing that varied between them was the OUTPUT FILENAME, split into a prefix
    and a suffix because ``weathergenr::write_netcdf`` takes them separately.
    Snakemake already knows that path: it is rule 3.07's own declared output, so
    it is now passed as an argument and the rule is gone.

    The per-member file also copied in the whole ``stress_test.temp`` and
    ``stress_test.precip`` blocks — step counts and monthly min/max ranges — of
    which the R read only the two transient flags (finding F6). Anyone opening
    one to see what a run did read plausible perturbation ranges that had no part
    in it; the real values come from ``cst_<m>.csv``. Only the two flags survive
    here, so the file no longer implies otherwise.
    """
    yml_snake = read_yml(snake_config_path)
    experiment_cfg = yml_snake["workflows"]["climate_experiment"]

    yml_dict = read_yml(default_config_path)
    yml_add = {
        "output.path": output_path,
        "sim.year.start": 2010,
        "sim.year.num": compute_nr_years(middle_year, sim_years),
        "nc.file.prefix": nc_file_prefix,
        "realizations_num": experiment_cfg["realizations_num"],
    }
    for k, v in yml_add.items():
        yml_dict["generateWeatherSeries"][k] = v

    # Read by impose_climate_change.R (rule 3.07). Only the flags, not the
    # perturbation magnitudes — those live in cst_<m>.csv and are read from there.
    stress_test_cfg = experiment_cfg["stress_test"]
    yml_dict["temp"] = {"transient_change": _transient_flag(stress_test_cfg, "temp")}
    yml_dict["precip"] = {"transient_change": _transient_flag(stress_test_cfg, "precip")}

    return yml_dict


def write_weagen_config(yml_dict, weagen_config_path):
    """Write the assembled weagen config dict to ``weagen_config_path``."""
    if not os.path.isdir(os.path.dirname(weagen_config_path)):
        os.makedirs(os.path.dirname(weagen_config_path))
    with open(weagen_config_path, "w") as f:
        yaml.dump(yml_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            weagen_config = sm.output.weagen_config
            log_row(
                f"Preparing and writing the weather generator config file {weagen_config}",
                module="weagen",
            )
            yml_dict = build_weagen_config(
                snake_config_path=sm.params.snake_config,
                output_path=sm.params.output_path,
                nc_file_prefix=sm.params.nc_file_prefix,
                default_config_path=sm.params.default_config,
                middle_year=sm.params.middle_year,
                sim_years=sm.params.sim_years,
            )
            write_weagen_config(yml_dict, weagen_config)
    else:
        raise ValueError("This script should be run from a snakemake environment")
