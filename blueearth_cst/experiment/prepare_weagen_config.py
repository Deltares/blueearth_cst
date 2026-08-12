"""Assemble a weathergenr config YAML for a generate or stress-test run.

The config-assembly body was previously module-level code reading the
``snakemake`` global on import, which made it un-importable for unit tests.
R5 extracts it into named functions (``build_weagen_config`` /
``compute_nr_years``) above a nested ``__main__`` / ``globals()`` guard so the
year math is reachable without a live ``snakemake`` global. Behavior-neutral:
the same dict is assembled and written.
"""

import math
import os

import yaml

from blueearth_cst.shared.snake_utils import water_year_start_number


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
    seed,
    water_year_start,
    dry_spell_factor,
    wet_spell_factor,
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
    Snakemake already knows that path: it is rule 3.12's own declared output, so
    it is now passed as an argument and the rule is gone.

    The per-member file also copied in the whole ``stress_test.temp`` and
    ``stress_test.precip`` blocks — step counts and monthly min/max ranges — of
    which the R read only the two transient flags (finding F6). Anyone opening
    one to see what a run did read plausible perturbation ranges that had no part
    in it; the real values come from ``st_<m>.csv``. Only the two flags survive
    here, so the file no longer implies otherwise.
    """
    yml_snake = read_yml(snake_config_path)
    experiment_cfg = yml_snake["workflows"]["climate_experiment"]

    yml_dict = read_yml(default_config_path)
    # Section and key names are weathergenr 1.2.0's own function and argument
    # names (renamed 2026-08-12 from `generateWeatherSeries`, a function 1.2.0
    # does not export). The values below are the per-run ones the template
    # cannot carry; every other argument comes from the template verbatim.
    yml_dict["generate_weather"].update(
        {
            "out_dir": output_path,
            "start_year": 2010,
            "n_years": compute_nr_years(middle_year, sim_years),
            "n_realizations": experiment_cfg["realizations_num"],
            # Resolved by the Snakefile from `shared.seed` (integer or `auto`)
            # against `defaults.seed`. Injected rather than templated so there
            # is ONE default: a `seed:` left in the weagen template would be a
            # second one, and the two would drift the first time either moved.
            "seed": seed,
            # weathergenr wants the month NUMBER; the config carries the
            # three-letter name, which is the spelling every other consumer
            # of `shared.water_year_start` uses. Converted here, at the seam.
            "year_start_month": water_year_start_number(water_year_start),
            # Moved out of the weathergen template 2026-08-12: these are
            # stress-test knobs, so they live beside temp/precip under
            # `stress_test`, and the Snakefile validates their length.
            "dry_spell_factor": dry_spell_factor,
            "wet_spell_factor": wet_spell_factor,
        }
    )
    # Belongs to write_netcdf, not to the generator: it names the realization
    # files rule 3.11 emits. Kept under `generateWeatherSeries` until the 1.2.0
    # rename, where it had no matching argument.
    yml_dict["write_netcdf"]["file_prefix"] = nc_file_prefix

    # Read by impose_climate_change.R (rule 3.12). Only the flags, not the
    # perturbation magnitudes — those live in st_<m>.csv and are read from there.
    stress_test_cfg = experiment_cfg["stress_test"]
    yml_dict["temp"] = {"transient_change": _transient_flag(stress_test_cfg, "temp")}
    yml_dict["precip"] = {
        "transient_change": _transient_flag(stress_test_cfg, "precip")
    }

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
                seed=sm.params.seed,
                water_year_start=sm.params.water_year_start,
                dry_spell_factor=sm.params.dry_spell_factor,
                wet_spell_factor=sm.params.wet_spell_factor,
            )
            write_weagen_config(yml_dict, weagen_config)
    else:
        raise ValueError("This script should be run from a snakemake environment")
