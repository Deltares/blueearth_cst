"""Assemble a weathergenr config YAML for a generate or stress-test run.

The config-assembly body was previously module-level code reading the
``snakemake`` global on import, which made it un-importable for unit tests.
R5 extracts it into named functions (``build_weathergen_config`` /
``compute_nr_years``) above a nested ``__main__`` / ``globals()`` guard so the
year math is reachable without a live ``snakemake`` global. Behavior-neutral:
the same dict is assembled and written.
"""

import os

import yaml

from blueearth_cst.shared.wf3_science import water_year_start_number


def read_yml(yml_path):
    """Read a yml file and return a dictionary."""
    with open(yml_path, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)
    return yml


def compute_nr_years(sim_end, historical_end):
    """Number of weathergen years to generate.

    Spans from the end of the historical record (``historical_end``, the last
    year of the configured climate window) to the end of the simulation window,
    plus a 2-year pad. The ``+2`` literal is the pad.
    """
    # `C-67`: the window END is declared now, so the generated record runs to
    # it directly instead of through a horizon-plus-half-length estimate that
    # rounded differently for an odd run length.
    return int(sim_end) - int(historical_end) + 2


#: `C-32`'s enum, closed. `transient` ramps the perturbation through the run;
#: `constant` holds it flat across the run.
#:
#: **RULED 2026-08-28.** The register row was ruled 2026-08-24 as
#: `transient | constant` and P1 shipped `transient | step`, so for four days
#: the design record and the code disagreed. `constant` won on three counts:
#: the register row is the decided one and the code was the deviation; both
#: words describe how the perturbation behaves over TIME, which is what
#: `trajectory` names, where `step` describes a shape; and the design, its
#: scoping note and P6's brief already agreed.
#:
#: The weathergen config this module WRITES still carries the boolean
#: `transient_change`; that is weathergenr's own vocabulary and K1/S5 put it
#: out of reach.
TRAJECTORY_KINDS = frozenset({"transient", "constant"})


def _transient_flag(stress_test_cfg, variable):
    """Read ``climate_perturbations.<variable>.trajectory``, refusing a default.

    Absent, this would decide whether a perturbation ramps or is held flat, and
    nobody would know which they got. The house rule for a missing required key is to
    refuse and name it (``variable_spec.parse``), not to guess.

    **The enum is CHECKED, not compared.** ``trajectory == "transient"`` alone
    would make every other string mean ``constant`` — so a typo, or the older
    ``transient_change: true`` written under the new key, would silently select
    the opposite behaviour and run to completion. That is the same silent
    default the required-key refusal above exists to prevent, one level down.
    """
    try:
        value = stress_test_cfg[variable]["trajectory"]
    except (KeyError, TypeError):
        raise ValueError(
            f"workflows.generate_scenarios.climate_perturbations.{variable}"
            ".trajectory is required (`C-32`; it was `transient_change: true`). "
            "It decides whether the perturbation ramps over the run "
            "(`transient`) or is held flat across it (`constant`), and the "
            "weather generator has no defensible default for it."
        ) from None
    if value not in TRAJECTORY_KINDS:
        raise ValueError(
            f"workflows.generate_scenarios.climate_perturbations.{variable}"
            f".trajectory must be one of {sorted(TRAJECTORY_KINDS)}, got "
            f"{value!r}. `C-32` replaced the boolean `transient_change` with "
            "this enum; `true` becomes `transient` and `false` becomes "
            "`constant`."
        )
    return value == "transient"


def build_weathergen_config(
    realizations_num,
    stress_test_cfg,
    output_path,
    nc_file_prefix,
    default_config_path,
    sim_end,
    seed,
    water_year_start,
    dry_spell_factor,
    wet_spell_factor,
    historical_end,
):
    """Assemble the ONE weathergenr config the experiment uses.

    Seeds from the default weathergen template, then overrides the output path,
    historical start year, number of years (``compute_nr_years``), file prefix
    and realization count from the project config. Adds the two
    ``transient_change`` flags that ``impose_climate_change.R`` reads.

    **C29 removed the second, per-member config** this function used to build.
    A former per-member rule emitted one ``weathergen_config_rlz_<n>_cst_<m>.yml`` per member —
    RLZ_NUM x ST_NUM files, each with its own log and benchmark — and the only
    thing that varied between them was the OUTPUT FILENAME, split into a prefix
    and a suffix because ``weathergenr::write_netcdf`` takes them separately.
    Snakemake already knows that path: it is rule 3.08's own declared output, so
    it is now passed as an argument and the rule is gone.

    **The project config is no longer read from disk** (R13 D-10.6). This
    function took a path and re-opened it for exactly two things -- the
    realization count and the two ``transient_change`` flags -- both of
    which the Snakefile has already composed and validated. They now arrive
    as params, finishing a conversion the six params above had already taken
    three-quarters of the way, and the disk read disappears rather than
    being redirected at the split layout.

    The per-member file also copied in the whole ``stress_test.temp`` and
    ``stress_test.precip`` blocks — step counts and monthly min/max ranges — of
    which the R read only the two transient flags (finding F6). Anyone opening
    one to see what a run did read plausible perturbation ranges that had no part
    in it; the real values come from ``st_<m>.csv``. Only the two flags survive
    here, so the file no longer implies otherwise.
    """
    yml_dict = read_yml(default_config_path)
    # Section and key names are weathergenr's own function and argument names
    # (renamed 2026-08-12 from `generateWeatherSeries`, a function 1.2.0 does
    # not export; tracking 2.0.0 since 2026-08-17). The values below are the
    # per-run ones the template cannot carry; every other argument comes from
    # the template verbatim.
    yml_dict["generate_weather"].update(
        {
            "out_dir": output_path,
            # The generated record starts where the historical one ends.
            "start_year": int(historical_end),
            "n_years": compute_nr_years(sim_end, historical_end),
            "n_realizations": realizations_num,
            # Resolved by the Snakefile from the generate_scenarios file's
            # top-level `seed` (integer or `auto`)
            # against `defaults.seed`. Injected rather than templated so there
            # is ONE default: a `seed:` left in the weathergen template would be a
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
    # files rule 3.07 emits. Kept under `generateWeatherSeries` until the 1.2.0
    # rename, where it had no matching argument.
    yml_dict["write_netcdf"]["file_prefix"] = nc_file_prefix

    # Read by impose_climate_change.R (rule 3.08). Only the flags, not the
    # perturbation magnitudes — those live in st_<m>.csv and are read from there.
    yml_dict["temp"] = {"transient_change": _transient_flag(stress_test_cfg, "temp")}
    yml_dict["precip"] = {
        "transient_change": _transient_flag(stress_test_cfg, "precip")
    }

    return yml_dict


def write_weathergen_config(yml_dict, weathergen_config_path):
    """Write the assembled weathergen config dict to ``weathergen_config_path``."""
    if not os.path.isdir(os.path.dirname(weathergen_config_path)):
        os.makedirs(os.path.dirname(weathergen_config_path))
    with open(weathergen_config_path, "w") as f:
        yaml.dump(yml_dict, f, default_flow_style=False, sort_keys=False)
