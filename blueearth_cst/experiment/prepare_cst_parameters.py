import os
import sys
from os.path import join
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import yaml

# Import the shared grid helper regardless of the working directory. The
# Snakefile prepends its basedir to sys.path before invoking script: rules, but
# guard here so the module is import-clean for unit tests too.
# parents[2] is the REPO ROOT (file -> experiment/ -> blueearth_cst/ ->
# root); parent.parent stopped at the package dir, from which
# `import blueearth_cst.shared...` cannot resolve (O-07).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# The SAME twelve-to-one reduction the indicator tables use (month-length
# weighted mean). Imported rather than reimplemented: C28 makes `validate_hm7`
# assert that a results row's temp_change/precip_change equals the design
# table's row for that st_id, and two independent collapses of the same twelve
# monthly values would make that check fail on rounding rather than on a defect.
from blueearth_cst.experiment.export_wflow_results import (
    annual_perturbation,
    perturbation_axes,
)
from blueearth_cst.shared.snake_utils import index_width, stress_test_grid

#: The stress-test axes this module knows how to enumerate. A third axis needs a
#: new design-table column AND a new results column, so it must arrive as a
#: refusal rather than as a silently missing dimension (C28's second obligation).
_KNOWN_AXES = ("temp", "precip")

#: Keys that live under `stress_test` but are NOT perturbation axes: they are
#: monthly spell-length coefficients handed to the weather generator, with no
#: design-table column and no grid contribution. Listed so the axis guard
#: below still refuses a typo'd or genuinely new AXIS while admitting these.
_NON_AXIS_KEYS = ("dry_spell_factor", "wet_spell_factor")

#: Design-table header (C23/C24). `st_id` is the DESIGNED axis; `realization` is
#: the sampled one and deliberately absent -- run identity is `(rlz, st)`, and a
#: draw has no design parameters to record.
DESIGN_COLUMNS = ("st_id", "temp_change", "precip_change", "precip_variance_change")


def prep_cst_parameters(
    config_fn: Union[str, Path],
    csv_fns: List[Union[str, Path]],
    design_fn: Union[str, Path, None] = None,
):
    """
    Prepare a csv file for each stress test scenario, and the design table.

    Parameters
    ----------
    config_fn : str, Path
        Path to the config file
    csv_fns : List[str, Path]
        List of paths to the output csv files. If None saves in same directory as
        config_fn and names from stress test parameters.
    design_fn : str, Path, optional
        Path to ``stress_test_design.csv`` (C23). Written from the SAME loop that
        writes the per-member files, so the two cannot disagree about what run
        ``m`` is -- which is the property C26 exists for. Skipped when None.
    """

    # Read the yaml config (R01 sectioned schema)
    with open(config_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)

    stress_test_cfg = yml["workflows"]["run_stress_test"]["stress_test"]

    # A third stress dimension must REFUSE, not silently vanish from the design
    # table (C28). The grid arithmetic, the CSV loop and DESIGN_COLUMNS below all
    # assume exactly two axes; adding one without touching them would emit a
    # table that describes a different experiment than the one that ran.
    unknown_axes = sorted(set(stress_test_cfg) - set(_KNOWN_AXES) - set(_NON_AXIS_KEYS))
    if unknown_axes:
        raise ValueError(
            f"stress_test carries unsupported axes {unknown_axes}: this module "
            f"enumerates exactly {list(_KNOWN_AXES)} (plus the non-axis keys "
            f"{list(_NON_AXIS_KEYS)}). Adding a dimension means "
            f"adding a design-table column and a results column together (C28); "
            f"see dev/milestones/r09/wf3-change-requests.md."
        )

    # Grid step counts + total via the shared helper (single source of truth,
    # strict on a missing step_num). temp_step_num / precip_step_num are the
    # per-axis counts (step_num + 1) that size the linspaces and the loops below.
    temp_step_num, precip_step_num, ST_NUM = stress_test_grid(stress_test_cfg)

    # Temperature change attributes
    delta_temp_mean_min = stress_test_cfg["temp"]["mean"]["min"]
    delta_temp_mean_max = stress_test_cfg["temp"]["mean"]["max"]

    # Precip change attributes
    delta_precip_mean_min = stress_test_cfg["precip"]["mean"]["min"]
    delta_precip_mean_max = stress_test_cfg["precip"]["mean"]["max"]
    delta_precip_variance_min = stress_test_cfg["precip"]["variance"]["min"]
    delta_precip_variance_max = stress_test_cfg["precip"]["variance"]["max"]
    # Stress test values per variables
    temp_values = np.linspace(
        delta_temp_mean_min, delta_temp_mean_max, temp_step_num, axis=1
    )
    precip_values = np.linspace(
        delta_precip_mean_min, delta_precip_mean_max, precip_step_num, axis=1
    )
    precip_var_values = np.linspace(
        delta_precip_variance_min, delta_precip_variance_max, precip_step_num, axis=1
    )

    # The design table's ids are padded to the same count-derived width as the
    # filenames (C27), so `st_id` and the member filename are textually the same
    # token and a consumer joining a plot to its run needs no coercion.
    st_width = index_width(ST_NUM)

    # C23: the reserved unperturbed baseline is a REAL row with every change
    # zero. A response surface missing its own origin forces every downstream
    # consumer to reconstruct it.
    design_rows = [
        {
            "st_id": f"{0:0{st_width}d}",
            "temp_change": 0.0,
            "precip_change": 0.0,
            "precip_variance_change": 0.0,
        }
    ]

    # Generate csv file for each stress test scenario
    i = 0
    for j in range(temp_step_num):
        temp_j = temp_values[:, j]
        for k in range(precip_step_num):
            precip_k = precip_values[:, k]
            precip_var_k = precip_var_values[:, k]

            # Create df and save to csv
            data = {
                "temp_mean": temp_j,
                "precip_mean": precip_k,
                "precip_variance": precip_var_k,
            }
            df = pd.DataFrame(data=data, dtype=np.float32, index=np.arange(1, 13))
            df.index.name = "month"
            if csv_fns is None:
                # Auto-naming fallback (no Snakemake): pad to the same
                # count-derived width the rule's output: declaration uses, so
                # the two spellings of a member name cannot diverge (C27).
                csv_fn = join(
                    os.path.dirname(config_fn),
                    f"st_{i + 1:0{index_width(ST_NUM)}d}.csv",
                )
            else:
                csv_fn = csv_fns[i]
            df.to_csv(csv_fn)

            # Derive the design row from the PERSISTED file, not from `df`
            # (R11 P3, 2026-08-08). `df` is float32; `df.to_csv` writes it as
            # text, and every downstream reader -- the weather generator at 3.12
            # and the results writer at 3.16 -- reads that text back as float64.
            # So a design row computed from `df` records a perturbation NOBODY
            # APPLIED: float32(0.7) is 0.69999998807, giving -30.000001%, while
            # the run actually imposed the round-tripped 0.7, i.e. -30.0%.
            #
            # Found by C28's own consistency check the first time it ran against
            # real data (P3). It had never run on the fixture: the integration
            # test called validate_hm7 without a `design=`, so the check that was
            # supposed to make this artifact trustworthy was skipped entirely.
            #
            # The two sides stay independent in the way C28 needs -- different
            # code, different rule, different job -- but now read the same bytes,
            # so the check verifies the design against the results rather than
            # against a precision artifact. What it no longer catches is a lossy
            # CSV write; that is a deliberate trade, ruled 2026-08-08.
            persisted = pd.read_csv(csv_fn, index_col="month")
            temp_change, precip_change = perturbation_axes(persisted, csv_fn)
            design_rows.append(
                {
                    "st_id": f"{i + 1:0{st_width}d}",
                    "temp_change": temp_change,
                    "precip_change": precip_change,
                    # Percent, matching precip_change: both are factors in the
                    # parameter file, and one table must not mix conventions.
                    "precip_variance_change": (
                        annual_perturbation(persisted, "precip_variance", csv_fn) * 100
                        - 100
                    ),
                }
            )

            i += 1

    if design_fn is not None:
        design = pd.DataFrame(design_rows, columns=list(DESIGN_COLUMNS))
        Path(design_fn).parent.mkdir(parents=True, exist_ok=True)
        design.to_csv(design_fn, index=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            prep_cst_parameters(
                config_fn=sm.input.config,
                csv_fns=sm.output.st_csv_fns,
                design_fn=sm.output.design_csv,
            )
    else:
        prep_cst_parameters(
            config_fn=join(os.getcwd(), "config", "snake_config_model_test.yml"),
        )
