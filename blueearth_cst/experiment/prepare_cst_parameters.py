import os
import sys
from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

# Import the shared grid helper regardless of the working directory. The
# Snakefile prepends its basedir to sys.path before invoking script: rules, but
# guard here so the module is import-clean for unit tests too -- and for the
# PARSE-TIME call to refuse_out_of_domain_multipliers below (D35), which
# run_stress_test.smk makes before the DAG is built.
# parents[2] is the REPO ROOT (file -> experiment/ -> blueearth_cst/ ->
# root); parent.parent stopped at the package dir, from which
# `import blueearth_cst.shared...` cannot resolve (O-07).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blueearth_cst.shared.snake_utils import index_width, stress_test_grid

#: The stress-test axes this module knows how to enumerate. A third axis needs a
#: new lookup COLUMN, and adding one requires a C28 ruling -- removing the shape
#: barrier (one table instead of ST_NUM files) did not remove the contract
#: barrier, so this still arrives as a refusal rather than a silently missing
#: dimension.
_KNOWN_AXES = ("temp", "precip")

#: Keys that live under `stress_test` but are NOT perturbation axes: they are
#: monthly spell-length coefficients handed to the weather generator, with no
#: lookup column and no grid contribution. Listed so the axis guard below still
#: refuses a typo'd or genuinely new AXIS while admitting these.
_NON_AXIS_KEYS = ("dry_spell_factor", "wet_spell_factor")

#: The lookup's header, in order (WG-2). `st_id` x `month` is the key; the three
#: value columns are the whole vocabulary. `realization` is deliberately absent:
#: run identity is `(rlz, st)`, and a DRAW has no design parameters to record.
LOOKUP_COLUMNS = (
    "st_id",
    "month",
    "temp_change",
    "precip_change",
    "precip_variance_change",
)

#: Multipliers below this are refused (D35). The bound WG-2 states -- that the
#: generator's reconstructed multiplier is within one `float64` ulp of the grid
#: level -- is TRUE over this domain and FALSE below it, so the domain is the
#: bound's precondition rather than a caveat on it. There is deliberately no
#: ceiling: the bound was measured to hold out to 1e6, so a cap would refuse
#: configurations the arithmetic serves correctly.
MULTIPLIER_DOMAIN = (0.5, None)

#: The `stress_test` keys whose values cross the Python->R seam as PERCENT and
#: are therefore reconstructed as `1 + p/100` on the other side. `temp` is absent
#: because it is additive degC and crosses unconverted, so no reconstruction
#: happens and none can be out of bound.
_PERCENT_CONVERTED_KEYS = (("precip", "mean"), ("precip", "variance"))


class MultiplierDomainError(ValueError):
    """A precipitation multiplier outside the domain WG-2 states its bound over."""


def refuse_out_of_domain_multipliers(stress_test_cfg: dict) -> None:
    """Refuse a config declaring a precipitation multiplier below the domain (D35).

    Called at Snakefile PARSE time, so the refusal lands before the DAG is built:
    ``--dry-run`` fails, and no member file, no realization and no wflow run is
    produced under a config whose reconstruction the contract cannot bound.

    Checks the endpoints only, and that is exact rather than a shortcut: the grid
    levels are ``np.linspace`` between ``min`` and ``max`` and are therefore
    monotone between them, so validating the endpoints validates every level.

    BOTH percent-converted keys are checked, not just ``mean``. The conversion is
    a rule over the percent COLUMNS rather than a formula for one of them, so a
    domain covering ``precip.mean`` alone would re-create that defect one layer
    up. No shipped config declares a variance below 1.0, so this costs nothing
    today; it is stated as a decision rather than left to be found by whoever
    first writes one.
    """
    floor, _ceiling = MULTIPLIER_DOMAIN
    for section, key in _PERCENT_CONVERTED_KEYS:
        block = (stress_test_cfg.get(section) or {}).get(key) or {}
        for bound in ("min", "max"):
            values = block.get(bound)
            if values is None:
                continue
            offenders = [
                (month, value)
                for month, value in enumerate(np.atleast_1d(values), start=1)
                if float(value) < floor
            ]
            if offenders:
                months = ", ".join(f"{m}: {v}" for m, v in offenders)
                raise MultiplierDomainError(
                    f"stress_test.{section}.{key}.{bound} declares multipliers "
                    f"below {floor} ({months}). The lookup crosses the "
                    f"Python->R seam in PERCENT and the generator reconstructs "
                    f"`1 + p/100`; that reconstruction is within one float64 ulp "
                    f"of the level only for multipliers >= {floor}, which is the "
                    f"bound WG-2 states. Below it the error grows without limit "
                    f"(68 ulps at 0.0136). There is no upper bound."
                )


def _level(value: float) -> float:
    """The grid level a member actually carries: its `float32` shortest repr.

    The grid the user asked for is quantized to `float32` -- that is what the
    member frames have always been built and written as -- so this is the value
    the run imposes, and the value any reconstruction must return.
    """
    return float(str(np.float32(value)))


def _percent(level: float) -> float:
    """A multiplier level as a percent change (D3's forward direction).

    Spelled ``level * 100 - 100`` and NOT ``(level - 1) * 100``: measured,
    ``1.3 * 100 - 100 == 30.0`` exactly while ``(1.3 - 1.0) * 100`` is
    ``30.000000000000004``. This is the formula the retired
    ``export_wflow_results.perturbation_axes`` already used -- preserved, not
    invented.

    The result is a Python ``float``, so the CSV carries its `float64` shortest
    repr. Re-quantizing the PERCENT to `float32` would degrade the reconstructed
    multiplier by roughly eight orders of magnitude (one `float32` ulp, ~6e-08,
    against ~1e-16 here). The column that must stay coarse is the GRID; the
    column that must stay fine is the TEXT.
    """
    return level * 100 - 100


def prep_cst_parameters(
    config_fn: Union[str, Path],
    lookup_fn: Union[str, Path, None] = None,
):
    """Write the stress-test lookup: one long table, `12 x ST_NUM` rows.

    Replaces the per-member ``_work/st_<m>.csv`` grid AND the derived
    ``stress_test_design.csv`` with a single artifact at monthly grain (WG-2).
    One loop still enumerates the members, so the enumeration that NAMES a member
    and the one that DESCRIBES it cannot disagree -- C26's property, preserved
    through the shape change.

    There is no second derivation and no disk round trip: R11 P3's read-it-back
    hack existed to keep a denormalised annual cache in step with the member
    files it was computed from, and it retires with that cache.

    Parameters
    ----------
    config_fn : str, Path
        Path to the config file.
    lookup_fn : str, Path, optional
        Path to ``stress_test_lookup.csv``. Defaults to the config's own
        directory, for use outside Snakemake.
    """

    # Read the yaml config (R01 sectioned schema)
    with open(config_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)

    stress_test_cfg = yml["workflows"]["run_stress_test"]["stress_test"]

    # A third stress dimension must REFUSE, not silently vanish from the lookup
    # (C28). The grid arithmetic, the row loop and LOOKUP_COLUMNS below all
    # assume exactly two axes; adding one without touching them would emit a
    # table that describes a different experiment than the one that ran.
    unknown_axes = sorted(set(stress_test_cfg) - set(_KNOWN_AXES) - set(_NON_AXIS_KEYS))
    if unknown_axes:
        raise ValueError(
            f"stress_test carries unsupported axes {unknown_axes}: this module "
            f"enumerates exactly {list(_KNOWN_AXES)} (plus the non-axis keys "
            f"{list(_NON_AXIS_KEYS)}). Adding a dimension means adding a lookup "
            f"column, and that needs a C28 ruling; "
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

    # The lookup's ids are padded to the same count-derived width the member
    # filename token uses (C27), so `st_id` and that token are ONE token and a
    # consumer joining results to the lookup needs no coercion.
    st_width = index_width(ST_NUM)

    # `st_0` has NO row: the table is the PARAMETER GRID, and the reserved
    # unperturbed baseline has no parameters -- it is produced by rule 3.11, not
    # by perturbation, and rule 3.12 never runs for it.
    #
    # Its absence is LOAD-BEARING, and this supersedes C23's recorded rationale
    # ("a response surface missing its own origin forces every downstream
    # consumer to reconstruct it"), which assumed st_0 IS the surface origin. It
    # is not: an all-zero st_0 row would be indistinguishable from an identity
    # member's row while denoting a differently-processed climate -- the raw
    # generated series, not that series round-tripped through a perturbation
    # that is NOT the identity at unit factors (measured: five of eleven `q`
    # metrics move by a factor). Absence is the strongest available marking, and
    # it makes "not on the surface" structural rather than conventional.
    rows = []

    i = 0
    for j in range(temp_step_num):
        temp_j = temp_values[:, j]
        for k in range(precip_step_num):
            precip_k = precip_values[:, k]
            precip_var_k = precip_var_values[:, k]

            st_id = f"{i + 1:0{st_width}d}"
            for month in range(1, 13):
                m = month - 1
                rows.append(
                    {
                        "st_id": st_id,
                        "month": month,
                        # Additive degC: it crosses the seam unconverted, so no
                        # reconstruction happens on the R side.
                        "temp_change": _level(temp_j[m]),
                        "precip_change": _percent(_level(precip_k[m])),
                        "precip_variance_change": _percent(_level(precip_var_k[m])),
                    }
                )

            i += 1

    if lookup_fn is None:
        lookup_fn = join(os.path.dirname(config_fn), "stress_test_lookup.csv")

    lookup = pd.DataFrame(rows, columns=list(LOOKUP_COLUMNS))
    Path(lookup_fn).parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(lookup_fn, index=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            prep_cst_parameters(
                config_fn=sm.input.config,
                lookup_fn=sm.output.lookup_csv,
            )
    else:
        # Direct invocation takes the config path as an argument; naming one
        # here would be a path that goes stale the next time a seed is renamed.
        prep_cst_parameters(config_fn=sys.argv[1])
