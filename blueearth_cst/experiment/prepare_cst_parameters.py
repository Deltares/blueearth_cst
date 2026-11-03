import os
import sys
from os.path import join
from pathlib import Path
from typing import Union

# numpy, pandas and yaml are DEFERRED into the functions that use them.
# `generate_scenarios.smk` imports this module at PARSE time for the single refusal
# `refuse_out_of_domain_multipliers` (D35), which reads a config dict and
# nothing else -- so a module-level import bought the numeric stack (~4.6s) on
# every WF3 dry-run and every real run in order to validate a YAML section. The
# `script:` entry point pays the same imports it always did, one call deeper.

# Import the shared grid helper regardless of the working directory. The
# Snakefile prepends its basedir to sys.path before invoking script: rules, but
# guard here so the module is import-clean for unit tests too -- and for the
# PARSE-TIME call to refuse_out_of_domain_multipliers below (D35), which
# generate_scenarios.smk makes before the DAG is built.
# parents[2] is the REPO ROOT (file -> experiment/ -> blueearth_cst/ ->
# root); parent.parent stopped at the package dir, from which
# `import blueearth_cst.shared...` cannot resolve (O-07).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blueearth_cst.shared.run_log_core import log_row, plural
from blueearth_cst.shared.wf3_science import index_width, stress_test_grid

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
_NON_AXIS_KEYS = ("spell_factors",)  # C-33: one group, was two flat keys

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


def _as_month_list(values) -> list:
    """A bound declared as one scalar, or as twelve monthly values, as a list.

    Was ``np.atleast_1d``. Plain Python instead, so the PARSE-TIME refusal below
    reaches no third-party import at all: a bound is a YAML scalar or a YAML
    sequence, and nothing here needs array semantics. Deferring the import
    without this would not have helped -- the refusal itself would still have
    triggered it.
    """
    return list(values) if isinstance(values, (list, tuple)) else [values]


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
                for month, value in enumerate(_as_month_list(values), start=1)
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
    import numpy as np

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
    stress_test_cfg: Union[dict, None] = None,
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
    stress_test_cfg : dict, optional
        The already-resolved ``stress_test`` section. The rule passes it as a
        param (R13 D-10.6) so this module never re-reads the config from
        disk: since the split, ``workflows.generate_scenarios.stress_test``
        exists in no single file a caller could hand over. Passing the
        section is also what gives rule 3.03 a rerun trigger at all -- its
        only config input is ``ancient()``, which by construction triggers
        nothing, so before this the grid values could change and the rule
        stay satisfied with a stale lookup table.

        Omitted, the section is composed from ``config_fn``, which is what
        the direct-invocation branch below does.
    """

    import numpy as np
    import pandas as pd

    if stress_test_cfg is None:
        # Direct invocation: compose the project file the same way a run
        # does, so `python -m ...` keeps working against the one path
        # anybody would think to pass.
        from blueearth_cst.shared.config_composition import load_composed_config

        composed = load_composed_config(
            config_fn,
            entry="generate_scenarios",
            declared_sections=("workflows.generate_scenarios",),
        )
        stress_test_cfg = composed["workflows"]["generate_scenarios"][
            "climate_perturbations"
        ]

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
            f"column and updating the producer and consumers together (C28); "
            f"see dev/reference/toolbox-weather-generator-interface.md"
            f"#wg-2--stress-test-perturbation-grid."
        )

    # Grid step counts + total via the shared helper (single source of truth,
    # strict on a missing n_levels). temp_step_num / precip_step_num are the
    # per-axis LEVEL counts -- since `C-31` that is `n_levels` itself, not
    # `step_num + 1` -- and they size the linspaces and the loops below.
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
    # unperturbed baseline has no parameters. Rule 3.08 still runs for it, at
    # unit factors synthesized from the `identity` member token, so it takes
    # the same quantile-mapping path as the grid (t2608151154) -- before that it
    # was the raw generated series, and five of eleven `q` indicators differed
    # from the grid's own identity member by a factor.
    #
    # Its absence stays LOAD-BEARING: the lookup is the grid a user configured,
    # and the baseline is not one of its points whatever values it takes.
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
    # Rule 3.03 declares a `log:`, so `merge_logs` opened a section for it on
    # every run and found nothing to put under it. The member count is the
    # grid this experiment will actually run -- the one number a reader of this
    # rule wants -- and twelve rows per member is WG-2's monthly grain.
    log_row(
        f"Wrote {lookup_fn} ({len(lookup)} rows, {plural(len(lookup) // 12, 'member')})",
        module="experiment",
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.run_log_core import tee_to_log

        with tee_to_log(sm.log[0]):
            prep_cst_parameters(
                config_fn=sm.input.config,
                lookup_fn=sm.output.lookup_csv,
                stress_test_cfg=sm.params.stress_test_cfg,
            )
    else:
        # Direct invocation takes the config path as an argument; naming one
        # here would be a path that goes stale the next time a seed is renamed.
        prep_cst_parameters(config_fn=sys.argv[1])
