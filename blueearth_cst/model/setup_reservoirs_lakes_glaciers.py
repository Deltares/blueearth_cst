"""Update a wflow model with reservoirs, lakes and glaciers.

This runs as a **separate rule** from the build (1.07 ``build_wflow_model``)
for two reasons, and neither is an upstream limitation.

Per-method no-data: a basin may legitimately lack reservoirs, lakes or
glaciers, so each method runs individually here and a ``NoDataException`` skips
only that one. This much COULD move into the build — ``build_wflow_model``
applies its own steps through the same kind of loop
(``_apply_parameter_steps``), so the tolerance is ours to add, not hydromt's to
fix.

Failure isolation is why it does not move. Rule 1.06 resamples soil, land-use
and LAI grids and scales with basin size, so on a production basin it is one of
the expensive rules; the waterbody methods depend on external catalog sources
that are often absent or mis-resolved. Folding them in would redo the whole
parameterization whenever a catalog lookup fails — trading a fixed few seconds
for a rebuild cost that grows with the basin.

The earlier rationale claimed the split existed because hydromt 1.3's
``hydromt build`` cannot tolerate per-method no-data, with a removal trigger
waiting on upstream. That trigger is withdrawn: rule 1.06 never invokes the
``hydromt build`` CLI, so the condition could not have fired. See
``dev/reference/wf1-model-construction.md`` for the current construction contract.
"""

import os
from os.path import join
from pathlib import Path
from typing import Union

import yaml

from blueearth_cst.shared.progress import hydromt_progress
from blueearth_cst.shared.wflow_write import write_model_except_forcing


def _run_waterbody_methods(mod, config, no_data_errors):
    """Run each configured method on ``mod``, capturing per-method outcome.

    Returns a list of ``{"method", "status", "reason"}`` dicts where status is
    ``"ok"`` or ``"skipped"`` (the method's data source was absent for this
    basin — a legitimate outcome, not a failure).
    """
    results = []
    for method, kwargs in config.items():
        kwargs = kwargs or {}
        # ONE mapping, used for the call and for the record -- the same
        # discipline as build_wflow_model's steps. Nothing is normalized here
        # (these kwargs reach hydromt as configured), but recording them from a
        # second read of the config file would let the two drift the moment
        # normalization is ever added.
        entry = {"method": method, "status": "ok", "reason": "", "values_used": kwargs}
        try:
            getattr(mod, method)(**kwargs)
        except no_data_errors as error:
            entry["status"] = "skipped"
            entry["reason"] = str(error)
        results.append(entry)
    return results


def write_values_used(path, results, config_fn):
    """Record what each waterbody method was handed, and whether it ran.

    The status is half the provenance: a skipped method leaves no trace in the
    model, so a record of values alone would describe reservoirs and lakes that
    were never added. "Skipped" here is a legitimate outcome -- the basin has
    no such water bodies in the source -- not a failure.
    """
    document = {
        "schema_version": 1,
        "waterbodies_config": str(config_fn),
        "steps": [
            {
                "method": entry["method"],
                "status": entry["status"],
                "reason": entry["reason"],
                "values_used": entry["values_used"],
            }
            for entry in results
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)


def write_sentinel(path, results):
    """Write a structured (TSV) sentinel: one ``method status reason`` row.

    Replaces the previous free-text repr so the log sweep can parse it. The
    sentinel exists only for Snakemake tracking; it is not a manifest target.
    """
    lines = ["method\tstatus\treason"]
    for r in results:
        reason = r["reason"].replace("\t", " ").replace("\n", " ")
        lines.append(f"{r['method']}\t{r['status']}\t{reason}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def update_wflow_waterbodies_glaciers(
    wflow_root: Union[str, Path],
    config_fn: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
    values_used_path: Union[str, Path, None] = None,
):
    """Update a wflow model with reservoirs, lakes and glaciers.

    Runs each method in ``config_fn`` individually, skipping any whose data is
    absent for the basin, and writes a structured sentinel for Snakemake
    tracking.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    config_fn : Union[str, Path]
        Path to the reservoirs/lakes/glaciers setup config.
    data_catalog : str
        Name of the data catalog to use.
    """
    # Lazy import: heavy plugin deps, only needed to touch the model. Keeps this
    # module importable (e.g. for unit tests) without hydromt installed/stubbed.
    from hydromt.error import NoDataException
    from hydromt_wflow import WflowSbmModel

    mod = WflowSbmModel(wflow_root, mode="r+", data_libs=data_catalog)

    with open(config_fn, "r") as f:
        config = yaml.safe_load(f) or {}

    results = _run_waterbody_methods(mod, config, (NoDataException, FileNotFoundError))

    if any(r["status"] == "ok" for r in results):
        with hydromt_progress("model"):
            # Everything but the forcing: rule 1.09 owns that file, and a bare
            # `write()` re-flushed it on every re-run into a duplicate under a
            # generated name. See `shared/wflow_write`.
            write_model_except_forcing(mod)
        mod.close()  # commits deferred staticmaps writes

    write_sentinel(
        join(wflow_root, "staticgeoms", "reservoirs_lakes_glaciers.txt"), results
    )

    if values_used_path is not None:
        write_values_used(values_used_path, results, config_fn)
    from blueearth_cst.shared.snake_utils import log_row

    for result in results:
        if result["status"] == "skipped":
            reason = " ".join(result["reason"].splitlines())
            details = (
                f"; setup outcomes → {values_used_path}" if values_used_path else ""
            )
            log_row(f"{result['method']} skipped: {reason}{details}", module="model")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            update_wflow_waterbodies_glaciers(
                wflow_root=os.path.dirname(sm.input.basin_nc),
                data_catalog=sm.params.data_catalog,
                config_fn=sm.params.config,
                values_used_path=sm.output.values_used,
            )
    else:
        update_wflow_waterbodies_glaciers(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            config_fn=join(os.getcwd(), "config", "wflow_update_waterbodies.yml"),
        )
