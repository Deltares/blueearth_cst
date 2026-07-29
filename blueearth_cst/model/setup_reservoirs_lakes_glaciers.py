"""Update a wflow model with reservoirs, lakes and glaciers.

This runs as a **separate rule** (rather than inside create_model's build
config) because hydromt 1.3's ``hydromt build`` cannot tolerate per-method
no-data for reservoirs/lakes/glaciers — a basin may legitimately lack any of
them. Running each method individually here lets a NoDataException skip only
that one method.

Removal trigger: fold these methods back into create_model's build config and
delete this module once upstream hydromt handles per-method no-data gracefully
during build. (Tracked upstream; no issue number recorded yet.)
"""
import os
from os.path import join
from pathlib import Path
from typing import Union

import yaml


def _run_waterbody_methods(mod, config, no_data_errors):
    """Run each configured method on ``mod``, capturing per-method outcome.

    Returns a list of ``{"method", "status", "reason"}`` dicts where status is
    ``"ok"`` or ``"skipped"`` (the method's data source was absent for this
    basin — a legitimate outcome, not a failure).
    """
    results = []
    for method, kwargs in config.items():
        kwargs = kwargs or {}
        try:
            getattr(mod, method)(**kwargs)
            results.append({"method": method, "status": "ok", "reason": ""})
        except no_data_errors as error:
            results.append(
                {"method": method, "status": "skipped", "reason": str(error)}
            )
    return results


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
    from hydromt_wflow import WflowSbmModel
    from hydromt.error import NoDataException

    mod = WflowSbmModel(wflow_root, mode="r+", data_libs=data_catalog)

    with open(config_fn, "r") as f:
        config = yaml.safe_load(f) or {}

    results = _run_waterbody_methods(
        mod, config, (NoDataException, FileNotFoundError)
    )

    if any(r["status"] == "ok" for r in results):
        mod.write()
        mod.close()  # commits deferred staticmaps writes

    write_sentinel(
        join(wflow_root, "staticgeoms", "reservoirs_lakes_glaciers.txt"), results
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            update_wflow_waterbodies_glaciers(
                wflow_root=os.path.dirname(sm.input.basin_nc),
                data_catalog=sm.params.data_catalog,
                config_fn=sm.params.config,
            )
    else:
        update_wflow_waterbodies_glaciers(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            config_fn=join(os.getcwd(), "config", "wflow_update_waterbodies.yml"),
        )
