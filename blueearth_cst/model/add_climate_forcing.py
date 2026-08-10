"""Assemble the hydromt forcing recipe and apply it — rules 1.07 + 1.08, merged.

`dev/followups-archive.md` `[R10-1]`. Rule 1.07 wrote a `steps:` YAML whose **only**
consumer was rule 1.08, which ran `hydromt update wflow_sbm -i` against it. Two
rules, one job — and a recipe that never leaves the pair needs no rule name of
its own, which is why 1.07's R10 rename was withdrawn rather than replaced.

**Why a `script:` and not a `shell:`.** Snakemake allows one of the two per rule,
and the halves were one of each. Driving hydromt's CLI from Python keeps the
command byte-identical to what rule 1.08 issued, which is what makes this merge
behaviour-preserving; calling hydromt's Python API instead would have been a
second change wearing the same commit.

**Why the output is streamed rather than inherited.** `tee_to_log` redirects
``sys.stdout``/``sys.stderr`` at the Python level — it does not touch a child
process's file descriptors. A bare ``subprocess.run`` inheriting stdout would
therefore write hydromt's ``-vv`` output to the console and leave the rule's log
part empty, silently losing what the `shell:` rule used to capture through
``run_logged``. Reading the pipe and re-printing puts it back inside the tee, and
line-by-line so a long update stays visible while it runs.

The recipe builder itself is untouched and still lives in
`blueearth_cst/shared/setup_time_horizon.py`, with `tests/test_setup_time_horizon.py`
covering it directly — the merge moves the *invocation*, not the logic.
"""

# NO `from __future__ import annotations` here: Snakemake's `script:` directive
# prepends its own preamble, so a future import is no longer the first statement
# and raises at rule run time. Every `script:` module in this repo omits it.
import os
import subprocess
from pathlib import Path
from typing import Sequence, Union

from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
    prepare_clim_data_catalog,
)
from blueearth_cst.shared.setup_time_horizon import prep_hydromt_update_forcing_config


def _as_list(data_catalog):
    """The catalogs as a plain list, whether one path or several came in."""
    if isinstance(data_catalog, (list, tuple)):
        return [os.fspath(item) for item in data_catalog]
    return [os.fspath(data_catalog)]


def _catalog_flags(data_catalog):
    """Render one or several catalogs as repeated ``-d`` flags.

    The shell rule this replaces interpolated ``-d "{DATA_SOURCES}"`` once, which
    is correct for the single-path case every shipped config uses and would have
    passed a Python list repr for anything else. Repeating the flag is what
    hydromt actually accepts, so a list now works rather than failing obscurely.
    """
    if isinstance(data_catalog, (list, tuple)):
        catalogs = [os.fspath(item) for item in data_catalog]
    else:
        catalogs = [os.fspath(data_catalog)]
    flags = []
    for catalog in catalogs:
        flags += ["-d", catalog]
    return flags


def _run_streaming(command: Sequence[str]) -> None:
    """Run a command, re-printing its output so ``tee_to_log`` captures it."""
    print(f"$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with process.stdout:
        for line in process.stdout:
            print(line.rstrip("\n"), flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"hydromt update failed with exit code {code}: {' '.join(command)}"
        )


def add_climate_forcing(
    starttime: str,
    endtime: str,
    clim_source: str,
    basin_dir: Union[str, os.PathLike],
    data_catalog,
    forcing_yml: Union[str, os.PathLike],
    climate_nc: Union[str, os.PathLike, None] = None,
    store_catalog: Union[str, os.PathLike, None] = None,
) -> None:
    """Write the forcing recipe, then apply it to the model with hydromt.

    When ``climate_nc`` is given the forcing is built from the CLIMATE STORE --
    the extraction rule 1.04 already produced -- instead of re-reading the
    global dataset from the catalog. That read was the second full pass over the
    same source in one workflow, and the store is a basin-sized clip of it.

    The store is handed to hydromt the way every other source is: as a catalog
    entry, generated here from the real source's entry so units and renames are
    inherited rather than hand-written (``prepare_clim_data_catalog`` drops the
    unit adapters, which is correct -- the extraction already applied them, and
    keeping them would convert twice).
    """
    store_source = None
    if climate_nc is not None:
        prepare_clim_data_catalog(
            fns=[climate_nc],
            data_libs_like=data_catalog,
            source_like=clim_source,
            fn_out=store_catalog,
        )
        # `prepare_clim_data_catalog` keys each entry on the file stem.
        store_source = Path(climate_nc).stem
        data_catalog = [*_as_list(data_catalog), store_catalog]

    prep_hydromt_update_forcing_config(
        starttime=starttime,
        endtime=endtime,
        fn_yml=forcing_yml,
        precip_source=clim_source,
        wflow_root=basin_dir,
        store_source=store_source,
    )
    _run_streaming(
        [
            "hydromt",
            "update",
            "wflow_sbm",
            os.fspath(basin_dir),
            "-i",
            os.fspath(forcing_yml),
            *_catalog_flags(data_catalog),
            "-vv",
        ]
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            add_climate_forcing(
                starttime=sm.params.starttime,
                endtime=sm.params.endtime,
                clim_source=sm.params.clim_source,
                basin_dir=sm.params.basin_dir,
                data_catalog=sm.params.data_catalog,
                forcing_yml=sm.output.forcing_yml,
                climate_nc=sm.input.climate_nc,
                store_catalog=sm.output.store_catalog,
            )
            log_row(
                f"Added climate forcing {sm.params.starttime}..{sm.params.endtime} "
                f"(clim_source={sm.params.clim_source}) -> {sm.output.forcing_path}",
                module="forcing",
            )
