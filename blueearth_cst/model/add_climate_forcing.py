"""Assemble the hydromt forcing recipe and apply it — rules 1.07 + 1.08, merged.

`dev/followups.md` `[R10-1]`. Rule 1.07 wrote a `steps:` YAML whose **only**
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
import sys
from typing import Sequence, Union

from blueearth_cst.shared.setup_time_horizon import prep_hydromt_update_forcing_config


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
) -> None:
    """Write the forcing recipe, then apply it to the model with hydromt."""
    prep_hydromt_update_forcing_config(
        starttime=starttime,
        endtime=endtime,
        fn_yml=forcing_yml,
        precip_source=clim_source,
        wflow_root=basin_dir,
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
            )
            log_row(
                f"Added climate forcing {sm.params.starttime}..{sm.params.endtime} "
                f"(clim_source={sm.params.clim_source}) -> {sm.output.forcing_path}",
                module="forcing",
            )
