"""Writing a Wflow model back to disk without touching the forcing.

Rule 1.09 (``add_climate_forcing``) owns the forcing netCDF: it builds it, it
declares it as a Snakemake output, and its name is on the file. Rules 1.07
(``add_reservoirs_lakes_glaciers``) and 1.09 (``declare_gauges_and_outputs``) change
staticmaps, geoms and the TOML and have no business writing forcing at all --
but ``WflowBaseModel.write()`` flushes every component, so on a re-run they
wrote it anyway.

What that cost, measured on a real re-run (2026-09-05, project dir
``.tmp/test_run``). ``WflowForcingComponent.write`` opens with
``if self._data_is_empty()``, and that test reads ``self.data`` -- the PUBLIC
property, which lazily calls ``read()`` whenever the root is in reading mode.
``r+`` is a reading mode, so merely asking whether the forcing was empty loaded
it from ``input.path_forcing``. It was then non-empty, so hydromt wrote it out;
the target already existed and overwriting was off, so it warned, invented
``inmaps_<precip>_<temp>d_<pet_method>_<press>_<start>_<end>.nc``
(``components/forcing.py:182``) and repointed ``input.path_forcing`` at that
duplicate (same file, line 265). The model directory was left holding 3.24 MB
of duplicate beside 3.52 MB of real forcing, declared by no rule and cleaned up
by nothing. Forcing scales with basin area times run length, so on a production
basin it is the largest single file the build writes, duplicated once per
re-run of either rule.

On a FRESH build there is no file to read, so the component stays empty and
hydromt logs ``Write forcing skipped: dataset is empty`` -- which is why this
never showed up on a clean run.

Rule 1.09 runs after both and repoints the key back, so a COMPLETE run ends
with a correct TOML. The window is what makes this worth fixing rather than
tidying: between rule 1.08 finishing and rule 1.09 rewriting the forcing, the
model config names a file Snakemake does not track. A run that stops there -- a
failure, an interrupt, ``--until``, or rule 1.09 judged up to date -- leaves it
that way, and rule 1.13 would then run Wflow against it.

**Why the sequence is replicated rather than parameterised.** ``hydromt``'s
base ``Model.write`` takes a ``components`` list, and naming every component
except ``forcing`` would have been the whole fix. ``WflowBaseModel`` overrides
``write`` with its own signature (``wflow_base.py:1716``) -- filenames only, no
component selection -- so that call raises ``TypeError``. Verified the hard way
on 2026-09-05; read the plugin's override, not the base class.

**Why not empty the component instead.** Making ``_data_is_empty()`` true would
be the smaller change, but there is no public way to do it: ``set()`` refuses
data without a ``time`` dimension (``components/forcing.py:288``), and the only
other route is assigning the private ``_data``. Replicating a documented
sequence of public component writes keeps this to API hydromt supports.

**The cost of replicating, and what pays it.** This now has to stay in step
with ``WflowBaseModel.write``, and drift would be SILENT -- a component hydromt
adds would simply stop being written, and the run would still succeed.
``tests/test_wflow_write.py`` closes that by reading the plugin's own source and
asserting its component writes are exactly ours plus the forcing, so a hydromt
upgrade that changes the sequence fails a test instead of quietly dropping an
artifact.
"""

from __future__ import annotations

from pathlib import Path

#: The one component these rules must not flush, under the key hydromt registers
#: it with (``hydromt_wflow/wflow_base.py:97``).
FORCING_COMPONENT = "forcing"

#: hydromt's own default, repeated here because the calls below pass it
#: explicitly rather than relying on a default that could move.
GEOMS_FOLDER = "staticgeoms"


def write_model_except_forcing(model) -> None:
    """``WflowBaseModel.write()`` with the forcing step left out.

    Step for step the plugin's own ``write`` (``wflow_base.py:1750-1767``), in
    its order and with its defaults, minus ``self.forcing.write(...)``. The
    config is written LAST for the reason hydromt states there: the other write
    methods can still set config values.

    Two things the plugin does are deliberately NOT replicated.

    Its opening ``Write model data to <root>`` row. Restoring it was tried and
    reverted: a record from a ``blueearth_cst`` logger does not reach the
    console at all (hydromt's rows arrive through a handler hydromt installs on
    its OWN logger, bound to the tee'd stdout), so the row would only have
    landed by binding ``logging.getLogger("hydromt_wflow.wflow_base")`` and
    signing hydromt's name to our output. It is no loss: rules 1.07 and 1.08
    are not silent without it -- a 1.09 run prints thirteen rows naming every
    grid, geom and config file as it is written, and a generic "writing the
    model" above them is the repetition this console work has been removing.

    Its ``is_writing_mode()`` guard, which warns and returns early. Both call
    sites open the model ``r+``, so the guard is unreachable there, and
    swallowing a misconfigured mode silently is the wrong failure for a rule
    that has already done its work.
    """
    model.write_data_catalog()
    # hydromt does this before writing the config: it forces the default
    # template to be read if nothing has set the config yet.
    _ = model.config.data
    model.staticmaps.write()
    model.staticmaps.write_region(
        filename=str(Path(GEOMS_FOLDER) / "region.geojson"), to_wgs84=True
    )
    model.geoms.write(folder=GEOMS_FOLDER)
    # `model.forcing.write()` deliberately absent -- see the module docstring.
    model.tables.write()
    model.states.write()
    model.config.write()
