"""Size WF3's Wflow batches against the disk they will actually fill.

Rule 3.15 runs ``B`` Wflow members per Julia session. Both of the sweep's
``temp()`` classes -- rule 3.14's per-member forcing NC and 3.15's own
per-member outstates NC -- are held for a whole batch rather than a whole
member, and Snakemake keeps ``p`` batch jobs in flight at ``-c N``. So peak
transient disk is ``p x B x (forcing + state)``: raising ``B`` for throughput
buys that throughput with disk.

P3-3 design section 6.1 calls this the **binding** constraint on large
``RLZ_NUM x ST_NUM`` runs, and the landed default did not implement it. It was
``ceil(K / -c N)`` -- the *parallelism* ceiling alone -- which scales ``B`` UP
with sweep size and therefore grows peak disk as the sweep grows, backwards
from what the disk ceiling asks. Commit ``3392587`` bounded the blast radius
with a constant ``batch_size_max`` (default 8), which is a clamp, not a disk
computation: on a real basin ``8`` can still be tens of gigabytes, and nothing
consulted the free space it was about to spend.

This module supplies the missing term.

Estimating a member BEFORE it exists
------------------------------------
The obvious anchor is a per-member forcing NC, and it is unavailable: those are
``temp()`` outputs that do not exist at parse time and are reclaimed as the run
proceeds. Modelling the size instead (grid cells x run length x variable count
x dtype) means predicting NetCDF's compressed size, which no arithmetic here
can do honestly.

So the estimate is **measured, not modelled**, off two artifacts WF1 leaves on
disk permanently -- the same model, the same grid, the same writer:

===============  ==================================================  ============
quantity         anchor                                              scaling
===============  ==================================================  ============
forcing          ``<basin>/forcing/inmaps_historical.nc``             by timestep
state            ``<basin>/run_default/outstate/outstates.nc``        none
===============  ==================================================  ============

The forcing anchor works because both files carry the same three variables on
the same grid at the same dtype through the same zlib level, chunked one
timestep at a time -- so bytes-per-timestep is a property of the MODEL, not of
the window, and a member's size is that constant times its own timestep count.
Measured on the rapid fixture (2026-08-18): the historical file is 1379
bytes/timestep and a member's is 1378, a 0.07 % difference; predicting the
4.530 MB member from the 3.527 MB historical file lands within 0.07 %.

The state anchor needs no scaling at all: outstates is a single snapshot of the
model's state variables, so its size is set by the grid and is independent of
how long the run was.

Both anchors are the WF1 model's own files, so they cannot describe a different
grid than the members will use. If either is missing -- a fresh project, a
``--dry-run`` before WF1 has ever run -- the estimate is simply unavailable and
the disk ceiling does not apply. **This is a safety cap, so it never raises and
never blocks**: an unavailable estimate degrades to the previous behaviour
rather than becoming a new way for a run to fail.
"""

import math
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from blueearth_cst.experiment.forcing_window import forcing_window_years

#: WF1 artifacts the estimate is anchored on, relative to the wflow basin dir.
#: Both are ordinary persisted outputs -- neither is ``temp()`` -- which is the
#: whole reason they can be read at WF3 parse time.
FORCING_ANCHOR = ("forcing", "inmaps_historical.nc")
STATE_ANCHOR = ("run_default", "outstate", "outstates.nc")

#: Binary GB (GiB). The unit a file manager shows for free space, which is the
#: number a user compares `disk_headroom_gb` against when choosing it.
BYTES_PER_GB = 1 << 30


def _gb(nbytes):
    """Render bytes as a short ``12.3 GB`` string for one console row."""
    if nbytes is None:
        return "?"
    value = nbytes / BYTES_PER_GB
    if value >= 10:
        return f"{value:.0f} GB"
    if value >= 0.1:
        return f"{value:.1f} GB"
    return f"{nbytes / (1 << 20):.0f} MB"


@dataclass(frozen=True)
class MemberFootprint:
    """The transient disk one stress-test member holds while its batch runs."""

    forcing_bytes: int
    state_bytes: int
    #: Where the numbers came from, for the row that reports the decision.
    basis: str

    @property
    def total_bytes(self):
        return self.forcing_bytes + self.state_bytes


@dataclass(frozen=True)
class BatchSizing:
    """The chosen ``B``, and enough context to say why it is that number."""

    batch_size: int
    #: Which ceiling produced it: ``explicit``, ``parallelism``, ``batch_size_max``
    #: or ``disk``. Reported, because a ``B`` the user did not ask for should be
    #: able to name the constraint that chose it.
    bound_by: str
    peak_bytes: int | None = None
    headroom_bytes: int | None = None
    footprint: MemberFootprint | None = None
    #: Set only when even ``B = 1`` cannot fit the headroom -- the one case the
    #: cap cannot fix by shrinking, so it has to be said out loud.
    warning: str | None = None

    def summary(self):
        """One short ``key value`` row for the WF3 run header."""
        text = f"B={self.batch_size} ({self.bound_by})"
        if self.peak_bytes is not None and self.headroom_bytes is not None:
            text = (
                f"{text}, peak {_gb(self.peak_bytes)}"
                f" of {_gb(self.headroom_bytes)} headroom"
            )
        return text


def _netcdf_timesteps(path):
    """Return the length of ``path``'s time dimension, or ``None``.

    ``netCDF4`` is imported HERE rather than at module scope: it costs ~0.75 s
    to import and the Snakefile parses on every invocation, including
    ``--unlock`` and ``--dry-run``. Reading the dimension itself is ~9 ms.

    Every failure returns ``None`` -- a missing backend, an unreadable file, a
    forcing file with no time dimension. The caller treats that as "no estimate
    available", which is the safe direction for a cap.
    """
    try:
        import netCDF4
    except Exception:  # noqa: BLE001 -- absent backend must not break a parse
        return None
    try:
        with netCDF4.Dataset(str(path)) as ds:
            steps = ds.dimensions["time"].size
    except Exception:  # noqa: BLE001 -- unreadable anchor is "no estimate"
        return None
    return int(steps) or None


def _days_in_years(startyear, endyear):
    """Whole days from ``startyear``-01-01 to ``endyear``-12-31, inclusive.

    Counted through :mod:`datetime` rather than as ``years x 365`` so leap days
    are in the number -- on a 30-year window that is a week of timesteps.
    """
    return (date(endyear, 12, 31) - date(startyear, 1, 1)).days + 1


def measure_member_footprint(basin_dir, horizontime_climate, run_length):
    """Estimate one member's ``temp()`` footprint, or ``None`` if it cannot be.

    ``None`` is an ordinary outcome, not an error: WF1 may not have run yet.
    """
    basin = Path(basin_dir)
    forcing_anchor = basin.joinpath(*FORCING_ANCHOR)
    state_anchor = basin.joinpath(*STATE_ANCHOR)
    if not forcing_anchor.is_file() or not state_anchor.is_file():
        return None

    hist_steps = _netcdf_timesteps(forcing_anchor)
    if not hist_steps:
        return None

    member_steps = _days_in_years(
        *forcing_window_years(horizontime_climate, run_length)
    )
    try:
        hist_bytes = os.path.getsize(forcing_anchor)
        state_bytes = os.path.getsize(state_anchor)
    except OSError:
        return None

    forcing_bytes = int(round(hist_bytes / hist_steps * member_steps))
    return MemberFootprint(
        forcing_bytes=forcing_bytes,
        state_bytes=int(state_bytes),
        basis=(
            f"{hist_bytes / hist_steps:.0f} B/timestep x {member_steps} steps"
            f" + {state_bytes} B state"
        ),
    )


def disk_headroom_bytes(path, fraction, headroom_gb=None):
    """How many bytes the batched footprint may occupy, or ``None`` if unknown.

    ``headroom_gb`` is an absolute budget and wins outright. Absent, the budget
    is ``fraction`` of what is FREE right now -- a share rather than a constant,
    because the only honest default is one that adapts to the machine. A
    constant generous enough for a workstation would be a wrong answer on a
    laptop and vice versa.

    Walks up to the nearest existing ancestor, so a project directory that the
    run is about to create still yields the free space of the volume it will
    land on.
    """
    if headroom_gb is not None:
        return max(1, int(float(headroom_gb) * BYTES_PER_GB))

    probe = Path(path).absolute()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(str(probe)).free
    except OSError:
        return None
    return max(1, int(free * float(fraction)))


def resolve_batch_size(
    member_count,
    cores,
    batch_size_max,
    explicit=None,
    footprint=None,
    headroom_bytes=None,
):
    """Choose ``B`` as the smallest of the ceilings that apply.

    ``explicit`` (the project's own ``batch_size``) wins outright and is not
    clamped -- an operator who names a batch size has said something this
    estimate cannot know, and silently overriding it would make the key a lie.
    The peak is still computed and reported for that case, so an explicit choice
    that overruns the disk is visible rather than merely unchecked.
    """
    cores = max(1, int(cores))
    member_count = max(1, int(member_count))

    if explicit is not None:
        chosen, bound_by = int(explicit), "explicit"
    else:
        # Ceiling 1, parallelism: enough batches to keep the cores busy.
        ceilings = {
            "parallelism": max(1, math.ceil(member_count / cores)),
            "batch_size_max": int(batch_size_max),
        }
        # Ceiling 2, disk -- but only when the WHOLE sweep cannot fit anyway.
        # Peak is `min(K, p x B) x per_member`, not `p x B x per_member`: once
        # every member is resident at once the batch structure has stopped
        # mattering and the cap degenerates to the sweep's own footprint, which
        # no choice of B can reduce. P3-3 GN-3 measured exactly that degenerate
        # case on the fixture (12 of 12 forcing NCs resident at B=4/p=3).
        if footprint is not None and headroom_bytes is not None:
            per_member = max(1, footprint.total_bytes)
            if member_count * per_member > headroom_bytes:
                ceilings["disk"] = max(1, headroom_bytes // (cores * per_member))
        bound_by = min(ceilings, key=lambda k: (ceilings[k], k != "disk"))
        chosen = max(1, ceilings[bound_by])

    peak = headroom = warning = None
    if footprint is not None:
        per_member = footprint.total_bytes
        peak = min(member_count, cores * chosen) * per_member
        headroom = headroom_bytes
        if headroom is not None and peak > headroom:
            # Reachable two ways, and both deserve the row: an explicit
            # `batch_size` that overruns, or a per-member footprint so large
            # that even one member per core does not fit. The cap cannot shrink
            # below B=1, so this is reported rather than enforced.
            warning = (
                f"WF3 batch peak disk {_gb(peak)} exceeds the"
                f" {_gb(headroom)} headroom at B={chosen}"
                f" ({_gb(per_member)}/member x {min(member_count, cores * chosen)}"
                f" resident); free disk or lower run_length/cores"
            )

    return BatchSizing(
        batch_size=chosen,
        bound_by=bound_by,
        peak_bytes=peak,
        headroom_bytes=headroom,
        footprint=footprint,
        warning=warning,
    )
