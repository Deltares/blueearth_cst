"""The WF3 forcing window — the year span every stress-test member is run over.

Its own module rather than living beside its main consumer in
``downscale_climate_forcing``, because that module imports ``hydromt_wflow`` at
module scope. ``generate_scenarios.smk`` needs this window at PARSE time to size
Wflow batches against the disk (:mod:`blueearth_cst.experiment.batch_sizing`),
and a Snakefile cannot pay a hydromt import to learn two integers.

**There is no arithmetic left here** (`C-72`). :func:`forcing_window` renders
the ISO pair rule 3.14 hands to hydromt, from the year pair the config now
declares directly (`C-67`).

What was deleted, and why it cannot come back: ``forcing_window_years()``
derived that pair from a horizon year and a run length, snapping ``ceil``
backwards and ``np.round`` forwards. The batch estimator counted days between
the two integers rather than multiplying the run length by 365, because the
snapping made the span ``run_length + 1`` calendar years whenever the halves
went outward. `C-67` moved the window into the config, which removed both the
derivation and the trap — and the derivation could not have been kept anyway,
since it was not single-valued for an odd run length.
"""


def forcing_window(startyear, endyear):
    """Render an INCLUSIVE year pair as the ``(starttime, endtime)`` ISO pair.

    As hydromt and the run TOML want it. Since `C-67` the two years are
    DECLARED, as ``simulation_window: {start, end}``, rather than derived — so
    this formats a window, it does not compute one. See the module docstring
    for what `C-72` removed.

    ``endyear`` renders as MIDNIGHT on 31 December, not end-of-day. That is the
    string the shipped configs carry, and reproducing it verbatim is the point:
    changing it would move every WF3 run's end boundary by a day.
    """
    return f"{startyear}-01-01T00:00:00", f"{endyear}-12-31T00:00:00"
