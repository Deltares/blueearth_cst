# -*- coding: utf-8 -*-
"""The WF2 projection figure family — one definition of what it contains.

**This module is the single source of the figure set.** Every declaration
downstream is derived from :func:`figure_relative_paths`: the Snakefile's
``WF2_TARGETS``, rule 2.06's ``figure_names`` promise, rule 2.07's outputs, the
``gather_logs`` / ``gather_benchmarks`` edges, the project-tree inventory and
``check_baseline``. That is deliberate — the defect this replaces is eight
figures written where three were declared, so five were invisible to Snakemake:
not cleaned on failure, not remade when deleted, and unusable as a dependency.
A set defined in one place cannot drift from the set that is written.

Why this module exists rather than the arrangement `dc40a22` used
-----------------------------------------------------------------
The reverted integration split the family across the two producers, which left
the page furniture (panel labels, the caveat line, the scenario palette) with no
home either of them could share. Both WF2 producers draw figures from this
family — ``get_change_climate_proj_summary`` draws the change-factor cloud from
the stage-B merge, ``plot_proj_timeseries`` draws the series and monthly
figures — so the family's contract, palette and furniture live here and both
import them. Nothing else changed about the design.

The adopted set (owner ruling, 2026-08-17)
------------------------------------------
Two annual overviews, two views of the change-factor cloud, and one monthly
change-factor figure per configured horizon. The combined cloud is emitted only
when more than one horizon is configured: it exists to answer *how far does the
cloud travel between horizons*, which is not a question a single horizon has.

Layout conventions, all of them owner rulings from 2026-08-11 and toolbox-wide
rather than WF2-specific:

* **No titles anywhere.** Panels carry ``a)`` / ``b)`` labels. What a title used
  to say goes where a journal figure keeps it — the variable and its unit in the
  y-label, and the horizon, reference window and trace counts in the caveat.
* **The WF1 page spec**: ``cartographic_map._publication_rc()`` at
  ``series_figure_size(...)``, ``layout="constrained"``, and the caveat carried
  by ``fig.supxlabel(..., wrap=True)`` — never ``tight_layout`` and never a
  hand-placed ``fig.text``, which does not clip but silently loses its tail.
* **Scenario is the only visual encoding.** No model or member gets its own
  colour, marker or line style, and no legend names one. The cloud annotates
  each point with its model name, which identifies a point without making model
  a visual channel.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

#: The two full-period overviews, in the order they appear in the set.
_OVERVIEW_PATHS = (
    "overview/annual-precipitation.png",
    "overview/annual-temperature.png",
)

#: The faceted change-factor cloud — always present, one panel per horizon.
CLOUD_FACETED_PATH = "overview/change-factor-cloud.png"

#: Every horizon on one pair of axes. Emitted only for a multi-horizon config;
#: see :func:`figure_relative_paths`.
CLOUD_COMBINED_PATH = "overview/change-factor-cloud-combined.png"

#: One per configured horizon, under its own window directory.
_MONTHLY_BASENAME = "monthly-change-factors.png"


def parse_horizon_period(period: Sequence[int] | str) -> tuple[int, int]:
    """Return one inclusive ``(start_year, end_year)`` horizon pair.

    R01 delivers ``future_horizons`` as lists (``[2070, 2090]``); pre-R01
    configs delivered comma-separated strings (``"2070, 2090"``). Both are
    accepted, because a project config written against either is still valid.
    """
    values = period.split(",") if isinstance(period, str) else list(period)
    if len(values) != 2:
        raise ValueError(f"horizon period must contain two years, got {period!r}")
    try:
        start, end = (int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"horizon years must be integers, got {period!r}") from exc
    if start > end:
        raise ValueError(f"horizon start must not exceed end, got {start}-{end}")
    return start, end


def sanitize_horizon_name(name: str) -> str:
    """Sanitize a configured horizon name into a portable directory name.

    A horizon name is free text in the config, and it reaches the filesystem
    here. Anything outside ``[a-z0-9]`` collapses to a single hyphen, so a name
    that would otherwise produce an unopenable path on win-64 is caught at parse
    time rather than at the moment a figure is written.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    if not slug:
        raise ValueError(f"horizon name has no portable characters: {name!r}")
    return slug


def horizon_directory(name: str, period: Sequence[int] | str) -> str:
    """Return ``<sanitized-name>-<start>-<end>`` for one horizon.

    The years are in the directory name, not only the horizon's label, because
    a project that redefines ``far`` from 2070-2090 to 2060-2080 would otherwise
    overwrite the old figure in place and leave nothing saying the window moved.
    """
    start, end = parse_horizon_period(period)
    return f"{sanitize_horizon_name(name)}-{start}-{end}"


def figure_relative_paths(
    horizons: Mapping[str, Sequence[int] | str],
) -> list[str]:
    """Every WF2 figure path relative to ``plots/``, in stable order.

    The order is the reading order of the set — overviews, clouds, then one
    monthly figure per horizon in configured order — and it is stable because
    ``figure_names`` is threaded into rule 2.06 as a `params:` value, so a
    reordering would re-trigger the rule for no reason.

    The combined cloud appears only for a multi-horizon config. It answers "how
    far does the cloud travel between horizons"; with one horizon it would be
    the faceted figure drawn again under a second name, which is a second path
    to the same fact.
    """
    if not horizons:
        raise ValueError("no horizons configured; WF2 cannot name its figure set")

    window_dirs = [horizon_directory(name, period) for name, period in horizons.items()]
    duplicates = {d for d in window_dirs if window_dirs.count(d) > 1}
    if duplicates:
        raise ValueError(
            "configured horizon names and years resolve to duplicate paths: "
            + ", ".join(sorted(duplicates))
        )

    paths = [*_OVERVIEW_PATHS, CLOUD_FACETED_PATH]
    if len(window_dirs) > 1:
        paths.append(CLOUD_COMBINED_PATH)
    paths.extend(
        f"windows/{window_dir}/{_MONTHLY_BASENAME}" for window_dir in window_dirs
    )
    return paths


def monthly_figure_paths(
    horizons: Mapping[str, Sequence[int] | str],
) -> dict[str, str]:
    """Map each configured horizon NAME to its monthly figure's relative path.

    The producer needs the horizon a path belongs to; the Snakefile only needs
    the paths. Deriving both from :func:`horizon_directory` is what keeps them
    from disagreeing about which window a figure describes.
    """
    return {
        name: f"windows/{horizon_directory(name, period)}/{_MONTHLY_BASENAME}"
        for name, period in horizons.items()
    }
