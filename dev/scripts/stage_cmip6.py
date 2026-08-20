"""Stage CMIP6 slices for one region, outside any project.

What it does
------------
The same thing WF2's rule 2.04 does — open the remote CMIP6 store, clip to a
polygon plus a buffer, slice the acquisition window, write a small netCDF — but
driven by a YAML file and writing wherever you point it, with no project and no
Snakemake run.

**The slices are WF2-cache-compatible.** Each file is stamped with the same
``cst_raw_digest`` the pipeline computes, so dropping one into a project's
``data/climate/projections/<clim_project>/raw/`` makes WF2 treat it as a cache
hit and skip the download. That is the point of the tool: WF2's raw cache is
project-scoped, so every new project on the same basin re-pays the remote open,
measured at **1142 s per source** (``fetch_gcm_raw`` docstring). Stage once
here, reuse everywhere.

For that to work the REGION must be the same polygon the project uses — the
digest covers it (``series_identity.region_fingerprint``). A different polygon
is a different slice by design, and its file will simply be re-fetched by the
project rather than silently accepted.

Why this is not a `stage_data.py` dataset type
----------------------------------------------
``stage_data.py`` mirrors a directory tree under one ``source_root`` by joining
``path``. CMIP6 is ``(model, scenario, member, variable)`` tuples resolved
through a GENERATED catalog with ``{member}`` / ``{variable}`` placeholders, on
an object store. The addressing has nothing in common, so this is a sibling
tool rather than a new ``type:``.

There is exactly ONE fetch implementation: ``fetch_gcm_raw.fetch_raw_slice``,
which WF2's rule 2.04 also calls. This module contributes the config and the
digest recipe, never a second way to read the store.

Configuration
-------------
All knobs live in a YAML file (default: ``dev/scripts/stage_cmip6.yml``)::

    region: C:/basins/gabon/region.geojson
    target_root: C:/data/cmip6_slices
    catalog: config/catalogs/cmip6_data.yml
    store_index: config/catalogs/cmip6_store_index.json
    clim_project: cmip6
    buffer_cells: 1
    models:    [NOAA-GFDL/GFDL-ESM4, INM/INM-CM4-8]
    scenarios: [historical, ssp245]
    members:   [r1i1p1f1]
    variables:                       # POST-rename names, not pr/tas
      precip: {units: kg m-2 s-1}
      temp:   {units: K}

The acquisition window is NOT a knob: it is derived per scenario by
``series_identity.acquisition_window``, exactly as the Snakefile derives it, so
a staged slice covers the window the pipeline expects.

Usage
-----
    python dev/scripts/stage_cmip6.py
    python dev/scripts/stage_cmip6.py --config my_basin.yml
    python dev/scripts/stage_cmip6.py --dry-run          # list jobs, fetch none
    python dev/scripts/stage_cmip6.py --models NOAA-GFDL/GFDL-ESM4
    python dev/scripts/stage_cmip6.py --workers 8        # 4 slices at once by default
    python dev/scripts/stage_cmip6.py --workers 1        # serial, for a real traceback

The requested cross product is PRE-FILTERED against the catalog, which already
records which (model, scenario) pairs exist and which members each carries. Much
of a naive cross product is not real: of 65 models with `historical`, 19 never
published `ssp245`, and 70 of 289 entries do not offer `r1i1p1f1`. Those are
refused before a worker starts, with the reason -- and for a missing member, the
members the entry DOES have. Nothing is auto-substituted: choosing a different
realisation is a methodological decision, not a convenience.

Slices are staged in parallel PROCESSES because the cost is the remote open, not
compute -- see `DEFAULT_WORKERS` for the measurement and for why threads are the
wrong tool here. Progress is reported in completion order, and one unavailable
source is reported without ending the run.

The closing report separates what the catalog refused, what could not be read
(a few models publish `Amon` on a non-rectilinear grid that hydromt's raster
accessor cannot take), and what simply failed -- because those call for
different actions.

Console output
--------------
The surface is `stage_data.py`'s, through the same vendored `console.py`:
Description / Parameters / Stage / Total regions, the `+ = - x` glyph
vocabulary coloured on the glyph, the 0/2/4/6 indent ladder, and one verdict
line. The two tools are read by the same person on the same afternoon, so a
reader should not have to relearn where the counts are. What is NOT borrowed is
the shape of what this tool has to say: the `[n/total]` completion counter (a
slice can take twenty minutes, so a silent console is a defect) and the
three-way failure split above, which `stage_data.py` has no equivalent for.

Headings are bold Title Case with NO decoration, and the only rule drawn is the
one above `Total` -- the hard break before the final aggregate. They read
`━━ SECTION ━━` in any copy predating `console.py` 0.10.0; check
`console.__version__` against the `console-formatting` skill before assuming
the older shape is current.

What the FETCH says is not this tool's console surface and is not reimplemented
here. Each slice runs inside WF2's own `tee_to_log`, so hydromt's records are
compacted, the mute table drops the rows that state a property of the run, and
what remains is written to `<target_root>/logs/<key>.log` as well as shown --
identically to what the same fetch prints under Snakemake. Two consequences
worth stating: a console fix belongs in `shared/snake_utils.py`, where both
paths get it, and a slice that failed has a full log part with its traceback in
it, which the closing report names.

Not part of a run: this is an authoring/staging helper
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from blueearth_cst.projections import series_identity as _si  # noqa: E402
from blueearth_cst.projections.fetch_gcm_raw import (  # noqa: E402
    AMBIGUOUS_VERSION_PHRASE,
    fetch_raw_slice,
    resolve_entry_name,
)
from blueearth_cst.shared.snake_utils import tee_to_log  # noqa: E402

# `console.py` sits beside this file, and `stage_data.py` reaches it exactly
# this way. The two staging tools share one console vocabulary, so they share
# its implementation rather than each growing colour helpers of its own.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from console import (  # noqa: E402
    banner,
    bold,
    cyan,
    dim,
    fmt_path,
    glyph,
    green,
    pad,
    red,
    rule,
    section_banner,
    yellow,
)

CONFIG_DEFAULT = Path(__file__).resolve().parent / "stage_cmip6.yml"

#: Mirrors `analyze_projections.smk`'s REGION_BUFFER_CELLS. Duplicated rather
#: than imported because a Snakefile is not an importable module. It IS a digest
#: component, so a value differing from the pipeline's yields slices the pipeline
#: will not accept -- which `tests/test_stage_cmip6.py`'s fixture case catches,
#: since it recomputes against files WF2 actually wrote.
#:
#: A COUNT OF GRID CELLS, not degrees: hydromt spends `buffer` as resolution
#: multiplicity. Named `DEFAULT_BUFFER_DEGREES` until t2608182238; the value is
#: unchanged, so no slice this tool already staged is wrong -- but they were
#: written under schema 4 and the pipeline now demands 5, so they re-stage.
DEFAULT_BUFFER_CELLS = 1

#: Mirrors the `clim_project` config key. Only `cmip6` has a generated catalog
#: today, but the grammar below is the Snakefile's, not a cmip6 literal.
DEFAULT_CLIM_PROJECT = "cmip6"


def series_key(clim_project, model, experiment, member):
    """The pipeline's filename stem for one slice.

    Verbatim from `analyze_projections.smk`'s `SERIES` comprehension — the
    slash in a model id becomes an underscore. Staged files carry this name so
    they can be copied into `raw/` unchanged.
    """
    return f"{clim_project}_{model.replace('/', '_')}_{experiment}_{member}"


def catalog_entry_name(clim_project, model, experiment):
    """The catalog entry, `{member}` placeholder still unresolved.

    Also verbatim from the Snakefile: the generated catalog expands the member
    at generation time, so the entry NAME carries it and `fetch_raw_slice`
    resolves it through the catalog's own grammar.
    """
    return f"{clim_project}_{model}_{experiment}_{{member}}"


def digest_components(cfg, model, experiment, member):
    """Rebuild the pipeline's raw-digest components for one slice.

    This is the one place the tool must agree with
    `analyze_projections.smk::series_digest_components`, and
    `tests/test_stage_cmip6.py` pins that agreement rather than trusting it.

    `reducer_module_hash` is passed as the empty string on purpose:
    `raw_components` strips it, and inventing a value here would suggest the
    raw slice depends on a reduction it has never seen.
    """
    entry_name = catalog_entry_name(cfg["clim_project"], model, experiment)
    entry = _si.load_catalog_entry(cfg["catalog"], entry_name)
    return _si.raw_components(
        _si.digest_components(
            catalog_entry=entry_name,
            entry=entry,
            members=[member],
            pins_by_member={
                member: _si.load_pins(cfg["store_index"], entry_name, member)
            },
            buffer_cells=cfg["buffer_cells"],
            variable_spec=list(cfg["variables"]),
            experiment=experiment,
            reducer_module_hash="",
        )
    )


def load_config(path):
    """Read the YAML and apply defaults; raise on anything required and absent."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    for key in ("region", "target_root", "models", "scenarios", "members", "variables"):
        if not cfg.get(key):
            raise SystemExit(f"{path}: '{key}' is required and missing or empty")

    cfg.setdefault("clim_project", DEFAULT_CLIM_PROJECT)
    cfg.setdefault("buffer_cells", DEFAULT_BUFFER_CELLS)
    cfg.setdefault("catalog", str(_REPO_ROOT / "config/catalogs/cmip6_data.yml"))
    cfg.setdefault(
        "store_index", str(_REPO_ROOT / "config/catalogs/cmip6_store_index.json")
    )
    if not Path(cfg["region"]).is_file():
        raise SystemExit(f"{path}: region not found: {cfg['region']}")
    return cfg


def catalog_availability(cfg):
    """`(model, scenario) -> [members]`, read from the generated catalog.

    The catalog is an inventory of what the bucket actually holds: one entry
    per (model, scenario) that has BOTH `pr` and `tas` at `Amon`, each listing
    exactly the members that exist. So it can answer "is this combination
    real?" without a single network call, which is what lets `plan` refuse an
    impossible request instead of spending a worker discovering it.

    Read ONCE per run rather than per slice: the file is ~3 900 lines, and the
    Snakefile memoises its own lookups for the same reason.
    """
    with open(cfg["catalog"], encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}

    prefix, suffix = f"{cfg['clim_project']}_", "_{member}"
    available = {}
    for key, spec in catalog.items():
        if not (
            isinstance(key, str) and key.startswith(prefix) and key.endswith(suffix)
        ):
            continue
        body = key[len(prefix) : -len(suffix)]
        # Neither a model id nor a scenario contains `_`, so the last one
        # separates them. `NOAA-GFDL/GFDL-ESM4_ssp245` -> (model, "ssp245").
        model, _, experiment = body.rpartition("_")
        if model:
            available[(model, experiment)] = list(
                (spec.get("placeholders") or {}).get("member") or []
            )
    return available


def plan(cfg):
    """Split the requested cross product into runnable jobs and skipped ones.

    Returns `(jobs, skipped)`, where `skipped` is `(key, reason)` pairs.

    PRE-FILTERED against the catalog, because a large share of the cross
    product is not real and asking anyway is both slow and confusing. Measured
    on the shipped catalog: of 65 models with `historical`, 19 never published
    `ssp245`; and 70 of 289 entries do not offer `r1i1p1f1`, so a single global
    `members:` list misses a quarter of the catalog.

    The second case is the one worth filtering hardest. An absent (model,
    scenario) at least fails with a KeyError naming the entry, but an absent
    MEMBER used to reach hydromt, which could not resolve the name, fell back
    to treating it as a local path, and reported
    `NoDataException: ... found no files at C:\\...\\cmip6_NIMS-KMA\\UKESM1-0-LL_...`
    -- the model's `/` read as a directory separator. Nothing in that said
    "wrong member". UKESM1-0-LL publishes `r13i1p1f2`, `r14i1p1f2`,
    `r15i1p1f2`: the `f2` forcing variant, realisations from 13. Now the
    request is refused up front, naming what the entry does have.

    Nothing is auto-substituted. Falling back to whichever member exists would
    silently change WHICH realisation is analysed, and that is a methodological
    choice for the caller, not a convenience for the tool.
    """
    available = catalog_availability(cfg)
    jobs, skipped = [], []
    for model in cfg["models"]:
        for experiment in cfg["scenarios"]:
            for member in cfg["members"]:
                key = series_key(cfg["clim_project"], model, experiment, member)
                members = available.get((model, experiment))
                if members is None:
                    skipped.append((key, f"{model} published no {experiment}"))
                    continue
                if member not in members:
                    skipped.append(
                        (
                            key,
                            f"member {member} not available — this entry has "
                            f"{', '.join(members) or 'none'}",
                        )
                    )
                    continue
                jobs.append(
                    {
                        "key": key,
                        "model": model,
                        "experiment": experiment,
                        "member": member,
                        "out": str(Path(cfg["target_root"]) / f"{key}.nc"),
                    }
                )
    return jobs, skipped


#: Slices staged at once. PROCESSES, not threads, and the reason is measured:
#: `fetch_gcm_raw`'s benchmark puts one source's OPEN at 1142 s against 19 s to
#: transfer and 0.2 s to reduce, so the job is dominated by waiting on store
#: metadata -- and a worker's whole startup cost, importing geopandas + hydromt
#: + xarray, was measured at 6.7-7.4 s. Paying 7 s to remove every thread-safety
#: question is a 0.6 % overhead on the thing being parallelised.
#:
#: Threads were the alternative and are rejected on THIS workload: netCDF4/HDF5
#: serialises writes behind a global lock (the same lock `fetch_gcm_raw`'s
#: eager-`load()` comment records a deadlock against), and hydromt, GDAL and
#: gcsfs would all have to be thread-safe together. Each slice writes its own
#: file and shares nothing, so processes are the natural fit.
#:
#: 4 rather than cpu_count(): the limit is the remote store and ~311 MiB of
#: resident memory per worker, not cores.
DEFAULT_WORKERS = 4


#: Strips ANSI so a row can be classified after `_Tee` has painted it. The tee
#: colours body text on its way to a live console, so the sink below sees
#: `\x1b[...m` around the very fields it has to read.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: One row of the toolbox log grammar: `HH:MM:SS - <module> - [LEVEL - ]message`.
#: The level is present only when it is NOT info (`snake_utils._log_row_text`
#: renders INFO by omitting it), so an absent `level` group MEANS info and no
#: wording of a message can fake a raised one.
_LOG_ROW_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2} - [^\s-]+ - (?:(?P<level>[A-Z]+) - )?(?P<message>.*)$"
)


#: One console per stream per PROCESS, and the memoization is load-bearing.
#: `tee_to_log` repoints library log handlers by STREAM IDENTITY: hydromt binds
#: a StreamHandler the first time a worker parses a catalog, and
#: `_detach_handlers_bound_to` hands it back to whatever `sys.stdout` was on
#: entry. A pool worker stages many slices, so a fresh object per slice would
#: leave that handler pointing at the PREVIOUS slice's -- which the next slice's
#: identity check cannot match, so hydromt's records bypass the tee entirely and
#: arrive uncompacted, unmuted and missing from the log.
#:
#: Observed exactly that on a 6-slice run before this was memoized: two raw
#: `2026-08-19 17:29:45,254 - hydromt.data_catalog.sources.data_source - ...`
#: rows on the console, from the two slices whose worker had already been used.
_CONSOLE_WRAPPERS = {}


def _wrapped_console(name, stream):
    """This process's `_SliceConsole` for `name`, created once and reused.

    Reused only while it still wraps the SAME underlying stream. A worker
    restores the real stream after every slice and hands back the identical
    object, so the cache hits and the handler identity above is preserved; a
    caller that genuinely swapped the console gets one around the new stream
    rather than a writer onto the old. `capsys` is the second case, once per
    test, and without this check the first test to call `stage_one` would own
    every later test's output.
    """
    cached = _CONSOLE_WRAPPERS.get(name)
    if cached is None or cached.wraps is not stream:
        cached = _CONSOLE_WRAPPERS[name] = _SliceConsole(stream)
    return cached


class _SliceConsole:
    """The tee's live sink in a worker. Sorts a row into one of three fates.

    A worker cannot number its own slice -- the counter is assigned by the
    parent in completion order -- so nothing it prints can join the entry list
    that is the tool's console. This sorts instead:

    * an **INFO** row is DROPPED. `fetching <entry>`, `pinned gn/v1`,
      `store calendar=noleap` each say a third time that this slice is being
      worked on. Dropped from the console only: every row still reaches the
      slice's own log part in full, because this is the tee's live sink and
      not its log handle.
    * a **WARNING or ERROR** row is COLLECTED, stripped of its stamp and module,
      and returned to the parent, which prints it INDENTED UNDER the slice's
      entry. `irregular grid, applying the bbox directly` and `more than one
      version on the store` are findings about one source, and a reader should
      not have to match a name across two rows to learn which.
    * **anything else** is PRINTED, live and flushed. A traceback, a library's
      bare `print`, and above all the heartbeat's `... still running, 4m00s
      elapsed` -- the only thing on the console during a twenty-minute open, and
      the one row that must never wait for the slice to finish.

    The trade the middle case makes, stated plainly: a warning no longer appears
    at the moment it is raised, but when its slice lands. That is up to a couple
    of minutes later. It buys attribution, which the timestamp never gave --
    under four workers the previous shape printed a warning between two
    unrelated entries and left the reader to match names.
    """

    def __init__(self, stream):
        self._stream = stream
        self.notices = []

    @property
    def wraps(self):
        """The stream underneath, so `_wrapped_console` can tell it changed."""
        return self._stream

    def take_notices(self):
        """The notices collected since the last call, and reset.

        Reset by READING, because the console outlives the slice: one worker
        stages many, and notices left behind would be reported against whichever
        slice happened to land next.
        """
        collected, self.notices = self.notices, []
        return collected

    def write(self, text):
        row = _LOG_ROW_RE.match(_ANSI_RE.sub("", text).rstrip("\n"))
        if row is None:
            self._stream.write(text)
            # Flushed HERE, not left to the stream. A worker is a separate
            # process whose stdout is block-buffered the moment the run is
            # captured to a file, so the heartbeat's notice would otherwise
            # arrive at process exit -- long after the stall it reports.
            self._stream.flush()
        elif row.group("level"):
            self.notices.append(f"{row.group('level')} - {row.group('message')}")
        return len(text)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        # `_Tee` asks, to decide whether a carriage return overwrites. Answer
        # for the REAL console: this wrapper does not change what a terminal
        # does with the bytes it forwards.
        try:
            return self._stream.isatty()
        except Exception:
            return False


def slice_log_path(target_root, key):
    """Where one slice's fetch log lands: ``<target_root>/logs/<key>.log``.

    The ``logs`` component is not decoration. :func:`snake_utils._log_path_parts`
    anchors on it to split the path into ``(project_root, log_id)``, which is
    what gives the log its header and shortens every path inside it against
    ``target_root``; a flat ``<target_root>/<key>.log`` falls into the
    ``("", basename)`` fallback and loses both. One file per slice, because the
    workers are separate processes and a shared handle would interleave.
    """
    return str(Path(target_root) / "logs" / f"{key}.log")


def _drop_entry_name(notices, catalog_entry, member):
    """Remove this slice's own name from its notices; the entry above carries it.

    `fetch_gcm_raw` names the source in both shapes -- `<entry>: <message>` and
    `<message>: <entry>` -- and it must, because under Snakemake several fetch
    jobs interleave on one console and a row that cannot say which source it is
    about is useless. Here the notice is printed under the entry it belongs to,
    so the same name would be stated twice on adjacent lines.

    Matched against the RESOLVED entry rather than guessed from the slice key:
    the key sanitizes `/` to `_` and CMIP6 model names contain both `-` and `_`
    (`NOAA-GFDL/GFDL-ESM4`), so the sanitizing does not invert. Nothing is
    stripped unless it matches exactly.
    """
    entry = resolve_entry_name(catalog_entry, member)
    trimmed = []
    for notice in notices:
        if notice.endswith(f": {entry}"):
            notice = notice[: -len(f": {entry}")]
        else:
            head, sep, tail = notice.partition(" - ")
            if sep and tail.startswith(f"{entry}: "):
                notice = f"{head} - {tail[len(entry) + 2 :]}"
        trimmed.append(notice)
    return trimmed


def stage_one(cfg, job):
    """Stage one slice in a worker; returns (key, error, seconds, notices).

    `notices` are the WARNING/ERROR rows the fetch raised, stripped of stamp and
    module by `_SliceConsole` and handed back so the PARENT can print them under
    this slice's entry. A worker cannot number its own row, so a notice it
    printed itself would land between two unrelated entries and leave the reader
    matching names -- which is what it did until 2026-08-19.

    Returns rather than raises so one unavailable source cannot end a staging
    run that may have hours of other work in it -- a model missing a member is
    an ordinary catalog fact. The error is stringified HERE because an arbitrary
    exception may not survive the trip back to the parent process intact.

    The duration is measured INSIDE the worker, so it is the slice's own cost
    and excludes process startup and queue wait. That is the number worth
    reporting: it is dominated by the remote store OPEN (benchmarked at 19 s to
    transfer against a far larger open), so a slow slice means a slow store, not
    a slow machine or a saturated pool.

    The fetch runs inside ``tee_to_log`` -- WF2's own context manager, not a
    local imitation of it. That is what makes this tool a WRAPPER around the
    workflow's fetch rather than a second implementation of it: the compaction
    of hydromt's log format, the console mute table, path shortening, the
    severity colouring and the silence watchdog are all the pipeline's, so a
    row reads the same here as it does under Snakemake and only ONE of them has
    to be maintained. Without it, hydromt's ``StreamHandler`` writes straight
    past this process at its full ``<date> - <dotted.name> - <module> - INFO -``
    width, and every fetch row lands on the console beside the entry this tool
    prints for the same slice.

    ``tee_to_log`` re-raises, so the ``with`` sits INSIDE the ``try``: the
    traceback is written into the slice's log part on the way out and the
    return-not-raise contract above is unchanged.
    """
    started = time.perf_counter()
    notices = []
    units = {
        name: (spec or {}).get("units", "") for name, spec in cfg["variables"].items()
    }
    try:
        # Installed BEFORE the tee, so it becomes the `live` sink the tee wraps
        # and every console-bound row is classified on its way out. Restored in
        # the `finally` even though this is a worker process about to be reused
        # by the pool for the next slice -- which is exactly why it matters.
        real_stdout, real_stderr = sys.stdout, sys.stderr
        console = _wrapped_console("stdout", real_stdout)
        errors = _wrapped_console("stderr", real_stderr)
        console.take_notices()  # discard anything the previous slice left
        errors.take_notices()
        sys.stdout, sys.stderr = console, errors
        try:
            with tee_to_log(slice_log_path(cfg["target_root"], job["key"])):
                fetch_raw_slice(
                    region_path=cfg["region"],
                    raw_nc_out=job["out"],
                    catalog_path=cfg["catalog"],
                    catalog_entry=catalog_entry_name(
                        cfg["clim_project"], job["model"], job["experiment"]
                    ),
                    member=job["member"],
                    variables=list(cfg["variables"]),
                    variable_units=units,
                    buffer=cfg["buffer_cells"],
                    acquisition_window=tuple(_si.acquisition_window(job["experiment"])),
                    components=digest_components(
                        cfg, job["model"], job["experiment"], job["member"]
                    ),
                )
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
            notices = _drop_entry_name(
                console.take_notices() + errors.take_notices(),
                catalog_entry_name(
                    cfg["clim_project"], job["model"], job["experiment"]
                ),
                job["member"],
            )
    except Exception as exc:  # noqa: BLE001 -- reported per slice, never fatal
        return (
            job["key"],
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
            notices,
        )
    return job["key"], None, time.perf_counter() - started, notices


#: Outcome glyphs and the colour each carries, borrowed from `stage_data.py`'s
#: `RunReport.print_entry` so the two staging tools read alike: `+` written,
#: `=` already present, `-` skipped, `x` failed.
#:
#: The colour goes on the GLYPH and never on the name. A slice key is 40-odd
#: characters, and colouring it turns the entry list into a wash that no longer
#: keys by state -- which is the one job the glyph column has.
WRITTEN_GLYPH, WRITTEN_COLOR = "+", green
EXISTS_GLYPH, EXISTS_COLOR = "=", dim
SKIPPED_GLYPH, SKIPPED_COLOR = "-", yellow
FAILED_GLYPH, FAILED_COLOR = "x", red

#: Width of the key column in every `label   value` row. Matches
#: `stage_data.py`'s parameter rows, so the two tools' blocks break at the same
#: column and a reader moving between them re-uses the same eye position.
_LABEL_WIDTH = 12


def _format_elapsed(seconds):
    """A compact duration, byte-for-byte `stage_data.py`'s renderer.

    COPIED rather than imported, because importing it would import
    `stage_data`: that module sets four GDAL/VSI environment variables and
    pulls in geopandas + rasterio + xarray at module scope, and every worker
    process here re-imports the module it was launched from -- against a
    startup cost already measured at 6.7-7.4 s. Two small formatters are the
    cheaper duplication.

    This replaces `snake_utils.format_elapsed`, whose `h:mm:ss` is pinned to
    the benchmark tables' own column -- a pipeline concern a dev staging tool
    has no part in, and the reason a 19-second slice used to read `0:00:19`.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining:02d}s"
    return f"{minutes}m{remaining:02d}s"


def _format_bytes(size_bytes):
    """A compact byte size, also `stage_data.py`'s -- see `_format_elapsed`."""
    if size_bytes < 1_000:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1_000
        if value < 1_000 or unit == "TB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


def _entry(state_glyph, color, name, detail="", prefix="", reason="", notices=()):
    """One outcome row, on ONE line: glyph, counter, name, then the detail dim.

    The indent ladder is `stage_data.py`'s -- 0 section, 2 subject, 4 entry,
    6 detail. `detail` used to take the column-6 line of its own, which made a
    161-slice run 322 rows of which half said only a size; it now trails the
    name on the same line, dim, so the eye runs down one column of names
    instead of stepping over a detail row between each. `prefix` is printed
    already coloured, and carries the completion counter.

    `notices` are the fetch's own WARNING/ERROR rows, printed under the entry
    they belong to rather than live from a worker that cannot number itself.

    `reason` is the other thing given a line of its own, and only failures
    pass it: the multi-version message runs to ~380 characters, so inlining it
    would push the name off the first screen-width and defeat the collapse for
    every row around it.

    The parameter is `state_glyph`, never `glyph`: that name is the imported
    `console.glyph()` fallback, and binding it here would shadow the function
    for this whole body -- latent until someone adds a `glyph('·')` to one of
    these lines and gets `UnboundLocalError` on whichever branch is rarest.
    """
    head = f"{prefix} " if prefix else ""
    tail = f"  {dim(detail)}" if detail else ""
    print(f"    {color(state_glyph)} {head}{name}{tail}")
    for notice in notices or ():
        # Column 6, the ladder's detail rung -- the same one `reason` uses,
        # because a notice and a failure's reason are the same kind of thing:
        # something subordinate to the entry above. The stamp and module are
        # already gone (`_SliceConsole`); what is left opens with the LEVEL,
        # which is what makes it scannable now that no colour-coded glyph
        # introduces it.
        paint = red if notice.startswith(("ERROR", "FAILURE")) else yellow
        print(f"      {paint(notice)}")
    if reason:
        print(f"      {dim(reason)}")


def _row(label, value, note=""):
    """A `label   value` line at column 2: dim key, then the value.

    With `note`, the value is padded and cyan and the note trails dim, which is
    `stage_data.py`'s `flags:` shape; without it, the plainer `inputs:` shape.
    """
    if note:
        print(
            f"  {pad(label, _LABEL_WIDTH, dim)} {pad(str(value), 22, cyan)} {dim(note)}"
        )
    else:
        print(f"  {pad(label, _LABEL_WIDTH, dim)}  {value}")


def _recap(title, items, state_glyph, color, note=""):
    """A titled failure/refusal list, each reason on its own indented line.

    `stage_data.py` keeps its reason on the entry line, which works because a
    staging detail is short. Here a reason is a member list or a stringified
    exception, so it takes the second line rather than pushing the key off the
    right edge.

    `note` is the section's own explanation, printed ONCE under the list. A
    diagnosis that is identical for every key in a bucket is a property of the
    bucket, and repeating it per key buries the keys it is about.
    """
    print()
    print(f"{bold(title)} ({len(items)}):")
    for key, reason in items:
        print(f"  {color(state_glyph)} {key}")
        if reason:
            print(f"    {dim(reason)}")
    if note:
        print(f"  {dim(note)}")


def _unreadable(error):
    """True when the store EXISTS but this toolbox cannot read its grid.

    RESIDUAL since 2026-08-18. The case this was written for -- a GAUSSIAN
    latitude axis, which hydromt's `.raster` accessor refuses with
    `ValueError: The 'raster' accessor only applies to regular grids` -- is now
    handled inside `fetch_raw_slice`, which re-reads unclipped and applies the
    bbox itself (`dev/reference/workflows/wf2-cmip6-store-readability.md`
    section 1). That was 27 of 67 models, including CanESM5 and every
    EC-Earth3 variant.

    Kept rather than deleted because it classifies a phrase, not a cause: a
    slice reaching this bucket now means the branch did NOT engage, which is
    worth seeing separately from an ordinary download failure. What still fails
    outright is a grid the branch cannot read either -- 2-D/curvilinear
    coordinates, or a store with no x/y axis at all (MPI-M/ICON-ESM-LR, an
    unstructured ICON mesh) -- and those raise their own messages, so they
    land in `broke` with the reason attached. UA/MCM-UA-1-0 is NOT one of
    them: it spells its axes `latitude`/`longitude`, which hydromt accepts,
    and it reads through the branch like any other irregular model.

    Matched on the accessor's own phrase rather than the exception type,
    because ValueError is far too common to key on alone.
    """
    return "only applies to regular grids" in error


def _ambiguous_versions(error):
    """True when the store index recorded several versions and the glob read both.

    Matched on the PHRASE `fetch_gcm_raw` builds the message from, imported so
    the two cannot drift apart, and not on the exception type -- what comes back
    from combining two stores is whatever xarray raised, which has been at least
    `MergeError` (values disagree) and `OutOfBoundsDatetime` (the two versions
    cover different spans, so aligning their indexes upcasts the longer one into
    nanoseconds and it overflows past 2262).

    Reported separately because the action differs from every other bucket: the
    source EXISTS and is readable, and what is missing is a decision about which
    published version the assessment uses. See
    `dev/reference/workflows/wf2-cmip6-store-readability.md` section 2.
    """
    return AMBIGUOUS_VERSION_PHRASE in error


def _unavailable(error):
    """True when the slice does not EXIST rather than having failed to download.

    A KeyError from `load_catalog_entry` is the catalog saying it carries no
    such (model, scenario). Since `plan` now pre-filters against the catalog
    this should be rare -- it survives for the case where the catalog and the
    store index disagree, which is exactly when a loud report is wanted. The
    test is the exception NAME rather than the message, so a reworded error
    does not silently reclassify.
    """
    return error.startswith("KeyError")


def resolve_workers(requested, n_jobs):
    """How many workers to actually start.

    Capped at the slice count because each worker costs ~7 s of imports and
    ~311 MiB resident, so idle ones are a real expense rather than untidiness.
    Floored at 1 so a nonsense `--workers 0` still runs.
    """
    return max(1, min(requested, n_jobs))


def _print_report(
    cfg, jobs, existed, out_by_key, errors, seconds, workers, wall, skipped, requested
):
    """The end-of-run recap: what was asked for, what arrived, what did not."""
    ok = [job["key"] for job in jobs if job["key"] not in errors]
    written = [k for k in ok if not existed[k]]
    present = [k for k in ok if existed[k]]
    unavailable = {k: e for k, e in errors.items() if _unavailable(e)}
    unreadable = {k: e for k, e in errors.items() if _unreadable(e)}
    ambiguous = {k: e for k, e in errors.items() if _ambiguous_versions(e)}
    broke = {
        k: e
        for k, e in errors.items()
        if not _unavailable(e) and not _unreadable(e) and not _ambiguous_versions(e)
    }
    size = sum(out_by_key[k].stat().st_size for k in ok if out_by_key[k].exists())

    print()
    print(rule())
    print(section_banner("total"))
    # The pill states every outcome ONCE, in `stage_data.py`'s order and
    # colours. A catalog refusal counts as `skipped` there too: nothing was
    # spent on it, which is exactly what the word means in the other tool.
    print(
        f"{green(f'written: {len(written)}')} {dim(glyph('·'))} "
        f"{dim(f'existing: {len(present)}')} {dim(glyph('·'))} "
        f"{yellow(f'skipped: {len(skipped)}')} {dim(glyph('·'))} "
        f"{red(f'failed: {len(errors)}')}"
    )
    print(
        f"{dim('elapsed:')} {_format_elapsed(wall)} {dim(glyph('·'))} "
        f"{dim('size:')} {_format_bytes(size)}"
    )

    print()
    _row(
        "requested",
        f"{len(cfg['models'])} model(s) x {len(cfg['scenarios'])} scenario(s) "
        f"x {len(cfg['members'])} member(s)  =  {requested} slice(s)",
    )
    _row("attempted", len(jobs))
    _row("staged", f"{len(ok)} of {len(jobs)}")
    _row("target", fmt_path(cfg["target_root"]))
    if seconds:
        slice_total = sum(seconds.values())
        # Slice time SUMMED across workers against wall clock. The ratio is the
        # speed-up the pool actually delivered -- worth printing because this
        # workload is bound by the remote store's open, so more workers help
        # until the store rate-limits and then stop. A ratio well below
        # `workers` says adding more will not pay.
        ratio = f", {slice_total / wall:.1f}x wall" if wall >= 1.0 else ""
        _row(
            "slice time",
            f"{_format_elapsed(slice_total)} summed over {workers} worker(s){ratio}",
        )
        slowest = max(seconds.items(), key=lambda kv: kv[1])
        _row("slowest", f"{slowest[0]}  ({_format_elapsed(slowest[1])})")

    # The verdict, stated once and in `stage_data.py`'s three shapes. The
    # nothing-to-do case is not a failure of this tool: it is the catalog
    # having refused everything asked for, which the recap below spells out.
    print()
    if not errors and ok:
        print(green(bold(f"OK — all {len(ok)} slice(s) staged successfully.")))
    elif not errors:
        print(yellow(bold("nothing to do — no slice survived the catalog filter.")))
    else:
        print(red(bold(f"FAILED — {len(errors)} slice(s) did not stage; see below.")))

    # Catalog refusals are NOT recapped here. They are already stated twice
    # above -- once in the pill's `skipped` count, once per model in the
    # parameters block's availability ratio -- and nothing was spent on them,
    # so a third listing padded the tail of every run with the one outcome
    # that needs no action.

    # The two failure kinds are listed SEPARATELY because they call for
    # different actions: an absent (model, scenario) will never appear however
    # many times it is retried, while a broken one may well succeed next run.
    if unavailable:
        _recap(
            "not available in the catalog",
            [(key, "") for key in sorted(unavailable)],
            FAILED_GLYPH,
            FAILED_COLOR,
        )
    # Third kind: the model is THERE and the request was right, but its grid
    # was refused. Since the irregular-grid branch landed this should be EMPTY
    # -- a Gaussian axis is read now -- so anything here means the branch did
    # not engage and is worth reporting on its own line rather than among the
    # ordinary download failures.
    if unreadable:
        _recap(
            "irregular grid, cannot be read",
            [(key, "") for key in sorted(unreadable)],
            FAILED_GLYPH,
            FAILED_COLOR,
            note=(
                "hydromt's raster accessor refused the grid AND "
                "fetch_raw_slice's irregular-grid branch did not engage"
            ),
        )

    # Fourth kind, and the only one whose fix is a DECISION rather than a
    # repair: the store is there and readable, but the index recorded several
    # published versions and nobody has said which one this assessment uses.
    if ambiguous:
        _recap(
            "several published versions, none chosen",
            [(key, ambiguous[key]) for key in sorted(ambiguous)],
            FAILED_GLYPH,
            FAILED_COLOR,
            note=(
                "pin one version in the catalog to stage these; "
                "the versions are named on each row"
            ),
        )

    if broke:
        _recap(
            "could not be downloaded",
            [(key, broke[key]) for key in sorted(broke)],
            FAILED_GLYPH,
            FAILED_COLOR,
            # The message below is the exception's one line. `tee_to_log`
            # writes the whole traceback, and every row the fetch printed
            # before it, into the slice's own log part -- so the note names
            # where to look rather than making the reader guess.
            note=(
                "the fetch log and traceback for each are under "
                f"{fmt_path(str(Path(cfg['target_root']) / 'logs'))}"
            ),
        )


def _plan_by_model(cfg, jobs, skipped):
    """`model -> (planned, refused)` slice counts, for the parameters block.

    Rebuilt from `plan`'s two lists rather than returned by it, because `plan`
    is a tested contract and this is presentation. The refusal side is matched
    on the exact `series_key`, not on a prefix -- a model id contains hyphens
    and slashes, and a prefix test would mis-attribute
    `INM/INM-CM4-8` and `INM/INM-CM5-0` to each other on a shorter name.
    """
    planned = dict.fromkeys(cfg["models"], 0)
    refused = dict.fromkeys(cfg["models"], 0)
    refused_keys = {key for key, _reason in skipped}
    for job in jobs:
        planned[job["model"]] = planned.get(job["model"], 0) + 1
    for model in cfg["models"]:
        for experiment in cfg["scenarios"]:
            for member in cfg["members"]:
                key = series_key(cfg["clim_project"], model, experiment, member)
                if key in refused_keys:
                    refused[model] += 1
    return planned, refused


def _print_parameters(cfg, config_path, jobs, skipped, workers, dry_run):
    """Inputs, the per-model plan, and the flags -- `stage_data.py`'s three blocks."""
    print(banner("Parameters"))

    print(bold("inputs:"))
    for label, value in (
        ("config", fmt_path(config_path)),
        ("region", fmt_path(cfg["region"])),
        ("target_root", fmt_path(cfg["target_root"])),
        ("catalog", fmt_path(cfg["catalog"])),
        ("store_index", fmt_path(cfg["store_index"])),
    ):
        _row(label, value)
    print()

    # Listed per MODEL, not per slice. The models are the config's own list,
    # while the slices are its cross product and run to hundreds on a full
    # ensemble -- and the question this block answers is which models the
    # catalog refused before a worker started. The slices themselves are the
    # STAGE section's business.
    planned, refused = _plan_by_model(cfg, jobs, skipped)
    width = max((len(m) for m in cfg["models"]), default=0) + 2
    print(bold(f"models ({len(cfg['models'])}):"))
    for model in cfg["models"]:
        # Stated as a RATIO, not a count plus a refusal: the question here is
        # how much of what was asked for this model can actually be staged,
        # and `4/5` answers it at a glance where `4 slice(s), 1 not in the
        # catalog` made the reader do the addition.
        note = f"{planned[model]}/{planned[model] + refused[model]} slice(s) available"
        print(f"  {pad(model, width, cyan)}  {dim(note)}")
    print()

    print(bold("flags:"))
    _row("buffer", cfg["buffer_cells"], "grid cells around the region polygon")
    _row("variables", ", ".join(cfg["variables"]), "post-rename names, not pr/tas")
    _row("workers", workers, "slices staged at once")
    if dry_run:
        _row("dry run", "yes", "list the plan; open nothing")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage WF2-cache-compatible CMIP6 slices for one region."
    )
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--region", help="override the config's region polygon")
    parser.add_argument("--target-root", help="override where slices are written")
    parser.add_argument("--models", nargs="+", help="override the model list")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the slices and their destinations; open nothing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"slices to stage at once (default: {DEFAULT_WORKERS}). "
            "1 runs in this process, which is what to use when a failure needs "
            "a readable traceback"
        ),
    )
    args = parser.parse_args(argv)

    # The workers flush every row as the tee writes it, while this process's
    # `print` is BLOCK-buffered the moment stdout is not a terminal -- which is
    # exactly how a staging run is captured (`stage_cmip6.py > run.log`). The
    # result is a file where the workers' warnings sit above the banner and
    # nowhere near the slice rows they belong to; measured on a 6-slice run,
    # two `irregular grid` warnings landed at lines 4-5 of a 40-line report.
    # Line buffering costs nothing at this row count and restores true order.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass  # a stream that cannot be reconfigured is not worth failing over

    cfg = load_config(args.config)
    if args.region:
        cfg["region"] = args.region
    if args.target_root:
        cfg["target_root"] = args.target_root
    if args.models:
        cfg["models"] = args.models

    jobs, skipped = plan(cfg)
    requested = len(cfg["models"]) * len(cfg["scenarios"]) * len(cfg["members"])

    # Resolved BEFORE the parameters block so the flag it prints is the number
    # the pool will actually start, not the number asked for.
    workers = resolve_workers(args.workers, len(jobs))
    _print_parameters(cfg, args.config, jobs, skipped, workers, args.dry_run)

    if args.dry_run:
        # The same glyph vocabulary as a real run, so a dry run previews the
        # console it is a dry run of: `+` would be fetched, `=` already there,
        # `-` refused by the catalog.
        print(banner("Dry Run"))
        print(
            f"  {len(jobs)} of {requested} slice(s) would be staged into "
            f"{fmt_path(cfg['target_root'])}"
        )
        print()
        for job in jobs:
            if Path(job["out"]).exists():
                _entry(
                    EXISTS_GLYPH,
                    EXISTS_COLOR,
                    job["key"],
                    "already present; would be revalidated, not re-fetched",
                )
            else:
                # No detail line: inside a DRY RUN section a `+` already says
                # "would be fetched", and repeating it once per slice buys a
                # second column of nothing on a 200-slice ensemble.
                _entry(WRITTEN_GLYPH, WRITTEN_COLOR, job["key"])
        for key, reason in skipped:
            _entry(SKIPPED_GLYPH, SKIPPED_COLOR, key, detail=reason)
        return 0

    if not jobs:
        # Every requested combination was refused before any worker started.
        _print_report(cfg, [], {}, {}, {}, {}, 0, 0.0, skipped, requested)
        return 1

    os.makedirs(cfg["target_root"], exist_ok=True)
    print(banner("Stage"))
    print(f"  {len(jobs)} slice(s) through {workers} worker(s)")
    # Named once, here, rather than on every entry: the console below carries
    # one row per slice and the log path is derivable from the key on all of
    # them. Said at all because the fetch chatter no longer scrolls past -- a
    # reader who wants it has to know it was kept somewhere.
    print(
        f"  {dim('fetch logs: ' + fmt_path(slice_log_path(cfg['target_root'], '<slice>')))}"
    )
    print()

    # Which outputs were ALREADY there before this run. `fetch_raw_slice`
    # revalidates and returns without touching the network when a slice is
    # present and its digest matches, so without this the report could not tell
    # "fetched" from "was already good" -- and on a re-run that is the whole
    # story.
    existed = {job["key"]: Path(job["out"]).exists() for job in jobs}
    out_by_key = {job["key"]: Path(job["out"]) for job in jobs}

    done = 0
    seconds = {}
    errors = {}
    wall_started = time.perf_counter()

    def _report(key, error, elapsed, notices=()):
        """One line per finished slice, in COMPLETION order.

        Completion order, not config order: with several workers in flight the
        useful signal is which slice just landed, and a staging run is long
        enough that waiting to sort would mean a silent console for minutes.
        The end-of-run report re-states everything in a stable order.
        """
        nonlocal done
        done += 1
        seconds[key] = elapsed
        # Padded OUTSIDE the bracket, not inside it. Right-aligning the count
        # within `[...]` kept the names in one column but put the gap where a
        # reader sees it -- `[  1/161]` reads as a typo where `[1/161]  ` reads
        # as a column. Same total width either way, so the names still line up.
        width = len(str(len(jobs)))
        prefix = dim(f"[{done}/{len(jobs)}]".ljust(2 * width + 3))
        if error:
            errors[key] = error
            # STDOUT, not stderr, and deliberately: a failing slice is one row
            # of the same entry list as a successful one, and two streams out
            # of a process pool interleave in an order neither of them chose.
            # The recap re-states every failure in a stable order anyway.
            _entry(
                FAILED_GLYPH,
                FAILED_COLOR,
                key,
                _format_elapsed(elapsed),
                prefix=prefix,
                reason=error,
                notices=notices,
            )
            return
        path = out_by_key[key]
        size = path.stat().st_size if path.exists() else 0
        state_glyph, color = (
            (EXISTS_GLYPH, EXISTS_COLOR)
            if existed[key]
            else (WRITTEN_GLYPH, WRITTEN_COLOR)
        )
        detail = _format_bytes(size)
        # `stage_data.py`'s rule: a cached re-run should not print a column of
        # `elapsed: 0.1s`, so the duration is shown when the slice was actually
        # fetched or when it cost more than about a second.
        #
        # Two spaces rather than `; elapsed: ` now that the row is one line: the
        # label was earning its keep on a detail line of its own, and beside a
        # size on the same line a bare duration is unambiguous.
        if not existed[key] or elapsed > 1.0:
            detail += f"  {_format_elapsed(elapsed)}"
        _entry(state_glyph, color, key, detail, prefix=prefix, notices=notices)

    if workers == 1:
        # Deliberately NOT a one-worker pool: staying in-process keeps
        # tracebacks and any debugger attached to the real failure.
        for job in jobs:
            _report(*stage_one(cfg, job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(stage_one, cfg, job): job["key"] for job in jobs}
            for future in as_completed(futures):
                try:
                    _report(*future.result())
                except Exception as exc:  # noqa: BLE001 -- a worker died outright
                    # Distinct from a slice that failed: the worker never got to
                    # return, so `stage_one`'s own handler never ran. Most likely
                    # a killed process (memory) rather than a bad source.
                    key = futures[future]
                    # No notices: the worker never returned, so whatever it
                    # collected died with it. The slice's log part still has it.
                    _report(key, f"worker died: {type(exc).__name__}: {exc}", 0.0)

    _print_report(
        cfg,
        jobs,
        existed,
        out_by_key,
        errors,
        seconds,
        workers,
        time.perf_counter() - wall_started,
        skipped,
        requested,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
