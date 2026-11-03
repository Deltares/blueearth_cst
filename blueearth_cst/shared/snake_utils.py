# Current guidance: dev/reference/toolbox-spatial-foundation.md
"""Shared Snakefile helpers and compatibility exports.

WF3 imports its scientific and neutral logging helpers from ``wf3_science``
and ``run_log_core`` directly. This module keeps the historical API for other
workflows and owns Wflow-specific terminal handling.
"""

# ruff: noqa: F401 -- imported names preserve the historical helper surface.

import contextlib
import gc
import io
import json
import logging
import math
import os
import posixpath
import re
import subprocess
import sys
import threading
import time
import traceback
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from blueearth_cst.climate_analysis.source_capabilities import (
    PRECIP_ONLY_SOURCES as PRECIP_ONLY_SOURCES,
)
from blueearth_cst.shared.advanced_settings import (
    _ADVANCED_SETTINGS_SCHEMA,
    _MONTH_ABBREVS,
    _QUANTILE_STAT_RE,
    _VALIDATORS,
    _VERSION_RE,
    ADVANCED_SETTINGS,
    ADVANCED_SETTINGS_PATH,
    _catalog_entry_name,
    _month_abbrev,
    _monthly_factors,
    _nonnegative_int,
    _positive_float,
    _positive_int,
    _statistic_names,
    _unit_fraction,
    _version_string,
    load_advanced_settings,
)
from blueearth_cst.shared.run_log_core import (
    _ANSI_ALERT,
    _ANSI_BODY,
    _ANSI_DIM,
    _ANSI_FAIL,
    _ANSI_RESET,
    _ASCII_GLYPH_FALLBACK,
    _COMPONENT_PREFIX_RE,
    _CONSOLE_PATH_TOKENS_ENV,
    _DATA_SOURCE_FROM_RE,
    _DATA_SOURCE_READ_RE,
    _DEMOTED_WARNINGS,
    _EXCEPTHOOK_MARKERS,
    _FIGURE_BUNDLE_FLUSHING,
    _FIGURE_BUNDLE_MODULES,
    _FIGURE_BUNDLES,
    _FORCING_EXISTS_RE,
    _HEARTBEAT_LABEL_RE,
    _HYDROMT_LOG_RE,
    _JULIA_RECORD_HEAD,
    _JULIA_RECORD_MID,
    _JULIA_RECORD_TAIL,
    _LOG_LEVEL_RANK,
    _MEMBER_PART_RE,
    _PATH_TOKENS_ENV,
    _PROJECT_ROOT_ENV,
    _REMOTE_PREFIXES,
    _REMOTE_READ_ECHO_RE,
    _REPO_ROOT,
    _ROW_PREFIX_RE,
    _SEVERITY_PATTERNS,
    _SITE_PACKAGES_RE,
    _STRIPPED_TAIL_RE,
    _TEE_CONSOLE_MUTED,
    _WARNING_TALLY_ENV,
    _ansi,
    _compact_log_line,
    _console_colour,
    _cr_overwrite,
    _declared_tokens,
    _demoted_warning,
    _detach_handlers_bound_to,
    _drop_redraw_frames,
    _drop_repeated_source_name,
    _flush_pending,
    _folder_rows,
    _Heartbeat,
    _heartbeat_identity,
    _is_clean_exit,
    _is_shutdown_noise,
    _JuliaRecordFolder,
    _line_reset,
    _log_header_lines,
    _log_path_parts,
    _log_row_text,
    _muted_matches,
    _muted_on_console,
    _NoFrameRelay,
    _pad_line_over,
    _paint_body,
    _paint_line,
    _path_tokens,
    _redirect_console_log_handlers,
    _relativize_paths,
    _restore_log_handlers,
    _set_handler_stream,
    _severity_code,
    _spelt_tokens,
    _strip_prefix,
    _Tee,
    _tokenize_prefix,
    flush_figure_bundles,
    format_elapsed,
    log_level_floor,
    log_row,
    note_warning,
    plural,
    rule_id,
    tee_to_log,
)
from blueearth_cst.shared.wf3_science import (
    _AXIS_SUBKEYS,
    _SEED_MODULUS,
    CLIMATE_STORE_SCRIPT,
    DEFAULT_BASIN_INDEX,
    DEFAULT_HYDROGRAPHY,
    DEFAULT_SEED,
    DEFAULT_SPELL_FACTOR,
    DEFAULT_WATER_YEAR_START,
    REGION_SCRIPT,
    ClimateStoreRule,
    RegionRule,
    _reject_unknown_axis_subkeys,
    _require_n_levels,
    climate_store_rule,
    derive_seed,
    historical_window_bounds,
    index_width,
    region_geojson_path,
    region_rule,
    resolve_seed,
    resolve_water_year_start,
    slugify_window,
    stress_test_grid,
    validate_spell_factor,
    water_year_start_number,
    window_year_pair,
)


def declare_path_tokens(*, console_only=None, **folders):
    """Declare a workflow's key folders once, by short name. Returns the pairs.

    Called at Snakefile PARSE time, before any rule runs. Two things then read
    the declaration and they must not disagree, which is why it is one call:

    * :func:`run_header` and :func:`_log_header_lines` print the folders, so the
      console and every rule log open by saying where the run's data lives.
    * :func:`_relativize_paths` rewrites each declared folder to ``<name>`` in
      every line that mentions it, so the same twelve directory components stop
      being repeated on line after line of a rule's output.

    Together those are the point: a path is stated in FULL once, at the top, and
    referred to by name after that. A token with no header row would be worse
    than the long path it replaced -- an undefined symbol in a log someone reads
    months later -- so nothing tokenizes that is not also declared.

    Empty and ``None`` values are dropped rather than declared, so a workflow
    passes what it has (WF2 has no model) without a caller-side conditional.
    Paths are stored ABSOLUTE and normalized: a rule's output names a resolved
    path, and a relative ``project_dir`` from the config would never match it.

    ``console_only`` optionally maps display aliases to paths (including files).
    These appear in the console legend and terminal rows, while saved logs keep
    the shared folder spelling. Each declaration replaces both sets of aliases.
    """
    tokens = {}
    for name, folder in folders.items():
        if folder is None or not str(folder).strip():
            continue
        tokens[name] = os.path.normpath(os.path.abspath(os.fspath(folder)))
    os.environ[_PATH_TOKENS_ENV] = json.dumps(tokens)
    os.environ[_CONSOLE_PATH_TOKENS_ENV] = json.dumps(
        {
            name: os.path.normpath(os.path.abspath(os.fspath(path)))
            for name, path in (console_only or {}).items()
            if path is not None and str(path).strip()
        }
    )
    return tokens


def declare_project_root(project_dir):
    """Declare the run's project directory for :func:`log_row` to shorten against.

    ``tee_to_log`` already relativizes everything a rule with a ``log:``
    directive prints, so this exists for the rules that have NONE -- the config
    snapshot (0.01/1.01/2.01/3.01) is the standing example. Those rows reach the
    console without passing the tee, so they were the only ones in a run still
    printing ``test_case\\test_rapid\\config\\build_model\\...`` -- full length, and
    backslashed, while every neighbouring row was short and forward-slashed.
    The project dir is stated once in the header either way, so nothing is lost
    by shortening them to match.

    Declared alongside :func:`declare_path_tokens` at Snakefile parse time.
    Absent (a bare ``log_row`` in a test, or a script run by hand) the message
    is left alone, which is the behaviour this improves on.
    """
    if project_dir is None or not str(project_dir).strip():
        os.environ.pop(_PROJECT_ROOT_ENV, None)
        return ""
    os.environ[_PROJECT_ROOT_ENV] = os.fspath(project_dir)
    return os.environ[_PROJECT_ROOT_ENV]


def catalog_root(data_sources):
    """Return a data catalog's local root directory, or ``""`` when it has none.

    The external data tree is the one folder in this toolbox a project does not
    own and cannot derive: it is declared inside the hydromt catalog, as
    ``meta.roots``, and every ``Reading <source> data from ...`` line a build
    prints is under it. Reading it here is what lets that folder be named once
    in the header instead of repeated on every one of those lines.

    Accepts the config's own shape, which is a path or a list of them, and takes
    the FIRST local root: a catalog whose root is remote (``gs://``, ``s3://``)
    has no prefix worth shortening. Fail-open in every direction — an
    unreadable, malformed or rootless catalog yields ``""``, and the paths
    simply print in full.
    """
    if isinstance(data_sources, (list, tuple)):
        data_sources = next((item for item in data_sources if item), "")
    if not data_sources:
        return ""
    try:
        with open(os.fspath(data_sources), "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError, TypeError):
        return ""
    meta = document.get("meta") if isinstance(document, Mapping) else None
    if not isinstance(meta, Mapping):
        return ""
    roots = meta.get("roots") or meta.get("root") or []
    if isinstance(roots, str):
        roots = [roots]
    for root in roots:
        if isinstance(root, str) and root.strip() and "://" not in root:
            return root
    return ""


def get_config(config, arg, default=None, optional=True):
    """Read a config key, returning a default for optional missing keys.

    Parameters
    ----------
    config : Mapping
        Config section to read from.
    arg : str
        Key to look up.
    default : Any, optional
        Value returned when ``arg`` is absent and ``optional`` is True.
    optional : bool, optional
        When False, a missing ``arg`` raises ``ValueError`` instead of
        returning ``default``.

    Returns
    -------
    Any
        ``config[arg]`` when present — including ``None`` and other falsey
        values, which are returned as-is rather than replaced by ``default``.
        Otherwise ``default`` for optional keys.

    Raises
    ------
    ValueError
        If ``arg`` is absent and ``optional`` is False.
    """
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config")


def file_digest_or_absent(path) -> str:
    """Return the SHA-256 hex digest of a file's bytes, or ``"ABSENT"``.

    Absence-tolerant digest helper for the wf3 drift guard's params
    (dev/records/milestones/p31/experiment-structure-design.md §3b/§3c, ext2-2). Called at
    Snakefile parse time for the wf1/wf2 project-snapshot digests, so a fresh
    project (no snapshot yet) still parses, ``--dry-run``s, and ``--unlock``s
    cleanly — snapshot absence surfaces at the guard *rule* via its
    ``ancient()`` input declaration (``MissingInputException``), never as a
    parse-time traceback.

    - **present:** SHA-256 hex digest of the file bytes — any content change
      flips the returned string, tripping Snakemake's params rerun-trigger.
    - **missing (or unreadable):** the literal sentinel string ``"ABSENT"`` —
      never raises. ``"ABSENT"`` cannot collide with a real digest (uppercase,
      non-hex, wrong length), and the ABSENT->present transition itself flips
      the param, so the first post-wf1 invocation re-evaluates the guard.
    """
    import hashlib

    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "ABSENT"


_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})

_EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

_EXPERIMENT_NAME_MAX_LEN = 64

_WINDOWS_RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


def project_slug(project_dir, reserve: int = 0) -> str:
    """Slugify ``project_dir``'s basename into an experiment-name stem.

    Legacy stem for dated-name utilities. Current runners select numbered
    families independently of the project directory's basename.

    ``reserve`` is how many characters the caller will append afterwards; the
    stem is truncated to leave room, so the total still fits the length limit.

    Raises ``ValueError`` if the basename has no alphanumeric characters at all.
    """
    base = os.path.basename(str(project_dir).replace("\\", "/").rstrip("/"))
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError(
            f"cannot derive an experiment_name from project_dir basename "
            f"{base!r}: it contains no alphanumeric characters"
        )
    return slug[: _EXPERIMENT_NAME_MAX_LEN - reserve].rstrip("_")


def suggest_experiment_name(project_dir, today: str) -> str:
    """Suggest an ``experiment_name`` from ``project_dir`` and a date stamp.

    This pure legacy utility is retained for callers working with dated names.
    Execution runners and the suggestion CLI now use numbered reservation
    selection; they do not call this helper or write a date into configuration.

    ``project_dir``'s basename is **slugified**, because it is not guaranteed
    to satisfy the grammar ``validate_experiment_name`` enforces (repo-7):
    ``examples/Gabon`` was live in six shipped configs, and production
    ``project_dir`` values routinely carry uppercase, hyphens or spaces. The
    slug is lowercased, every character outside ``[a-z0-9]`` becomes ``_``,
    runs of ``_`` collapse, leading non-alphanumerics are stripped, and the
    result is truncated to fit the length limit once the date suffix is added.

    This deliberately differs from ``validate_experiment_name``'s
    never-silently-lowercase stance: that function VALIDATES a value a human
    chose, where a silent case change would be a surprise; this one PROPOSES a
    value from a path the user did not write as a slug. The proposal is passed
    back through ``validate_experiment_name`` before being returned, so the two
    can never disagree.

    Parameters
    ----------
    project_dir : str | Path
        the run's output root; only its basename is used
    today : str
        date stamp to append, ``YYYYMMDD``. Passed in rather than read from the
        clock so the helper stays deterministic and testable.

    Returns the validated suggestion, or raises ``ValueError`` if no valid slug
    can be derived (e.g. a basename with no alphanumerics at all).
    """
    suffix = f"_{today}" if today else ""
    slug = project_slug(project_dir, reserve=len(suffix))
    return validate_experiment_name(f"{slug}{suffix}", project_dir)


def validate_experiment_name(name: str, project_dir) -> str:
    """Validate ``experiment_name`` as a safe ``experiments/<name>/`` path segment.

    Centralized slug validation for the WF4 experiment subtree
    (dev/records/milestones/p31/experiment-structure-design.md §2b). Called once at
    simulation configuration preflight, BEFORE ``exp_dir`` (and every
    derived output/params path) is built, so all paths are constructed only from
    a vetted value. Parse-time is correct here: a malformed name makes the entire
    DAG ill-defined, so failing under ``--dry-run`` is the intended behavior
    (unlike the drift *guard*, which is a rule so ``--unlock`` stays usable).

    Grammar: ``^[a-z0-9][a-z0-9_]*$`` (lowercase alnum + underscore, must start
    with an alnum), nonempty, at most 64 chars — a strict subset of
    ``dev/reference/dev-naming-conventions.md``'s snake_case rule that deliberately excludes
    hyphens and dots so the value can never introduce a path component or an
    extension. Uppercase is REJECTED (never silently lowercased). After the
    grammar, a containment assertion confirms the resolved target is a direct
    child of ``<project_dir>/experiments`` (belt to the grammar's braces).

    Returns the validated ``name`` unchanged, or raises ``ValueError`` naming the
    offending input.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"experiment_name must be a non-empty string, got {name!r}")
    if len(name) > _EXPERIMENT_NAME_MAX_LEN:
        raise ValueError(
            f"experiment_name {name!r} exceeds the {_EXPERIMENT_NAME_MAX_LEN}-char "
            "limit"
        )
    # Case-insensitive Windows-reserved-name check (including any extension):
    # the bare stem before the first dot must not be a reserved device name.
    stem = name.split(".", 1)[0].lower()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"experiment_name {name!r} is a Windows-reserved device name "
            "(case- and extension-insensitive); choose another name"
        )
    if not _EXPERIMENT_NAME_RE.match(name):
        raise ValueError(
            f"experiment_name {name!r} does not match the required grammar "
            r"^[a-z0-9][a-z0-9_]*$ (lowercase alphanumerics and underscores, "
            "starting with an alphanumeric; no separators, dots, hyphens, "
            "absolute forms, or uppercase)"
        )
    # Containment assertion (independent of the grammar): the resolved target
    # must be a DIRECT child of <project_dir>/experiments. .resolve() at parse is
    # safe — it does not require the dir to exist.
    experiments_root = os.path.abspath(os.path.join(str(project_dir), "experiments"))
    target = os.path.abspath(os.path.join(experiments_root, name))
    if os.path.dirname(target) != experiments_root:
        raise ValueError(
            f"experiment_name {name!r} does not resolve to a direct child of "
            f"{experiments_root!r}"
        )
    return name


MIN_HISTORICAL_YEARS = ADVANCED_SETTINGS["constraints"]["min_historical_years"]

JULIA_VERSION = ADVANCED_SETTINGS["runtime"]["julia_version"]

DEFAULT_JULIA_THREADS = ADVANCED_SETTINGS["runtime"]["julia_threads"]


def validate_julia_threads(value) -> int:
    """Validate a Julia thread count as a positive whole number.

    The count comes from ``advanced_settings.runtime.julia_threads`` or a
    Wflow rule's ``--set-threads``; a project config cannot set it (``C-54``
    removed ``shared.julia_threads``), so the refusal names those two sources.

    Parse-time, like the other config validators here: the value lands in a
    ``shell:`` body, so a bad one would otherwise surface as a Julia usage error
    inside a rule rather than as a config problem. Same predicate the settings
    file's own ``defaults.julia_threads`` is held to.
    """
    return _positive_int(
        value,
        "julia threads (advanced_settings.runtime.julia_threads or --set-threads)",
    )


def water_year_end_anchor(month: str) -> str:
    """The pandas resample anchor for a water year STARTING in ``month``.

    A year that starts in October ends in September, so the anchor is the month
    BEFORE the start — ``YE-SEP``. The off-by-one is the whole reason this is a
    function: ``YE-OCT`` would silently aggregate Nov→Oct and every annual
    extreme would be attributed to the wrong year.

    A January water year yields ``YE-DEC``, which pandas treats as identical to
    a bare ``YE`` — so adopting this helper at the default changes no number.
    """
    index = _MONTH_ABBREVS.index(_month_abbrev(month, "water_year_start"))
    return f"YE-{_MONTH_ABBREVS[(index - 1) % 12].upper()}"


DEFAULT_WATER_YEAR_ANCHOR = water_year_end_anchor(DEFAULT_WATER_YEAR_START)


def julia_prefix(threads=DEFAULT_JULIA_THREADS) -> str:
    """The ``julia ... `` prefix both Wflow-running rules share.

    ``--project=.`` resolves against Snakemake's working directory, which is the
    repository root — where ``Project.toml``/``Manifest.toml`` live.
    """
    return f"julia +{JULIA_VERSION} --project=. --threads {validate_julia_threads(threads)}"


def _shift_years(moment, years):
    """``moment`` shifted by whole calendar years; Feb 29 clamps to Feb 28.

    Duck-typed on ``.replace()``/``.year`` so it accepts both ``datetime`` (the
    parse-time path, from config strings) and ``pandas.Timestamp`` (the
    extraction path, from the data's own time axis).
    """
    try:
        return moment.replace(year=moment.year + years)
    except ValueError:  # 29 Feb -> a non-leap year
        return moment.replace(year=moment.year + years, month=2, day=28)


def format_years(years: float) -> str:
    """A record length for the console: whole years, rounded half up.

    One decimal is kept only when rounding would lift a span that falls short of
    ``MIN_HISTORICAL_YEARS`` onto the floor itself -- ``~30 years, below the
    30-year minimum`` would contradict itself, so that case reads ``~29.6``.
    """
    rounded = math.floor(years + 0.5)
    if years < MIN_HISTORICAL_YEARS <= rounded:
        return f"{years:.1f}"
    return str(rounded)


def meets_min_historical_years(start, end) -> bool:
    """Does ``start..end`` span at least ``MIN_HISTORICAL_YEARS`` calendar years?

    Calendar arithmetic, not ``days / 365.25``: the requirement is on ANNUAL
    observations, so "16 years later, same date" is the honest comparison and it
    stays exact across leap years.
    """
    return end >= _shift_years(start, MIN_HISTORICAL_YEARS)


def historical_window_days(historical_window) -> int:
    """Calendar days spanned by a ``climate.window`` mapping.

    Written in terms of ``historical_window_bounds``, so it inherits the same
    parsing and the same fail-loud errors.
    """
    start, end = historical_window_bounds(historical_window)
    return (end - start).days


def validate_historical_window(historical_window) -> int:
    """Reject a ``climate.window`` shorter than ``MIN_HISTORICAL_YEARS``.

    Called at ``build_model.smk`` parse time, so a window that cannot
    support a full CST run is rejected BEFORE any rule executes — the same
    parse-time stance as ``clim_historical: eobs`` and
    ``validate_experiment_name``, and for the same reason: no execution can
    rescue it, so the earliest possible failure is the most legible one.

    This checks what the config REQUESTS. Whether the staged source actually
    covers it is unknowable until extraction, and is checked against the same
    floor there (``extract_historical_climate._check_window_coverage``).

    Returns the span in days, or raises ``ValueError`` naming the requested
    window, its length and the floor.
    """
    start, end = historical_window_bounds(historical_window)
    days = historical_window_days(historical_window)
    if not meets_min_historical_years(start, end):
        raise ValueError(
            f"climate.window {start.date()} .. {end.date()} spans "
            f"{format_years(days / 365.25)} years, below the "
            f"{MIN_HISTORICAL_YEARS}-year minimum this toolbox requires: "
            f"weathergenr's wavelet decomposition needs at least "
            f"{MIN_HISTORICAL_YEARS} annual observations, so a shorter record "
            f"cannot support a climate stress test. Widen "
            f"climate.window to >= {MIN_HISTORICAL_YEARS} years"
            + ("" if days >= 0 else " (endtime is BEFORE starttime — check the order)")
        )
    return days


def resolve_simulation_window(
    climate_cfg, model_cfg, *, shared_source=None, model_source=None
):
    """The window the hydrological model SIMULATES, which is not the record.

    Two different questions were one config key until 2026-08-10:

    * ``climate.window`` — how much climate record to EXTRACT. It
      feeds the climate store, the climate figures, and (through that store)
      weathergenr, whose wavelet decomposition sets ``MIN_HISTORICAL_YEARS``.
      This is analysis input, and it is what a future standalone climate
      workflow would be parameterised on.
    * ``workflows.build_model.simulation_window`` — the period the model is
      RUN over. It sets the forcing hydromt prepares and the ``[time]``
      ``starttime``/``endtime`` in the wflow TOML, which are necessarily the
      same span: forcing outside the run period is built and never read, and a
      run period outside the forcing has nothing to read.

    The simulation window must sit INSIDE the record, and that is a change from
    how this shipped on 2026-08-10. It was written unconstrained, correctly at
    the time: rule 1.09 declared no climate-store input and read its forcing
    from the data catalog, so the two windows were genuinely independent. Rule
    1.09 now builds the forcing FROM the store, to stop re-reading the same
    source twice, and a simulation period outside the extraction therefore has
    no data behind it. Caught here, at parse time, rather than as a truncated or
    empty forcing twenty rules downstream.

    ``MIN_HISTORICAL_YEARS`` is likewise not applied here. It exists for
    weathergenr's record, so a project can now run a short simulation while
    keeping the >=16-year record a stress test needs — which the single-key
    form could not express.

    REQUIRED: declare the model run period explicitly, even when it equals
    ``climate.window``. A long climate record must not silently become a long
    Wflow simulation when this key is omitted or commented out.

    Returns a mapping with inclusive ``start``/``end`` years; raises ``ValueError``
    naming the offending key if the window is malformed.
    ``shared_source`` and ``model_source`` name the FILES the two values were
    authored in, and appear in the refusal below. Both default to ``None``, so
    the message shape is unchanged when they are absent -- which is what keeps
    this additive for every caller that has only the two mappings.

    They exist because the comparison became genuinely cross-file at R13: the
    simulation window is authored in the build_model settings file and the
    record in the project file. This is the clearest demonstration that the
    shared-seam placement rule is right rather than arbitrary --
    ``historical_window`` is read by three workflows, so it belongs in the
    project file, and a copy planted in a workflow file is refused at parse
    time rather than allowed to become a second record that disagrees.
    """
    where_model = f" (in {model_source})" if model_source else ""
    where_shared = f" (in {shared_source})" if shared_source else ""
    window = get_config(model_cfg, "simulation_window", None)
    if window is None:
        raise ValueError(
            f"workflows.build_model.simulation_window{where_model} is required: "
            "declare {start: YEAR, end: YEAR} in the build_model config. "
            "The Wflow run period must sit inside climate.window; it does not "
            "default to the full climate record"
        )
    # R14 `C-71`: years on this side too, so the two windows a user compares are
    # written in one unit. `historical_window_bounds` is the only parser, so a
    # malformed value is diagnosed once, in one voice, wherever it was authored.
    try:
        start, end = historical_window_bounds(window)
    except ValueError as exc:
        raise ValueError(f"workflows.build_model.simulation_window: {exc}") from None
    if end <= start:
        raise ValueError(
            f"workflows.build_model.simulation_window {start.date()} .. "
            f"{end.date()} ends on or before it starts — check the order"
        )
    rec_start, rec_end = historical_window_bounds(
        get_config(climate_cfg, "window", optional=False)
    )
    if start < rec_start or end > rec_end:
        raise ValueError(
            f"workflows.build_model.simulation_window {start.date()} .. "
            f"{end.date()}{where_model} is not inside climate.window "
            f"{rec_start.date()} .. {rec_end.date()}{where_shared}. The forcing "
            "is built from "
            "the extracted climate store, so a simulation period outside the "
            "record has no data behind it — widen climate.window, or narrow "
            "the simulation window to fit inside it"
        )
    return window


SPATIAL_UNITS_SCRIPT = "blueearth_cst/spatial/delineate_spatial_units.py"


@dataclass(frozen=True)
class SpatialUnitsRule:
    """The producer contract for the shared vector foundation.

    The THIRD member of the shared-rule family, beside :class:`RegionRule` and
    :class:`ClimateStoreRule` and with the same shape. All three workflows
    declare ``delineate_subbasins_and_rivers`` from this object, so the three rule
    bodies cannot drift apart.

    Named ``_rule``, not ``_spec``: the object holds a rule's script, inputs,
    outputs and params, so it IS a rule definition minus its labels. This was
    the first of the three to carry the suffix; the other two joined it in the
    R10 step-6 sweep (``pre-board record 45e3a564`` ``[R10-7]``), so the family is
    uniform again.

    **Name the next one ``<thing>_rule``.** ``_contract`` was rejected — this
    repo uses "contract" for interchange surfaces
    (the engine-interface references, ``SPATIAL_CONTRACT_VERSION``,
    ``test_climate_store_contract.py``) and overloading it would be worse than
    the jargon it replaced; ``_definition`` was the runner-up, rejected on
    verbosity at the call sites.
    """

    spatial_dir: str
    hydrography_nc: str
    script: str
    inputs: Mapping
    outputs: Mapping
    params: Mapping


def spatial_units_rule(project_dir, spatial_config, data_sources) -> SpatialUnitsRule:
    """Build the one producer contract for the shared vector foundation.

    ONE rule definition, declared in all three workflows (``1.01c`` / ``2.03c``
    / ``3.01f``), over the vector half of what rule 1.01 used to do alone.
    Before the shared-region refactor, WF2 and WF3 could not reach the basin and subbasin
    boundaries without declaring ``prepare_land_and_soil_maps`` — whose real product
    is ``spatial_maps.nc`` — so a projections-only run would have resampled
    ``vito``, ``modis_lai`` and ``soilgrids`` to draw a subbasin outline.

    Seven outputs. Six are the declared vector artifacts; the seventh,
    ``hydrography.nc``, is the SEAM INTERMEDIATE. The raster half declares
    it as an input because the whole hydrography grid stack used to cross this
    boundary in memory, and recomputing it would make WF1 read the hydrography
    twice with two grids that can drift. It is deliberately absent from
    ``spatial_catalog.yml``.

    **The params are a pure function of the shared project and basin settings,
    and that is a requirement, not a convenience.** The five projections-only
    configs contain no ``workflows.build_model`` keys at all, so a params
    payload drawn from that section would differ per invoking workflow — the
    input/params asymmetry ``ext1-02`` forbade for the climate store — and
    ``config_path`` itself differs between a full config and a single-workflow
    one, so declaring it as an input would thrash this rule on every WF1/WF2
    alternation. Hence: no ``config_snake`` input, and the deprecated
    ``workflows.build_model.output_locations`` fallback in
    ``resolve_gauge_points_path`` CANNOT feed this rule. Callers must resolve
    ``spatial_config`` with ``parse_spatial_config(basin_cfg)`` — no model
    section. What makes that safe is rule 3.00b, which already guarantees
    ``shared.basin`` agrees across the workflows that share the rule.

    The thematic source names (``lulc``/``lai``/``soil``) are deliberately NOT
    carried: they belong to the raster half, and carrying them would make a
    change to one of them re-run the vector rule in all three workflows.

    Parameters
    ----------
    project_dir : str
        ``project.project_dir``.
    spatial_config : SpatialConfig
        The parsed ``shared.basin`` contract
        (``blueearth_cst.spatial.config.parse_spatial_config``). Read
        attribute-wise rather than imported, so ``shared/`` keeps importing
        nothing from ``spatial/`` — the dependency runs the other way.
    data_sources : str
        ``project.data_sources`` — the hydromt catalog path.

    Returns
    -------
    SpatialUnitsRule
        ``spatial_dir``, ``hydrography_nc``, ``script``, ``inputs``,
        ``outputs``, ``params``.
    """
    spatial_dir = f"{project_dir}/data/spatial"
    geoms_dir = f"{spatial_dir}/geoms"
    hydrography_nc = f"{spatial_dir}/hydrography.nc"

    inputs = {
        "data_catalogs": data_sources,
        # Imported scientific helpers are inputs so Snakemake tracks their edits.
        # Keep this declaration shared by WF0, WF1 and WF2.
        "scientific_code": [
            f"blueearth_cst/spatial/{name}.py"
            for name in (
                "config",
                "products",
                "hydrography",
                "delineation",
                "identity",
                "delineate_region",
            )
        ],
        # `region_rule` owns the path, so the two helpers cannot disagree about
        # where the one project polygon lives -- the same reason
        # `climate_store_rule` resolves it through the helper rather than
        # restating the string.
        "region_geojson": region_rule(
            project_dir,
            spatial_config.region,
            data_sources,
            hydrography=spatial_config.hydrography,
            basin_index=spatial_config.basin_index,
        ).region_geojson,
    }
    # OPTIONAL, and an unset key contributes no entry at all -- the shape rule
    # 1.02 already used for `output_locations`. `resolve_gauge_points_path` has
    # already collapsed both unset spellings (YAML null and the legacy "None"
    # string) to None, so this is the whole test. Declared as an INPUT rather
    # than a param so editing the FILE re-triggers the rule: as a param
    # Snakemake compares the path, and renumbering the gauge points would leave
    # the registry on the old ids in silence.
    if spatial_config.gauge_points_path is not None:
        inputs["gauge_points"] = spatial_config.gauge_points_path

    outputs = {
        "basins": f"{geoms_dir}/basins.geojson",
        "subbasins": f"{geoms_dir}/subbasins.geojson",
        "catchments": f"{geoms_dir}/catchments.geojson",
        # The river NETWORK, derived from this project's own flow direction at
        # `river_uparea_km2` -- the same threshold gauge snapping and the wflow
        # river map use, so all three call the same cells river.
        "rivers": f"{geoms_dir}/rivers.geojson",
        # The catalog's river vector, kept for its WIDTH and BANKFULL DISCHARGE
        # attributes, which hydromt's `setup_rivers` takes as `river_geom_fn`
        # and which cannot be derived from flow direction. Not a network: it
        # carries the global product's own drainage-area floor, and drawing it
        # as one is what left station 1030 with no branch.
        "river_attributes": f"{geoms_dir}/river_attributes.geojson",
        # Physical waterbodies for the study-area map (t2608091730); an empty
        # layer where the basins hold none or the source is missing.
        "reservoirs": f"{geoms_dir}/reservoirs.geojson",
        "lakes": f"{geoms_dir}/lakes.geojson",
        "glaciers": f"{geoms_dir}/glaciers.geojson",
        "locations": f"{geoms_dir}/locations.geojson",
        "location_registry": f"{spatial_dir}/location_registry.csv",
        "hydrography": hydrography_nc,
    }
    return SpatialUnitsRule(
        spatial_dir=spatial_dir,
        hydrography_nc=hydrography_nc,
        script=SPATIAL_UNITS_SCRIPT,
        inputs=inputs,
        outputs=outputs,
        params={
            "hydrography": spatial_config.hydrography,
            "resolution": spatial_config.resolution,
            "river_uparea_km2": spatial_config.river_uparea_km2,
            "rivers_source": spatial_config.sources.rivers,
            "gauge_snap_tolerance_m": spatial_config.gauge_snap_tolerance_m,
            "max_subbasins_per_basin": spatial_config.max_subbasins_per_basin,
        },
    )


def member_pointer_base(config_out_fn) -> tuple[str, str]:
    """Derive one stress-test member's ``run_name`` and output prefix.

    Pure, and it lives here rather than in ``downscale_climate_forcing.py``
    because that module reads the ``snakemake`` global at import time and so
    cannot be imported by a test. Separating it makes the PER-MEMBER KEYING
    testable without a Wflow run.

    Every member-specific pointer the caller writes -- the output CSV, the
    outstate NetCDF, and ``[logging] path_log`` -- is built from these two
    values, so distinct values per member imply distinct pointers per member.
    That is the cheap half of R9 P2's concurrency falsifier. The expensive half
    -- content attribution under a real concurrent batch -- still needs a run,
    because two members could be keyed apart here and still collide if wflow
    ignored the pointer.

    ``out_prefix`` is relative, POSIX and trailing-separated: wflow resolves
    output pointers against ``dirname(toml) + dir_output``, and ``dir_output``
    stays ``"."``, so the config/ -> output/ sibling hop rides in the pointers
    themselves. Keeping the hop out of ``dir_output`` also keeps
    ``semantic_tree_diff``'s TOML comparator correct: it resolves these fields
    lexically against the toml's own directory and does not read ``dir_output``.

    Parameters
    ----------
    config_out_fn : str | Path
        The member's DECLARED run-TOML path, e.g.
        ``experiments/<id>/hydrology/wflow/config/rlz_1_st_2.toml``.

    Returns
    -------
    (run_name, out_prefix)
        The TOML stem, and a relative prefix pointing at the sibling
        ``output/`` directory.
    """
    config_out_fn = Path(config_out_fn)
    config_out_root = os.path.dirname(config_out_fn)
    run_output_dir = Path(config_out_root).parent / "output"
    out_prefix = Path(os.path.relpath(run_output_dir, config_out_root)).as_posix() + "/"
    return config_out_fn.stem, out_prefix


DEFAULT_WFLOW_OUTVARS = ["river discharge", "actual evapotranspiration"]


def member_index_regex(width: int) -> str:
    """Wildcard-constraint regex for a padded member index, excluding all-zeros.

    Two jobs, and the second is why this is exact-width rather than the laxer
    ``0*[1-9][0-9]*``:

    1. **Bar the reserved baseline.** ``st_0`` (``st_00`` at width 2) is written
       by ``generate_weather_realizations``; rule 3.11 must never become a
       second producer of it, which surfaces as a ``CyclicGraphException``
       (``generate_scenarios.smk``, rule 3.08's own comment).
    2. **Reject an UNPADDED name outright.** At width 2, ``st_1`` fails to match
       and Snakemake raises ``MissingRuleException`` rather than routing it.
       A lax pattern would accept both spellings, so a producer that forgot to
       pad would silently agree with the DAG — the same invisible
       producer/declaration disagreement that broke this milestone's first two
       rename attempts.

    **ANCHOR-FREE, and that is not a style choice.** The obvious spelling is a
    negative lookahead, ``(?!0+$)[0-9]{width}``. It is WRONG here: Snakemake
    embeds a wildcard's constraint inside the regex for the WHOLE path, so the
    ``$`` anchors to the end of that path rather than to the end of the
    wildcard. With ``.nc`` always following, ``0+$`` can never match, the
    lookahead always succeeds, and the constraint silently degenerates to
    ``[0-9]{width}`` -- which admits the baseline and makes rule 3.11 a second
    producer of it. Caught by ``test_cross_workflow_inputs`` and
    ``test_guard_invalidation`` as a ``CyclicGraphException``, and NOT by a
    plain ``--dry-run``, because whether the ambiguity surfaces depends on the
    DAG shape: where the baseline is also reachable from its own plural rule,
    Snakemake prefers that one (fewer wildcards) and the degeneracy stays
    hidden.

    So the not-all-zeros condition is spelled positionally instead: the index is
    exactly ``width`` digits, of which the first NON-zero one sits at some
    position ``k``. Alternating over ``k`` covers every value except all-zeros,
    with no anchor and no lookahead.

        width 1 -> [1-9]
        width 2 -> [1-9][0-9]|0[1-9]        (10..99 and 01..09, never 00)
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(
            f"member_index_regex needs a positive int width, got {width!r}"
        )
    branches = [
        f"0{{{k}}}[1-9][0-9]{{{width - 1 - k}}}".replace("0{0}", "").replace(
            "[0-9]{0}", ""
        )
        for k in range(width)
    ]
    return branches[0] if width == 1 else "(?:" + "|".join(branches) + ")"


def _wflow_frame_relay():
    """A :class:`~blueearth_cst.shared.progress.WflowFrameRelay`, or a no-op.

    Imported HERE rather than at module scope because ``progress`` pulls in
    ``dask.callbacks``, and every Snakefile imports this module at PARSE time --
    where the tee is never used. ``tee_to_log`` runs in the ``run_logged.py``
    child, so the cost lands only where the bar is actually drawn.
    """
    try:
        from blueearth_cst.shared.progress import WflowFrameRelay
    except Exception:  # pragma: no cover - defensive; see _NoFrameRelay
        return _NoFrameRelay()
    return WflowFrameRelay()


def listed(values, limit=6):
    """``a, b, c (+37 more)`` -- a list that cannot outgrow its console line.

    A row interpolating a joined list has a length the DATA decides, not the
    author: one basin strands two locations off the river network and another
    strands forty. Rewording cannot bound that, and the rows it affects are
    exactly the diagnostic ones a reader needs to be able to read.

    **The overflow is always stated.** A list that silently stopped at six
    would be a tool bounding its own coverage without saying so, which this
    repo refuses on principle (AGENTS.md, "no silent caps"): the reader would
    have no way to tell a basin with six stranded locations from one with
    sixty. The full set stays in the rule's log part either way.

    ``limit`` is generous on purpose. The point is to stop a pathological row,
    not to make a normal one terse -- a five-element list reads better whole
    than as three and a count.
    """
    items = [str(value) for value in values]
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit} more)"


def warning_tally_path(project_dir, log_name):
    """The file a run counts its warnings in.

    Beside the merged log it describes and named after it, NOT under
    ``logs/_parts/``: :func:`~blueearth_cst.shared.merge_logs.merge_logs`
    prunes that directory once it has folded the parts into the merged log,
    and the verdict reads this tally AFTER that rule has run. A tally living
    there would either be litter the prune leaves behind or keep the directory
    alive past the rule meant to remove it.
    """
    return os.path.join(os.fspath(project_dir), "logs", f".{log_name}.tally")


def declare_warning_tally(project_dir, log_name):
    """Open a fresh tally for this run and publish it to every job process.

    Called once per Snakefile at parse time, beside
    :func:`declare_path_tokens`. Truncating HERE rather than at first write is
    what keeps a rerun honest: the file would otherwise carry every previous
    run's rows and the verdict would report a number that only grows.

    **Mutates the process environment**, which in a run is the point and in a
    test is a leak: a test calling this must let ``monkeypatch`` touch
    ``_WARNING_TALLY_ENV`` FIRST, or the variable outlives it and every later
    warning in the session lands in one tally.
    """
    path = warning_tally_path(project_dir, log_name)
    os.environ[_WARNING_TALLY_ENV] = path
    try:
        # REMOVED, not truncated-in-place: parsing is not running. A `--dry-run`
        # and a DAG contract test both reach this line, and creating the file
        # here put an artifact inside every project either one touched. The
        # first warning creates it (:func:`note_warning`); a run that prints
        # none leaves nothing behind, which is the honest trace of a clean run.
        os.remove(path)
    except OSError:
        # Absent already, or unreachable. A tally that cannot be managed costs
        # the verdict its field and nothing else. Never the run.
        pass
    return path


def warning_count():
    """How many warning rows this run printed, or ``None`` when untracked.

    ``None`` and ``0`` are different answers and the verdict spells them
    differently: no tally was declared, versus a run that printed no warning.
    """
    path = os.environ.get(_WARNING_TALLY_ENV)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        # Declared but never written: a run that printed no warning. That is a
        # count of zero, not an absent tally.
        return 0
    except OSError:
        return None


def save_figure(
    path, module="plot", fig=None, *, annotation="none", title="", caption="", **kwargs
):
    """Save a matplotlib figure to ``path`` and announce it as part of a bundle.

    Centralizes the "write a figure + log one line" pattern for the plotting
    ``script:`` rules: every produced map/plot is accounted for in the rule's
    log instead of the log being empty or showing only upstream library
    chatter. Parent directories are created. ``kwargs`` pass through to
    ``Figure.savefig`` (e.g. ``dpi``, ``bbox_inches``). matplotlib is
    imported lazily so this module stays light for the Snakefiles that import it
    only for ``get_config`` / ``stress_test_grid``.
    The neutral exporter owns parent creation and the explicit-figure save.

    **One row per DIRECTORY, not per figure.** A single wf0 plotting rule
    writes 33 figures, and a row each made the figures the bulk of the whole
    workflow's console -- 54 of wf0's ~117 rows, all of them the same sentence
    with one word changed. The rows are accumulated by directory and emitted on
    flush::

        21:50:31 - plot - 9 figures -> <climate>/era5_20000101_20161231/plots
        21:50:31 - plot - 24 figures -> <climate>/era5_.../plots/subbasins

    The previous scheme printed the directory once and then indented bare file
    names under it, which halved the width but not the row COUNT, and it broke
    down exactly where it was needed most: these rules alternate between a
    parent directory and its ``subbasins`` child, so "state the directory when
    it changes" restated it on nearly every other row. Accumulating per
    directory rather than tracking only the last one is what makes the
    alternation collapse to two rows instead of twelve.

    Naming the directory rather than dropping the paths entirely is the
    deliberate half: a log is the artifact someone sends you when a run went
    wrong, and a bare count would leave "where did they go" answerable only
    from the Snakefile. A per-file record still exists -- on disk, and in the
    rule's declared outputs.

    Flushing is automatic (:func:`flush_figure_bundles`): any other ``log_row``
    closes the open bundles first, so a module's own summary line still reads
    after the figures it summarizes, and ``tee_to_log`` drains whatever is left
    when the log closes.

    ``fig`` defaults to the current figure, which is what every historical
    caller relies on. Pass it explicitly when one plot writes MORE THAN ONE
    file (a vector deliverable plus a raster preview): two saves that both
    resolve "current figure" through global pyplot state are a silent
    correctness trap the moment any intervening code creates a figure.
    """
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.export import export_figure

    if fig is None:
        # Preserve failure before gcf creates an implicit figure. The exporter
        # repeats this idempotent parent creation after resolving the figure.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig = plt.gcf()
    export_figure(
        fig, path, annotation=annotation, title=title, caption=caption, **kwargs
    )
    # Grouped by ABSOLUTE directory, so two callers spelling one directory
    # differently (relative vs absolute, `./plots` vs `plots`) still group. The
    # row is printed from this spelling and relativized by the tee like any
    # other path, so the absolute form never reaches a reader.
    directory = os.path.dirname(os.path.abspath(path))
    _FIGURE_BUNDLES.setdefault(directory, []).append(os.path.basename(path))
    _FIGURE_BUNDLE_MODULES[directory] = module


def patch_psutil_windows_benchmark():
    """Work around Snakemake's benchmark sampler crashing on Windows.

    Snakemake's benchmark monitor reads ``psutil.memory_full_info().pss`` on
    every sample, but on Windows psutil's ``pfullmem`` has ``uss`` and **no**
    ``pss`` — the resulting ``AttributeError`` aborts every sample before the
    record is marked collected, so ALL indicators (rss/vms/uss/io/load/cpu_time,
    not just pss) come out ``NA``. This shim exposes ``pss`` (= ``uss`` as a
    Windows proxy) so the sampler succeeds and the real indicators populate.

    No-op off Windows, when psutil is absent, or when ``pss`` already exists.
    Called at the top of each Snakefile so it is active in the Snakemake process
    that runs the benchmark threads. Upstream Snakemake bug; shimmed in our own
    code rather than editing the vendored package.
    """
    if sys.platform != "win32":
        return
    try:
        import psutil
    except ImportError:
        return
    from collections import namedtuple

    orig = psutil.Process.memory_full_info
    if getattr(orig, "_cst_pss_shim", False):
        return  # already patched

    def _with_pss(self):
        meminfo = orig(self)
        if hasattr(meminfo, "pss"):
            return meminfo
        tuple_with_pss = namedtuple("pfullmem_pss", list(meminfo._fields) + ["pss"])
        return tuple_with_pss(*meminfo, meminfo.uss)

    _with_pss._cst_pss_shim = True
    psutil.Process.memory_full_info = _with_pss


def run_and_tee(command, log_path):
    """Preserve the WF0/WF4 Wflow frame relay on the legacy helper surface."""
    from blueearth_cst.shared.run_log_core import run_and_tee as run_core

    return run_core(command, log_path, frame_relay_factory=_wflow_frame_relay)
