"""Shared helpers for the BlueEarth-CST Snakefiles.

Imported by all three ``Snakefile_*`` entry points (and ``tests/conftest.py``)
so the ``get_config`` contract lives in exactly one place. Each Snakefile makes
this module importable regardless of the working directory by prepending its
own directory to ``sys.path`` before importing — see
``dev/r03/model-builder-design.md`` §3.
"""

import contextlib
import logging
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


# hydromt formats every log record as
# ``<ts> - <name> - <module> - <LEVEL> - <message>`` (its hardcoded
# ``_LOG_FORMAT``; no CLI/env/config override exists). ``<ts>`` is a full
# ``YYYY-MM-DD HH:MM:SS,mmm`` stamp, ``<name>`` the dotted logger path
# (``hydromt.model.model``), ``<module>`` its leaf (``model``) — all verbose or
# redundant per row. We cannot change hydromt's format (vendored, off-limits),
# so both tee paths below rewrite matching lines into *our* logs: drop the
# dotted ``<name>`` (keep ``<module>`` as a short subsystem tag) and shorten the
# stamp to ``HH:MM:SS`` (the date lives once in the log header, not on every
# row). Only lines matching this exact shape are rewritten; everything else
# (Julia/Wflow output, tracebacks, plain prints) passes through verbatim.
_HYDROMT_LOG_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2}),\d{3} - \S+ - (\S+) - (\w+) - (.*)$"
)


def _compact_log_line(text):
    """Compact a hydromt-format log line: ``HH:MM:SS`` stamp, drop dotted name.

    ``<YYYY-MM-DD HH:MM:SS,mmm> - <name> - <module> - <LEVEL> - <msg>`` becomes
    ``<HH:MM:SS> - <module> - <LEVEL> - <msg>``. A trailing newline is preserved.
    Non-matching text is returned unchanged, so the tee stays faithful for all
    output that is not a single hydromt log record.
    """
    had_newline = text.endswith("\n")
    core = text[:-1] if had_newline else text
    match = _HYDROMT_LOG_RE.match(core)
    if not match:
        return text
    hms, module, level, message = match.groups()
    return f"{hms} - {module} - {level} - {message}" + ("\n" if had_newline else "")


def _log_path_parts(log_path):
    """Return ``(project_root, log_id)`` derived from a rule log path.

    The parent of the first ``logs`` / ``benchmarks`` path component is the
    project dir; the path below that anchor is the rule-log id (so wildcard
    sub-logs read e.g. ``3.10_run_wflow/rlz_1_cst_1.log``). Both are ``""`` /
    the bare basename when the anchor is absent (e.g. an ad-hoc test path).
    """
    log_path = os.fspath(log_path)
    parts = os.path.normpath(log_path).split(os.sep)
    for anchor in ("logs", "benchmarks"):
        if anchor in parts:
            i = parts.index(anchor)
            root = os.sep.join(parts[:i]) if i > 0 else ""
            log_id = "/".join(parts[i + 1:]) or os.path.basename(log_path)
            return root, log_id
    return "", os.path.basename(log_path)


def _relativize_paths(text, project_root):
    """Rewrite absolute project paths in ``text`` as project-relative.

    Strips the ``project_root`` prefix (in both native and forward-slash forms,
    since hydromt emits either) so a log line like
    ``Writing geoms to C:\\...\\gabon\\hydrology_model\\...\\basins.geojson``
    reads ``Writing geoms to hydrology_model\\...\\basins.geojson``. Paths
    outside the project (data catalogs, the pixi env) are left absolute.
    """
    if not project_root:
        return text
    text = text.replace(project_root + os.sep, "")
    text = text.replace(project_root.replace(os.sep, "/") + "/", "")
    return text


def _log_header_lines(path, kind="log", time_label="started", markdown=False):
    """Return the provenance header block for a rule log or merged artifact.

    Carries the project name and run date (the date dropped from each row by
    ``_compact_log_line``), the full project dir, and the artifact id + a
    timestamp, followed by a blank line separating it from the body.

    ``kind``/``time_label`` name the third line for the artifact type — a log is
    ``log: <id> | started <t>``, a benchmark table ``benchmark: <id> | generated
    <t>``. With ``markdown=True`` the same lines are wrapped in a fenced code
    block so they render as one metadata box in a ``.md`` file instead of as a
    stack of ``#`` H1 headings; otherwise each line is a ``#`` comment (a log's
    plain-text convention).
    """
    now = datetime.now()
    root, log_id = _log_path_parts(path)
    project = os.path.basename(root) if root else ""
    project_field = f"project: {project} | " if project else ""
    lines = [f"BlueEarth-CST | {project_field}{now:%Y-%m-%d}"]
    if root:
        lines.append(f"project dir: {root.replace(os.sep, '/')}")
    lines.append(f"{kind}: {log_id} | {time_label} {now:%H:%M:%S}")
    if markdown:
        body = "\n".join(lines)
        return f"```text\n{body}\n```\n\n"
    # plain-text log: each line a `# ` comment, then a blank line before the body
    return "".join(f"# {line}\n" for line in lines) + "\n"


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
    (dev/p31/experiment-structure-design.md §3b/§3c, ext2-2). Called at
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


# Directory names under the repository root that are EXEMPT from the
# in-repo project_dir warning. Only the tracked test fixture: the baseline seed
# config is version-controlled and a tracked config cannot carry a
# machine-specific absolute path (design § "Two-tier project_dir rule").
_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})


def warn_if_project_dir_in_repo(project_dir, repo_root) -> bool:
    """Warn when ``project_dir`` resolves inside the repository tree.

    Makes the two-tier rule mechanical instead of documentary: production runs
    write outside the toolbox source, and the one exemption is the in-repo
    ``test_case/`` fixture. Called at parse time from all three Snakefiles with
    ``workflow.basedir`` as ``repo_root``.

    Warns; never raises. An in-repo project_dir is a smell, not an error --
    raising would break the fixture-driven baseline gate and anyone who
    deliberately keeps a scratch run inside a checkout.

    ``repo_root`` is a parameter rather than derived from ``__file__``:
    deriving it inside the module silently breaks if the package is ever
    installed rather than imported from the checkout, and an absolute constant
    is not portable across machines. The call sites already hold the value.

    Returns True when a warning was emitted, so callers and tests can assert on
    the decision rather than on captured output.
    """
    try:
        pd_resolved = Path(project_dir).expanduser().resolve()
        root_resolved = Path(repo_root).expanduser().resolve()
    except (OSError, ValueError):  # unresolvable path: nothing to warn about
        return False

    # commonpath, not startswith: "test_caseX" must not read as inside
    # "test_case", and str-prefix comparisons get that wrong.
    try:
        inside = os.path.commonpath([pd_resolved, root_resolved]) == str(
            root_resolved
        )
    except ValueError:  # different drives on Windows -> definitively outside
        return False
    if not inside:
        return False

    rel = pd_resolved.relative_to(root_resolved)
    if rel.parts and rel.parts[0] in _PROJECT_DIR_EXEMPT_NAMES:
        return False

    warnings.warn(
        f"project_dir resolves inside the repository tree "
        f"({rel.as_posix()!r} under {root_resolved}). Generated model and "
        f"result artifacts should be written OUTSIDE the toolbox source; set "
        f"project_dir to an absolute path elsewhere. Exempt: "
        f"{'/'.join(sorted(_PROJECT_DIR_EXEMPT_NAMES))}/.",
        UserWarning,
        stacklevel=2,
    )
    return True


_EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_EXPERIMENT_NAME_MAX_LEN = 64
# Windows reserved device names (compared case-insensitively, incl. any
# extension): CON, PRN, AUX, NUL, COM1-9, LPT1-9. A path segment equal to one of
# these (with or without an extension) is invalid on Windows.
_WINDOWS_RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


def validate_experiment_name(name: str, project_dir) -> str:
    """Validate ``experiment_name`` as a safe ``experiments/<name>/`` path segment.

    Centralized slug validation for the wf3 experiment subtree
    (dev/p31/experiment-structure-design.md §2b). Called once at
    ``Snakefile_climate_experiment`` parse time, BEFORE ``exp_dir`` (and every
    derived output/params path) is built, so all paths are constructed only from
    a vetted value. Parse-time is correct here: a malformed name makes the entire
    DAG ill-defined, so failing under ``--dry-run`` is the intended behavior
    (unlike the drift *guard*, which is a rule so ``--unlock`` stays usable).

    Grammar: ``^[a-z0-9][a-z0-9_]*$`` (lowercase alnum + underscore, must start
    with an alnum), nonempty, at most 64 chars — a strict subset of
    ``dev/conventions/naming.md``'s snake_case rule that deliberately excludes
    hyphens and dots so the value can never introduce a path component or an
    extension. Uppercase is REJECTED (never silently lowercased). After the
    grammar, a containment assertion confirms the resolved target is a direct
    child of ``<project_dir>/experiments`` (belt to the grammar's braces).

    Returns the validated ``name`` unchanged, or raises ``ValueError`` naming the
    offending input.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"experiment_name must be a non-empty string, got {name!r}"
        )
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


def slugify_window(start, end) -> str:
    """Render a window ``(start, end)`` to a compact ``YYYYMMDD_YYYYMMDD`` slug.

    Builds the dataset-store key component for the wf3 historical-climate store
    (dev/p31/experiment-structure-design.md §4/§4c/§4d). The store dir is
    ``climate_historical/<clim_source>_<start>_<end>/`` where ``<start>``/``<end>``
    are this function's output. The window endpoints are ISO
    ``YYYY-MM-DDTHH:MM:SS``; ``:`` is illegal in Windows paths, so time-of-day and
    separators are stripped to ``YYYYMMDD``.

    Day-resolution invariant (§4c): the store is keyed at day resolution, so two
    windows differing ONLY below the day boundary would render to the same key
    yet request different bounds — a silent stale-reuse. This helper therefore
    **asserts** ``HH:MM:SS == 00:00:00`` on both endpoints and raises
    ``ValueError`` otherwise, failing loud instead of colliding.

    Parameters
    ----------
    start, end : str
        Window endpoints as ISO ``YYYY-MM-DDTHH:MM:SS`` (or ``YYYY-MM-DD``).

    Returns
    -------
    str
        ``"<YYYYMMDD>_<YYYYMMDD>"``.

    Raises
    ------
    ValueError
        If an endpoint is not parseable at day resolution, or carries a nonzero
        time-of-day component.
    """
    def _day_slug(value, which):
        text = str(value).strip()
        # Split date from an optional time-of-day on the 'T' separator (or a space).
        if "T" in text:
            date_part, time_part = text.split("T", 1)
        elif " " in text:
            date_part, time_part = text.split(" ", 1)
        else:
            date_part, time_part = text, ""
        try:
            dt = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"historical_window {which} {value!r} is not a YYYY-MM-DD date"
            ) from exc
        if time_part:
            # Accept only an all-zero time-of-day; anything else is sub-day
            # resolution the day-keyed store cannot represent (§4c). Drop any
            # fractional seconds, then check every digit is zero.
            hms = time_part.split(".", 1)[0]
            if hms.replace(":", "").strip("0") != "":
                raise ValueError(
                    f"historical_window {which} {value!r} has a nonzero "
                    "time-of-day; the store key is day-resolution (§4c) — "
                    "sub-day windows are not supported"
                )
        return dt.strftime("%Y%m%d")

    return f"{_day_slug(start, 'starttime')}_{_day_slug(end, 'endtime')}"


def _require_step_num(axis_cfg, axis_name):
    """Read and validate a required ``step_num`` from a stress-test axis section.

    Strict by contract: a missing axis section or ``step_num`` raises
    ``KeyError`` (parity with ``prepare_cst_parameters.py``'s direct read); a
    ``step_num`` that is not a non-negative integer raises ``ValueError``.
    ``bool`` is rejected — ``True``/``False`` are not valid grid step counts.
    """
    step_num = axis_cfg[axis_name]["step_num"]  # KeyError on missing axis/key
    if isinstance(step_num, bool) or not isinstance(step_num, int):
        raise ValueError(
            f"stress_test.{axis_name}.step_num must be a non-negative int, "
            f"got {step_num!r}"
        )
    if step_num < 0:
        raise ValueError(
            f"stress_test.{axis_name}.step_num must be non-negative, got {step_num}"
        )
    return step_num


def stress_test_grid(stress_test_cfg: Mapping) -> tuple[int, int, int]:
    """Return ``(temp_step_count, precip_step_count, st_num)`` for a stress_test cfg.

    Single source of truth for the stress-test grid arithmetic, which was
    previously derived twice (inline in ``Snakefile_climate_experiment`` and in
    ``blueearth_cst/experiment/prepare_cst_parameters.py``). Both call sites now read this helper.

    STRICT: ``temp.step_num`` and ``precip.step_num`` are REQUIRED — a missing
    axis section or ``step_num`` raises ``KeyError``, and a value that is not a
    non-negative integer raises ``ValueError``. The helper never silently
    invents a grid. Per-axis step count is ``step_num + 1`` (endpoints
    inclusive), and ``st_num = temp_step_count * precip_step_count``.

    Parameters
    ----------
    stress_test_cfg : Mapping
        The ``workflows.climate_experiment.stress_test`` config section, with
        ``temp`` and ``precip`` axis sub-sections each carrying ``step_num``.

    Returns
    -------
    tuple[int, int, int]
        ``(temp_step_count, precip_step_count, st_num)``.

    Raises
    ------
    KeyError
        If the ``temp``/``precip`` axis section or its ``step_num`` is absent.
    ValueError
        If a ``step_num`` is not a non-negative integer.
    """
    temp_step_count = _require_step_num(stress_test_cfg, "temp") + 1
    precip_step_count = _require_step_num(stress_test_cfg, "precip") + 1
    return temp_step_count, precip_step_count, temp_step_count * precip_step_count


def _fmt_elapsed(seconds):
    """Format a duration compactly: ``45s``, ``2m14s``, ``1h03m20s``."""
    seconds = int(round(seconds))
    hours, minutes, secs = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _Heartbeat:
    """Console-only watchdog that makes a stalled rule visible while it runs.

    Snakemake prints only a start and a finish timestamp, so a hung job looks
    identical to a slow one until it (never) finishes. This daemon prints an
    elapsed-time notice when the rule has produced no output for ``interval``
    seconds, and a one-line ``done in <elapsed>`` summary when it stops.

    Silence-triggered, not periodic: callers stamp ``touch()`` on every real
    write, so a rule that is actively logging or drawing a progress bar keeps
    resetting the clock and never beeps — the notice appears exactly when the
    console would otherwise be frozen, which is the "is it stuck?" case. A lone
    ``time.monotonic()`` float assignment is atomic under the GIL, so ``touch()``
    needs no lock.

    Writes **only** to ``stream`` (the live console, captured before any tee
    swap); nothing here ever reaches the rule's log file — the persisted log
    stays clean. Set ``CST_HEARTBEAT_SECS`` (``0`` disables entirely) to override
    the interval without touching a Snakefile.
    """

    def __init__(self, label, stream, interval=60.0):
        self._label = label
        self._stream = stream
        raw = os.environ.get("CST_HEARTBEAT_SECS")
        try:
            self._interval = float(raw) if raw is not None else float(interval)
        except ValueError:
            self._interval = float(interval)
        self._enabled = self._interval > 0
        self._start = time.monotonic()
        self._last = self._start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def touch(self):
        self._last = time.monotonic()

    def _emit(self, text):
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            pass  # console I/O must never break the job

    def _run(self):
        while not self._stop.wait(self._interval):
            if time.monotonic() - self._last >= self._interval:
                elapsed = _fmt_elapsed(time.monotonic() - self._start)
                self._emit(f"   ... {self._label}: still running, {elapsed} elapsed\n")

    def start(self):
        if self._enabled:
            self._thread.start()
        return self

    def stop(self, failed=False):
        if not self._enabled:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        elapsed = _fmt_elapsed(time.monotonic() - self._start)
        verb = "failed after" if failed else "done in"
        self._emit(f"   ... {self._label}: {verb} {elapsed}\n")


def _cr_overwrite(line):
    """Collapse a carriage-return-redrawn line to its final visible text.

    Emulates a terminal: each ``\\r`` returns the cursor to column 0 so later
    text overwrites earlier text on the same line. Progress bars (e.g. dask's
    ``[####] | 100% Completed | 7.08 s``) redraw the full-width bar on every
    ``\\r``, so the *last non-empty* segment is the final state. Filtering empty
    segments is load-bearing: dask ends its stream with a bare ``\\r`` before the
    newline, and a plain ``rsplit`` would keep that trailing empty piece and blank
    the whole bar. A line with no ``\\r`` is returned unchanged.
    """
    if "\r" not in line:
        return line
    segments = [s for s in line.split("\r") if s]
    return segments[-1] if segments else ""


class _Tee:
    """Text stream mirroring in-process output to a live console and a log file.

    Deliberately not an ``io`` subclass: ``script:`` rules only ``print`` /
    log through ``sys.stdout``/``sys.stderr``, so ``write`` + ``flush`` (plus
    ``isatty``) is all that is needed. Note: this operates at the Python
    stream level, so output from *shell* subprocesses (which inherit the real
    file descriptors) is not captured — only in-process Python output is.

    The ``live`` sink (console) gets output verbatim, so a carriage-return
    progress bar still animates in place during a long ``to_netcdf``. The
    ``logfile`` sink instead receives each line *after* carriage-return overwrite
    (see ``_cr_overwrite``), so the persisted log keeps only the final rendered
    state of an in-place-updated line rather than every redraw. Partial (not yet
    newline-terminated) output is held in ``_pending`` and collapsed on the fly,
    so a bar redrawing for hours never grows the buffer beyond one line.
    """

    def __init__(self, live, logfile, project_root="", on_activity=None):
        self._live = live
        self._logfile = logfile
        self._project_root = project_root
        self._on_activity = on_activity  # called on each write (heartbeat reset)
        self._pending = ""  # current, not-yet-newline-terminated log line

    def write(self, text):
        if self._on_activity is not None:
            self._on_activity()
        out = _relativize_paths(_compact_log_line(text), self._project_root)
        self._live.write(out)  # verbatim: keeps the live console animation
        buf = self._pending + out
        lines = buf.split("\n")
        self._pending = lines.pop()  # trailing fragment, no newline yet
        for line in lines:
            self._logfile.write(_cr_overwrite(line) + "\n")
        self._pending = _cr_overwrite(self._pending)  # keep the buffer bounded
        return len(text)

    def flush(self):
        # Flush the sinks but NOT ``_pending``: emitting a mid-progress fragment
        # would re-clutter the log with every partial redraw.
        self._live.flush()
        self._logfile.flush()

    def close(self):
        # Flush any trailing partial line (e.g. a progress bar cut short by an
        # error before its final newline) so nothing is silently dropped.
        if self._pending:
            self._logfile.write(_cr_overwrite(self._pending) + "\n")
            self._pending = ""
        self._logfile.flush()

    def isatty(self):
        return False


# Benign CPython interpreter-shutdown noise. A subprocess -- notably the verbose
# ``hydromt build wflow_sbm ... -vv`` step -- can emit a repeating
# ``Error in sys.excepthook:`` / ``Original exception was:`` cascade with EMPTY
# bodies *after* it has finished successfully (rc=0), when a stderr write fails
# during interpreter finalization (many GDAL/rasterio datasets torn down at once
# on Windows). It floods the tail of an otherwise-clean log. Triaged as cosmetic
# in dev/phase-1/m01/warnings.md; ``run_and_tee`` collapses a *pure* trailing run
# of these into one summary line. A real traceback puts non-empty content
# between the markers, so it is never collapsed (see ``_is_shutdown_noise``).
_EXCEPTHOOK_MARKERS = ("Error in sys.excepthook:", "Original exception was:")


def _is_shutdown_noise(line):
    """True if ``line`` is a shutdown-excepthook marker or a blank line.

    Only pure marker/blank lines are collapsible. A genuine excepthook failure
    interleaves the markers with an actual traceback (``Traceback (most recent
    call last):`` ...); those body lines return False here, which breaks the
    candidate block and forces it to be emitted verbatim -- so no real error is
    ever hidden by the collapse.
    """
    stripped = line.strip()
    return stripped == "" or stripped in _EXCEPTHOOK_MARKERS


def run_and_tee(command, log_path):
    """Run ``command`` (an argv list), streaming combined stdout+stderr to the
    console AND ``log_path``, and return the child's exit code.

    Replaces the ``<cmd> 2>&1 | tee {log}`` idiom in ``shell:`` rules. A bare
    ``| tee`` pipeline returns *tee*'s exit status, not the command's, unless
    bash ``pipefail`` is active -- and Snakemake injects no ``pipefail`` prefix
    on Windows/cmd.exe, so a failed ``hydromt``/``julia`` step is misread as
    success (t260721a; dev/followups.md). Teeing in-process restores exit-code
    fidelity while keeping live console output. The child runs with
    ``shell=False`` so argument quoting is preserved identically across cmd.exe
    and bash (e.g. Julia's ``-e "using Wflow; Wflow.run()"`` stays one argv).

    A *pure* trailing run of benign interpreter-shutdown excepthook noise (see
    ``_EXCEPTHOOK_MARKERS``) is collapsed into a single summary line so it does
    not bury the real end of the log. The collapse is conservative: candidate
    lines are buffered, and any real content flushes them verbatim, so the
    filter only ever fires on a genuinely empty-bodied shutdown cascade.

    Parameters
    ----------
    command : list[str]
        Program and arguments, already tokenized (as a ``shell:`` rule's words
        arrive after ``--``).
    log_path : str | os.PathLike
        Destination log file; parent directories are created.

    Returns
    -------
    int
        The child process's return code.
    """
    log_path = os.fspath(log_path)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    project_root, log_id = _log_path_parts(log_path)
    label = os.path.splitext(log_id)[0]
    if label.startswith("_parts/"):
        label = label[len("_parts/"):]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(_log_header_lines(log_path))  # header to file only, not console
        log.flush()

        def emit(text):
            # Compact hydromt's redundant log format (see _compact_log_line) and
            # show project files relative to the project dir; non-hydromt lines
            # and out-of-project paths pass through unchanged.
            text = _relativize_paths(_compact_log_line(text), project_root)
            # The log file is UTF-8. The live console mirror may be a legacy
            # code page (cp1252 on Windows) that cannot encode glyphs the child
            # emits (e.g. Julia/Wflow progress-bar blocks); fall back to a lossy
            # encode for the console only — the log always gets the real text.
            try:
                sys.stdout.write(text)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                sys.stdout.write(text.encode(enc, "replace").decode(enc))
            sys.stdout.flush()
            log.write(text)
            log.flush()

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            # Decode the child's pipe as UTF-8. Julia/Wflow (and Python under
            # UTF-8 mode) emit UTF-8; without this, text mode uses the Windows
            # locale code page (cp1252) and mangles non-ASCII — a `█` (UTF-8
            # E2 96 88) decoded as cp1252 becomes "â–ˆ". ASCII-only children
            # (hydromt logs) are unaffected; `errors="replace"` guards any
            # genuinely non-UTF-8 byte instead of crashing the tee.
            encoding="utf-8",
            errors="replace",
        )
        # Silence watchdog: prints an elapsed-time notice to the console (stderr,
        # never the log) if the child goes quiet — so a hung Julia/Wflow/hydromt
        # step is visible live. Touched on every line read from the child.
        heartbeat = _Heartbeat(label, sys.stderr).start()
        # ``pending`` holds a trailing run of candidate shutdown-noise lines that
        # are withheld until we know whether real content follows (flush
        # verbatim) or the stream ends (collapse if it is a true cascade).
        rc = None
        try:
            pending = []
            for line in proc.stdout:
                heartbeat.touch()
                if _is_shutdown_noise(line):
                    pending.append(line)
                    continue
                for buffered in pending:
                    emit(buffered)
                pending = []
                emit(line)
            rc = proc.wait()
            _flush_pending(pending, emit, rc)
        finally:
            heartbeat.stop(failed=(rc is None or rc != 0))
        return rc


def _flush_pending(pending, emit, rc):
    """Emit the trailing candidate block: collapse a real cascade, else verbatim.

    Collapse only when the block holds at least two markers (one full
    ``excepthook``/``original`` unit); a smaller or marker-free tail is emitted
    unchanged so nothing real is dropped.
    """
    marker_count = sum(1 for ln in pending if ln.strip() in _EXCEPTHOOK_MARKERS)
    if marker_count >= 2:
        emit(
            f"[run_logged] collapsed {len(pending)} benign interpreter-shutdown "
            f"lines (repeated 'Error in sys.excepthook:' / 'Original exception "
            f"was:'; child rc={rc})\n"
        )
    else:
        for buffered in pending:
            emit(buffered)


def _set_handler_stream(handler, stream):
    """Repoint a logging handler's stream, using ``setStream`` when available."""
    if hasattr(handler, "setStream"):
        handler.setStream(stream)  # flushes the old stream first (py3.7+)
    else:
        handler.stream = stream


def _redirect_console_log_handlers(orig_out, orig_err, stdout_tee, stderr_tee):
    """Route pre-existing console logging handlers through the tees.

    A library can install a ``StreamHandler`` bound to the real ``sys.stdout`` /
    ``sys.stderr`` at import time — hydromt does, on the ``hydromt`` logger, in
    its full ``<date> - <name> - <module> - <LEVEL> - <msg>`` format. Because it
    captured the stream object *before* ``tee_to_log`` swaps the streams, its
    records bypass ``_Tee`` entirely: uncompacted on the console and **missing
    from the log file**. Repointing each such handler at the matching ``_Tee``
    makes those records flow through the one shared pipeline (``_compact_log_line``
    + path relativization + log file), so every workflow — in-process (hydromt
    Python API) or subprocess (``run_and_tee``) — emits one identical style.

    Matches the console streams by *identity*, so real ``FileHandler``s (whose
    stream is a file, never ``is`` the console) are untouched. Returns a list of
    ``(handler, original_stream)`` for ``_restore_log_handlers`` to undo.
    """
    loggers = [logging.getLogger()]  # root, then every concrete (non-placeholder) logger
    loggers += [
        lg for lg in logging.Logger.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
    ]
    saved = []
    for lg in loggers:
        for handler in getattr(lg, "handlers", []):
            stream = getattr(handler, "stream", None)
            if stream is orig_out:
                target = stdout_tee
            elif stream is orig_err:
                target = stderr_tee
            else:
                continue
            saved.append((handler, stream))
            _set_handler_stream(handler, target)
    return saved


def _restore_log_handlers(saved):
    """Undo ``_redirect_console_log_handlers`` (restore each handler's stream)."""
    for handler, stream in saved:
        _set_handler_stream(handler, stream)


@contextlib.contextmanager
def tee_to_log(log_path, heartbeat_interval=60.0):
    """Tee ``sys.stdout``/``sys.stderr`` to ``log_path`` for a ``script:`` rule.

    Snakemake does not auto-redirect ``script:`` output to the rule's ``log:``
    (unlike ``shell:`` rules), so a script wraps its body in this manager and
    passes ``snakemake.log[0]``.

    Contract (R3 design §6):
    - creates ``log_path`` and any missing parent directories;
    - both streams are restored in a ``finally`` — the redirection cannot leak
      past the ``with`` block even if the body raises;
    - the exception is **re-raised** (not swallowed), so the traceback still
      reaches Snakemake and the rule fails loudly rather than leaving an empty
      log that Snakemake would read as a finished product.

    A silence watchdog (``_Heartbeat``) prints an elapsed-time notice to the
    live console when the rule goes quiet for ``heartbeat_interval`` seconds, so
    a stalled job is visible while it runs. It writes to the console only — the
    log file never receives a heartbeat line. ``CST_HEARTBEAT_SECS`` overrides
    the interval (``0`` disables it).

    Library logging bound to the console before entry (hydromt's ``StreamHandler``
    on ``sys.stdout``) is repointed through the tee for the duration, so its
    records get the same compacted ``HH:MM:SS - <module> - <LEVEL> - <msg>`` form
    and land in the log file instead of bypassing it (see
    ``_redirect_console_log_handlers``).

    Parameters
    ----------
    log_path : str | os.PathLike
        Destination log file. Callers pass the rule's unique
        ``snakemake.log[0]`` so concurrent jobs never share a path.
    heartbeat_interval : float
        Seconds of silence before the console heartbeat fires (default 60).
    """
    log_path = os.fspath(log_path)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    project_root, log_id = _log_path_parts(log_path)
    label = os.path.splitext(log_id)[0]
    if label.startswith("_parts/"):
        label = label[len("_parts/"):]
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(_log_header_lines(log_path))  # header to file only
        handle.flush()
        # heartbeat writes to the real console (orig_err), never the log handle
        heartbeat = _Heartbeat(label, orig_err, interval=heartbeat_interval)
        stdout_tee = _Tee(orig_out, handle, project_root=project_root, on_activity=heartbeat.touch)
        stderr_tee = _Tee(orig_err, handle, project_root=project_root, on_activity=heartbeat.touch)
        sys.stdout, sys.stderr = stdout_tee, stderr_tee
        # route library logging (hydromt) bound to the old console through the tee
        saved_handlers = _redirect_console_log_handlers(orig_out, orig_err, stdout_tee, stderr_tee)
        heartbeat.start()
        try:
            yield
        finally:
            # Restore log handlers first (before their target tees close), stop
            # the watchdog (console-only summary), flush trailing partial lines
            # while ``handle`` is open, then restore the streams — all always run,
            # even if the body raised.
            _restore_log_handlers(saved_handlers)
            heartbeat.stop(failed=sys.exc_info()[0] is not None)
            for tee in (stdout_tee, stderr_tee):
                tee.close()
            sys.stdout, sys.stderr = orig_out, orig_err


def log_row(message, module="cst", level="INFO"):
    """Print one log row in the standard compact format used across rule logs.

    ``HH:MM:SS - <module> - <LEVEL> - <message>`` — the same shape
    ``_compact_log_line`` produces for hydromt records, so a ``script:`` rule's
    own messages sit uniformly among the hydromt/library lines rather than as
    bare, timestamp-less text. Use this instead of a plain ``print`` for anything
    meant to appear in a rule log. The row is already compact, so the tee passes
    it through (only any project paths in it are relativized).
    """
    print(f"{datetime.now():%H:%M:%S} - {module} - {level} - {message}")


def save_figure(path, module="plot", **kwargs):
    """Save the current matplotlib figure to ``path`` and announce it cleanly.

    Centralizes the "write a figure + log one line" pattern for the plotting
    ``script:`` rules: every produced map/plot appears in the rule's log as a
    standard row ``HH:MM:SS - <module> - INFO - Saved figure: <path>`` (via
    ``log_row``) instead of the log being empty or showing only upstream
    library chatter. Parent directories are created. ``kwargs`` pass through to
    ``matplotlib.pyplot.savefig`` (e.g. ``dpi``, ``bbox_inches``). matplotlib is
    imported lazily so this module stays light for the Snakefiles that import it
    only for ``get_config`` / ``stress_test_grid``.
    """
    import matplotlib.pyplot as plt

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(path, **kwargs)
    log_row(f"Saved figure: {path}", module=module)


def patch_psutil_windows_benchmark():
    """Work around Snakemake's benchmark sampler crashing on Windows.

    Snakemake's benchmark monitor reads ``psutil.memory_full_info().pss`` on
    every sample, but on Windows psutil's ``pfullmem`` has ``uss`` and **no**
    ``pss`` — the resulting ``AttributeError`` aborts every sample before the
    record is marked collected, so ALL metrics (rss/vms/uss/io/load/cpu_time,
    not just pss) come out ``NA``. This shim exposes ``pss`` (= ``uss`` as a
    Windows proxy) so the sampler succeeds and the real metrics populate.

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


def rule_banner(number, name):
    """Return a rule's ``message:`` string: a bold, numbered console banner.

    Shows ``<W.NN>  <name>`` (the ``W.NN`` matching the rule's log/benchmark
    filenames) so the live Snakemake console is easy to track. The number+name
    are wrapped in bold cyan **only** when stderr is a TTY and ``NO_COLOR`` is
    unset — so piping/redirecting the console to a file leaves no escape codes.
    Evaluated once at Snakefile parse time (a plain string, no wildcards).
    """
    tag = f"{number}  {name}"
    if sys.stderr.isatty() and not os.environ.get("NO_COLOR"):
        return f"\033[1;36m{tag}\033[0m"  # bold cyan
    return tag
