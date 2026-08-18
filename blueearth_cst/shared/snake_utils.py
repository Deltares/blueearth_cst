"""Shared helpers for the BlueEarth-CST Snakefiles.

Imported by all three ``*.smk`` entry points (and ``tests/conftest.py``)
so the ``get_config`` contract lives in exactly one place. Each Snakefile makes
this module importable regardless of the working directory by prepending its
own directory to ``sys.path`` before importing — see
``dev/milestones/r03/model-builder-design.md`` §3.
"""

import contextlib
import gc
import io
import json
import logging
import os
import posixpath
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

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

# hydromt_wflow ALSO prefixes many messages with the model component they write:
# ``wflow_sbm.geoms: Writing geoms to ...``. Where that prefix's leaf repeats the
# ``<module>`` column the row already carries, one row names the same subsystem
# twice — measured on a full WF1 build, 35 of 272 rows (and 31 of them `geoms`).
_COMPONENT_PREFIX_RE = re.compile(r"^\w+\.(\w+): (.*)$")


#: Drops the data TYPE from hydromt's catalog-read line, the single commonest
#: row a build prints:
#:
#:     data_source - Reading merit_hydro_ihu RasterDataset data from <data>/...
#:     data_source - Reading merit_hydro_ihu from <data>/...
#:
#: The type is already implied twice over -- by the ``data_source`` column, and
#: by the entry's own declaration in the catalog the header names -- so on a
#: build that reads two dozen sources it is a column of noise between the two
#: facts a reader wants, WHICH source and from WHERE.
#:
#: The five names are ENUMERATED rather than matched as ``\S+`` (they are
#: ``hydromt.data_catalog.sources``' concrete ``data_type`` values). A type this
#: does not know then survives into the line instead of being silently eaten,
#: which is the right way round for a cosmetic filter sitting on someone else's
#: message: an upstream addition shows up as a slightly noisy row, never as a
#: quietly mangled one. Longest-first, so ``Dataset`` cannot claim the tail of
#: ``RasterDataset``.
_DATA_SOURCE_READ_RE = re.compile(
    r"^(Reading \S+) "
    r"(?:RasterDataset|GeoDataFrame|GeoDataset|DataFrame|Dataset)"
    r" data from "
)

#: Second half of the same row: once the TYPE is gone, ``Reading <name> from
#: <dir>/<name>.nc`` still says ``<name>`` twice. That is not a corner case in
#: WF3 — every downscale rule reads its realization by its own file stem, so a
#: rapid experiment prints 20 such rows, each ~100 chars of which ~18 are the
#: repeat. The name is dropped only when it is EXACTLY the file's stem, so a
#: catalog key that genuinely differs from its file name still shows both.
_DATA_SOURCE_FROM_RE = re.compile(r"^Reading (\S+) from (\S+)$")


def _log_row_text(hms, module, level, message):
    """Assemble one log row: ``HH:MM:SS - <module> - <message>``.

    **The level is shown only when it is not INFO**, and that is the point
    rather than a saving. On a full WF1 build 259 of 272 rows are INFO, so the
    field is a constant column on all but a handful — and a WARNING or an ERROR,
    the rows someone is scanning for, hides inside a wall of identical
    ``- INFO -``. Omitting the common case makes the uncommon one visible:

        11:13:01 - geoms - Writing geoms to basins.geojson.
        11:13:04 - states - WARNING - state file not found, using cold start

    Absence therefore MEANS ``INFO``; the log header says so. Every row in these
    logs goes through here -- :func:`_compact_log_line` for hydromt's records,
    :func:`log_row` for our own, and the heartbeat's stall markers -- because
    three emitters with three spellings of one grammar is how the grammar stops
    being one.
    """
    if str(level).strip().upper() == "INFO":
        return f"{hms} - {module} - {message}"
    return f"{hms} - {module} - {level} - {message}"


def _compact_log_line(text):
    """Compact a hydromt-format log line: ``HH:MM:SS`` stamp, drop dotted name.

    ``<YYYY-MM-DD HH:MM:SS,mmm> - <name> - <module> - <LEVEL> - <msg>`` becomes
    ``<HH:MM:SS> - <module> - <LEVEL> - <msg>``. A trailing newline is preserved.
    Non-matching text is returned unchanged, so the tee stays faithful for all
    output that is not a single hydromt log record.

    A message-leading ``<model>.<component>: `` prefix is dropped when
    ``<component>`` is the ``<module>`` column — the same rule that drops the
    dotted ``<name>``, applied one field along: a row states its subsystem once.
    ``geoms - INFO - wflow_sbm.geoms: Writing geoms to x.geojson`` becomes
    ``geoms - Writing geoms to x.geojson`` (the level goes too -- see
    :func:`_log_row_text`).

    The prefix is KEPT when the two differ, which is a third of them:
    ``spatial - INFO - wflow_sbm.staticmaps: ...`` names hydromt's code module
    and the model component being written, and those are two different facts.

    Trimming the message rather than the ``<module>`` column is deliberate. The
    column is what makes every row in these logs the same four fields, which is
    what a reader scans and what ``merge_logs`` and ``log_row`` both assume;
    dropping it only on the rows that happen to carry a prefix would make the
    shape conditional to buy the same characters.
    """
    had_newline = text.endswith("\n")
    core = text[:-1] if had_newline else text
    match = _HYDROMT_LOG_RE.match(core)
    if not match:
        return text
    hms, module, level, message = match.groups()
    prefixed = _COMPONENT_PREFIX_RE.match(message)
    if prefixed and prefixed.group(1) == module:
        message = prefixed.group(2)
    message = _DATA_SOURCE_READ_RE.sub(r"\1 from ", message)
    message = _drop_repeated_source_name(message)
    return _log_row_text(hms, module, level, message) + ("\n" if had_newline else "")


def _drop_repeated_source_name(message):
    """Rewrite ``Reading <name> from <dir>/<name>.ext`` as ``Reading <dir>/<name>.ext``."""
    match = _DATA_SOURCE_FROM_RE.match(message)
    if not match:
        return message
    name, path = match.groups()
    stem = posixpath.splitext(posixpath.basename(path.replace("\\", "/")))[0]
    return f"Reading {path}" if stem == name else message


def _log_path_parts(log_path):
    """Return ``(project_root, log_id)`` derived from a rule log path.

    The parent of the first ``logs`` / ``benchmarks`` path component is the
    project dir; the path below that anchor is the rule-log id (so wildcard
    sub-logs read e.g. ``3.10_run_wflow/rlz_1_st_1.log``). Both are ``""`` /
    the bare basename when the anchor is absent (e.g. an ad-hoc test path).
    """
    log_path = os.fspath(log_path)
    parts = os.path.normpath(log_path).split(os.sep)
    for anchor in ("logs", "benchmarks"):
        if anchor in parts:
            i = parts.index(anchor)
            root = os.sep.join(parts[:i]) if i > 0 else ""
            log_id = "/".join(parts[i + 1 :]) or os.path.basename(log_path)
            return root, log_id
    return "", os.path.basename(log_path)


#: Repo root — three levels up from ``blueearth_cst/shared/snake_utils.py``.
#: Used only to shorten log lines, never to resolve anything.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

#: Everything up to and including a ``site-packages`` component. An installed
#: dependency's own file (hydromt_wflow's ``parameters_data.yml``) says nothing
#: useful in its first hundred characters, and those characters differ per
#: machine and per environment.
_SITE_PACKAGES_RE = re.compile(r"[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])*?site-packages[\\/]")


#: The path-looking run immediately after a stripped prefix. Stops at
#: whitespace and at the quote/bracket characters that end a path in a
#: traceback (``File "C:\\...\\x.py", line 3``) or a log message.
_STRIPPED_TAIL_RE = r"[^\s\"'<>|,;)\]}]*"


def _strip_prefix(text, prefix, replacement=""):
    """Drop ``prefix`` from ``text`` in both native and forward-slash spellings.

    The remainder is normalized to FORWARD SLASHES. Without that, one log mixes
    both spellings of the same tree -- ``data/climate/historical/...`` from a
    library that builds paths with ``/`` beside
    ``data\\climate\\historical\\...\\era5_precip_annual_clim_map_basin_ext.png``
    from one that used
    ``os.path.join`` -- and the two read as different locations at a glance.

    Normalization is deliberately scoped to the run of text FOLLOWING a stripped
    prefix, not applied to the whole line. A log line is prose as well as paths,
    and a blanket replace would rewrite Windows paths this function deliberately
    leaves absolute (a data catalog under ``C:\\data\\``, whose location is the
    information), regex escapes, and any other literal backslash.

    **Both spellings are derived from the prefix, not from the prefix as
    given.** The earlier form appended ``os.sep`` to the prefix verbatim and
    then replaced ``os.sep`` within it, which silently produced ONE spelling
    whenever the prefix arrived already forward-slashed -- exactly what a
    ``project_dir`` read from a shipped config is (``test_case/test_rapid``).
    A rule that printed the same folder OS-natively (``test_case\\test_rapid\\
    config\\runs``, from a ``pathlib`` value interpolated into a message) then
    matched neither spelling, so the row kept the full path AND its
    backslashes while every neighbouring row was stripped and forward-slashed.
    Canonicalizing first makes the two spellings genuinely independent of how
    the caller happened to write the prefix.
    """
    if not prefix:
        return text
    canonical = os.fspath(prefix).replace("\\", "/")
    for spelling in (canonical + "/", canonical.replace("/", "\\") + "\\"):
        text = re.sub(
            re.escape(spelling) + f"({_STRIPPED_TAIL_RE})",
            lambda m: replacement + m.group(1).replace("\\", "/"),
            text,
        )
    return text


#: Where a workflow's declared key folders travel. A rule's output is written by
#: a CHILD of the process that parses the Snakefile -- a ``script:`` module, or
#: an R/Julia command under ``run_logged.py`` -- and none of them is handed the
#: config. An env var is what crosses that boundary without threading a
#: parameter through fifteen ``tee_to_log`` call sites, and it is inherited by
#: every child for free.
_PATH_TOKENS_ENV = "CST_PATH_TOKENS"


def declare_path_tokens(**folders):
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
    """
    tokens = {}
    for name, folder in folders.items():
        if folder is None or not str(folder).strip():
            continue
        tokens[name] = os.path.normpath(os.path.abspath(os.fspath(folder)))
    os.environ[_PATH_TOKENS_ENV] = json.dumps(tokens)
    return tokens


#: The run's project directory, for rules that print a path OUTSIDE a rule log.
#: Travels the same way and for the same reason as `_PATH_TOKENS_ENV`.
_PROJECT_ROOT_ENV = "CST_PROJECT_ROOT"


def declare_project_root(project_dir):
    """Declare the run's project directory for :func:`log_row` to shorten against.

    ``tee_to_log`` already relativizes everything a rule with a ``log:``
    directive prints, so this exists for the rules that have NONE -- the config
    snapshot (0.01/1.01/2.01/3.01) is the standing example. Those rows reach the
    console without passing the tee, so they were the only ones in a run still
    printing ``test_case\\test_rapid\\config\\runs\\...`` -- full length, and
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


def _declared_tokens(environ=None):
    """Read the declared folders as ``(name, path)`` in DECLARATION order.

    The order a workflow declared them in is the order they read best in a
    header -- external data, then the model, then the run's own outputs -- and
    it survives the round trip because both ``dict`` and JSON objects preserve
    insertion order. Matching wants a different order entirely; see
    :func:`_path_tokens`.
    """
    raw = (os.environ if environ is None else environ).get(_PATH_TOKENS_ENV)
    if not raw:
        return ()
    try:
        tokens = json.loads(raw)
    except ValueError:
        return ()
    if not isinstance(tokens, dict):
        return ()
    return tuple(
        (str(name), str(path))
        for name, path in tokens.items()
        if isinstance(path, str) and path.strip()
    )


def _path_tokens(environ=None):
    """Read the declared folders as ``(name, path)``, LONGEST PATH FIRST.

    The order is the whole reason this is not a plain dict iteration: an
    experiment directory sits under the project and a model directory can sit
    under either, so a shorter path that prefixes a longer one would claim it
    first and ``<experiment>/hydrology/wflow`` would come out as
    ``<project>/experiments/x/hydrology/wflow``. Longest first, always.

    Fail-open: an absent or malformed variable yields no tokens and every path
    prints in full, which is the behaviour this whole mechanism improves on.
    """
    pairs = _declared_tokens(environ)
    return tuple(sorted(pairs, key=lambda pair: len(pair[1]), reverse=True))


def _tokenize_prefix(text, prefix, token):
    """Replace ``prefix`` (and anything below it) with ``<token>/...``.

    Unlike :func:`_strip_prefix` this also matches the directory NAMED ON ITS
    OWN, with no trailing separator -- ``Write model data to <...>/wflow`` is
    one of the commonest lines a build prints, and a rewrite that needed a
    separator would leave exactly the line that states the folder untouched
    while shortening every line below it.

    The trailing lookahead is what stops ``<...>/wflow`` from also claiming
    ``<...>/wflow_extra``: with no separator-led tail, the character after the
    prefix must be one that cannot continue a path.

    Both separator spellings are derived from a canonicalized prefix, for the
    reason set out in :func:`_strip_prefix`: deriving them from the prefix as
    given collapses to one spelling whenever the caller already used ``/``.
    """
    if not prefix:
        return text
    tail = f"([\\\\/]{_STRIPPED_TAIL_RE})?"
    guard = r"(?![^\s\"'<>|,;)\]}])"
    canonical = os.fspath(prefix).replace("\\", "/")
    for spelling in (canonical + "/", canonical.replace("/", "\\") + "\\"):
        spelling = spelling[:-1]  # match the folder named on its own too
        text = re.sub(
            re.escape(spelling) + tail + guard,
            lambda m: f"<{token}>" + (m.group(1) or "").replace("\\", "/"),
            text,
        )
    return text


def _relativize_paths(text, project_root, tokens=()):
    """Shorten the three absolute prefixes that dominate a log line.

    Every rule log is full of paths whose leading two-thirds are the same on
    one machine and different on the next, which buries the part that carries
    information. Three prefixes are rewritten, in decreasing specificity:

    * the **project** — dropped, so ``C:\\...\\gabon\\hydrology_model\\...\\
      basins.geojson`` reads ``hydrology_model\\...\\basins.geojson``. The
      project root is stated once in the log header, so no information is lost.
    * the **repository** — dropped and marked, so a config or script under the
      checkout reads ``<repo>/config/catalogs/deltares_data.yml``. Marked rather
      than bare because a repo-relative path and a project-relative one would
      otherwise be indistinguishable in the same line.
    * an installed **dependency** — everything up to ``site-packages`` becomes
      ``<site-packages>/``, so hydromt_wflow's own
      ``.../envs/default/Lib/site-packages/hydromt_wflow/data/parameters_data.yml``
      reads ``<site-packages>/hydromt_wflow/data/parameters_data.yml``. What
      matters is which package the file came from, not where pixi put the env.

    Order matters: the repo contains the pixi env, so ``site-packages`` is
    matched FIRST or a repo-relative rewrite would hide it. A path in none of
    the three (a data catalog under ``C:\\data\\``) is left absolute — its
    location is the information.

    ``tokens`` are the run's DECLARED key folders (see
    :func:`declare_path_tokens`), and they are applied BEFORE the project strip
    for a mechanical reason: by then a path under the project has already lost
    its root, so a token registered in absolute form — which is the only form
    that can match what a rule prints — would find nothing left to match. They
    are also what makes an external data root shortenable at all; the paragraph
    above is still true of an UNDECLARED one.
    """
    text = _SITE_PACKAGES_RE.sub("<site-packages>/", text)
    # Each token in BOTH the absolute spelling it was declared in and the
    # project-relative one, longest first so a nested token still wins.
    #
    # The relative spelling is not redundant. A plotting rule builds its output
    # path from the config's own relative `project_dir`, so it prints
    # `test_case/test_rapid/data/climate/historical/<store>/plots/x.png` --
    # already relative, and therefore unmatched by an absolute token. The
    # declaration was then applied to nothing and the project strip below
    # reduced the row to `data/climate/historical/<store>/plots/x.png`: the
    # header declared `<climate>` as a legend for lines that never used it,
    # which is worse than not declaring it, and left 25 characters of constant
    # prefix on every figure row.
    for token, path in _spelt_tokens(tokens, project_root):
        text = _tokenize_prefix(text, path, token)
    # BOTH spellings of the project root, absolute first. `project_dir` is
    # relative in every shipped config (`test_case/test_rapid`), and the two
    # emitters disagree about which form they print: hydromt resolves, so it
    # prints `C:\...\pipeline\test_case\test_rapid\models\...`, while `log_row`
    # prints the configured `test_case\test_rapid\config\runs` as given.
    #
    # Stripping only the relative form was actively WRONG on the absolute one:
    # `_strip_prefix` is unanchored, so it excised the root from the MIDDLE and
    # left the head, which the repo rewrite below then labelled -- turning
    # `<...>/test_case/test_rapid/models/hydrology/wflow` into
    # `<repo>/models/hydrology/wflow`, a path that does not exist, presented
    # as if it did. Absolute first because it is the longer of the two.
    if project_root:
        text = _strip_prefix(text, os.path.abspath(os.fspath(project_root)))
    text = _strip_prefix(text, project_root)
    return _strip_prefix(text, _REPO_ROOT, "<repo>/")


def _spelt_tokens(tokens, project_root):
    """Return ``(token, path)`` in every spelling a rule might print, longest first.

    A declared folder is stored absolute (:func:`declare_path_tokens`), but a
    rule that joined its output onto a relative ``project_dir`` prints the
    relative form. Both are yielded so the token matches either, and the result
    is sorted longest-path-first for the nesting reason in :func:`_path_tokens`
    -- a relative spelling is shorter than its own absolute one, so appending
    without re-sorting would let a short token claim a longer token's path.
    """
    spellings = []
    root = os.path.abspath(os.fspath(project_root)) if project_root else ""
    for token, path in tokens:
        spellings.append((token, path))
        if not root:
            continue
        relative = _strip_prefix(path, root)
        # `_strip_prefix` is unanchored and returns the text unchanged when it
        # does not match, so an unrelated token (an external data root) yields
        # its own absolute path back and must not be added a second time.
        if relative != path and relative:
            spellings.append((token, os.path.join(os.fspath(project_root), relative)))
    return tuple(sorted(spellings, key=lambda pair: len(pair[1]), reverse=True))


def _folder_rows(project_root, tokens=None):
    """Return ``(<name>, path)`` rows defining the run's declared key folders.

    A folder under the project is shown PROJECT-RELATIVE, because that is
    exactly how every path below it prints in the body, and one that is not
    (an external data root) is shown absolute for the same reason. The header
    carrying these rows always states the project dir itself, so the two forms
    cannot be confused.
    """
    # ABSOLUTE, because the tokens are: `project_dir` is relative in every
    # shipped config, and stripping a relative root off an absolute token
    # excises it from the middle -- the `<model>` row then read
    # `C:/.../pipeline/models/hydrology/wflow`, a path that does not exist,
    # offered as the definition of the token. abspath resolves against the
    # working directory, which is where Snakemake resolves `project_dir` too.
    root = os.path.abspath(os.fspath(project_root)) if project_root else ""
    rows = []
    for token, path in _declared_tokens() if tokens is None else tokens:
        shown = _strip_prefix(path, root) if root else path
        rows.append((f"<{token}>", shown.replace(os.sep, "/")))
    return rows


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

    A log also gets a one-line legend for its rows, because
    :func:`_log_row_text` omits the level on the INFO rows that are almost all
    of them: a reader who does not know that reads the absence as a defect, and
    the next person to "fix" it puts the constant column back. Logs only — a
    benchmark table has no rows of this shape.
    """
    now = datetime.now()
    root, log_id = _log_path_parts(path)
    project = os.path.basename(root) if root else ""
    project_field = f"project: {project} | " if project else ""
    lines = [f"BlueEarth-CST | {project_field}{now:%Y-%m-%d}"]
    if root:
        lines.append(f"project dir: {root.replace(os.sep, '/')}")
    lines.append(f"{kind}: {log_id} | {time_label} {now:%H:%M:%S}")
    if kind == "log":
        # The declared key folders, because the rows below refer to them by
        # name. A `<model>/staticmaps.nc` in a log read months from now is
        # strictly worse than the long path it replaced unless the log itself
        # says what `<model>` was -- so the definition travels with the rows,
        # not just with the console that scrolled away.
        lines.extend(f"{label}: {value}" for label, value in _folder_rows(root))
        lines.append("rows: HH:MM:SS - module - message | level shown unless INFO")
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
    (dev/milestones/p31/experiment-structure-design.md §3b/§3c, ext2-2). Called at
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
        inside = os.path.commonpath([pd_resolved, root_resolved]) == str(root_resolved)
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


def project_slug(project_dir, reserve: int = 0) -> str:
    """Slugify ``project_dir``'s basename into an experiment-name stem.

    The shared stem behind both naming paths: ``suggest_experiment_name``
    (which appends a date and writes the result into a config) and the
    workflow's own unset-key default (which appends a date only when no dated
    experiment for this project exists yet). Extracted so the two cannot derive
    a different stem from one ``project_dir``.

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

    R07 B8. A *suggestion writer*, never a runtime generator: a name derived
    fresh on every run would make each invocation target a new
    ``experiments/<id>/``, so nothing would ever be up to date, incremental
    reruns would be impossible, ``--dry-run`` would mislead, and the baseline
    gate would have no fixed path. The helper is invoked once, deliberately,
    and the value it writes is then read as an ordinary config key.

    The workflow's unset-key default (``allocate.resolve_default_experiment_name``)
    reaches the same name without a config edit, and avoids the trap by
    **reusing** an existing dated experiment instead of minting today's. This
    command remains the way to pin a deliberate name, choose one with
    ``--name``, or start a fresh experiment beside an existing one.

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

    Centralized slug validation for the wf3 experiment subtree
    (dev/milestones/p31/experiment-structure-design.md §2b). Called once at
    ``run_stress_test.smk`` parse time, BEFORE ``exp_dir`` (and every
    derived output/params path) is built, so all paths are constructed only from
    a vetted value. Parse-time is correct here: a malformed name makes the entire
    DAG ill-defined, so failing under ``--dry-run`` is the intended behavior
    (unlike the drift *guard*, which is a rule so ``--unlock`` stays usable).

    Grammar: ``^[a-z0-9][a-z0-9_]*$`` (lowercase alnum + underscore, must start
    with an alnum), nonempty, at most 64 chars — a strict subset of
    ``dev/reference/naming.md``'s snake_case rule that deliberately excludes
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


#: The advanced-settings file: toolbox-wide constraints and defaults that no
#: normal project edits. Repo root is two levels up from
#: ``blueearth_cst/shared/``. NOT a ``--configfile`` target — the Snakefiles
#: take a per-project ``config/workflows/snake_config_*.yml``; this one is read
#: once, here, and applies to every project.
ADVANCED_SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "advanced_settings.yml"
)

#: The closed schema: section -> {key: validator}. Closed on purpose — an
#: unknown section or key is REJECTED rather than ignored, so a typo in the
#: settings file fails loudly instead of silently leaving the built-in value in
#: force (the same fail-loud stance ``get_config`` takes for project configs).
#: A new setting is added HERE and in the file together.
_ADVANCED_SETTINGS_SCHEMA = {
    "constraints": {"min_historical_years": "positive_int"},
    "defaults": {
        "julia_threads": "positive_int",
        "seed": "nonnegative_int",
        "water_year_start": "month_abbrev",
    },
    "runtime": {"julia_version": "version_string"},
}

#: Three-part ``X.Y.Z``. Two parts would let juliaup resolve a different patch
#: than ``Manifest.toml`` was built against.
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _positive_int(value, where: str) -> int:
    """A whole number >= 1, rejecting the bool that ``isinstance(x, int)`` admits."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{where} must be >= 1, got {value}")
    return value


def _nonnegative_int(value, where: str) -> int:
    """A whole number >= 0. Separate from ``_positive_int`` because 0 is a
    legitimate randomization seed, and rejecting it would be an arbitrary hole
    in the accepted range rather than a constraint anything needs."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{where} must be >= 0, got {value}")
    return value


#: Three-letter month abbreviations, index + 1 == calendar month number. The
#: config surface for the water year is this spelling rather than an integer:
#: ``Oct`` cannot be misread, whereas ``10`` invites "tenth month" vs "offset of
#: ten", and it is the spelling ``start_month_hyd_year`` already used.
_MONTH_ABBREVS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)  # fmt: skip


def _month_abbrev(value, where: str) -> str:
    """A three-letter month name, normalized to ``Jan``-style capitalization.

    Defined up here with the other settings validators rather than beside the
    water-year helpers below: ``_VALIDATORS`` is built at module level, so a
    later definition would be a NameError at import.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a three-letter month name like 'Oct', got "
            f"{value!r} ({type(value).__name__})"
        )
    token = value.strip().capitalize()
    if token not in _MONTH_ABBREVS:
        raise ValueError(
            f"{where} must be one of {', '.join(_MONTH_ABBREVS)}, got {value!r}"
        )
    return token


def _version_string(value, where: str) -> str:
    """A quoted three-part version.

    The non-string rejection is load-bearing rather than defensive: unquoted
    ``1.11`` in YAML parses to the FLOAT 1.11, which would silently become the
    selector ``+1.11`` and let juliaup pick whatever patch it likes.
    """
    if not isinstance(value, str):
        raise ValueError(
            f'{where} must be a quoted string like "1.11.7", got {value!r} '
            f"({type(value).__name__}) — an unquoted X.Y is parsed as a number"
        )
    if not _VERSION_RE.match(value):
        raise ValueError(f"{where} must be a three-part version X.Y.Z, got {value!r}")
    return value


_VALIDATORS = {
    "positive_int": _positive_int,
    "nonnegative_int": _nonnegative_int,
    "month_abbrev": _month_abbrev,
    "version_string": _version_string,
}


def load_advanced_settings(path=None) -> dict:
    """Read and validate ``config/advanced_settings.yml``.

    Returns ``{section: {key: value}}``. Raises ``ValueError`` naming the
    offending section or key on anything the schema does not admit: a missing
    section, a missing key, an unknown section, an unknown key, or a value that
    fails its validator.

    Deliberately has NO built-in fallback. A silent fallback would mean a
    deleted or mistyped settings file changes what the toolbox enforces without
    saying so — exactly the failure mode the closed schema exists to prevent.
    The file is tracked; if it is absent the checkout is broken, and that should
    be said plainly at import.
    """
    settings_path = Path(path) if path is not None else ADVANCED_SETTINGS_PATH
    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"advanced settings file not found at {settings_path}. It is "
            f"tracked in the repository; a checkout without it cannot state "
            f"what the toolbox enforces"
        ) from None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{settings_path} is not a YAML mapping")

    unknown_sections = sorted(set(raw) - set(_ADVANCED_SETTINGS_SCHEMA))
    if unknown_sections:
        raise ValueError(
            f"{settings_path}: unknown section(s) {unknown_sections}; expected "
            f"{sorted(_ADVANCED_SETTINGS_SCHEMA)}"
        )

    resolved = {}
    for section, keys in _ADVANCED_SETTINGS_SCHEMA.items():
        if section not in raw:
            raise ValueError(f"{settings_path}: missing section {section!r}")
        body = raw[section]
        if not isinstance(body, Mapping):
            raise ValueError(f"{settings_path}: section {section!r} is not a mapping")
        unknown_keys = sorted(set(body) - set(keys))
        if unknown_keys:
            raise ValueError(
                f"{settings_path}: unknown key(s) {unknown_keys} in section "
                f"{section!r}; expected {sorted(keys)}"
            )
        resolved[section] = {}
        for key, validator in keys.items():
            if key not in body:
                raise ValueError(f"{settings_path}: missing {section}.{key}")
            resolved[section][key] = _VALIDATORS[validator](
                body[key], f"{section}.{key}"
            )
    return resolved


ADVANCED_SETTINGS = load_advanced_settings()

#: THE minimum historical window, in whole calendar years — one floor for the
#: whole toolbox, enforced identically wherever the window is checked (owner
#: ruling 2026-08-01). The VALUE lives in
#: ``config/advanced_settings.yml`` under ``constraints:``, with the reasoning
#: for the number itself; what follows is why the check is shaped this way.
#:
#: Deliberately NOT a per-workflow floor. An earlier revision enforced 365 days
#: hard and warned at 16 years, so WF1 could build a model on a record WF3 would
#: later reject — the failure simply moved to the workflow least able to explain
#: it. One number, one message, checked in both places a different fact is
#: knowable: the REQUESTED window at parse time
#: (``validate_historical_window``) and the ACTUAL extracted span in
#: ``extract_historical_climate._check_window_coverage``.
#:
#: It subsumes the other length requirement in the tree: rule 1.11
#: (Retired 2026-08-09, ADR 0006.) ``plot_results`` wrote the subcatchment
#: climate figures only from >= 365
#: TIMESTEPS, which 16 years clears for any daily source (dev/followups-archive.md
#: R7-6).
MIN_HISTORICAL_YEARS = ADVANCED_SETTINGS["constraints"]["min_historical_years"]


#: juliaup version selector for every Julia invocation a workflow makes. The
#: VALUE lives in ``config/advanced_settings.yml`` under ``runtime:``, together
#: with why Julia sits outside pixi at all.
#:
#: THREE files declare it and must agree — the settings file, ``pixi.toml``'s
#: ``install-julia`` task, and ``Manifest.toml``'s ``julia_version``. Only the
#: first is readable from here; the other two cannot read YAML, so the equality
#: is enforced by ``tests/test_julia_runtime.py`` rather than by single-sourcing.
JULIA_VERSION = ADVANCED_SETTINGS["runtime"]["julia_version"]

#: Default ``--threads`` for Wflow.jl. The VALUE lives in
#: ``config/advanced_settings.yml`` under ``defaults:``; a project may override
#: it with ``shared.julia_threads`` (P3-3 design §6.3, which sanctions exactly
#: this: "optionally promote --threads to a config value so a deployment can
#: tune it to its basin without a Snakefile edit").
#:
#: Deliberately NOT wired to Snakemake's ``threads:`` directive. Snakemake CAPS
#: a rule's threads at ``--cores``, so ``-c 3`` would quietly hand Wflow 3
#: threads instead of 4 — a thread-allocation change disguised as a refactor,
#: and precisely what §5.6 forbids. The two numbers are independent by design:
#: the nominal budget is ``N x t <= C_logical``.
DEFAULT_JULIA_THREADS = ADVANCED_SETTINGS["defaults"]["julia_threads"]


def validate_julia_threads(value) -> int:
    """Validate ``shared.julia_threads`` as a positive whole number of threads.

    Parse-time, like the other config validators here: the value lands in a
    ``shell:`` body, so a bad one would otherwise surface as a Julia usage error
    inside a rule rather than as a config problem. Same predicate the settings
    file's own ``defaults.julia_threads`` is held to.
    """
    return _positive_int(value, "shared.julia_threads")


#: Default randomization seed for every stochastic step. The VALUE lives in
#: ``config/advanced_settings.yml`` under ``defaults:``; a project overrides it
#: with ``shared.seed``, which also accepts the literal ``auto``.
#:
#: One key, deliberately, rather than a seed per stochastic component. Today the
#: only consumer is weathergenr (generation AND perturbation, C34/F15), but a
#: second one reading its own key is how two halves of a run end up with
#: different reproducibility guarantees that nobody chose.
DEFAULT_SEED = ADVANCED_SETTINGS["defaults"]["seed"]

#: ``derive_seed`` reduces modulo this. R's integer type tops out at 2**31 - 1
#: and ``set.seed`` takes an integer, so a larger value would arrive in R as a
#: double and either warn or truncate.
_SEED_MODULUS = 2**31


def derive_seed(experiment_name: str) -> int:
    """The seed ``shared.seed: auto`` resolves to, from the experiment name.

    Deterministic on the name alone: the same experiment re-runs with the same
    seed (so nothing downstream re-runs), a different experiment gets a
    different one, and "what seed did experiment X use?" stays answerable
    forever without reading any artifact.

    **``zlib.crc32``, never the builtin ``hash``.** ``hash`` is salted per
    process by ``PYTHONHASHSEED``, so it would return a different seed for the
    same experiment in every interpreter — silently breaking the one property
    this function exists to provide, and doing it in a way that looks like
    reproducible behaviour until two runs are compared.
    """
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError(
            f"cannot derive a seed from experiment_name={experiment_name!r}: "
            "`shared.seed: auto` needs the experiment's name, which WF3 always "
            "resolves before rule 3.10 runs"
        )
    return zlib.crc32(experiment_name.encode("utf-8")) % _SEED_MODULUS


def resolve_seed(value, experiment_name: str) -> int:
    """Resolve ``shared.seed`` to the integer the generator is handed.

    ``None`` (key absent) takes ``defaults.seed``; ``auto`` derives from the
    experiment name; anything else must be a non-negative integer. A string
    that is not ``auto`` is refused rather than coerced — ``seed: "123"`` and
    ``seed: random`` would otherwise both reach weathergenr, one working by
    accident and one as ``NULL``.
    """
    if value is None:
        value = DEFAULT_SEED
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(
                f"shared.seed must be an integer or the literal 'auto', got {value!r}"
            )
        return derive_seed(experiment_name)
    return _nonnegative_int(value, "shared.seed")


#: First month of the water (hydrological) year, for EVERY workflow that
#: aggregates to an annual value. The VALUE lives in
#: ``config/advanced_settings.yml`` under ``defaults:``; a project overrides it
#: with ``shared.water_year_start``.
#:
#: One key, because the alternative is what this replaced: WF2 read
#: ``workflows.analyze_projections.start_month_hyd_year`` (and silently ignored
#: it), WF3's generator read ``year_start_month`` as an integer, and the WF3
#: indicators and WF1 figures had no concept at all — four consumers of one
#: physical idea, agreeing by accident when they agreed.
DEFAULT_WATER_YEAR_START = ADVANCED_SETTINGS["defaults"]["water_year_start"]


def resolve_water_year_start(value) -> str:
    """Resolve ``shared.water_year_start`` to a canonical ``Jan``-style month."""
    if value is None:
        value = DEFAULT_WATER_YEAR_START
    return _month_abbrev(value, "shared.water_year_start")


def water_year_start_number(month: str) -> int:
    """Calendar month number 1..12 — what weathergenr's ``year_start_month`` takes."""
    return _MONTH_ABBREVS.index(_month_abbrev(month, "water_year_start")) + 1


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


#: The resample anchor for the DEFAULT water year — the fallback every annual
#: reduction carries when a caller passes none. Derived, never written out, so
#: it cannot drift from ``defaults.water_year_start``: it was literal
#: ``"YE-DEC"`` in two modules until 2026-08-13, which is two declarations of
#: "January" that nothing kept in step.
#:
#: Named for what it anchors. The bare ``DEFAULT_ANCHOR`` it replaces said
#: neither *what* was anchored nor that the value is a pandas/xarray frequency
#: alias, and it read as a generic knob at module scope.
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


def meets_min_historical_years(start, end) -> bool:
    """Does ``start..end`` span at least ``MIN_HISTORICAL_YEARS`` calendar years?

    Calendar arithmetic, not ``days / 365.25``: the requirement is on ANNUAL
    observations, so "16 years later, same date" is the honest comparison and it
    stays exact across leap years.
    """
    return end >= _shift_years(start, MIN_HISTORICAL_YEARS)


def historical_window_days(historical_window) -> int:
    """Calendar days spanned by a ``shared.historical_window`` mapping.

    Endpoints are the ISO ``starttime``/``endtime`` every config carries. Raises
    ``ValueError`` naming the offending key when either is missing or
    unparseable — the same fail-loud stance ``slugify_window`` takes on the same
    two values.
    """
    if not isinstance(historical_window, Mapping):
        raise ValueError(
            f"historical_window must be a mapping with starttime/endtime, got "
            f"{historical_window!r}"
        )
    bounds = {}
    for key in ("starttime", "endtime"):
        if key not in historical_window:
            raise ValueError(f"historical_window is missing {key!r}")
        try:
            bounds[key] = datetime.fromisoformat(str(historical_window[key]).strip())
        except ValueError:
            raise ValueError(
                f"historical_window.{key} is not an ISO datetime: "
                f"{historical_window[key]!r}"
            ) from None
    return (bounds["endtime"] - bounds["starttime"]).days


def historical_window_bounds(historical_window):
    """``(starttime, endtime)`` of a ``shared.historical_window``, as datetimes.

    Same parsing and same fail-loud errors as ``historical_window_days``, which
    is written in terms of this.
    """
    if not isinstance(historical_window, Mapping):
        raise ValueError(
            f"historical_window must be a mapping with starttime/endtime, got "
            f"{historical_window!r}"
        )
    bounds = []
    for key in ("starttime", "endtime"):
        if key not in historical_window:
            raise ValueError(f"historical_window is missing {key!r}")
        try:
            bounds.append(datetime.fromisoformat(str(historical_window[key]).strip()))
        except ValueError:
            raise ValueError(
                f"historical_window.{key} is not an ISO datetime: "
                f"{historical_window[key]!r}"
            ) from None
    return tuple(bounds)


def validate_historical_window(historical_window) -> int:
    """Reject a ``shared.historical_window`` shorter than ``MIN_HISTORICAL_YEARS``.

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
            f"historical_window {start.date()} .. {end.date()} spans "
            f"{days / 365.25:.1f} years, below the "
            f"{MIN_HISTORICAL_YEARS}-year minimum this toolbox requires: "
            f"weathergenr's wavelet decomposition needs at least "
            f"{MIN_HISTORICAL_YEARS} annual observations, so a shorter record "
            f"cannot support a climate stress test. Widen "
            f"shared.historical_window to >= {MIN_HISTORICAL_YEARS} years"
            + ("" if days >= 0 else " (endtime is BEFORE starttime — check the order)")
        )
    return days


def resolve_simulation_window(shared_cfg, model_cfg):
    """The window the hydrological model SIMULATES, which is not the record.

    Two different questions were one config key until 2026-08-10:

    * ``shared.historical_window`` — how much climate record to EXTRACT. It
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
    the time: rule 1.10 declared no climate-store input and read its forcing
    from the data catalog, so the two windows were genuinely independent. Rule
    1.10 now builds the forcing FROM the store, to stop re-reading the same
    source twice, and a simulation period outside the extraction therefore has
    no data behind it. Caught here, at parse time, rather than as a truncated or
    empty forcing twenty rules downstream.

    ``MIN_HISTORICAL_YEARS`` is likewise not applied here. It exists for
    weathergenr's record, so a project can now run a short simulation while
    keeping the >=16-year record a stress test needs — which the single-key
    form could not express.

    OPTIONAL, and absent means EXACT passthrough of ``historical_window`` — not
    a default that happens to coincide. Every config written before this key
    existed therefore behaves identically.

    Returns a mapping with ``starttime``/``endtime``; raises ``ValueError``
    naming the offending key if the window is malformed.
    """
    window = get_config(model_cfg, "simulation_window", None)
    if window is None:
        return get_config(shared_cfg, "historical_window", optional=False)
    if not isinstance(window, Mapping):
        raise ValueError(
            f"workflows.build_model.simulation_window must be a mapping "
            f"with starttime/endtime, got {window!r}"
        )
    for key in ("starttime", "endtime"):
        if key not in window:
            raise ValueError(
                f"workflows.build_model.simulation_window is missing {key!r}"
            )
        try:
            datetime.fromisoformat(str(window[key]).strip())
        except ValueError:
            raise ValueError(
                f"workflows.build_model.simulation_window.{key} is not an "
                f"ISO datetime: {window[key]!r}"
            ) from None
    start, end = (
        datetime.fromisoformat(str(window["starttime"]).strip()),
        datetime.fromisoformat(str(window["endtime"]).strip()),
    )
    if end <= start:
        raise ValueError(
            f"workflows.build_model.simulation_window {start.date()} .. "
            f"{end.date()} ends on or before it starts — check the order"
        )
    record = get_config(shared_cfg, "historical_window", optional=False)
    rec_start, rec_end = historical_window_bounds(record)
    if start < rec_start or end > rec_end:
        raise ValueError(
            f"workflows.build_model.simulation_window {start.date()} .. "
            f"{end.date()} is not inside shared.historical_window "
            f"{rec_start.date()} .. {rec_end.date()}. The forcing is built from "
            "the extracted climate store, so a simulation period outside the "
            "record has no data behind it — widen historical_window, or narrow "
            "the simulation window to fit inside it"
        )
    return window


def slugify_window(start, end) -> str:
    """Render a window ``(start, end)`` to a compact ``YYYYMMDD_YYYYMMDD`` slug.

    Builds the dataset-store key component for the wf3 historical-climate store
    (dev/milestones/p31/experiment-structure-design.md §4/§4c/§4d). The store dir is
    ``data/climate/historical/<clim_source>_<start>_<end>/`` where
    ``<start>``/``<end>`` are this function's output. The window endpoints are ISO
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


#: Catalog ENTRY NAMES the model-free basin delineation defaults to. Equal to
#: the shipped ``config/defaults/wflow_build_model.yml`` ``setup_basemaps``
#: values, so an existing config that declares neither key keeps building the
#: same basin (and rule 3.00b's guard digest stays byte-identical, since the
#: digest serializes the config dict as-is).
DEFAULT_HYDROGRAPHY = "merit_hydro_ihu"
DEFAULT_BASIN_INDEX = "merit_hydro_index"

#: Climate sources that carry PRECIPITATION ONLY.
#:
#: The extraction pairs them with era5 for temperature, radiation and pressure,
#: because a precipitation-only source cannot force a hydrological model on its
#: own -- so the STORE is a blend by necessity. What must not follow is a
#: REPORT that presents the borrowed fields as this source's own: a reader
#: choosing between sources would read an era5 temperature panel as a property
#: of CHIRPS. Reporting therefore narrows to what the source actually carries
#: (``climate_figures.source_climate_vars``), rather than the borrowed values
#: being drawn under the source's name.
#:
#: Named for the PROPERTY, not the members, so a future precipitation-only
#: source declares itself by joining the set.
PRECIP_ONLY_SOURCES = ("chirps", "chirps_global")

#: The store producer's script, relative to the declaring Snakefile. Both
#: Snakefiles sit at the repository root, so one relative path serves both
#: (``script:`` resolves against ``workflow.basedir``).
CLIMATE_STORE_SCRIPT = "blueearth_cst/climate_analysis/extract_historical_climate.py"

#: The region producer's script (ADR 0006), same resolution rule as above.
REGION_SCRIPT = "blueearth_cst/spatial/delineate_region.py"

#: The vector-foundation producer's script (ADR 0006 §8), same rule again.
SPATIAL_UNITS_SCRIPT = "blueearth_cst/spatial/delineate_spatial_units.py"


@dataclass(frozen=True)
class RegionRule:
    """The producer contract for the one project region artifact (ADR 0006).

    Same shape and same purpose as :class:`ClimateStoreRule`: all three
    workflows declare ``delineate_region`` from this object, so the three rule
    bodies cannot drift apart.
    """

    region_geojson: str
    script: str
    inputs: Mapping
    outputs: Mapping
    params: Mapping


def region_rule(
    project_dir,
    model_region,
    data_sources,
    hydrography=DEFAULT_HYDROGRAPHY,
    basin_index=DEFAULT_BASIN_INDEX,
) -> RegionRule:
    """Build the one producer contract for ``spatial/geoms/region.geojson``.

    ONE rule definition, declared in all three workflows (``1.01b`` / ``2.03b``
    / ``3.01b``), over the ONE delineation of ``shared.basin.region``. Before
    ADR 0006 the polygon was derived twice — by rule 1.02 on its way to
    ``basins.geojson``, and per climate-store key as ``store_region.geojson`` —
    and WF2 ran the whole climate-store producer to obtain it.

    The artifact lives in ``spatial/geoms/`` beside ``basins.geojson``,
    ``catchments.geojson`` and ``locations.geojson``: that is where this project
    keeps the vector description of where the model is.

    Model-free by construction, which is the property R07 B1 bought for the
    climate store and this must preserve — the single declared input is the data
    catalog, and the params carry only the region specification and the two
    catalog ENTRY NAMES (not paths) for the delineation.

    Parameters
    ----------
    project_dir : str
        ``project.project_dir``.
    model_region : str | Mapping
        ``shared.basin.region`` — the hydromt region specification (usually a
        Python-dict-literal string). Carried in ``params``, never resolved here.
    data_sources : str
        ``project.data_sources`` — the hydromt catalog path. The single declared
        input.
    hydrography, basin_index : str
        ``shared.basin.hydrography`` / ``shared.basin.basin_index`` — catalog
        entry names for the delineation. The defaults equal the shipped build
        template's ``setup_basemaps`` values; rule 1.02 fails loud if the two
        ever disagree.

    Returns
    -------
    RegionRule
        ``region_geojson``, ``script``, ``inputs``, ``outputs``, ``params``.
    """
    # R9 P2 commit 2: engine-neutral geometry lives under `data/` (design v10).
    # Defined ONCE here and splatted into all three workflows' delineate_region
    # rule, so the move lands in every workflow at the same instant --
    # tests/test_region_rule.py parses all three and fails on any difference.
    region_geojson = f"{project_dir}/data/spatial/geoms/region.geojson"
    return RegionRule(
        region_geojson=region_geojson,
        script=REGION_SCRIPT,
        inputs={"catalog": data_sources},
        outputs={"region_geojson": region_geojson},
        params={
            "model_region": model_region,
            "hydrography": hydrography,
            "basin_index": basin_index,
        },
    )


@dataclass(frozen=True)
class SpatialUnitsRule:
    """The producer contract for the shared vector foundation (ADR 0006 §8).

    The THIRD member of the shared-rule family, beside :class:`RegionRule` and
    :class:`ClimateStoreRule` and with the same shape. All three workflows
    declare ``delineate_spatial_units`` from this object, so the three rule
    bodies cannot drift apart.

    Named ``_rule``, not ``_spec``: the object holds a rule's script, inputs,
    outputs and params, so it IS a rule definition minus its labels. This was
    the first of the three to carry the suffix; the other two joined it in the
    R10 step-6 sweep (``dev/followups-archive.md`` ``[R10-7]``), so the family is
    uniform again.

    **Name the next one ``<thing>_rule``.** ``_contract`` was rejected — this
    repo uses "contract" for interchange surfaces
    (``dev/reference/contracts/``, ``SPATIAL_CONTRACT_VERSION``,
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
    / ``3.01f``), over the vector half of what rule 1.02 used to do alone.
    Before ADR 0006 §8, WF2 and WF3 could not reach the basin and subbasin
    boundaries without declaring ``prepare_spatial_maps`` — whose real product
    is ``spatial_maps.nc`` — so a projections-only run would have resampled
    ``vito``, ``modis_lai`` and ``soilgrids`` to draw a subbasin outline.

    Seven outputs. Six are the declared vector artifacts; the seventh,
    ``hydrography.nc``, is the SEAM INTERMEDIATE (§8a). The raster half declares
    it as an input because the whole hydrography grid stack used to cross this
    boundary in memory, and recomputing it would make WF1 read the hydrography
    twice with two grids that can drift. It is deliberately absent from
    ``spatial_catalog.yml``.

    **The params are a pure function of ``project`` + ``shared.basin`` (§8b),
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


@dataclass(frozen=True)
class ClimateStoreRule:
    """The complete producer contract for the shared historical-climate store.

    Attribute-accessible and dict-splattable: a Snakefile writes

    ``input: **SPEC.inputs`` / ``output: **SPEC.outputs`` /
    ``params: **SPEC.params`` / ``script: SPEC.script``

    so every content- or execution-determining field of the two declarations
    comes from one object rather than from two hand-maintained rule bodies.
    """

    store_dir: str
    script: str
    inputs: Mapping
    outputs: Mapping
    params: Mapping


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


def climate_store_rule(
    project_dir,
    model_region,
    clim_source,
    historical_window: Mapping,
    data_sources,
    hydrography=DEFAULT_HYDROGRAPHY,
    basin_index=DEFAULT_BASIN_INDEX,
    enforce_min_years=True,
) -> ClimateStoreRule:
    """Build the one producer contract for ``data/climate/historical/<key>/``
    (R07 B1).

    ONE rule definition, declared in ``build_model.smk`` (rule 1.04) and
    ``run_stress_test.smk`` (rule 3.08) as ``extract_historical_climate``, and
    generated per candidate source by ``analyze_climate.smk`` (rule 0.04) as
    ``extract_historical_climate_<source>``. All three resolve to the same
    store directory, so whichever workflow runs first extracts and the others
    read what is already there. Over the
    model-independent region specification + data catalog. wf1's `wf1_raw/`
    store and its `staticmaps.nc`-derived bbox are retired: the extent is now a
    pure function of ``shared.basin`` + the catalog, so a climate-only run needs
    no ``models/hydrology/wflow/`` on disk and a region change re-extracts through
    Snakemake's params rerun-trigger (design § B1).

    **The input set is exactly one entry — the catalog — in both DAGs.** An
    asymmetric input set re-creates the wf1<->wf3 re-extraction oscillation
    (design P2(b) / ext1-02); the catalog **file** is the store's freshness
    boundary (ext2-01), so it is declared plain, never ``ancient()``. Data
    *behind* an unchanged catalog entry is out of scope — edit the entry, or use
    ``snakemake --forcerun extract_historical_climate`` (in wf0, the generated
    name for the source you mean, e.g. ``extract_historical_climate_chirps``)
    (``dev/milestones/r07/migration_project-layout.md`` §2f).

    Parameters
    ----------
    project_dir : str
        ``project.project_dir``; the store lands under
        ``<project_dir>/data/climate/historical/``.
    model_region : str | Mapping
        ``shared.basin.region`` — the hydromt region specification (usually a
        Python-dict-literal string). Carried in ``params``, never resolved here.
    clim_source : str
        ``shared.clim_historical``. Selects the chirps orography branch.
    historical_window : Mapping
        The ``shared.historical_window`` section, with ``starttime`` and
        ``endtime``. Keyed at day resolution by ``slugify_window``.
    data_sources : str
        ``project.data_sources`` — the hydromt catalog path. The single
        declared input.
    hydrography, basin_index : str
        ``shared.basin.hydrography`` / ``shared.basin.basin_index`` — catalog
        ENTRY NAMES for the delineation, not paths. Optional config keys; the
        defaults equal the shipped build template's ``setup_basemaps`` values,
        and rule 1.02 fails loud if the two ever disagree.
    enforce_min_years : bool, optional
        Whether a DELIVERED record below ``MIN_HISTORICAL_YEARS`` fails the
        extraction. ``True`` for every store that feeds the pipeline — which is
        every caller except wf0's extra ``candidate_sources``, whose stores end
        at a comparison figure (2026-08-16 owner ruling; see
        ``shared/climate_window.py``).

        ``True`` emits **no param at all**, rather than ``enforce_min_years:
        True``. The params dict is a Snakemake rerun trigger, so adding a key to
        the default path would re-extract every store already on disk and break
        the byte-identity ``tests/test_climate_store_contract.py`` pins across
        the four workflows. Only the relaxed candidates carry the key — and
        because they carry it, promoting one to ``shared.clim_historical``
        changes the params WF1/WF3 declare and re-extracts it under the floor.

    Returns
    -------
    ClimateStoreRule
        ``store_dir``, ``script``, ``inputs``, ``outputs``, ``params``.

    Raises
    ------
    TypeError
        If ``historical_window`` is not a mapping.
    ValueError
        If either window endpoint is missing, or carries a sub-day component
        the day-resolution store key cannot represent (``slugify_window``).
    """
    if not isinstance(historical_window, Mapping):
        raise TypeError(
            "climate_store_rule: historical_window must be the shared."
            "historical_window mapping with 'starttime'/'endtime', got "
            f"{type(historical_window).__name__}"
        )
    starttime = get_config(historical_window, "starttime", optional=False)
    endtime = get_config(historical_window, "endtime", optional=False)

    # Byte-for-byte the key wf3 built inline before R07 (P3-1 §4/§4c/§4d): two
    # experiments sharing clim_historical + historical_window resolve to the
    # same dir and reuse the extraction.
    store_key = f"{clim_source}_{slugify_window(starttime, endtime)}"
    # R9 P2 commit 2: the store moves under `data/climate/`, and the KEY IS
    # RETAINED. `<clim_source>_<window>` is a cache key, not multi-window
    # support (R9 design Finding 3): two experiments sharing a source and a
    # window must still resolve to the same directory and reuse the extraction,
    # so the path stays EXPERIMENT-INVARIANT across the move. That invariant is
    # this commit's, and it is why the key survives the relocation unchanged.
    store_dir = f"{project_dir}/data/climate/historical/{store_key}"

    outputs = {
        "climate_nc": f"{store_dir}/extract_historical.nc",
        # Which extracted cells the basin TOUCHES. Part of the store contract
        # rather than a WF3-local artifact: the store is what both workflows
        # share, and the mask is a property of this extraction's grid, so it is
        # only derivable where the grid and the region polygon meet. Consumers
        # (rule 3.11) average over exactly these cells instead of over every
        # cell the bbox+buffer read happened to include.
        "basin_cells": f"{store_dir}/basin_cells.csv",
    }
    if clim_source in ("chirps", "chirps_global"):
        # Resolved at parse time from clim_historical, so there are no dynamic
        # outputs. The filename is clim_source-INDEPENDENT (R07 standardises the
        # two pre-R07 spellings on `orography.nc`).
        outputs["oro_nc"] = f"{store_dir}/orography.nc"

    params = {
        "model_region": model_region,
        "clim_source": clim_source,
        "starttime": starttime,
        "endtime": endtime,
        "hydrography": hydrography,
        "basin_index": basin_index,
    }
    # Present ONLY when relaxed — see the parameter's docstring for why the
    # default path must emit no key.
    if not enforce_min_years:
        params["enforce_min_years"] = False

    return ClimateStoreRule(
        store_dir=store_dir,
        script=CLIMATE_STORE_SCRIPT,
        # Two declared inputs since ADR 0006. The catalog is the store's
        # freshness boundary (ext2-01); the region is the extent it cuts to,
        # produced once per project by `delineate_region` rather than
        # re-delineated per store key. `region_rule` owns the path, so the two
        # helpers cannot disagree about where the polygon lives.
        inputs={
            "catalog": data_sources,
            "region_geojson": region_rule(
                project_dir,
                model_region,
                data_sources,
                hydrography=hydrography,
                basin_index=basin_index,
            ).region_geojson,
        },
        outputs=outputs,
        params=params,
    )


#: The sub-keys each stress-test axis accepts. Temperature has NO ``variance``:
#: only precipitation variance reaches the generator
#: (``prepare_cst_parameters`` reads ``precip.variance.{min,max}`` and
#: ``LOOKUP_COLUMNS`` carries only ``precip_variance_change``). Owner ruling
#: 2026-08-13: temperature variance is not a supported dimension.
#:
#: Closed per axis, not just at the top level. The axis guard in
#: ``prepare_cst_parameters`` already refused an unknown AXIS, but a sub-key of
#: a known axis passed unexamined — so ``stress_test.temp.variance`` was
#: accepted in silence and changed nothing.
_AXIS_SUBKEYS = {
    "temp": frozenset({"step_num", "transient_change", "mean"}),
    "precip": frozenset({"step_num", "transient_change", "mean", "variance"}),
}


def _reject_unknown_axis_subkeys(stress_test_cfg: Mapping) -> None:
    """Refuse a sub-key no axis reads, naming it and what the axis accepts."""
    for axis, allowed in _AXIS_SUBKEYS.items():
        axis_cfg = stress_test_cfg.get(axis)
        if not isinstance(axis_cfg, Mapping):
            continue
        unknown = sorted(set(axis_cfg) - allowed)
        if not unknown:
            continue
        detail = ""
        if axis == "temp" and "variance" in unknown:
            detail = (
                " Temperature variance is not a supported stress dimension: "
                "only precipitation variance reaches the weather generator. "
                "Remove it rather than expecting it to perturb anything."
            )
        raise ValueError(
            f"workflows.run_stress_test.stress_test.{axis} carries "
            f"unsupported key(s) {unknown}; it accepts {sorted(allowed)}.{detail}"
        )


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
    previously derived twice (inline in ``run_stress_test.smk`` and in
    ``blueearth_cst/experiment/prepare_cst_parameters.py``). Both call sites now read this helper.

    STRICT: ``temp.step_num`` and ``precip.step_num`` are REQUIRED — a missing
    axis section or ``step_num`` raises ``KeyError``, and a value that is not a
    non-negative integer raises ``ValueError``. The helper never silently
    invents a grid. Per-axis step count is ``step_num + 1`` (endpoints
    inclusive), and ``st_num = temp_step_count * precip_step_count``.

    Parameters
    ----------
    stress_test_cfg : Mapping
        The ``workflows.run_stress_test.stress_test`` config section, with
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
    _reject_unknown_axis_subkeys(stress_test_cfg)
    temp_step_count = _require_step_num(stress_test_cfg, "temp") + 1
    precip_step_count = _require_step_num(stress_test_cfg, "precip") + 1
    return temp_step_count, precip_step_count, temp_step_count * precip_step_count


#: The wflow output variables a model produces when a config names none.
#: ONE definition because WF1 and WF3 both default it: WF1 to these two,
#: WF3 to `[]` until 2026-08-13 — and `[]` means zero indicator tables with
#: no error, so a config omitting the key ran to completion and wrote
#: nothing. `snake_config_baseline_linux.yml` omits it, so that was a
#: shipped reproducer.
DEFAULT_WFLOW_OUTVARS = ["river discharge", "actual evapotranspiration"]

#: Twelve 1.0s — no spell-length adjustment, the identity for both factors.
DEFAULT_SPELL_FACTOR = [1.0] * 12


def validate_spell_factor(value, where: str) -> list[float]:
    """Validate a monthly spell-length coefficient list from ``stress_test``.

    Twelve numbers, one per calendar month. ``None`` (key absent) yields the
    identity, because "no adjustment" is a defensible default in a way that,
    say, ``transient_change`` is not — there the house rule is to refuse.

    The LENGTH check is the point. weathergenr indexes these by month, so a
    ten-element list would be recycled or truncated by R rather than rejected,
    and the run would silently perturb the wrong months.
    """
    if value is None:
        return list(DEFAULT_SPELL_FACTOR)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{where} must be a list of 12 monthly coefficients, got "
            f"{value!r} ({type(value).__name__})"
        )
    if len(value) != 12:
        raise ValueError(
            f"{where} must have 12 entries, one per month, got {len(value)}"
        )
    out = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where}[{index}] must be a number, got {item!r}")
        out.append(float(item))
    return out


def index_width(count: int) -> int:
    """Digits needed to render ``0..count`` so LEXICAL order matches NUMERIC.

    C27: the width is derived from the COUNT, never fixed. A 6-member grid gets
    width 1 (``st_1``) because ``st_1 … st_6`` already sort correctly; a
    12-member grid gets width 2 (``st_01 … st_12``) because ``st_1, st_11,
    st_2`` does not. 100 members gets 3.

    This is what makes an ``ls``, a glob expansion, an IDE tree and the WG-5
    catalog's key order read in run order. It is NOT cosmetic for the design
    table: C28 puts ``st_id`` in the indicator tables, and padding both from
    this one function makes the column and the filename textually identical, so
    a consumer joining a plot to its run needs no integer coercion.

    **The width is stable for an experiment's life.** It is a function of
    ``ST_NUM`` / ``RLZ_NUM``, and both live in the ``run_stress_test``
    section that ``experiment.yml`` freezes at first successful run — so a grid
    change that would move the width already forces a new experiment via
    ``check_not_frozen``. No existing tree can be renamed underneath itself.

    Raises
    ------
    ValueError
        If ``count`` is not a positive integer. A zero or negative count has no
        width, and returning 1 for it would paper over a broken grid.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"index_width needs a positive int count, got {count!r}")
    if count < 1:
        raise ValueError(f"index_width needs a positive count, got {count}")
    return len(str(count))


def member_index_regex(width: int) -> str:
    """Wildcard-constraint regex for a padded member index, excluding all-zeros.

    Two jobs, and the second is why this is exact-width rather than the laxer
    ``0*[1-9][0-9]*``:

    1. **Bar the reserved baseline.** ``st_0`` (``st_00`` at width 2) is written
       by ``generate_weather_realizations``; rule 3.12 must never become a
       second producer of it, which surfaces as a ``CyclicGraphException``
       (``run_stress_test.smk``, rule 3.12's own comment).
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
    ``[0-9]{width}`` -- which admits the baseline and makes rule 3.12 a second
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
    seconds, and closes with a one-line ``done in <elapsed>`` summary — but
    only where that summary is not already on the console under another name
    (see :meth:`stop`).

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
        self._wall_start = datetime.now()
        self._last = self._start
        #: Closed quiet periods as ``(monotonic_start, monotonic_end)``. Appended
        #: only by the watchdog thread and read only after ``stop()`` has joined
        #: it, so the list needs no lock.
        self._quiet = []
        #: Whether a stall notice was ever printed. Written by the watchdog
        #: thread and read in ``stop()`` only after it has been joined, so it
        #: needs no lock -- the same argument ``_quiet`` above makes.
        self._noticed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _wall_at(self, monotonic_value):
        """Wall-clock ``HH:MM:SS`` for a monotonic stamp taken by this watchdog.

        The class times with ``monotonic`` (immune to clock changes) but a log
        reader needs a wall clock to line a gap up against the rows around it,
        so one offset is captured at construction and applied here.
        """
        return (
            self._wall_start + timedelta(seconds=monotonic_value - self._start)
        ).strftime("%H:%M:%S")

    def quiet_rows(self):
        """One ``log_row``-shaped line per quiet period, for the rule's log file.

        The console notice is live and ephemeral; this is the same fact made
        durable. Without it a stall is invisible in the file someone sends you —
        it shows only as a GAP between two timestamps, which the reader has to
        notice and then infer, and which is indistinguishable from a rule that
        simply logged nothing while working.

        Emitted at ``stop()`` rather than inline at the moment of the stall, and
        that is a thread-safety decision, not a preference: the watchdog runs on
        a daemon thread while the rule body writes through the tee on the main
        one, so an inline write would interleave mid-line with whatever the rule
        was printing. By ``stop()`` the thread is joined and the main thread owns
        the handle. The rows carry their own start/end times, so position is
        preserved even though placement is not.
        """
        return [
            f"... quiet for {_fmt_elapsed(end - start)} "
            f"({self._wall_at(start)} -> {self._wall_at(end)})"
            for start, end in self._quiet
        ]

    def touch(self):
        self._last = time.monotonic()

    def _emit(self, text, code=None):
        # `None` rather than `_ANSI_BODY` as the default: a default argument is
        # evaluated when this class is DEFINED, and the colour constants are
        # declared further down the module (beside the console handler that owns
        # the scheme). Naming one here would be a NameError at import.
        try:
            # Console-only by design (`quiet_rows` is the durable copy), so the
            # colour here can never reach a file.
            self._stream.write(
                _paint_body(text, _console_colour(self._stream), code or _ANSI_BODY)
            )
            self._stream.flush()
        except Exception:
            pass  # console I/O must never break the job

    def _run(self):
        # `quiet_since` is the timestamp of the last real write BEFORE the
        # current silence, i.e. where the gap starts. It is carried across
        # iterations so one contiguous silence yields ONE recorded period no
        # matter how many notices it prints.
        quiet_since = None
        while not self._stop.wait(self._interval):
            now = time.monotonic()
            last = self._last
            if now - last >= self._interval:
                if quiet_since is None:
                    quiet_since = last
                elapsed = _fmt_elapsed(now - self._start)
                self._noticed = True
                self._emit(
                    f"   ... {self._label}: still running, {elapsed} elapsed\n",
                    _ANSI_WARN,
                )
            elif quiet_since is not None:
                # Output resumed: `last` is when, so the gap closes there.
                self._quiet.append((quiet_since, last))
                quiet_since = None
        if quiet_since is not None:
            # Still silent when the rule ended -- close the period at the stop,
            # not at `_last`, or the final and usually most interesting gap is
            # recorded as ending when the silence BEGAN.
            self._quiet.append((quiet_since, time.monotonic()))

    def start(self):
        if self._enabled:
            self._thread.start()
        return self

    def stop(self, failed=False):
        """Close the watchdog, printing a verdict only where one is NEWS.

        The success verdict is emitted only when this watchdog actually beeped
        (or when the job failed), because Snakemake's own finish line already
        carries both facts it states: ``DONE Rule 3.12: perturb_climate_
        realization  [rlz 1 | st 2]  0:00:19`` names the job and its duration,
        and ``   ... <label>: done in 19s`` follows it saying the same thing in
        a second duration grammar. On a fanned-out rule that is one redundant
        line per member, and because the two come from different writers -- the
        job's own process, versus Snakemake's log handler in the parent -- they
        interleave out of order under ``-c 3``, so the duplicate does not even
        land next to what it duplicates.

        The two cases kept are the ones the finish line cannot cover:

        * ``failed`` -- there IS no DONE line for a job that raised, so this is
          the only place the console says what happened to it.
        * a watchdog that beeped -- ``still running, 4m00s elapsed`` is an open
          bracket, and leaving it unclosed is worse than the duplicate. This
          also keeps the line on exactly the long, silent rules a person is
          sitting and watching.

        The log file is unaffected in every case: the heartbeat has always been
        console-only, and ``quiet_rows`` is the durable record of a stall.
        """
        if not self._enabled:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        if not (failed or self._noticed):
            return
        elapsed = _fmt_elapsed(time.monotonic() - self._start)
        verb = "failed after" if failed else "done in"
        # Yellow on the failure verdict only. `done in` is the all-clear that
        # closes a yellow `still running`, and painting it too would make the
        # resolution as loud as the alarm.
        self._emit(
            f"   ... {self._label}: {verb} {elapsed}\n", _ANSI_WARN if failed else None
        )


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


def _drop_redraw_frames(text, in_redraw):
    """Split a console chunk into what to print, dropping carriage-return redraws.

    Snakemake multiplexes several jobs onto ONE console, so an in-place progress
    bar cannot work here even on a real terminal: job A's redraw lands in the
    middle of job B's log row, which is exactly the interleaved mess a WF3 run
    with ``-c 3`` produced (measured 2026-08-17 — 214 of 624 console rows were
    dask bar frames, against 10 in the persisted log). Progress that is worth
    showing goes through ``blueearth_cst.shared.progress``, which is TTY-aware;
    a library bar written straight to ``sys.stdout`` is dropped here.

    Returns ``(console_text, in_redraw)``. ``in_redraw`` carries across calls
    because a bar arrives as one ``write`` per frame and the LAST frame arrives
    without a ``\\r`` of its own -- dask terminates the sequence with the
    completed bar plus a plain newline. A rule that only looked for ``\\r``
    therefore let exactly one 68-character ``[####...] | 100% Completed`` row
    through per bar, which is what a WF1 run still showed once the per-frame
    writes were suppressed.

    The trade-off, stated: a line whose only content precedes a ``\\r`` is lost
    from the console. The log file keeps it — ``_cr_overwrite`` collapses rather
    than drops — so the durable record is complete either way.
    """
    if "\r" in text:
        # Everything before the last `\r` was overwritten; the piece after it is
        # the final state of that line, which is the bar frame we are dropping.
        _head, sep, rest = text.rsplit("\r", 1)[-1].partition("\n")
        if not sep:
            return "", True  # line not finished; more frames may follow
        return rest, False
    if in_redraw:
        # Still inside an unterminated redraw: this chunk completes the bar's
        # final frame, so it belongs to the bar and not to the console.
        _tail, sep, rest = text.partition("\n")
        return (rest, False) if sep else ("", True)
    return text, False


#: Julia wraps ONE log record across several lines with box-drawing glyphs: a#: ``┌`` head, zero or more ``│`` continuations, and a ``└`` tail. Wflow emits
#: dozens per run, so a WF3 experiment that runs the model 20 times spent ~500
#: console rows on records that are 20 distinct sentences.
#:
#: Two continuation shapes, and they fold differently. Julia hard-wraps a long
#: message at the terminal width, indenting the wrapped part by ONE space --
#: ``┌ Info: Set atmosphere_water__precipitation_volume_flux using netCDF
#: variable`` / ``└ precip as forcing parameter.`` is a single sentence cut in
#: half. Keyword arguments, by contrast, are indented by THREE and are a list --
#: ``┌ Info: General model settings`` / ``│   snow = true`` / ``│   glacier =
#: false``. Joining the second shape with spaces would read as prose and lose
#: that it is a table, so it becomes a parenthesised list instead.
_JULIA_RECORD_HEAD = "\u250c"
_JULIA_RECORD_MID = "\u2502"
_JULIA_RECORD_TAIL = "\u2514"


class _JuliaRecordFolder:
    """Fold Julia's multi-line log records into one line each.

    ``feed(line)`` returns the lines to emit now -- empty while a record is
    still open, and the folded record on its tail. ``flush()`` closes an
    unterminated record at end of stream.

    **It fails open in every ambiguous case.** A record interrupted by an
    unrelated line (another thread, a bare ``print``) releases what it buffered
    VERBATIM followed by that line, and a stream that ends mid-record flushes
    the same way. Losing a Wflow diagnostic to a cosmetic filter would cost far
    more than the rows the filter saves, so the filter only ever fires on a
    complete, well-formed ``┌ … └`` record.
    """

    def __init__(self):
        self._buffer = []

    def feed(self, line):
        core = line.rstrip("\n")
        if not self._buffer:
            if core.startswith(_JULIA_RECORD_HEAD):
                self._buffer.append(core)
                return []
            return [line]
        if core.startswith(_JULIA_RECORD_MID):
            self._buffer.append(core)
            return []
        if core.startswith(_JULIA_RECORD_TAIL):
            self._buffer.append(core)
            folded = self._fold(self._buffer)
            self._buffer = []
            return [folded + "\n"]
        # Something else arrived inside a record: release it all, unchanged.
        return self.flush() + [line]

    def flush(self):
        released = [line + "\n" for line in self._buffer]
        self._buffer = []
        return released

    @staticmethod
    def _fold(lines):
        head = lines[0][len(_JULIA_RECORD_HEAD) :].strip()
        prose, kwargs = [], []
        for line in lines[1:]:
            rest = line[1:]
            # Three-space indent marks a keyword argument; one marks a message
            # Julia hard-wrapped at the terminal width. See the note above.
            (kwargs if rest.startswith("   ") else prose).append(rest.strip())
        text = " ".join([head] + prose).strip()
        if kwargs:
            text = f"{text} ({', '.join(kwargs)})"
        return text


#: ASCII stand-ins for the glyphs a child process draws with, used ONLY when the
#: console cannot encode them. The alternative is ``errors="replace"``, which
#: turns Wflow's 20-cell progress bar into twenty literal question marks — a row
#: that looks like a decoding fault rather than a bar. A legacy Windows code
#: page is the normal case here, not an exotic one, so the degraded rendering is
#: worth choosing rather than inheriting. The log file always gets the real
#: glyphs; this is the console mirror only.
_ASCII_GLYPH_FALLBACK = str.maketrans(
    {
        "\u2588": "#",  # full block -- a filled progress cell
        "\u2589": "#",
        "\u258a": "#",
        "\u258b": "=",
        "\u258c": "=",
        "\u258d": "=",
        "\u258e": "-",
        "\u258f": "-",
        "\u2591": ".",  # light shade -- an empty progress cell
        "\u2592": ".",
        "\u2593": "=",
        "\u2500": "-",  # box drawing, as Julia's log records use
        "\u2502": "|",
        "\u250c": "+",
        "\u2514": "+",
        "\u251c": "+",
        "\u2026": "...",
        "\u2192": "->",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
    }
)


#: Body lines muted on the CONSOLE only -- matched on the message, after
#: :func:`_compact_log_line` has normalized the row.
#:
#: ``Parsing data catalog from <path>`` is hydromt announcing a catalog read.
#: It is one line per catalog PER JOB, and a fanned-out rule reads the same two
#: catalogs in every member: nine parallel jobs of rule 3.14 put eighteen
#: identical lines on the console in the same second. Nothing distinguishes
#: them, and nothing can -- the line names the catalog, which is a property of
#: the run, not of the member.
#:
#: Muting is the only lever available, because collapsing is not: each job is a
#: SEPARATE PROCESS with its own tee, so no in-process buffer can see that the
#: line it is about to print was just printed by a sibling. That is also why
#: this differs from ``_CONSOLE_MUTED_PREFIXES``, which mutes Snakemake's own
#: records in the parent.
#:
#: The line SURVIVES IN EVERY LOG PART, which is where a question about what a
#: job read is actually answered, and the run header names the catalogs up
#: front. What the console loses is a fact it was stating up to eighteen times
#: and that nobody reads on the eighteenth.
#:
#: The rest are hydromt's MODEL-OPEN BOILERPLATE, muted for the same reason and
#: on the same terms. Every rule that touches the built model reopens it, and
#: each reopen announces the plugin version, the TOML it read and the Wflow
#: version it supports -- three rows, eight times in one WF1 build and once per
#: member in WF3. All three state a property of the RUN, not of the job, and
#: the run header states two of them already.
#:
#: The two ``skip writing`` rows are the other shape worth muting: a report that
#: NOTHING happened. ``No tables found, skip writing.`` is hydromt confirming an
#: absence on every single write, which is the definition of a row that cannot
#: distinguish one job from another.
#:
#: Nothing here is ever a warning or an error -- see the INFO restriction in
#: :func:`_muted_on_console`, which is what keeps a prefix muted for VOLUME from
#: ever silencing a row someone is scanning for.
_TEE_CONSOLE_MUTED = (
    ("data_catalog", "Parsing data catalog from "),
    ("model", "Initializing wflow_sbm model from hydromt_wflow"),
    ("config", "Reading model config file from "),
    ("config", "Reading default config file from "),
    ("wflow_base", "Supported Wflow.jl version "),
    ("tables", "Reading model table files."),
    ("tables", "No tables found, skip writing."),
    ("grid", "No grid data found, skip writing."),
    # hydromt echoes the object store URI it is about to read. WF2 already
    # printed that URI one row earlier, from `fetch_gcm_raw`, in BOTH the
    # pinned and the globbed branch -- so this row is a duplicate that costs
    # 162-175 characters four times per run, the longest rows the workflow
    # produces. A pattern rather than a prefix because the entry name sits
    # between the two fixed parts.
    ("data_source", re.compile(r"^Reading \S+ from \w+://")),
)


def _muted_matches(message, muted):
    """Whether ``message`` matches a ``_TEE_CONSOLE_MUTED`` right-hand side.

    A plain string is a PREFIX test (the message must open with it, not merely
    contain it); a compiled pattern is searched. Prefixes stay the default —
    they cannot silence a row whose opening words differ — and a pattern is
    used only where a variable field sits between two fixed parts.
    """
    if isinstance(muted, str):
        return message.startswith(muted)
    return muted.search(message) is not None


def _muted_on_console(text):
    """Whether a tee chunk is a muted-on-console body line.

    Three restrictions, all failing in the direction of PRINTING:

    * the chunk must be exactly one newline-terminated line. A tee is handed
      arbitrary chunks, not lines, so a partial write or a multi-line block is
      never matched — it prints, which is the behaviour we started from.
    * the row must carry the muted module in its own column and the message
      must match the muted entry — a PREFIX test for a plain string, or a
      search for a compiled pattern — rather than the phrase being allowed to
      appear anywhere in the text.
    * the row must be at INFO, which :func:`_log_row_text` renders by OMITTING
      the level field. A ``data_catalog - WARNING - ...`` row therefore has a
      third field before the message and cannot match — the same restriction
      ``_ConsoleHandler._muted`` applies, and for the same reason: a prefix
      muted for volume must never be able to silence a warning.
    """
    if not text.endswith("\n") or text.count("\n") != 1:
        return False
    fields = text[:-1].split(" - ", 2)
    if len(fields) != 3:
        return False
    _stamp, module, message = fields
    # `_compact_log_line` KEEPS a `<model>.<component>: ` prefix when the
    # component is not the module column -- `grid - wflow_sbm.states: No grid
    # data found, skip writing.` is that shape, and the prefix would otherwise
    # defeat every `startswith` below. Strip it for the test only; the row is
    # unchanged either way.
    message = _COMPONENT_PREFIX_RE.sub(r"\2", message)
    return any(
        module == muted_module and _muted_matches(message, muted)
        for muted_module, muted in _TEE_CONSOLE_MUTED
    )


class _Tee:
    """Text stream mirroring in-process output to a live console and a log file.

    Deliberately not an ``io`` subclass: ``script:`` rules only ``print`` /
    log through ``sys.stdout``/``sys.stderr``, so ``write`` + ``flush`` (plus
    ``isatty``) is all that is needed. Note: this operates at the Python
    stream level, so output from *shell* subprocesses (which inherit the real
    file descriptors) is not captured — only in-process Python output is.

    The ``live`` sink (console) gets output verbatim EXCEPT for carriage-return
    redraw frames, which are dropped (see ``_console_text``) — verbatim in SHAPE,
    that is, less those frames and less the handful of high-volume boilerplate
    rows ``_TEE_CONSOLE_MUTED`` drops from the console and keeps in the file. The
    ``logfile`` sink instead receives each line *after* carriage-return overwrite
    (see ``_cr_overwrite``), so the persisted log keeps only the final rendered
    state of an in-place-updated line rather than every redraw. Partial (not yet
    newline-terminated) output is held in ``_pending`` and collapsed on the fly,
    so a bar redrawing for hours never grows the buffer beyond one line.
    """

    def __init__(self, live, logfile, project_root="", on_activity=None, tokens=None):
        self._live = live
        # The declared key folders, resolved ONCE per tee rather than per line.
        # Per line it would be a global read on the hot path of every write, and
        # a test that declared tokens would leak into the next one; here the
        # value is fixed when the redirect is set up, which is also when the
        # header naming those folders is written.
        self._tokens = _path_tokens() if tokens is None else tuple(tokens)
        # Animate an in-place bar only where redrawing means something. On a
        # pipe -- `snakemake ... *> run.txt`, or the GUI capturing a run -- a
        # carriage return does NOT overwrite, so streaming the frames turns one
        # bar into a row per frame in the very artifact someone reads later.
        # ``run_and_tee`` has always made this check for `shell:` rules; making
        # it here puts `script:` rules on the same footing.
        try:
            self._stream_frames = bool(live.isatty())
        except Exception:
            self._stream_frames = False
        #: The last frame seen while frames are NOT streamed, held so the
        #: console still gets the bar's summary -- once, as an ordinary row.
        self._held_frame = ""
        # Decided once, against the REAL console this tee wraps -- and it paints
        # the live copy ONLY. The log file must never receive an escape code:
        # it is read months later, by tools and by `merge_logs`, where a colour
        # is corruption rather than styling.
        self._colour = _console_colour(live)
        self._logfile = logfile
        self._project_root = project_root
        self._on_activity = on_activity  # called on each write (heartbeat reset)
        self._pending = ""  # current, not-yet-newline-terminated log line
        self._in_redraw = False  # console-side state for _drop_redraw_frames
        # Set by ``close``. A tee OUTLIVES its log file: ``tee_to_log`` closes
        # the file when its `with open(...)` exits, and anything still holding a
        # reference to this object then has a live handle onto a dead sink.
        # That is not hypothetical -- a library configuring logging lazily
        # inside the rule body (hydromt does, per data catalog) installs a
        # StreamHandler bound to whatever ``sys.stdout`` was AT THAT MOMENT,
        # which is this tee, and nothing restores a handler that did not exist
        # when the redirect was set up.
        self._closed = False

    def write(self, text, _redraw=False):
        if self._on_activity is not None:
            self._on_activity()
        out = _relativize_paths(
            _compact_log_line(text), self._project_root, self._tokens
        )
        # Painted, and stripped of carriage-return redraw frames -- see
        # `_drop_redraw_frames` for why an in-place bar cannot work under a
        # multi-job snakemake console. A muted line skips the console and takes
        # the log-file path below unchanged: this is the one place the two sinks
        # are allowed to differ in CONTENT rather than in formatting, and the
        # durable record is the one that keeps everything.
        shown, self._in_redraw = _drop_redraw_frames(out, self._in_redraw)
        if _redraw:
            shown, self._in_redraw = out, False
        if shown and not _muted_on_console(shown):
            self._live.write(_paint_body(shown, self._colour))
        # After close the console is still open and still the right place for
        # this text; only the log file is gone. Writing to a closed file raises
        # ValueError, and a raise HERE is the expensive kind: these late writes
        # happen during interpreter finalization, where the exception cannot be
        # reported (module globals are already torn down) and CPython prints the
        # bare `Error in sys.excepthook:` / `Original exception was:` pair with
        # EMPTY bodies instead. Degrading to console-only keeps the output and
        # removes that whole failure class.
        if self._closed:
            return len(text)
        buf = self._pending + out
        lines = buf.split("\n")
        self._pending = lines.pop()  # trailing fragment, no newline yet
        for line in lines:
            self._logfile.write(_cr_overwrite(line) + "\n")
        self._pending = _cr_overwrite(self._pending)  # keep the buffer bounded
        return len(text)

    def write_redraw(self, text):
        """Write a carriage-return frame the console is allowed to KEEP.

        ``write`` drops redraw frames because a library bar written straight to
        ``sys.stdout`` cannot animate under a multi-job console. Our own bar
        (``blueearth_cst.shared.progress``) is the sanctioned exception: it is
        installed by one rule at a time, sizes itself to the stream, and is the
        thing a reader is meant to watch. It reaches this method by duck-typing —
        ``getattr(stream, "write_redraw", stream.write)`` — so it degrades to a
        plain write on any stream that is not a tee.

        The exception is granted only where a carriage return OVERWRITES. Off a
        terminal the frames would append, so they are held instead and the last
        one is emitted as a single ordinary row when the bar closes its line —
        the same summary the log file keeps, and the same row count.
        """
        if self._stream_frames:
            return self.write(text, _redraw=True)
        if "\r" in text:
            frame = _cr_overwrite(text)
            if frame.strip():
                self._held_frame = frame
            return self.write(text)
        if self._held_frame:
            # The bar's closing newline: release the summary, then let `write`
            # terminate the log's own pending line as usual.
            self._live.write(_paint_body(self._held_frame, self._colour) + "\n")
            self._held_frame = ""
        return self.write(text)

    def flush(self):
        # Flush the sinks but NOT ``_pending``: emitting a mid-progress fragment
        # would re-clutter the log with every partial redraw.        self._live.flush()
        # Same reasoning as ``write``: ``logging.shutdown`` flushes every handler
        # at exit, so a handler left pointing here must not raise.
        if not self._closed:
            self._logfile.flush()

    def close(self):
        # Flush any trailing partial line (e.g. a progress bar cut short by an
        # error before its final newline) so nothing is silently dropped.
        if self._closed:
            return
        if self._pending:
            self._logfile.write(_cr_overwrite(self._pending) + "\n")
            self._pending = ""
        self._logfile.flush()
        self._closed = True

    def isatty(self):
        return False


# Benign CPython interpreter-shutdown noise. A subprocess -- notably the verbose
# ``hydromt build wflow_sbm ... -vv`` step -- can emit a repeating
# ``Error in sys.excepthook:`` / ``Original exception was:`` cascade with EMPTY
# bodies *after* it has finished successfully (rc=0), when a stderr write fails
# during interpreter finalization (many GDAL/rasterio datasets torn down at once
# on Windows). It floods the tail of an otherwise-clean log. Triaged as cosmetic
# in dev/milestones/phase-1/m01/warnings.md; ``run_and_tee`` collapses a *pure* trailing run
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
    success (t260721a; dev/tasks/). Teeing in-process restores exit-code
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
        label = label[len("_parts/") :]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(_log_header_lines(log_path))  # header to file only, not console
        log.flush()

        colour = _console_colour(sys.stdout)
        # Resolved once per RUN, for the same reason the tee resolves once per
        # construction: `emit` is called per line.
        tokens = _path_tokens()
        # Redraw a progress bar in place only where that means something. On a
        # pipe -- `snakemake ... *> run.txt`, or the GUI capturing a run -- the
        # frames are not overwritten but APPENDED, so streaming them turns one
        # bar into ~40 rows in the very artifact someone reads afterwards.
        try:
            stream_frames = sys.stdout.isatty()
        except Exception:
            stream_frames = False

        def _console_write(text):
            # The log file is UTF-8. The live console mirror may be a legacy
            # code page (cp1252 on Windows) that cannot encode glyphs the child
            # emits (e.g. Julia/Wflow progress-bar blocks); fall back to ASCII
            # stand-ins for the console only — the log always gets the real text.
            try:
                sys.stdout.write(text)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                folded = text.translate(_ASCII_GLYPH_FALLBACK)
                sys.stdout.write(folded.encode(enc, "replace").decode(enc))
            sys.stdout.flush()

        def emit(text, redraw=False, had_cr=False):
            # Collapse a carriage-return-redrawn line to the frame that was
            # actually left on screen, then compact hydromt's redundant log
            # format (see _compact_log_line) and show project files relative to
            # the project dir; non-hydromt lines and out-of-project paths pass
            # through unchanged.
            text = _relativize_paths(
                _compact_log_line(_cr_overwrite(text)), project_root, tokens
            )
            # The log is written unconditionally; only the console mirror drops
            # the repeated boilerplate. A `shell:` rule and a `script:` rule run
            # the same hydromt code and used to disagree about which of its rows
            # reached the console, purely because they reach it through
            # different tees -- `_Tee` consulted this and `run_and_tee` did not.
            if not _muted_on_console(text):
                # Body tier on the console only: a shell rule's output is
                # detail, and `log` below must stay free of escape codes.
                shown = _paint_body(text, colour)
                # `redraw` means the frames of this line were already streamed,
                # so the cursor sits mid-bar: return to column 0 and overwrite
                # it with the final frame rather than printing a second line.
                # Where they were NOT streamed, the bar never appeared and its
                # final frame would be a row of its own -- drop it, the call
                # `_Tee._drop_redraw_frames` makes for the same reason (a bar
                # cannot animate under a multi-job console). The log keeps it.
                if had_cr and not redraw:
                    pass
                else:
                    _console_write(("\r" + shown) if redraw else shown)
            log.write(text)
            log.flush()

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        # Decode the child's pipe as UTF-8. Julia/Wflow (and Python under UTF-8
        # mode) emit UTF-8; without this, text mode uses the Windows locale code
        # page (cp1252) and mangles non-ASCII — a `█` (UTF-8 E2 96 88) decoded
        # as cp1252 becomes "â–ˆ". ASCII-only children (hydromt logs) are
        # unaffected; `errors="replace"` guards any genuinely non-UTF-8 byte
        # instead of crashing the tee.
        #
        # Wrapped HERE rather than via `Popen(text=True)` for `newline=""`,
        # which Popen cannot pass on. Its default is UNIVERSAL NEWLINES, and
        # that turned every `\r` into a `\n` — so Wflow's progress bar, which
        # redraws one line ~40 times per model run, arrived as ~40 separate
        # lines. A WF3 experiment runs the model 20 times, and the result was
        # ~2000 of the log's 4000 rows being frames of a bar that is meant to
        # occupy ONE. Preserving `\r` lets `_cr_overwrite` do to a Julia bar
        # exactly what it already did to a Python one.
        stream = io.TextIOWrapper(
            proc.stdout, encoding="utf-8", errors="replace", newline=""
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
            folder = _JuliaRecordFolder()
            # Carriage-return frames of a line still being redrawn, held until
            # its terminating newline arrives.
            frames = []

            def emit_folded(folded, redraw=False, had_cr=False):
                nonlocal pending
                if _is_shutdown_noise(folded):
                    pending.append(folded)
                    return
                # Collapse here too, not just at end of stream. The block was
                # flushed verbatim until 2026-08-10, which made the filter fire
                # only when the cascade happened to be the LAST thing in the
                # stream. Under `-c 3` it usually is not: several jobs finalize
                # concurrently, another job's line lands after the markers, and
                # the collapse the log needed never ran.
                _flush_pending(pending, emit)
                pending = []
                emit(folded, redraw=redraw, had_cr=had_cr)

            def deliver(line, redraw):
                # `redraw` applies to the FIRST row only: it means the cursor is
                # sitting on a half-drawn bar, and one row overwrites it.
                had_cr = "\r" in line
                for folded in folder.feed(line):
                    emit_folded(folded, redraw, had_cr)
                    redraw = False

            for raw in stream:
                heartbeat.touch()
                # `\r\n` is ONE line ending, not a redraw. Normalizing it first
                # is load-bearing: `_cr_overwrite("text\r\n")` would otherwise
                # split on the `\r` and keep only the `\n`, blanking the row.
                raw = raw.replace("\r\n", "\n")
                if raw.endswith("\r"):
                    if stream_frames:
                        _console_write(raw)
                    frames.append(raw)
                    continue
                line, frames = "".join(frames) + raw, []
                deliver(line, redraw=stream_frames and "\r" in line)
            if frames:
                deliver("".join(frames), redraw=stream_frames)
            # An unterminated record is released verbatim, and must NOT go back
            # through `folder.feed` — its head line would simply be buffered
            # again and lost with it.
            for released in folder.flush():
                emit_folded(released)
            rc = proc.wait()
            _flush_pending(pending, emit, rc)
        finally:
            heartbeat.stop(failed=(rc is None or rc != 0))
        return rc


def _flush_pending(pending, emit, rc=None):
    """Emit a candidate block: collapse a real cascade, else verbatim.

    Collapse only when the block holds at least two markers (one full
    ``excepthook``/``original`` unit); a smaller or marker-free block is emitted
    unchanged so nothing real is dropped.

    Called BOTH mid-stream (``rc=None`` — the child is still running) and once
    the stream ends (``rc`` known). The exit code is worth naming in the summary
    because the whole point of the collapse is that this noise follows a
    SUCCESSFUL run; mid-stream that is not yet knowable, so the summary says
    where it happened instead.
    """
    if not pending:
        return
    marker_count = sum(1 for ln in pending if ln.strip() in _EXCEPTHOOK_MARKERS)
    if marker_count >= 2:
        where = "mid-run" if rc is None else f"child rc={rc}"
        emit(
            f"[run_logged] collapsed {len(pending)} benign interpreter-shutdown "
            f"lines (repeated 'Error in sys.excepthook:' / 'Original exception "
            f"was:'; {where})\n"
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
    loggers = [
        logging.getLogger()
    ]  # root, then every concrete (non-placeholder) logger
    loggers += [
        lg
        for lg in logging.Logger.manager.loggerDict.values()
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


def _detach_handlers_bound_to(tees, orig_out, orig_err):
    """Repoint any handler still bound to a tee back at the real console.

    ``_restore_log_handlers`` can only undo what ``_redirect_console_log_handlers``
    SAVED, and that snapshot is taken on entry. A handler created *during* the
    rule body is invisible to it — and libraries do exactly that: hydromt
    installs a StreamHandler when it parses a data catalog, which happens inside
    the body, bound to the tee that ``sys.stdout`` then was.

    Left alone, such a handler outlives the log file it points into. Every later
    record through it is dropped, and its exit-time flush touches a closed file.
    Sweeping by stream IDENTITY (never by logger name) repoints exactly those and
    nothing else, so a genuine FileHandler is untouched.
    """
    targets = {id(tee) for tee in tees}
    loggers = [logging.getLogger()]
    loggers += [
        lg
        for lg in logging.Logger.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        for handler in getattr(lg, "handlers", []):
            stream = getattr(handler, "stream", None)
            if id(stream) not in targets:
                continue
            _set_handler_stream(handler, orig_err if stream is tees[-1] else orig_out)


def _is_clean_exit(exc) -> bool:
    """True for a deliberate ``SystemExit(0)`` — a SUCCESS, not a failure.

    ``sys.exc_info()`` is populated during *any* unwinding, including the clean
    early return a ``script:`` module makes with ``raise SystemExit(0)``. That is
    how every WF2 cache-hit job leaves its body (``fetch_gcm_raw.py``,
    ``get_stats_climate_proj.py``), so the previous "any exception is a failure"
    test printed ``... <rule>: failed after Ns`` to the console for jobs Snakemake
    then reported as ``Finished`` — on the most common path in the workflow.
    Observed 2026-07-31 on a forced cached fetch.

    Only the exit CODE decides: ``SystemExit(1)`` is still a failure, and so is
    every other exception. The log file is unaffected either way (the heartbeat
    writes to the console only) — this is the status line a user actually watches.
    """
    return isinstance(exc, SystemExit) and exc.code in (None, 0)


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
      log that Snakemake would read as a finished product;
    - on failure the formatted **traceback is written into the log part** before
      unwinding. Snakemake prints ``check log file(s) for error details`` and
      nothing more, so a log that stops mid-rule actively misdirects: it sends
      an operator to the one file that cannot explain the failure. A deliberate
      ``SystemExit(0)`` is a success and writes nothing (see ``_is_clean_exit``).

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
        label = label[len("_parts/") :]
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(_log_header_lines(log_path))  # header to file only
        handle.flush()
        # heartbeat writes to the real console (orig_err), never the log handle
        heartbeat = _Heartbeat(label, orig_err, interval=heartbeat_interval)
        stdout_tee = _Tee(
            orig_out, handle, project_root=project_root, on_activity=heartbeat.touch
        )
        stderr_tee = _Tee(
            orig_err, handle, project_root=project_root, on_activity=heartbeat.touch
        )
        sys.stdout, sys.stderr = stdout_tee, stderr_tee
        # route library logging (hydromt) bound to the old console through the tee
        saved_handlers = _redirect_console_log_handlers(
            orig_out, orig_err, stdout_tee, stderr_tee
        )
        heartbeat.start()
        # A new log starts a new figure group, so its bundle rows describe only
        # the figures this log's own rules wrote. Makes that invariant a
        # property of the FILE, which is the unit a reader has in front of them.
        _FIGURE_BUNDLES.clear()
        _FIGURE_BUNDLE_MODULES.clear()
        try:
            yield
        except BaseException as exc:  # noqa: BLE001 - re-raised below, never swallowed
            # Write the traceback INTO the log part before unwinding. Snakemake
            # prints `check log file(s) for error details` and nothing else, so
            # without this the one artifact it names ends mid-rule with no
            # reason -- the cause reaches the interactive console and is absent
            # from the file a user would send you ([R10-13], t2608071219).
            #
            # Written to ``handle`` directly, NOT through ``stderr_tee``: the
            # interpreter prints its own traceback to the real stderr once the
            # exception leaves this manager, so teeing would put two copies on
            # the console. Direct writes bypass the tee, so relativize here to
            # match the path spelling of every other line in the file.
            if not _is_clean_exit(exc):
                handle.write(
                    _relativize_paths(
                        "\n"
                        + "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                        project_root,
                        stdout_tee._tokens,
                    )
                )
                handle.flush()
            raise
        finally:
            # Drain any figures the rule wrote but never followed with another
            # row, BEFORE the tee closes -- `flush_figure_bundles` prints
            # through `log_row`, so it needs the log's stdout still in place.
            # A plotting rule whose last act is to save a figure is the normal
            # case, so without this the whole bundle would be lost.
            try:
                flush_figure_bundles()
            except Exception:  # noqa: BLE001 -- never fail a rule over a log row
                pass
            # Collect FIRST, while the interpreter is healthy and this block is
            # still fully set up. A `script:` rule's data catalogs and model
            # objects are frame locals of the function the body just called, so
            # by now they are unreachable — but hydromt's catalog and model
            # objects reference each other, so what holds their GDAL/rasterio
            # handles is a REFERENCE CYCLE. A cycle is freed only by the cyclic
            # collector, and if that does not run until interpreter finalization
            # the handles are all torn down there instead. On Windows that makes
            # a stderr write fail, CPython's excepthook cannot run that late
            # (module globals are already gone), and it prints a bare
            # `Error in sys.excepthook:` / `Original exception was:` pair with
            # EMPTY bodies, repeatedly, after a rule that SUCCEEDED.
            #
            # Here rather than per-rule because the population is every `script:`
            # rule, present and future. Three modules carry a local
            # `gc.collect()` and it was measured to work in only one of them
            # (`delineate_region.py`, 14 lines -> 0); the other two collect with
            # the catalog still BOUND, so the collector cannot claim it. This is
            # the one place that sees every rule after its frame has gone.
            #
            # Ordering matters: before the handler restore and the tee close, so
            # that a ``__del__`` which logs or warns during collection still
            # lands in the rule's log instead of on the bare console — the exact
            # late-write class the tee-close fix addressed.
            gc.collect()
            # Restore log handlers first (before their target tees close), stop
            # the watchdog (console-only summary), flush trailing partial lines
            # while ``handle`` is open, then restore the streams — all always run,
            # even if the body raised.
            _restore_log_handlers(saved_handlers)
            # ...then the ones that snapshot could not know about, which are the
            # ones that would otherwise still be writing into a closed log file
            # after this block returns.
            _detach_handlers_bound_to((stdout_tee, stderr_tee), orig_out, orig_err)
            _exc = sys.exc_info()[1]
            heartbeat.stop(failed=_exc is not None and not _is_clean_exit(_exc))
            # The stalls the watchdog announced on the console, made durable in
            # the log. `stop()` has joined the watchdog thread, so this is the
            # main thread writing alone -- the reason it happens here and not at
            # the moment of the stall. Written to `handle` directly for the same
            # reason as the traceback above: the tees are about to close, and a
            # console copy would duplicate a notice already printed live.
            for _row in heartbeat.quiet_rows():
                handle.write(
                    _log_row_text(
                        f"{datetime.now():%H:%M:%S}", "heartbeat", "INFO", _row
                    )
                    + "\n"
                )
            handle.flush()
            for tee in (stdout_tee, stderr_tee):
                tee.close()
            sys.stdout, sys.stderr = orig_out, orig_err


#: Rank for the ``CST_LOG_LEVEL`` floor. Names and order follow ``logging``'s,
#: so an operator who knows one knows the other.
_LOG_LEVEL_RANK = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def log_level_floor():
    """The minimum rank ``log_row`` will emit, from ``CST_LOG_LEVEL``.

    Unset or unrecognized means ``DEBUG`` — i.e. emit everything, the behaviour
    every caller had before the floor existed. Read per call rather than cached
    so the variable can be set inside a session without a reimport.
    """
    return _LOG_LEVEL_RANK.get(os.environ.get("CST_LOG_LEVEL", "").strip().upper(), 10)


def log_row(message, module="cst", level="INFO"):
    """Print one log row in the standard compact format used across rule logs.

    ``HH:MM:SS - <module> - <message>``, with the level shown only when it is
    not INFO (:func:`_log_row_text`) — the same shape ``_compact_log_line``
    produces for hydromt records, so a ``script:`` rule's own messages sit
    uniformly among the hydromt/library lines rather than as bare,
    timestamp-less text. Use this instead of a plain ``print`` for anything
    meant to appear in a rule log. The row is already compact, so the tee passes
    it through (only any project paths in it are relativized).

    Rows below ``CST_LOG_LEVEL`` are dropped, which is the toolbox's quiet mode:
    ``CST_LOG_LEVEL=WARNING`` leaves only warnings and errors. Two properties
    make that safe to add to an existing caller population:

    * **Unset means emit everything**, so nothing changes until someone opts in.
    * **An unrecognized ``level`` is never suppressed.** A caller passing a
      level this table does not know keeps printing — a filter that silently
      swallowed an unfamiliar level would hide exactly the unusual row worth
      seeing.

    Still ``print`` rather than ``logging``, deliberately: the tee captures
    ``sys.stdout``, so a row that went through a logging handler would bypass
    the mechanism that puts it in the rule's log file. The floor therefore
    belongs INSIDE this function, not in a migration to ``logging``.
    """
    rank = _LOG_LEVEL_RANK.get(str(level).strip().upper())
    if rank is not None and rank < log_level_floor():
        return
    # Any ordinary row closes an open figure bundle first, so the bundle line
    # appears where the figures were actually written rather than after the
    # message that followed them. See `save_figure`.
    if not _FIGURE_BUNDLE_FLUSHING:
        flush_figure_bundles()
    # Shorten paths HERE as well as in the tee, so a rule with no `log:`
    # directive prints them the same way as every rule that has one. A row that
    # does go through the tee is simply relativized twice, which is a no-op:
    # the second pass finds no prefix left to strip.
    message = _relativize_paths(
        str(message), os.environ.get(_PROJECT_ROOT_ENV, ""), _path_tokens()
    )
    print(_log_row_text(f"{datetime.now():%H:%M:%S}", module, level, message))


# Figures written by `save_figure` since the last flush, as
# {absolute directory: [file name, ...]} in first-written order, plus the module
# each was announced under. Reset and drained by `tee_to_log`, which makes the
# grouping a property of the LOG FILE rather than of the process.
_FIGURE_BUNDLES = {}
_FIGURE_BUNDLE_MODULES = {}
#: Guards `log_row` against recursing while the bundle rows are themselves
#: being printed through it.
_FIGURE_BUNDLE_FLUSHING = False


def flush_figure_bundles():
    """Emit one row per directory of pending figures, then forget them.

    Called automatically -- by `log_row` before any other row, and by
    `tee_to_log` when a rule's log closes -- so no caller has to remember it.
    Idempotent: with nothing pending it prints nothing.
    """
    global _FIGURE_BUNDLE_FLUSHING
    if not _FIGURE_BUNDLES:
        return
    pending = tuple(_FIGURE_BUNDLES.items())
    _FIGURE_BUNDLES.clear()
    modules = dict(_FIGURE_BUNDLE_MODULES)
    _FIGURE_BUNDLE_MODULES.clear()
    _FIGURE_BUNDLE_FLUSHING = True
    try:
        for directory, names in pending:
            module = modules.get(directory, "plot")
            if len(names) == 1:
                # A rule that writes ONE figure names the file, exactly as
                # before: "1 figure -> <dir>" would be a longer way of saying
                # less, and several rules here write a single map.
                log_row(os.path.join(directory, names[0]), module=module)
            else:
                log_row(f"{len(names)} figures -> {directory}", module=module)
    finally:
        _FIGURE_BUNDLE_FLUSHING = False


def save_figure(path, module="plot", fig=None, **kwargs):
    """Save a matplotlib figure to ``path`` and announce it as part of a bundle.

    Centralizes the "write a figure + log one line" pattern for the plotting
    ``script:`` rules: every produced map/plot is accounted for in the rule's
    log instead of the log being empty or showing only upstream library
    chatter. Parent directories are created. ``kwargs`` pass through to
    ``Figure.savefig`` (e.g. ``dpi``, ``bbox_inches``). matplotlib is
    imported lazily so this module stays light for the Snakefiles that import it
    only for ``get_config`` / ``stress_test_grid``.

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

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    (fig if fig is not None else plt.gcf()).savefig(path, **kwargs)
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


# Colour by WHAT KIND OF LINE it is, not by which field it is -- three tiers,
# whole lines, so a run scrolling past reads as structure rather than text.
#
# * ``_ANSI_RUN`` (blue) -- a job STARTED. The line that says what you are now
#   waiting for, so it is the one the eye goes to.
# * ``_ANSI_DONE`` (green) -- a job FINISHED. Green because that is what green
#   means everywhere else a machine reports on work, and the pair then reads
#   without a legend: blue is in flight, green is behind you.
# * ``_ANSI_BODY`` (light grey) -- everything in between: a rule's own output,
#   the heartbeat's status notices, and Snakemake's informational lines. This is
#   the bulk of the output and it recedes.
#
# Superseded the earlier per-FIELD scheme (bold cyan identity, grey qualifiers)
# on 2026-08-14: within one line the identity did stand out, but across a long
# run every line looked alike, and what a reader scrolls for is "where did this
# job start" / "what finished", not a field.
#
# WARNING and ERROR records are never recoloured, so Snakemake's own red stays
# the loudest thing on screen -- three tiers describe the ROUTINE path only.
#
# The body is a FIXED light grey (256-colour 250) rather than the ``2m`` faint
# it was until 2026-08-14. Faint has one real advantage -- it dims whatever
# colour the terminal already uses, so it cannot land close to the background on
# a light theme -- and one practical cost, which is that on a dark terminal it
# dims far enough to be work to read, which is what a body tier must not be when
# it is most of the output. A fixed grey is the deliberate trade, and it is
# **one constant to swap** if it reads wrong on your terminal: back to ``2``, or
# to ``90`` (bright black) for something between the two.
#
# The other two are the bright 8-colour codes, which follow the terminal's own
# palette rather than pinning an exact hue -- so they stay in whatever key the
# theme is written in. All three are swappable here and nowhere else.
#
# Two more sit OUTSIDE the routine three, for the two lines that report that
# something went wrong or might have. They are deliberately not a fourth and
# fifth tier: the three above classify every routine line, while these mark the
# exceptional one, so they appear at most twice in a run.
#
# * ``_ANSI_FAIL`` (red) -- ``run_summary``'s FAILED verdict, and nothing else.
#   Snakemake's own error block is already red, and the verdict is the line a
#   reader scrolls to the bottom to find; in the routine three it was grey.
#   The SUCCESS verdict stays uncoloured on purpose. Colouring both would make
#   the pair a status field to be read, when the whole value here is that a
#   failed run looks different from every other run without being read.
# * ``_ANSI_WARN`` (yellow) -- the heartbeat's ``still running`` notice and its
#   ``failed after`` verdict. A stall notice is the console saying it does not
#   know whether anything is wrong, which is neither routine nor an error, and
#   is exactly what yellow means everywhere else. Its ``done in`` all-clear
#   stays body-tier: the alarm is the news, the resolution is not.
_ANSI_RUN = "94"  # bright blue
_ANSI_DONE = "92"  # bright green
_ANSI_BODY = "38;5;250"  # light grey
_ANSI_FAIL = "91"  # bright red
_ANSI_WARN = "93"  # bright yellow
_ANSI_RESET = "\033[0m"


def _ansi(text, code):
    """Wrap ``text`` in an SGR code. Callers decide WHETHER to colour."""
    return f"\033[{code}m{text}{_ANSI_RESET}"


def _console_colour(stream):
    """Whether to colour output written to ``stream``.

    A live terminal and no ``NO_COLOR``. Asked of the REAL console stream, never
    of a tee -- ``_Tee.isatty`` reports False by design, so a tee asked about
    itself would answer "not a terminal" while writing to one.
    """
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty()) and not os.environ.get("NO_COLOR")


def _paint_body(text, colour, code=_ANSI_BODY):
    """Paint a console-bound chunk, in the body tier by default.

    Returns ``text`` unchanged when not colouring. Named for the TIER and
    not for the hue: the colour is one constant away from being something
    else, and the name this replaced said "grey" while ``_ANSI_BODY`` was
    what actually decided.

    ``code`` overrides the tier for the heartbeat's stall notices, which are
    the same shape of chunk -- one console-bound line, never a log line -- but
    are not routine. Passing the code rather than adding a second function is
    what keeps the carriage-return and whitespace rules below applying to
    every painted chunk, since those are properties of the CHUNK, not of the
    colour.

    **Chunks containing a carriage return pass through untouched.** Those are
    in-place progress bars (dask's ``[####] | 100% Completed``), which redraw
    many times a second: wrapping each redraw would put an SGR pair around every
    frame, and a reset landing mid-bar flickers. They stay uncoloured and stay
    animated, which is the trade the tee already makes for them.

    Whitespace-only chunks are left alone too -- ``print`` commonly arrives as
    two writes, the text then the newline, and colouring a bare newline emits an
    escape pair around nothing.

    Wrapping is per LINE within the chunk, never around the whole of it, so a
    colour never spans a newline. A chunk is frequently ``"...text\\n"`` and can
    hold several lines; wrapping it whole would leave the reset sitting at the
    start of the NEXT line, so a terminal reflowing on resize carries the styling
    across the break.
    """
    if not colour or "\r" in text or not text.strip():
        return text
    return "\n".join(
        _ansi(line, code) if line.strip() else line for line in text.split("\n")
    )


# Rule name -> its ``W.NN`` number, filled by every ``rule_banner`` call at
# Snakefile PARSE time. The console handler below needs the number when a job
# FINISHES, and Snakemake's finish record carries only a jobid and a rule name
# -- never the ``message:`` the start record carried. Recovering the number by
# parsing the assembled banner would mean parsing prose (the summary clause and
# the per-job context both sit in that one string); a registry keyed on the one
# field both records DO share costs nothing and cannot misparse. One Snakefile
# is parsed per process, so a name maps to one number.
_RULE_NUMBERS = {}

# Rule name -> its constant ``summary`` clause, filled by the same call. The
# console handler prints that clause on a rule's FIRST start line only and trims
# it from the rest (see ``_ConsoleHandler._start_line``), which needs the exact
# substring ``rule_banner`` inserted -- so it is recorded here rather than
# recovered by splitting the assembled banner on its separators. A rule with no
# summary is absent, and trimming then does nothing.
_RULE_SUMMARIES = {}


def rule_id(number):
    """Return a rule number in the console's spelling: ``Rule 1.08:``.

    One definition, because three places have to agree on it: the ``message:``
    banner a rule declares, the START line the console handler renders from
    that banner, and the FINISH line it builds from a record that carries only
    a rule name. Two spellings of one identity on adjacent lines is how a pair
    stops reading as one job.
    """
    return f"Rule {number}:"


def rule_banner(number, name, context=None, summary=None):
    """Return a rule's ``message:`` string: a numbered console banner.

    Shows ``<W.NN>  <name>`` (the ``W.NN`` matching the rule's log/benchmark
    filenames) so the live Snakemake console is easy to track.

    **Returns PLAIN TEXT — it never colours.** Colour belongs to whoever writes
    the line, and this string is written to three places with three answers:
    the console (where ``_ConsoleHandler`` paints the whole start line blue),
    ``.snakemake/log/*.snakemake.log`` (a file, which must stay clean), and
    Snakemake's ``JOB_ERROR`` block (where red is the point and our styling
    would fight it). It coloured its own fields until 2026-08-14, which made
    the second of those a file with escape codes in it whenever stderr happened
    to be a terminal.

    ``context`` appends a per-job suffix, and is the answer for FAN-OUT
    rules. This helper is evaluated once at Snakefile parse time, so without it
    every member of a fanned-out rule prints an IDENTICAL banner: on a
    multi-hour WF3 run the console says *something is running*, never *which
    member*. The return value becomes ``message:``, which Snakemake formats per
    job — so a ``context`` holding ``{wildcards.<name>}`` resolves per member
    even though the banner itself was built once.

    Two constraints on what a caller may pass:

    * Only wildcards the rule actually declares. The message is formatted
      against that job's namespace, so a stray field fails at RUN time, not at
      parse time — `tests/test_snake_utils.py` pins the shape, and
      `tests/test_cli.py`'s dry-run is what exercises the real namespaces.
    * **ASCII only.** A Windows console defaults to cp1252 and raises
      ``UnicodeEncodeError`` on the typographic separators that would read
      better here. Use ``|``, not ``·``.

    A rule with no wildcards may still pass a constant context; only the
    *interpolation* needs a wildcard, not the suffix itself.

    The context is BRACKETED (``[rlz 1 | st 2]``). Without ANSI -- a pipe, a
    redirect, CI -- the banner's fields rest on nothing but ``-`` and a double
    space, and the trimming described below removes the ``-`` from most start
    lines, leaving two adjacent runs of text separated by whitespace alone.
    Brackets keep the qualifier separable in every one of the three
    destinations, at the cost of two characters.

    ``summary`` is a plain-language clause saying what the rule DOES, for the
    rules a person waits on: ``1.14  run_wflow`` is an identifier, and someone
    watching a multi-hour run should not have to know the codebase to read the
    console. Applied to the LONG-RUNNING rules only, per the parked note this
    discharges — a sentence on all 47 would lengthen every line to say what the
    fast ones already say by name, and the value is precisely in the rules where
    you are waiting and wondering. It is constant, so unlike ``context`` it must
    not contain a wildcard.

    Being constant is exactly why the CONSOLE prints it once per rule and not
    once per job: the long-running rules are also the FANNED-OUT ones, so a
    summary reprinted per member is the same sentence on every line of the
    longest stretch of a WF3 run — 400 identical clauses on a 10 x 20 grid,
    since rules 3.12 and 3.14 are one job per member. The string returned here
    is unchanged and always carries it; ``_ConsoleHandler._start_line`` does the
    trimming, because the other two destinations want the whole sentence on
    every line. The log file is grepped a line at a time, and a ``JOB_ERROR``
    block is read in isolation from whatever scrolled past hours earlier.

    Order is ``Rule <number>: <name> - <summary>  <context>``: identifier first
    because it is what the log filenames, the benchmark table and this file's
    own rule comments all key on.

    The identifier is SPELLED OUT (``Rule 1.08: build_wflow_model``) rather than
    left as a bare ``1.08  build_wflow_model``. Two digits and a dot are a rule
    number to someone who already knows this console; to everyone else they are
    an unexplained figure sitting where a version or a count could equally well
    be, and the word costs five characters once per line.

    Side effect: records ``name -> number`` in ``_RULE_NUMBERS`` so
    :func:`install_console_style` can put the number on a job's FINISH line,
    which Snakemake reports by rule name only, and ``name -> summary`` in
    ``_RULE_SUMMARIES`` for the once-per-rule trimming above. See those
    registries' comments.
    """
    _RULE_NUMBERS[name] = str(number)
    if summary:
        _RULE_SUMMARIES[name] = summary
    tag = f"{rule_id(number)} {name}"
    if summary:
        tag = f"{tag} - {summary}"
    if not context:
        return tag
    return f"{tag}  [{context}]"


def format_elapsed(seconds):
    """``h:mm:ss`` for a duration, matching the benchmark tables' own column."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def run_summary(
    workflow,
    project_dir,
    log_name,
    benchmarks_name,
    elapsed_seconds=None,
    failed=False,
    log_parts_dir=None,
):
    """Return the end-of-run console block for an ``onsuccess``/``onerror``.

    Snakemake ends a run with its own one-line verdict and nothing about what
    the run PRODUCED. Two artifacts every run of this toolbox writes are
    consequently invisible unless you already know they exist: the merged log
    (rule W.17/W.18 folds the per-rule parts into one file, then deletes them)
    and the benchmark table (a rule column plus a TOTAL row). This names both.

    Reported as PATHS rather than contents: they are the two things a person
    needs in order to answer "what happened", and printing either inline would
    reproduce on every run the noise these console changes exist to remove.

    ``elapsed_seconds`` is optional because a Snakefile has to measure it
    itself -- Snakemake exposes no run duration to these handlers. It is
    wall-clock from Snakefile PARSE, so it includes DAG construction; that is a
    second or two on these workflows and is not worth a second clock.

    Deliberately absent: a job count. Neither handler is given one, and
    reconstructing it from the DAG would report jobs SCHEDULED rather than jobs
    RUN -- a number that reads as authoritative and is wrong whenever anything
    was already up to date.

    The failure form names the log-parts directory as well, because on failure
    the merged log does not exist yet: rule W.17 is a normal rule and does not
    run when an upstream job fails, so the per-rule parts are still the only
    record. ``show-failed-logs`` (profiles/default/config.yaml) prints the
    failing job's own log inline; this points at everything around it.

    ``log_parts_dir`` is passed rather than derived: WF3 keys its parts by
    experiment (``logs/_parts/<experiment>``), so a derived ``logs/_parts``
    would send the reader to the parent of the directory they want.

    Grouped and labelled like :func:`run_header`, so a run closes in the shape
    it opened in: a head line, a blank, then a labelled group whose rows indent
    one step further. The label is a VERB (``wrote``) because that is what these
    rows are -- artifacts this run produced -- as against the header's ``run``
    (what this run is) and ``path tokens`` (how to read the lines between).

    A FAILED verdict is painted red (``_ANSI_FAIL``), gated on stderr being a
    colour console. The success verdict is not painted at all: what matters is
    that a failed run looks different from every other run WITHOUT being read,
    and colouring both would turn the pair into a field to be read instead.
    """
    project_dir = os.fspath(project_dir)
    lines = []
    verdict = "FAILED" if failed else "done"
    head = f"{workflow} {verdict}"
    if elapsed_seconds is not None:
        head = f"{head} in {format_elapsed(elapsed_seconds)}"
    if failed:
        # Coloured HERE, unlike `rule_banner`, and the difference is where the
        # string goes: a banner reaches a log file and an error block, where an
        # escape code is corruption. This block reaches the console and nothing
        # else -- every Snakefile writes it to stderr and no log receives it --
        # so painting it here cannot leak. Asked of stderr for the same reason.
        head = _paint_body(head, _console_colour(sys.stderr), _ANSI_FAIL)
    lines.extend([head, "", "  wrote"])
    if failed:
        parts = os.fspath(log_parts_dir or f"{project_dir}/logs/_parts")
        rows = [("log parts", f"{parts}/")]
    else:
        rows = [
            ("log", f"{project_dir}/logs/{log_name}"),
            ("benchmarks", f"{project_dir}/benchmarks/{benchmarks_name}"),
        ]
    width = max(len(key) for key, _ in rows)
    lines.extend(f"    {key.ljust(width)}  {value}" for key, value in rows)
    if failed:
        # A NOTE, not a row: it names no artifact, so giving it a key column
        # would file a sentence under a heading meaning "paths this run wrote".
        lines.extend(["", "  the failing job's own log is printed above"])
    return "\n".join(lines)


def target_banner(number, name, targets, project_dir=None):
    """Return a `rule all` ``message:``: the banner, then one target per line.

    Snakemake joins a job's ``input:`` with ``", "``, which collapses a target
    aggregator's whole product list onto one unreadable line — nine absolute
    paths in a single wrap-around blob on WF2. No CLI flag changes that joiner;
    a rule's ``message:`` is the only lever, and it REPLACES the default block
    (``rule``/``input``/``output``/``jobid``/``resources``) rather than
    reformatting part of it.

    That trade is free for `rule all` specifically, which is why this helper is
    scoped to it: a target aggregator has no ``output:``, its jobid is always the
    root, and it declares no resources, so the replaced block carried nothing
    the target list does not. Do NOT reach for this on a working rule — there it
    would hide the output paths and the jobid a failure report needs.

    Indented four spaces to sit where Snakemake's own ``input:`` values sit.

    With ``project_dir`` the targets print RELATIVE to it, which is what makes a
    deep tree legible — ``data/climate/projections/cmip6/summary/x.csv`` rather
    than the same path behind 40 characters of absolute prefix. The root is then
    appended to the banner in brackets, because a relative path with no stated
    root is ambiguous: the reader must still be able to reconstruct the full
    path, and one root on one line beats repeating it nine times. Without
    ``project_dir`` the targets print exactly as given.

    Relativization is :func:`_relativize_paths`, so it strips the root in both
    native and forward-slash form — Snakefiles build these paths with ``/``
    while ``project_dir`` may arrive either way.

    Evaluated once at Snakefile parse time, like :func:`rule_banner`.
    """
    banner = rule_banner(number, name)
    listed = [os.fspath(target) for target in targets]
    if project_dir:
        root = os.path.normpath(os.fspath(project_dir))
        tokens = _path_tokens()
        listed = [_relativize_paths(target, root, tokens) for target in listed]
        banner = f"{banner}  [{root.replace(os.sep, '/')}]"
    body = "\n".join(f"    {target}" for target in listed)
    return f"{banner}\n{body}" if body else banner


# --------------------------------------------------------------------------
# Console style: Snakemake's own terminal output, in this toolbox's grammar
# --------------------------------------------------------------------------

# Plain INFO lines Snakemake emits that tell a reader of THIS console nothing.
# Matched on leading text because Snakemake attaches neither an event nor a
# quietness class to them -- there is nothing structural to key on. Matching
# prose is fragile across Snakemake versions, and it fails in the SAFE
# direction: a reworded line stops matching and simply prints, which is the
# behaviour we started from.
#
# `Removing temporary output` is the one with volume: WF3 wraps every
# per-realization netCDF in `temp()`, so a full run prints hundreds of them.
# They stay in `.snakemake/log/*.snakemake.log`, which this handler does not
# touch -- muted on the terminal, not lost.
#
# Snakemake's fixed PREAMBLE (`Assuming unrestricted shared filesystem usage.`,
# `host: ...`, `Provided cores: N`) is deliberately absent: it is printed before
# `onstart`, so no Snakefile hook can be installed early enough to mute it. It
# is a handful of lines once per run; the volume was never there.
_CONSOLE_MAX_TRACKED_JOBS = 4096

#: The two words that open a job line, PADDED TO ONE WIDTH so the identity that
#: follows starts at the same column on a start and a finish -- the pair reads
#: as a column, not as two sentences. Upper case because they are the only
#: markers on the console and nothing else competes for that weight.
_MARKER_RUN = "RUN "
_MARKER_DONE = "DONE"

_CONSOLE_MUTED_PREFIXES = (
    "Select jobs to execute...",
    "Removing temporary output ",
    "Would remove temporary output ",
    "Touching output file ",
)


def _console_wildcards(wildcards):
    """Render a job's wildcards in the banner's own grammar (``[rlz 1 | st 0]``).

    Snakemake's ``format_wildcards`` spells these ``rlz=1, st=0``; the finish
    line sits directly under a start line whose context clause reads
    ``[rlz 1 | st 0]``, and two spellings of one fact on adjacent lines is
    exactly the inconsistency the console work has been removing. That includes
    the brackets :func:`rule_banner` puts around its context. Derived from the
    job's wildcards rather than parsed back out of the banner, so no prose is
    ever parsed.

    A trailing ``_num`` is DROPPED from the key, and that is what makes the
    claim above true rather than merely intended. The wildcard is ``rlz_num``
    because `dev/reference/naming.md` says a count-or-index field carries the
    suffix; every banner context in WF3 writes ``rlz {wildcards.rlz_num}``,
    dropping it because on a console the column header and the value are
    adjacent and the suffix says nothing the value does not. Until this, the
    start line said ``[rlz 1 | st 2]`` and the finish line under it said
    ``[rlz_num 1 | st_num 2]`` — one fact, two spellings, which is the exact
    defect this function's own docstring claimed to have fixed. The convention
    is now stated in one place instead of being restated per banner.
    """
    try:
        items = list(wildcards.items())
    except AttributeError:
        return ""
    if not items:
        return ""
    fields = " | ".join(f"{_console_wildcard_key(key)} {value}" for key, value in items)
    return f"[{fields}]"


def _console_wildcard_key(key):
    """``rlz_num`` -> ``rlz``: the console's spelling of an index wildcard.

    Only a TRAILING ``_num``, and only when something is left of it, so a
    wildcard actually named ``num`` keeps its name.
    """
    return (
        key[: -len("_num")] if key.endswith("_num") and len(key) > len("_num") else key
    )


class _ConsoleHandler(logging.StreamHandler):
    """Snakemake's terminal handler, restyled: one line per job start and end.

    Snakemake spends SIX console lines on a job -- a blank separator, a bare
    ``[Thu Aug 13 23:26:41 2026]``, ``Job 8: <message>``, a second bare
    timestamp, ``Finished jobid: 8 (Rule: check_project_consistency)`` and
    ``1 of 37 steps (3%) done`` -- plus ``Select jobs to execute...`` and
    ``Execute N jobs...`` per scheduling wave. On a WF3 run that is thousands
    of lines in which one line per job is ours. This renders two::

        23:26:44 - RUN  Rule 3.11: generate_weather_realizations - generate ...
        23:27:14 - DONE Rule 3.11: generate_weather_realizations  0:00:30  [8/37]

    On a FANNED-OUT rule the summary clause is printed on the first member only
    and trimmed from the rest (:meth:`_trim_summary`), so the run's longest
    stretch reads as a list of members rather than one sentence restated a few
    hundred times::

        16:16:57 - RUN  Rule 3.14: downscale_climate_realization - downscale ...  [rlz 2 | st 0]
        16:16:57 - RUN  Rule 3.14: downscale_climate_realization  [rlz 2 | st 2]
        16:16:58 - RUN  Rule 3.14: downscale_climate_realization  [rlz 2 | st 3]

    The stamp is followed by ``- `` and a padded marker, so these lines open in
    the same grammar as every row :func:`log_row` and the tee emit between them
    (``23:26:51 - geoms - Writing geoms to ...``) and the identity lands in one
    column on both. Note the two ``-`` on a start line separate different
    things: the first ends the stamp, the second introduces the rule's
    plain-language summary.

    Start lines are painted blue and finish lines green, WHOLE-LINE, with
    everything else on the console light grey (``_ANSI_RUN`` / ``_ANSI_DONE``
    / ``_ANSI_BODY``). Colour is applied here rather than inside ``rule_banner``
    because the banner string also reaches a log file and an error block, where
    an escape code is corruption in one and a fight with red in the other.

    Both lines carry the same ``W.NN  name`` identifier the merged log, the
    benchmark table and the rule comments key on, so one grep finds a rule's
    whole story. Timestamps are ``HH:MM:SS``, matching :func:`log_row` rather
    than introducing ``time.asctime`` as a second clock format.

    **Only the terminal changes.** Snakemake's own
    ``.snakemake/log/*.snakemake.log`` is written by a separate handler this
    never touches, so the verbose record survives for anyone who wants it.

    **Errors are delegated, never reformatted.** ``JOB_ERROR`` /
    ``GROUP_ERROR`` and every unrecognized record go to Snakemake's own
    formatter, which is what carries the ``show-failed-logs`` inline log dump
    and the input/output/shellcmd block a failure report needs. This class
    reshapes the routine path and stays out of the way on the one that matters.

    Events are compared as PLAIN STRINGS. ``LogEvent`` is a ``StrEnum``, so
    ``record.event == "job_info"`` holds for the member itself -- which keeps
    this class importable, and unit-testable, without snakemake installed.

    **``--quiet`` still means what it means**, because the inherited filter runs
    BEFORE this handler and can remove either half of a pair. ``--quiet rules``
    drops the start records, so a finish falls back to Snakemake's own
    ``Finished jobid: N (Rule: x)`` text plus our stamp and counter -- still one
    line, still naming the rule. ``--quiet progress`` drops the finish records
    entirely, leaving only start lines, which is what asking for no progress
    reporting should give; the start memo is capped so that mode cannot grow a
    dict without bound. Neither combination was hypothetical: both were run.
    """

    def __init__(self, base):
        super().__init__(stream=getattr(base, "stream", sys.stderr))
        # Inherit the handler being replaced rather than reconstructing it:
        # its formatter and filter were built from the run's real settings
        # (`quiet`, `show_failed_logs`, `printshellcmds`, `dryrun`), which are
        # not reachable from a Snakefile.
        self.name = getattr(base, "name", None)
        formatter = getattr(base, "formatter", None)
        if formatter is not None:
            self.setFormatter(formatter)
        for inherited in list(getattr(base, "filters", ()) or ()):
            self.addFilter(inherited)
        self._started = {}  # jobid -> (rule name, wildcards, monotonic start)
        self._finished = []  # jobids awaiting the progress record's counter
        # Rule names whose summary clause has already been printed once. Held on
        # the INSTANCE, unlike `_RULE_NUMBERS`/`_RULE_SUMMARIES`, which are
        # module-level and outlive one workflow: `run_workflows.py` drives four
        # Snakefiles, and a shared set would give the second and later ones a
        # console on which no summary was ever printed at all.
        self._summarized = set()
        self._emit_lock = threading.Lock()
        isatty = getattr(self.stream, "isatty", None)
        self._color = bool(isatty and isatty()) and not os.environ.get("NO_COLOR")

    # -- emission ----------------------------------------------------------

    def emit(self, record):
        try:
            with self._emit_lock:
                text = self._render(record)
            if not text:
                return
            self.stream.write(text + self.terminator)
            self.flush()
        except BrokenPipeError:
            raise
        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception:  # noqa: BLE001 -- a console style must never end a run
            self.handleError(record)

    def close(self):
        """Flush any finish line still waiting for its progress counter."""
        try:
            pending = "\n".join(self._drain(None, None))
            if pending:
                self.stream.write(pending + self.terminator)
                self.flush()
        except Exception:  # noqa: BLE001 -- teardown must not raise
            pass
        super().close()

    # -- rendering ---------------------------------------------------------

    def _render(self, record):
        fields = record.__dict__
        event = fields.get("event")
        event = None if event is None else str(event)

        # A finish line is HELD until the progress record that follows it, which
        # is the only place Snakemake reports the done/total counter. Holding
        # rather than counting ourselves keeps one authority for the number: an
        # independent counter drifts the moment a job is restarted, grouped, or
        # already up to date. The scheduler emits progress immediately after
        # each finish (`scheduler.py`), and anything else arriving first drains
        # the queue uncounted, so a held line cannot be lost.
        if event == "job_finished":
            # The record's own text is kept as a fallback identity. Under
            # `--quiet rules` the inherited filter drops JOB_INFO, so there is
            # no memo to look up and `Finished jobid: 5 (Rule: seed)` is the
            # only thing naming the job -- see `_done_line`.
            self._finished.append((fields.get("job_id"), str(record.msg or "")))
            return None

        if event == "progress":
            lines = self._drain(fields.get("done"), fields.get("total"))
        else:
            lines = self._drain(None, None)
            if event == "job_info":
                lines.append(self._start_line(fields, record))
            elif event == "job_started":
                pass  # "Execute N jobs..." -- scheduler bookkeeping
            elif not self._muted(record, event):
                # Body tier for what is informational only. A WARNING or an ERROR keeps
                # Snakemake's own colouring, which is the one thing on this
                # console that must stay louder than a start line.
                #
                # Shortened the same way a rule's own output is (the tee calls
                # the same function): Snakemake's `Complete log(s):` line ends a
                # run with a 100-character absolute path, in OS separators,
                # directly above a summary block whose every path is short and
                # forward-slashed. One console, one spelling of a path.
                shown = _relativize_paths(self.format(record), "", _path_tokens())
                if record.levelno <= logging.INFO:
                    shown = self._paint(shown, _ANSI_BODY)
                lines.append(shown)

        return "\n".join(line for line in lines if line) or None

    def _muted(self, record, event):
        """Whether a plain INFO line is one of the muted ones.

        Restricted to event-less records at INFO or below, so a warning or an
        error can never be silenced by a prefix that happens to match.
        """
        if event is not None or record.levelno > logging.INFO:
            return False
        return str(record.msg or "").startswith(_CONSOLE_MUTED_PREFIXES)

    def _start_line(self, fields, record):
        # Memoized BEFORE the branch below, so a rule with no `message:` still
        # gets a finish line naming it. Skipping the memo left those jobs
        # finishing as a bare `done  job 9`, which is the one thing the finish
        # line exists to avoid.
        self._started[fields.get("jobid")] = (
            fields.get("rule_name"),
            _console_wildcards(fields.get("wildcards")),
            time.monotonic(),
        )
        # Bounded, because under `--quiet progress` the inherited filter drops
        # JOB_FINISHED and nothing ever pops an entry -- an unbounded dict on a
        # run with thousands of jobs. Insertion order makes the oldest entry the
        # one least likely to still be running. The cap is far above any real
        # in-flight count (WF3's widest fan-out is one job per rlz x st), so a
        # normal run never reaches it.
        while len(self._started) > _CONSOLE_MAX_TRACKED_JOBS:
            self._started.pop(next(iter(self._started)))
        message = fields.get("rule_msg")
        if not message:
            # A rule with no `message:`. Snakemake's default block names its
            # input/output/jobid, which is the whole of what a reader gets for
            # that rule -- reshaping it into one line would delete it.
            return self.format(record)
        message = self._trim_summary(fields.get("rule_name"), message)
        return self._paint(f"{self._now()} - {_MARKER_RUN} {message}", _ANSI_RUN)

    def _trim_summary(self, rule_name, message):
        """Drop the rule's constant summary clause after its FIRST start line.

        The clause says what the rule DOES, which is a fact about the RULE and
        not about the job -- so on a fanned-out rule it is the same sentence on
        every line for the length of the fan-out. Printed once, the reader has
        it; repeated, it is the widest column on screen carrying no per-job
        information. What remains is ``Rule 3.14: downscale_climate_realization
        [rlz 2 | st 2]``, which is the grammar :meth:`_done_line` already builds
        -- it has never carried a summary -- so the pair converges rather than
        the start line acquiring a format of its own.

        Removal is by the EXACT substring ``rule_banner`` inserted, looked up by
        rule name in ``_RULE_SUMMARIES``. Nothing is parsed out of the assembled
        message: a rule whose banner was built some other way is simply absent
        from the registry and prints unchanged, which is also what happens if
        the two ever fall out of step.

        Keyed on the rule NAME, so WF3's per-batch rules (``run_wflow_batch_1``,
        ``_2``, ...) are distinct rules and each prints its summary once. That
        is a handful of lines on a run, and it is deliberate: they are separate
        rules with separate numbers everywhere else on this console.
        """
        if not rule_name:
            return message
        if rule_name not in self._summarized:
            self._summarized.add(rule_name)
            return message
        summary = _RULE_SUMMARIES.get(rule_name)
        if not summary:
            return message
        return message.replace(f" - {summary}", "", 1)

    def _drain(self, done, total):
        if not self._finished:
            return []
        finished, self._finished = self._finished, []
        # `done` counts every job finished so far, so a batch of held lines
        # ends AT it and counts backwards from there.
        first = None if done is None else done - len(finished) + 1
        return [
            self._done_line(
                jobid, fallback, None if first is None else first + offset, total
            )
            for offset, (jobid, fallback) in enumerate(finished)
        ]

    def _done_line(self, jobid, fallback, counter, total):
        rule_name, wildcards, started = self._started.pop(jobid, (None, "", None))
        if rule_name is None and fallback:
            # No start line was seen for this job, so there is nothing to render
            # in our grammar -- under `--quiet rules` the filter dropped it, and
            # a grouped job never emits one. Snakemake's own text still names
            # the rule, which is the whole point of the line; it gets our stamp
            # and the counter and nothing else.
            tail = f"  [{counter}/{total}]" if counter is not None and total else ""
            return self._paint(
                f"{self._now()} - {_MARKER_DONE} {fallback}{tail}", _ANSI_DONE
            )
        number = _RULE_NUMBERS.get(rule_name)
        if rule_name:
            identity = f"{rule_id(number)} {rule_name}" if number else rule_name
        else:
            identity = f"job {jobid}"
        # The identity is coloured HERE, not inherited from the start line's
        # banner: this line is built from the finish record, and the two must
        # look alike or a pair reads as two unrelated events.
        parts = [identity]
        if wildcards:
            parts.append(wildcards)
        tail = []
        if started is not None and time.monotonic() - started >= 1:
            # Measured from the START record, which Snakemake logs at submission
            # rather than at first instruction. The difference is the scheduler's
            # own dispatch, well under a second on these workflows; the
            # `benchmark:` TSVs remain the authority for a quotable duration.
            #
            # A sub-second job shows NO duration rather than `0:00:00`, which
            # reads as a broken clock. Many rules here are bookkeeping that
            # finishes instantly, so this is the common case, not an edge one.
            tail.append(format_elapsed(time.monotonic() - started))
        if counter is not None and total:
            tail.append(f"[{counter}/{total}]")
        line = f"{self._now()} - {_MARKER_DONE} " + "  ".join(parts)
        if tail:
            line = f"{line}  " + "  ".join(tail)
        return self._paint(line, _ANSI_DONE)

    # -- decoration --------------------------------------------------------

    def _now(self):
        return f"{datetime.now():%H:%M:%S}"

    def _paint(self, text, code):
        """Colour a WHOLE line. Nothing here paints a field."""
        return _ansi(text, code) if self._color and text else text


def install_console_style():
    """Restyle Snakemake's terminal output; return whether it took effect.

    **Call it from ``onstart:``, not from the top of the Snakefile.** That is
    measured, not stylistic: Snakemake parses the workflow BEFORE it builds its
    logging stack, so at parse time ``logger_manager.queue_listener`` is still
    ``None`` and this returns ``False`` having done nothing (probed
    2026-08-14 -- the first attempt put the call beside
    :func:`patch_psutil_windows_benchmark`, and every run came out in
    Snakemake's own style with nothing reporting why). By ``onstart`` the
    listener exists and no job record has been emitted yet, so the whole
    execution phase is covered; only the fixed preamble above it is not.
    ``queue_listener.handlers`` is read per record inside
    ``QueueListener.handle``, so replacing the tuple takes effect immediately
    and needs no restart.

    Selection is by ``handler.name == "DefaultStreamHandler"``, and that
    exactness is load-bearing: the same listener also holds
    ``DefaultLogFileHandler``, which writes ``.snakemake/log/*.snakemake.log``.
    Restyling that one would destroy the verbose durable record whose whole
    value is being verbose.

    Everything is guarded and FAIL-OPEN. None of this is public API, so a
    Snakemake upgrade that renames the handler or restructures the manager
    leaves the console exactly as Snakemake ships it and the run proceeds --
    a cosmetic layer must never be able to stop a workflow. The return value
    is for tests; no caller should branch on it.

    Not a Snakemake logger plugin, deliberately. That interface
    (``--logger <name>``) resolves through installed entry points, so using it
    would mean shipping and depending on a separate package to restyle our own
    console -- a new dependency for a cosmetic layer.
    """
    try:
        from snakemake.logging import logger_manager

        listener = getattr(logger_manager, "queue_listener", None)
        handlers = list(getattr(listener, "handlers", None) or ())
        for index, handler in enumerate(handlers):
            if getattr(handler, "name", None) != "DefaultStreamHandler":
                continue
            if isinstance(handler, _ConsoleHandler):
                return True  # a second Snakefile in one process (tests)
            handlers[index] = _ConsoleHandler(handler)
            listener.handlers = tuple(handlers)
            return True
    except Exception:  # noqa: BLE001 -- never fail a run over console styling
        return False
    return False


def run_header(workflow, project_dir, config_path=None, **details):
    """Return the start-of-run console block: what run this is, in one place.

    The mirror image of :func:`run_summary`, and the same grammar (a head line,
    then indented ``key   value`` rows) so a run opens and closes alike -- the
    header groups and labels its rows, for the reason set out below, and is the
    longer of the two. Snakemake's own preamble names the host and job counts but
    never the PROJECT -- so a console scrolled back to, or pasted into a
    message, could not be attributed to a project, a config or an experiment
    without asking. The merged log has carried this header for some time; the
    terminal had nothing.

    ``details`` are extra rows in call order (WF3 passes ``experiment=``),
    keeping the block to what a workflow actually has rather than a fixed set
    with blanks in it.

    The run's DECLARED key folders (:func:`declare_path_tokens`) close the
    block, and they are read from the declaration rather than passed in: these
    same names are what every path in every rule's output is rewritten to, so a
    header row and a body line that disagreed would be worse than no header at
    all. One declaration, two readers.

    **Two GROUPS, blank-line separated, each under a label.** The rows are two
    different kinds of thing and read as one wall of text without the split:
    the first three answer "which run is this", while the ``<name>`` rows are a
    LEGEND -- they define the tokens every path in every line below is printed
    with. Nothing said so, which left a reader to infer from angle brackets
    alone why ``<model>`` was written that way, and the block ran on into
    Snakemake's own ``Job stats:`` with no air between them.

    The labelled rows indent one step further than the label, which is what
    makes the grouping visible without a rule or a box. :func:`run_summary`
    closes the run in the same shape, under one group labelled ``wrote``.

    Every value is FORWARD-SLASHED, including ``config`` -- which was the one
    row that was not, so a block whose whole purpose is to state paths mixed
    ``C:\\a\\b`` with ``a/b`` and read as two trees. ``config`` is also printed
    through the repo strip that every rule log line already uses, so it reads
    ``<repo>/test_case/snake_config_rapid.yml`` rather than the same fact
    behind 60 characters of machine-specific prefix.
    """
    # Forward slashes, like every path the folder rows below and the log
    # headers print: one block mixing `C:\a\b` with `a/b` reads as two trees.
    run_rows = [("project", os.fspath(project_dir).replace(os.sep, "/"))]
    if config_path:
        # No project root passed: the config is not a project artifact, and
        # stripping one would render a config that happens to live INSIDE the
        # project as a bare relative path indistinguishable from an output.
        # This still applies the `<repo>` and `<site-packages>` rewrites.
        run_rows.append(("config", _relativize_paths(os.fspath(config_path), "")))
    run_rows.extend((key, str(value)) for key, value in details.items())
    token_rows = _folder_rows(project_dir)
    # One width across BOTH groups, so the value column is a single column down
    # the whole block rather than restarting at each label.
    width = max(len(key) for key, _ in run_rows + token_rows)

    def row(key, value):
        return f"    {key.ljust(width)}  {value}"

    lines = [workflow, "", "  run"]
    lines.extend(row(key, value) for key, value in run_rows)
    if token_rows:
        lines.append("")
        lines.append(
            "  path tokens -- these folders print as <name> in every line below"
        )
        lines.extend(row(key, value) for key, value in token_rows)
    return "\n".join(lines)
