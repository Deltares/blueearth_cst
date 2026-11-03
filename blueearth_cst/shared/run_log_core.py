"""Neutral log and console helpers shared with WF3.

The generation code inventory includes this whole file. WF4 frame handling is
injected by ``snake_utils``; this module does not import that adapter.
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
from datetime import datetime, timedelta
from pathlib import Path

from blueearth_cst.shared.console_terminal import restore_terminal_streams

restore_terminal_streams()

_HYDROMT_LOG_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} (\d{2}:\d{2}:\d{2}),\d{3} - \S+ - (\S+) - (\w+) - (.*)$"
)

_COMPONENT_PREFIX_RE = re.compile(r"^\w+\.(\w+): (.*)$")

_DATA_SOURCE_READ_RE = re.compile(
    r"^(Reading \S+) "
    r"(?:RasterDataset|GeoDataFrame|GeoDataset|DataFrame|Dataset)"
    r" data from "
)

_DATA_SOURCE_FROM_RE = re.compile(r"^Reading (\S+) from (\S+)$")


_GENERATION_MESSAGES = (
    (
        r"Reading weather netcdf: (.+)",
        r"Reading historical climate → \1",
    ),
    (
        r"Resampling on (\d+) basin cell\(s\) of (\d+) in the store",
        lambda m: (
            f"Weather resampling uses {plural(int(m[1]), 'basin-intersecting cell')} from the {m[2]}-cell extracted grid"
        ),
    ),
    (
        r"Generating (\d+) weather realization\(s\)",
        lambda m: f"Generating {plural(int(m[1]), 'stochastic weather realization')}",
    ),
    (
        r"Moving (\d+) figure\(s\) to (.+)",
        r"Moving diagnostic figures → \2",
    ),
    (r"Saving realization (\d+) of (\d+)", r"Writing stochastic realization \1 of \2"),
    (r"(\S+) reading realization (.+)", r"\1: reading source realization \2"),
    (
        r"(\S+) applying perturbations \((.*)\)",
        r"\1: applying climate perturbations (\2)",
    ),
)


def _presentation_message(module: str, message: str) -> str:
    """Present retained-producer messages without editing their scientific sources."""
    if module == "weathergen":
        for pattern, replacement in _GENERATION_MESSAGES:
            if re.fullmatch(pattern, message):
                return re.sub(pattern, replacement, message)
    if module == "experiment":
        if message == "Model reference matches the live model":
            return "Live model matches the recorded experiment identity"
        reference = re.fullmatch(
            r"Model reference: (.+) digest (\w+)\.\.\. over (\d+) inputs?", message
        )
        if reference:
            path, digest, count = reference.groups()
            return f"Using hydrological model: {path}; {plural(int(count), 'input file')}; checksum {digest}…"
        match = re.fullmatch(r"Wrote (.+) \((\d+) rows, (\d+) members?\)", message)
        if match:
            path, rows, points = match.groups()
            return f"Perturbation lookup written → {path} ({rows} monthly rows; {plural(int(points), 'perturbation point')})"
    if module == "indicators":
        match = re.fullmatch(
            r"(\S+): (\d+) declarations? over (\d+) result keys?", message
        )
        if match:
            identity, declarations, keys = match.groups()
            return f"Indicator set {identity} planned: {plural(int(declarations), 'indicator declaration')}; {plural(int(keys), 'expected result key')}"
    if module == "responses" and message.startswith(
        "Duplicate series on one model point, kept once: "
    ):
        return message.replace(
            "Duplicate series on one model point, kept once: ",
            "Duplicate series at the same model location will be used once: ",
            1,
        )
    if module == "wflow":
        match = re.fullmatch(
            r"FAILED \[(\d+/\d+)\] Run (\S+) batch=(\S+)  (.*)", message
        )
        if match:
            position, run_id, batch, reason = match.groups()
            return f"FAILED run {run_id} in batch {batch} [{position}]: {reason}"
    return message


def _log_row_text(hms, module, level, message):
    """Assemble one log row: ``HH:MM:SS - <module> - <message>``.

    **The level is shown only when it is not INFO**, and that is the point
    rather than a saving. On a full WF1 build 259 of 272 rows are INFO, so the
    field is a constant column on all but a handful — and a WARNING or an ERROR,
    the rows someone is scanning for, hides inside a wall of identical
    ``- INFO -``. Omitting the common case makes the uncommon one visible:

        11:13:01 - geoms - Writing geoms to basins.geojson.
        11:13:04 - states - WARNING - state file not found, using cold start

    On a colour-capable console that second row is also painted (orange for a
    warning, red for a failure) — see ``_SEVERITY_PATTERNS``, which keys off
    this very spelling. The orange console row omits the WARNING field;
    plain output and log files retain it as the durable severity signal.

    Absence therefore MEANS ``INFO``; the log header says so. Every row in these
    logs goes through here -- :func:`_compact_log_line` for hydromt's records,
    :func:`log_row` for our own, and the heartbeat's stall markers -- because
    three emitters with three spellings of one grammar is how the grammar stops
    being one.
    """
    message = _presentation_message(module, message)
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
        own = re.fullmatch(
            r"(\d{2}:\d{2}:\d{2}) - (weathergen|experiment|indicators|responses|wflow) - (?:(WARNING|ERROR|CRITICAL|DEBUG|INFO) - )?(.*)",
            core,
        )
        if own:
            hms, module, level, message = own.groups()
            return _log_row_text(hms, module, level or "INFO", message) + (
                "\n" if had_newline else ""
            )
        return text
    hms, module, level, message = match.groups()
    prefixed = _COMPONENT_PREFIX_RE.match(message)
    if prefixed and prefixed.group(1) == module:
        message = prefixed.group(2)
    # The component column already says what is being written.
    message = re.sub(rf"^Writing {re.escape(module)} to ", "Writing to ", message)
    message = _DATA_SOURCE_READ_RE.sub(r"\1 from ", message)
    message = _drop_repeated_source_name(message)
    message = _FORCING_EXISTS_RE.sub(
        r"\1 exists and overwriting is off; writing under a generated name",
        message,
    )
    # hydromt ends most messages with a full stop and our own rows end with
    # none, so a log mixed `staticmaps.nc.` and `wflow_sbm.toml.` with
    # `-> basin_cells.csv` on adjacent rows -- and a stop after a path is the
    # one place it reads as part of the path. One convention: no stop. Only a
    # SINGLE trailing stop goes, so an ellipsis (`Writing...`) is left alone.
    if message.endswith(".") and not message.endswith(".."):
        message = message[:-1]
    return _log_row_text(hms, module, level, message) + ("\n" if had_newline else "")


_FORCING_EXISTS_RE = re.compile(
    r"^Netcdf forcing file `([^`]+)` already exists and overwriting is not "
    r"enabled\. .*? A default name will be generated\.?$"
)


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
    sub-logs read e.g. ``3.09_run_wflow/rlz_1_st_1.log``). Both are ``""`` /
    the bare basename when the anchor is absent (e.g. an ad-hoc test path).
    """
    log_path = os.fspath(log_path)
    parts = os.path.normpath(log_path).split(os.sep)
    for anchor in ("logs", "benchmarks"):
        if anchor in parts:
            i = parts.index(anchor)
            root = os.sep.join(parts[:i]) if i > 0 else ""
            suffix = parts[i + 1 :]
            # Managed executions add a folder and child namespace before the
            # rule parts; keep those opaque identities out of heartbeat rows.
            if (
                len(suffix) >= 5
                and suffix[1] == "_parts"
                and suffix[3] in {"logs", "benchmarks"}
            ):
                suffix = suffix[4:]
            log_id = "/".join(suffix) or os.path.basename(log_path)
            return root, log_id
    return "", os.path.basename(log_path)


_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_SITE_PACKAGES_RE = re.compile(r"[A-Za-z]:[\\/](?:[^\\/\s]+[\\/])*?site-packages[\\/]")

_REMOTE_PREFIXES = (("gs://cmip6/CMIP6/", "<cmip6>/"),)

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
    config\\build_model``, from a ``pathlib`` value interpolated into a message) then
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


_PATH_TOKENS_ENV = "CST_PATH_TOKENS"
_CONSOLE_PATH_TOKENS_ENV = "CST_CONSOLE_PATH_TOKENS"

_PROJECT_ROOT_ENV = "CST_PROJECT_ROOT"


def _declared_tokens(environ=None, *, variable=_PATH_TOKENS_ENV):
    """Read the declared folders as ``(name, path)`` in DECLARATION order.

    The order a workflow declared them in is the order they read best in a
    header -- external data, then the model, then the run's own outputs -- and
    it survives the round trip because both ``dict`` and JSON objects preserve
    insertion order. Matching wants a different order entirely; see
    :func:`_path_tokens`.
    """
    raw = (os.environ if environ is None else environ).get(variable)
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

    A fourth, remote rather than local: the object-store prefixes in
    :data:`_REMOTE_PREFIXES`, so a CMIP6 URI reads ``<cmip6>/ScenarioMIP/...``.
    Applied first, since a URI shares no structure with the three below it.

    Order matters: the repo contains the pixi env, so ``site-packages`` is
    matched FIRST or a repo-relative rewrite would hide it. A path in none of
    the three (a data catalog under ``C:\\data\\``) is left absolute — its
    location is the information.

    ``tokens`` are the run's DECLARED key folders (see
    :func:`declare_path_tokens`), and they are applied BEFORE the project strip
    for a mechanical reason: by then a path under the project has already lost
    its root, so a token would find nothing left to match. Each is tried in its
    absolute form and, given ``project_root``, in the root-relative form too
    (:func:`_spelt_tokens`): a rule prints either, and Snakemake's own error
    block prints the relative spelling the Snakefile declared. They
    are also what makes an external data root shortenable at all; the paragraph
    above is still true of an UNDECLARED one.
    """
    text = _SITE_PACKAGES_RE.sub("<site-packages>/", text)
    # Before the local rewrites: a remote URI shares none of their structure, so
    # nothing below can match it and nothing above can be hidden by it.
    for prefix, token in _REMOTE_PREFIXES:
        text = text.replace(prefix, token)
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
    # prints the configured `test_case\test_rapid\config\build_model` as given.
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

    A folder under the project starts with <project>/ so its base is explicit.
    An external folder stays absolute. The header defines <project> alongside
    these aliases; config remains a metadata label.
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
        shown = _tokenize_prefix(path, root, "project") if root else path
        rows.append((f"<{token}>", shown.replace(os.sep, "/")))
    return rows


def _shorten_console_paths(text, project_root, tokens):
    """Apply terminal-only aliases after the shared log-path rewrites.

    Match the spelling already used by the tee, such as ``<model>/config``.
    Saved log rows retain their original catalog identities and config paths.
    Scenario data paths name the export tree available after successful completion.
    """
    aliases = _declared_tokens(variable=_CONSOLE_PATH_TOKENS_ENV)
    for token, path in sorted(aliases, key=lambda pair: len(pair[1]), reverse=True):
        shown = _relativize_paths(path, project_root, tokens)
        text = _tokenize_prefix(text, shown, token)
    # The complete collection data tree is copied to scenarios/ on success.
    # Metadata elsewhere under _engine has no equivalent and keeps its path.
    text = re.sub(
        r"(<scenarios>|<project>/scenarios)/_engine/data/[0-9a-f]{12}(?=/|\s|$)",
        r"\1",
        text,
    )
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
        lines.append(f"<project>: {root.replace(os.sep, '/')}")
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


_HEARTBEAT_LABEL_RE = re.compile(r"^(\d+\.\d+[a-z]?)_([^/]+)(?:/(.+))?$")

_MEMBER_PART_RE = re.compile(r"^[a-z]+_\d+(?:_[a-z]+_\d+)*$")


def _heartbeat_identity(label):
    """Spell a rule-log label the way the RUN and DONE lines spell the job.

    The watchdog is built from the log path, so it knows the job as
    ``2.03_fetch_cmip6_projections/cmip6_INM_...`` -- the parts directory -- while the
    lines above and below its notice say ``Rule 2.03: fetch_cmip6_projections
    [cmip6_INM_...]``. One job, two spellings on adjacent lines, which is the
    defect the console grammar exists to prevent; this is the translation.

    A member part of the ``rlz_1_st_2`` shape is rendered ``[rlz 1 | st 2]``,
    the banner's own grammar for an index wildcard (`rule_banner`); any other
    part is bracketed as it is. A label that is not ``<W.NN>_<name>`` -- a
    test's ``busy_rule``, an ad-hoc path -- is returned unchanged.
    """
    match = _HEARTBEAT_LABEL_RE.match(str(label))
    if not match:
        return str(label)
    number, name, part = match.groups()
    identity = f"{rule_id(number)} {name}"
    if not part:
        return identity
    if _MEMBER_PART_RE.match(part):
        tokens = part.split("_")
        part = " | ".join(f"{k} {v}" for k, v in zip(tokens[::2], tokens[1::2]))
    return f"{identity}  [{part}]"


class _Heartbeat:
    """Silence watchdog that keeps a long-running rule visibly alive.

    Snakemake prints only a start and a finish timestamp, so a hung job looks
    identical to a slow one until it (never) finishes. On a terminal this daemon
    maintains one replaceable status frame. On a redirected stream it emits
    durable rows every interval. Both forms report elapsed execution time.

    Silence-triggered, not periodic: callers stamp ``touch()`` on every real
    write, so a rule that is actively logging or drawing a progress bar keeps
    resetting the clock and never reports — the status appears exactly when the
    console would otherwise be frozen, which is the "is it stuck?" case. A lone
    ``time.monotonic()`` float assignment is atomic under the GIL, so ``touch()``
    needs no lock.

    Set ``CST_HEARTBEAT_SECS`` (``0`` disables entirely) to override the
    interval without touching a Snakefile.
    """

    def __init__(self, label, stream, interval=15.0, on_stall=None):
        self._label = label
        self._stream = stream
        #: Called INSTEAD of printing the notice; a truthy return means the
        #: stall was already answered elsewhere on the console (see `_run`).
        self._on_stall = on_stall
        raw = os.environ.get("CST_HEARTBEAT_SECS")
        try:
            self._interval = float(raw) if raw is not None else float(interval)
        except ValueError:
            self._interval = float(interval)
        self._enabled = self._interval > 0
        isatty = getattr(stream, "isatty", None)
        self._interactive = bool(isatty and isatty())
        self._start = time.monotonic()
        self._wall_start = datetime.now()
        self._last = self._start
        #: Closed quiet periods as ``(monotonic_start, monotonic_end)``. Appended
        #: only by the watchdog thread and read only after ``stop()`` has joined
        #: it, so the list needs no lock.
        self._quiet = []
        #: Whether a silence status was ever shown. Written by the watchdog
        #: thread and read in ``stop()`` only after it has been joined, so it
        #: needs no lock -- the same argument ``_quiet`` above makes.
        self._noticed = False
        # The notice schedule, advanced by `_tick`; `_run` resets it on start.
        self._quiet_since = None
        self._next_notice = self._first_notice()
        self._seen = self._last
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
            f"... quiet for {format_elapsed(end - start)} "
            f"({self._wall_at(start)} -> {self._wall_at(end)})"
            for start, end in self._quiet
        ]

    def touch(self):
        self._last = time.monotonic()

    def _emit(self, text, code=None, *, redraw=False):
        # `None` rather than `_ANSI_BODY` as the default: a default argument is
        # evaluated when this class is DEFINED, and the colour constants are
        # declared further down the module (beside the console handler that owns
        # the scheme). Naming one here would be a NameError at import.
        #
        # `text` is the MESSAGE; the row is assembled here so every notice this
        # watchdog prints has the stamp and module column the lines around it
        # have. It used to print `   ... 2.03_fetch/<key>: still running, 2m00s
        # elapsed` -- no stamp, the log-parts spelling of the job, and a fourth
        # duration format -- between a RUN and a DONE line that agreed on all
        # three. The identity is the job's console spelling (`_heartbeat_identity`).
        row = _log_row_text(
            f"{datetime.now():%H:%M:%S}",
            "fetch" if self._label.startswith("2.03") else "execution",
            "INFO",
            f"{_heartbeat_identity(self._label)} {text}",
        )
        try:
            ending = "" if redraw else "\n"
            self._stream.write(
                _line_reset(self._stream)
                + _paint_body(
                    row + ending,
                    _console_colour(self._stream),
                    code or _ANSI_BODY,
                )
            )
            self._stream.flush()
        except Exception:
            pass  # console I/O must never break the job

    def _clear_status(self):
        """Erase a standing terminal frame without adding a console row."""
        if not (self._interactive and self._noticed):
            return
        try:
            self._stream.write(_line_reset(self._stream))
            self._stream.flush()
        except Exception:
            pass  # console I/O must never break the job

    def _first_notice(self):
        """Silence length at which a fresh silence is first reported."""
        return self._interval

    def _run(self):
        """Wake every interval and hand the current time to :meth:`_tick`."""
        # `_quiet_since` is the timestamp of the last real write BEFORE the
        # current silence, i.e. where the gap starts. It is carried across
        # ticks so one contiguous silence yields ONE recorded period no
        # matter how many notices it prints.
        self._quiet_since = None
        self._next_notice = self._first_notice()
        # The last `touch()` this loop has already accounted for. Resumption is
        # detected by this value CHANGING, not by catching a tick while
        # `now - last < interval`: the thread wakes every `interval` and the
        # gap it is measuring is also `interval`, so whether any tick lands
        # inside a short burst of output is down to alignment. It was a coin
        # flip before 2026-09-06. A missed reset delays the next silence report.
        self._seen = self._last
        # Wake every interval so output resuming resets the notice schedule.
        while not self._stop.wait(self._interval):
            self._tick(time.monotonic())
        if self._quiet_since is not None:
            # Still silent when the rule ended -- close the period at the stop,
            # not at `_last`, or the final and usually most interesting gap is
            # recorded as ending when the silence BEGAN.
            self._quiet.append((self._quiet_since, time.monotonic()))

    def _tick(self, now):
        """One watchdog step at monotonic time ``now``.

        Separate from the thread so the schedule can be driven with exact
        timestamps: tests call this directly instead of sleeping on a real
        clock and hoping the thread woke on time.
        """
        last = self._last
        if last != self._seen:
            # Output happened since the previous tick. `last` is when, so a
            # quiet period closes exactly where it did before.
            if self._quiet_since is not None:
                self._quiet.append((self._quiet_since, last))
                self._quiet_since = None
            self._next_notice = self._first_notice()
            self._seen = last
        silence = now - last
        if silence >= self._interval:
            if self._quiet_since is None:
                self._quiet_since = last
            if silence < self._next_notice:
                return  # still silent, not yet time to say so again
            # A rule that is drawing a progress bar answers the stall in the
            # bar's own line: the hook redraws it with the clock advanced,
            # which is the only fact the notice carries, and the notice's
            # row would otherwise land ON the line the bar occupies. The
            # silence is still REAL and is still recorded in `_quiet` below
            # -- only its console presentation changed, so `quiet_rows` is
            # unaffected. `_noticed` stays unset too: no watchdog frame was
            # opened here, so `stop()` has none to clear.
            #
            # The schedule does not advance here: it counts watchdog
            # presentations, and this branch prints none.
            if self._on_stall is not None and self._on_stall():
                return
            self._noticed = True
            self._emit(
                f"{format_elapsed(now - self._start)} elapsed",
                redraw=self._interactive,
            )
            self._next_notice += self._interval

    def start(self):
        if self._enabled:
            self._thread.start()
        return self

    def stop(self, failed=False):
        """Close the watchdog and report failures that have no DONE row."""
        if not self._enabled:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._clear_status()
        if not failed:
            return
        elapsed = format_elapsed(time.monotonic() - self._start)
        self._emit(f"failed after {elapsed}", _ANSI_FAIL)


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


def _pad_line_over(text, columns):
    """Pad ``text`` so it covers a progress frame it is about to overwrite.

    A frame is written as ``<bar>\\r``: the text lands, then the carriage return
    puts the cursor back at column 0 WITHOUT erasing anything. Whatever is
    written next therefore overwrites the frame from the left and leaves
    everything past its own end standing -- which is how a bar's summary row
    kept a stale ``eta 0:12`` hanging off it, and how an ordinary log row landed
    looking like it had been appended to the bar with no line break between them
    (both observed 2026-08-18, on WF4's batched Wflow runs).

    Padding goes before the trailing newline, and on the FIRST line of a
    multi-line chunk, because the frame occupies the line the cursor is on and
    not the one the text ends on.

    Returns ``(padded_text, columns_still_dirty)`` -- the second being what to
    pass back next time: the width now standing on the console line, or ``0``
    once a newline has moved past it.
    """
    head, sep, tail = text.partition("\n")
    if len(head) < columns:
        head = head.ljust(columns)
    return head + sep + tail, 0 if sep else len(head)


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

_REMOTE_READ_ECHO_RE = re.compile(
    r"^Reading \S+ from (?:\w+://|"
    + "|".join(re.escape(token) for _, token in _REMOTE_PREFIXES)
    + ")"
)

_TEE_CONSOLE_MUTED = (
    # Rule start/end rows already identify these per-scenario steps. Keep the
    # repeated file/parameter detail in the durable log.
    (
        "weathergen",
        re.compile(
            r"^\S+: (reading source realization|applying climate perturbations)"
        ),
    ),
    ("data_catalog", "Parsing data catalog from "),
    ("model", "Initializing wflow_sbm model from hydromt_wflow"),
    ("config", "Reading model config file from "),
    ("config", "Reading default config file from "),
    ("wflow_base", "Supported Wflow.jl version "),
    # Spelled WITHOUT hydromt's closing full stop: `_compact_log_line` drops it
    # before the row reaches this table, so a prefix that carried it matched
    # nothing (2026-09-05 -- three rows came back on the console the day the
    # stop went).
    ("tables", "Reading model table files"),
    ("tables", "No tables found, skip writing"),
    ("grid", "No grid data found, skip writing"),
    # `Write forcing file` announces the write hydromt is ABOUT to do; the next
    # row, `Writing file <path>`, states the same fact and names the target. The
    # pair does bracket a decision -- hydromt renames the file when one already
    # exists and overwriting is off -- but that case announces ITSELF at
    # WARNING and names both paths, so the opening row is never the anchor that
    # makes the rename legible. Muting it costs nothing and saves a row per
    # forcing write: once in a WF1 build, and once per run in WF4, where
    # every downscale rule writes one.
    ("forcing", "Write forcing file"),
    # Rule 1.09 drives `hydromt update` through its CLI at `-vv`, which is what
    # puts hydromt's rows in the rule's log at all -- and at that level the CLI
    # also announces its version (three times, once per logger it configures),
    # names each `setup_*` step it is about to run, and echoes every keyword
    # of that step as `setup_x.param=value`. The version is a property of the
    # environment, and the parameter echo is the build recipe the rule's own
    # `-i` file already holds; ~25 rows per forcing write that say nothing a
    # reader of the CONSOLE is waiting for. The log part keeps all of them.
    ("log", "HydroMT version: "),
    ("model", "update: "),
    ("model", "setup_"),
    # hydromt echoes the object store URI it is about to read. WF2 already
    # printed that URI one row earlier, from `fetch_gcm_raw`, in BOTH the
    # pinned and the globbed branch -- so this row is a duplicate that costs
    # 162-175 characters four times per run, the longest rows the workflow
    # produces. A pattern rather than a prefix because the entry name sits
    # between the two fixed parts.
    ("data_source", _REMOTE_READ_ECHO_RE),
    # WF2's own fetch rows that state a property of the RUN, or a fact the
    # artifact already carries. Same terms as everything above: INFO only, and
    # the row survives in every log part. The row naming WHICH slice a worker is
    # on -- `Fetching <resolved entry name>` -- is
    # deliberately NOT here: with several fetch jobs in flight (Snakemake under
    # `-c 3`, or `stage_cmip6.py` under four workers) it is the only thing that
    # attributes a twenty-minute wait to a source, and a stall nobody can
    # attribute is the defect these rows exist to prevent.
    #
    # `gcsfs extended-filesystem switch = 'false'` is the value this module
    # needs, reported once per fetch job -- one identical row per slice, and a
    # full staging run is 161 of them. The muted spelling PINS `'false'`:
    # `hns_switch_row` reports every other value at WARNING, which carries a
    # level field and so cannot match here, which is what keeps the row that
    # explains a 14x slowdown out of reach of a mute added for volume.
    ("fetch", "gcsfs extended-filesystem switch = 'false'"),
    # `store calendar=noleap (tas)` -- a property of the store, stamped on the
    # slice as `cst_calendar`, so the console copy is the transient one.
    ("fetch", "store calendar="),
    # `Wrote raw <key>.nc (0.07 MB, 780 steps)` -- the completion row. Under
    # Snakemake the rule's DONE line already says the job finished, and under
    # `stage_cmip6.py` the tool's own entry restates the size beside the
    # elapsed time; either way this row lands next to a second statement of
    # itself. The size and step count stay in the log part.
    ("fetch", "Wrote raw "),
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


def _write_console(stream, text):
    """Keep console glyphs safe on legacy code pages; leave UTF-8 log text intact."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        folded = text.translate(_ASCII_GLYPH_FALLBACK)
        stream.write(folded.encode(encoding, "replace").decode(encoding))


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
        #: Whether this tee is part-way through writing ONE logical row, i.e.
        #: the next console write is the rest of the line this one started.
        #: That is the only state in which the cursor may not be reset -- see
        #: `write`, and `_line_reset` for what is being reset and why the
        #: condition cannot be "is one of MY frames standing".
        self._mid_row = False
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
        # An empty replay from captured_output is not console activity.
        if not text:
            return 0
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
        shown = _shorten_console_paths(shown, self._project_root, self._tokens)
        if shown and not _muted_on_console(shown):
            # The CONSOLE copy only. `out` below feeds `_pending` and the log
            # file, and the module's standing contract is that no escape code
            # ever reaches `logs/` -- which is also why `shared.progress` may
            # not emit this itself: its one string goes to both sinks, while
            # here the two are already separate.
            # UNCONDITIONAL at the start of a row, like the two other writers on
            # this console (`console_style._ConsoleHandler.emit` and the
            # heartbeat). It was gated on "did I leave a frame standing" until
            # 2026-09-17, and that flag cannot see the case it most needs to:
            # a FANNED rule runs its members as separate PROCESSES (`-c 3` on
            # wf0's `extract_climate_datasets_{era5,chirps}`), so the tee
            # holding the bar and the tee writing the row are in different
            # interpreters and share no state. Only the cursor is common, which
            # is why `_line_reset` is an escape rather than a flag.
            #
            # Resetting a line nothing is standing on erases an empty line and
            # costs nothing. The one state that must NOT be reset is this tee
            # part-way through its own row: a library writing one line in two
            # calls would lose the first half.
            reset = "" if (self._mid_row or _redraw) else _line_reset(self._live)
            _write_console(self._live, reset + _paint_body(shown, self._colour))
            # A redraw frame leaves the cursor on the bar, which the next row is
            # expected to clear -- so it does not count as a row in progress.
            self._mid_row = not _redraw and not shown.endswith("\n")
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
            _write_console(
                self._live, _paint_body(self._held_frame, self._colour) + "\n"
            )
            self._held_frame = ""
        return self.write(text)

    def flush(self):
        # Flush the sinks but NOT ``_pending``: emitting a mid-progress fragment
        # would re-clutter the log with every partial redraw.
        self._live.flush()
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


class _NoFrameRelay:
    """Stand-in used when :mod:`blueearth_cst.shared.progress` cannot be
    imported. Refuses every line, so the tee behaves exactly as it did before
    the bar existed -- a progress bar must never be able to fail a run."""

    active = False

    def feed(self, line, stream=None):
        return None

    def tick(self):
        return None

    def close(self):
        return None


def run_and_tee(command, log_path, *, frame_relay_factory=None):
    """Run ``command`` (an argv list), streaming combined stdout+stderr to the
    console AND ``log_path``, and return the child's exit code.

    Replaces the ``<cmd> 2>&1 | tee {log}`` idiom in ``shell:`` rules. A bare
    ``| tee`` pipeline returns *tee*'s exit status, not the command's, unless
    bash ``pipefail`` is active -- and Snakemake injects no ``pipefail`` prefix
    on Windows/cmd.exe, so a failed ``hydromt``/``julia`` step is misread as
    success (t260721a; dev/tasks/). Teeing in-process restores exit-code
    fidelity while keeping live console output. The child runs with
    ``shell=False`` so argument quoting is preserved identically across cmd.exe
    and bash (a quoted ``julia -e "..."`` body stays one argv -- rules 1.13 and
    3.15 now pass a driver *file* instead, but other callers still rely on it).

    Wflow progress frames (``[cst-progress] <label> <fraction>``, emitted by
    ``shared/wflow_progress.jl``) are re-rendered here as the house progress bar;
    see :class:`~blueearth_cst.shared.progress.WflowFrameRelay`. While such a bar
    is open the silence watchdog redraws it instead of printing its own
    elapsed-time status (``_bar_tick``), and every console write is padded
    over whatever frame is standing on the line (``_pad_line_over``).

    A *pure* trailing run of benign interpreter-shutdown excepthook noise (see
    ``_EXCEPTHOOK_MARKERS``) is collapsed into a single summary line so it does
    not bury the real end of the log. The collapse is conservative: candidate
    lines are buffered, and any real content flushes them verbatim, so the
    filter only ever fires on a genuinely empty-bodied shutdown cascade.

    The weather template's routine dimensions row appears once per WF3
    invocation. Parallel jobs claim a marker beside the warning tally; each
    job's detailed log retains every row, including suppressed repetitions.

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
    template_seen = False
    tally = os.environ.get(_WARNING_TALLY_ENV)
    invocation = os.environ.get("CST_GENERATION_INVOCATION_ID", "")
    template_marker = (
        Path(tally).with_name(f".{invocation}.template-loaded")
        if tally and re.fullmatch(r"[0-9a-f]{32}", invocation)
        else None
    )
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

        # Columns of a progress frame currently standing on the console line, or
        # 0 when the cursor sits on a clean one. A frame ends in a carriage
        # return, which moves the cursor back without erasing, so every later
        # write has to cover it -- see `_pad_line_over`. ONE counter, because all
        # three writers (a streamed frame, the watchdog's tick, an ordinary row)
        # share the one console line.
        bar_line = {"columns": 0}

        def _console_frame(text):
            """Write one in-place frame, padded over what it overwrites."""
            body = _cr_overwrite(text)
            _console_write(body.ljust(bar_line["columns"]) + "\r")
            bar_line["columns"] = len(body)

        def emit(text, redraw=False, had_cr=False):
            nonlocal template_seen
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
            muted = _muted_on_console(text)
            if re.fullmatch(
                r"\d{2}:\d{2}:\d{2} - io - Template loaded: nx=[\d,]+, ny=[\d,]+, nt=[\d,]+\n?",
                text,
            ):
                muted = template_seen
                template_seen = True
                if not muted and template_marker is not None:
                    try:
                        # Exclusive creation coordinates parallel job processes.
                        with template_marker.open("x", encoding="utf-8"):
                            pass
                    except FileExistsError:
                        muted = True
                    except OSError:
                        # An unavailable marker still allows one notice per job.
                        muted = False
            if not muted:
                # Body tier on the console only: a shell rule's output is
                # detail, and `log` below must stay free of escape codes.
                #
                # Padded for the CONSOLE only, and painted afterwards:
                # `log.write` below takes the original `text`, so a bar's
                # summary row does not reach the file with trailing spaces on
                # it, and the padding is measured on the text rather than on the
                # escape codes `_paint_body` wraps around it.
                console_text = _shorten_console_paths(text, project_root, tokens)
                console_text, columns = _pad_line_over(
                    console_text, bar_line["columns"]
                )
                shown = _paint_body(console_text, colour)
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
                    # Only where the row was actually WRITTEN: a row the console
                    # dropped leaves whatever frame is standing exactly where it
                    # was, so the next writer still has to cover it.
                    bar_line["columns"] = columns
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
        # lines. A WF4 experiment runs the model 20 times, and the result was
        # ~2000 of the log's 4000 rows being frames of a bar that is meant to
        # occupy ONE. Preserving `\r` lets `_cr_overwrite` do to a Julia bar
        # exactly what it already did to a Python one.
        stream = io.TextIOWrapper(
            proc.stdout, encoding="utf-8", errors="replace", newline=""
        )
        # Wflow reports its timestep progress as a bare fraction (see
        # `shared/wflow_progress.jl`); this turns each one into a frame of the
        # same bar wf0's and wf2's long writes animate. Built BEFORE the
        # watchdog because the watchdog consults it -- see `_bar_tick`.
        wflow_bar = (frame_relay_factory or _NoFrameRelay)()

        def _bar_tick():
            """Answer a stall by redrawing the open bar; False if there is none.

            The watchdog's elapsed-time status and a live bar carry
            the same fact, and printing the first onto the second's line is what
            leaves a row with a frame's tail hanging off it. So while a bar is
            open the stall is answered IN the bar, and the notice is kept for
            the rules that have no bar to draw -- 3.06 (weathergenr), 3.12 and
            3.14, where silence really is the only thing there is to report.

            False off a terminal as well: there the frames are not streamed at
            all (`stream_frames`), so a redraw would print nothing and the
            notice is the only liveness signal a captured log gets.

            Called on the watchdog THREAD while the main one owns the console.
            It is not a race in practice -- the watchdog fires only after a full
            interval in which the main thread wrote nothing, which is to say
            while it is blocked reading the child's pipe -- and the failure mode
            if it ever were is a stray frame that the next padded write covers,
            never a corrupted log: this path is console-only.
            """
            if not stream_frames:
                return False
            frame = wflow_bar.tick()
            if frame is None:
                return False
            _console_frame(frame)
            return True

        # Silence watchdog: prints an elapsed-time notice to the console (stderr,
        # never the log) if the child goes quiet — so a hung Julia/Wflow/hydromt
        # step is visible live. Touched on every line read from the child.
        heartbeat = _Heartbeat(label, sys.stderr, on_stall=_bar_tick).start()
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
                # A Wflow progress frame is REWRITTEN into a frame of the house
                # bar, then falls through to the ordinary carriage-return path
                # below -- which already streams frames to the console and
                # collapses them to one row in the log. Rendering here and
                # writing there keeps one implementation of each job; see
                # `WflowFrameRelay` for why the child reports only a number.
                rendered = wflow_bar.feed(raw, stream=sys.stdout)
                if rendered is not None:
                    # An empty render means the relay recognised the frame and
                    # chose not to draw it (the duplicate final frame). Dropping
                    # it here is what keeps the raw `[cst-progress]` sentinel out
                    # of the log -- `None`, by contrast, means "not mine", and
                    # that line must pass through untouched.
                    if not rendered:
                        continue
                    if not stream_frames and rendered.endswith("\n"):
                        # Off a terminal no frame reached the console, so the
                        # finished bar goes out as an ordinary row: it is a
                        # batch member's only record there (it carries
                        # `[k/N]`), and dropping it like a redraw would leave a
                        # CI log with no per-member line at all.
                        frames = []
                        deliver(rendered, redraw=False)
                        continue
                    raw = rendered
                if raw.endswith("\r"):
                    if stream_frames:
                        _console_frame(raw)
                    frames.append(raw)
                    continue
                line, frames = "".join(frames) + raw, []
                deliver(line, redraw=stream_frames and "\r" in line)
            # A child that died mid-bar leaves its last frame unterminated, so
            # close the row before anything else is written to it.
            trailing = wflow_bar.close()
            if frames:
                chunk = "".join(frames)
                if trailing is not None:
                    # The terminator REPLACES the dangling carriage return
                    # rather than following it: `_cr_overwrite` keeps the last
                    # non-empty `\r`-separated segment, so an appended newline
                    # would become that segment and blank the bar it closes.
                    chunk = chunk.rstrip("\r") + trailing
                deliver(chunk, redraw=stream_frames)
            elif trailing is not None and stream_frames and bar_line["columns"]:
                # Only when a frame is still standing. The bar's line may have
                # been closed already -- by its own summary, or by an ordinary
                # row that overwrote it -- and a newline written onto a clean
                # line is a blank one, which reads as output that went missing.
                # The LOG is unaffected either way: this branch is console-only.
                _console_write(trailing)
                bar_line["columns"] = 0
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
def tee_to_log(log_path, heartbeat_interval=15.0):
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
        Seconds of silence before the console timer updates (default 15).
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


def plural(count, singular, plural_form=None):
    """``3 areas`` / ``1 area`` -- a count and its noun, agreeing.

    Replaces ``f"{n} area(s)"``, which was the toolbox's habit in ~50 emitted
    rows. The parenthetical hedges a question the line has already answered:
    the count deciding the plural is right there, and on the rows where it is
    ``1`` -- which is most of them on a small basin -- the row reads as a
    defect in the code rather than as a fact about the run.

    An irregular plural is passed rather than derived: ``plural(n, "reach",
    "reaches")``. Deriving it would mean an English rule table for the four
    nouns in this package that need one.

    ``count`` is not required to be an integer. One caller formats a value
    read from a report that may be ``"?"``, and a helper that raised on it
    would turn a cosmetic row into a failure; anything that is not ``1``
    pluralizes, which is the right answer for ``"?"``.
    """
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


_WARNING_TALLY_ENV = "CST_WARNING_TALLY"
RUN_WARNING_TALLY_ENV = "CST_RUN_WARNING_TALLY"


def note_warning(module="cst"):
    """Record that one warning row reached the console. Best effort.

    One short append per row. ``a`` seeks to the end on every write, so the
    parallel jobs this toolbox runs interleave rows rather than overwriting
    each other's -- a line count is all this file is ever read for.
    """
    # The runner tally survives resets between WF3 source/generation phases.
    # Each workflow keeps its existing tally and verdict independently.
    paths = {os.environ.get(key) for key in (_WARNING_TALLY_ENV, RUN_WARNING_TALLY_ENV)}
    for path in paths - {None, ""}:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"{module}\n")
        except OSError:
            pass


def log_row(message, module="cst", level="INFO"):
    """Print one log row in the standard compact format used across rule logs.

    **What a row SAYS.** Six conventions, arrived at by reading every emitted
    row in the package against a rendered transcript of all five workflows
    (2026-09-17). They are here, on the function every author calls, rather
    than in a document nobody opens while writing a row:

    1. **A row states its subsystem once.** It must not open with a noun that
       restates its own ``module`` column: ``collection - Collection <id>: ...``
       says it twice. This is the rule :func:`_compact_log_line` already
       applies to hydromt's records; we cleaned someone else's output of it
       and exempted our own until this was written down. The boundary is that
       rule's own: it strips a LABEL, not a verb, so ``fetch - Fetching ...``
       stays -- without the verb the row is a bare identifier that never says
       what is happening to it.
    2. **A value never repeats the key that introduces it.**
       ``reference_window reference_window_start=... reference_window_years=...``
       said one word five times; ``reference_window start=... years=...`` says it
       once. Strip the prefix at the ROW, never in the record it came from --
       a durable record wants self-describing keys.
    3. **A count and its noun agree**, through :func:`plural`. Never
       ``{n} thing(s)``: the count deciding the plural is on the same line, and
       on the rows where it is 1 the hedge reads as a defect in the code.
       The fix drags agreement with it -- ``are`` becomes ``is``, ``They were``
       becomes ``It was`` -- so it is never a search-and-replace.
    4. **A warning states the FAULT first**, then what it means for this run.
       Not what the output artifacts will show ("and the figures say so" is the
       figures' business), and not a count when the count is not the news: a
       row reporting that two subsystems disagree opened with how many
       locations were affected and buried the disagreement in its third clause.
    5. **No definite article on a verb row**, and no clause that cannot be
       false. ``Reading monthly change-factor table``, not ``Reading the ...``;
       ``Model reference matches the live model``, without ``simulation may
       proceed`` -- a mismatch raises.
    6. **A row opens with a capital unless it opens with an identifier.** The
       rows that opened lowercase were written as continuation fragments of a
       multi-part f-string and are one line on the console. A source id
       (``chirps is precipitation-only: ...``) is exempt: it is a literal key
       from the config and the catalog, and capitalising it prints a name that
       does not exist.

    One more, about LENGTH: a row is one console line and is never wrapped --
    wrapping splits what ``grep`` is meant to find on one line in the rule's
    log part, so the answer to a long row is a shorter row. Where the length is
    the data's rather than the author's, cap the list with :func:`listed`,
    which always states what it dropped.

    ``HH:MM:SS - <module> - <message>``, with the level shown only when it is
    not INFO (:func:`_log_row_text`) — the same shape ``_compact_log_line``
    produces for hydromt records, so a ``script:`` rule's own messages sit
    uniformly among the hydromt/library lines rather than as bare,
    timestamp-less text. Use this instead of a plain ``print`` for anything
    meant to appear in a rule log. The row is already compact, so the tee passes
    it through (only any project paths in it are relativized).

    A row at ``WARNING`` or above also increments the run's warning tally
    (:func:`note_warning`), which is what puts the count on the verdict line.
    Counted after the ``CST_LOG_LEVEL`` floor, so the tally reports warnings
    the reader was actually shown.

    Rows below ``CST_LOG_LEVEL`` are dropped, which is the toolbox's quiet mode:
    ``CST_LOG_LEVEL=WARNING`` leaves only warnings and errors. Two properties
    make that safe to add to an existing caller population:

    * **Unset means emit everything**, so nothing changes until someone opts in.
    * **An unrecognized ``level`` is never suppressed.** A caller passing a
      level this table does not know keeps printing — a filter that silently
      swallowed an unfamiliar level would hide exactly the unusual row worth
      seeing.

    Still ``sys.stdout`` rather than ``logging``, deliberately: the tee captures
    ``sys.stdout``, so a row that went through a logging handler would bypass
    the mechanism that puts it in the rule's log file. The floor therefore
    belongs INSIDE this function, not in a migration to ``logging``.

    **One ``write`` per row, not ``print``.** ``print`` emits the text and the
    newline as two calls, and :func:`_muted_on_console` refuses any chunk that
    is not exactly one newline-terminated line — deliberately, since a tee is
    handed chunks rather than lines. So a row emitted through ``print`` can
    never match ``_TEE_CONSOLE_MUTED``, and until this changed the mute table
    reached hydromt's records (``StreamHandler.emit`` writes ``msg +
    terminator`` in one call) but not a single row of our own. The visible
    output is identical either way; what changes is that our rows are now
    mutable on the same terms as everyone else's. Same defect and same fix as
    ``add_climate_forcing._run_streaming`` in the wf1-wf3 console lean.
    """
    rank = _LOG_LEVEL_RANK.get(str(level).strip().upper())
    if rank is not None and rank < log_level_floor():
        return
    # Counted AFTER the floor and before the write, so the tally matches what
    # the console actually showed: a row suppressed by `CST_LOG_LEVEL` is not a
    # warning the reader was given and must not appear in the verdict's count.
    if rank is not None and rank >= _LOG_LEVEL_RANK["WARNING"]:
        note_warning(module)
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
    # Painted here only when stdout IS the terminal: a `run:` rule writes from
    # Snakemake's own process, with no tee to dim the stamp and module column.
    # A tee reports `isatty() == False`, so a teed row is painted once, by it.
    _write_console(
        sys.stdout,
        _paint_body(
            _log_row_text(f"{datetime.now():%H:%M:%S}", module, level, message) + "\n",
            _console_colour(sys.stdout),
        ),
    )
    # Flushed, because off a terminal Python block-buffers stdout: under a
    # redirect or in CI a rule's rows then all arrived AFTER its DONE line,
    # which Snakemake writes from the parent process. On a terminal the stream
    # is line-buffered and this is a no-op. Guarded, since the tee and the
    # test doubles standing in for stdout do not all offer `flush`.
    flush = getattr(sys.stdout, "flush", None)
    if flush is not None:
        flush()


_FIGURE_BUNDLES = {}

_FIGURE_BUNDLE_MODULES = {}

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
                log_row(
                    f"Figure written → {os.path.join(directory, names[0])}",
                    module=module,
                )
            else:
                log_row(f"Wrote {len(names)} figures → {directory}", module=module)
    finally:
        _FIGURE_BUNDLE_FLUSHING = False


_ANSI_BODY = None  # the terminal's OWN foreground -- no SGR at all

_ANSI_DIM = "38;5;243"  # dim grey -- the plan block, and a row's scaffolding

_ANSI_FAIL = "91"  # bright red

_ANSI_ALERT = "38;5;208"  # orange

_ANSI_RESET = "\033[0m"

_SEVERITY_PATTERNS = (
    (
        re.compile(
            r"\b(?:ERROR|ERRORS|FAIL|FAILURE|FAILED|CRITICAL|FATAL)\b"
            r"|Traceback \(most recent call last\)"
            r"|\bError in\b|\bError:"
        ),
        _ANSI_FAIL,
    ),
    (
        re.compile(r"\b(?:WARNING|WARN)\b|\bWarning message\b|\b\w*Warning:"),
        _ANSI_ALERT,
    ),
)

_DEMOTED_WARNINGS = (("forcing", "Write forcing skipped: dataset is empty"),)


def _demoted_warning(line):
    """Whether ``line`` is a WARNING row on the enumerated demotion list."""
    fields = line.strip().split(" - ", 3)
    if len(fields) != 4 or fields[2] != "WARNING":
        return False
    _stamp, module, _level, message = fields
    return any(
        module == demoted_module and message.startswith(prefix)
        for demoted_module, prefix in _DEMOTED_WARNINGS
    )


def _severity_code(line):
    """The SGR code a line's own text demands, or ``None`` for the caller's.

    A row on ``_DEMOTED_WARNINGS`` answers ``None`` before the patterns are
    consulted, so it takes the caller's tier -- body, on every console path.
    """
    if _demoted_warning(line):
        return None
    for pattern, code in _SEVERITY_PATTERNS:
        if pattern.search(line):
            return code
    return None


def _ansi(text, code):
    """Wrap ``text`` in an SGR code. Callers decide WHETHER to colour."""
    return f"\033[{code}m{text}{_ANSI_RESET}"


def _line_reset(stream):
    """The escape that puts ``stream``'s cursor on a CLEAN line, or ``""``.

    A progress frame is left STANDING on the console line between redraws: the
    bar in ``shared.progress`` writes ``\\r<frame>`` and stops there, so the
    cursor sits at the frame's end with the frame still visible. Any writer that
    then starts a new logical line appends to it -- ``13:18:23 - DONE Rule 0.01``
    landing on the tail of an ``era5 store`` bar with no line break between them
    (observed 2026-09-03, on a ``-c 3`` WF0 run). This is the same defect
    :func:`_pad_line_over` fixes for ``run_and_tee``; that path never reached the
    writers here, which is why the convention has to be restated as an escape.

    ``\\r`` returns to column 0 and ``\\x1b[2K`` erases the line, so the caller's
    text lands on a clean one whatever was standing and whoever drew it. That
    matters because the writers are in DIFFERENT PROCESSES -- the bar in a rule's
    job, the finish line in Snakemake's parent -- so no in-process flag can
    coordinate them and only cursor state can.

    Gated on ``isatty`` ALONE, deliberately, and not on :func:`_console_colour`:
    ``NO_COLOR`` asks for no colour, not for an unmanaged cursor, and folding the
    two together would hand anyone who sets it the bug back. Off a terminal the
    escape would be literal text in a captured run artifact, so there it is
    ``""`` -- and off a terminal nothing overwrites anything, so no frame is ever
    left standing to clear.
    """
    isatty = getattr(stream, "isatty", None)
    return "\r\033[2K" if bool(isatty and isatty()) else ""


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

    **A line's own severity outranks ``code``.** ``_severity_code`` is consulted
    per line, so a ``WARNING`` or ``ERROR`` arriving as ordinary body text --
    from our scripts, or from hydromt / R / Julia / wflow stdout -- is painted
    for what it says rather than for the tier its emitter assumed. Applied here
    because this is the ONE funnel every console-bound chunk passes through:
    the tee, the heartbeat and the console handler all call it, and none of them
    can see inside the text they forward. Nothing here reaches a log file --
    ``_paint_body`` returns ``text`` unchanged when ``colour`` is false, and the
    log branch never asks for colour -- so the file stays free of escapes.

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
        _paint_line(line, code) if line.strip() else line for line in text.split("\n")
    )


_ROW_PREFIX_RE = re.compile(r"^(\d\d:\d\d:\d\d - [a-z_][a-z0-9_]* - )(.*)$")


def _paint_line(line, code):
    """Paint one line: its severity if it has one, else its tier by field.

    **Severity outranks everything, and is painted WHOLE-LINE.** A warning or
    an error arriving as body text -- from our scripts, or from hydromt, R,
    Julia or wflow stdout -- is painted for what it says rather than for the
    tier its emitter assumed, stamp and module column included. Dimming half of
    such a row would say it is partly routine. A structured warning omits its
    WARNING field after severity detection; orange carries it on this console.

    Otherwise a row in our own grammar has its stamp and module column dimmed
    and its message left in the body tier, so a run's messages stand clear of
    the scaffolding that repeats on every line. Anything else takes the tier
    whole, which is what every non-row line already did.
    """
    severity = _severity_code(line)
    if severity:
        # Detect severity before removing its field from the coloured console.
        # Keep durable rows and unstructured warning text intact.
        match = _ROW_PREFIX_RE.match(line)
        if match and match.group(2).startswith("WARNING - "):
            line = match.group(1) + match.group(2).removeprefix("WARNING - ")
        return _ansi(line, severity)
    # Splitting a line into fields is a property of the BODY tier and of
    # nothing else. A caller that passes an explicit tier is saying this whole
    # line is not routine -- an explicitly coded failure is the heartbeat case
    # now that routine silence frames use the body tier.
    if code:
        return _ansi(line, code)
    match = _ROW_PREFIX_RE.match(line)
    if not match:
        return line
    prefix, message = match.groups()
    return _ansi(prefix, _ANSI_DIM) + message


def rule_id(number):
    """Return a rule number in the console's spelling: ``Rule 1.07:``.

    One definition, because three places have to agree on it: the ``message:``
    banner a rule declares, the START line the console handler renders from
    that banner, and the FINISH line it builds from a record that carries only
    a rule name. Two spellings of one identity on adjacent lines is how a pair
    stops reading as one job.
    """
    return f"Rule {number}:"


def format_elapsed(seconds):
    """``h:mm:ss`` for a duration, matching the benchmark tables' own column."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
