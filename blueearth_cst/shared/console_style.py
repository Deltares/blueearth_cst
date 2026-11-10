"""Snakemake's own terminal output, in this toolbox's grammar.

Split out of `shared/snake_utils.py` on 2026-09-17 (t2609171028). The reason is
not tidiness -- it is that a console edit was invalidating generated scientific
artifacts.

`experiment/content_identity.repository_code_inventory` fingerprints a
TRANSITIVE STATIC-IMPORT CLOSURE from each stage's declared entry paths, and
hashes every file in it. WF3 generation now imports its scientific helpers
from `wf3_science` and its neutral tee helpers from `run_log_core`. The mixed
`snake_utils` module remains a compatibility surface for the other workflows
and owns Wflow-specific relay injection; it is outside the WF3 closure.

Nothing in this module was ever one of those nineteen. Measured before the
split: every name here is imported by ZERO closure modules -- they are reached
only from Snakefiles, which are not entry paths. They were inside the
fingerprint purely by sharing a 5,768-line file with helpers that earned their
place there. So renaming a banner invalidated a scenario collection, and a
rapid-fixture run cost ~12 minutes to recover. Twice in one session, for changes
that decide which characters reach a terminal.

**The dependency runs one way: this module imports from `snake_utils`, never the
reverse.** A convenience re-export in `snake_utils` would pull this file into
its import closures. `tests/test_module_import_direction.py` pins that direction.

The tee tier (`_Tee`, `tee_to_log`, `run_and_tee`, `log_row`, `_paint_body`,
`_compact_log_line` and friends) lives in `run_log_core`. It is genuinely
reachable from WF3's `script:` modules and remains in their code inventory.
This module holds Snakemake's parse-time console presentation.

`warn_row` lives here rather than beside `log_row`, which is a cohesion cost
paid deliberately: it is the PARSE-TIME counterpart, called from a Snakefile's
top-of-file checks before Snakemake's logging stack exists, and it is imported
by no closure module. Keeping it next to `log_row` would keep it in the
fingerprint for no benefit.

Several names imported below are private compatibility exports of
`snake_utils`. The WF3 scientific and tee modules do not import this module.
"""

import atexit
import logging
import os
import re
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from blueearth_cst.shared.command_display import sanitize_argv
from blueearth_cst.shared.run_log_core import (
    _CONSOLE_PATH_TOKENS_ENV,
    _declared_tokens,
    _shorten_console_paths,
)
from blueearth_cst.shared.snake_utils import (
    _ANSI_BODY,
    _ANSI_FAIL,
    _PROJECT_DIR_EXEMPT_NAMES,
    _PROJECT_ROOT_ENV,
    _ansi,
    _console_colour,
    _folder_rows,
    _line_reset,
    _log_row_text,
    _paint_body,
    _path_tokens,
    _relativize_paths,
    _tokenize_prefix,
    format_elapsed,
    note_warning,
    plural,
    rule_id,
)


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

    warn_row(
        f"project_dir is inside the repo tree ({rel.as_posix()!r}); "
        f"write run artifacts outside the source",
        module="config",
    )
    return True


def warn_row(message, module="cst"):
    """Print one WARNING row to stderr, in :func:`log_row`'s grammar and colour.

    The PARSE-TIME counterpart of `log_row`. A Snakefile's top-of-file checks
    run before Snakemake builds its logging stack and before any rule opens a
    tee, so neither :func:`install_console_style` nor :class:`_Tee` is there to
    style what they print -- and each site had invented its own spelling:
    ``warnings.warn`` (which prepends ``<file>:<line>: UserWarning:`` and echoes
    the source line under it, three lines for one sentence), a bare ``print``
    with a hand-written ``WARNING <workflow>:`` prefix, and a ``log_row`` on
    stdout. Four spellings of one thing, none of them the colour the same
    warning gets once a rule is running.

    ``HH:MM:SS - <module> - WARNING - <message>``, where ``module`` names the
    component that noticed (``config``, ``reference_window``) exactly as an
    in-run row names ``states`` or ``data_source``. **The colour is not chosen
    here**: :func:`_paint_body` reads the severity out of the row's own text, so
    this is painted by the same rule that paints a warning arriving from
    hydromt -- one funnel, not a second scheme that has to be kept in step.

    stderr rather than stdout, because a parse-time row belongs to the RUN's
    console and has no rule log to land in; that also keeps it out of a piped
    ``--dry-run`` DAG, which callers redirect.

    **From a Snakefile, keep ``module=...`` off the start of a line.**
    ``module`` is a Snakemake KEYWORD, and its parser reads a line opening with
    ``module=`` as the ``module`` directive -- ``SyntaxError: Expected name or
    colon after module keyword``, raised at parse time before any rule exists.
    A single-line call is fine (the token is then mid-expression); a call broken
    across lines is not. Build the message into a local first and pass it in one
    line, as ``generate_scenarios.smk`` does. The parameter keeps the name anyway,
    because :func:`log_row` has always spelled it that way and two names for one
    field costs more at every call site than this costs at one.
    """
    note_warning(module)
    text = _log_row_text(f"{datetime.now():%H:%M:%S}", module, "WARNING", str(message))
    sys.stderr.write(_paint_body(text + "\n", _console_colour(sys.stderr)))


#: Parse-time WARNING rows held for the run header, oldest first. Popped by
#: :func:`_drain_deferred_warnings` -- see :func:`defer_warning`.
_DEFERRED_WARNINGS = []
_HEADER_PRINTED = False


def defer_warning(message, module="cst"):
    """Hold a parse-time WARNING until the run header has been written.

    :func:`warn_row`'s grammar at the right MOMENT. A Snakefile's top-of-file
    checks run while Snakemake is still PARSING the workflow -- before
    ``Building DAG of jobs...``, and long before ``onstart:`` fires and
    :func:`install_console_style` and :func:`open_run_header` run. So a row
    written from there lands in the middle of Snakemake's preamble, above a
    title band that does not exist yet. No change to how it prints can move it;
    only deferral can.

    Queued here, the row is printed under the ``-- RUN ---`` block, where this
    toolbox's own output begins. The flush is the last thing the opening block
    does, in both the styled and the unstyled path, so it is not something the
    next workflow to need it has to remember.

    The stamp is taken NOW rather than at flush. It says when the workflow
    NOTICED, which is the honest reading and the only one that stays true when
    the DAG takes a while to build.

    **A row whose run never reaches a header is printed OUT OF POSITION, never
    dropped.** Two runs never get there: a DAG that fails to build, and
    ``--dry-run``, which does not fire ``onstart:`` at all -- and a dry run is
    exactly where "which combinations did not resolve" is what the reader came
    for. :func:`_flush_undrained_warnings` catches both at interpreter exit, so
    the worst case is a row at the bottom of the output rather than a row
    nobody sees. Position is what this function is for; silence is not a
    trade it is allowed to make.

    ``module`` follows :func:`warn_row`'s rules, including the one about never
    opening a Snakefile line with the ``module=`` token.
    """
    # Counted on DEFERRAL, not on flush: a row held for a header its run never
    # reaches is still printed (`_flush_undrained_warnings`), so a tally keyed
    # to the flush would drop the rows of exactly the runs that went wrong.
    note_warning(module)
    _DEFERRED_WARNINGS.append((f"{datetime.now():%H:%M:%S}", str(module), str(message)))
    if _HEADER_PRINTED:
        sys.stderr.write(
            "\n".join(_drain_deferred_warnings(_console_colour(sys.stderr))) + "\n"
        )
        sys.stderr.flush()


def defer_rows(text):
    """Hold already-printed WARNING rows for the run header, like :func:`defer_warning`.

    For output captured at parse time (``captured_output(replay=defer_rows)``):
    a ``HH:MM:SS - <module> - WARNING - <message>`` row keeps its stamp and is
    queued; anything else is written to stderr now, since only a warning row
    has a place under the header. Not re-counted: the ``log_row`` that printed
    the row already noted it in the warning tally.
    """
    for line in text.splitlines():
        fields = line.split(" - ", 3)
        if len(fields) == 4 and fields[2] == "WARNING":
            _DEFERRED_WARNINGS.append((fields[0], fields[1], fields[3]))
        elif line.strip():
            sys.stderr.write(line + "\n")
    if _HEADER_PRINTED:
        held = _drain_deferred_warnings(_console_colour(sys.stderr))
        if held:
            sys.stderr.write("\n".join(held) + "\n")
    sys.stderr.flush()


def _drain_deferred_warnings(colour):
    """Pop every held row, painted, as lines ready to join with newlines.

    POPS rather than reads: the opening block is assembled on whichever record
    arrives first and the no-header path runs a second builder, so a queue that
    survived its own flush would print the same row twice.

    Painted through :func:`_paint_body`, not through the handler's tier map: a
    WARNING is coloured by what it SAYS, by the same funnel that paints a
    warning arriving from hydromt. Running these through the block's body tier
    would make them the one warning on the console that reads as routine.
    """
    if not _DEFERRED_WARNINGS:
        return []
    held = list(_DEFERRED_WARNINGS)
    del _DEFERRED_WARNINGS[:]
    return [
        _paint_body(
            message
            if message.startswith("Reference period is short:")
            else _log_row_text(hms, module, "WARNING", message),
            colour,
        )
        for hms, module, message in held
    ]


@atexit.register
def _flush_undrained_warnings():
    """Print anything the run header never reached, at the bottom.

    The two runs that never reach a header: a DAG that fails to build, and
    ``--dry-run``, which never fires ``onstart:`` -- measured 2026-09-17, a dry
    run shows Snakemake's own ``Job stats:`` and nothing of this module.
    ``--dry-run`` is in AGENTS.md's Key Commands as the check to run after
    editing a rule, and the resolution report is one of the things its reader
    is looking for, so deferral must not be the reason it vanishes.

    Out of position, and that is the whole point: the ordered flush sites have
    already popped the queue on any run that reached them, so this fires only
    when the alternative is silence. It writes nothing on a healthy run.

    Guarded end to end -- at interpreter shutdown ``sys.stderr`` may already be
    closed, and a warning row is never worth a traceback on the way out.
    """
    try:
        if _RUN_HEADER is not None and not _HEADER_PRINTED:
            workflow, project_dir, config_path, details = _RUN_HEADER
            sys.stderr.write(
                "\n" + run_header(workflow, project_dir, config_path, **details) + "\n"
            )
        held = _drain_deferred_warnings(_console_colour(sys.stderr))
        if held:
            sys.stderr.write("\n".join(held) + "\n")
    except Exception:  # noqa: BLE001 -- never raise from an exit hook
        pass


# Colour by WHAT KIND OF LINE it is, not by which field it is -- three tiers,
# whole lines, so a run scrolling past reads as structure rather than text.
#
# * ``_ANSI_RUN`` (blue) -- a job STARTED. The line that says what you are now
#   waiting for, so it is the one the eye goes to.
# * ``_ANSI_DONE`` (green) -- a job FINISHED. Green because that is what green
#   means everywhere else a machine reports on work, and the pair then reads
#   without a legend: blue is in flight, green is behind you.
# * ``_ANSI_BODY`` (light grey) -- everything in between: a rule's own output,
#   the heartbeat's status frame, and Snakemake's informational lines. This is
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
# Two more sit OUTSIDE the routine three, for the lines that report that
# something went wrong or might have. They are deliberately not a fourth and
# fifth tier: the three above classify every routine line, while these mark the
# exceptional ones.
#
# * ``_ANSI_FAIL`` (red) -- every failure: ``run_summary``'s and run_workflows'
#   FAILED verdicts, a FAILED outcome row, the heartbeat's ``failed after``, and
#   any row ``_SEVERITY_PATTERNS`` reads as an error.
#   Snakemake's own error block is already red, and the verdict is the line a
#   reader scrolls to the bottom to find; in the routine three it was grey.
#   The SUCCESS verdict stays uncoloured on purpose. Colouring both would make
#   the pair a status field to be read, when the whole value here is that a
#   failed run looks different from every other run without being read.
# * ``_ANSI_ALERT`` (orange) -- every warning row. There is no yellow: a
#   failure is red and a warning orange, whoever prints it.
_ANSI_RUN = "94"  # bright blue


_ANSI_DONE = "92"  # bright green


#: What an up-to-date run prints in place of Snakemake's "Nothing to be done".
_UP_TO_DATE_LINE = "Up to date"


_ANSI_DIM = "38;5;243"  # dim grey -- the plan block's up-to-date rows


_ANSI_TITLE = "1"  # bold -- the workflow title, opening and closing


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


# Steps a workflow completes before Snakemake builds its DAG -- WF3's planning in
# scripts/generate_scenarios.py. They keep a rule number so the plan table reads
# as the whole pipeline, but they are not Snakemake jobs. name -> number, filled
# by `pre_dag_step`.
_PRE_DAG_STEPS = {}

# The plan head's tally label for pre-DAG steps. Their table cell is empty,
# like an up-to-date rule's: the head states the count once.
_PRE_DAG_LABEL = "completed before DAG execution"

# Rules whose jobs Snakemake adds only after a checkpoint resolves, so they are
# absent from the opening job table without being up to date. name -> the job
# count the rule plans once the checkpoint resolves (``planned_jobs``), or None
# when it declared none; filled by `rule_banner`'s `after_checkpoint`. The plan
# block shows the planned count as that rule's jobs, and an empty cell when
# there is none to show.
_AFTER_CHECKPOINT_RULES = {}

# Checkpoint-gated rule name -> the rule whose job count in this run's table
# replaces its declared ``planned_jobs``; filled by `rule_banner`'s
# `planned_from`. WF4 declares one 4.08 job per indicator set, but adding a set
# beside a current one runs one 4.07 plan, hence one 4.08 publication.
_PLANNED_FROM = {}


# Rule names belonging to a dynamic DAG whose progress denominator can change
# after checkpoint expansion. A handler activates this mode only after seeing
# one of these jobs, so registries left by another parsed Snakefile cannot
# affect the current workflow.
_DYNAMIC_PROGRESS_RULES = set()


@dataclass(frozen=True)
class RuleIdentity:
    """One rule's identity, so its label is written once instead of four times.

    A rule spells the same label in four places -- ``message:`` via
    :func:`rule_banner`, the ``log:`` path, the ``benchmark:`` path, and its
    entry in ``LOG_RULES`` -- and a fan-out rule spells it a fifth time in
    ``name:``. Keeping them in step by hand is how stale and unlisted labels
    kept appearing (`[R10-8]`, and three rules added without their label).

    Built through :class:`RuleRegistry`, never directly: the registry is what
    puts the label in ``LOG_RULES``, and an identity created outside one would
    write parts nothing merges.

    **The number lives in the VALUE, not in the constant's name.** Call the
    constant after the rule (``DELINEATE_REGION``), so a RENUMBER is a one-line
    edit. That is the direction history argues for -- renumbers are the routine
    event here and rule renames are rare -- and naming the constant
    ``RULE_0_02`` would invert it.

    ``part`` is the FAN-OUT argument, and it is a parse-time value, not a
    wildcard. WF0 generates one rule per candidate source in a Python loop, so
    the source is baked into the rule's name and into a per-label directory of
    parts. That is a different mechanism from :func:`rule_banner`'s ``context``,
    which resolves ``{wildcards.*}`` per JOB for a single wildcard rule; pass
    ``context`` through for that case and leave ``part`` unset.
    """

    number: str
    name: str
    log_parts_dir: str
    benchmark_parts_dir: str
    summary: str | None = None
    logged: bool = True
    #: Passed through to :func:`rule_banner` unchanged; see its arguments.
    quiet_start: bool = False
    dynamic_progress: bool = False
    after_checkpoint: bool = False
    planned_jobs: int | None = None
    planned_from: str | None = None

    @property
    def label(self) -> str:
        """``<W.NN>_<name>`` -- the LOG_RULES entry and the parts directory."""
        return f"{self.number}_{self.name}"

    def job_name(self, part=None) -> str:
        """The rule's own ``name:``. A fan-out rule suffixes the part."""
        return self.name if part is None else f"{self.name}_{part}"

    def banner(self, part=None, context=None) -> str:
        """The ``message:`` string, with this rule's summary applied."""
        return rule_banner(
            self.number,
            self.job_name(part),
            context=context,
            summary=self.summary,
            quiet_start=self.quiet_start,
            dynamic_progress=self.dynamic_progress,
            after_checkpoint=self.after_checkpoint,
            planned_jobs=self.planned_jobs,
            planned_from=self.planned_from,
        )

    def log(self, part=None) -> str:
        """The ``log:`` path -- flat for one job, under the label for a fan-out.

        Raises for a banner-only rule rather than returning a path. A part
        written under a label ``LOG_RULES`` does not carry is the `[R10-8]`
        defect: its section never reaches the merged log and its files are never
        cleaned up, and neither raises at run time. Refusing here turns that into
        a PARSE-time error, which is earlier than
        ``tests/test_log_rules_contract.py`` can catch it.
        """
        if not self.logged:
            raise ValueError(
                f"rule {self.label} was registered as banner-only, so a log part "
                "under its label would be merged by nothing. Register it with "
                "`registry.logged(...)` if it should write one."
            )
        return f"{self.log_parts_dir}/{self._stem(part)}.log"

    def benchmark(self, part=None) -> str:
        """The ``benchmark:`` path, in the same two shapes as :meth:`log`.

        Logging and benchmarking declarations are independent.
        """
        return f"{self.benchmark_parts_dir}/{self._stem(part)}.tsv"

    def _stem(self, part):
        return self.label if part is None else f"{self.label}/{part}"


class RuleRegistry:
    """Mints :class:`RuleIdentity` objects and accumulates ``LOG_RULES``.

    ``LOG_RULES`` is the MERGE ORDER for ``merge_logs``, and this builds it in
    declaration order. That is safe rather than fragile: the contract test
    asserts the list reads in rule-number order, so declaring out of order fails
    loudly instead of quietly reordering a merged log. It is strictly stronger
    than the hand-maintained literal it replaces, which could be reordered
    independently of the rules it names.

    A conditionally-declared rule needs no special handling and no
    ``LOG_RULES.append`` beside it -- registering inside the ``if`` is what puts
    the label in the list. Removing that append is not a tidy-up: reading it
    required parsing Snakefile source text, which cannot see a statement, and
    that blindness is what had grown a second parser in the test suite.

    Usage in a Snakefile::

        RULES = RuleRegistry(LOG_PARTS_DIR, f"{project_dir}/benchmarks/_parts")
        LOG_RULES = RULES.log_rules          # the SAME list, not a copy
        DELINEATE_REGION = RULES.logged("0.02", "delineate_region")
        SNAPSHOT_CONFIG = RULES.banner_only("0.01", "snapshot_config")

    ``LOG_RULES`` may be aliased before every rule is registered, because it is
    the registry's own mutable list. A consumer that SNAPSHOTS it -- ``list()``,
    ``sorted()``, a comprehension -- must come after the last registration.
    """

    def __init__(self, log_parts_dir, benchmark_parts_dir):
        self.log_parts_dir = str(log_parts_dir)
        self.benchmark_parts_dir = str(benchmark_parts_dir)
        self.log_rules = []

    def _make(self, number, name, logged, **banner):
        # Registered at DECLARATION, not only when a `message:` renders the
        # banner, so the plan table lists every rule the workflow declares --
        # including those this run's DAG does not define (WF3 reusing a
        # collection defines no generation rules) -- as rows with no jobs.
        _RULE_NUMBERS.setdefault(name, str(number))
        return RuleIdentity(
            number=number,
            name=name,
            log_parts_dir=self.log_parts_dir,
            benchmark_parts_dir=self.benchmark_parts_dir,
            logged=logged,
            **banner,
        )

    def logged(
        self,
        number,
        name,
        *,
        summary=None,
        quiet_start=False,
        dynamic_progress=False,
        after_checkpoint=False,
        planned_jobs=None,
        planned_from=None,
    ) -> RuleIdentity:
        """Register a rule that writes a log part, and record its label.

        ``summary``, ``quiet_start`` and ``dynamic_progress`` are stored on the
        identity and forwarded to :func:`rule_banner` by :meth:`RuleIdentity.banner`.
        """
        identity = self._make(
            number,
            name,
            True,
            summary=summary,
            quiet_start=quiet_start,
            dynamic_progress=dynamic_progress,
            after_checkpoint=after_checkpoint,
            planned_jobs=planned_jobs,
            planned_from=planned_from,
        )
        self.log_rules.append(identity.label)
        return identity

    def banner_only(
        self,
        number,
        name,
        *,
        summary=None,
        quiet_start=False,
        dynamic_progress=False,
        after_checkpoint=False,
        planned_jobs=None,
        planned_from=None,
    ) -> RuleIdentity:
        """A rule with a banner and no log part -- bookkeeping and terminal rules.

        Not in ``LOG_RULES``, so ``merge_logs`` never looks for its section.
        Registering one of these by mistake shows up as an orphaned label in
        ``test_every_declared_label_has_a_producing_rule``; failing to register a
        rule that DOES log shows up in ``test_every_logging_rule_is_declared``.
        Both directions are covered, which is why this is two methods rather
        than a boolean nobody would read at the call site.
        """
        return self._make(
            number,
            name,
            False,
            summary=summary,
            quiet_start=quiet_start,
            dynamic_progress=dynamic_progress,
            after_checkpoint=after_checkpoint,
            planned_jobs=planned_jobs,
            planned_from=planned_from,
        )


#: Rules whose per-job START line is not printed, filled by `rule_banner`'s
#: `quiet_start`. See that argument for what earns a rule a place here.
_QUIET_START_RULES = set()


def rule_banner(
    number,
    name,
    context=None,
    summary=None,
    quiet_start=False,
    dynamic_progress=False,
    after_checkpoint=False,
    planned_jobs=None,
    planned_from=None,
):
    """Return a rule's ``message:`` string: a numbered console banner.

    Shows ``<W.NN>  <name>`` (the ``W.NN`` matching the rule's log/benchmark
    filenames) so the live Snakemake console is easy to track.

    **Returns PLAIN TEXT â€” it never colours.** Colour belongs to whoever writes
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
    job â€” so a ``context`` holding ``{wildcards.<name>}`` resolves per member
    even though the banner itself was built once.

    Two constraints on what a caller may pass:

    * Only wildcards the rule actually declares. The message is formatted
      against that job's namespace, so a stray field fails at RUN time, not at
      parse time â€” `tests/test_snake_utils.py` pins the shape, and
      `tests/test_cli.py`'s dry-run is what exercises the real namespaces.
    * **ASCII only.** A Windows console defaults to cp1252 and raises
      ``UnicodeEncodeError`` on the typographic separators that would read
      better here. Use ``|``, not ``Â·``.

    A rule with no wildcards may still pass a constant context; only the
    *interpolation* needs a wildcard, not the suffix itself.

    The context is BRACKETED (``[rlz 1 | st 2]``). Without ANSI -- a pipe, a
    redirect, CI -- the banner's fields rest on nothing but ``-`` and a double
    space, and the trimming described below removes the ``-`` from most start
    lines, leaving two adjacent runs of text separated by whitespace alone.
    Brackets keep the qualifier separable in every one of the three
    destinations, at the cost of two characters.

    ``summary`` is a plain-language clause saying what the rule DOES, for the
    rules a person waits on: ``1.13  run_historical_simulation`` is an identifier, and someone
    watching a multi-hour run should not have to know the codebase to read the
    console. Applied to the LONG-RUNNING rules only, per the parked note this
    discharges â€” a sentence on all 47 would lengthen every line to say what the
    fast ones already say by name, and the value is precisely in the rules where
    you are waiting and wondering. It is constant, so unlike ``context`` it must
    not contain a wildcard.

    Being constant is exactly why the CONSOLE prints it once per rule and not
    once per job: the long-running rules are also the FANNED-OUT ones, so a
    summary reprinted per member is the same sentence on every line of the
    longest stretch of a WF3 run â€” 400 identical clauses on a 10 x 20 grid,
    since rules 3.11 and 3.14 are one job per member. The string returned here
    is unchanged and always carries it; ``_ConsoleHandler._start_line`` does the
    trimming, because the other two destinations want the whole sentence on
    every line. The log file is grepped a line at a time, and a ``JOB_ERROR``
    block is read in isolation from whatever scrolled past hours earlier.

    Order is ``Rule <number>: <name> - <summary>  <context>``: identifier first
    because it is what the log filenames, the benchmark table and this file's
    own rule comments all key on.

    The identifier is SPELLED OUT (``Rule 1.07: build_wflow_model``) rather than
    left as a bare ``1.07  build_wflow_model``. Two digits and a dot are a rule
    number to someone who already knows this console; to everyone else they are
    an unexplained figure sitting where a version or a count could equally well
    be, and the word costs five characters once per line.

    ``quiet_start`` suppresses this rule's per-job START line; its finish line
    is unchanged. For BOOKKEEPING rules only -- a rule whose per-member work is
    a file copy attached to the member another rule just produced. WF3's
    ``3.09 retain_scenario_forcing`` is the case it was added for: it takes
    about a second, interleaves with the rule it follows, and spent two console
    lines per member, so on a 10 x 20 grid it put 800 lines into the longest
    stretch of a run to report file copies.

    What is given up, and why it is affordable: the finish line still carries
    the duration and the progress counter, so the rule is still visible and
    still timed; the heartbeat still reports a stall, since it watches the job
    wrapper rather than the console line; and a failure still prints
    Snakemake's own error block. What goes is the ability to see that such a
    rule has STARTED -- which for a rule that finishes in a second is a line
    that was already obsolete by the time it was read.

    Do NOT reach for this on a rule anyone waits on. The console's whole job
    during a long rule is to say that something is running.

    ``after_checkpoint`` marks a rule whose jobs Snakemake adds only once a
    checkpoint resolves, so it is absent from the opening job table without
    being up to date. ``planned_jobs`` is the count it will run then; the plan
    block shows it as the rule's job count, or an empty cell when not given.
    ``planned_from`` names a rule with one job per job of this one; when that
    rule is in the opening job table, its count replaces ``planned_jobs``.

    ``dynamic_progress`` marks a rule in a workflow whose job total can change
    after checkpoint expansion. Once a handler sees one, finish lines keep the
    stable completed-job numerator and omit Snakemake's provisional total.

    Side effect: records ``name -> number`` in ``_RULE_NUMBERS`` so
    :func:`install_console_style` can put the number on a job's FINISH line,
    which Snakemake reports by rule name only, and ``name -> summary`` in
    ``_RULE_SUMMARIES`` for the once-per-rule trimming above. See those
    registries' comments.
    """
    _RULE_NUMBERS[name] = str(number)
    if summary:
        _RULE_SUMMARIES[name] = summary
    if quiet_start:
        _QUIET_START_RULES.add(name)
    if dynamic_progress:
        _DYNAMIC_PROGRESS_RULES.add(name)
    if after_checkpoint:
        _AFTER_CHECKPOINT_RULES[name] = planned_jobs
    if planned_from:
        _PLANNED_FROM[name] = planned_from
    tag = f"{rule_id(number)} {name}"
    if summary:
        tag = f"{tag} - {summary}"
    if not context:
        return tag
    return f"{tag}  [{context}]"


def declared_step(number, name):
    """Register a pipeline step that has no job in this run's DAG.

    It lists at its number as a row with no jobs, so the plan table still
    shows the whole pipeline when a run skips the step outright -- WF3's
    3.04-3.06 when an existing collection is reused rather than created.
    """
    _RULE_NUMBERS.setdefault(name, str(number))


def pre_dag_step(number, name):
    """Register a step completed before Snakemake built the DAG.

    The plan table lists it at its number, dimmed with an empty count cell,
    and the head tallies it as ``completed before DAG execution``, so a workflow whose
    planning runs outside Snakemake (WF3) still reads as one numbered
    pipeline. Call it only once the step has actually run.
    """
    _PRE_DAG_STEPS[name] = str(number)


def run_summary(
    workflow,
    project_dir,
    log_name,
    benchmarks_name,
    elapsed_seconds=None,
    failed=False,
    log_parts_dir=None,
    warnings=None,
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

    A FAILURE is shaped like :func:`run_header` -- a ruled title, a blank, then
        one aligned column -- so a run closes in the shape it opened in. The
        ``wrote`` label and the second indent level went with the header's ``run``
        and ``path tokens`` groups on 2026-09-06.

        A SUCCESS is one line, and the asymmetry is the point: on success rule
        ``all`` has just listed every target it produced, the log and the benchmark
        table among them, so a block repeating two of those paths said nothing new.
        On FAILURE rule ``all`` never ran, the merged log does not exist yet, and
        the log-parts directory is the only record there is -- so the block that
        names it is the only place that fact appears.

        A FAILED verdict is painted red (``_ANSI_FAIL``), gated on stderr being a
        colour console. The success verdict is not painted at all: what matters is
        that a failed run looks different from every other run WITHOUT being read,
        and colouring both would turn the pair into a field to be read instead.
    """
    project_dir = os.fspath(project_dir)
    lines = []
    verdict = "FAILED" if failed else "completed"
    head = f"{workflow} {verdict}"
    if elapsed_seconds is not None:
        head = f"{head} in {format_elapsed(elapsed_seconds)}"
    # A run that printed a warning must not end in a line indistinguishable
    # from a clean one. The rows themselves have scrolled past by now, so the
    # verdict carries the COUNT and the log carries the text -- this stays one
    # line and one statement, which is the whole point of a verdict.
    #
    # PIPED, the same separator the plan head uses for its fields. Passed IN
    # rather than read from the environment here: `declare_warning_tally` sets a
    # process-global variable at Snakefile PARSE, so anything that parses a
    # Snakefile in-process -- a DAG contract test, a dry run -- leaves this
    # function reading a tally belonging to a run it is not summarizing. The
    # Snakefile that declared the tally is the one that reads it back.
    if warnings:
        head = f"{head}  |  {plural(warnings, 'warning')}"
    plain_head = head
    if failed:
        # Coloured HERE, unlike `rule_banner`, and the difference is where the
        # string goes: a banner reaches a log file and an error block, where an
        # escape code is corruption. This block reaches the console and nothing
        # else -- every Snakefile writes it to stderr and no log receives it --
        # so painting it here cannot leak. Asked of stderr for the same reason.
        head = _paint_body(head, _console_colour(sys.stderr), _ANSI_FAIL)
    else:
        # Success: the NAME is bold and the verdict is not, which matches the
        # opening title. A failure stays wholly red instead -- one loud signal
        # beats two competing ones, and red already says everything bold would.
        head = (
            _paint_body(workflow, _console_colour(sys.stderr), _ANSI_TITLE)
            + head[len(workflow) :]
        )
    # Sized on the plain text built above, before any painting: `head` may
    # already carry escape codes, which are not columns.
    if not failed:
        # ONE LINE on success. The `log` and `benchmarks` rows this used to
        # carry are already among rule `all`'s targets, printed by
        # `target_banner` a few lines above -- so the block restated two paths
        # the reader had just seen, under a rule, after a blank. What is NOT
        # said anywhere else is the duration, so that is what survives.
        #
        # It survives rather than going with them because two readers depend on
        # it: a workflow run DIRECTLY has no other statement of how long it
        # took, and `scripts/run_workflows.py` deliberately prints no band of
        # its own on success precisely because this line exists (its module
        # docstring says so, and `tests/test_run_workflows.py` pins it).
        return head
    lines.extend([head, title_rule(plain_head), ""])
    parts = os.fspath(log_parts_dir or f"{project_dir}/logs/_parts")
    lines.extend(meta_row_lines([("log parts", f"{parts}/")]))
    if failed:
        # A NOTE, not a row: it names no artifact, so giving it a key column
        # would file a sentence under a heading meaning "paths this run wrote".
        lines.extend(["", "  the failing job's own log is printed above"])
    return "\n".join(lines)


def display_root(project_dir):
    """The run's project root as the console STATES it: absolute, then marked.

    Two readers print this root -- :func:`run_meta_rows`' ``<project>`` row and
    :func:`target_banner`'s bracket -- and until 2026-09-16 they printed
    whatever form the Snakefile happened to hold. WF0 to WF3 read a relative
    ``project_dir`` straight from the config; WF4 writes
    ``Path(...).resolve().as_posix()``. One ``run_workflows.py`` run therefore
    stated one fact two ways, a few lines apart:

        wf3   project      test_case/test_rapid
        wf4   project      C:/workspace/.../test_case/test_rapid

    Absolutized HERE rather than at each Snakefile's definition, because the
    held value is not only printed: it is the root `_relativize_paths` and
    `target_banner` strip against, and a workflow whose targets are built from
    a relative ``project_dir`` needs a relative root to strip with. Display and
    stripping want different forms of the same directory, so they are computed
    separately -- see the two variables in `target_banner`.

    Then passed through :func:`_relativize_paths` with NO project root, which is
    exactly what the ``config`` row beside it already does: it applies the
    ``<repo>`` and ``<site-packages>`` rewrites without stripping a project
    prefix. That is what keeps the absolute form readable. A dev tree inside the
    checkout reads ``<repo>/test_case/test_rapid`` -- shorter than the relative
    spelling it replaces AND unambiguous about which checkout, which matters on
    a machine with six worktrees. A production ``project_dir``, which lives
    outside the repository tree, matches nothing and prints in full, which is
    correct: there is no shared root to imply.
    """
    if project_dir is None or not str(project_dir).strip():
        return ""
    absolute = os.path.abspath(os.fspath(project_dir)).replace(os.sep, "/")
    return _relativize_paths(absolute, "")


def target_banner(name, targets, project_dir=None, *, compact=False):
    """Return a target rule's ``message:``: the banner, then one target per line.

    A target carries NO number: it does no work, so numbering it put a step in
    the pipeline overview that is not one. The banner reads ``Target: <name>``
    and registers nothing in ``_RULE_NUMBERS``, so the finish line shows the
    bare name; the name must also be listed in ``_PLAN_EXCLUDED_RULES``.

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

    With ``compact=True``, show only provided/requested counts. Use only for
    target aggregators: Snakemake runs them once all requested inputs exist.
    As with other job messages, a dry-run shows the planned completion.
    Pass ``targets=None`` for checkpoint-dependent inputs; the console handler
    counts those from the resolved job record at execution time.
    """
    if compact:
        if targets is None:
            return "Confirm requested outputs"
        count = sum(1 for _ in targets)
        return f"Requested outputs: {count}/{count} provided"
    banner = f"Target: {name}"
    action = {
        "all": "confirm requested outputs",
        "simulations_only": "confirm retained native responses",
        "simulations_and_indicators": "confirm outputs for the selected operation",
    }.get(name)
    if action:
        banner += f" - {action}"
    listed = [os.fspath(target) for target in targets]
    if project_dir:
        # TWO forms of one directory, deliberately not one variable.
        #
        # `strip_root` keeps the form the CALLER passed, because that is the
        # form its targets were built from: WF3 writes
        # `f"{project_dir}/logs/..."` against a relative config value, so an
        # absolutized root would match nothing, the strip would fail, and every
        # target would print LONGER than before the bracket existed.
        # (`_relativize_paths` tries both spellings, but only of the root it is
        # handed.)
        #
        # `shown` is what the reader sees, and it is absolute and `<repo>`-marked
        # so that the same fact reads the same way in every workflow -- see
        # `display_root`.
        strip_root = os.path.normpath(os.fspath(project_dir))
        tokens = _path_tokens()
        listed = [_relativize_paths(target, strip_root, tokens) for target in listed]
        banner = f"{banner}  [{display_root(project_dir)}]"
    body = "\n".join(f"    {target}" for target in listed)
    return f"{banner}\n{body}" if body else banner


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
# is filtered before handler installation by the owned runners' snakemake_console
# entry point. Bare Snakemake, dry runs and verbose runs retain native startup output.
#
# Nor can `--quiet` reach it, which is the obvious next idea and was measured
# route by route on 2026-09-18 (`dev/tasks/t2609181302-*`): the category named
# `host` has no effect at all, and every category that DOES remove a preamble
# line starves this handler of the records it draws from -- `progress` takes
# the plan block and every finish row, `all` leaves nothing but the rule's own
# output. `--quiet` filters at the record level, upstream of any handler, so
# the two cannot be separated from here. Installing this handler at PARSE time
# rather than at `onstart` does not help either: the lines still print and
# Snakemake's raw `Job stats:` table comes back.
_CONSOLE_MAX_TRACKED_JOBS = 4096


#: The two words that open a job line, PADDED TO ONE WIDTH so the identity that
#: follows starts at the same column on a start and a finish -- the pair reads
#: as a column, not as two sentences. Upper case because they are the only
#: markers on the console and nothing else competes for that weight.
_MARKER_RUN = "RUN "


_MARKER_DONE = "DONE"


_CONSOLE_MUTED_PREFIXES = (
    "Assuming unrestricted shared filesystem usage.",
    "host: ",
    "Building DAG of jobs...",
    "Provided cores: ",
    "Rules claiming more threads will be scaled down.",
    "Indicator details: ",
    "DAG of jobs will be updated after completion.",
    "Updating checkpoint dependencies.",
    "Select jobs to execute...",
    "Removing temporary output ",
    "Would remove temporary output ",
    "Touching output file ",
)

# Presentation labels only: numerical request ids and rule names stay unchanged.
_INDICATOR_CONTEXTS = {}
_INDICATOR_RULE_LABELS = {
    "publish_wflow_outputs": "Retain simulation outputs",
    "prepare_indicator_plan": "Check simulation outputs for indicators",
    "derive_system_indicators": "Calculate indicators",
}


def register_indicator_context(key, identity, names):
    """Associate an opaque job wildcard with its selected indicator-set names."""
    _INDICATOR_CONTEXTS[(key, str(identity))] = ", ".join(names)


#: PUBLIC: `scripts/run_workflows.py` sets it. Set on the environment every
#: workflow the runner
#: invokes inherits. It means "something upstream has already named this
#: workflow to this console", and the only thing it suppresses is the title.
#:
#: An ENV VAR rather than a config key or a CLI flag: the fact it carries is
#: about the INVOCATION, not about the project, and the runner already owns
#: the child's environment (`simulation_command` builds its own from
#: `os.environ`, so one assignment covers both spawn paths).
ANNOUNCED_ENV = "CST_RUN_ANNOUNCED"
_ENGINE_ARGV = None
_INPUT_CHECKS = []
_COUNTED_INPUT_CHECKS = set()


def source_variable_labels(variables):
    """Name requested fields without asserting unverified source availability."""
    labels = {
        "kin": "radiation",
        "press": "pressure",
        "sh": "specific humidity",
        "wind10": "wind",
        "wind10_u": "eastward wind",
        "wind10_v": "northward wind",
        "kout": "top-of-atmosphere radiation",
    }
    return ", ".join(
        f"{labels[name]} ({name})" if name in labels else name for name in variables
    )


def set_engine_command(argv):
    """Retain finalized engine arguments after owned configuration capture."""
    global _ENGINE_ARGV
    _ENGINE_ARGV = list(argv)


def engine_command():
    """Return known Snakemake CLI arguments, or no CLI metadata for embedded APIs."""
    if _ENGINE_ARGV is not None:
        return list(_ENGINE_ARGV)
    entry = Path(sys.argv[0].replace("\\", "/"))
    native = entry.name.lower() in {"snakemake", "snakemake.exe", "snakemake-script.py"}
    native_module = (
        entry.name == "__main__.py" and entry.parent.name.lower() == "snakemake"
    )
    return ["snakemake", *sys.argv[1:]] if native or native_module else None


def _engine_command_line():
    argv = engine_command()
    if not argv:
        return "Unavailable (embedded API invocation)"
    argv = sanitize_argv(argv)
    return shell_command(argv, multiline=True)


def shell_command(argv, *, powershell=None, multiline=False):
    """Quote literal arguments for the invoking platform's shell."""
    powershell = os.name == "nt" if powershell is None else powershell
    separator = " `\n  " if powershell else " \\\n  "
    parts = []
    for index, arg in enumerate(argv):
        arg = str(arg)
        quoted = "'" + arg.replace("'", "''") + "'" if powershell else shlex.quote(arg)
        new_option = re.match(r"--?[A-Za-z-]", arg) is not None
        parts.append(
            separator if multiline and index and new_option else " " if index else ""
        )
        parts.append(quoted)
    return ("& " if powershell else "") + "".join(parts)


def _dry_run():
    return any(
        arg in {"-n", "--dry-run", "--dryrun"}
        or (re.fullmatch(r"-[np]+", arg) and "n" in arg)
        for arg in (engine_command() or [])[1:]
    )


def _compact_header():
    """Identify a no-op without printing settings, paths or scheduled-input checks."""
    if _RUN_HEADER is None:
        return []
    workflow, _, _, details = _RUN_HEADER
    lines = ["", workflow]
    if details.get("cores") is not None:
        lines.append(f"cores  {details['cores']}")
    lines.extend(["", "Snakemake command", _engine_command_line()])
    if _dry_run():
        lines.extend(["", "DRY RUN — no jobs will execute"])
    return lines


def defer_input_check(rule, summary, details, *, module="resolution"):
    """Hold an acquisition warning until the plan establishes its applicability."""
    check = (str(rule), str(summary), tuple(str(row) for row in details), str(module))
    if check not in _INPUT_CHECKS:
        _INPUT_CHECKS.append(check)


def _input_check_lines(counts):
    lines = []
    for rule, summary, details, module in _INPUT_CHECKS:
        if not counts.get(rule):
            continue
        if not lines:
            lines.extend([("Input checks", "section")])
        key = (rule, summary, details, module)
        if key not in _COUNTED_INPUT_CHECKS:
            note_warning(module)
            _COUNTED_INPUT_CHECKS.add(key)
        lines.append(("WARN  " + summary, "warning"))
        lines.extend(("      " + row, "body") for row in details)
    return lines


def opening_block(
    workflow, project_dir, config_path=None, details=None, plan=None, counts=None
):
    """Build the canonical workflow opening from declared metadata and actual jobs.

    Order: identity/cores, literal engine command, applicable input checks, work
    plan, settings, paths, and execution or planning boundary. Each workflow owns
    its identity even inside a batch. Checkpoint/pre-DAG states remain explicit.
    See dev/reference/dev-console-style.md for the maintained presentation contract.
    """
    details = dict(details or {})
    cores = details.pop("cores", None)
    meta = meta_row_lines(
        run_meta_rows(project_dir, config_path, details), grouped=True
    )
    head, rows = ("", []) if plan is None else (plan[0], list(plan[1]))
    out = [("", "body"), (workflow, "title")]
    if cores is not None:
        out.append((f"cores  {cores}", "body"))
    out.extend(
        [
            ("", "body"),
            ("Snakemake command", "section"),
            (_engine_command_line(), "command"),
        ]
    )
    if _dry_run():
        out.extend([("", "body"), ("DRY RUN — no jobs will execute", "warning")])
    checks = _input_check_lines(counts or {})
    if checks:
        out.extend([("", "body"), *checks])
    if rows:
        out.extend([("", "body"), ("Work plan", "section")])
        out.extend(
            (
                text,
                "run" if running is True else "body" if running is None else "dim",
            )
            for text, running in rows
        )
        out.append((head, "body"))
    elif head:
        out.extend([("", "body"), (head, "body")])
    if meta:
        out.append(("", "body"))
        out.extend(
            (line, "section" if line in {"Run settings", "Paths"} else "body")
            for line in meta
        )
    if rows:
        out.extend(
            [
                ("", "body"),
                ("Planning complete" if _dry_run() else "Execution log", "section"),
            ]
        )
        if _dry_run():
            out.append(
                (
                    "No jobs executed; source data have not been fetched or validated by this dry run.",
                    "body",
                )
            )
    return out


def _console_wildcards(wildcards):
    """Render a job's wildcards in the banner's own grammar (``[rlz 1 | st 0]``).

    Snakemake's ``format_wildcards`` spells these ``rlz=1, st=0``; the finish
    line sits directly under a start line whose context clause reads
    ``[rlz 1 | st 0]``, and two spellings of one fact on adjacent lines is
    exactly the inconsistency the console work has been removing. That includes
    the brackets :func:`rule_banner` puts around its context. Derived from the
    job's wildcards rather than parsed back out of the banner, so no prose is
    ever parsed. CMIP6 ``series_key`` is shown as its identifier alone on DONE
    rows, without the redundant ``series`` label.

    A trailing ``_num`` is DROPPED from the key, and that is what makes the
    claim above true rather than merely intended. The wildcard is ``rlz_num``
    because `dev/reference/dev-naming-conventions.md` says a count-or-index field carries the
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
    if len(items) == 1 and tuple(map(str, items[0])) in _INDICATOR_CONTEXTS:
        return f"[{_INDICATOR_CONTEXTS[tuple(map(str, items[0]))]}]"
    # CMIP6 series identifiers already describe the source, experiment and
    # member; the extra "series" label adds no information to a DONE row.
    fields = " | ".join(
        str(value) if key == "series_key" else f"{_console_wildcard_key(key)} {value}"
        for key, value in items
    )
    return f"[{fields}]"


def _console_wildcard_key(key):
    """``rlz_num`` -> ``rlz``: the console's spelling of an index wildcard.

    Only a TRAILING ``_num``, and only when something is left of it, so a
    wildcard actually named ``num`` keeps its name.

    ``_key`` is dropped on the same terms: WF2 fans out over ``series_key``,
    and its banners write ``series {wildcards.series_key}``, so without this
    the START line said ``[series cmip6_...]`` and the FINISH line under it
    ``[series_key cmip6_...]`` -- the one-fact-two-spellings defect again, one
    suffix along. All three suffixes are `dev/reference/dev-naming-conventions.md` conventions
    for what KIND of field a wildcard is, which is why they belong off a
    console that shows the value beside the name.

    ``_id`` joined them in 2026-09-16's WF3/WF4 console restoration, and it is
    the suffix with the most wildcards behind it: ``collection_id``, ``run_id``,
    ``indicator_request_id`` and ``indicator_set_id`` are every fan-out wildcard the
    two R12 workflows have. WF4's 4.04 banner already wrote
    ``run {wildcards.run_id}`` against a finish line reading ``[run_id 07]``, so
    this was a live defect before those workflows had banners to mismatch.
    """
    for suffix in ("_num", "_key", "_id"):
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)]
    return key


#: The run header, held for the console handler to print once Snakemake has
#: reported the job counts. `onstart:` fires BEFORE that record, so a header
#: written there can only sit above the plan; holding it is what lets the
#: rules follow the title directly and the path tokens sit next to the lines
#: that use them. `None` until a Snakefile declares one.
_RUN_HEADER = None


#: Whether `install_console_style` took effect. When it did not there is no
#: handler to print the held header, so `open_run_header` writes it itself --
#: the header must never be the thing a styling failure silently removes.
_CONSOLE_STYLE_ACTIVE = False
_CONSOLE_STYLE_PENDING = False


def open_run_header(workflow, project_dir, config_path=None, **details):
    """Declare the run's header. Call from ``onstart:``, after the style.

    Holds the header for :class:`_ConsoleHandler`, which prints it together
    with the plan block so the title, the rules and the path tokens form one
    structure. Falls back to writing it immediately -- in the old order, which
    is the only order available without the job counts -- when the console
    style is not active.

    **The deferred warnings are flushed where the header is PRINTED, not
    here.** In the styled path this function writes nothing at all, so draining
    the queue on the way out would put the rows above the title band -- the
    same fault, moved. The styled flush therefore lives in
    :meth:`_ConsoleHandler._opening`, which every styled console goes through;
    this branch, which does print, flushes on the spot.
    """
    global _RUN_HEADER, _HEADER_PRINTED
    _HEADER_PRINTED = False
    if _CONSOLE_STYLE_ACTIVE or _CONSOLE_STYLE_PENDING:
        _RUN_HEADER = (workflow, project_dir, config_path, dict(details))
        return True
    sys.stderr.write(
        "\n" + run_header(workflow, project_dir, config_path, **details) + "\n\n"
    )
    _HEADER_PRINTED = True
    held = _drain_deferred_warnings(_console_colour(sys.stderr))
    if held:
        sys.stderr.write("\n".join(held) + "\n\n")
    return False


#: Rules kept OUT of the plan block: the TARGET rules, which :func:`target_banner`
#: labels and which carry no number. A target declares no output and does no
#: work, but Snakemake counts it as a job -- so the block's "5 of 19 rules" and
#: Snakemake's own "6 jobs" differ by exactly the target, deliberately. ``all``
#: is every workflow's; the other two are WF4's partial targets.
_PLAN_EXCLUDED_RULES = ("all", "simulations_only", "simulations_and_indicators")

#: Targets that stop before every checkpoint-gated rule: WF4's
#: ``simulations_only`` ends at the retained responses, before 4.07.
_PRE_CHECKPOINT_TARGETS = ("simulations_only",)


def _after_checkpoint_pending(counts):
    """Whether this run's targets reach the rules a checkpoint will add.

    Under ``--target simulations_only`` the indicator rules are absent from the
    job table because they will not run, not because a checkpoint is pending.
    """
    targets = [name for name in _PLAN_EXCLUDED_RULES if counts.get(name)]
    return not targets or any(name not in _PRE_CHECKPOINT_TARGETS for name in targets)


def _planned_count(name, counts):
    """Jobs a checkpoint-gated rule absent from ``counts`` will run, or None.

    Its ``planned_from`` rule's count in this run's table when listed there,
    else its declared ``planned_jobs``. Shared by the plan block and the job
    counter so the two agree.
    """
    source = _PLANNED_FROM.get(name)
    if source in counts:
        return counts[source]
    return _AFTER_CHECKPOINT_RULES[name]


def _run_info_counts(text):
    """Parse Snakemake's ``Job stats:`` table into ``{rule name: job count}``.

    By the ``<name>  <count>`` shape of a table row, so the header, the ruled
    line and the ``total`` row are skipped by that shape alone. A message that
    yields nothing -- a reworded table, or some other ``run_info`` -- returns
    an empty mapping and every caller falls back to Snakemake's own text.
    """
    counts = {}
    for row in text.splitlines():
        match = re.fullmatch(r"(\S+)\s+(\d+)", row.strip())
        if match and match.group(1) != "total":
            counts[match.group(1)] = int(match.group(2))
    return counts


def _plan_rule_name(names):
    """One display name for the rules SHARING a number.

    WF0 builds ``extract_historical_climate_<source>`` in a Python loop, so
    three rule objects carry the number ``0.03``; the shared prefix is the rule
    as a person names it. Falls back to the first name when the group has no
    usable common prefix, which is the safe direction: a real name from the
    Snakefile beats a truncation.
    """
    ordered = sorted(names)
    if len(ordered) == 1:
        return ordered[0]
    shared = os.path.commonprefix(ordered).rstrip("_")
    return shared or ordered[0]


def _plan_rows(counts):
    """``[(number, name, jobs), ...]`` in rule-number order.

    Joins the two things that each know half the answer: ``_RULE_NUMBERS``
    (every rule the Snakefile DECLARED, filled by :func:`rule_banner` at parse
    time) and Snakemake's job-stats counts (the rules that will actually RUN).
    A rule absent from ``counts`` is up to date. Steps registered by
    :func:`pre_dag_step` join as rows whose job count is ``None``: they ran
    before the DAG existed.

    Grouped by NUMBER, not by name, because a number is not unique: see
    :func:`_plan_rule_name`. Sorted lexicographically on the number, which
    orders ``1.13`` before ``1.14`` before ``1.15`` without a version parser.
    """
    by_number = {}
    for name, number in _RULE_NUMBERS.items():
        if name in _PLAN_EXCLUDED_RULES:
            continue
        by_number.setdefault(number, []).append(name)
    rows = []
    pending = _after_checkpoint_pending(counts)
    for number, names in by_number.items():
        jobs = sum(counts.get(name, 0) for name in names)
        if (
            not jobs
            and pending
            and all(name in _AFTER_CHECKPOINT_RULES for name in names)
        ):
            # Absent only because the checkpoint has not resolved: its planned
            # count, or an empty cell when it declared none.
            planned = [_planned_count(name, counts) for name in names]
            jobs = sum(planned) if all(n is not None for n in planned) else ""
        rows.append((number, _plan_rule_name(names), jobs))
    if rows:
        rows.extend(
            (number, name, None)
            for name, number in _PRE_DAG_STEPS.items()
            if number not in by_number
        )
    rows.sort(key=lambda row: row[0])
    return rows


def _plan_lines(counts):
    """Render explicit run/reuse/done/pending rows and the scheduled work-job total."""
    rows = _plan_rows(counts)
    if not rows:
        return None
    pending_numbers = {
        _RULE_NUMBERS[name]
        for name in _AFTER_CHECKPOINT_RULES
        if _after_checkpoint_pending(counts)
        and name in _RULE_NUMBERS
        and name not in counts
    }
    display = []
    states = []
    for number, name, jobs in rows:
        pending = number in pending_numbers
        status = (
            "done"
            if jobs is None
            else "pending"
            if pending or isinstance(jobs, str)
            else "run"
            if jobs
            else "reuse"
        )
        cell = (
            "-"
            if jobs is None or jobs == 0
            else "?"
            if jobs == ""
            else f"{jobs} planned"
            if pending
            else str(jobs)
        )
        display.append((status, number, _INDICATOR_RULE_LABELS.get(name, name), cell))
        states.append(
            True if status == "run" else None if status == "pending" else False
        )
    widths = [
        max(len(row[i]) for row in [*display, ("status", "step", "rule", "jobs")])
        for i in range(4)
    ]
    lines = [
        (
            f"{'status':<{widths[0]}}  {'step':<{widths[1]}}  {'rule':<{widths[2]}}  {'jobs':>{widths[3]}}",
            None,
        )
    ]
    lines.extend(
        (
            f"{status:<{widths[0]}}  {number:<{widths[1]}}  {name:<{widths[2]}}  {cell:>{widths[3]}}",
            state,
        )
        for (status, number, name, cell), state in zip(display, states)
    )
    unlisted = sum(
        1
        for name in counts
        if name not in _RULE_NUMBERS and name not in _PLAN_EXCLUDED_RULES
    )
    jobs = sum(
        count for name, count in counts.items() if name not in _PLAN_EXCLUDED_RULES
    )
    scheduled = sum(state is True for state in states)
    reused = sum(row[0] == "reuse" for row in display)
    head = f"{plural(jobs, 'job')}  |  {plural(scheduled, 'rule')} scheduled  |  {plural(reused, 'rule')} up to date"
    pending = sum(row[0] == "pending" for row in display)
    done = sum(row[0] == "done" for row in display)
    if pending:
        head += f"  |  {plural(pending, 'rule')} pending checkpoints"
    if done:
        head += f"  |  {done} {_PRE_DAG_LABEL}"
    if unlisted:
        head += f"  |  {unlisted} unlisted"
    return head, lines


class _ConsoleHandler(logging.StreamHandler):
    """Snakemake's terminal handler, restyled: one line per job start and end.

    Snakemake spends SIX console lines on a job -- a blank separator, a bare
    ``[Thu Aug 13 23:26:41 2026]``, ``Job 8: <message>``, a second bare
    timestamp, ``Finished jobid: 8 (Rule: check_project_consistency)`` and
    ``1 of 37 steps (3%) done`` -- plus ``Select jobs to execute...`` and
    ``Execute N jobs...`` per scheduling wave. On a WF3 run that is thousands
    of lines in which one line per job is ours. This renders two::

        23:26:44 - RUN  Rule 3.10: generate_weather_realizations - generate ...
        23:27:14 - DONE Rule 3.10: generate_weather_realizations  0:00:30  [8/37]

    On a FANNED-OUT rule the summary clause is printed on the first member only
    and trimmed from the rest (:meth:`_trim_summary`), so the run's longest
    stretch reads as a list of members rather than one sentence restated a few
    hundred times::

        16:16:57 - RUN  Rule 3.14: downscale_scenario_series - downscale ...  [rlz 2 | st 0]
        16:16:57 - RUN  Rule 3.14: downscale_scenario_series  [rlz 2 | st 2]
        16:16:58 - RUN  Rule 3.14: downscale_scenario_series  [rlz 2 | st 3]

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
        #: Last counter Snakemake reported, replayed onto the START line so a
        #: long fan-out shows its position while it runs rather than only as
        #: each member finishes. Snakemake's number, shown earlier -- NOT a
        #: second counter, for the reason set out in :meth:`_render`.
        self._progress = (None, None)
        self._dynamic_progress = False
        # Target jobs (`all`, WF4's partial targets) do no work, so the job
        # counter leaves them out: `_target_jobs` from the job table, and
        # `_targets_done` so jobs finishing AFTER a mid-run target keep counting
        # from where the work actually is.
        self._target_jobs = 0
        self._targets_done = 0
        # Work jobs the plan expects, including jobs Snakemake adds only after a
        # checkpoint resolves; None until the job table is seen. See
        # `_progress_label`.
        self._planned_jobs = None
        # Rule names whose summary clause has already been printed once. Held on
        # the INSTANCE, unlike `_RULE_NUMBERS`/`_RULE_SUMMARIES`, which are
        # module-level and outlive one workflow: `run_workflows.py` drives five
        # Snakefiles, and a shared set would give the second and later ones a
        # console on which no summary was ever printed at all.
        self._summarized = set()
        # The opening block is printed once, by whichever record is first.
        self._opened = False
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
            # Reset the line first: a job running under `-c 3` may have a
            # progress frame standing on it, drawn from ANOTHER PROCESS, and
            # this handler's row would otherwise be appended to that frame with
            # no break between them. See `_line_reset`; unconditional, because
            # every row this handler writes is whole and terminator-suffixed.
            self.stream.write(_line_reset(self.stream) + text + self.terminator)
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
                self.stream.write(_line_reset(self.stream) + pending + self.terminator)
                self.flush()
        except Exception:  # noqa: BLE001 -- teardown must not raise
            pass
        super().close()

    # -- rendering ---------------------------------------------------------

    def _render(self, record):
        global _HEADER_PRINTED
        fields = record.__dict__
        event = fields.get("event")
        event = None if event is None else str(event)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", str(record.msg or "")).strip()
        if plain.startswith(("Nothing to be done", _UP_TO_DATE_LINE)):
            lines = [] if self._opened else _compact_header()
            self._opened = True
            _HEADER_PRINTED = True
            lines.extend(
                ["", self._paint(_UP_TO_DATE_LINE + " — no jobs scheduled", _ANSI_DONE)]
            )
            lines.extend(_drain_deferred_warnings(self._color))
            return "\n".join(lines)
        if _dry_run() and event in {
            "job_info",
            "job_started",
            "job_finished",
            "progress",
        }:
            return None

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
            self._progress = (fields.get("done"), fields.get("total"))
        else:
            lines = self._drain(None, None)
            if (
                event in {"job_info", "job_error", "group_error"}
                or record.levelno >= logging.ERROR
            ):
                # Whichever record comes first opens the run; `run_info`
                # normally does and carries the plan with it.
                opening = self._opening()
                if opening:
                    lines.append("\n".join(opening) + "\n")
            if event == "job_info":
                lines.append(self._start_line(fields, record))
            elif event == "job_started":
                pass  # "Execute N jobs..." -- scheduler bookkeeping
            elif event == "run_info":
                lines.extend(self._plan_block(record))
            elif not self._muted(record, event):
                if event in {"job_error", "group_error"}:
                    lines.extend(self._failure_lines(fields, event))
                # Body tier for what is informational only. A WARNING or an ERROR keeps
                # Snakemake's own colouring, which is the one thing on this
                # console that must stay louder than a start line.
                #
                # Shortened the same way a rule's own output is (the tee calls
                # the same function): Snakemake's `Complete log(s):` line ends a
                # run with a 100-character absolute path, in OS separators,
                # directly above a summary block whose every path is short and
                # forward-slashed. One console, one spelling of a path.
                #
                # With the declared project root, as `log_row` passes it: the
                # error block prints paths in the RELATIVE spelling the
                # Snakefile declared, which only the root-relative token
                # spelling can match (t2609172119).
                shown = _relativize_paths(
                    self.format(record),
                    os.environ.get(_PROJECT_ROOT_ENV, ""),
                    _path_tokens(),
                )
                shown = _shorten_console_paths(
                    shown, os.environ.get(_PROJECT_ROOT_ENV, ""), _path_tokens()
                )
                if (
                    shown == "None"
                    and record.msg is None
                    and record.levelno <= logging.INFO
                ):
                    shown = ""
                if record.levelno <= logging.INFO:
                    shown = self._paint(shown, _ANSI_BODY)
                lines.append(shown)

        return "\n".join(line for line in lines if line) or None

    def _failure_lines(self, fields, event):
        """Identify failed jobs before the original traceback and log dump.

        Use Snakemake's structured fields, including each member of a failed
        group. Do not infer a failed rule from the last job that started: jobs
        can run concurrently. Records without an identity retain native output.
        """
        jobs = fields.get("job_error_info", ()) if event == "group_error" else [fields]
        lines = []
        for job in jobs:
            name = job.get("name") if event == "group_error" else job.get("rule_name")
            if not name:
                continue
            number = _RULE_NUMBERS.get(name)
            identity = f"{rule_id(number)} {name}" if number else name
            jobid = job.get("jobid")
            suffix = f"  |  job {jobid}" if jobid is not None else ""
            head = f"FAILED {identity}{suffix}"
            lines.extend(
                [
                    self._paint(head, _ANSI_FAIL),
                    self._paint(title_rule(head), _ANSI_FAIL),
                ]
            )
            for path in job.get("log", ()):
                shown = _relativize_paths(
                    os.fspath(path),
                    os.environ.get(_PROJECT_ROOT_ENV, ""),
                    _path_tokens(),
                )
                lines.append(f"  log  {shown}")
            lines.append("")
        return lines

    def _opening(self, plan=None, counts=None):
        """Emit the declared identity, command, plan and metadata once; see dev-console-style.md."""
        global _HEADER_PRINTED
        if _RUN_HEADER is None or self._opened:
            return []
        self._opened = True
        _HEADER_PRINTED = True
        workflow, project_dir, config_path, details = _RUN_HEADER
        # ONE builder with `run_header`, so a run that loses its styling does
        # not also lose its layout. The tiers come back with the text; mapping
        # them onto colour is this class's business and nobody else's.
        tiers = {
            "title": _ANSI_TITLE,
            "run": _ANSI_DONE,
            "dim": _ANSI_DIM,
            "body": _ANSI_BODY,
            "section": "1;36",
            "command": _ANSI_BODY,
            "warning": "33",
        }
        block = opening_block(workflow, project_dir, config_path, details, plan, counts)
        lines = [self._paint(text, tiers[tier]) for text, tier in block]
        # Generic deferred diagnostics retain their severity after the opening.
        # Acquisition-specific notices are placed by opening_block before the plan.
        held = _drain_deferred_warnings(self._color)
        if held:
            lines.append("")
            lines.extend(held)
        # The opening labels Execution log for real execution; keep a blank
        # line before the timestamped job rows that follow that boundary.
        lines.append("")
        return lines

    def _plan_block(self, record):
        """The run's rules, one per line, keyed on rule id.

        REPLACES the collapsed ``N jobs across M rules`` line rather than
        joining it: that line's whole content is this block's head line.

        Why this earns its lines when the ``Job stats:`` table it descends from
        did not. A rule that is up to date prints NOTHING for the rest of the
        run, so on a re-run the console showed five start lines and could not
        say whether the workflow had five rules or nineteen. Measured on the
        rapid fixture 2026-09-05: WF1 declares 19 rules and a completed build
        leaves 5 with work. That delta exists nowhere else on the console --
        which is the half of the argument for collapsing the table that did not
        survive contact with a re-run.

        Falls back to :meth:`_run_info_line` whenever the plan cannot be built
        -- an unparsable table, or a workflow whose rules never called
        :func:`rule_banner` -- the same fail-open direction every other
        cosmetic rule in this module takes.
        """
        text = self.format(record)
        counts = _run_info_counts(text)
        if _dry_run() and self._opened:
            return []
        self._target_jobs = sum(counts.get(name, 0) for name in _PLAN_EXCLUDED_RULES)
        if counts:
            # The same planned counts the plan block lists for checkpoint-gated
            # rules, so the counter's denominator agrees with the plan above it.
            pending = _after_checkpoint_pending(counts)
            self._planned_jobs = (
                sum(counts.values())
                - self._target_jobs
                + sum(
                    _planned_count(name, counts) or 0
                    for name in _AFTER_CHECKPOINT_RULES
                    # Only this workflow's rules, as `_plan_rows` lists them;
                    # the registry outlives one parsed Snakefile.
                    if pending and name in _RULE_NUMBERS and name not in counts
                )
            )
        plan = _plan_lines(counts) if counts else None
        opening = self._opening(plan, counts)
        if opening:
            painted = opening
        elif plan is None:
            return [self._paint(self._run_info_line(record), _ANSI_BODY)]
        else:
            # No header declared -- a bare `snakemake -s` without `onstart:`,
            # and in the tests. The plan stands on its own, in the same shape
            # it has inside the opening block: the table, then the summary that
            # captions it. Unruled, like the block, and for the same reason --
            # a list of numbered rules does not need to be told what it is.
            head, rows = plan
            painted = [
                self._paint(row, _ANSI_RUN if running else _ANSI_DIM)
                for row, running in rows
            ]
            painted.append(self._paint(head, _ANSI_BODY))
        # ONE element, newlines and all. `_render` joins its lines through a
        # truthiness filter, so a blank passed as its own element is dropped --
        # the block's internal air has to travel inside a single string. The
        # trailing newline is what separates the block from the first RUN line.
        return ["\n".join(painted) + "\n"]

    def _run_info_line(self, record):
        """Collapse Snakemake's ``Job stats:`` table to one line.

        The table is one row per rule plus a total: 22 lines on WF1, where
        every count is 1 and the rule names are the same ones about to scroll
        past on the RUN lines. What a reader wants from it is the SIZE of the
        run and which rules fan out, so that is what the line keeps::

            37 jobs across 21 rules  (downscale_scenario_series x10, perturb_climate_realization x8)

        Parsed from the message text, because ``run_info`` carries only that
        text (``dag.stats`` formats the table before logging it). Parsing is
        by the ``<name>  <count>`` shape of a table row; the header, the rule
        line and the ``total`` row are skipped by that shape, and a message
        that yields no rows -- a reworded table, or some other ``run_info`` --
        is passed through as Snakemake formatted it. Fails open, like every
        other cosmetic rule here.
        """
        text = self.format(record)
        counts = _run_info_counts(text)
        if not counts:
            return text
        jobs = sum(counts.values())
        rules = len(counts)
        line = (
            f"{jobs} job{'s' if jobs != 1 else ''} across "
            f"{rules} rule{'s' if rules != 1 else ''}"
        )
        fanned = [
            f"{name} x{count}"
            for name, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if count > 1
        ]
        if fanned:
            line = f"{line}  ({', '.join(fanned)})"
        return line

    def _muted(self, record, event):
        """Whether a plain INFO line is one of the muted ones.

        Restricted to event-less records at INFO or below, so a warning or an
        error can never be silenced by a prefix that happens to match.
        """
        if event is not None or record.levelno > logging.INFO:
            return False
        if (
            _dry_run()
            and "--verbose" not in (engine_command() or [])[1:]
            and str(record.msg or "").startswith(
                (
                    "Reasons:",
                    "Some jobs were triggered by provenance information",
                    "If you prefer that only modification time",
                    "If you are sure that a change",
                    "Rules with provenance triggered jobs:",
                    "This was a dry-run (flag -n).",
                )
            )
        ):
            return True
        return record.msg is None or str(record.msg or "").startswith(
            _CONSOLE_MUTED_PREFIXES
        )

    def _start_line(self, fields, record):
        # Memoized BEFORE the branch below, so a rule with no `message:` still
        # gets a finish line naming it. Skipping the memo left those jobs
        # finishing as a bare `done  job 9`, which is the one thing the finish
        # line exists to avoid.
        rule_name = fields.get("rule_name")
        if rule_name in _DYNAMIC_PROGRESS_RULES:
            self._dynamic_progress = True
        self._started[fields.get("jobid")] = (
            rule_name,
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
        if fields.get("rule_name") in _QUIET_START_RULES:
            # A BOOKKEEPING rule: its finish line is the whole of what it has to
            # say (see `rule_banner`'s `quiet_start`).
            return None
        message = self._trim_summary(fields.get("rule_name"), message)
        if rule_name in _PLAN_EXCLUDED_RULES and fields.get("input") is not None:
            count = len(fields["input"])
            message = f"Requested outputs: {count}/{count} provided"
        if rule_name in _INDICATOR_RULE_LABELS:
            context = _console_wildcards(fields.get("wildcards"))
            message = f"{rule_id(_RULE_NUMBERS[rule_name])} {_INDICATOR_RULE_LABELS[rule_name]}"
            if context:
                message += f"  {context}"
        # NO progress counter here. It is replayed from the last progress
        # record, so on a start line it is the tally BEFORE this job: two
        # consecutive starts repeat a number, and the last start of a run reads
        # `[13/14]` with nothing finished. It looks like a fact about the job it
        # sits on and is a fact about the previous one. The finish line keeps
        # it, where it is true -- and keeps getting it from `_render`'s held
        # line, which remains the one authority for the number.
        # THREE tiers on one line, rather than one colour across it. The
        # stamp is scaffolding and recedes; the marker carries the state and
        # keeps the blue; the payload takes the terminal's own foreground,
        # where it is easiest to read.
        #
        # The finish line stays WHOLE-LINE green, and the asymmetry is the
        # point. The 2026-08-14 revision that made both lines solid argued from
        # scroll-back -- "what a reader scrolls for is where did this job start
        # / what finished" -- and that is still true of the finish line, which
        # is also the one carrying the numbers. A start line does not need
        # finding: while its job runs it is the last thing on the screen.
        return (
            self._paint(f"{self._now()} - ", _ANSI_DIM)
            + self._paint(f"{_MARKER_RUN} ", _ANSI_RUN)
            + self._paint(message, _ANSI_BODY)
        )

    def _trim_summary(self, rule_name, message):
        """Drop the rule's constant summary clause after its FIRST start line.

        The clause says what the rule DOES, which is a fact about the RULE and
        not about the job -- so on a fanned-out rule it is the same sentence on
        every line for the length of the fan-out. Printed once, the reader has
        it; repeated, it is the widest column on screen carrying no per-job
        information. What remains is ``Rule 3.14: downscale_scenario_series
        [rlz 2 | st 2]``, which is the grammar :meth:`_done_line` already builds
        -- it has never carried a summary -- so the pair converges rather than
        the start line acquiring a format of its own.

        Removal is by the EXACT substring ``rule_banner`` inserted, looked up by
        rule name in ``_RULE_SUMMARIES``. Nothing is parsed out of the assembled
        message: a rule whose banner was built some other way is simply absent
        from the registry and prints unchanged, which is also what happens if
        the two ever fall out of step.

        Keyed on the rule NAME, so WF4's per-batch rules (``run_wflow_simulations_batch_1``,
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
        if rule_name in _PLAN_EXCLUDED_RULES:
            # A target is not a job of work: no counter on its own line, and
            # every later counter shifts down by one.
            self._targets_done += 1
            counter = None
        if rule_name == "all" and os.environ.get("CST_SIMULATION_FINALIZE"):
            # The owned launcher still has result exports to publish.
            return ""
        if rule_name is None and fallback:
            # No start line was seen for this job, so there is nothing to render
            # in our grammar -- under `--quiet rules` the filter dropped it, and
            # a grouped job never emits one. Snakemake's own text still names
            # the rule, which is the whole point of the line; it gets our stamp
            # and the counter and nothing else.
            progress = self._progress_label(counter, total)
            tail = f"  [{progress}]" if progress else ""
            return self._paint(
                f"{self._now()} - {_MARKER_DONE} {fallback}{tail}", _ANSI_DONE
            )
        number = _RULE_NUMBERS.get(rule_name)
        if rule_name:
            label = _INDICATOR_RULE_LABELS.get(rule_name, rule_name)
            identity = f"{rule_id(number)} {label}" if number else label
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
        progress = self._progress_label(counter, total)
        if progress:
            tail.append(f"[{progress}]")
        line = f"{self._now()} - {_MARKER_DONE} " + "  ".join(parts)
        if tail:
            line = f"{line}  " + "  ".join(tail)
        return self._paint(line, _ANSI_DONE)

    # -- decoration --------------------------------------------------------

    def _progress_label(self, counter, total):
        """Render a stable completed-job counter for static or dynamic DAGs.

        Snakemake's total grows when a checkpoint adds jobs (WF4's 4.08 after
        4.07), which read as `[job 112/112]` then `[job 113/114]`. The
        denominator therefore starts at the plan's count, which already includes
        those rules' declared jobs, and grows only if Snakemake reports more.
        """
        if counter is None:
            return ""
        counter -= self._targets_done
        if total:
            total -= self._target_jobs
            if self._planned_jobs:
                total = max(total, self._planned_jobs)
        if self._dynamic_progress:
            return str(counter)
        return f"{counter}/{total}" if total else ""

    def _now(self):
        return f"{datetime.now():%H:%M:%S}"

    def _paint(self, text, code):
        """Colour ``text``, or return it unchanged.

        ``code`` of ``None`` is the BODY tier: no SGR at all, so the terminal's
        own foreground applies. That is a colour decision, not an absence of
        one -- see ``_ANSI_BODY``.

        Whole lines, with ONE exception that arrived with the start line's
        three tiers (see :meth:`_start_line`): the handler may paint fields of
        a line it assembles itself, because nothing else can be looking at that
        string. It still never paints fields of a line that came from
        elsewhere; that is ``_paint_body``'s territory and its own rules.
        """
        return _ansi(text, code) if self._color and code and text else text


def _console_style_took():
    """Record that a console handler is live, and say so.

    `open_run_header` reads this: with a handler there is something to print
    the held header, and without one the header has to be written on the spot.
    A styling failure must never be the reason a run loses its header.
    """
    global _CONSOLE_STYLE_ACTIVE, _CONSOLE_STYLE_PENDING
    _CONSOLE_STYLE_ACTIVE = True
    _CONSOLE_STYLE_PENDING = False
    return True


def install_console_style():
    """Restyle only Snakemake's terminal handler, immediately or through its factory.

    At parse time the listener does not yet exist; register a deferred factory
    while retaining its signature and filter chain. Pending is distinct from an
    active handler. Later calls replace only DefaultStreamHandler in the live
    listener and leave the verbose file handler untouched. Unknown upstream
    seams fail open; held metadata is flushed if no handler ever appears.
    See dev/reference/dev-console-style.md for lifecycle and fallback behavior.
    """
    global _HEADER_PRINTED, _CONSOLE_STYLE_ACTIVE, _CONSOLE_STYLE_PENDING
    try:
        from snakemake.logging import logger_manager

        listener = getattr(logger_manager, "queue_listener", None)
        handlers = list(getattr(listener, "handlers", None) or ())
        for index, handler in enumerate(handlers):
            if getattr(handler, "name", None) != "DefaultStreamHandler":
                continue
            if isinstance(handler, _ConsoleHandler):
                return _console_style_took()  # a second Snakefile (tests)
            _HEADER_PRINTED = False
            handlers[index] = _ConsoleHandler(handler)
            listener.handlers = tuple(handlers)
            return _console_style_took()
        if listener is None:
            _HEADER_PRINTED = False
            factory = getattr(logger_manager, "_default_streamhandler", None)
            if callable(factory):
                _CONSOLE_STYLE_ACTIVE = False
                if not getattr(factory, "_cst_console_style", False):

                    def styled_handler(*args, **kwargs):
                        handler = factory(*args, **kwargs)
                        if not isinstance(handler, _ConsoleHandler):
                            handler = _ConsoleHandler(handler)
                        _console_style_took()
                        return handler

                    styled_handler._cst_console_style = True
                    logger_manager._default_streamhandler = styled_handler
                _CONSOLE_STYLE_PENDING = True
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

    ``Run settings`` groups the workflow's parameters. ``Paths`` groups the
    user-facing configuration first, then project and path-token definitions.
    A blank separates the groups, and their values share one aligned column.

    Every value is FORWARD-SLASHED, including ``config`` -- which was the one
    row that was not, so a block whose whole purpose is to state paths mixed
    ``C:\\a\\b`` with ``a/b`` and read as two trees. ``config`` is also printed
    through the repo strip that every rule log line already uses, so it reads
    ``<repo>/test_case/project_config_rapid.yml`` rather than the same fact
    behind 60 characters of machine-specific prefix.
    """
    # Forward slashes, like every path the folder rows below and the log
    # headers print: one block mixing `C:\a\b` with `a/b` reads as two trees.
    #
    # Through `opening_block`, which the console handler also uses -- the two
    # rendered this block differently until 2026-09-17, so a run whose styling
    # failed to install printed a different shape from every other run. No plan
    # is passed because none exists yet: this fallback is written during parsing, before
    # Snakemake has reported a job count.
    return "\n".join(
        text for text, _ in opening_block(workflow, project_dir, config_path, details)
    )


def header_config_path(
    config_path: str | os.PathLike[str], capture_record: dict | None = None
) -> str:
    """Return the user's project config path, retaining snapshots for execution."""
    if capture_record is None:
        return os.fspath(config_path)
    return next(
        source["original_path"]
        for source in capture_record["source_files"]
        if source["id"] == capture_record["source_config"]
    )


def run_meta_rows(project_dir, config_path=None, details=None):
    """``[(key, value)]`` for the run's metadata block: what run this is.

    Returns flat data for callers; :func:`meta_row_lines` groups parameters
    and path definitions when rendering the opening block.
    """
    # Through `display_root`, like `target_banner`'s bracket: the two print the
    # same directory a few lines apart and disagreed about its spelling until
    # 2026-09-16 (t2609162114).
    rows = []
    if config_path:
        # Keep configs inside the project distinguishable from output paths.
        # External configs also need forward slashes when no token matches.
        config_display = os.fspath(config_path)
        if project_dir:
            root = os.path.abspath(os.fspath(project_dir))
            config_display = _tokenize_prefix(config_display, root, "project")
        rows.append(
            ("config", _relativize_paths(config_display, "").replace("\\", "/"))
        )
    rows.append(("<project>", display_root(project_dir)))
    rows.extend((key, str(value)) for key, value in (details or {}).items())
    rows.extend(_folder_rows(project_dir))
    rows.extend(
        _folder_rows(project_dir, _declared_tokens(variable=_CONSOLE_PATH_TOKENS_ENV))
    )
    return rows


def meta_row_lines(rows, *, grouped=False):
    """Group run settings and paths, keeping path-token definitions together."""
    if not rows:
        return []
    width = max(len(key) for key, _ in rows)
    if not grouped:
        return [f"{key.ljust(width)}  {value}" for key, value in rows]
    paths = [
        (key, value) for key, value in rows if key.startswith("<") or key == "config"
    ]
    settings = [
        (key, value)
        for key, value in rows
        if not key.startswith("<") and key != "config"
    ]
    lines = []
    for title, group in (("Run settings", settings), ("Paths", paths)):
        if group:
            if lines:
                lines.append("")
            lines.append(title)
            lines.extend(f"{key.ljust(width)}  {value}" for key, value in group)
    return lines


def title_rule(text):
    """The ``-`` rule under a workflow title (treatment T2).

    Plain ASCII, so it reaches a redirect, a log file and CI, where the bold
    does not -- the same reason the plan block marks its running rules with a
    ``>`` as well as a colour. Drawn to the text it underlines, so it can never
    wrap a narrow console.
    """
    return "-" * len(text)


def up_to_date_verdict(stream):
    """Format the shared green verdict for a workflow with no work to run."""
    return "\n" + _paint_body(
        _UP_TO_DATE_LINE, _console_colour(stream), _ANSI_DONE
    ).rstrip("\n")


class _UpToDateVerdict(logging.Filter):
    """Restate Snakemake's "Nothing to be done" as a green up-to-date verdict.

    A logger FILTER, not a branch in :class:`_ConsoleHandler`: that record is
    logged only when there are no jobs, and then ``onstart:`` never fires, so
    the handler is never installed to see it. A filter on Snakemake's own
    logger rewrites the record as it is logged, whichever handler prints it.
    Set off by a blank line; painted only on a colour-capable stderr, which is
    where Snakemake's default handler writes.
    """

    def filter(self, record):
        try:
            if str(record.msg).startswith("Nothing to be done"):
                record.msg, record.args = up_to_date_verdict(sys.stderr), ()
        except Exception:  # noqa: BLE001 -- a console style must never end a run
            pass
        return True


def _install_up_to_date_verdict():
    """Attach :class:`_UpToDateVerdict` to Snakemake's logger, once.

    Runs at import, which for a Snakefile is parse time -- before the DAG is
    built and the verdict logged. Only when Snakemake is already loaded, so
    importing this module elsewhere pulls nothing in. Fail-open.
    """
    module = sys.modules.get("snakemake.logging")
    target = getattr(module, "logger", None)
    if not isinstance(target, logging.Logger):
        return
    if not any(isinstance(item, _UpToDateVerdict) for item in target.filters):
        target.addFilter(_UpToDateVerdict())


_install_up_to_date_verdict()
