# Workflow console style

Last updated: 2026-11-10

Use this reference when changing workflow orientation, planning or launcher output. The shared
[`console_style.py`](../../blueearth_cst/shared/console_style.py) owns the layout for all five workflows;
[`snakemake_console.py`](../../blueearth_cst/shared/snakemake_console.py) owns the early terminal adapter.
Rule implementations retain their compact `log_row` output. Vendored `dev/scripts/console.py` is unrelated.

## Workflow blocks

Each workflow prints its identity and effective cores before its rule rows, including inside a batch.
The opening order is identity, **Snakemake command**, applicable **Input checks**, **Work plan**,
**Run settings**, **Paths**, then the labelled **Execution log** boundary. Section headings are cyan;
scheduled plan rows are green, reused rows gray, and attention notices amber. Explicit labels preserve
meaning without color. Existing live rule progress, failure detail and warning severity remain intact.

The command is one physical line, with terminal soft wrapping. PowerShell uses literal single-quoted
arguments with doubled apostrophes and an invocation operator; POSIX uses shell quoting. Arguments are
never reconstructed from display text or Python representations. Owned launches supply the finalized
engine arguments after configuration capture, including the real execution YAML, with credential-like
values redacted under the same policy as invocation records. Ordinary arguments remain shell literals.
**Paths** separately names the user's source config. The command is diagnostic evidence; its temporary
execution inputs may be removed after the invocation.

The work plan labels status, step, rule and jobs. `run` means scheduled work; `reuse` means a current rule,
whose previous job count is unknown. `done` identifies completed work before the DAG. `pending` identifies
checkpoint-dependent work, with explicitly provisional counts where known. The job summary excludes target
collectors such as `all`; unknown or unlisted work is reported rather than silently counted as reused.

## Planning and no-op runs

A dry run prominently states **DRY RUN — no jobs will execute** before its plan and ends the header with
**Planning complete**. It emits no live RUN rows or Execution log boundary. Scheduling a fetch during a dry
run does not prove remote availability. A no-op prints only workflow identity/cores, the actual command and
**Up to date — no jobs scheduled**, plus any independently relevant warnings.

The batch runner identifies the selected sequence and cores per workflow; child blocks own their complete
workflow identity and finalized commands. `CST_RUN_ANNOUNCED` remains child-scoped and must not leak into the
calling environment or suppress the child's identity. Verbose invocations retain native startup output.

## Input checks and failure behavior

The projections availability notice belongs before its work plan, without timestamp/module prefixes. It
appears only when acquisition is scheduled and counts once as a warning then. Catalog certification of
precipitation/temperature does not establish absence of other requested variables; those remain checked by
the existing fetch admission. Analysis-only and no-op invocations do not repeat or count this notice.
Other source-resolution warnings and detailed file logs retain their existing meaning.

Workflow metadata is declared during parsing. If Snakemake's terminal handler is not ready, the adapter
defers styling through its existing handler factory; pending installation differs from an active handler.
It never replaces the detailed file handler. Repeated installation is idempotent. Unknown upstream seams
fail open; early parse errors remain visible, and held identity/command metadata is flushed if no handler
appears. Journaling, admission and scientific work retain their existing lifecycle hooks.

Owned runners filter the empty upstream workflow-start marker before parsing. Direct native Snakemake
may print upstream startup text before project code loads; its subsequent header follows this contract.
Verbose mode retains native startup diagnostics.
Embedded API callers without finalized CLI metadata show the command as unavailable. The native fallback
recognizes Snakemake executable or module entry points; it never presents the host program's arguments as
a Snakemake command or invents an API execution mode.

## Verification

Use the focused console/plan/launcher tests, literal argument round trips, and native/owned execute,
dry-run, no-op and early-error fixtures. Check redirected output, `NO_COLOR`, batch sequencing and dynamic
checkpoint states. Handler timing changes require the affected lifecycle contract cases under the
[validation policy](dev-validation.md). Numerical checks are selected only for an actual computational delta.
