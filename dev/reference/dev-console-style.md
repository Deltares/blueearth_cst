# Workflow console style

Last updated: 2026-11-14

Use this reference when changing workflow orientation, planning, launcher output or rule logs. The shared
[`console_style.py`](../../blueearth_cst/shared/console_style.py) owns the layout for all five workflows;
[`snakemake_console.py`](../../blueearth_cst/shared/snakemake_console.py) owns the early terminal adapter.
Rule implementations retain their compact `log_row` output. Vendored `dev/scripts/console.py` is unrelated.

## Workflow blocks

Each workflow opens with a heavy full-width band carrying its identity (`━━ wf1 build_model ━━…`, bold
cyan) and closes with a light band carrying its verdict (`── wf1 build_model completed in 0:03:02  |
5 warnings ──…`, green, or red on failure), so workflow boundaries stand out in a batch. Bands span the
terminal up to 100 columns. The opening order is the band, **Snakemake command**, applicable
**Input checks**, **Work plan**, **Run settings**, **Paths**, then the labelled **Execution log**
boundary, which the first job row follows directly. Section headings are cyan; scheduled plan rows are
green, reused rows gray, and attention notices amber. Explicit labels preserve meaning without color.
Existing live rule progress, failure detail and warning severity remain intact.

The command prints once, and its `-c`/`--cores` states the workflow's cores; the batch runner states them
once for all workflows. Arguments keep argv order and are quoted only outside a shell-safe whitelist
(letters, digits and `_ . / : = + -`, plus `\` in PowerShell); PowerShell quotes with single quotes and
doubled apostrophes and adds the `&` operator only when the program itself is quoted; POSIX uses shell
quoting. `--configfile` and everything after it go on one continuation line (backtick in PowerShell,
backslash in POSIX), so the command stays pasteable. Arguments are never reconstructed from display text or
Python representations. Owned launches supply the finalized engine arguments after configuration capture,
including the real execution YAML, with credential-like values redacted under the same policy as invocation
records. **Paths** separately names the user's source config. The command is diagnostic evidence; its
temporary execution inputs may be removed after the invocation.

The work plan labels status, rule and jobs; each rule's RUN and DONE rows carry its step number. `run`
means scheduled work; `reuse` means a current rule, whose previous job count is unknown. `off` means a
declared rule whose outputs this run's config does not request, such as a disabled figure switch; register
it with `enabled=False` on `RuleRegistry.logged`/`banner_only`, or it reads as `reuse`. A rule that does not
apply at all stays unregistered and is not listed. `done` identifies completed work before the DAG.
A rule behind a checkpoint (WF4's 4.07/4.08) is absent from Snakemake's opening job table but still runs in
the invocation, so it lists as `run` with its planned job count (`?` when none is declared), and the summary
counts those jobs. A full WF4 run is two Snakemake passes, simulations then indicators, so temp() files go
early; `simulation_phases` tags each pass (`CST_SIMULATION_PHASE`) and the console presents one run. The
first pass plans 4.07/4.08 and hides its `simulations_only` target lines. The second prints no opening block
and continues the `[n/N]` counter through the `CST_SIMULATION_PROGRESS` handoff file.

The summary line counts jobs, then rules by status (`12 jobs  |  3 scheduled  |  3 up to date  |  1 off  |
3 completed`). It excludes target collectors such as `all`; unknown or unlisted work is reported rather than
silently counted as reused.

## Planning and no-op runs

A dry run prominently states **DRY RUN — no jobs will execute** before its plan and ends the header with
**Planning complete**. It emits no live RUN rows or Execution log boundary. Scheduling a fetch during a dry
run does not prove remote availability. A no-op prints only the workflow's opening band, the actual command
and **Up to date — no jobs scheduled**, plus any independently relevant warnings.

The batch runner opens with the same flat sections: **Run settings** (project, cores per workflow, basin,
experiment when known, toolbox), **Paths** (`config`, `<project>`, `repository`) and **Workflow sequence**.
The project is named by [`project_label.py`](../../blueearth_cst/shared/project_label.py), which DAG render
filenames share: an explicit top-level `project_name`, else the case folder of a `create_case.py` trial,
else the `project_dir` basename. Child blocks own their complete workflow identity and finalized commands.
`CST_RUN_ANNOUNCED` remains child-scoped and must not leak into the calling environment or suppress the
child's identity. Verbose invocations retain native startup output.

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

## Rule logs

Each logged rule writes its own log part; the launcher merges the parts after success or failure.

- [`run_logged.py`](../../blueearth_cst/shared/run_logged.py) is the tee wrapper for `shell:` rules:
  `python -u {run_logged} {log} -- <command> [args...]`. It streams combined output to the console and the
  log and returns the command's own exit code, which `<cmd> | tee {log}` masks under `cmd.exe`.
  Python scripts that run a subprocess use `run_and_tee` from `snake_utils.py` instead.
- [`run_log_core.py`](../../blueearth_cst/shared/run_log_core.py) holds the neutral log and console
  helpers shared with WF3; WF4's Wflow frame relay is injected by `snake_utils`, so this module does not
  import that adapter.
- [`merge_logs.py`](../../blueearth_cst/shared/merge_logs.py), called by `execution_outputs.py`, merges
  parts into one workflow log in `LOG_RULES` order, with one provenance header, one section per rule
  and one sub-header per fan-out member. It discovers parts only for listed rule labels, so an orphan
  part from a renamed rule is never read or deleted, and it reports a rule without a part instead of
  skipping it. Raw parts are retained after a failure.

The [workflow map](toolbox-workflow-map.md#trace-a-rule-and-its-consumers) owns rule labels and
`LOG_RULES` registration. Some of these helpers sit near the code inventories that WF3 and WF4
identities hash; check with the [run identity reference](toolbox-run-identity.md#code-inventories) before
moving code between them. Tests: [`test_run_logged.py`](../../tests/test_run_logged.py),
[`test_merge_logs.py`](../../tests/test_merge_logs.py) and
[`test_log_rules_contract.py`](../../tests/test_log_rules_contract.py).

## Verification

Use the focused console/plan/launcher tests, literal argument round trips, and native/owned execute,
dry-run, no-op and early-error fixtures. Check redirected output, `NO_COLOR`, batch sequencing and dynamic
checkpoint states. Handler timing changes require the affected lifecycle contract cases under the
[validation policy](dev-validation.md). Numerical checks are selected only for an actual computational delta.
