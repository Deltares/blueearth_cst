# Task Brief — P0: pin Snakemake lifecycle-handler behaviour

### Context

`AGENTS.md`; design `design-v3.md` §5.2 (implementation step 0), §5.7.

- The design moves journal emission out of rule `X.01` into workflow-level
  `onstart:` / `onsuccess:` / `onerror:` handlers, because a rule that is up to
  date does not execute and therefore cannot record an invocation.
- The whole journal contract rests on handlers firing when no job runs. That is
  an assumption about the pinned Snakemake, not a fact yet.
- Snakemake version is pinned via `pixi.lock`; run under `pixi`.

### Goal

Establish, by observation on the pinned Snakemake, whether the three handlers
fire on (a) a normal invocation, (b) a "Nothing to be done" no-op, (c) a failed
invocation, and (d) `--dry-run`.

### Non-goals

No production code. No changes to any Snakefile that survives this phase.

### Allowed scope

- **Permitted:** a throwaway probe Snakefile and config under the scratch dir
  (`.tmp/`, gitignored), or a temporary handler block in a scratch copy.
- **Forbidden:** the three real `Snakefile_*`; anything under `project_dir`.

### Required changes (checklist)

1. Minimal probe workflow with all three handlers appending a line to a file.
2. Run it four ways: fresh, again unchanged (no-op), with a deliberately failing
   rule, and with `--dry-run`.
3. Record which handlers fired in each case, and the Snakemake version.

### Validation

Rung 1 only — this phase *is* an experiment.

**Falsifier for the design's claim** ("handlers fire on every non-dry
invocation"): a no-op run that appends **no** terminal line disproves it. That
is the observation to hunt for, not to hope against — design §5.2 says the
contract is the terminal `onsuccess`/`onerror` line, so a missing `onstart` on
no-op is tolerable while a missing terminal line is not.

### Acceptance criteria

- A short report naming the Snakemake version and a 4×3 fired/not-fired table.
- An explicit verdict: does design §5.2's journal contract hold as written?
- If it does not: **STOP at Gate 1** (master brief) and report. Do not design a
  workaround.

### Output requirements

`p0-probe-result.md` in this directory: version, table, verdict, and the exact
commands run.

### Task constraints

Delete the probe workflow afterwards; it is not a deliverable. Nothing from this
phase is committed except the report.
