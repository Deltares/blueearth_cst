---
title: "Run records and reproducibility"
format: html
aliases:
  - execution-and-provenance.html
---

Every supported run command writes a readable summary at
`<project>/logs/<UTC timestamp>__<workflow scope>/summary.md`. An all-workflow run has one parent summary
covering its explicitly linked children; a standalone workflow has its own summary.
Different attempts keep separate folders. The run command's closing output links to the report.
The run record JSON is authoritative when a report is missing or stale.

## Run records and saved configurations {#invocation-hierarchy-and-configuration-captures}

Each run command attempt writes a record to `<project>/config/_engine/invocations/<id>.json`.

### Record contents {#record-identity}

Each record names the workflow, entry point, command, targets and working directory. Workflow records from
an all-workflow run link to their parent record. All-workflow children share their parent's execution folder;
standalone launches allocate their own.

### Status fields

- `mode` distinguishes `execute` and `dry_run`.
- `status` starts as `running`, then becomes `succeeded` or `failed` on observed completion.
- `exit_code`, `error`, `started_at_utc`, `ended_at_utc` and `elapsed_seconds` describe that attempt.
- `work_performed` is `no` for dry runs and otherwise remains `unknown` without evidence.

Disabled/not-invoked workflows are report classifications, not successful computations. Launch failures
remain linked explicitly. Credential-like command values are sanitized; records are not environment-variable
dumps.

### Saved configuration

`configuration` records hashes of the source, effective and input files, `run_record` and archive
availability.

- Historical climate analysis, model building and projection analysis latest captures live in
  `config/<workflow>/run_record.yml` and `sources/`; saved configuration snapshots live in
  `config/_engine/snapshots/<workflow>/`.
- A snapshot of the current generation holds only its `run_record.yml`, because the readable copy holds its
  sources; a superseded generation's snapshot holds its sources, which are then their only copy.
- Scenario generation and hydrological simulation captures belong to their collection/experiment.

Edit source YAML, never generated archives. Archives preserve exact source bytes, resolved paths and
relevant input references; recording them does not copy external datasets.

Generated execution YAML uses the current public climate keys, including
`climate.forcings.<name>.precip_source` and `climate.forcings.<name>.temp_source`.

### Execution folder

The folder contains `summary.md`, `console.log`, `wfN.log`, `wfN-benchmarks.md` and `.tsv` where available,
with job diagnostics in `_parts/wfN-<child-id>/`.
Hydrological simulation adds `wf4-indicators.log`. Console counters and final verdicts aid monitoring;
run records and ready product records answer different questions about execution and product validity.

## Run commands and supported options {#launcher-arguments-and-safety-boundaries}

The argument owners are `scripts/run_workflows.py`, `scripts/run_workflow.py` and
`scripts/simulate_system.py`. Use the [running guide](../guide/running.md) for operating procedures.

| Argument | Contract |
|---|---|
| `--config` | Required project YAML, naming workflow files |
| `--project-dir` | Output directory used to record start-up failures; must equal the resolved `project.project_dir` |
| `--cores` | Positive integer, default 3 |
| `--new-experiment` | Request the next experiment allocation |
| `--note` | Optional experiment note on the all-workflow and hydrological simulation run commands |
| `--recover-reservation` | Request guarded reservation recovery on those same run commands |
| `--dry-run` | Direct hydrological simulation option; all-workflow execution forwards it after `--` |
| `--forceall` | Direct hydrological simulation option or forwarded execution flag; does not bypass checks of the recorded inputs |

### Single-workflow command

`run_workflow.py` requires workflow name, `--config` and `--project-dir`; its `--target` can repeat and it
supports `--dry-run` and `--keep-going`. Scenario generation and hydrological simulation are routed to their
supported run commands. Routine scenario generation execution uses the all-workflow run command with only
scenario generation enabled. Named hydrological simulation requests infer the target from operations; omit
explicit `--target`/`--simulation-target` overrides.

### All-workflow command

The all-workflow run command executes enabled workflows in fixed order and stops at the first workflow
failure. Hydrological simulation can continue independent sets/batches internally. Forwarded hydrological
simulation execution flags are limited to an allowlist; arbitrary config or target overrides are rejected. Use
`--help` to inspect supported flags rather than assuming every Snakemake flag can be forwarded.

### Unlock and rebuild

- `--unlock` after `--` unlocks enabled workflows without generation, execution-source capture or writing
  indicators. It does not validate products or repair mismatches with the recorded scientific inputs.
- The all-workflow `--rebuild` permanently erases generated namespaces, experiments and allocations,
  including disabled workflows' outputs, before the usual input checks and execution. It requires default-no
  interactive confirmation or `--yes`. `--rebuild -- --dry-run` previews that reset without writing records
  or invoking workflows. Source files, unrelated paths, logs and attempt history survive.

### After a failure

After a failed or interrupted attempt, existing files remain evidence rather than proof of readiness. A
mismatch with the recorded inputs calls for a separate experiment; forced scheduling never makes incompatible
inputs equivalent. Reservation recovery is guarded; it is not a general overwrite switch, nor a repair tool
for partial weather generation or damaged model-response files.

## Run status and elapsed time {#timing-and-attempt-lifecycle}

The source is `<project>/config/_engine/invocations/<id>.json`, not console output or
per-rule benchmark tables. Records keep numeric `elapsed_seconds` alongside UTC timestamps for
chronology. The total covers the recorded run attempt and is not the sum of parallel job durations.
Time before the run record is created is not measured.

Reports distinguish execution from dry runs, disabled workflows, workflows not invoked, launch
failures and unfinished attempts. Warning counts appear only when recorded. Exit zero does not prove
that computation ran, outputs are complete or scientific results are adequate; `work_performed`
remains unknown unless the run command has evidence. A no-op with every workflow disabled records no work.
Records without measured elapsed seconds keep an unavailable duration; UTC differences are not substituted.

A report starts as a running snapshot and refreshes as the run command observes children and completion.
A hard-killed attempt keeps its last snapshot without a fabricated end time. Report-writing failures
are reported separately and preserve the original execution outcome. Structured run records
remain authoritative, including when a report is missing or stale.
Startup failures can be recorded once the project root is known. Supply `--project-dir` to
the all-workflow or hydrological simulation run command to record failures that occur while reading
configuration.

## Workflow benchmark reports

Each invoked workflow writes `wfN-benchmarks.md` and `wfN-benchmarks.tsv` in its execution folder
when measurements are available. The Markdown report shows the recorded workflow elapsed time,
followed by a rule overview: distinct jobs, collected samples, summed job time, median and slowest
job, and peak individual-job RSS. Wildcard jobs are grouped under their rule. Repeated measurements
count as samples; per-job medians determine job-duration statistics so repeats do not inflate the sum.
Expand the detailed table for job identifiers and individual measurements, or use the adjacent TSV
for further analysis.

Summed job time is not workflow elapsed time: parallel jobs overlap and run-command overhead is separate.
Peak job RSS is not peak concurrent workflow memory. Missing or invalid measurements are unavailable,
never zero. Coverage remains unknown unless explicitly recorded; a successful attempt can reuse existing
outputs and collect few or no new benchmark samples. Partial failures keep the collected measurements
without claiming complete workflow coverage. Each retry starts with its own parts and reports.

Hydrological simulation reports also summarize available Julia timings: Wflow loading, compilation in the
first observed
member of each Julia process, and later member wall times. These observations help distinguish startup
cost from sustained simulation speed. Missing phases remain unmeasured, and failed observations are
counted separately. First and later members refer to observations within each process.

## Hydrological simulation resource diagnostics {#wf4-resource-diagnostics}

Resource sampling remains limited to hydrological simulation, automatically through either supported run
command. Dry runs do not start resource monitoring. No external monitoring service is used.

### Finding the diagnostics

Find the hydrological simulation child record in `<project>/config/_engine/invocations/<id>.json`.
Its `resources` field links to `<id>/diagnostics/` and summarizes elapsed time,
sampled CPU core use, peak process-tree RAM, minimum available system RAM and run-command phases.
Use the hydrological simulation child ID, rather than the all-workflow parent ID.

| File | Evidence |
|---|---|
| `resources.jsonl` | Host context, timestamped resource samples, phase boundaries and final summary |
| `batch-*.json` | Batch selection, memory/core/disk constraints, concurrent-batch limit, requested threads and run membership |
| `julia-<pid>.tsv` | Package loading and per-member wall time, allocations, GC and compilation |

### What is sampled

Resource samples are taken approximately every two seconds. They include the hydrological simulation run
command and its descendants, with process IDs, CPU seconds, RSS, logical read/write counters, available system
RAM, swap counters, system CPU load, reported CPU frequency and power-source status.

Julia records its actual thread count; Snakemake can reduce the rule's requested count to fit its cores.
Process affinity and priority are recorded when first observed. Host context includes processor
information, physical/logical CPU counts, RAM capacity, free output-volume space and the Windows
power plan at startup. No process command lines or environment dumps are collected.

### Reading the timings

The run-command phases distinguish preparation, Snakemake execution, finalization and indicator exports.
Within Snakemake, correlate sample timestamps with rule logs and `wf4-benchmarks.md` in the execution
folder.

- Summed job durations are not workflow wall time, and the table does not include all run-command overhead.
- Julia's first member includes its cold compilation costs; later members provide a better comparison of
  sustained simulation speed.
- Compilation and GC timings are components of wall time; do not add them to it.
- Allocated bytes measure allocation traffic, not peak or retained RAM.
- A failed member records wall time with unavailable allocation/GC/compilation fields left empty.

### Measurement limits

Treat these measurements as diagnostic evidence with explicit limits:

- Process CPU and I/O totals keep the last sample of exited processes, but miss children that start
  and exit between samples. Peaks are sampled. Summed RSS may count shared pages more than once.
- The initial process census excludes prior usage of existing processes. Descendants discovered later
  contribute their observed lifetime counters, without comparing process creation and wall-clock times.
  If the initial census is incomplete, first-observed counters are subtracted for all processes;
  this conservatively omits some work, with read failures recorded in the measurement errors.
- System CPU, swap and disk counters include other applications. Process I/O can include cached I/O;
  it does not directly measure physical storage throughput or I/O wait.
- Reported CPU MHz can be nominal on Windows. The trace cannot prove effective clock speed, thermal
  throttling or CPU temperature; those require platform-specific hardware profiling.
- Unavailable measurements appear as `null`; failed reads are counted, rather than reported as zero usage.
  Swap-in/out counters may remain zero on platforms that do not expose them.
- Samples are flushed during execution. A hard termination can leave a partial trace and a running
  run record. Only an observed completion produces a final summary.

### Effect on the run

The summary records sampler collection time so its cost can be inspected. Diagnostic failures emit a warning
and do not change the workflow result. Records belong to the run and do not modify saved model time series or
indicator outputs. Existing runs cannot acquire historical resource evidence retroactively. Resource sampling
does not change what decides simulation reuse: a code change alone does not require a new experiment, but a
changed [method revision](saved-results.md#identities-and-reuse) or key numerical package does.

## Tracing climate-projection results to their source run {#wf2-product-attribution}

Runs launched through `scripts/run_workflows.py` or `scripts/run_workflow.py` keep
`_engine/execution/production.json` at the ensemble root. It names, for every current product file,
the run whose bytes are present, with that run's outcome, configuration snapshot under
`config/_engine/snapshots/analyze_projections/`, and `region.geojson` checksum.
Each contributing run's record lists the paths of its `wf2.log` and `wf2-benchmarks.md`/`.tsv` in
its `logs/` folder; they are referenced, not copied. `dev/scripts/prune_logs.py` never lists a log folder
a product record refers to.

- `state: complete` means the latest run that changed the product succeeded. `incomplete` means it
  failed after replacing some files: tables, figures or caches then come from different runs.
- A run that writes no product file, successful or failed, leaves the record untouched. A file is
  credited when the run changed its bytes or rewrote it identically (detected by write time under the
  project lock), so a partial plotting rerun credits only the figures and figure manifests it rewrote.
- `producer: null` marks bytes no run command observed being written: products from before this record
  existed, direct Snakemake runs, or edits by hand. The record never credits them to a later run.
- The `delineate_region` and `delineate_subbasins_and_rivers` steps run inside projection analysis but write
  the shared `data/spatial/geoms/`. They appear in the copied log and are listed as `shared_steps`; their
  files are not product files.
