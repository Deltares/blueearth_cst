# M1 warnings triage

Per the roadmap, M1 exit criteria require buckets 2 (config / data-catalog)
and 3 (our-code) to be empty. This document records the M1 triage as
actually performed and is **not exhaustive** — see "Coverage gap" below.

## Bucket 1 — upstream (accept)

Frame originates inside `site-packages/`, vendored third-party paths, or
external tools (hydromt, xarray, R packages, Julia, Wflow.jl).

- **`hydromt model_api`: "Model dir already exists and files might be
  overwritten: ...staticgeoms"**. Emitted on every `hydromt update wflow`
  invocation that runs after the initial build. Source:
  `examples/test_local/hydrology_model/hydromt.log:134`. Hydromt's
  `update` mode is intentionally permissive about overwriting; the
  warning is informational and unavoidable in the current pipeline shape.
  Accept.

## Bucket 2 — config / data-catalog (fix in M1 if cheap)

*No entries captured.* See coverage gap below — this should not be read
as "no such warnings exist."

## Bucket 3 — our-code (fix in M1)

*No entries captured.* Same caveat as bucket 2.

## Coverage gap

This triage relied on log files written to disk by the workflow rules.
Only `hydromt`'s build/update steps produce a persistent `.log` file
(`examples/test_local/hydrology_model/hydromt.log` and
`.../run_climate_experiment/hydromt.log`). Every other rule —
`run_wflow`, `generate_weather_realization`, the `src/*.py` scripts
invoked via `script:`, the R weathergen scripts invoked via `shell:` —
emits stderr to the terminal and is not captured anywhere reusable.
Snakemake's own logs (`.snakemake/log/*.snakemake.log`) record rule
orchestration only, not rule stderr.

The roadmap-defined remedy is the M3 cross-cutting deliverable
"per-rule `log:` and `benchmark:` directives on every non-trivial rule"
— once that lands, every rule's stderr is captured to a file under
`logs/` (or wherever the convention puts it), and a comprehensive
triage becomes mechanical.

## Decision and follow-up

- Accept the M1 closure with this incomplete triage rather than
  blocking on the M3 prerequisite. The risk is that real bucket 2 or
  3 warnings exist and aren't being addressed at M1. That risk is
  judged acceptable because M2 (env swap) and M3 (workflow-1 cleanup)
  will both surface and capture warnings as part of their own work.
- A follow-up entry is added under M3 in `dev/followups.md` to
  redo this triage exhaustively once `log:` directives are in place,
  and to fix any bucket 2/3 entries that surface then.

---

## Exhaustive re-triage — 2026-07-21 (t260716a′, `fix/pre-r6-followups`)

The coverage gap is now closed: R3/R4/R5 added per-rule `log:` directives to
every non-trivial rule across all three Snakefiles, so rule stderr is captured to
disk. Swept **82 `.log` files** under `examples/test_local/logs/` (all three
workflows: wf1 from the 2026-07-21 restored `--forceall` rebuild, wf2/wf3 from the
2026-07-20 R5 seal run) plus the two hydromt `hydromt.log`s. Method: tool-aware
grep for Python (`FutureWarning`/`DeprecationWarning`/`UserWarning`/
`RuntimeWarning`), R (`Warning message`), Julia/Wflow (`┌ Warning`, `Warning:`),
hydromt `WARNING`, and tracebacks.

**Bucket 3 (our-code): EMPTY (verified).** No warnings whose frame is in `src/*.py`,
the Snakefiles, `dev/scripts/`, or the R weathergen scripts. No Python/R/Julia
warnings surfaced from our code in any captured log.

**Bucket 2 (config / data-catalog): one, intended — WON'T FIX.**
- `basemaps - WARNING - Model resolution 0.00833 does not match the hydrography
  resolution 0.008333333333325754 ... using hydrography resolution instead`. The
  config's `shared.basin.resolution: 0.00833` is a human-readable truncation of
  1/120°; hydromt snaps the model grid to the native hydrography resolution, which
  is the **desired** behavior (the model grid should match the source grid). The
  built model already uses the native resolution regardless of the config value.
  Not fixed: matching the config to the full-precision float is fragile
  (source-specific literal, float-equality-sensitive) and would drift the tracked
  `snake_config_model_creation.yml` manifest fingerprint → a baseline re-record for
  **zero** model change. Documented as understood.

**Bucket 1 (upstream: accept).**
- `states - WARNING - CRS not found in states data, setting to model CRS` (×3) —
  hydromt applies the model CRS correctly. Benign.
- `forcing - WARNING - Write forcing skipped: dataset is empty` (×3) — the build
  step writes no forcing (forcing is added by the separate `add_forcing` rule).
  Benign.
- `Model dir already exists ...` on `hydromt update` steps — the original bucket-1
  entry above; still applies.
- **NEW (now captured): `Error in sys.excepthook:` / `Original exception was:`
  cascade (×62) at the tail of `create_model.log`.** Empty-body Python
  interpreter-shutdown noise from the `hydromt build wflow_sbm ... -vv` subprocess,
  emitted **after** a successful build (rc=0; `staticmaps.nc` + all downstream
  targets produced; `check_baseline` 5/5). The frame is inside the hydromt
  subprocess at shutdown, **not** our tee wrapper: `src/run_logged.py` faithfully
  captures the child's combined output, and the cascade is **absent** from
  `run_wflow.log` (Julia) and the hydromt `update`-step logs, which use the same
  wrapper — so it is specific to the verbose hydromt build's shutdown, not a
  wrapper defect. Cosmetic. **Resolved (test/pre06):** `run_and_tee`
  (`src/snake_utils.py`) now collapses a *pure* trailing run of these empty-bodied
  markers into one `[run_logged] collapsed N ... child rc=<rc>` summary line. The
  filter is conservative — a real traceback puts non-empty content between the
  markers, breaking the candidate block so it is emitted verbatim; only the
  benign empty-bodied cascade is collapsed (`tests/test_run_logged.py`). Still a
  candidate for an upstream hydromt report (Windows `-vv` shutdown excepthook
  noise) since the root cause is inside the hydromt subprocess.

**Conclusion.** Buckets 2 (actionable) and 3 are empty of real defects; the single
bucket-2 item is intended hydromt behavior. The M1 coverage gap is closed. No code
changes result from the sweep — its value is the now-exhaustive, captured-log
verification.
