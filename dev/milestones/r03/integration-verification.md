# R3 — deferred integration verification (before seal)

> **COMPLETED 2026-07-19.** Ran a full `--forceall` WF1 rebuild into
> `examples/test_local`: `check_baseline` 14/14, every per-rule log written
> (silent scripts leave 0-byte logs, as expected), all `benchmark` TSVs
> present, `outlet_index.csv` and the structured sentinel correct, and
> `plot_map` wired + refactored without moving `basin_area.png`. R3 sealed
> and tagged `r03-model-builder`. Retained below as the executed record.

R3 is **code-complete but not sealed**. Per the 2026-07-19 decision to defer
the build batch, several changes are verified only by dry-run + hermetic unit
tests; their *runtime* behavior needs one real model build/run before the
milestone is tagged. This is the checklist for that session.

## Run

Run workflow 1 end-to-end into a **fresh, dedicated project dir** (never the
tracked `examples/test_local` baseline dir — the in-place `staticmaps` mutation
under `ancient()` inputs can otherwise bless stale artifacts):

```
snakemake all -c 3 -s Snakefile_model_creation \
  --configfile config/snake_config_model_test.yml \
  --config project='{"project_dir": "<fresh_tmp_dir>", ...}'   # or a temp config copy
```

(Simplest: copy the test config, repoint `project.project_dir` to a clean dir.)

## Verify

1. **Script-rule logs are actually written** (the deferred bit of commit 4b).
   Under `{project_dir}/logs/` expect non-empty logs for every wired script
   rule: `prepare_build_config`, `add_reservoirs_lakes_glaciers`,
   `add_gauges_and_outputs`, `write_outlet_index`, `setup_runtime`,
   `plot_results`, `plot_forcing` — plus the shell rules `create_model`,
   `add_forcing`, `run_wflow`. Confirm a rule that raises still writes its
   traceback to its log (tee_to_log re-raises).
2. **`benchmark:` TSVs** land under `{project_dir}/benchmarks/` for every
   non-trivial rule.
3. **`outlet_index.csv`** (`{basin_dir}/staticgeoms/outlet_index.csv`) exists
   with columns `station_name,subcatchment_id,x,y`, one row per outlet,
   `station_name = wflow_{1..N}`, real subcatchment IDs in `subcatchment_id`.
4. **Structured sentinel** `reservoirs_lakes_glaciers.txt` is the TSV format
   (`method\tstatus\treason` header + one row per method), not the old
   free-text repr.
5. **Baseline** — `check_baseline check` reports **14/14** (R3 is
   behavior-preserving). If any of the 3 workflow-1 PNGs drift, treat as
   advisory (size-only fingerprint) and inspect; no manifest edit expected.

## Remaining code item

- **`src/plot_map.py` tee-wiring is NOT done.** It is a bare top-level script
  (runs at module top referencing `snakemake` directly), so it needs a small
  refactor to a guarded `if "snakemake" in globals()` entry before it can be
  wrapped in `tee_to_log`, the same shape as the other scripts. Its `log:`
  directive is already declared (commit 4) but stays empty until wired. Do
  this in the integration session and verify its log.

## Then seal (commit 11)

- Update `dev/roadmap.md` R3 section to **sealed** with final suite counts.
- Update `dev/branches-and-tags.md` with the `r03-model-builder` branch/tag.
- Tag `r03-model-builder`.

## Not part of R3 (do not do here)

- CSDMS constant-parameter restoration and its baseline move → task
  `t260719a` (adds `staticmaps.nc`/TOML fingerprints; carries the scientific
  decision).
- Any workflow-2 / workflow-3 content (R4 / R5); the workflow-3
  `CyclicGraphException` ratchet in `tests/test_cli.py` stays.
