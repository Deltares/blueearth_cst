---
title: Make the WF3 batch-size default actually disk-aware
type: todo-item
status: done
effort: 2
area: wf3 batching
origin: P3-3
queue: 3
created: 2026-08-07
updated: 2026-08-18
closed: 2026-08-18
---

> [!note] Overview
> **What** — Make the WF3 batch-size default actually disk-aware.
> **Why** — The design names the disk ceiling as the binding constraint, and the default ignores it, so a large run can fill the disk mid-flight.
> **Effort** — Medium: the arithmetic is known, the unknown is how to read available disk portably.

## Progress

- [x] **Hazard re-confirmed with measured numbers**, as the note asked. Against the
  seed fixture the cap is still inert — `min(ceil(12/3), 8) = 4`, peak 53 MB of
  26 GB — so every P3-3 measurement stands. It binds at scale: at `-c 6` with the
  fixture's own 4.6 MB/member, `B` is driven to 3 and then to 1 as the budget
  falls, and warns once even `B=1` overruns. The 2026-07-25 finding is unchanged,
  not superseded.
- [x] **(b), the part the note called hard, is solved by MEASURING rather than
  modelling.** The note assumed the estimate had to come from grid dims × run
  length × variable count, because the member forcing NCs are `temp()` and absent
  at parse time. But WF1 leaves two ordinary persisted artifacts that carry the
  same per-model constants — `<basin>/forcing/inmaps_historical.nc` and
  `<basin>/run_default/outstate/outstates.nc`. Bytes-per-timestep is a property
  of the MODEL (grid, variable count, compression), not of the window, so it
  transfers: measured on a live rapid run, the historical file gives 1379 B/step
  and a real member forcing 1378 — **0.07 %**. A model of the encoded size would
  have had to predict zlib complevel-4 output, which is not predictable; reading
  a file the same writer produced sidesteps that entirely.
- [x] **The state anchor is NOT scaled by run length.** `outstates.nc` is a single
  snapshot (103.6 KB rapid, 106 KB the seed fixture); scaling it would be wrong,
  not merely imprecise.
- [x] **The peak formula is `min(K, cores × B) × per_member`, not `p × B × …`.**
  Once every member of the sweep is resident at once the batch structure has
  stopped mattering and the cap degenerates to the whole sweep's footprint —
  which is exactly the degenerate case GN-3 measured (12 of 12 forcing NCs
  resident at `B=4`/`p=3`). The disk ceiling is therefore only applied when
  `K × per_member` actually exceeds the headroom.
- [x] **(a), the headroom key, is TWO keys with deliberately different units.**
  `defaults.batch_disk_headroom_fraction` (advanced settings, 0.25) is a POLICY —
  never eat more than a quarter of what is free; `workflows.run_stress_test.disk_headroom_gb`
  is a BUDGET — you may use N GB, and wins outright. A single constant generous
  enough for a workstation is a wrong answer on a laptop, so the default had to
  be a share.
- [x] Free disk read via `shutil.disk_usage`, walking up to the nearest existing
  ancestor so a `project_dir` the run is about to create still resolves.

### Guarantees the cap deliberately does NOT have

- It **only ever lowers `B`** — never past the parallelism ceiling, never past
  `batch_size_max`, never above.
- An unavailable estimate (fresh project, WF1 not yet run, `netCDF4` absent,
  unreadable anchor) degrades silently to the previous behaviour. **It never
  raises**: a safety cap that becomes a new failure mode is a worse defect than
  the one it fixes.
- An explicit `batch_size` **wins and is not clamped** — the operator knows
  something the estimate does not, and silently overriding it would make the key
  a lie — but its peak is still computed and warned about, so an overrun is
  visible rather than unchecked.
- `netCDF4` is imported **lazily**, inside the reader. Measured: it is not
  preloaded at WF3 parse (the Snakefile's existing imports cost 3.04 s and leave
  `netCDF4`/`xarray`/`h5netcdf` all unloaded), so a module-scope import would add
  0.75 s to every parse; the read itself costs 9 ms.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**Make the wf3 batch-size default genuinely disk-aware.** Design §6.1 names
three ceilings on `B` and calls the **disk ceiling the BINDING constraint** on
large `RLZ_NUM×ST_NUM` runs, capped so `p × B × (forcing_size + state_size)`
stays inside a stated headroom. The landed default implements only the
*parallelism* ceiling (`ceil(K / -c N)`), which scales `B` **up** with sweep
size and therefore grows peak temp disk as the sweep grows — backwards from
what §6.1 asks. Commit `3392587` bounds it with an overridable
`batch_size_max` (default 8); that caps the blast radius but is a constant, not
a disk computation. A real cap needs (a) a stated disk-headroom config key and
(b) a per-run forcing+state size estimate, and (b) is the hard part: at parse
time the forcing NCs are `temp()` and do not exist yet, so the estimate has to
come from the wflow grid dimensions × run length × variable count, or from a
measured prior run recorded in config. Verified 2026-07-25: fixture (K=12,
`-c 3`) is unaffected — `min(ceil(12/3), 8) = 4`, so every P3-3 measurement
stands; the clamp only binds from K > 24 at `-c 3`. Confirm the hazard still
applies before fixing (it is scale-dependent and invisible on the seed
fixture, whose peak footprint is 120 MB).

## Resolution

`blueearth_cst/experiment/batch_sizing.py` (the estimator) and
`blueearth_cst/experiment/forcing_window.py` (the window arithmetic, moved out of
`downscale_climate_forcing.py` so the Snakefile can reach it without paying that
module's `hydromt_wflow` import at parse time). `run_stress_test.smk` calls
`resolve_batch_size(...)` and reports `B` plus the ceiling that chose it as a
`run_header` row, so a machine-dependent `B` can always name its constraint.

**`forcing_window` is wider than `run_length`, and the estimator uses the window.**
`run_length: 8` at horizon 2050 spans 2046..2054 -- NINE calendar years, 3287
days. `run_length x 365` under-counts by 12 %. Pinned by a test.

40 tests in `tests/test_batch_sizing.py`, including a fixture-backed reproduction
of the measured bytes-per-timestep -- the one that would notice if the anchor
assumption ever stopped holding.

**Residual, stated rather than implied.** The state anchor's transferability is
reasoned, not measured: no WF3 `outstates_*.nc` existed on disk to compare
against `<basin>/run_default/outstate/outstates.nc`. Both come from `Wflow.run()`
on the same model, so they should match, but that is inference. It is also the
small term -- 106 KB against 4.5 MB of forcing -- so an error there moves the
estimate by well under a percent. `p = cores` is likewise an assumption: the
batch rules declare no `threads:`, so Snakemake will run up to `-c N` of them.
