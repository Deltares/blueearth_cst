---
title: Snakefile parse imports the heavy stack via script modules
type: todo-item
status: backlog
effort: 2
area: performance / workflow parse
origin: test-runtime profiling (2026-08-20)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Five script: modules are imported at Snakefile parse time and each pulls hydromt, geopandas or xarray. Defer those heavy imports into the functions that use them, so a parse costs ~0.1s of our code instead of 8-19s.
> **Why** — 78-89% of every dry-run is our own parse, not Snakemake's 2.35s startup. A script: module is executed by Snakemake in its own process at rule runtime, so importing it at parse buys the heavy stack for nothing -- on every dry-run, every real run, and every one of the 11 test_cli dry-runs.
> **Effort** — large

## Progress

- [ ] <first step>

## Measured, not inferred — 2026-08-20

Wall-clock, `snakemake.exe` invoked directly so no `pixi run` wrapper is counted:

| dry-run | total | snakemake startup | ours |
|---|---|---|---|
| `analyze_climate.smk` | 10.70s | 2.35s | **8.35s** |
| `build_model.smk` | 15.01s | 2.35s | **12.66s** |
| `analyze_projections.smk` | 21.40s | 2.35s | **19.06s** |
| `run_stress_test.smk` | 11.29s | 2.35s | **8.94s** |

`import blueearth_cst.shared.snake_utils` costs 0.11s over bare Python, so the
shared module is not the problem.

`cProfile` on the WF2 dry-run:

```
analyze_projections.smk:1(<module>)           20.49s
  get_stats_climate_proj.py:1(<module>)       14.27s
    hydromt data_catalog.py:1(<module>)        8.65s
    yaml.safe_load  (x10)                      5.63s
```

## The five modules

`projections/get_stats_climate_proj` · `projections/get_change_climate_proj` ·
`climate_analysis/climate_figures` · `climate_analysis/compare_sources` ·
`shared/plot_spatial_maps`

## Why the import cannot simply be dropped

`analyze_projections.smk:333` imports `get_stats_clim_projections` so
`REDUCER_KERNEL` can hold the FUNCTION OBJECT. `kernel_hash` hashes the
behaviour of the functions it is given and follows no call graph, so the
enumeration is what stops a changed weighting from being silently reused
across the series cache. The object is required; the heavy stack is not.

## Two things the fix must handle

1. **`import hydromt` is side-effecting** — it registers the xarray `.raster`
   accessor. Deferring it means guaranteeing it runs before any `.raster`
   access, not merely before the function returns.
2. **Deferring an import changes the function's bytecode, so `kernel_hash`
   changes**, forcing one re-derivation of the WF2 series cache. The Snakefile
   calls that the safe direction, but on an existing project it is a real cost
   and should be stated in the change, not discovered.

## Expected payoff

`test_cli` ~147s toward ~40s, and the same saving on every real `snakemake`
invocation. Verify by re-running the four dry-runs above and comparing the
`ours` column.

