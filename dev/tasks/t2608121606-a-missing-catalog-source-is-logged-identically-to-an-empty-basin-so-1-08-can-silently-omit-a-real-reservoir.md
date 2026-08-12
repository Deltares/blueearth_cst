---
title: A missing catalog source is logged identically to an empty basin, so 1.08 can silently omit a real reservoir
type: watch-item
area: wf1 / data catalog
origin: 2026-08-12 t2608091730 investigation
created: 2026-08-12
updated: 2026-08-12
---

> [!note] Overview
> **What** — hydromt logs the RESOLVED URI before opening a source, then reports a missing file as 'Skipping method, as no data has been found' -- the same line an empty clip produces. Rule 1.08 add_reservoirs_lakes_glaciers therefore exits 0 whether the basin has no reservoirs or the reservoir dataset is absent from the machine.
> **Why** — On a real basin with a major reservoir, a missing or misconfigured catalog entry yields a wflow model with no reservoir, no warning, and a green run. The stress test then routes water through a basin whose largest control structure is not represented. It also defeats log-based evidence: a 'Reading X from Y' line does not establish that Y exists, which is how t2608091730 recorded a blocker three weeks newer than it was.
> **Trigger** — Any basin whose results look implausible around a known reservoir, lake or glacier; or a model build on a fresh data root. Cheap check: assert the catalog's resolved URIs exist before the build, rather than trusting the run's exit code.

## The observation

Today's WF1 run in the primary (`test_case/snake_config_model_test.yml`,
2026-08-12), rule 1.08, verbatim:

```
13:28:05 - data_source - INFO - Reading hydro_reservoirs GeoDataFrame data from C:\data\wflow_global\hydromt\hydrography\reservoirs\reservoir-db.gpkg
13:28:06 - wflow_sbm  - INFO - Skipping method, as no data has been found
13:28:06 - data_source - INFO - Reading hydro_lakes GeoDataFrame data from C:\data\wflow_global\hydromt\hydrography\lakes\lake-db.gpkg
13:28:06 - wflow_sbm  - INFO - Skipping method, as no data has been found
13:28:06 - data_source - INFO - Reading rgi GeoDataFrame data from C:\data\wflow_global\hydromt\hydrography\rgi\rgi.gpkg
13:28:06 - wflow_sbm  - INFO - Skipping method, as no data has been found
```

None of those three files exists. `hydrography/` holds only `rivers_lin2019`
and has a LastWriteTime of 24 July; a recursive search of `C:\data` finds no
`reservoir-db.gpkg`, `lake-db.gpkg` or `rgi.gpkg` anywhere on the machine. The
rule still exited 0, the model TOML records `glacier = false`, and
`staticgeoms/` carries no waterbody layers — which is exactly what a basin with
no reservoirs would produce.

**The fixture basin genuinely has none, so nothing here is wrong today.** That
is the point: the fixture cannot tell the two cases apart either, so no test in
this repo can catch the difference, and a green suite says nothing about it.

## Why this is upstream behaviour and not ours to patch

`AGENTS.md` is explicit that hydromt's data handling is not re-engineered here.
The workaround, if this ever bites, belongs in our own code: a pre-build
assertion that every catalog entry the build will read resolves to a file that
exists. That is cheap — the catalog is already parsed — and it converts a silent
INFO line into a hard failure naming the missing source.
