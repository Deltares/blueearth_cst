# Archived workflow configs

Single-workflow `--configfile` targets for climate projections against sources
other than the CMIP6 setup the live configs use. Parked here on 2026-08-10:
nothing in `tests/`, `scripts/`, or any Snakefile referenced them, and they had
drifted out of step with the configs that are exercised on every run.

**These are unmaintained examples, not supported entry points.** They are kept
because an ISIMIP3 or CMIP5 run is a reasonable thing to want and rewriting one
of these from scratch is harder than repairing it. Expect to reconcile a config
here against `../snake_config.template.yml` before using it — the R01 sectioned
schema has moved since these were last run, and a stale key fails at parse time
rather than being ignored.

| File | Source |
| --- | --- |
| `snake_config_projections_cmip5_full.yml` | CMIP5 |
| `snake_config_projections_cmip5_full_linux.yml` | CMIP5, Linux catalog paths |
| `snake_config_projections_cmip6_full.yml` | CMIP6, full model set |
| `snake_config_projections_isimip3.yml` | ISIMIP3 |
| `snake_config_projections_isimip3_linux.yml` | ISIMIP3, Linux catalog paths |

The `_linux` variants differ from their siblings only in the data-catalog path
(`config/catalogs/*_linux.yml`); a 2026-07-25 review measured two of them as a
one-line difference. Fold them together rather than maintaining both if any of
these is ever restored to active use.

## What still covers them

`tests/test_gridded_outputs_removed.py` walks this directory with `rglob`, so a
config here is still checked for the removed `save_grids:` / `save_gridded:`
keys. Nothing else validates them — in particular no test parses them against
the current schema, which is why the reconciliation above is on you.

## Live configs

`../snake_config.template.yml` is the clean starting point;
`../../../test_case/snake_config_baseline.yml` is the tracked baseline seed and
the worked example. Neither is a single-workflow config: both carry a `workflows:`
section
and drive `scripts/run_workflows.py`, which the files here deliberately do not.
