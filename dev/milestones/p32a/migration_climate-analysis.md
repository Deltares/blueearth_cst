# P3-2a migration note — climate_analysis module map

P3-2a (2026-07-24) lifted the model-independent climate helpers into the
`blueearth_cst/climate_analysis/` subpackage (design
`dev/milestones/p32a/climate-analysis-design.md` §5.1/§5.4). The old module paths carried
transitional star-import shims during the migration window (commits 1–5) and
were **deleted at milestone close** (commit 6). They were internal package
modules, not a contract surface — no external consumer imports them; the
platform surface (the three Snakefile entry points, `run_workflows.py`,
config schema) is unchanged.

## Old → new module map

| Old path (deleted) | New path | Notes |
| --- | --- | --- |
| `blueearth_cst/experiment/extract_historical_climate.py` | `blueearth_cst/climate_analysis/extract_historical_climate.py` | Moved verbatim; gained keyword-only `bbox=` (P3-2a arch-1; wf3 call unchanged). wf3 rule 3.02 `script:` repointed. |
| `blueearth_cst/model/climate_forcing.py` | `blueearth_cst/climate_analysis/subcatchment_climate.py` | File renamed (OQ-1); function name `climate_forcing_by_subcatchment` kept, signature/algorithm unchanged. |
| `blueearth_cst/projections/prepare_climate_data_catalog.py` | `blueearth_cst/climate_analysis/prepare_climate_data_catalog.py` | Moved verbatim. wf3 rule 3.08 `script:` repointed. |

## New model-side modules (not moves)

- `blueearth_cst/model/extract_climate_wf1.py` — wf1 rule 1.10 `script:`
  wrapper (staticmaps-derived bbox; chirps sidecar relocation; eobs guard).
- `blueearth_cst/model/climate_parity.py` — `model_parity_climate(...)`, the
  build-parity regrid + corrections + PET transform for the re-sourced wf1
  climate plots.

## Import rewrites

`from blueearth_cst.<old> import ...` → `from blueearth_cst.climate_analysis.<new> import ...`
at every live site: `blueearth_cst/model/plot_results.py`, wf3 rules 3.02/3.08
`script:` strings, and the three moved-module test files
(`test_extract_historical_climate.py`, `test_prepare_climate_data_catalog.py`,
`test_climate_forcing.py` — assertions verbatim).
