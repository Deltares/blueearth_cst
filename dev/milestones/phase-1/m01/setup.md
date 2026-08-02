# M1 setup

Provenance and reproduction notes for the M1 replication baseline.
Windows-only by intent — Linux/Docker replication is deferred (see
`roadmap.md` → "Deferred: Linux replication").

## Test-data provenance

- **Where it lives.** All input data the workflows consume on this
  machine sits at `C:/data/wflow_global/hydromt`. This is a
  bbox-clipped local mirror of the upstream Deltares data root
  (`P:/wflow_global/hydromt`), produced by `dev/scripts/stage_data.py`
  (config in `dev/scripts/stage_data.yml`, bbox `[8.5, -0.5, 11.0, 1.5]`).
  The matching data catalog is `config/deltares_data_local.yml` (its
  `meta.root` points at `C:/data/wflow_global/hydromt`). Climate
  projections come from `config/cmip6_data.yml`, which reads CMIP6
  directly from its configured source (no staging needed).
- **How a fresh clone gets it.** Not yet automated. Two known paths:
  (a) obtain access to the Deltares P-drive and run
  `python dev/scripts/stage_data.py` once with P: mounted to populate
  `C:/data/wflow_global/hydromt`; (b) copy the staged
  `C:/data/wflow_global/hydromt` tree from a machine that already has
  it. P-drive access is not currently set up on this machine, so the
  staged C: copy *is* the source of truth here. Productionizing
  fresh-clone setup is a separate problem tracked under deferred
  Linux/Docker work.

## Configs in use

| Snake config | Data catalog | Data root | Status |
|---|---|---|---|
| `config/snake_config_model_test_local.yml` | `config/deltares_data_local.yml` | `C:/data/wflow_global/hydromt` | **canonical for M1 on this machine** |
| `config/snake_config_model_test.yml` | `config/deltares_data.yml` | `P:/wflow_global/hydromt` | reference / collaborators with P: mounted |

`project_dir` resolves to `examples/test_local/` for the local config
(gitignored).

## Reproducing the M1 baseline

```powershell
conda activate cst
.\run_snake_test.cmd
```

`run_snake_test.cmd` activates the env, generates DAG PNGs, unlocks
each Snakefile, and runs the three workflows in order against
`config/snake_config_model_test_local.yml`. To run against the
P-drive catalog instead (assuming P: is mounted), invoke `snakemake`
directly with `--configfile config/snake_config_model_test.yml`.

## External dependencies not in the conda env

- **Wflow.jl.** Currently installed via Julia package manager outside
  the conda env. Pinned by the Dockerfile only; M2 will surface the
  pin in the declarative env file.
- **weathergenr (R package).** Pinned to tag `v1.2.0` of
  `tanerumit/weathergenr` and installed at runtime via
  `dev/scripts/install_weathergenr.R`. M2 will move this into the
  declarative env or vendor it.

## What was hand-tweaked to get M1 green

- `historical: 2000, 2020` in the local snake config (the canonical
  config still says `1980, 2010` — staged era5 starts in 2000, and the
  weathergenr wavelet decomposition needs ≥ 16 historical years). Both
  the silent truncation and the hardcoded date range are tracked in
  `dev/followups.md` for resolution in M3 / M5.
- weathergenr `spatial_ref` attribute workaround in
  `src/weathergen/generate_weather.R` — see commit 75d455f and the
  weathergenr items in `dev/followups.md` (M5 section).
