---
title: "Software environment and dependencies"
format: html
---

Pixi manages the Python/R environment through `pixi.toml` and `pixi.lock`. Julia is managed by juliaup and
must
be available on `PATH`; `Project.toml` and `Manifest.toml` describe the Julia project. Do not install Julia
into Pixi or edit generated locks by hand. Use the [installation guide](../guide/install.md) for setup.

## Roles of Python, R and Julia {#language-boundaries}

| Runtime | Operations | Authoritative files |
|---|---|---|
| Python | Runners, configuration, HydroMT preparation, reductions and plots | `pixi.toml`, `pixi.lock` |
| R | `weathergenr` and its generation scripts | `dev/scripts/install_weathergenr.R`, Pixi tasks |
| Julia | Wflow execution | `Project.toml`, `Manifest.toml`, `config/advanced_settings.yml` |
| Snakemake | Workflow scheduling | Five root `.smk` entry points and included rule files |

`config/advanced_settings.yml` declares required software versions and default resources. Its closed schema is
implemented
in `blueearth_cst/shared/advanced_settings.py`. `tests/test_julia_runtime.py` checks Julia pin consistency.
The installation script specifies the installed weather-generator revision.

Run supported commands with `pixi run`. Snakemake Python steps read `snakemake.input`, `output` and `params`.
R workflow scripts use `Rscript --vanilla` and positional arguments. netCDF carries gridded climate and
forcing between languages. Hydrological simulation retains rounded per-variable CSV responses with explicit
interpretation metadata.

## Data access and environment records {#availability-and-provenance}

Installing the runtime does not install the external forcing or model datasets. Catalog roots, network
access, storage and project write access are separate prerequisites. Successful runtime checks do not
establish data coverage, scientific suitability or a qualified model.

Configuration snapshots and collection/simulation/indicator records capture environment and source
identities for each processing stage. These identities detect incompatible reuse; they do not recreate
an environment on another machine. Projection analysis bundles likewise record lock hashes and toolbox
revision without
installing dependencies. See [execution records](execution-and-provenance.md) and
[results transfer](results-transfer.md).
