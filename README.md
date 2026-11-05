# Climate Stress Testing toolbox

[![CI](https://github.com/Deltares/blueearth_cst/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/blueearth_cst/actions/workflows/ci.yml)

[Documentation website](https://deltares.github.io/blueearth_cst/)

The Climate Stress Testing toolbox (`blueearth_cst`) provides Python, R, and Julia workflows for
bottom-up climate risk assessments of water systems. It combines historical climate analysis,
hydrological modelling and weather generation to explore basin responses across a temperature ×
precipitation stress-test grid. Hydrological indicators help identify vulnerabilities, assess robustness
and prioritize further investigation. Results support first-order screening; basin-specific decisions
require scrutiny of data, model performance and assessment assumptions.

The stress-test experiment is scenario-neutral: CMIP6 projections
provide a plausibility overlay for the tested changes; they do not select or weight grid cells.

The toolbox uses [weathergenr](https://github.com/Deltares-research/weathergenr) for climate series
and [Wflow](https://github.com/Deltares/Wflow.jl) for hydrological simulation. This is a
[personal fork](https://github.com/Deltares/blueearth_cst) of the Deltares project.

## Get started

1. Follow [Windows installation](docs/guide/install.md) for Pixi and juliaup-managed Julia.
2. Check your project's data catalogs. Installation does not supply the external datasets;
   the example expects Deltares P-drive access and CMIP6 network access.
3. Follow the [quick start](docs/guide/quick-start.md) to run the example basin and inspect its results.
   Its small grid and short simulation period are intended for learning the toolbox.

## Run your own assessment

Follow [Creating a project](docs/guide/new-case.md) to create case files and choose settings.
Then [run/resume](docs/guide/running.md) and [read the output tables](docs/guide/outputs.md).
Keep production cases and outputs outside this checkout.

Five Snakemake workflows analyze historical climate, build a hydrological model, analyze projections,
generate climate scenarios and simulate hydrological responses. Model building and scenario generation
are independent prerequisites of simulation. Historical climate analysis helps users understand
variability and select forcing; projection analysis provides context for interpreting the tested changes.

## Documentation and development

- [User Guide](docs/guide/quick-start.md): installation, the example basin, project creation,
  workflow execution, outputs and troubleshooting.
- [Technical Manual](docs/technical-manual/overview.md): models and methods, datasets, workflow
  configuration, indicators, saved results and advanced extension.
- [Methodology](docs/site/scientific-approach/overview.qmd): the bottom-up assessment approach
  and interpretation of climate vulnerability and robustness.
- [Documentation map](docs/README.md): all documentation entry points and optional notebooks.
- [Build the publications](docs/publication/README.md): render the navigable website and complete
  User Guide and Technical Manual in HTML and PDF. The rendered site's Downloads page provides both volumes.
- [Development reference](dev/reference/README.md): current maintainer practices and architecture.
- [Changelog](CHANGELOG.md): release history.

## License

Copyright (c) 2021, Deltares. Licensed under the
[GNU General Public License, version 3 or later](LICENSE).
