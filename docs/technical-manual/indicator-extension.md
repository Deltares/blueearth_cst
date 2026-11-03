---
title: "Adding custom indicators"
format: html
---

Use the [indicator module reference](../../blueearth_cst/indicators/README.md) to implement a custom
calculation in Python. It describes input and result objects, accepted parameters, paired variables,
diagnostic tables, independent numerical checks and the records used to identify a calculation.
Hydrological simulation handles input selection, warm-up handling, file publication and reuse guards.

## Calculation references {#canonical-sources}

The following repository references describe the indicator module and calculation sets. They remain
separate from the manual and require repository access when reading offline.

| Reference identifier | Calculation reference |
|---|---|
| `indicator-module` | [Module README](../../blueearth_cst/indicators/README.md) |
| `indicator-default` | [Default](../../blueearth_cst/indicators/sets/default/README.md) |
| `indicator-flow-regime` | [Flow regime](../../blueearth_cst/indicators/sets/flow_regime/README.md) |
| `indicator-eflow-erfa` | [ERFA](../../blueearth_cst/indicators/sets/eflow_erfa/README.md) |
| `indicator-crop-water` | [Crop-water](../../blueearth_cst/indicators/sets/crop_water/README.md) |
| `indicator-flood-risk` | [Flood hazards](../../blueearth_cst/indicators/sets/flood_risk/README.md) |

Each set reference describes its formulas, units, missingness and qualification limits.
The crop-water reference includes its preparation procedure, including the historical low-flow threshold.

## Supporting evidence and interpretation limits {#evidence-availability}

The ERFA comparison against the R implementation and its test data are available in the repository.
Those files, upstream literature and external software manuals require separate access when reading offline.
Including calculation documentation establishes documented behavior, not new ecological, agricultural
or flood-risk qualification. Source checks and software tests do not establish basin-model adequacy.

## Requirements for custom calculations {#extension-boundary}

Add installed trusted code using the module README's instructions. YAML chooses it; YAML cannot import a path.
An unrelated set change does not enter a selected calculation's identity, while shared contributing source
changes do. Check each new calculation against independently derived expected values. Review how results
are combined across realizations before introducing a new aggregation method. The
[scientific approach](../site/scientific-approach/overview.qmd) explains the assessment methodology.
