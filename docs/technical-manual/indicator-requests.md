---
title: "Indicator selection and calculation settings"
format: html
---

Indicator settings specify which measures to calculate from simulated hydrological responses and
which thresholds or other parameters they require. This reference describes the selection syntax and
input requirements. The [indicator module chapter](indicator-calculations.md) explains the measures
and their interpretation; the [User Guide](../guide/running.md#calculate-different-indicators-from-existing-simulations)
shows how to recalculate indicators from saved simulations.

YAML selects installed calculations and their permitted parameters. It does not define new formulas
or change their units, calendars, missing-data rules or aggregation methods.

## Calculation modes and indicator sets {#operations-and-sets}

In the hydrological simulation file, `operations` is a nonempty list containing `simulations`, `indicators`,
or both.
Omitted means both. Duplicates and unknown entries fail; ordering does not change meaning.
Simulations alone validate request structure without opening set files. Indicators alone require retained
responses and do not launch Wflow.

`indicator_sets` defaults to `[default]`. It accepts string IDs or inline mappings. IDs start with a
lowercase letter, then use lowercase letters, digits, `_` or `-`, up to 64 characters. Windows reserved
names are rejected. Selected IDs must be unique. `default` is the reserved bundled preset at
`config/indicator_sets/default.yml`; a custom file cannot replace it.

```yaml
operations: [indicators]
indicator_sets: [default, example_subset]
indicator_set_definitions:
  example_subset:
    config_path: indicator_sets/example_subset.yml
```

`config_path` is absolute or relative to the declaring hydrological simulation YAML. Selected sources are
read once per
invocation and captured. Installed bundled IDs resolve against the toolbox, never the current directory.
The default preset follows configured output variables for simulation requests and retained variables for
indicators-only requests. A selection cannot recover a variable the simulation did not retain.

## Selecting indicators and setting parameters {#inline-parameters-and-selection}

An inline mapping's `set` names a registered set. Optional `id` defaults to `set`.
`title` and `description` are presentation metadata. Other keys name registered indicators.
Omitted indicators stay enabled; omitted parameters inherit registered defaults.

```yaml
indicator_sets:
  - set: crop-water
    summer_low_flow_days:
      thresholds:
        baseline_quantile: {}
    summer_longest_dry_spell:
      dry_day_mm: 2.0
    winter_longest_dry_spell:
      enabled: false
  - set: default
```

Each supplied parameter replaces that parameter's complete value; nested threshold mappings are not
merged. `enabled` is Boolean and defaults to true. Disabled declarations still undergo static parameter
validation, but require no responses or historical preparation. Disabling all indicators fails.
Parameters encoded in a registered name cannot contradict that name. An inline set cannot also have a
file definition. Standalone flat files add `schema_version: 1` to this structure.

`indicator_selection` further narrows a selected set with one nonempty `include` or `exclude` list.
Names are published indicator keys and retain set order. It cannot re-enable an inline-disabled indicator.

```yaml
indicator_sets: [default, crop-water]
indicator_selection:
  crop-water:
    exclude: [summer_low_flow_days]
```

A deselected indicator is neither evaluated nor published nor checked for required inputs.
Unknown names, an unselected set, both list modes, duplicates or an empty final selection fail.
Duplicate YAML mapping keys also fail. Remove an entire set entry rather than requesting an empty set.

## Defining indicator sets in separate files {#standalone-definitions}

Copy `config/templates/indicator_set.template.yml` for an authored subset or aliases.
`selection: {registry: flow_regime}` chooses every registration in that installed set.
The registry mapping form accepts `{set: <id>, overrides: {<indicator>: {<parameter>: <value>}}}`.
Authored entries instead list `selection.indicators` with a stable `id`, title, description, references
and calculation. `calculation.set` names trusted installed code, `method` its versioned callable selector,
and `parameters` the method's closed values. For example, this file chooses one qualified flood declaration:

```yaml
schema_version: 1
id: flood_10
title: Ten-year discharge hazard
description: Annual maximum discharge; no exposure or damage calculation.
selection:
  indicators:
    - id: peak_10
      title: Ten-year annual maximum discharge
      description: Qualified production GEV estimator and screening requirements.
      references: []
      calculation:
        set: flood_risk
        method: flood_return_level/1
        parameters: {return_period: 10}
```

The full `flood_risk` set includes unsupported probabilities and therefore fails under the shipped
qualification. Only its 2- and 10-year examples are eligible, subject to sample/source checks.
The [flood-indicator reference](../../blueearth_cst/indicators/sets/flood_risk/README.md) describes
qualification
limits.

## Required inputs and handling incomplete results {#inputs-evaluation-and-recovery}

Before simulation, selected requirements are checked against `model.outvars`. Indicators-only requests
check retained variables and locations. Paired methods require aligned inputs. Supplied malformed
parameters always fail. `indicator_evaluation` is `strict` by default; `partial` permits absent inputs.

| Mode/input | Result |
|---|---|
| `strict`: any missing input | Refuses; names the affected calculation and input |
| `partial`: absent response | Omits the calculation from value tables; records its reason in dictionary/manifest |
| `partial`: missing location parameter | Publishes unavailable rows for affected locations |
| Either mode: data refuse a calculation | Publishes unavailable rows for that group and location |

An absent response drops the whole affected calculation into manifest `unavailable_calculations`, with a
reason such as `missing_response:pet`; no value rows are invented for that calculation. A set-declared
missing parameter at retained locations instead produces blank value rows with `status: unavailable` and
a reason such as `missing_parameter:thresholds`, recorded in `unavailable_results`.
A calculation can also refuse one group's data at one location while the other groups, locations and
indicators complete. Return levels do so with too few annual blocks (`5 annual blocks; 10 required`) or a
refused fit (`estimator refused: invalid_range`). These blank rows are recorded in manifest
`refused_results` (`indicator-set/7`), and their full screening evidence in `return_level_evidence.json`.
Publication then follows the mode: `partial` warns and publishes; `strict` keeps the computed set but
refuses to publish it.
The dictionary retains unavailable declarations and reports `partial` or `unavailable` coverage.
A request with no calculable declarations can therefore have an empty `tables` inventory.
This is successful only under partial mode. Calculated NaNs instead have `status: undefined`.
Unexpected programming errors and configuration faults, such as a return period outside the qualified
domain, are never converted into successful undefined/unavailable results.
See [table statuses](retained-handoffs.md#indicator-tables-and-dictionary).

Providing a missing parameter or selecting another calculation can create a new immutable set over
unchanged saved model time series. Earlier sets remain unchanged. A missing response requires a new experiment
with that variable retained. Named sets publish independently, so inspect the verdict and committed pointer
for each set after a multi-set failure.

## Set-specific preparation and conventions

The [crop-water README](../../blueearth_cst/indicators/sets/crop_water/README.md) is the complete owner of
Qcrit syntax, historical prerequisites, preparation timing, dry-run behavior, provenance and compatibility.
Its baseline-quantile preparation uses model building historical model discharge; neither synthetic
observations nor
scenario discharge supplies the reference. A minimal selection above does not bypass those prerequisites.
All five set READMEs and the module source are declared for offline inclusion by
[the local-source manifest](local-source-appendix.yml).

## Integrity and numerical rechecking

```powershell
pixi run python scripts/inspect_indicators.py --experiment <experiment-root> --list
pixi run python scripts/inspect_indicators.py --experiment <experiment-root> --manifest <indicators.json>
pixi run python scripts/inspect_indicators.py --experiment <experiment-root> --manifest <indicators.json> --recheck
```

Default inspection checks saved hashes, completion identity and structure without the current registry or
original response input. Matching saved check receipts are historical evidence. Coordinated changes to data
and manifests cannot be excluded by hash consistency alone.

`--recheck` requires matching execution sources, definitions, original input identity and recorded
dependencies. It invokes independent current checkers and writes a new attributed receipt. Checker-only
changes are allowed and receive fresh attribution. Missing/changed inputs block rechecking but preserve
artifact inspection. A requested unavailable or failed recheck exits nonzero. `--traceback` exposes causes.
Inspection never relaxes production reuse guards.

Evidence owners: `experiment/indicator_sets.py`, `indicator_registry.py`, `indicator_artifacts.py`,
`indicator_inspection.py`, `indicator_export.py` and `named_export.py`;
`tests/test_indicator_sets.py`, `test_partial_indicators.py` and `test_indicator_inspection.py`.
