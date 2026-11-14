# Weather-generator interface

Last updated: 2026-11-14

Use this reference when changing scenario generation, replacing the generator, or handing generated forcing
to a simulator. **WG** stands for **weather generator**. WG-1 through WG-6 are stable interface identifiers.
WG-1 through WG-4 describe stochastic generation;
WG-5 and WG-6 describe the current HydroMT/Wflow preparation binding. An alternative simulator need not
implement that binding. The GCM delta method is a second generation path; see
[delta-change collections](#delta-change-collections) for which contracts it shares.

## Boundary and ownership

The current provider wraps weathergenr.
[`scenario_provider.py`](../../blueearth_cst/experiment/scenario_provider.py) owns generation and
perturbation calls; [`generation_plan.py`](../../blueearth_cst/experiment/generation_plan.py)
prepares inputs; [`generation_publication.py`](../../blueearth_cst/experiment/generation_publication.py)
publishes collections. A replacement generator must preserve scenario membership, ancestry, perturbation
meaning, and the forcing identities, descriptors, and preparation inputs required by the collection reader.
Its configuration and preparation adapter may differ. The [architecture reference](toolbox-architecture.md)
explains workflow responsibilities; the
[retained handoffs](../../docs/technical-manual/retained-handoffs.md) describe saved products and reuse.
For figure appearance, panel layout and export, see
[styling weathergenr figures in CST](../../blueearth_cst/plotting/docs/weathergenr-style.md).

## WG-1 — historical climate extraction

The selected historical store supplies the current generator's seven-variable input.
[`extract_historical_climate.py`](../../blueearth_cst/climate_analysis/extract_historical_climate.py)
produces it. `validate_wg1` in
[`interchange_contracts.py`](../../blueearth_cst/shared/interchange_contracts.py) checks time, latitude and
longitude dimensions, float32 spatial axes and variables, and required CRS and metadata. `_WG1_VARS_UNITS`
owns the exact variable and native-label expectations. A `units` attribute is checked only when present;
matching native labels does not establish effective physical units. The validator does not establish daily
cadence, value magnitudes, or scientific suitability.
[Forcing-unit admission](toolbox-forcing-units.md) owns conversion and interpretation.

Selected CHIRPS includes ERA5 companion variables. A precipitation-only CHIRPS store used for WF0 comparison
can report WG-1 differences without being unusable for that comparison; selecting it as generator forcing
requires the full input. The orography sidecar is outside `validate_wg1`.

## WG-2 — stress-test perturbation grid

The grid has temperature and precipitation axes only; unsupported axes are rejected. Adding one requires
coordinated producer, lookup, and consumer changes. The collection retains `perturbation_lookup.csv`.
[`prepare_cst_parameters.py`](../../blueearth_cst/experiment/prepare_cst_parameters.py) owns its columns
and arithmetic. `validate_wg2` checks the ordered header, uniform ID padding, and exactly twelve distinct
months per member. Supply the independently expected member count to detect a truncated grid; an internally
consistent table cannot establish complete requested membership. It does not check CSV row order or prove
that `st_id` was parsed as text. Unperturbed rows have no perturbation-grid member. Read `st_id` as text to
preserve padding and joins.

Temperature changes are additive degrees Celsius. Precipitation mean and variance changes are percentages;
the R binding reconstructs multipliers as `1 + percent / 100`. The producer quantizes requested multiplier
levels to float32 and writes percent changes at float64 shortest representation. For admitted multipliers
**at least 0.5**, reconstruction is within **one float64 ulp** of the quantized level; bit-for-bit equality
is not promised. Both precipitation mean and variance bounds are refused below 0.5 before the DAG is built;
there is no upper ceiling. This is a bound on multiplier reconstruction, not on generated weather or
indicators. `validate_wg2` checks table structure, not the numerical bound.

## WG-3 — weathergenr configuration

One prepared configuration serves the generation request; the provider and lookup supply scenario-specific
identities and perturbations.
[`prepare_weathergen_config.py`](../../blueearth_cst/experiment/prepare_weathergen_config.py)
and the R readers under [`weathergen/`](../../blueearth_cst/weathergen) own this upstream binding. Preserve
weathergenr argument spellings. `_WG3_SECTIONS` and `validate_wg3` check consumed keys and types, not the
generator algorithm or all parameter ranges. A replacement generator may use another configuration surface.

## WG-4 — durable generated forcing

Publish one inventoried forcing artifact per scenario row, including unperturbed rows. Collection metadata,
not filenames, establishes membership and ancestry. Simulator preparation must not delete published forcing.
`validate_wg4` checks the raster subset: time and latitude/longitude axes (long or short names), at least
precipitation and temperature, and EPSG:4326. Global `crs` and `category` attributes are checked for
contradictions only when present. It does not validate collection membership or content hashes. Collection
readers and forcing descriptors establish inventory, identity, effective units, calendar, endpoints, and
grid; the simulator adapter checks its own physical requirements.

## WG-5 — per-run HydroMT catalog

The Wflow preparation binding emits a temporary catalog pointing to each run's retained forcing.
`validate_wg5` checks the emitted driver and metadata subset; HydroMT owns the full catalog schema.
`validate_wg5_catalog_runs` compares catalogs with an independently supplied evaluated-run set and checks
that each catalog names its intended run. Neither check opens the referenced netCDF. Machine-specific
catalog URIs are not the collection portability contract.

## WG-6 — prepared Wflow forcing

The Wflow adapter prepares temporary model-grid forcing for execution. Its content contract is
[HM-2](toolbox-hydrological-model-interface.md#hm-2--prepared-forcing), where **HM** stands for
**hydrological model**; `validate_wg6` delegates to
`validate_hm2`. Temporal compatibility and preparation provenance remain adapter responsibilities.

## Delta-change collections

The GCM delta method applies monthly WF2 projection changes to the case's historical record. It is the
projection-conditioned alternative to stochastic generation and is selected in the WF3 file with
`method: delta_change` and a `delta-change/3` block. Stochastic generation never reads WF2; only this path
does. The [delta-change guide](../../docs/technical-manual/delta-change-scenarios.md) explains settings,
transformations, set layout and plots to users.

Boundary and ownership:

- [`delta_wf2.py`](../../blueearth_cst/experiment/delta_wf2.py) is the WF2 boundary. `resolve_windows` reads
  the ensemble's `analysis/gridded_changes/` with its `_engine/gridded_changes_inventory.json` (schema 2 or
  later) and `_engine/analysis_record.json`, selects windows and their members, and refuses missing or older
  products. WF2's kernels own the change arithmetic; this module selects, checks and regrids factors.
- [`delta_scenarios.py`](../../blueearth_cst/experiment/delta_scenarios.py) validates settings, derives
  factors, applies them and publishes the collection (`delta-scenario-collection/2`). `delta-change/1` and
  `/2` collections remain readable; new generation refuses those settings.
- [`delta_generation.py`](../../blueearth_cst/experiment/delta_generation.py) captures the authored
  configuration and input hashes for the owned runner and rechecks them before publication. The all-workflow
  runner orders WF2 before WF3 and refuses `--rebuild` with delta generation.
- [`collection_method.py`](../../blueearth_cst/experiment/collection_method.py) is the single method
  discriminator. WF4 consumers call `collection_method` or `is_delta` instead of comparing schema strings,
  so a new collection format changes only that module.
- [`delta_plots.py`](../../blueearth_cst/experiment/delta_plots.py) and
  [`plot_delta_scenarios.py`](../../scripts/plot_delta_scenarios.py) render retained factors;
  [`delta_annual.py`](../../blueearth_cst/experiment/delta_annual.py) supplies annual diagnostics.
  [`generate_delta_scenarios.py`](../../scripts/generate_delta_scenarios.py) is a retired stub that exits
  with a pointer to the runner.

Contracts shared with the stochastic path:

- **WG-1:** the historical record is the same climate-store extract, with units supplied for supported
  sources rather than read from extraction labels.
- **WG-2 and WG-3:** not applicable. There is no perturbation grid or weathergenr configuration;
  `factor_lookup.csv` and `scenario_run_lookup.csv` bind runs to factor files, members and WF2 windows.
- **WG-4:** one self-contained forcing series per run, including the historical and neutral controls.
  Windows are never pooled.
- **WG-5 and WG-6:** WF4 prepares delta series through the same simulator adapter. A collection with PET
  ratios supplies PET, so the preparation identity records `pet_method: provided`
  (`wf4_ancillary_descriptor.py`).

WF4 reserves a pending delta selection in `delta-generation` mode with the authored settings and historical
location only, then binds the published set. It checks that the set's climate source and historical
forcing recipe match the WF4 configuration (`simulation_runner.py`). Collection identity and reuse follow
the [run identity reference](toolbox-run-identity.md#three-identities); the delta producer hashes its own
file rather than an import closure. Tests: [`test_delta_scenarios.py`](../../tests/test_delta_scenarios.py),
[`test_delta_wf2.py`](../../tests/test_delta_wf2.py),
[`test_delta_annual.py`](../../tests/test_delta_annual.py) and
[`test_collection_method.py`](../../tests/test_collection_method.py).

## Verification

The interchange validators accept parsed objects and return differences. They are not a universal runtime
gate: WF0 reports WG-1 readiness, while collection and simulator readers enforce their own contracts.
A missing temporary artifact or skipped fixture check is unavailable coverage. See
[`test_interchange_contracts.py`](../../tests/test_interchange_contracts.py),
[`test_prepare_cst_parameters.py`](../../tests/test_prepare_cst_parameters.py), and affected
[provider](../../tests/test_scenario_provider.py) and
[publication](../../tests/test_generation_publication.py)
tests. Use the [validation ladder](dev-validation.md) for behavior changes.
