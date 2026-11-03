# Weather-generator interface

Last updated: 2026-10-20

Use this reference when changing generation inputs, replacing the weather generator or handing forcing
to a simulator. Preserve WG-1 through WG-6 as stable interface identifiers. They identify different
boundaries, not a universal schema that every engine must implement unchanged.

## Boundary and ownership

The current provider wraps weathergenr; its configuration and R wrappers are engine-specific.
`blueearth_cst/experiment/scenario_provider.py` owns generation and perturbation calls,
`generation_plan.py` owns preparation, and `generation_publication.py` owns collection publication.
A replacement must preserve scenario membership, ancestry and perturbation semantics and publish forcing
with the identities, descriptors and preparation inputs required by the collection reader.
It may need its own configuration and preparation adapter.

[Retained handoffs](../../docs/site/toolbox-reference/retained-handoffs.qmd) owns artifact layout and reuse.
[Toolbox architecture](toolbox-architecture.md) owns workflow boundaries.
WG-1 through WG-4 describe generation; WG-5 and WG-6 describe the current HydroMT/Wflow preparation
binding downstream of generation. They are not requirements to use Wflow for every replacement simulator.

## WG-1 — historical climate extraction

The selected historical store supplies the current generator's seven-variable forcing input.
`blueearth_cst/climate_analysis/extract_historical_climate.py` owns extraction;
`validate_wg1` in `blueearth_cst/shared/interchange_contracts.py` checks its structural subset:
time/latitude/longitude dimensions, float32 spatial axes and variables, and the required CRS/metadata.
The exact variable and native-label expectations belong to `_WG1_VARS_UNITS`.

Native labels are not effective physical units. The validator checks `units` only when present;
it does not establish daily cadence, value magnitudes or scientific suitability.
[Forcing-unit admission](toolbox-forcing-units.md) owns the reviewed interpretation and conversion policy.

Selected CHIRPS includes ERA5 companions; comparison-only CHIRPS can contain precipitation alone and
therefore report WG-1 differences without being unusable for WF0. Promotion to selected forcing requires
the full input. The orography sidecar is outside `validate_wg1`.

## WG-2 — stress-test perturbation grid

The current grid supports temperature and precipitation axes only; unsupported dimensions are rejected.
Adding an axis requires updating the producer, lookup columns and downstream consumers together.

The collection retains `perturbation_lookup.csv`. Its producer owns the closed column vocabulary;
`validate_wg2` checks the ordered header, textual member IDs, uniform padding, ordered rows and twelve
distinct months per member. Pass the independently expected member count to check completeness against
the requested design; an internally consistent table alone cannot establish that no member is missing.

Temperature changes are additive degrees Celsius. Precipitation mean and variance changes are percentages:
the current R binding reconstructs multipliers with `1 + percent / 100`. Unperturbed scenario rows have
no perturbation-grid member. Read `st_id` as text to preserve padding and joins.

The current producer quantizes requested multiplier levels to float32 and writes the percent conversion
at float64 shortest representation. Reconstruction is bounded to one float64 ulp for admitted multipliers
of at least `0.5`; bit-for-bit equality is not promised. The producer admits no upper ceiling.
`blueearth_cst/experiment/prepare_cst_parameters.py` owns that arithmetic and domain;
`validate_wg2` checks table structure, not this numerical bound.

## WG-3 — weathergenr configuration

One prepared configuration serves the generation request; scenario-specific output identities and
perturbations come from the provider and lookup, not duplicated configuration files.
`blueearth_cst/experiment/prepare_weathergen_config.py` and the R readers under
`blueearth_cst/weathergen/` own the current configuration interface.

`_WG3_SECTIONS` and `validate_wg3` own checks for consumed keys and types. Preserve upstream argument
spellings; the validator does not validate the generator's algorithm or all parameter ranges.
A replacement generator can have a different configuration surface; WG-3 describes weathergenr's binding.

## WG-4 — durable generated forcing

Publish one inventoried forcing artifact per scenario row, including unperturbed rows.
Membership and ancestry come from collection metadata, not reconstructed filenames.
Published forcing is durable and must not be deleted by simulator preparation.

`validate_wg4` checks the raster subset: time, latitude/longitude axes (long or short names),
at least precipitation and temperature, and EPSG:4326. Global `crs` and `category` attributes are
checked for contradictions when present. It does not validate collection membership or content hashes.
Collection readers and forcing descriptors separately establish inventory, identity, effective units,
calendar, endpoints and grid; the simulator adapter checks whether these meet its physical requirements.

## WG-5 — per-run HydroMT catalog

The Wflow preparation binding emits a temporary catalog pointing to each selected run's retained forcing.
`validate_wg5` checks the emitted HydroMT driver/metadata subset; HydroMT owns the catalog schema.
`validate_wg5_catalog_runs` compares catalogs with an independently supplied evaluated-run set and
requires each catalog to name its intended run. Neither check opens or validates the referenced netCDF.
Machine-specific catalog URIs are not the collection's portability contract.

## WG-6 — prepared Wflow forcing

The Wflow adapter prepares temporary model-grid forcing for execution.
Its content contract is [HM-2](toolbox-hydrological-model-interface.md#hm-2--prepared-forcing).
`validate_wg6` delegates to `validate_hm2`; do not maintain a parallel forcing schema here.
Temporal compatibility and preparation provenance remain adapter responsibilities.

## Verification and evidence

The interchange validators accept parsed objects and return differences. They are not a universal
runtime admission gate: WF0 uses WG-1 for readiness reporting, while collection and simulator readers
enforce their own contracts. A missing temporary artifact or skipped fixture test is an unavailable check.
Use `tests/test_interchange_contracts.py`, the provider/publication tests and affected reader checks
according to the [validation ladder](dev-validation.md).

Use the supported [WF3](../../docs/site/toolbox-reference/workflow-generate-scenarios.qmd) and
[WF4](../../docs/site/toolbox-reference/workflow-simulate-system.qmd) launchers for execution. Current
behavior is not certified by historical captures: CHIRPS unit evidence (private-checkout-only citation:
`../records/milestones/r12/implementation/evidence/chirps-unit-verification-2026-09-27.md`) preserves its
measured scope, and P2 acceptance (private-checkout-only citation:
`../records/milestones/r12/implementation/evidence/p2/acceptance.md`) records bounded preservation evidence.
