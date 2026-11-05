# Weather-generator interface

Last updated: 2026-11-05

Use this reference when changing scenario generation, replacing the generator, or handing generated forcing
to a simulator. **WG** stands for **weather generator**. WG-1 through WG-6 are stable interface identifiers.
WG-1 through WG-4 describe generation;
WG-5 and WG-6 describe the current HydroMT/Wflow preparation binding. An alternative simulator need not
implement that binding.

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

## Verification

The interchange validators accept parsed objects and return differences. They are not a universal runtime
gate: WF0 reports WG-1 readiness, while collection and simulator readers enforce their own contracts.
A missing temporary artifact or skipped fixture check is unavailable coverage. See
[`test_interchange_contracts.py`](../../tests/test_interchange_contracts.py),
[`test_prepare_cst_parameters.py`](../../tests/test_prepare_cst_parameters.py), and affected
[provider](../../tests/test_scenario_provider.py) and
[publication](../../tests/test_generation_publication.py)
tests. Use the [validation ladder](dev-validation.md) for behavior changes.
