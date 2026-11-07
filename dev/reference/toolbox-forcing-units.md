# Forcing unit admission

Last updated: 2026-11-06

Use this reference when changing historical climate catalogs or interpreting extracted forcing. The admission
rule is implemented by
[`resolved_unit_interpretation`](../../blueearth_cst/climate_analysis/prepare_climate_data_catalog.py)
and the shared [`forcing_data`](../../blueearth_cst/shared/forcing_data.py) admission functions.
Checks live in [`test_prepare_climate_data_catalog.py`](../../tests/test_prepare_climate_data_catalog.py)
and [`test_forcing_data.py`](../../tests/test_forcing_data.py).

## Reviewed variables

The daily conversion checks cover `precip`, `temp`, `temp_min`, `temp_max`, `press_msl`, `kin`, and `kout`
after renaming. They compare the catalog adapter's renames and conversions with the expected definitions.
They reject a missing or changed reviewed conversion, an extra conversion affecting a reviewed variable, or
another native variable renamed onto a reviewed name. Descriptions of unconsumed variables are allowed;
catalog shape alone does not establish scientific compatibility.

The conversions put temperature, pressure, and radiation into the units expected by historical extraction;
precipitation is not scaled. The owning code defines the exact arithmetic, renames, and binding revision.
Check that arithmetic when changing sources: source attributes alone do not establish the units of values
after extraction.

CHIRPS (`chirps` or `chirps_global`) uses an exact precipitation adapter check for the reviewed rename and
one-day time alignment, with an empty `unit_mult` map: precipitation must not be scaled. It takes six other
reviewed variables from the ERA5 extraction path, whose adapter must pass the daily binding check. The
general binding's allowance for surplus variables does not loosen the CHIRPS precipitation and time check.

## Configurable recipes

Non-default extraction records admission evidence per role (precipitation, companion and supplied PET),
including the source and consumed variables. Roles may share a catalog entry without overwriting evidence.
The reviewed ERA5 binding is recognized by its arithmetic, independently of the entry name. Its definition
is shared by historical and generation admission.

The other admitted profile is canonical daily input with no numerical conversion on the consumed variables.
It requires actual post-adapter `units`/`unit` attributes consistent with mm/day, degC, hPa and W/m2, as
applicable. Conflicting labels fail; values are never interpreted from their magnitudes. Catalog
`metadata.time_label: interval_end` explicitly declares daily intervals ending at their midnight timestamp.
Contiguous midnight timestamps alone cannot establish that meaning. This profile is a mechanism checked
with controlled fixtures, not qualification of a new public dataset or a new conversion profile.
Supplied PET may use HydroMT's native `mm` label for an interval-end daily total, numerically equivalent
to mm/day on the required one-day axis. Subdaily and multi-day accumulations fail the daily-axis check.

Recipe extraction intersects actual daily source coverage, rejects gaps and unsupported calendars, and
requires finite fields over the basin. Supplied PET must also be nonnegative; legitimate missing cells
outside the required domain are allowed. Its source must represent reference PET. Validation runs before
HydroMT's PET preparation can replace missing values with zero. The extraction fingerprint and role evidence
travel in the netCDF; preparation-only settings remain outside those bytes and the generator seed inputs.
See [configuration](../../docs/technical-manual/configuration.md) for the public recipe keys.

## Provenance and verification

The returned `UnitInterpretation` records the binding revision, interpreted units, selected precipitation
source, catalog paths and SHA-256 hashes, and extraction-module hash. Its evidence field also contains fixed
historical path strings; keep those serialized provenance values stable when moving documentation. The
current admission rule is the code linked above, not the historical label in that field.

Use the admission and rejection cases in the linked test file for catalog changes. For changes that can alter
forcing values, follow the [validation ladder](dev-validation.md).
