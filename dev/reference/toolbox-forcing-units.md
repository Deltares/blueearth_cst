# Forcing unit admission

Last updated: 2026-11-05

Use this reference when changing historical climate catalogs or interpreting extracted forcing. The admission
rule is implemented by
[`resolved_unit_interpretation`](../../blueearth_cst/climate_analysis/prepare_climate_data_catalog.py)
and checked by [`test_prepare_climate_data_catalog.py`](../../tests/test_prepare_climate_data_catalog.py).

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

## Provenance and verification

The returned `UnitInterpretation` records the binding revision, interpreted units, selected precipitation
source, catalog paths and SHA-256 hashes, and extraction-module hash. Its evidence field also contains fixed
historical path strings; keep those serialized provenance values stable when moving documentation. The
current admission rule is the code linked above, not the historical label in that field.

Use the admission and rejection cases in the linked test file for catalog changes. For changes that can alter
forcing values, follow the [validation ladder](dev-validation.md).
