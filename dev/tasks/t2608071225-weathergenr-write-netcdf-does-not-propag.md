---
title: Generated netCDFs lose `spatial_ref` from their template
type: watch-item
area: upstream / weathergenr
created: 2026-08-07
updated: 2026-08-07
---

> [!note] Overview
> **What** — Generated netCDFs lose `spatial_ref` from their template.
> **Why** — The workaround holds in-repo, but any consumer reading CRS off a generated netCDF gets nothing.
> **Trigger** — Upstream `tanerumit/weathergenr` propagates the attribute, or a consumer starts needing the CRS from that file.

## Refs

- Migrated from `dev/followups.md` on 2026-08-07, when the board replaced it. Prose below is that
  entry verbatim; it is the reproducible context, not a summary.

## Detail

**`weathergenr::write_netcdf` does not propagate `spatial_ref` attributes
from `template_path` to the output.** Confirmed 2026-05-07: the historical
template (`extract_historical.nc`) has `x_dim='longitude'` and
`y_dim='latitude'` on its `spatial_ref` variable, but the realization
files written by `write_netcdf` (`rlz_*_cst_0.nc`) have an *empty*
attribute list on their `spatial_ref` variable. Downstream
(`impose_climate_change.R`) then crashes when it uses the realization
as its own template, because `write_netcdf`'s `x_dim` lookup returns
`0` (numeric, from `ncatt_get` on a missing attr) — which slips past
the existence check and causes
`Error in nc_in$dim[[x_dim_name]] : attempt to select less than one element`.

*Workaround applied 2026-05-07:* in `src/weathergen/generate_weather.R`,
after each `write_netcdf` call, manually copy `spatial_ref` attributes
from the historical input file to the just-written realization file
via `ncdf4::ncatt_get` / `ncatt_put`. Marked clearly so it can be
removed when weathergenr is fixed.

*Proper fix:* in `tanerumit/weathergenr` `R/io_netcdf.R`, the
attribute-copy loop in `write_netcdf` looks correct on the surface
(`ncatt_get(nc_in, spatial_ref)` → `ncatt_put`) but evidently isn't
executing or isn't writing through. Investigate why the loop produces
zero attributes on the output. Separately, the missing-attribute check
should also assert `hasatt = TRUE` on the `ncatt_get` result, not just
test the value for NA / NULL — the current check accepts the numeric
`0` returned for a missing attribute and crashes one line later.
