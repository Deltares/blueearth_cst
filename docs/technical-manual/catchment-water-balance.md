---
title: "Observed catchment water balance"
format: html
---

Optional `evaluation.budyko: true` in the model-building YAML screens observed runoff against selected
prepared historical precipitation and PET. It requires the project file's
[`observations.river_discharge`](configuration.md#observations), whose `time_label` it uses:

```yaml
evaluation:
  budyko: true
```

## Discharge declaration

Omitted, null or false disables the diagnostic. Enabling it asserts that the discharge file holds daily
mean m³/s over midnight-to-midnight UTC days. Naive timestamps are asserted
UTC; aware timestamps normalize to UTC and must land at midnight. Start labels shift one day in a derived
series; the input CSV and ordinary model evaluation keep their labels. The toolbox checks structure
and records the declaration as user-supplied support; it cannot establish from the values where the
measurements came from.

## Catchment areas and forcing

The diagnostic uses original snapped upstream catchments and hydrography cell areas. Prepared model
forcing is sampled at those cell centres using nearest cells, then area-weighted. The same contributing
area converts daily mean Q to depth with `Q * 86400 * 1000 / area_m2`. This discretization is a screening
approximation; inspect station areas and forcing coverage before interpreting departures.

## Ratios

Only complete matched water years of precipitation, PET and observed Q enter the ratios. The covered day
(interval end minus one day) determines window and water-year membership. Missing days remain missing;
partial edge years are excluded, and no spin-up year is removed.

Aridity is total PET / total P and runoff coefficient is total runoff depth / total P over the same saved
years. Zero P or no eligible years produces an unavailable result. These are ratios of totals, rather than
averages of annual ratios.

## Saved files

Outputs live in each forcing-set folder, `models/hydrology/wflow/historical/<id>/`:

- `tables/budyko_stations.csv`, `budyko_annual.csv` and `budyko_summary.csv`;
- the diagram `plots/budyko_observed_runoff_coefficient_stations.png` (drawn by the evaluation rule);
- daily evidence, input hashes and support declarations in `_engine/budyko_daily.csv` and
  `_engine/budyko_methods.json`.

The target `evaluate_budyko` requires model/forcing preparation but no Wflow simulation. Saved tables support
`scripts/render_figures.py budyko --project <project_dir> --out <new_dir>`.

## Interpretation

Water and energy limits describe long-term equilibrium assumptions. Short/discontinuous records,
storage changes, regulation, transfers, basin area and forcing/observation errors can explain departures.
The residual `1-Q/P` is not measured ET. Unusual points remain visible; no automatic pass/fail or source
ranking is supplied. Tables keep counts, actual years and exclusion reasons for inspection.
