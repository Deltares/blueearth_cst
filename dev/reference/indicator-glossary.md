# Indicator glossary — every spelling of every WF3 output variable

One output variable carries **five names** by the time it reaches a result table,
and they are not interchangeable. This file is the one place all five sit in one
row, so "what is `gwr`?" and "which of these do I put in a config?" have an
answer that does not require reading four modules.

**Derived, not authoritative.** Each column below is owned by a dict in the code,
named in the column heading. `tests/test_indicator_glossary.py` parses the two
tables in this file and fails when they disagree with those dicts, so the
glossary cannot go quietly stale the way an unchecked prose copy would — but the
fix for a failure is always to correct THIS file, never the code. Add a variable
to the dicts first and to the tables here second.

## 1. Variables

| `wflow_outvars` label | CSDMS name | csv code | token | table | metric |
| --- | --- | --- | --- | --- | --- |
| `river discharge` | `river_water__volume_flow_rate` | `Q` | `q` | `q_indicators.csv` | *11 names — see §2* |
| `precipitation` | `atmosphere_water__precipitation_volume_flux` | `p` | `precip` | `precip_indicators.csv` | `precip_annual_total` |
| `actual evapotranspiration` | `land_surface__evapotranspiration_volume_flux` | `aet` | `aet` | `aet_indicators.csv` | `aet_annual_total` |
| `groundwater recharge` | `soil_water_saturated_zone_top__net_recharge_volume_flux` | `gwr` | `gwr` | `gwr_indicators.csv` | `gwr_annual_total` |
| `overland flow` | `land_surface_water__volume_flow_rate` | `qof` | `overland_flow` | `overland_flow_indicators.csv` | `overland_flow_annual_mean` |
| `snow` | `snowpack_liquid_water__depth` | `swe` | `snow` | `snow_indicators.csv` | `snow_annual_max` |

Column by column, with the dict that owns it:

- **`wflow_outvars` label** — what a **user writes in a config**, and the only
  spelling in this table that appears in one. Owned by
  `blueearth_cst/model/setup_gauges_and_outputs.py::WFLOW_VARS`. A `naming.md` §6
  **tier-2** contract: grandfathered, renameable only with a migration note.
- **CSDMS name** — the Wflow.jl 1.x standard name the model build writes into
  `wflow_sbm.toml`. Same dict. `naming.md` §6 **tier 1**: upstream-owned, *not
  renameable at all*, not even with a note.
- **csv code** — the `[output.csv]` `header`, so the run csv column is
  `<code>_<subcatchment-or-gauge-id>` (`gwr_101`). Owned by
  `blueearth_cst/shared/wflow_outputs.py::CODES`. Discharge is deliberately
  **absent from `CODES`**: it does not travel the basin-average path, and its
  header is the fixed `Q` that `shared/gauges.py` and `export_wflow_results` both
  key on. Changing a code renames the csv column, the hydromt variable
  (`<code>_subcatchment`), rule 1.14b's derived tables, and figure filenames.
- **token** — the short name in the result **filename** and in the composite
  metric. Owned by
  `blueearth_cst/shared/indicator_tables.py::VARIABLE_TOKENS`. Published as a
  contract in `contracts/hydrological-model-seam.md` §HM-7, which is the
  authority for this column; the row above is a copy of it.
- **table / metric** — derived from the token by `indicator_table_filename` and
  `basin_metric_name`; no dict of its own.

**Why the token and the code differ at all, for three of six.** They answer to
different owners: the code reaches the csv through the TOML, the token names our
files. Where the repo already had a canonical short name the token takes it —
which is why `aet` and `gwr` coincide with their code, and why `gwr` replaced the
locally-invented `recharge` on 2026-08-11. Where taking the code would have cost
something, it was not taken:

| token | rejected candidate | why |
| --- | --- | --- |
| `precip` | `p` | `naming.md` §6 tier 2 makes `precip` the canonical cross-tool stem; `p` would be a further spelling of a variable that already has enough |
| `snow` | `swe` | the CSDMS name is `snowpack_liquid_water__depth` — snowpack *liquid water*, not total water equivalent. `swe` would assert a physical claim upstream does not make |
| `overland_flow` | `qof` | no canonical short name exists, and `qof` is opaque in a filename |
| `aet` | `et` | `pet` is already canonical here; one letter apart in the same file is a misreading waiting to happen |

**The consequence that has bitten twice, stated plainly:** a reducer that uses
the *token* as a csv column prefix reads `aet` and `gwr` and silently finds
nothing for the other three. It must match on `CODES`. This is exactly how
8bd51de wrote two indicator tables as a header and zero rows with every rule
green — see `export_wflow_results.py::MissingOutputColumnError`, which now raises
instead.

### Two more spellings, for figures

Axis legends and the monthly resample rule live in
`wflow_outputs.py::PLOT_META`, keyed by **code** rather than by label, because a
figure sees only what the csv carried.

| code | resample | legend |
| --- | --- | --- |
| `p` | `sum` | Precipitation (mm month⁻¹) |
| `aet` | `sum` | Actual Evapotranspiration (mm month⁻¹) |
| `gwr` | `sum` | Groundwater Recharge (mm month⁻¹) |
| `qof` | `mean` | Overland Flow (m³s⁻¹) |
| `swe` | `sum` | Snowpack (mm month⁻¹) |

## 2. Metric vocabulary

A `metric` value is `<token>_<statistic>`, so a result file is self-contained
once it leaves the project tree and needs no separate `variable` column.

**Discharge** — 11 metrics, owned by `indicator_tables.py::Q_METRIC_SUFFIXES`:

| metric | statistic | grain |
| --- | --- | --- |
| `q_annual_mean` | mean of daily flow, per year | per-realization |
| `q_mean_annual_max` | annual maximum, meaned over years | per-realization |
| `q_mean_annual_min` | annual minimum, meaned over years | per-realization |
| `q_mean_annual_p95` | annual 95th percentile, meaned over years | per-realization |
| `q_mean_annual_7day_max` | annual max of the 7-day mean, meaned over years | per-realization |
| `q_mean_annual_7day_min` | annual min of the 7-day mean, meaned over years | per-realization |
| `q_baseflow_index` | baseflow index | per-realization |
| `q_return_level_10yr_max` | GEV return level of the annual maximum | pooled (`rlz_id = 0`) |
| `q_return_level_2yr_7day_min` | GEV return level of the annual 7-day minimum | pooled (`rlz_id = 0`) |
| `q_wettest_month_mean` | mean flow in the wettest month, chosen once from the pooled baseline | pooled (`rlz_id = 0`) |
| `q_driest_month_mean` | mean flow in the driest month, same rule | pooled (`rlz_id = 0`) |

The two return periods were the `Tpeak` / `Tlow` config keys until 2026-08-12,
which made these two names partly config-derived — a project with `Tpeak: 50`
emitted `q_return_level_50yr_max`, so `validate_hm7` had to match a *pattern*
rather than an enumeration. The owner retired those keys: a return period is a
property of the indicator set the toolbox defines, not of a project. Both shipped
at 10 and 2 everywhere, so no name changed. They are now
`indicator_tables.RETURN_PERIOD_PEAK_YR` / `RETURN_PERIOD_LOW_YR`, the vocabulary
is closed, and `validate_hm7` enumerates.

Three wordings in that list are deliberate and were each chosen against a
shorter alternative:

- **`_p95`, not `q95`.** Our statistic is the mean annual 95th percentile, a
  **high** flow. Conventional *Q95* is the flow exceeded 95% of the time, a
  **low**-flow drought index — the opposite end of the distribution. Verbose
  beats wrong.
- **`mean_annual_`, not `annual_`.** It says "annual statistic, then mean over
  years", a two-step reduction the pre-R11 names hid.
- **"return level", not "return period" or "return interval".** The return
  *period* is the input `T` in years; the return *level* is the output, a
  discharge magnitude — which is what the row's `value` holds. Naming the metric
  after the period would name the wrong quantity. "Return interval" is
  additionally non-standard; the established synonyms are *return period* and
  *recurrence interval*.

**Basin-scalar variables** — one metric each, owned by
`indicator_tables.py::BASIN_METRIC_SUFFIXES`. All are per-realization. Two
asymmetries in that table are ruled, not accidental:

- **Overland flow reduces with a `mean`, the rest with a `sum` or a `max`.** It
  is a volume flow rate (m³ s⁻¹), so summing daily values gives a quantity in no
  unit anyone wants; the annual mean preserves the native unit. ET, recharge and
  precipitation keep `annual_total` in mm/yr, because a daily sum of a mm Δt⁻¹
  flux is a legitimate time-integral. Scoped deliberately to overland flow.
- **These suffixes omit the `mean_` prefix the `q` vocabulary uses**
  (`snow_annual_max`, not `snow_mean_annual_max`), by owner ruling 2026-08-08.
  An accepted asymmetry: `q_mean_annual_max` and `snow_annual_max` describe the
  same reduction shape spelled two ways.

## 3. Row identifiers

Every indicator table carries the same seven columns, in this order
(`indicator_tables.py::INDICATOR_COLUMNS`):

| column | meaning |
| --- | --- |
| `metric` | `<token>_<statistic>` from §2 |
| `location` | the **bare** wflow id (`130000086`, not `Q_130000086`), so it joins `outlet_index.csv` with no crosswalk. Gauge/outlet ids in `q_indicators.csv`; **subcatchment** ids in every other table |
| `st_id` | stress-test member; `st_0` is the unperturbed baseline |
| `rlz_id` | realization `1..RLZ_NUM`, or **`0` meaning pooled over realizations** |
| `temp_change` | temperature perturbation, °C |
| `precip_change` | precipitation perturbation, multiplicative factor |
| `value` | `float32`, unrounded |

`basin` is a **reserved** `location` for a whole-basin scalar. Nothing emits it
today: since 8bd51de the csv columns are per-subcatchment means, so no
whole-basin column exists to carry the value, and deriving one by area-weighting
would silently assume subcatchments tile rather than nest (Q11).

## Renaming anything here

Every column in §1 except the CSDMS name is a `naming.md` §7 surface — old → new
mapping in a `dev/<milestone>/migration_<topic>.md` note. The CSDMS name is
§6 tier 1 and cannot be renamed locally at all.

Records so far, all against `dev/milestones/r11/migration_indicator-tables.md`
and its banners: the wide → long reshape (2026-08-08), the identifier-first
column reorder with `realization_id` → `rlz_id` (2026-08-11), and
`recharge` → `gwr` (2026-08-11). Two earlier ones are R9's:
`migration_indicator-axis-columns.md` (`tavg` → `temp_change`, `prcp` →
`precip_change`) and `migration_project-tree.md` (`Qstats.csv` →
`q_indicators.csv`).

**Pre-existing experiments are never migrated in place** — the standing ruling
since R7 (GA-2). A result tree written before a rename keeps its old names; re-run
the experiment to get the new ones, and delete the orphan by hand (`tree-check`
reports it as undeclared).
