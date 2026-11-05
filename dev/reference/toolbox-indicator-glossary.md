# Indicator variable and result glossary

Last updated: 2026-11-05

Use this lookup to distinguish configuration labels, Wflow standard names, native CSV codes, retained
tokens, and indicator names. The mappings are derived from code, not an alternative source of defaults.
`tests/test_indicator_glossary.py` checks the variable, plot-label, and discharge tables against their
owners; change the owning code first when the vocabulary changes, then update these tables.

## 1. Variables

| `model.outvars` label | CSDMS name | csv code | token | table | indicator |
| --- | --- | --- | --- | --- | --- |
| `river discharge` | `river_water__volume_flow_rate` | `Q` | `q` | `q_run_indicators.csv` | *11 names — see §2* |
| `precipitation` | `atmosphere_water__precipitation_volume_flux` | `p` | `precip` | `precip_run_indicators.csv` | `precip_annual_total` |
| `potential evapotranspiration` | `land_surface_water__potential_evaporation_volume_flux` | `pet` | `pet` | `pet_run_indicators.csv` | `pet_annual_total` |
| `actual evapotranspiration` | `land_surface__evapotranspiration_volume_flux` | `aet` | `aet` | `aet_run_indicators.csv` | `aet_annual_total` |
| `groundwater recharge` | `soil_water_saturated_zone_top__net_recharge_volume_flux` | `gwr` | `gwr` | `gwr_run_indicators.csv` | `gwr_annual_total` |
| `overland flow` | `land_surface_water__volume_flow_rate` | `qof` | `overland_flow` | `overland_flow_run_indicators.csv` | `overland_flow_annual_mean` |
| `snow` | `snowpack_liquid_water__depth` | `swe` | `snow` | `snow_run_indicators.csv` | `snow_annual_max` |
| `temperature` | `atmosphere_air__temperature` | `temp` | `temp` | `temp_run_indicators.csv` | `temp_annual_mean` |

Each run table has a paired `<token>_group_indicators.csv`; §3 defines their distinct identifier columns.

Column by column, with the dict that owns it:

- **`model.outvars` label** — what a **user writes in a config**, and the only
  spelling in this table that appears in one. Owned by
  `blueearth_cst/model/setup_gauges_and_outputs.py::WFLOW_VARS`. A `dev-naming-conventions.md` §6
  **tier-2** contract: grandfathered, renameable only with a migration note.
- **CSDMS name** — the Wflow.jl 1.x standard name the model build writes into
  `wflow_sbm.toml`. Same dict. `dev-naming-conventions.md` §6 **tier 1**: upstream-owned, *not
  renameable at all*, not even with a note.
- **csv code** — the `[output.csv]` `header`, so the run csv column is
  `<code>_<subcatchment-or-gauge-id>` (`gwr_101`). Owned by
  `blueearth_cst/shared/wflow_outputs.py::CODES`. Discharge is deliberately
  **absent from `CODES`**: it does not travel the basin-average path, and its
  header is the fixed `Q` that `shared/gauges.py` keys on. Changing a code renames the CSV column,
  the HydroMT variable
  (`<code>_subcatchment`), rule 1.14's derived tables, and figure filenames.
- **token** — the short name in the result **filename** and in the composite
  indicator. Owned by
  `blueearth_cst/shared/indicator_tables.py::VARIABLE_TOKENS`.
  [HM-7](toolbox-hydrological-model-interface.md#hm-7--retained-indicator-sets) describes its
  interface context;
  the code owns the exact mapping.
- **table / indicator** — derived from the token by `indicator_table_filename` and
  `basin_indicator_name`; no dict of its own.

CSV codes follow Wflow output declarations; tokens name retained files. The differences prevent
ambiguous or misleading filenames:

| token | rejected candidate | why |
| --- | --- | --- |
| `precip` | `p` | `dev-naming-conventions.md` §6 tier 2 makes `precip` the canonical cross-tool stem; `p` would be a further spelling of a variable that already has enough |
| `snow` | `swe` | the CSDMS name is `snowpack_liquid_water__depth` — snowpack *liquid water*, not total water equivalent. `swe` would assert a physical claim upstream does not make |
| `overland_flow` | `qof` | no canonical short name exists, and `qof` is opaque in a filename |
| `aet` | `et` | `pet` is already canonical here; one letter apart in the same file is a misreading waiting to happen |

Match CSV columns using their owning code mapping, not filename tokens. WF4 reduces the
series named by its retained response inventory; similar-looking spellings are not interchangeable.

### Two more spellings, for figures

Axis legends and the monthly resample rule live in
`wflow_outputs.py::PLOT_META`, keyed by **code** rather than by label, because a
figure sees only what the csv carried.

| code | resample | legend |
| --- | --- | --- |
| `p` | `sum` | Precipitation (mm month⁻¹) |
| `pet` | `sum` | Potential Evapotranspiration (mm month⁻¹) |
| `aet` | `sum` | Actual Evapotranspiration (mm month⁻¹) |
| `gwr` | `sum` | Groundwater Recharge (mm month⁻¹) |
| `qof` | `mean` | Overland Flow (m³s⁻¹) |
| `swe` | `mean` | Snowpack liquid water (mm) |
| `temp` | `mean` | Air Temperature (°C) |

Temperature is Wflow's air-temperature forcing retained as subcatchment means in degrees Celsius,
using the upstream standard name `atmosphere_air__temperature`.
It is a state variable: monthly and annual reductions use means, never depth sums. The default
`temp_annual_mean` averages water-year means with equal block weight, following the default set's
partial-block and missingness policies. Add `temperature` to `model.outvars` to enable it; existing
output selections are unchanged. Existing retained simulations lacking temperature require a new
simulation before temperature indicators can be calculated.

## 2. Indicator vocabulary

A `indicator` value is `<token>_<statistic>`, so a result file is self-contained
once it leaves the project tree and needs no separate `variable` column.

**Discharge** — 11 indicators, owned by `indicator_tables.py::Q_INDICATOR_SUFFIXES`:

| indicator | statistic | grain |
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

**Discharge outside the default preset** — 5 registered definitions, owned by
`indicator_registry.py::OPTIONAL_Q_CALCULATIONS`. A custom indicator set selects them; `current_default`
never expands them. IHA-style (eflowrisk `amax30d`, `amin30d`, `amin90d`, `mean_rate_rise`,
`mean_rate_fall`), but averaged over water years rather than the eflowrisk median.

- `q_mean_annual_30day_max`, `q_mean_annual_30day_min`, `q_mean_annual_90day_min`:
  `annual_rolling_extreme/1` with `window` and `extreme`; m3 s-1.
- `q_mean_rise_rate`, `q_mean_fall_rate`: `annual_daily_change/1` with `direction`; m3 s-1 d-1.

The default return periods are owned by `indicator_tables.RETURN_PERIOD_PEAK_YR` and
`RETURN_PERIOD_LOW_YR`; they are properties of the indicator definitions, not project settings.
The table's `rlz_id = 0` notation identifies pooled calculation grain; it is not a column in current
run or group result tables.

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
  discharge magnitude — which is what the row's `value` holds. Naming the indicator
  after the period would name the wrong quantity. "Return interval" is
  additionally non-standard; the established synonyms are *return period* and
  *recurrence interval*.

**Basin-scalar variables** — one indicator each, owned by
`indicator_tables.py::BASIN_INDICATOR_SUFFIXES`. All are per-realization. Two
asymmetries in that table are ruled, not accidental:

- **Overland flow reduces with a `mean`, the rest with a `sum` or a `max`.** It
  is a volume flow rate (m³ s⁻¹), so summing daily values gives a quantity in no
  unit anyone wants; the annual mean preserves the native unit. ET, recharge and
  precipitation keep `annual_total` and report mm: the mean over water-year
  blocks of each block's summed daily depth (partial edge blocks kept without
  duration normalization, missing days contributing zero), which is mean annual
  depth in mm/yr only for complete nonmissing blocks. A daily sum of a mm Δt⁻¹
  flux is a legitimate time-integral. Scoped deliberately to overland flow.
- **These suffixes omit the `mean_` prefix the `q` vocabulary uses**
  (`snow_annual_max`, not `snow_mean_annual_max`). `q_mean_annual_max` and `snow_annual_max`
  describe the same reduction shape with different established names.

## 3. Row identifiers

New `indicator-set/6` run and group tables carry six columns, in this order:

| column | meaning |
| --- | --- |
| `indicator` | `<token>_<statistic>` from §2 |
| `location` | the registry's bare Wflow ID, joinable to `outlet_index.csv` |
| `run_id` or `group_id` | prefixed text identifying a run or group in its respective table |
| `value` | result stored as plain decimal text to four significant digits, or empty |
| `status` | `available`, `undefined`, or `unavailable` |
| `reason` | reason for `unavailable`; empty otherwise |

`<token>_run_indicators.csv` uses `run_id` (`run_01`); `<token>_group_indicators.csv` uses
`group_id` (`group_01`). Join runs to the collection's `scenario_run_lookup.csv` for realization and
`st_id`. Join groups to `group_metadata.csv`, then by `st_id` to `perturbation_lookup.csv` for monthly
changes. `group_members.csv` records contributing runs, including singleton groups. An empty `st_id`
and `is_reference=true` identify an unperturbed group. Run-grain group summaries average realization
indicators with equal weight and NaN propagation; return levels pool member-local blocks.

An empty `value` with `undefined` is a calculated NaN; an empty `value` with `unavailable` names a
missing input in `reason`. Neither is zero. A missing response can also exclude a calculation from value
tables while retaining its declaration and reason in the manifest and named dictionary. See
[retained handoffs](../../docs/technical-manual/retained-handoffs.md) for current table and reuse rules.

Axis columns are derived at reporting time rather than stored in indicator tables. `precip_change`
in the perturbation lookup is a percent.
[HM-7](toolbox-hydrological-model-interface.md#hm-7--retained-indicator-sets)
defines the retained-set boundary; `blueearth_cst/shared/surface_axes.py` supplies monthly axis calculations.
Keep identifiers as text; their scope is the owning collection or indicator set.

`indicator_tables.py::INDICATOR_COLUMNS` describes legacy expanded rows; current six-column retained
tables are written by `indicator_plan.py`.

`basin` is a reserved `location` for a whole-basin scalar. Current native columns hold subcatchment
means, not a whole-basin value. Deriving one by area weighting would assume subcatchments tile rather
than nest; no current result row makes that claim.

## Renaming anything here

The CSDMS names are upstream owned and cannot be renamed locally. Other §1 names are project contracts;
follow [naming conventions](dev-naming-conventions.md) and provide a migration mapping for a rename.
Existing experiment trees keep their recorded names; rerun an experiment to obtain a new schema rather
than editing retained results in place.
