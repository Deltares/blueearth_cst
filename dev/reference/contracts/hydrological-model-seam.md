# Contract: hydrological-model seam (HM-1 .. HM-7)

> **Genre:** dev-facing interchange contract. **Audience:** a future *swapper* —
> someone replacing Wflow-SBM with an alternative hydrological engine, or the R6
> model-flexibility work — read end-to-end. Not an end-user doc (hence `dev/`,
> not `docs/`; precedent `dev/reference/workflows/run_stress_test.md`).
> **Source of record:** `dev/milestones/p32b/interchange-contracts-design.md` (ACCEPTED
> 2026-07-24, §5.3 / §5.4 / §5.6 / §5.5). Every load-bearing fact below cites a
> Snakefile line, a script line, or an observed fixture artifact; do not add a
> contract fact that is not so grounded.

## Scope and method

The **hydrological-model seam** spans `build_model.smk` (wf1 build) and
`run_stress_test.smk` (wf3 run) — the point where the hydrological
engine could be swapped without re-architecting the pipeline. Wflow-SBM (built by
hydromt) is the current occupant, but **this contract is model-agnostic**: it
pins what the pipeline hands *in* (forcing + static grid + run config) and
expects *out* (discharge CSVs → response surface), not Wflow's physics.

**Grounded in** the fixture tree `test_case/test_local` (era5 branch,
`config/workflows/snake_config_model_test.yml`) inspected with xarray, the base
vs per-cst `wflow_sbm.toml` diff, and the wf1/wf3 rules + scripts.

**CST-scope disclaimer (the governing constraint; `AGENTS.md` Hard
Constraints).** For `staticmaps.nc` and `wflow_sbm.toml` we pin **only the
names/fields OUR code and the run TOML reference** — this is *pinned-as-reliance*
on a consumed upstream schema, NOT a re-specification of the wflow static-grid
schema or the TOML physics parameterization. The remaining ~39 staticmaps
variables and the physics value blocks are labelled **"wflow schema, consumed
verbatim, unpinned"**. The validators *read* upstream artifacts to check OUR
reliance; they never assert upstream correctness. The numeric outlet id in a
gauge column (`Q_130000086`) is wflow's outlets-map cell value — its derivation
stays **wflow-owned**, recorded as reliance, never asserted (C3 boundary, HM-4→
HM-5→HM-7 invariant below).

**Fixture branch = era5.** Wflow-side contracts here are era5-grounded.

**Contract-surface tiers** (design §5.1): (1) **Pinned** — a structural fact a
swap MUST reproduce; (2) **Pinned-as-reliance** — OUR consumed/rewritten subset
of the upstream staticmaps/TOML schema; (3) **Deliberately unpinned** — internal
detail (physics blocks, state-variable schema, provenance attrs).

Per-artifact schema (design §5.4): *artifact id · path pattern · producer rule ·
consumer rule(s) · dims · coords · data_vars · CRS · time axis/calendar · naming
pattern · temp() lifecycle · pinned surface · deliberately unpinned · validator*.
Rendered one subsection per artifact.

---

## HM-1 — static grid (staticmaps.nc)

- **path pattern:** `models/hydrology/wflow/staticmaps.nc`.
- **producer:** rule 1.07 `create_model` (hydromt build).
- **consumers:** wf1 rules 1.08 / 1.09 / 1.04 / 1.15; wf3 rule 3.14
  (`WflowSbmModel(root)`).
- **coords:** `(latitude, longitude)` `float64` + `spatial_ref` EPSG:4326 +
  `GeoTransform`.
- **pinned surface (pinned-as-reliance — OUR-referenced names only):**
  `subcatchment` (zone raster — plot aggregation, P3-2a §5.2); `land_elevation`
  (`m` — parity DEM, `hydromt_wflow/naming.py:10`); plus the TOML-referenced
  `local_drain_direction`, `river_mask`, `outlets`, and the `[input.static]`
  name set the run resolves. **Grid definition** (the `(lat, lon)` axes +
  `GeoTransform`) is pinned as the **co-registration target forcing must match**.
- **temp() lifecycle:** not `temp()`.
- **deliberately unpinned:** the **~39 unpinned wflow vars** — the fixture
  `staticmaps.nc` has **44 data_vars total**, minus the pinned OUR-referenced set
  (`vegetation_*`, `soil_*`, `meta_*`, `river_*` beyond mask) — **wflow schema,
  consumed verbatim, unpinned** (design arch-6).
- **validator:** `validate_hm1`.

## HM-2 — Wflow forcing (inmaps)

- **path pattern:** `models/hydrology/wflow/forcing/inmaps_historical.nc` (wf1
  forcing); wf3 twin `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_st_<m>.nc`
  (= WG-6 on the weather-generator seam). R07 B5 files the wf3 twin on the
  HYDROLOGY side because it is model-grid forcing, symmetric with the wf1
  path above; R9 P2 flattened the `rlz_<n>/` level out of it.
- **producer → consumer:** rule 1.10 `add_climate_forcing` (hydromt update) → rule 1.14
  `run_wflow`; wf3 rule 3.14 → rule 3.15.
- **dims:** `(time, latitude, longitude)` on the **model grid** (`float64`
  lat/lon matching HM-1).
- **data_vars:** exactly `precip`, `pet`, `temp` — all `float32`, each
  `grid_mapping=spatial_ref`.
- **CRS / grid:** `spatial_ref` EPSG:4326 + `GeoTransform`.
- **time axis/calendar:** daily `proleptic_gregorian` (wf1). *(wf3 forcing axis
  is moved to `standard` — see HM-4.)*
- **pinned surface:** the dims, the model-grid `(lat, lon)`, and the **variable
  names** `precip` / `pet` / `temp` — the names are the consumer contract: they
  are the RHS values the TOML `[input.forcing]` block maps to (HM-4).
- **temp() lifecycle:** HM-2 wf1 `inmaps_historical.nc` — not `temp()`; the wf3
  twin (WG-6) — **`temp()`**, absent on the completed fixture.
- **UNITS NOT PINNED (design arch-2 / risk-4 / repo-3).** wflow is name-keyed, so
  no consumer reads the unit attr. **Observed attr layout** (recorded, asserted
  **only if present**): `precip` carries **both** `units='mm d**-1'` (plural) and
  `unit='mm'` (singular); `pet` `unit='mm'` (`units` absent); `temp`
  `unit='degree C.'` (`units` absent). The wflow-native values live under the
  `unit` **singular** key; the `units` plural key survives on `precip` only as an
  extraction leftover. Contrast WG-1, whose values are all under `units` plural.
- **deliberately unpinned:** **all forcing units** (`unit` / `units` attr
  values) — asserted-if-present, not required; `precip_fn` / `pet_method` /
  `temp_correction` provenance attrs.
- **validator:** `validate_hm2` (asserts unit attrs **only if present**, per the
  asserted-if-present semantics). Covers HM-2 wf1 `inmaps_historical.nc`
  (persists) and the WG-6 wf3 twin (temp(), skip-until-captured — see WG-6 /
  `validate_wg6` on the weather-generator seam).

## HM-3 — static vector geometries (staticgeoms)

- **path pattern:** `models/hydrology/wflow/staticgeoms/*` (`region.geojson`,
  `basins.geojson`, `outlets.geojson`, `rivers.geojson`, `outlet_index.csv`, …).
- **producer:** rule 1.07 side-effect + rules 1.09 / 1.11.
- **consumers:** wf1 plot rules; wf3 rule 3.08 (`region.geojson` via
  `ancient()`).
- **pinned surface (OUR-consumed vectors only):** `region.geojson` (basin extent
  polygon, EPSG:4326 — the wf3 extraction region + the `ancient()` DAG edge);
  `outlets.geojson` (gauge points → plots/outputs); `outlet_index.csv` (the
  `outlet position → subcatchment-ID` mapping, a `rule all` target). CRS
  EPSG:4326; geometry types (Polygon / Point).
- **temp() lifecycle:** not `temp()`.
- **deliberately unpinned:** the full attribute tables; the `basins` / `rivers` /
  `meta_*` layers we don't index.
- **validator:** `validate_hm3`.

## HM-4 — run configuration (wflow_sbm.toml)

- **path pattern:** `models/hydrology/wflow/wflow_sbm.toml` (base) and per-cst
  `<exp>/hydrology/wflow/config/rlz_<n>_st_<m>.toml`. The run TOMLs sit in
  their own `config/` directory beside `forcing/` and `output/`, so
  `input.path_forcing` is the sibling hop `../forcing/…` and
  `[state].path_output` / `[output.csv].path` are `../output/…`. `dir_output`
  stays `"."` and the hop rides in the pointers themselves
  (`snake_utils.member_pointer_base`). hydromt re-relativizes the absolute
  pointers on write -- none is hand-maintained.
- **producer:** tracked template / rule 3.14 rewrite.
- **consumer:** rule 1.14 / rule 3.15 `run_wflow` (`Wflow.run()`).
- **pinned surface (the TOML fields OUR code reads/rewrites — the wf3 rewrite
  sites, `downscale_climate_forcing.py:55-84` `setup_config`):**
  `[time].{calendar, starttime, endtime, timestepsecs}`, `dir_output`,
  `[state].{path_input, path_output}`, `[input].{path_static, path_forcing}`,
  `[output.csv].path`.
- **Rewrite-value facts (design arch-4, fixture-verified at
  `downscale_climate_forcing.py:55-84`):**
  - `time.timestepsecs = 86400`.
  - `time.calendar` rewritten to **`"standard"`** — distinct from the wf1 base +
    the HM-2 pin of `proleptic_gregorian`. The code comment (lines 57–61)
    grounds it: weathergenr writes `noleap`, and hydromt_wflow 1.x forcing
    validation would fail comparing `cftime.DatetimeNoLeap` vs
    `datetime.datetime`, so **both** the wf3 forcing time axis and the TOML are
    moved to `standard`.
  - `dir_output = "."` (flat, no `run_default/` subdir).
  - `state.path_output = "outstates_<climate_name>.nc"` — so the wf3 warm state
    lands **flat**, unlike wf1 (HM-6a).
- **Also pinned (read-reliance):**
  - `[input.forcing]` — the block **keys are wflow CSDMS Standard Names** (e.g.
    `atmosphere_water__precipitation_volume_flux`,
    `land_surface_water__potential_evaporation_volume_flux`,
    `atmosphere_air__temperature`) and the **VALUES are the forcing netCDF
    variable names** `precip` / `pet` / `temp`. The tie to HM-2 is on the **RHS
    values** (design arch-6, direction corrected).
  - `[output.csv].column` entries `{header, map, parameter}` — drives HM-5 column
    identity.
  - `[model].cold_start__flag` (section verified against both the base and
    per-cst fixture TOMLs — the flag lives under `[model]`, not top-level).
- **temp() lifecycle:** not `temp()`.
- **deliberately unpinned:** all `[input.static]` physics value blocks, layer
  thicknesses, kinematic-wave params — **wflow physics, unpinned**.
- **validator:** `validate_hm4`.

## HM-5 — per-run discharge CSV (output.csv)

- **path pattern:** wf1 `models/hydrology/wflow/run_default/output.csv`; wf3
  `<exp>/hydrology/wflow/output/rlz_<n>_st_<m>.csv` — R9 P2 dissolved the
  `rlz_<n>/` level and moved the realization index back into the file name, so
  one member is one filename in three flat directories (`config/`, `forcing/`,
  `output/`). This is the inverse of R07 B5.
- **producer:** rule 1.14 / rule 3.15 `run_wflow`.
- **consumers:** wf1 rule 1.15 plots; wf3 rule 3.16 `derive_wflow_indicators`.
- **pinned surface — column identity is config-driven, NOT a literal list:** a
  `time` index (ISO-8601, daily) + **one column per `[output.csv].column`
  entry**, named `<header>_<mapid>`. Fixture: `time,Q_130000086` (one gauge).
- **the single degree of freedom:** the gauge-column set flows **TOML
  `[output.csv].column` → `output_rlz` → q_indicators** as one degree of freedom — the
  key bounded-substitution invariant, checked end-to-end by
  `validate_hm_gauge_column_identity` (below).
- **consumer-prefix reliance (grounded, `export_wflow_results.py`):** rule 3.16
  selects gauge columns by a **hard-coded `Q_` prefix** (`gauge_columns`) and
  every other variable's columns by its `wflow_outputs.CODES` code plus a numeric
  subcatchment id (`subcatchment_columns`) — so the TOML `header` values are
  load-bearing **beyond mere identity**.

  **This reliance broke, undetected, and how it broke is the point.** 8bd51de
  (2026-08-10) changed the basin-average header from `<label>_basavg` to
  `<code>_<subcatchment>`; the consumer kept matching the retired spelling, found
  no column, and `continue`d — writing `aet_indicators.csv` and
  `recharge_indicators.csv` as a header and zero rows, with every rule green and
  nothing in any log. Three things had to fail together: the producer changed a
  header the consumer parses without either being a declared shared surface; the
  consumer treated "no column" as *skip* rather than *raise*; and
  `test_export_wflow_results.py`'s fixture wrote the `_basavg` header itself, so
  the unit suite agreed with the consumer and with nothing else. The matcher is
  now keyed off `CODES` — the same table the model build writes the TOML from —
  and a requested variable with no matching column raises `MissingOutputColumnError`
  rather than emptying its table. See also the `validate_hm7` note below: it has
  a "no rows" check that would have caught this, but is never invoked at run time.
- **temp() lifecycle:** SPLIT since 2026-08-10. wf1 `output.csv` **is** `temp()`
  (rule 1.14): it is an intermediate feeding rule 1.14b's derived per-variable
  tables and rule 1.15's metrics, and Snakemake drops it once both have run. A
  swapper must therefore treat the wf1 artifact as existing only *within* the
  run — `--notemp` is what materialises it, and the baseline procedure uses that
  flag for exactly this reason. The wf3 `output_rlz_*` CSVs still persist.
- **deliberately unpinned:** numeric discharge values (not a contract; they
  change per run).
- **validator:** `validate_hm5` (per-artifact column-identity); cross-file
  identity by the relational `validate_hm_gauge_column_identity`.

## HM-6a — wf1 warm state (persisted, no validator)

- **path pattern:** `models/hydrology/wflow/run_default/outstate/outstates.nc`.
- **producer → consumer:** rule 1.14 `run_wflow` → **(nothing in-repo)**.
- **THIN — "named output sink, unconsumed."** Persisted on the fixture.
- **contract surface:** name + location only — which **HM-4 already pins** via
  `[state].path_output`. **No validator (design risk-1):** a standalone existence
  check would pad the green count without verifying an independent contract. Kept
  as a **doc row only**; existence guaranteed **transitively through HM-4**.
- **path derivation (design arch-3):** the on-disk path
  `run_default/outstate/outstates.nc` = base TOML `dir_output = "run_default"`
  **+** `[state].path_output = "outstate/outstates.nc"`. A swapper that changes
  `dir_output` moves this path with it.
- **temp() lifecycle:** not `temp()`.
- **deliberately unpinned:** the entire state-variable schema
  (`[state.variables]` — wflow-owned).
- **validator:** **none** (existence pinned transitively via HM-4).

## HM-6b — wf3 warm state (temp, skip-until-captured)

- **path pattern:** `<exp>/hydrology/wflow/output/outstates_rlz_<n>_st_<m>.nc`
  (flattened with HM-5 at R9 P2).
- **producer → consumer:** rule 3.15 `run_wflow` → **(nothing in-repo)**.
- **THIN — "named output sink, unconsumed" (corrects the intake's chaining
  hint).** Verified: the per-cst TOML keeps `cold_start__flag = true` and
  declares **no `instates` input** on rule 3.15; wf3 fans out in parallel over
  `(rlz, cst)` with no cross-cst edge — **no warm-state chaining invariant** our
  DAG relies on (design §5.3 warm-state finding).
- **contract surface:** the file is a declared wflow state **output** whose name
  (`outstates_rlz_<n>_st_<m>.nc`, under the experiment's `output/` — wf3 keeps
`dir_output="."` and carries the `config/` → `output/` hop in the pointer, HM-4)
  our rewrite sets.
- **temp() lifecycle:** **`temp()`** in wf3 — deleted, absent on the fixture.
- **structural note:** the wf1/wf3 split mirrors HM-2/WG-6 structurally, but is
  **disanalogous on content** — forcing (HM-2/WG-6) is consumed, warm state is
  not — so the split does **not** imply an independent wf1 validator (hence HM-6a
  carries none).
- **deliberately unpinned:** the entire state-variable schema
  (`[state.variables]` — wflow-owned).
- **validator:** `validate_hm6b` — **skip-until-captured on disk** (temp()
  content absent by default); logic proven every suite by a synthetic pass/fail
  pair. See the `--notemp` capture procedure below.

## HM-7 — response-surface reduction (one indicator table per variable)

- **path pattern:** `<exp>/results/<token>_indicators.csv`, **one per variable in
  `workflows.build_model.wflow_outvars`** (R11 CR-2). `basin_indicators.csv` is
  gone; its contents are now per-variable tables. The set is config-dependent, so
  the DAG derives it — see the token vocabulary below.
- **producer:** rule 3.16 `derive_wflow_indicators` (renamed from
  `export_wflow_results` at R9 P3; the *module* it runs keeps the old name, so
  the `export_wflow_results.py:NN` citations below are current).
- **consumer:** CST-API / GUI (terminal in-repo).
- **pinned surface:** every table carries **exactly seven columns, in this order**:

      metric, location, st_id, rlz_id, temp_change, precip_change, value

  Reordered identifier-first and `realization_id` renamed to `rlz_id` by owner
  ruling 2026-08-11. The order is *what* (`metric`, `location`), *which member*
  (`st_id`, `rlz_id`), *where on the surface* (`temp_change`, `precip_change`),
  then the number — so the two id columns sit adjacent instead of being split
  around the perturbation axes. `rlz_id` matches the `rlz_` member token the run
  filenames already carry (`rlz_1_st_0.csv`) and `RLZ_NUM`. Same seven columns,
  same fixity: a reorder and one rename, not a shape change. (The count said
  "six" here until that ruling, having gone stale when C28 added `st_id`.)

  The header does not grow with the gauge count — locations are ROWS. `metric` is
  a composite `<token>_<statistic>`, so a result file is self-contained once it
  leaves the project tree and needs no `variable` column; `validate_hm7` asserts
  the metric agrees with the table it sits in, which is what normalisation would
  have given for free. `value` is `float32` and **unrounded**.

- **variable tokens (the contract CR-2 places here):**

  | `wflow_outvars` | token | table |
  | --- | --- | --- |
  | river discharge | `q` | `q_indicators.csv` |
  | precipitation | `precip` | `precip_indicators.csv` |
  | actual evapotranspiration | `aet` | `aet_indicators.csv` |
  | groundwater recharge | `gwr` | `gwr_indicators.csv` |
  | overland flow | `overland_flow` | `overland_flow_indicators.csv` |
  | snow | `snow` | `snow_indicators.csv` |

  Minting rule: where the repo already has a canonical short name, use it; only
  mint where none exists; disambiguate against names in use. Hence `precip` not
  `p` (`naming.md` §6 tier 2), `aet` not `et` (`pet` is canonical and one letter
  away), `snow` not `swe` (the CSDMS name is `snowpack_liquid_water__depth` —
  snowpack *liquid water*, so `swe` would assert a claim upstream does not make),
  and `gwr` not `recharge` — the *first* clause of the rule rather than the
  others' "only mint where none exists": `gwr` is the `wflow_outputs.CODES` code
  already in every run csv header, so `recharge` was a spelling the repo did not
  need. **Renamed 2026-08-11**; the metric moved with the table
  (`recharge_annual_total` → `gwr_annual_total`), since the metric is composed
  `<token>_<statistic>`. Record: `dev/milestones/r11/migration_indicator-tables.md`.

  Every spelling of every variable — config label, CSDMS name, csv code, token,
  metric, axis legend — is tabulated in `dev/reference/indicator-glossary.md`,
  which `tests/test_indicator_glossary.py` checks against the code tables. This
  table stays the CONTRACT for the token column; the glossary is the wider,
  derived view.

- **`rlz_id`, and the grain it encodes:** `0` means **pooled over realizations**;
  `1..RLZ_NUM` name one. Metrics linear in years are emitted per realization; the
  two GEV fits and the two month-selecting metrics are pooled only. The numeric
  sentinel is safe **only because no metric emits both grains** — if that ever
  changes it must become a string, or `groupby("rlz_id")` folds pooled rows in as
  another realization. `validate_hm7` asserts it, since `0` cannot announce
  itself. (Spelled `realization_id` before the 2026-08-11 ruling.)

- **`location`:** the **bare** id (`130000086`, not `Q_130000086`), which is the
  id wflow emits, so it joins `outlet_index.csv` with no crosswalk. In
  `q_indicators.csv` those are the outlets- and gauges-map ids; in every other
  table they are **subcatchment** ids, since those variables are declared
  `map = "subcatchment"` and a run emits one column per subcatchment. The two id
  sets are not 1:1 and need not be — a basin can have more gauges than
  subcatchments or the reverse.

  `basin` is **reserved** for a basin-scalar value, emitted independently rather
  than derived from per-location values (Q11): whether subcatchments nest or tile
  decides whether an area-weighted mean is valid at all, and a derived value would
  silently encode whichever answer the implementer assumed. **Nothing emits it
  today**, and Q11 is exactly why the reducer does not start: producing a genuine
  basin scalar means declaring a whole-basin column in the TOML, which is a WF1
  change rather than a reduction one.

- **`aggregate_rlz` is retired** (ruling b1). In the long shape "aggregated" is
  not a *shape* choice, so the table always carries the finest grain available and
  downstream aggregates as it likes.

- **axis-column rename (2026-08-05):** these two columns were `tavg` / `prcp`
  until the R9 followup recorded in
  `dev/milestones/r09/migration_indicator-axis-columns.md`. They were the repo's
  only violation of the `precip` / `temp` vocabulary `naming.md` §6 tier 2
  declares. The `_change` suffix is load-bearing: the columns hold the
  **perturbation** each member imposes — absolute degC and relative % — not the
  variable's value, so bare `temp` / `precip` would have been wrong in the other
  direction. Both spellings are named once in code, as
  `interchange_contracts._PERTURBATION_AXIS`.
- **axis VALUE, not just its name (2026-08-07, [R9-3]):** each member's
  perturbation is monthly — `st_<m>.csv` carries twelve rows — so the two axis
  columns are a **month-length-weighted annual mean** of those twelve values,
  taken by `export_wflow_results.annual_perturbation`. They held the JANUARY
  value until this date, which was indistinguishable from the annual figure for
  a flat perturbation vector and wrong for any seasonal one. The rule is fixed
  by the CMIP6 overlay rather than chosen: the GCM dots share these axes, and
  WF2 defines its annual change factor the same way
  (`get_change_climate_proj._annual`). The precip axis is that definition with a
  uniform-daily-rate weight standing in for the baseline climatology — exact for
  a flat vector, approximate under seasonality; see the function's docstring.
  A consumer may rely on the axis staying **evenly spaced** across the grid: the
  collapse is affine in the member's step index, so the surface is rectilinear.
- **temp() lifecycle:** not `temp()` (`rule all`, manifested).
- **removed at R9 P3:** the `RT_*.csv` response tables. They were non-manifest
  side products with no in-repo consumer, written via `params` rather than
  declared and therefore invisible to `--dry-run`. Nothing replaces them.
- **validator:** `validate_hm7`; the gauge-column tie to HM-4/HM-5 by the
  relational `validate_hm_gauge_column_identity`.

---

## The HM-4 → HM-5 → HM-7 gauge-column-identity invariant

The gauge-column set is a **single degree of freedom** flowing through three
artifacts; the per-artifact validators cannot see a break *between* them (each
can pass while a renamed or omitted gauge column silently corrupts the response
surface), so a dedicated **relational validator** exists:
`validate_hm_gauge_column_identity(toml_cfg, output_rlz_df, qstats_df) ->
list[str]`.

**Grounded consumer mechanics (`export_wflow_results.py`).** The reduction
derives its gauge set from the **first** csv's columns via a **hard-coded `Q_`
prefix filter** (`Q_vars = [x for x in sim.columns if x.startswith("Q_")]`, line
61) and indexes every other csv with that set (`sim = sim_all[Q_vars]`, lines
123 / 136). So:

- a **renamed gauge header in the first csv** empties `Q_vars` → a **silently
  gauge-less q_indicators**;
- a **mismatch in a later csv** `KeyError`s deep in the reduction.

Exactly the failure no per-artifact validator can see.

**The validator checks:**

1. every non-`time` `output_rlz_df` column traces to a declared
   `[output.csv].column` entry in `toml_cfg` (map-typed entries →
   `<header>_<id>` pattern; entries without `map` → exact `header`), and every
   declared entry is represented;
2. the map-typed gauge columns carry the `Q_` prefix rule 3.16 hard-codes;
3. `qstats_df`'s gauge columns (header minus `statistic` and the
- **`st_id` (C28, R11 P2).** The design point's id, zero-padded to the same
  count-derived width as the member filename, so the two are ONE token. It is
  emitted ALONGSIDE the perturbation columns rather than replacing them — ruled
  "at this stage", for plottable-without-a-join, with an explicit revisit when a
  third stress dimension arrives. Two obligations hold that in place:
  `validate_hm7` asserts `temp_change`/`precip_change` equal the design table's
  row for that `st_id` (they are a cached copy, derived independently by the
  writer, so they really can drift), and the writer REFUSES a design table
  carrying an axis this header cannot express.

  **Read it as a string.** `pd.read_csv` with no `dtype` infers `st_id` as an
  integer, so `01` returns as `1` and the join to `stress_test_design.csv`
  silently misses. Both tables carry the padded text on disk; a consumer must
  pass `dtype={"st_id": str}`.

   `_PERTURBATION_AXIS` columns `temp_change,precip_change`, ordered per
   `export_wflow_results.py:66-67`) are **list-equal** to the `output_rlz_df`
   gauge set.

**C3 scope boundary.** The numeric `<id>` in `Q_130000086` is wflow's
outlets-map cell value; the validator checks the `<header>_<id>` **pattern** and
the cross-file identity, **NOT** the id's derivation from `staticmaps.outlets`
(wflow-owned naming semantics — recorded as reliance, never asserted). The
parallel `basavg`-substring filter (line 62) is the same class of consumer-prefix
reliance; the fixture TOML declares no `basavg` column, so that branch is
**documented, not fixture-verified**.

**Fixture wiring.** All inputs persist (12 per-cst TOMLs, 12 `output_rlz_*` CSVs,
`q_indicators.csv` — rlz {1,2} × cst {0..6}), so this relational check is in the
**continuously-verified** class. The test parametrizes over all 12 fixture
`(toml, output_rlz)` pairs against the one `q_indicators.csv`.

---

## Considered and corrected — the warm-state finding (design §5.3, C4)

The intake's seam inventory hinted at "per-cst run chaining" via
`instates`/`outstates`. The fixture + TOML diff (`wflow_sbm_rlz_1_st_1.toml` vs
base `wflow_sbm.toml`) shows `cold_start__flag = true` in **both**, rule 3.15
declares only a `forcing_path` + `toml_path` input (no `instates`), and the
`path_input` the rewrite sets (`downscale_climate_forcing.py:79`) points at a
**non-existent** `models/hydrology/wflow/instate/instates.nc` that cold-start never
reads. So HM-6a / HM-6b are contracted as an **unconsumed named sink**, not a
chaining invariant (a chaining contract would be fiction).

---

## Bounded-substitution walkthrough — replacing Wflow-SBM

A drop-in model (design §5.6) must:

- **Consume** HM-2 / WG-6 forcing (`(time, lat, lon)` `precip` / `pet` / `temp`
  on its own grid — the variable **names** are the contract; a run-config maps
  them, as wflow does via `[input.forcing]` RHS values) and a static description
  equivalent to HM-1 (the OUR-referenced name set on a co-registered grid),
  driven by a run config equivalent to HM-4's rewrite fields — **including
  `time.timestepsecs`** and a calendar (wflow's rewrite sets `standard` on the
  wf3 forcing/TOML to match the weathergenr `noleap` origin, distinct from the
  wf1 `proleptic_gregorian` pin; arch-4).
- **Produce** HM-5 per-run discharge CSVs with the **`<header>_<mapid>` column
  identity** the reduction (rule 3.16) keys on — the single degree of freedom
  that flows HM-4 `[output.csv].column` → HM-5 → HM-7, and the `Q_` header prefix
  the reduction hard-codes (`export_wflow_results.py:61`). A swap that renames the
  gauge column silently breaks HM-7 — exactly the failure the relational
  `validate_hm_gauge_column_identity` exists to catch.
- **Repo files it replaces:** rule 1.07 build (`hydromt build`), rule 1.10 / 3.14
  forcing prep, rules 1.14 / 3.15 `run_wflow` `shell:` bodies
  (`julia … Wflow.run()`), and `downscale_climate_forcing.py`'s TOML-rewrite
  (HM-4 fields) if the run-config format changes.
- **Files it must NOT change:** rule 3.16 `derive_wflow_indicators` (the reduction)
  — provided HM-5's column identity is honored.
- **Contracts it must satisfy:** HM-1 (static reliance), HM-2 (forcing in), HM-4
  (run-config rewrite fields), HM-5 (output column identity), HM-7 (reduction
  input). HM-6a / HM-6b warm state is **not** a substitution constraint
  (unconsumed sink). Acceptance check: validators `validate_hm1`,
  `validate_hm2`, `validate_hm4`, `validate_hm5`, `validate_hm7` plus the
  relational `validate_hm_gauge_column_identity`.

---

## Validator index

Validators live in `blueearth_cst/shared/interchange_contracts.py` (added by a
later commit; this index is the spec that commit implements against). Each is a
pure `-> list[str]` divergence report (empty ⇒ pass); no `assert` /
`AssertionError` in the bodies (`-O`-safe liftability, design §6.5). Every
validator additionally carries a Layer-1 synthetic pass/fail test pair that
executes on **every** checkout, fixture or not. HM-2 unit attrs are asserted
**only if present** (asserted-if-present semantics, design §5.5).

| validator | artifact(s) | fixture path (era5) | continuously verified? |
|---|---|---|---|
| `validate_hm1` | HM-1 | `models/hydrology/wflow/staticmaps.nc` | **yes** (persists) |
| `validate_hm2` | HM-2 (+ WG-6 twin) | `models/hydrology/wflow/forcing/inmaps_historical.nc`; wf3 twin `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_st_<m>.nc` | **yes** for wf1 `inmaps_historical.nc`; wf3 twin (WG-6) `temp()` → skip-until-captured |
| `validate_hm3` | HM-3 | `models/hydrology/wflow/staticgeoms/{region.geojson, outlets.geojson, outlet_index.csv}` | **yes** (persists) |
| `validate_hm4` | HM-4 | `models/hydrology/wflow/wflow_sbm.toml`; `<exp>/hydrology/wflow/config/rlz_<n>_st_<m>.toml` | **yes** (both base + per-cst TOMLs persist) |
| `validate_hm5` | HM-5 | wf1 `run_default/output.csv`; wf3 `<exp>/hydrology/wflow/output/rlz_<n>_st_<m>.csv` | wf1 `output.csv` is `temp()` since 2026-08-10 → skip-until-captured, or run with `--notemp`; **yes** for the wf3 per-cst CSVs (NOT `temp()`) |
| `validate_hm_gauge_column_identity` (relational) | HM-4 → HM-5 → HM-7 gauge-column identity | per-cst TOMLs + the per-cst run CSVs + `q_indicators.csv` | **yes** (all inputs persist) |
| *(HM-6a)* | HM-6a | `models/hydrology/wflow/run_default/outstate/outstates.nc` | **no validator** — existence pinned transitively via HM-4's `[state].path_output` |
| `validate_hm6b` | HM-6b | `<exp>/hydrology/wflow/output/outstates_rlz_<n>_st_<m>.nc` | **no** — `temp()` content absent; skip-until-captured on disk, synthetic-proven every suite |
| `validate_hm7` | HM-7 | `<exp>/results/<token>_indicators.csv` (one per `wflow_outvars` entry) | **yes** (persists; `rule all`, manifested) |

### `--notemp` capture procedure (temp() on-disk validators)

The `temp()`-content validator `validate_hm6b` (HM-6b, the wf3 warm state) has
**no on-disk integration check on the default fixture**: `outstates_rlz_<n>_st_<m>.nc`
is wrapped in Snakemake `temp()` and deleted after rule 3.15 finishes, so it does
not survive a completed run. Its Layer-2 integration case
(`test_hm6b_integration`) carries **both** the `_FIXTURE_ABSENT` skipif and a
runtime `pytest.skip("temp() artifact absent; capture via --notemp")` guarding on
the NC's presence. Its logic is proven on **every** checkout by its Layer-1
synthetic pass/fail pair regardless.

**This milestone does NOT run the capture** — passing `--notemp` and letting the
artifact persist would modify the untracked `test_case/test_local` fixture, out of
a contracts-only milestone. The procedure below is the one-command lift a
**future run** performs when full on-disk coverage is wanted (design OQ-4).

**Capture sketch** (run from the repo root inside `pixi shell`, after the wf1
model exists — wf3 needs `models/hydrology/wflow/` artifacts):

```bash
snakemake all -c 3 -s run_stress_test.smk \
  --configfile config/workflows/snake_config_model_test.yml --notemp
```

`--notemp` tells Snakemake **not** to delete `temp()`-flagged outputs after their
consuming jobs complete, so the run leaves the intermediate netCDFs on disk.

**Paths that then appear** under `test_case/test_local` (the path the skip-guard
tests for):

| validator | artifact captured | fixture path (`<exp>` = `experiments/experiment`) |
|---|---|---|
| `validate_hm6b` | HM-6b wf3 warm state NC | `<exp>/hydrology/wflow/output/outstates_rlz_<n>_st_<m>.nc` |

(The same run also captures WG-4 `rlz_<n>_st_<m>.nc` and WG-6
`inmaps_rlz_<n>_st_<m>.nc` — documented in the weather-generator seam doc.)

**Which cases un-skip:** with these artifacts present, `test_hm6b_integration`
here (plus `test_wg4_integration` and `test_wg6_integration` on the other seam)
stop hitting their `pytest.skip` and run their on-disk assertion — the **three**
temp validators' *on-disk* integration checks flip from skip-until-captured to
green. No test code or validator changes; the guards resolve to the real-artifact
path automatically once the files exist. Re-running **without** `--notemp` (or a
`snakemake --delete-temp-output`) restores the default temp-deleted fixture state.
