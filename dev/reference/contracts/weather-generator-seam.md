# Contract: weather-generator seam (WG-1 .. WG-6)

> **Genre:** dev-facing interchange contract. **Audience:** a future *swapper* —
> someone replacing `weathergenr` with an alternative weather generator, or the
> R6 model-flexibility work — read end-to-end. Not an end-user doc (hence `dev/`,
> not `docs/`; precedent `dev/reference/workflows/run_stress_test.md`).
> **Source of record:** `dev/milestones/p32b/interchange-contracts-design.md` (ACCEPTED
> 2026-07-24, §5.2 / §5.4 / §5.6 / §5.5). Every load-bearing fact below cites a
> Snakefile line, a script line, or an observed fixture artifact; do not add a
> contract fact that is not so grounded.

## Scope and method

The **weather-generator seam** is the point in `run_stress_test.smk`
(wf3) where the stochastic weather generator could be swapped for an alternative
without re-architecting the rest of the pipeline. `weathergenr` (R) is the
current occupant, but **this contract is generator-agnostic**: it pins what wf3
hands *in* to the generator and expects *out* of it, not weathergenr's internals.

**Grounded in** the fixture tree `test_case/test_local` (era5 branch,
`config/workflows/snake_config_model_test.yml`) inspected with xarray for
dims/coords/vars/units/attrs, and the wf3 rules + scripts. **CST-scope
disclaimer** (`AGENTS.md` Hard Constraints): a contract surface pins only what
OUR pipeline's producer guarantees or OUR consumer relies on; upstream tool
internals (hydromt catalog machinery, weathergenr's algorithm) are consumed
verbatim and are *not* re-specified here. The producer-side (R) surface (WG-3
config keys, WG-4 output shape) is derived **read-only** from
`blueearth_cst/weathergen/{global.R,generate_weather.R,impose_climate_change.R}`
— those files are never edited.

**Fixture branch = era5.** Branch-specific facts (chirps precip-only, the chirps
orography sidecar) are documented from code and tagged **not fixture-verified
(no chirps fixture)** where no chirps fixture exists — never faked green.

**Contract-surface tiers** (design §5.1), applied per artifact below:

1. **Pinned (contract surface)** — a structural fact a swap MUST reproduce for
   the downstream consumer to work.
2. **Pinned-as-reliance** — OUR consumed subset of an upstream schema (e.g. the
   hydromt data-catalog schema); we pin the fields we emit/read, not the whole
   upstream schema.
3. **Deliberately unpinned** — internal detail (provenance attrs, encoding,
   machine-scoped paths) recorded as unpinned so the omission is auditably
   intentional, not an oversight.

Per-artifact schema (design §5.4): *artifact id · path pattern · producer rule ·
consumer rule(s) · dims · coords (dtype/units/calendar) · data_vars
(dtype/units) · CRS · time axis/calendar · naming pattern · temp() lifecycle ·
pinned surface · deliberately unpinned · validator*. Rendered as one subsection
per artifact (a literal 14-column table is illegible).

---

## WG-1 — historical climate extraction

- **path pattern:** `data/climate/historical/<key>/extract_historical.nc`, where
  `<key> = <clim_source>_<startYYYYMMDD>_<endYYYYMMDD>` (P3-1 keyed store).
- **producer:** rule `extract_historical_climate`
  (`blueearth_cst/climate_analysis/extract_historical_climate.py`) — ONE rule,
  declared identically in `run_stress_test.smk` (3.08) and
  `build_model.smk` (1.04) from `snake_utils.climate_store_rule`
  (R07 B1). Its inputs are the data catalog and the project region artifact
  `spatial/geoms/region.geojson`; the extent is still model-free, but it is
  delineated once per project by rule `delineate_region` (ADR 0003) rather
  than per store key. The store records the extent it cut to in the
  extraction's own attributes (`region_bbox`, `region_geojson_sha256`,
  `region_source`).
- **consumer:** rule 3.11 `generate_weather_realizations` (weathergenr
  `generate_weather.R`), passed in as `climate_nc`.
- **dims:** `(time, latitude, longitude)`.
- **coords:** `time` — `datetime64[ns]`, daily, `calendar=proleptic_gregorian`;
  `latitude` / `longitude` — `float32`, `degrees_north` / `degrees_east`;
  `spatial_ref` — EPSG:4326 (WKT).
- **data_vars** (all `float32 (time, latitude, longitude)`): `precip`
  (`mm d**-1`); `temp` / `temp_min` / `temp_max` (**K** — see units note);
  `kin` / `kout` (`J m**-2`); `press_msl` (`Pa`).
- **CRS:** EPSG:4326 (global attr `crs=4326`, `category=meteo`).
- **time axis/calendar:** daily `proleptic_gregorian`.
- **naming pattern:** `<clim_source>_<start>_<end>/extract_historical.nc`.
- **temp() lifecycle:** not `temp()`; consumed via `ancient()` on the DAG.
- **pinned surface:** the dims, the coord axes + CRS, the seven variable names
  and their `float32` dtype. **Every WG-1 unit is under the `units` (plural)
  attr key** — fixture-verified, NOT `unit` singular (contrast HM-2, which
  carries wflow-native values under `unit` singular — see the
  hydrological-model seam doc). `crs=4326` / `category=meteo` global attrs.
- **deliberately unpinned:** provenance attrs (`paper_*`, `source_*`, `notes`);
  chunk/encoding.
- **validator:** `validate_wg1`.

**Branch note (not fixture-verified — no chirps fixture).** The era5 branch
writes all seven variables. The **chirps** branch writes `precip` from
chirps-native data and reprojects era5 `temp`/radiation/`press_msl` onto the
chirps grid; the chirps orography sidecar is a chirps-only input. These
chirps-only facts are documented from code and asserted only under a chirps
fixture — tagged **not fixture-verified (no chirps fixture)** in the validator
index.

**Units note (grounded — corrects the p32a °C assumption; design §5.2).** WG-1
`temp*` is in **Kelvin** (`long_name` + observed value, under the `units` plural
key): the extraction writes native era5 K. The Kelvin→°C conversion happens
inside the forcing build / downscale, so the °C value lands on the model-grid
forcing (HM-2 `temp.attrs['unit'] = 'degree C.'`, fixture-verified). **Units are
NOT pinned as a hard contract surface** on either artifact — wflow maps forcing
by variable NAME via the TOML `[input.forcing]` block (HM-2), never by the
netCDF unit attribute — so the K-vs-°C divergence is an **observed, documented
cross-seam fact**, asserted only **if the attr is present** (§5.5), not pinned as
a required property. This avoids over-constraining a swap with a property no
consumer reads while keeping the divergence honestly on the record.

## WG-2 — stress-test perturbation grid

- **path pattern:** `<exp>/config/stress_test_lookup.csv` — **one file for the
  whole experiment**. Replaced the per-member
  `<exp>/climate/weathergenr/_work/st_<m>.csv` on 2026-08-16, together with the
  derived cache `<exp>/config/stress_test_design.csv`, which it absorbs.
  `_work/` is deleted. The rename to `lookup` signals that **the shape moved**,
  not merely the path. It sits beside the config snapshot whose settings
  produced it: it is a record of what ran, not scratch.
- **producer:** rule 3.09 `prepare_stress_test_grid`
  (`blueearth_cst/experiment/prepare_cst_parameters.py`).
- **consumer:** rule 3.12 `perturb_climate_realization` (weathergenr
  `impose_climate_change.R`), passed in as `lookup_csv` — a **constant** input,
  no longer carrying the member wildcard. The member id arrives as a positional
  argument and the script filters on it. It is **no longer an `input:` on rule
  3.16**: the reduction reads no parameter artifact at all, since the axis is
  derived at reporting time (HM-7).
- **shape:** a CSV with **header exactly**
  `st_id,month,temp_change,precip_change,precip_variance_change`, and
  **`12 × ST_NUM` rows** — twelve per member, `month ∈ 1..12`, members
  `1..ST_NUM`. The `(st_id, month)` grid is **complete and duplicate-free**, and
  rows are sorted by `(st_id, month)`.
- **semantics:** `temp_change` additive (°C); `precip_change` and
  `precip_variance_change` **percent**, not multipliers — `0.0` means no change,
  `-30.0` means a 0.7 factor. The multiplier convention survives only inside the
  generator: the R side reconstructs `1 + <col>/100` for **both** percent
  columns. The project config keeps its 12-element multiplier vectors; this is
  an artifact-unit rule, not a config-surface change.
- **precision:** the member levels are `float32` shortest-repr quantized (the
  grid the user asked for); the percent text is written at **`float64` shortest
  repr of the exact conversion**, so the reconstructed multiplier is within one
  `float64` ulp of the level. It is **not** bit-identical for every level, and
  cannot be made so — measured, 1,155 of 50,000 `float32` multipliers admit no
  `float64` percent that reconstructs them exactly under `1 + p/100`. A consumer
  may rely on the bound, not on exactness.
- **admissible multiplier domain: `multiplier ≥ 0.5`, with no upper bound.** This
  is the **precondition of the bound above**, not a caveat on it: the producer
  refuses a configuration declaring a precipitation mean or variance multiplier
  below `0.5` before the DAG is built, so every lookup this contract describes
  was written from admitted multipliers and the one-ulp bound holds over the
  whole table, unconditionally. Outside the domain it does not: at level
  `0.013596006` the specified conversion reconstructs `0.013596005999999883`,
  68 ulps out, because the percent's rounding scale stops shrinking with the
  level once `|percent|` crosses 64. There is deliberately **no ceiling** — the
  bound was measured to hold out to `1e6`, so an upper cap would refuse
  configurations the arithmetic serves correctly. A re-implementer needs the
  floor to *validate its own producer*; a consumer reading a lookup needs only
  the bound.
- **`st_id`:** the member id, **zero-padded** to a width derived from `ST_NUM`
  (C27: `01 … 12` at twelve points, unpadded below ten), **textually identical**
  to the member filename token, so the two are ONE token. **Read it as a
  string** — `pd.read_csv` with no `dtype` returns `01` as `1` and every join
  silently misses. R readers pass `colClasses = c(st_id = "character")`.
  **Every `st_id` in one table has the same width**, which is what lets a
  consumer infer the join key's width from the table itself rather than needing
  `ST_NUM` passed alongside it. A table mixing widths is malformed.
- **`st_0` has NO row.** The table covers members `1..ST_NUM` only. `st_0` is
  the reserved unperturbed baseline (naming.md §4): it has no parameters, is
  produced by rule 3.11 rather than by perturbation, and rule 3.12 never runs
  for it. Its **absence is load-bearing**, not incidental — it is what makes
  "not on the surface" a structural fact rather than a convention, and an
  all-zero `st_0` row would be indistinguishable from an identity member's row
  while denoting a differently-processed climate (the raw generated series, not
  that series round-tripped through a perturbation that is not the identity at
  unit factors).
- **column vocabulary:** **closed**. A new perturbation parameter is a new
  COLUMN, and adding one requires a C28 ruling — the shape barrier is gone, the
  contract barrier is not. Refused at write time by
  `prepare_cst_parameters._KNOWN_AXES`.
- **temp() lifecycle:** not `temp()`; a `rule all` target (`WF3_TARGETS` entry
  `stress_test_lookup`), so it persists.
- **pinned surface:** the exact header and column order; the `12 × ST_NUM` row
  count with a complete, duplicate-free `(st_id, month)` grid; the
  `(st_id, month)` sort order; `st_id` as zero-padded TEXT at the filename's
  width, **one width for the whole table**; `st_0` ABSENT; the
  additive-vs-percent column semantics.
- **deliberately unpinned:** the numeric values themselves (they are the
  experiment), and the percent text's digit count beyond the `float64`
  shortest-repr rule.
- **validator:** `validate_wg2`.

## WG-3 — weathergenr config surface

- **path pattern:** `<exp>/climate/weathergenr/config/weathergen_config.yml` —
  **one file** since C29.
- **producer:** rule 3.10 `prepare_weathergen_config`
  (`blueearth_cst/experiment/prepare_weagen_config.py`).
- **consumer:** rules 3.11 and 3.12 (both R side), which now read the same file.
- **removed at C29 (2026-08-05):** the per-member
  `_work/weathergen_config_rlz_<n>_cst_<m>.yml` and its producer rule 3.05
  `prepare_weagen_config_st`. Nothing in that file varied except the output
  filename — split into prefix and suffix because `weathergenr::write_netcdf`
  takes them separately — and Snakemake already knew it as rule 3.12's own
  declared output, so it is passed as the 4th CLI argument and split in R. Its
  two `transient_change` flags moved into this file and are now pinned here. At
  RLZ_NUM=10, ST_NUM=88 the removal drops 880 YAMLs plus their logs and
  benchmark parts. The rest of what it carried — copies of the `stress_test`
  step counts and monthly min/max ranges — was never read (finding F6) and
  deliberately did **not** move: the values that perturb a run come from
  `stress_test_lookup.csv`.
- **shape (YAML):** the weathergenr config surface — top-level
  `general.variables` (list ⊆ `{precip, temp, temp_min, temp_max}`) and
  `generateWeatherSeries.{warm.*, knn.sample.num, month.start, warm.variable,
  seed, evaluate.*, dry.spell.change[12], wet.spell.change[12], output.path,
  sim.year.start, sim.year.num, nc.file.prefix, realizations_num}`.
- **pinned surface:** **the key set + types the R side reads** (derived
  read-only from `global.R` / `generate_weather.R`), NOT weathergenr's
  semantics. Upstream-spelled keys (`warm.signif.level`, `dot.case`) are
  preserved verbatim per naming.md §2 (YAML under an upstream schema).
- **temp() lifecycle:** not `temp()`.
- **deliberately unpinned:** comment layout, key order.
- **validator:** `validate_wg3`.

**Depth note (design OQ-6).** WG-3 pins the config *key set + types*, not value
*ranges* — a replacement generator may define its own config surface entirely, so
WG-3 is the *current* generator's contract, not a universal one.

## WG-4 — generator output netCDFs (baseline + perturbed)

- **path pattern:** `<exp>/climate/weathergenr/output/rlz_<n>_st_0.nc` (baseline)
  and `<exp>/climate/weathergenr/output/rlz_<n>_st_<m>.nc` (`m ≥ 1`, perturbed).
  R07 B5 dissolved `realization_<n>/`; the index stays in the file name.
- **producer:** rule 3.11 (st_0) / rule 3.12 (st_m).
- **consumer:** rule 3.14 `downscale_climate_realization` (rule 3.13
  `write_climate_data_catalog` consumed the whole set until 2026-08-18; 3.14
  now reads only its own member).
- **shape:** the **generator OUTPUT contract** — a raster netCDF the hydromt
  catalog reads: `(time, lat, lon)` daily grid with **at least `precip`, `temp`**
  (+ `pet` if present) on an EPSG:4326 grid carrying a `spatial_ref` CRS
  descriptor (so `raster_xarray` + `harmonise_dims` load it — WG-5).
- **naming pattern:** `rlz_<n>_st_<m>.nc` — a **DAG-globbed pattern**
  (rule 3.14 wildcards; rule 3.12 `expand` over the grid).
- **temp() lifecycle:** **`temp()`** (both st_0 and st_m). Deleted after
  consumers finish — **absent on the completed fixture**.
- **pinned surface:** the `(time, lat, lon)` raster shape, the minimal
  `{precip, temp}` variable set, the `spatial_ref` CRS descriptor, the
  DAG-globbed naming pattern.
- **deliberately unpinned:** exact variable superset, internal attrs.
- **`crs` / `category`: asserted-IF-PRESENT, NOT required — corrected 2026-07-25
  by the first `--notemp` capture.** This contract was written expecting them as
  netCDF **global attrs**; the real artifact carries **empty global attrs**. Its
  CRS travels the CF/rioxarray way — the `spatial_ref` coordinate's `crs_wkt`,
  ending `ID["EPSG",4326]` — and `crs: 4326` / `category: meteo` are supplied by
  the generated **data catalog** (WG-5's `metadata.crs` / `metadata.category`),
  which is the surface hydromt actually reads and which `validate_wg5` already
  pins. So the original wording asserted the right values on the wrong surface:
  **the pipeline was never non-conformant — the contract was.** `validate_wg4`
  now flags a *present but contradictory* value and accepts absence.
- **validator:** `validate_wg4` — **captured and green** as of 2026-07-25 (see
  the `--notemp` capture procedure below); logic also proven every suite by
  synthetic pass/fail pairs, including absence-is-ok and
  contradiction-still-fails cases.

## WG-5 — hydromt climate data catalog (side channel)

- **path pattern:** `<runs>/config/rlz_<n>_st_<m>.yml` — ONE per member,
  `temp()`, beside that member's TOML. It was a single
  `<exp>/config/catalogs/data_catalog_run_stress_test.yml` naming every member
  until 2026-08-18; the entries differed only in `uri`, and the rule that built
  it fanned in over the whole sweep.
- **producer:** rule 3.14 `downscale_climate_realization`, calling
  `blueearth_cst/climate_analysis/prepare_climate_data_catalog.py` (rule 3.13
  `write_climate_data_catalog` was removed with the aggregate file).
- **consumer:** rule 3.14 `downscale_climate_realization` (as the `-d` catalog).
- **shape (pinned-as-reliance — hydromt data-catalog schema, OUR emitted
  subset):** one entry per `rlz_<n>_st_<m>` (**including `st_0`**), each
  `{uri, driver.name = raster_xarray, driver.options.preprocess = harmonise_dims,
  driver.options.lock = false, metadata.crs = 4326, metadata.category = meteo,
  data_type = RasterDataset}`.
- **cross-artifact invariant:** the **entry-key set = the realization × cst
  grid**. Checked against the intended grid by the relational validator
  `validate_wg5_catalog_grid` (below).
- **temp() lifecycle:** not `temp()` — persists (the NCs it points at do not).
- **pinned surface:** the per-entry driver/metadata fields above; the entry-key
  grid.
- **deliberately unpinned:** provenance metadata block values; **the `uri`
  value** — an absolute machine-scoped path (fixture:
  `C:\Users\...\rlz_1_st_1.nc`) emitted by `prepare_climate_data_catalog.py`.
  Portability is not a current contract; any future `uri`-resolving guard is
  machine-scoped (design arch-5).
- **validators:** `validate_wg5` (per-entry driver/metadata schema) **and** the
  relational `validate_wg5_catalog_grid` (entry-key grid completeness).

**WG-5 checks bookkeeping only, NOT WG-4/WG-6 NC content (design §5.5).** WG-5
pins that a well-formed catalog *entry* exists per realization × cst; it pins
**nothing** about the NC's dims, variable names, units, or grid (that content is
WG-4 / WG-6's). `validate_wg5_catalog_grid` strengthens the *entry-key
completeness* check but likewise says nothing about NC content. The NC-content
contract is skip-until-captured with **no indirect proxy** — an "inmaps_rlz shape
≈ inmaps_historical shape" proxy would be confirmation bias, not a check of the
real artifact.

## WG-6 — downscaled Wflow forcing (wf3)

- **path pattern:** `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_st_<m>.nc`.
  This is wflow-GRID forcing, so R07 B5 files it on the hydrology side, not
  under `climate/weathergenr/output/`; R9 P2 dissolved the `rlz_<n>/` level, so
  both indices are back in the file name.
- **producer:** rule 3.14 `downscale_climate_realization`
  (`blueearth_cst/experiment/downscale_climate_forcing.py`).
- **consumer:** rule 3.15 `run_wflow`.
- **shape:** **Wflow forcing on the MODEL grid** — the wf3 twin of
  `inmaps_historical.nc`; the same contract as HM-2 (see the hydrological-model
  seam doc, HM-2): `(time, lat, lon)` `float32` `precip` / `pet` / `temp` on the
  staticmaps grid, `spatial_ref` EPSG:4326 + `GeoTransform`, daily. This is the
  wflow-seam forcing input; **pinned once in HM-2, cross-referenced here.**
- **naming pattern:** `forcing/inmaps_rlz_<n>_st_<m>.nc`.
- **temp() lifecycle:** **`temp()`** — deleted after rule 3.15 finishes,
  **absent on the completed fixture**.
- **pinned surface:** as HM-2 (dims, `precip`/`pet`/`temp` names + `float32`,
  the model-grid `(lat,lon)`, EPSG:4326 + `GeoTransform`, daily).
- **deliberately unpinned:** as HM-2.
- **validator:** `validate_wg6` — **skip-until-captured on disk** (temp()
  content absent by default); logic proven by a synthetic pass/fail pair. See the
  `--notemp` capture procedure below.

---

## Considered and excluded (non-interchange artifacts)

Three persisted fixture artifacts were examined and **deliberately excluded** as
non-interchange (no downstream DAG-tracked consumer), so their absence from the
inventory is intentional, not an oversight (design §5.2, risk-5 / arch-7):

- `experiments/<exp>/climate/weathergenr/output/{sim_dates.csv, resampled_dates.csv}` —
  weathergenr-internal run diagnostics. Verified: neither name appears as a
  produced or consumed path in any Snakefile, Python module, or R script.
- `spatial/geoms/region.geojson` (rule `delineate_region`, ADR 0003) - the
  delineated polygon the extraction bbox came from. Provenance for WG-1. It
  IS a DAG-tracked input of `extract_historical_climate` and of rule 1.06; what
  has no DAG-tracked consumer is the extraction's `region_*` attributes,
  which record the same fact inside the data. Retired with ADR 0003: the
  per-store-key `data/climate/historical/<key>/store_region.geojson`.
  *(The pre-R07 `data/climate/historical/wf1_raw/extract_historical.nc`, rule 1.04
  `extract_climate_grid_wf1`, was retired by B1: wf1's model-parity plots now
  read WG-1 itself.)*

The completeness audit (both rule graphs walked) otherwise **confirms**
WG-1..WG-6 cover every interchange handoff at this seam; pipeline-internal
intermediates (build configs, guard/sequencing sentinels, log/benchmark gathers)
are correctly out.

---

## Bounded-substitution walkthrough — replacing weathergenr

A drop-in generator (design §5.6) must:

- **Consume** WG-1 (`extract_historical.nc`, the 7-var K grid) and WG-2 (the
  `stress_test_lookup.csv` perturbation grid) — or provide its own reader for
  them.
- **Produce** WG-4 netCDFs at the DAG-globbed paths
  `climate/weathergenr/output/rlz_<n>_st_<m>.nc` (incl. `st_0`), each a `(time, lat, lon)`
  EPSG:4326 raster with ≥ `precip`, `temp` and `crs=4326` / `category=meteo`, so
  the hydromt catalog (WG-5) loads it via `raster_xarray` + `harmonise_dims`.
- **Repo files it replaces:** rules 3.10–3.12 `shell:` / `script:` targets in
  `run_stress_test.smk` (the two `Rscript --vanilla` bodies pointing at
  `weathergen/*.R`, plus the two config-prep scripts if the WG-3 config surface
  changes).
- **Files it must NOT change (the pinned boundaries):** rule 3.08 (WG-1
  producer), rule 3.14 (WG-5 catalog + WG-6 downscale — it owns both since the
  catalog became per-member).
- **Contracts it must satisfy:** WG-1 / WG-2 (in), WG-4 shape + naming (out),
  and — if it emits its own catalog — WG-5 **including the catalog↔grid
  invariant** (an entry per realization × cst incl. `st_0`). Acceptance check:
  validators `validate_wg1`, `validate_wg2`, `validate_wg4`, `validate_wg5` plus
  the relational `validate_wg5_catalog_grid`.

---

## Validator index

Validators live in `blueearth_cst/shared/interchange_contracts.py` (added by a
later commit; this index is the spec that commit implements against). Each is a
pure `-> list[str]` divergence report (empty ⇒ pass); no `assert` /
`AssertionError` in the bodies (`-O`-safe liftability, design §6.5). Every
validator additionally carries a Layer-1 synthetic pass/fail test pair that
executes on **every** checkout, fixture or not.

| validator | artifact(s) | fixture path (era5) | continuously verified? |
|---|---|---|---|
| `validate_wg1` | WG-1 | `data/climate/historical/<key>/extract_historical.nc` | **yes** (persists); chirps facts **not fixture-verified (no chirps fixture)** |
| `validate_wg2` | WG-2 | `<exp>/config/stress_test_lookup.csv` | **yes** (persists) |
| `validate_wg3` | WG-3 | `<exp>/climate/weathergenr/config/weathergen_config.yml` (the per-member config is gone — C29) | **yes** (persists) |
| `validate_wg4` | WG-4 | `<exp>/climate/weathergenr/output/rlz_<n>_st_<m>.nc` | **captured 2026-07-25** — `temp()` content, absent until a `--notemp` capture; green on the real artifact **after** the `crs`/`category` correction; synthetic-proven every suite |
| `validate_wg5` | WG-5 | `<runs>/config/rlz_<n>_st_<m>.yml` | `temp()` since 2026-08-18 — absent until a `--notemp` capture, then every member's file is checked; synthetic-proven every suite |
| `validate_wg5_catalog_grid` (relational) | WG-5 entry-key grid vs intended `rlz × cst` (incl. `st_0`) | the UNION of `<runs>/config/rlz_<n>_st_<m>.yml` + the run's config snapshot | `temp()` since 2026-08-18 — the union of the per-member files is the set the aggregate catalog used to hold, so the validator is unchanged; needs a `--notemp` capture |
| `validate_wg6` | WG-6 | `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_st_<m>.nc` | **captured 2026-07-25** — `temp()` content, absent until a `--notemp` capture; green on the real artifact unchanged; synthetic-proven every suite |

`validate_wg5_catalog_grid(catalog_cfg, rlz_num, st_num) -> list[str]` checks the
WG-5 entry-key set against the **intended** grid: expected keys exactly
`{rlz_<n>_st_<m> : n ∈ 1..rlz_num, m ∈ 0..st_num}` (**st_0 included** — rule
3.14 is instantiated over both the st_0 members and the perturbed grid).
Applied to the UNION of the per-member catalogs. Missing and unexpected keys are each
reported. The intended grid is derived from the run's *recorded* P3-1 config
snapshot (`<exp>/config/snake_config_run_stress_test.yml`) via the same
`stress_test_grid` helper the Snakefile uses (`shared/snake_utils.py:336`), so
the check is self-consistent with the tree even if the tracked test config later
drifts. A dropped or extra catalog entry is invisible to per-artifact
`validate_wg5` (each remaining entry is well-formed) but breaks the
realization × cst fan-out rule 3.14 depends on.

### `--notemp` capture procedure (temp() on-disk validators)

The `temp()`-content validators `validate_wg4` (WG-4) and `validate_wg6` (WG-6)
have **no on-disk integration check on the default fixture**: both artifacts are
wrapped in Snakemake `temp()` and deleted after their consumers finish, so no
`rlz_<n>_st_<m>.nc` / `inmaps_rlz_<n>_st_<m>.nc` survive a completed run. Their
Layer-2 integration cases (`test_wg4_integration`, `test_wg6_integration`) carry
**both** the `_FIXTURE_ABSENT` skipif and a runtime
`pytest.skip("temp() artifact absent; capture via --notemp")` guarding on the
NC's presence. Their logic is proven on **every** checkout by their Layer-1
synthetic pass/fail pairs regardless.

**The capture WAS RUN 2026-07-25** (the P3-2b milestone deferred it as
out-of-scope for a contracts-only milestone; it was executed later as the
Post-R6/OQ-4 lift). **Outcome: 2 of the 3 validators passed on the real artifacts
unchanged; WG-4 FAILED and the contract — not the pipeline — was wrong.** See the
WG-4 `crs`/`category` note above: it demanded catalog metadata as netCDF global
attrs, and the real artifact carries none. Corrected to asserted-if-present. This
is precisely the class of error only a capture can find, which is the argument for
having run it. The procedure below is the repeatable lift.

**Cheaper targeted form.** The full-sweep command below works, but only three
artifact paths are actually needed (`rlz_1_st_1`), so naming them as targets is
enough and avoids re-running the batches that are already up to date:

```bash
snakemake -c 3 -s run_stress_test.smk \
  --configfile config/workflows/snake_config_model_test.yml --notemp \
  test_case/test_local/experiments/experiment/climate/weathergenr/output/rlz_1_st_1.nc \
  test_case/test_local/experiments/experiment/hydrology/wflow/forcing/inmaps_rlz_1_st_1.nc \
  test_case/test_local/experiments/experiment/hydrology/wflow/output/outstates_rlz_1_st_1.nc
```

Measured 2026-07-25: **19 jobs, 247.7 s**. Note the `temp()` cascade — asking for
one intermediate re-runs 3.11 (which emits **all** realizations) and therefore all
twelve 3.12 jobs plus `run_wflow_batch_0`; there is no cheaper single-cst path.

**Capture sketch** (run from the repo root inside `pixi shell`, after the wf1
model exists — wf3 needs `models/hydrology/wflow/` artifacts):

```bash
snakemake all -c 3 -s run_stress_test.smk \
  --configfile config/workflows/snake_config_model_test.yml --notemp
```

`--notemp` tells Snakemake **not** to delete `temp()`-flagged outputs after their
consuming jobs complete, so the run leaves the intermediate netCDFs on disk.

**Paths that then appear** under `test_case/test_local` (the paths the
skip-guards test for):

| validator | artifact captured | fixture path (`<exp>` = `experiments/experiment`) |
|---|---|---|
| `validate_wg4` | WG-4 generator output NC | `<exp>/climate/weathergenr/output/rlz_<n>_st_<m>.nc` |
| `validate_wg6` | WG-6 downscaled forcing NC | `<exp>/hydrology/wflow/forcing/inmaps_rlz_<n>_st_<m>.nc` |

(HM-6b's `output/outstates_rlz_<n>_st_<m>.nc` is captured by the same run — documented
in the hydrological-model seam doc.)

**Which cases un-skip:** with these artifacts present, `test_wg4_integration` and
`test_wg6_integration` here (plus `test_hm6b_integration` in the other seam doc)
stop hitting their `pytest.skip` and run their on-disk assertion — the **three**
temp validators' *on-disk* integration checks flip from skip-until-captured to
green. The guards resolve to the real-artifact path automatically once the files
exist.

**Correction (2026-07-25):** this section used to promise "**no test code or
validator changes**". That held for WG-6 and HM-6b but **not** for WG-4, whose
`crs`/`category` global-attr requirement did not survive contact with the real
artifact (above). The honest statement: the *guards* need no change, but a
capture can — and here did — reveal that a validator encoded an assumption the
artifact never satisfied. Budget for that when running a capture; a first-contact
failure is a likely outcome, not a surprise.

**Restore** the default temp-deleted fixture state with
`snakemake --delete-temp-output` (verified 2026-07-25 to return the tree to a
byte-identical state, checked with `dev/scripts/semantic_tree_diff.py`).
