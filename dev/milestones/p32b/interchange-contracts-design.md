# P3-2b — Model-swap interchange contracts: Design (ACCEPTED)

> **Status: ACCEPTED** — approved at human gate G2 (2026-07-24). The accepted
> revision (v3) converged inside the external round cap: internal lens panel
> (risk / architecture / repo-fit; 0 blocking / 7 major / 9 minor, all
> accepted in v2) → external GPT round 1 (`revise`: ext1-1 relational
> validators, ext1-2 all-skip-green defect in the v2 repo-1 fix; both
> resolved in v3, Fable-escalated revision) → external GPT round 2
> **`approve`, zero findings**, on this version unchanged. G1 framing
> approved 2026-07-24. Per-finding disposition (18/18 accepted, none
> rejected or deferred) in the consolidated review record
> (`interchange-contracts-design-review-record.md`); what changed per
> revision in §11.
> **Milestone:** P3-2b (interchange contracts at both substitution seams).
> **Genre:** decision-record (milestone design). **Commit prefix:** `p32b:`.
> **Author role:** cst-architect. **Run:** `p32b-interchange-contracts`.
> **Scope authority:** `dev/milestones/p32b/climate-interchange-intake.md` (its four
> "Confirmed scoping decisions" are fixed anchors, not reopened here).
> Structure precedent: `dev/milestones/p32a/climate-analysis-design.md` (accepted 2026-07-24).
> This doc is self-contained: a reviewer needs only this file plus the cited paths.

---

## 1. Problem statement

The CST toolbox has two **substitution seams** — the points where a component
could be swapped for an alternative implementation without re-architecting the
rest of the pipeline:

- **The weather-generator seam** — today `weathergenr` (R): it consumes the
  historical climate extraction + a config surface + the stress-test parameter
  grid, and produces per-realization / per-cst perturbed climate netCDFs that are
  downscaled to Wflow forcing.
- **The hydrological-model seam** — today Wflow-SBM (built by hydromt): it
  consumes forcing on the model grid + the built static grid + a run TOML, and
  produces discharge CSVs reduced to the stress-test response surface.

Phase-3's driver is **model flexibility** (the R6-handoff expectations map):
making either component swappable is a stated goal. But today both seams exist
**only as implicit file handoffs** — a set of netCDFs, CSVs, TOMLs, and YAMLs
whose shapes, names, units, and producer→consumer rules are encoded nowhere
except in the Snakefiles and scripts that happen to read them. Nothing states
*what a replacement must satisfy*; nothing checks that a candidate artifact
conforms; "swappable" is aspirational, not checkable — the same
aspirational-vs-checkable gap P3-2a closed for model-independence.

P3-2b makes the two seams **explicit, documented, and machine-checked**: per-seam
contract documents (dims/coords/vars/units/dtypes/CRS/time-axis/naming/
producer→consumer) plus hand-rolled validator functions exercised by pytest
against real fixture artifacts, so a generator/model swap becomes a **bounded
exercise against pinned contracts** rather than an archaeology expedition. It
wires **no** alternative implementation and changes **no** pipeline behavior — it
only pins and checks what already flows.

The cost of leaving this: any future swap (a different weather generator, a
non-Wflow hydrological engine, or the R6 model-flexibility work) starts by
reverse-engineering the seam from the Snakefiles; a subtly non-conforming
artifact fails deep in a run instead of at a contract boundary; and the platform
has no artifact it can point a third-party model author at.

## 2. Goals / Non-goals

### Goals

1. **G1 — a complete seam inventory.** Every interchange artifact at both seams is
   enumerated with its producer rule, its consumer rule(s), and a full shape
   record: dims, coords, variables, units, dtypes, CRS, time axis/calendar,
   naming pattern, and `temp()` lifecycle. Grounded in the three Snakefiles and
   the fixture tree, not aspiration (§5.2, §5.3).
2. **G2 — a validator per contract, green on fixture artifacts.** Each contract
   surface has a hand-rolled validator function (stdlib + xarray + existing deps
   only — no new dependency), exercised by pytest against the **real
   `examples/test_local` fixture**, green under `pixi run pytest` for every
   artifact that persists on disk (§5.5). "Contract" includes the two
   **cross-artifact invariants** (the HM-4→HM-5→HM-7 gauge-column identity; the
   WG-5 catalog↔grid coupling), each checked by a dedicated **relational
   validator** with an explicit multi-object signature (§5.5). Every validator's
   logic — per-artifact and relational — is additionally proven
   **fixture-independently** by synthetic pass/fail unit tests, so green never
   means nothing-was-executed (§5.5). Validators are pure functions written to
   be liftable into future in-pipeline guards, but wired into **nothing** this
   milestone.
3. **G3 — an explicit contract-surface boundary.** For each artifact the design
   states what is a **contract surface** (pinned, checked) vs an **internal detail
   deliberately left unpinned** — most importantly, our reliance on the
   hydromt/wflow static-grid and TOML schema is pinned as *OUR consumed subset*,
   never as a re-specification of the upstream schema (§5.1).
4. **G4 — a bounded-substitution walkthrough per seam.** Each seam's contract doc
   names exactly which repo files a replacement weather generator / hydrological
   model would touch, and which contracts it must satisfy — the roadmap's "bounded
   substitution" made concrete and reviewable (§5.6).
5. **G5 — zero behavior change, per-commit-runnable.** No Snakefile/pipeline/module
   edit; the full suite plus three dry-runs stay green at every commit; nothing is
   re-recorded. The deliverables are contract docs + validator code + tests only.

### Non-goals

- **No alternative implementation (PoC swap).** Neither seam gets a second
  generator/model wired. A swap remains a future bounded exercise (a P3-2c
  candidate; intake decision 1).
- **No in-pipeline contract enforcement.** Validators run under pytest only; no
  Snakefile rule body calls them, no runtime guard is inserted. They are written
  to be liftable later (intake decision 2).
- **No new dependencies.** Hand-rolled validators over stdlib + xarray +
  geopandas + pyyaml + tomllib (all already in the env). No schema library
  (`new-dependency-requires-approval` memory; intake decision 2).
- **No behavior change of any kind** — no output re-record, no manifest change, no
  DAG edit, no `blueearth_cst/**` runtime edit. `blueearth_cst/weathergen/*.R` is
  read-only (may be read to derive the R-side contract; never edited).
- **No absorption of P3-2a-deferred structural items** (intake decision 3): the
  OQ-3 project-level climate store, the OQ-8 model-free subcatchment-zone source /
  polygon-zonal aggregation, and the standalone climate-screening entry point all
  stay OUT — they are behavior-touching redesigns, recorded as later candidates.
- **No upstream re-specification.** Contracts describe OUR reliance on hydromt /
  hydromt_wflow / wflow / weathergenr formats; they never re-specify or patch
  upstream internals (AGENTS.md Hard Constraints).

## 3. Constraints (standing; restated for this milestone)

- **CST automation scope (the governing constraint).** hydromt / hydromt_wflow /
  wflow / weathergenr conventions are consumed verbatim. A contract at the
  hydrological-model seam pins **which variables/fields OUR pipeline reads or
  rewrites** (e.g. `staticmaps.nc["subcatchment"]`, the TOML `[input]` block,
  `output.csv`); it does **not** re-document the full wflow static-grid schema or
  the TOML physics parameterization. The validators *read* upstream artifacts to
  check our reliance; they never assert upstream correctness (AGENTS.md Hard
  Constraints; `stay-within-cst-automation-scope` memory).
- **Zero behavior change.** No Snakefile edits, no output changes, no manifest
  re-record. Per-commit gates are cheap: full suite + three dry-runs, all green.
- **No new dependencies** without prior user approval — hand-rolled validators
  only.
- **`blueearth_cst/weathergen/*.R` content untouched** (read-only for
  contract-derivation).
- **Naming** per `dev/reference/naming.md`: new identifiers snake_case, `_path`
  for path strings, `_ds`/`_df` for objects; contract-doc filenames kebab-case.
  Domain identifiers governed by an upstream tool or an established BlueEarth
  contract follow *that* contract, not local style (naming.md §2/§6) — the
  contract docs record those verbatim.
- **Commit prefix `p32b:`**; tag `p32b-interchange-contracts` at milestone close.
- **Fixture is era5** (`config/workflows/snake_config_model_test.yml`): validators
  are grounded on the era5 branch; branch-specific contract facts (chirps
  orography sidecar, chirps-only precip) are documented from code and marked
  *not continuously fixture-verified* where no chirps fixture exists (§5.5).

## 4. Decision criteria

A design choice below is judged against, in priority order:

- **C1 — Contracts are checkable, not aspirational.** Every pinned contract
  surface has a validator that runs green against a real fixture artifact, or an
  explicit, honest statement that it cannot be continuously checked (and why).
  "Pinned" without a validator or an honest gap statement fails C1.
- **C2 — Zero behavior change.** No Snakefile/pipeline/module runtime edit; full
  suite + three dry-runs green at every commit; nothing re-recorded. A validator
  module added to the package but wired into no rule does **not** violate this
  (it is unreferenced by the DAG).
- **C3 — CST automation scope respected.** Contracts pin OUR consumed/rewritten
  subset of upstream formats; they never re-specify or assert upstream internals.
  Any staticmaps/TOML/state contract must name the boundary between "we rely on
  this" and "wflow owns this, unpinned."
- **C4 — Grounded in the tree.** Every contract fact traces to a Snakefile line, a
  script line, or an observed fixture artifact — not to the intake's prose (whose
  artifact hints the design verifies and, where wrong, corrects: §5.3 warm-state,
  §5.3 output-column set).
- **C5 — Liftable, minimal-footprint validators.** Validators are pure functions
  (data in → pass/fail + reason), placed so a future milestone can call them from
  a rule body without moving them, and depend only on existing packages.
- **C6 — Honest coverage.** The design states plainly which contracts are
  continuously verified on the current fixture and which are not (temp()-deleted
  content, non-era5 branches), and never dresses an unverifiable contract as green
  (success criterion 2 read strictly).

## 5. Selected approach

### 5.1 Contract-surface boundary (what is pinned vs deliberately unpinned)

The governing principle, applied uniformly at both seams:

> **A contract surface is any property of an interchange artifact that OUR
> pipeline's producer guarantees or OUR consumer relies on.** Everything else an
> artifact happens to carry — most of the wflow static-grid schema, TOML physics
> parameters we neither read nor rewrite, netCDF provenance attrs — is
> **deliberately unpinned**: it is upstream's to define, and pinning it would both
> violate the CST automation scope (C3) and make the contract brittle to
> upstream version bumps.

Concretely this yields three tiers per artifact:

1. **Pinned (contract surface).** Structural facts a swap MUST reproduce for the
   downstream consumer to work: variable names + units the consumer reads, the
   dims/coord axes it indexes, the naming pattern the DAG globs, the CRS/grid
   co-registration a regrid assumes, the CSV column-identity a reduction keys on.
   Validated (§5.5).
2. **Pinned-as-reliance (consumed subset of an upstream schema).** For
   `staticmaps.nc` and `wflow_sbm.toml` we pin **only the names/fields OUR code
   and the run TOML reference** (e.g. `subcatchment`, `land_elevation`,
   `local_drain_direction`, `river_mask`, `outlets`; the TOML `[input]` /
   `[state]` / `[output.csv]` blocks our downscale step rewrites). The remaining
   ~30 staticmaps variables and the physics `[input.static]` value blocks are
   labelled **"wflow schema, consumed verbatim, unpinned"** (C3). The contract
   says "a replacement model must provide a static grid these OUR-referenced names
   resolve against, on a grid the forcing co-registers to" — not "reproduce the
   wflow schema."
3. **Deliberately unpinned (internal detail).** Provenance attrs
   (`paper_doi`/`source_url`/…), exact chunking/compression encoding, the
   float32-vs-float64 choice on coords where no consumer depends on it, and
   temp-file byte layout. Recorded as unpinned so a reviewer sees the omission is
   intentional, not an oversight.

The boundary is stated **per artifact** in §5.2/§5.3 (a "pinned?" column), so the
reviewer can audit every inclusion and exclusion.

### 5.2 Contract inventory — weather-generator seam

The seam sits inside `Snakefile_climate_experiment` (wf3). Producer/consumer rules
cited by number; shapes grounded in the fixture (`examples/test_local`, era5) and
the scripts. **weathergenr is the current occupant; the contract is generator-
agnostic** — it pins what wf3 hands *in* and expects *out*, not weathergenr's
internals.

| # | Artifact (path pattern) | Producer → Consumer | Pinned contract surface (grounded) | temp()? | Deliberately unpinned |
|---|---|---|---|---|---|
| WG-1 | `climate_historical/<key>/extract_historical.nc` (`<key>=<clim_source>_<startYYYYMMDD>_<endYYYYMMDD>`, P3-1 keyed store) | rule 3.02 `extract_climate_grid` (`climate_analysis/extract_historical_climate.py`) → rule 3.06 `generate_weather_realization` (weathergenr `generate_weather.R`) as `climate_nc` | **dims** `(time, latitude, longitude)`; **coords** `time` (`datetime64[ns]`, daily, `calendar=proleptic_gregorian`), `latitude`/`longitude` (`float32`, `degrees_north`/`east`), `spatial_ref` (EPSG:4326, WKT); **data_vars** `precip` (`mm d**-1`), `temp`/`temp_min`/`temp_max` (**K** — see units note), `kin`/`kout` (`J m**-2`), `press_msl` (`Pa`), all `float32 (time,lat,lon)`; **every WG-1 unit is under the `units` (plural) attr key** (fixture-verified — NOT `unit` singular; contrast HM-2 §5.3); **global attr** `crs=4326`, `category=meteo`. era5 branch writes all 7; chirps branch: `precip` chirps-native + era5 temp/rad/press reprojected (§5.3 note). | no (consumed `ancient()`) | provenance attrs (`paper_*`, `source_*`, `notes`); chunk/encoding |
| WG-2 | `<exp>/stress_test/cst_<m>.csv` (`m≥1`) | rule 3.03 `climate_stress_parameters` (`experiment/prepare_cst_parameters.py`) → rule 3.07 `generate_climate_stress_test` (weathergenr `impose_climate_change.R`) as `st_csv` | **header** exactly `month,temp_mean,precip_mean,precip_variance`; **12 rows** `month∈1..12`; `temp_mean` additive (°C), `precip_mean`/`precip_variance` multiplicative factors (fixture: `0.0,0.7,1.0`). One file per perturbation `m=1..ST_NUM`; `cst_0` reserved (no file — unperturbed baseline, naming.md §4). | no | — |
| WG-3 | `<exp>/weathergen_config.yml` and `<exp>/realization_<n>/weathergen_config_rlz_<n>_cst_<m>.yml` | rules 3.04/3.05 `prepare_weagen_config[_st]` (`experiment/prepare_weagen_config.py`) → rules 3.06/3.07 (R) | **The weathergenr config surface** (YAML): top-level `general.variables` (list ⊆ `{precip,temp,temp_min,temp_max}`), `generateWeatherSeries.{warm.*, knn.sample.num, month.start, warm.variable, seed, evaluate.*, dry.spell.change[12], wet.spell.change[12], output.path, sim.year.start, sim.year.num, nc.file.prefix, realizations_num}`. Contract = **the key set + types the R side reads** (derived read-only from `global.R`/`generate_weather.R`), not weathergenr's semantics. | no | comment layout, key order |
| WG-4 | `<exp>/realization_<n>/rlz_<n>_cst_0.nc` (baseline) and `rlz_<n>_cst_<m>.nc` (`m≥1`, perturbed) | rule 3.06 (cst_0) / rule 3.07 (cst_m) → rule 3.08 `climate_data_catalog` + rule 3.09 `downscale_climate_realization` | **The generator OUTPUT contract** — a raster netCDF the hydromt catalog reads: `(time, lat, lon)` daily grid with at least `precip`, `temp` (+ `pet` if present) on an EPSG:4326 grid, `crs=4326`/`category=meteo` metadata (so `raster_xarray` + `harmonise_dims` load it — WG-5). Naming `rlz_<n>_cst_<m>.nc` is a **DAG-globbed pattern** (rule 3.08 `expand`, rule 3.09 wildcards). | **yes** `temp()` (both) | exact var superset, internal attrs |
| WG-5 | `<exp>/data_catalog_climate_experiment.yml` (rule-3.08 side-channel) | rule 3.08 `climate_data_catalog` (`climate_analysis/prepare_climate_data_catalog.py`) → rule 3.09 `downscale_climate_realization` (`-d` catalog) | **hydromt data-catalog schema (our emitted subset):** one entry per `rlz_<n>_cst_<m>` (incl. `cst_0`), each `{uri, driver.name=raster_xarray, driver.options.preprocess=harmonise_dims, driver.options.lock=false, metadata.crs=4326, metadata.category=meteo, data_type=RasterDataset}`. Entry-key set = the realization×cst grid — a **cross-artifact invariant** checked against the intended grid by the relational validator `validate_wg5_catalog_grid` (§5.5). | no (persists) | provenance metadata block values; **`uri` value** — absolute machine-scoped path (fixture: `C:\Users\...\rlz_1_cst_1.nc`) emitted by `prepare_climate_data_catalog.py`; deliberately unpinned (portability is not a current contract; any future `uri`-resolving validator/guard is machine-scoped — arch-5) |
| WG-6 | `<exp>/realization_<n>/inmaps_rlz_<n>_cst_<m>.nc` (downscaled forcing) | rule 3.09 `downscale_climate_realization` (`experiment/downscale_climate_forcing.py`) → rule 3.10 `run_wflow` | **Wflow forcing shape on the MODEL grid** — same contract as HM-2 (§5.3): `(time,lat,lon)` `float32` `precip`/`pet`/`temp` on the staticmaps grid, `spatial_ref` EPSG:4326 + `GeoTransform`, daily. This is the **wf3 twin of `inmaps_historical.nc`** and the wflow-seam forcing input; pinned once in §5.3 HM-2, cross-referenced here. | **yes** `temp()` | — |

**Units note (grounded, corrects the p32a °C assumption; refined in v2 per
arch-2/risk-4/repo-3).** WG-1 `temp*` is in **Kelvin** (`long_name` + observed,
under the `units` plural key): the extraction writes native era5 K; the
Kelvin→°C conversion happens inside the forcing build / downscale, so the °C value
lands on the model-grid forcing (HM-2 `temp.attrs['unit'] = 'degree C.'`,
fixture-verified — the divergence note **is** fixture-grounded, correcting arch-2's
parenthetical that HM-2 temp carries no unit attribute). **Units are NOT pinned as
a hard contract surface on either artifact** — wflow maps forcing by variable NAME
via the TOML `[input.forcing]` block (HM-2, §5.3), never by the netCDF unit
attribute — so the K-vs-°C divergence is recorded as an **observed, documented
cross-seam fact** and asserted only **if the attr is present** (§5.5), not pinned
as a required property. This satisfies R1 (do not over-constrain a swap with a
property no consumer reads) while keeping the divergence honestly on the record.

**Excluded non-interchange artifacts (completeness audit, risk-5/arch-7).** Three
persisted fixture artifacts were examined and **deliberately excluded** as
non-interchange (no downstream DAG-tracked consumer), so their absence from the
inventory is intentional, not an oversight:

- `experiments/experiment/{sim_dates.csv, resampled_dates.csv}` — weathergenr-
  internal run diagnostics; verified: neither name appears as a produced/consumed
  path in any Snakefile, Python module, or R script (risk-5).
- `climate_historical/wf1_raw/extract_historical.nc` (rule 1.10
  `extract_climate_grid_wf1`) — shares WG-1's extraction schema but feeds the wf1
  model-parity **plots**, not either substitution seam (arch-7).

The completeness audit (both rule graphs walked) otherwise **confirms** WG-1..WG-6
/ HM-1..HM-7 cover every interchange handoff at the two seams; pipeline-internal
intermediates (build configs, guard/sequencing sentinels, log/benchmark gathers)
are correctly out (arch-7).

**Producer-side (R) contract surface, read-only.** WG-3's config keys and WG-4's
output shape are the surface a *replacement generator* must honor; they are
derived by reading `blueearth_cst/weathergen/{global.R,generate_weather.R,
impose_climate_change.R}` (never edited). The contract doc records the key set and
the output netCDF shape; it does not reproduce weathergenr's algorithm.

### 5.3 Contract inventory — hydrological-model seam

The seam sits across `Snakefile_model_creation` (wf1 build) and
`Snakefile_climate_experiment` (wf3 run). It pins OUR reliance on the
hydromt/wflow build + run artifacts (C3), grounded in the fixture and the TOML.

| # | Artifact | Producer → Consumer | Pinned contract surface (OUR reliance) | temp()? | Deliberately unpinned |
|---|---|---|---|---|---|
| HM-1 | `hydrology_model/staticmaps.nc` | rule 1.03 `create_model` (hydromt build) → wf1 rules 1.04/1.05/1.10/1.11; wf3 rule 3.09 (`WflowSbmModel(root)`) | **Only the OUR-referenced names**, all on `(latitude, longitude)` `float64` coords + `spatial_ref` EPSG:4326 + `GeoTransform`: `subcatchment` (zone raster — plot aggregation, P3-2a §5.2), `land_elevation` (`m` — parity DEM, `naming.py:10`), plus the TOML-referenced `local_drain_direction`, `river_mask`, `outlets`, and the `[input.static]` name set the run resolves. **Grid definition** (the `(lat,lon)` axes + `GeoTransform`) is pinned as the co-registration target forcing must match. | no | **~39 unpinned wflow vars** (fixture staticmaps.nc has **44** data_vars total, minus the pinned OUR-referenced set — arch-6) (`vegetation_*`, `soil_*`, `meta_*`, `river_*` beyond mask) — **wflow schema, consumed verbatim, unpinned** |
| HM-2 | `climate_historical/wflow_data/inmaps_historical.nc` (wf1 forcing); wf3 twin `inmaps_rlz_<n>_cst_<m>.nc` (WG-6) | rule 1.08 `add_forcing` (hydromt update) → rule 1.09 `run_wflow`; wf3 rule 3.09 → 3.10 | **PINNED:** **dims** `(time, latitude, longitude)` on the **model grid** (`float64` lat/lon matching HM-1); **data_vars** exactly `precip`, `pet`, `temp`, all `float32`, each `grid_mapping=spatial_ref`; `spatial_ref` EPSG:4326 + `GeoTransform`; daily `proleptic_gregorian`. Variable **names** are the consumer contract — they are the RHS values the TOML `[input.forcing]` block maps to (HM-4). **UNITS NOT PINNED (arch-2/risk-4/repo-3):** wflow is name-keyed, so no consumer reads the unit attr. Observed attr layout (recorded, asserted-if-present only): `precip` carries **both** `units='mm d**-1'` and `unit='mm'`; `pet` `unit='mm'` (`units` absent); `temp` `unit='degree C.'` (`units` absent). The wflow-native values live under the `unit` (**singular**) key; the `units` (plural) key survives on `precip` only as an extraction leftover. | HM-1 no; WG-6 **yes** | **all forcing units** (`unit`/`units` attr values) — asserted-if-present, not required; `precip_fn`/`pet_method`/`temp_correction` provenance attrs |
| HM-3 | `hydrology_model/staticgeoms/*` (`region.geojson`, `basins.geojson`, `outlets.geojson`, `rivers.geojson`, `outlet_index.csv`, …) | rule 1.03 side-effect + rules 1.05/1.06 → wf1 plot rules; wf3 rule 3.02 (`region.geojson` `ancient()`) | **OUR-consumed vectors only:** `region.geojson` (basin extent polygon, EPSG:4326 — the wf3 extraction region + `ancient()` DAG edge); `outlets.geojson` (gauge points → plots/outputs); `outlet_index.csv` (`outlet position → subcatchment-ID` mapping, `rule all` target). CRS EPSG:4326; geometry types (Polygon / Point). | no | the full attribute tables; `basins`/`rivers`/`meta_*` layers we don't index |
| HM-4 | `hydrology_model/wflow_sbm.toml` (base) and per-cst `<exp>/model_runs/wflow_sbm_rlz_<n>_cst_<m>.toml` | tracked template / rule 3.09 rewrite → rule 1.09 / rule 3.10 `run_wflow` (`Wflow.run()`) | **The TOML fields OUR code reads/rewrites** (the wf3 rewrite sites, `downscale_climate_forcing.py:55-84` `setup_config`): `[time].{calendar,starttime,endtime,timestepsecs}`, `dir_output`, `[state].{path_input,path_output}`, `[input].{path_static,path_forcing}`, `[output.csv].path`. **Rewrite-value facts (arch-4, fixture-verified):** wf3 sets `time.timestepsecs = 86400` and rewrites `time.calendar` to **`"standard"`** — distinct from the wf1 base + HM-2 pin of `proleptic_gregorian` (the code comment: weathergenr writes `noleap`, and hydromt_wflow 1.x forcing validation would fail comparing `cftime.DatetimeNoLeap` vs `datetime.datetime`, so both forcing axis and TOML are moved to `standard`). wf3 also sets `dir_output = "."` (flat, no `run_default/` subdir) and `state.path_output = "outstates_<climate_name>.nc"` — so the wf3 warm state lands flat, unlike wf1 (HM-6, §5.3). **Also pinned (read-reliance):** `[input.forcing]` — the block **keys are wflow CSDMS Standard Names** (e.g. `atmosphere_water__precipitation_volume_flux`, `land_surface_water__potential_evaporation_volume_flux`, `atmosphere_air__temperature`) and the **VALUES are the forcing netCDF variable names** `precip`/`pet`/`temp` (the tie to HM-2 is on the RHS values — arch-6, direction corrected); `[output.csv].column` entries `{header,map,parameter}` (drives HM-5 column identity), `cold_start__flag`. | no | all `[input.static]` physics value blocks, layer thicknesses, kinematic-wave params — **wflow physics, unpinned** |
| HM-5 | wf1 `hydrology_model/run_default/output.csv`; wf3 `<exp>/model_runs/output_rlz_<n>_cst_<m>.csv` | rule 1.09 / rule 3.10 `run_wflow` → wf1 rule 1.11 plots; wf3 rule 3.11 `export_wflow_results` | **column identity is config-driven, not a literal list:** a `time` index (ISO-8601, daily) + **one column per `[output.csv].column` entry**, named `<header>_<mapid>`. Fixture: `time,Q_130000086` (one gauge). The gauge-column set flows **TOML `[output.csv].column` → `output_rlz` → Qstats** as a *single degree of freedom* — the key bounded-substitution invariant (§5.6), checked end-to-end by the relational validator `validate_hm_gauge_column_identity` (§5.5). Consumer-prefix reliance (grounded): rule 3.11 selects gauge columns by a **hard-coded `Q_` prefix** and basin-average columns by a `basavg` substring (`export_wflow_results.py:61-62`) — so the TOML `header` values are load-bearing beyond mere identity. | no (persists) | numeric discharge values (not a contract; they change per run) |
| HM-6a | wf1 warm state: `hydrology_model/run_default/outstate/outstates.nc` | rule 1.09 `run_wflow` → (nothing in-repo) | **THIN — "named output sink, unconsumed."** Persisted on the fixture. **No validator (risk-1, decided in v2): its only contract surface is name + location, which HM-4 already pins via `[state].path_output`** — a standalone existence check would pad the green count without verifying an independent contract. Kept as a **doc row only**; its sole contract surface (name/location) is pinned by HM-4's `[state].path_output` declaration. **Path derivation (arch-3):** the on-disk path is `run_default/outstate/outstates.nc` = base TOML `dir_output = "run_default"` **+** `[state].path_output = "outstate/outstates.nc"` — a swapper that changes `dir_output` moves this path with it. | no | entire state-variable schema (`[state.variables]` — wflow-owned) |
| HM-6b | wf3 warm state: `<exp>/model_runs/outstates_rlz_<n>_cst_<m>.nc` | rule 3.10 `run_wflow` → (nothing in-repo) | **THIN — "named output sink, unconsumed" (corrects the intake's chaining hint).** Verified: the per-cst TOML keeps `cold_start__flag = true` and declares **no `instates` input** on rule 3.10; wf3 fans out in parallel over `(rlz,cst)` with no cross-cst edge — **no warm-state chaining invariant** our DAG relies on. Contract pins only: the file is a declared wflow state **output** whose name (`outstates_<climate_name>.nc`) and flat location (wf3 `dir_output="."`, HM-4) our rewrite sets, deleted `temp()` in wf3. **Validator = skip-until-captured** (temp() content absent on the fixture, §5.5); logic proven by a synthetic pass/fail unit test. **The wf1/wf3 split mirrors HM-2/WG-6 structurally, but is disanalogous on content** — forcing (HM-2/WG-6) is consumed, warm state is not — so the split does not imply an independent wf1 validator (hence HM-6a carries none). Internal state-variable schema is **wflow's, unpinned**. | wf3 **yes** | entire state-variable schema (`[state.variables]` — wflow-owned) |
| HM-7 | Reduction: `<exp>/model_results/Qstats.csv`, `basin.csv` | rule 3.11 `export_wflow_results` → CST-API/GUI (terminal in-repo) | `Qstats.csv`: header `statistic,tavg,prcp,<gauge-cols>` where `<gauge-cols>` = HM-5's `<header>_<mapid>` set (fixture `Q_130000086`); rows keyed by `statistic` × the `(tavg,prcp)` perturbation grid. `basin.csv`: header `tavg,prcp` (the perturbation-axis index). These are the **response-surface hand-off** to the platform. | no (`rule all`, manifested) | the RT_*.csv response tables (non-manifest side products) |

**Warm-state finding (C4, corrects the intake).** The intake's seam inventory
hinted at "per-cst run chaining" via `instates`/`outstates`. The fixture + TOML
diff (`wflow_sbm_rlz_1_cst_1.toml` vs base `wflow_sbm.toml`) shows
`cold_start__flag = true` in **both**, rule 3.10 declares only a `forcing_path` +
`toml_path` input (no `instates`), and the `path_input` the rewrite sets points at
a **non-existent** `hydrology_model/instate/instates.nc` that cold-start never
reads. So HM-6a/HM-6b are contracted as an **unconsumed named sink**, not a
chaining invariant — the honest, grounded shape (a chaining contract would be
fiction). HM-6 was **split wf1/HM-6a (persisted) vs wf3/HM-6b (temp)** in v2
(risk-1/arch-1) to keep one status per id.

**Units divergence (C4, refined in v2).** The cross-seam unit change is real and
fixture-grounded: WG-1/extraction temp = **K** (under `units`), HM-2 model-grid
forcing temp = **degree C.** (under `unit`, fixture-verified — correcting arch-2's
parenthetical). The conversion is part of the hydromt forcing-build reliance. But
because wflow reads forcing by **variable name**, not by unit attribute (HM-4
`[input.forcing]`), the units are **NOT a hard pinned contract surface** on either
artifact (arch-2): they are recorded as an observed, documented divergence and
asserted **only if the attr is present** (§5.5), never as a required property — so
a swap is not over-constrained by a property no consumer reads (R1).

### 5.4 Contract-doc format + placement

**Placement: `dev/`, not `docs/`.** The audience is future *swappers* and
dev-process (a model/generator author, the R6 model-flexibility work), not the
tool's end user — squarely the `dev/` side of the AGENTS.md `dev/`-vs-`docs/`
boundary. Precedent: the R5 contract doc `dev/reference/workflows/climate_experiment.md`
records current-behavior contracts under `dev/reference/workflows/`.

**One file per seam** (two docs — matching the `dev/reference/workflows/*.md`
one-file-per-workflow precedent and intake decision 2's "per-seam contract
documents"):

- `dev/reference/contracts/weather-generator-seam.md` — the WG-1..WG-6 inventory.
- `dev/reference/contracts/hydrological-model-seam.md` — the HM-1..HM-7 inventory.

(New `dev/reference/contracts/` subdir; kebab-case filenames per naming.md. Rejected: one
combined file — it buries the two independent substitution stories; per-artifact
files — 13 fragments, over-decomposed, breaks the "one doc a swapper reads
end-to-end" goal. §6.1.)

**Per-artifact table schema** (each doc, mirroring §5.2/§5.3 columns, expanded):

`artifact id · path pattern · producer rule · consumer rule(s) · dims · coords
(dtype, units, calendar) · data_vars (dtype, units) · CRS · time axis/calendar ·
naming pattern · temp() lifecycle · pinned surface · deliberately unpinned ·
validator name`.

Each doc opens with a scope/method note (grounded-in, generator/model-agnostic,
CST-scope disclaimer), carries the inventory table + per-artifact prose for the
non-trivial ones, then a **bounded-substitution walkthrough** (§5.6) and a
**validator index** (function → artifact(s) → fixture path(s) →
continuously-verified? flag; the two relational validators list every artifact
in their correlated set — §5.5).

### 5.5 Validator design

**Hand-rolled, one module, unwired (C2/C5).** A single new package module
`blueearth_cst/shared/interchange_contracts.py` holds pure validator functions,
one per artifact contract **plus two relational validators for the
cross-artifact invariants** (below, v3/ext1-1).

**Error idiom — pure `-> list[str]` divergence report (repo-2, decided in v2).**
Each validator is `validate_<id>(obj) -> list[str]`: it returns a list of
human-readable divergence messages, **empty list = pass**. This mirrors the house
drift-guard analog `compare_project_consistency`
(`blueearth_cst/experiment/check_project_consistency.py:142`, a documented pure
`-> list[str]` function; empty ⇒ consistent), rather than the other analog
`validate_experiment_name` (`shared/snake_utils.py:193`, `-> str` raising
`ValueError`). Rationale for choosing the report list over `ValueError`:

- **`-O`-safe liftability (C5).** `assert` / `AssertionError` is stripped under
  `python -O` / `PYTHONOPTIMIZE`, so a future in-pipeline guard running optimized
  would silently no-op — the exact lift the design is architected around would fail
  open. A returned report never vanishes. **`assert` / `AssertionError` is banned
  in the validator bodies** for this reason (the test file may `assert` on the
  returned list, but the validator itself must not).
- **A report list composes with a guard AND a test.** A future guard lifts the
  function and raises/logs when the list is non-empty; the test this milestone
  asserts the list is empty. Same function, both call sites, no move (C5).
- **`ValueError`-on-first-violation loses information** (stops at the first bad
  fact); a divergence report surfaces every violation at once, better for a swapper
  diagnosing a candidate artifact.

Placement rationale (against the zero-behavior constraint):

- **Placed in the package, not `tests/`, so it is liftable** into a future
  in-pipeline guard without a move (intake decision 2; C5). It is imported by the
  test file and **by nothing in any Snakefile** — adding an unreferenced module is
  not a behavior change (C2): no DAG edge, no rule body, no import from a
  `script:` target touches it.
- **Pure functions over PARSED objects** (`_ds`/`_df`/`_cfg` in → report list
  out; **never a path** — the caller owns file I/O; sharpened in v3 for
  ext1-2): data-source-agnostic, so the same function checks a fixture artifact
  today, a synthetic in-memory object in a unit test, and a live artifact in a
  guard later.
- **Stdlib + xarray + geopandas + pyyaml + tomllib only** (all in-env; tomllib is
  py3.11 stdlib) — no new dependency (C3/non-goal).

**Asserted-if-present semantics for units (arch-2/risk-4).** Where a property is
recorded but not a hard contract surface — chiefly the HM-2 forcing units (wflow is
name-keyed) — the validator appends a divergence message **only when the attr is
present and its value is wrong**; an absent attr yields no message and never blocks.
This encodes "pin what a consumer reads, not what happens to be present" directly
in the validator logic.

**Relational validators — cross-artifact invariants (ext1-1, added in v3).** The
per-artifact `validate_<id>(obj) -> list[str]` interface cannot check the two
invariants the inventory pins *between* artifacts: every individual validator can
pass while a renamed or omitted gauge column silently corrupts the response
surface. Both invariants therefore get **dedicated relational validators** —
same idiom, explicit multi-object signatures:

- **`validate_hm_gauge_column_identity(toml_cfg, output_rlz_df, qstats_df) ->
  list[str]`** — the HM-4→HM-5→HM-7 chain, the model seam's single degree of
  freedom (§5.6), made first-class checkable. Grounded consumer facts it
  encodes (`experiment/export_wflow_results.py`): the reduction derives its
  gauge set from the **first** csv's columns via a **hard-coded `Q_` prefix
  filter** (lines 60–61, `Q_vars = [x for x in sim.columns if
  x.startswith("Q_")]`) and indexes every other csv with that set
  (`sim_all[Q_vars]`, lines 123/136). So a renamed gauge header in the first
  csv empties `Q_vars` and yields a **silently gauge-less Qstats**, while a
  mismatch in a later csv KeyErrors deep in the reduction — exactly the failure
  no per-artifact validator can see. Checks: **(1)** every non-`time`
  `output_rlz_df` column traces to a declared `[output.csv].column` entry in
  `toml_cfg` (map-typed entries → `<header>_<id>` pattern; entries without
  `map` → exact `header`), and every declared entry is represented; **(2)** the
  map-typed gauge columns carry the `Q_` prefix rule 3.11 hard-codes; **(3)**
  `qstats_df`'s gauge columns (header minus `statistic,tavg,prcp`, ordered per
  `export_wflow_results.py:66-67`) are list-equal to the `output_rlz_df` gauge
  set. **Scope boundary (C3):** the numeric `<id>` in `Q_130000086` is wflow's
  outlets-map cell value; the validator checks the `<header>_<id>` pattern and
  the cross-file identity, **not** the id's derivation from
  `staticmaps.outlets` (wflow-owned naming semantics — recorded in the seam doc
  as reliance, never asserted). The parallel `basavg`-substring filter
  (line 62) is the same class of consumer-prefix reliance, recorded in
  HM-5/HM-7; the fixture TOML declares no basavg column, so that branch is
  documented, not fixture-verified (C6).
- **`validate_wg5_catalog_grid(catalog_cfg, rlz_num, st_num) -> list[str]`** —
  WG-5's entry-key set vs the **intended** grid: expected keys exactly
  `{rlz_<n>_cst_<m> : n ∈ 1..rlz_num, m ∈ 0..st_num}` (**cst_0 included** —
  rule 3.08 consumes both the cst_0 list and the perturbed `expand` grid,
  `Snakefile_climate_experiment:318-319`); missing keys and unexpected keys are
  each reported. A dropped or extra catalog entry is invisible to the
  per-artifact WG-5 check (each remaining entry is well-formed) but breaks the
  realization×cst fan-out rule 3.09 depends on.

**Relational fixture wiring — all inputs persist, so both relational checks sit
in the continuously-verified class.** Per-cst TOMLs
(`model_runs/wflow_sbm_rlz_<n>_cst_<m>.toml`), `output_rlz_<n>_cst_<m>.csv`
(**not** `temp()`), `Qstats.csv`, and the rule-3.08 catalog YAML all persist on
the fixture (verified: 12 TOMLs + 12 output CSVs + Qstats + a 14-entry catalog,
rlz {1,2} × cst {0..6}). The gauge-identity test parametrizes over all 12
fixture `(toml, output_rlz)` pairs against the one `Qstats.csv`. The
catalog-grid test derives the intended grid from the fixture's own P3-1 config
snapshot (`<exp>/config/snake_config_climate_experiment.yml`:
`realizations_num: 2`; stress_test temp `step_num: 1` × precip `step_num: 2` →
`ST_NUM = 6` via the same `stress_test_grid` helper the Snakefile uses,
`shared/snake_utils.py:336`) — the run's *recorded* intent, self-consistent
with the tree even if the tracked test config later drifts. Synthetic fail
cases (Layer 1, below) deliberately break **one member of each correlated
set**: rename one `Qstats` gauge column; drop one catalog entry key.

**Test driver — two layers, so green is never indistinguishable from
nothing-checked (repo-1 in v2; rebuilt in v3 for ext1-2).**
`tests/test_interchange_contracts.py` exercises every validator at two layers:

- **Layer 1 — synthetic pass/fail, fixture-independent, ALWAYS executed.**
  Every one of the **15** validators (13 per-artifact + 2 relational) ships a
  pair of synthetic unit tests: a tiny conforming in-memory object (asserting
  `validate_x(good) == []`) and a deliberately broken one — transposed dim,
  renamed gauge column, dropped catalog key, missing coord, wrong var name —
  (asserting `validate_x(bad) != []`). **30 tests, no file I/O**, objects built
  directly in the test module (validators take parsed objects, never paths —
  above). This generalizes the v2 risk-2 mechanism from the 3 temp validators
  to ALL validators, and repairs the **defect in the v2 repo-1 fix**: with
  skipif-only coverage, a fixtureless checkout (fresh clone, CI) reported green
  having executed **zero** validator logic — "green" was indistinguishable from
  "nothing was checked." Now every validator's pass AND fail paths execute on
  every checkout. Cheap by construction: the validators are pure functions over
  small in-memory objects.
- **Layer 2 — real-fixture integration checks, skipped VISIBLY when the
  fixture is absent.** Each of the 15 real-artifact/relational cases points at
  `examples/test_local` (**untracked**: `git ls-files examples/test_local` → 0
  files) and carries the repo's fixture-absent guard, mirroring
  `tests/test_extract_climate_wf1.py` and
  `tests/test_check_project_consistency.py` (the only two existing
  fixture-reading tests): `@pytest.mark.skipif(not os.path.exists(<fixture
  path>), reason=_FIXTURE_ABSENT)` with a **single module-level reason
  constant** `_FIXTURE_ABSENT = "untracked examples/test_local fixture tree not
  present (interchange-contract integration layer skipped)"`. Fixture absence
  is thereby a **named, reported condition, not silence**: the §9 acceptance
  step runs `pixi run pytest -rs tests/test_interchange_contracts.py` and
  requires the skip report to show exactly the 15 integration cases under
  `_FIXTURE_ABSENT` (fixtureless) or the 12-green/3-temp-skip split (fixture
  present) — an all-skip integration layer is an explicitly stated **unmet
  integration gate** (close-out note, §8 commit 5), never an implied pass. The
  guard still keeps the suite green either way (the G5/C2 suite-green anchor:
  without it, the persisted-artifact and relational integration cases would
  `xr.open_dataset` / `open` missing paths and ERROR on a fresh clone).

**The temp() problem — the load-bearing coverage decision (C1/C6).** The fixture
sweep is decisive: **every `temp()` netCDF is absent from disk** — no
`rlz_*_cst_*.nc`, no `inmaps_rlz_*.nc`, no `outstates_rlz_*.nc` survive a completed
run. So the artifacts split into two honestly-labelled classes:

- **Continuously verified (persisted survivors) — 10 per-artifact validators,
  plus the 2 relational validators (all their inputs persist, above) = 12
  continuously-verified integration checks.** WG-1 (keyed-store
  extract), WG-2 (stress CSVs), WG-3 (weathergen YAMLs), **WG-5 (the rule-3.08
  catalog YAML persists — the side-channel contract is fully checkable even though
  the NCs it points at are gone)**, HM-1 (staticmaps), HM-2 wf1
  `inmaps_historical.nc`, HM-3 (staticgeoms), HM-4 (both TOMLs persist — base +
  per-cst), HM-5 (both output CSVs persist — wf3 `output_rlz_*` are **not**
  `temp()`), HM-7 (Qstats/basin). These validators are **green when the fixture is
  present** under `pixi run pytest` (skip on a fixtureless checkout, above).
  **HM-6a (wf1 `run_default/outstate/outstates.nc`) gets NO validator** — its only
  surface (name+location) is already pinned by HM-4's `[state].path_output`
  rewrite, so a standalone existence check would pad the count without checking an
  independent contract (risk-1). It stays a **doc row**, existence guaranteed
  transitively through HM-4.
- **NOT continuously verified (temp()-deleted content) — 3 validators.** WG-4
  (`rlz_*_cst_*.nc` content), WG-6 (`inmaps_rlz_*.nc` content), HM-6b wf3
  `outstates_rlz_*.nc` content. Their real-artifact test case
  **`pytest.skip("temp() artifact absent; capture via --notemp")`** when the file
  is absent (the default fixture state), un-skipped only by a **documented
  `--notemp` capture procedure** the *implementer* runs. **Their assertion logic
  is NOT left unexecuted (risk-2 in v2; generalized in v3):** like every other
  validator, each carries a Layer-1 synthetic pass/fail pair that runs
  fixture-independently every suite — so each temp validator's pass AND fail
  paths are proven this milestone; only the *on-disk real-artifact* check stays
  gated. No fixture edit, no behavior change. **This design does not run the
  `--notemp` capture** (it would modify `examples/test_local`, outside the
  deliverable).

**Honest coverage statement (C6, success-criterion 2 read strictly).**
Success-criterion 2 ("validator per contract, green on fixture") is satisfied
**for every persisted artifact and both relational invariants** (12 integration
checks green when the fixture is present); for the three temp()-deleted content
contracts the *on-disk* check is *skip-until-captured*. **Every** validator's
logic — all 15, not just the temp three — is proven **on every checkout** by the
Layer-1 synthetic pass/fail tests (v3, ext1-2), so criterion-2's "validator per
contract" is met as **integration-checked where the artifact persists AND
synthetic-proven everywhere**, never as an unexecuted claim. The design
does **not** claim on-disk green for temp artifacts and does **not** rescue them
with re-derivation proxies (e.g. "inmaps_rlz shape ≈ inmaps_historical shape") — a
proxy would be confirmation bias, not a check of the real artifact. Coverage
bookkeeping is honestly bounded:

- **WG-5 checks the catalog bookkeeping only, NOT WG-4/WG-6 NC content (risk-3
  correction).** WG-5 pins the entry-key grid + driver options (`uri`,
  `driver.name`, `preprocess`, `crs`, `category`, `data_type`) — i.e. that a
  catalog *entry* exists per realization×cst. It pins **nothing** about the NC's
  dims, variable names, units, or grid — the content WG-4/WG-6 guarantee. The v3
  relational `validate_wg5_catalog_grid` strengthens the *bookkeeping* check
  (entry-key completeness vs the intended grid) but likewise says nothing about
  NC content. So the NC-content contract is **skip-until-captured with no
  indirect proxy** (consistent with §6.3's rejection of proxies); the earlier
  "WG-5 gives indirect coverage of WG-4/WG-6 content" framing is dropped as an
  overstatement.
- The `--notemp` capture procedure is documented in the seam doc's validator
  index so a future run can flip those three on-disk checks to green without design
  change.

**Granularity.** One validator per pinned artifact id plus the two relational
validators — **15 validator functions** (10 persisted per-artifact +
WG-4/WG-6/HM-6b + `validate_hm_gauge_column_identity` +
`validate_wg5_catalog_grid`; HM-6a carries none, above). Shared helpers
(assert dims/coords/CRS/dtype) factored into small internal helpers within the
module. Branch-awareness: era5 is the fixture; chirps-only facts (WG-1 chirps
precip-only + orography sidecar) are asserted only under a chirps fixture — marked
`not fixture-verified (no chirps fixture)` in the validator index, never faked
green.

**Counting axis (v3, the SINGLE authoritative statement — §7, §9 and §11
reference it; the v1→v2 lesson: one counting axis stated once, never
re-quoted):** *15 validator functions — 13 per-artifact across 14 doc rows (HM-6
split into HM-6a/HM-6b; HM-6a is a doc row with no validator) plus 2 relational
(gauge-column identity; catalog grid). Every validator has a Layer-1 synthetic
pass/fail test pair (30 synthetic tests) that executes on every checkout,
fixture or not. With the era5 fixture present, 12 integration checks are green
(10 persisted per-artifact + 2 relational) and 3 (WG-4 / WG-6 / HM-6b) are
skip-until-captured on disk; on a fixtureless checkout all 15 integration
checks skip under the named `_FIXTURE_ABSENT` reason — surfaced by the
documented `pytest -rs` acceptance step — while the 30 synthetic tests still
execute. The suite is green either way, but never green-by-executing-nothing.*

### 5.6 Bounded-substitution walkthrough

Each seam doc ends with the concrete file-touch map a replacement would need — the
roadmap's "bounded substitution" made reviewable.

**Weather-generator seam — replacing weathergenr.** A drop-in generator must:

- **Consume** WG-1 (`extract_historical.nc`, the 7-var K grid) and WG-2 (the
  `cst_<m>.csv` perturbation grid) — or provide its own reader for them.
- **Produce** WG-4 netCDFs at the DAG-globbed paths
  `realization_<n>/rlz_<n>_cst_<m>.nc` (incl. `cst_0`), each a `(time,lat,lon)`
  EPSG:4326 raster with ≥ `precip`,`temp` and `crs=4326`/`category=meteo` so the
  hydromt catalog (WG-5) loads it via `raster_xarray`+`harmonise_dims`.
- **Repo files it replaces:** rules 3.04–3.07 `shell:`/`script:` targets in
  `Snakefile_climate_experiment` (the two `Rscript --vanilla` bodies pointing at
  `weathergen/*.R`, plus the two config-prep scripts if the config surface WG-3
  changes). **Files it must NOT change:** rule 3.02 (WG-1 producer), rule 3.08
  (WG-5 catalog), rule 3.09 (WG-6 downscale) — those are the pinned boundaries.
- **Contracts it must satisfy:** WG-1/WG-2 (in), WG-4 shape + naming (out), and —
  if it emits its own catalog — WG-5 **including the catalog↔grid invariant**
  (an entry per realization×cst incl. `cst_0`). Validators WG-1,2,4,5 plus the
  relational `validate_wg5_catalog_grid` (§5.5) are the acceptance check.

**Hydrological-model seam — replacing Wflow-SBM.** A drop-in model must:

- **Consume** HM-2/WG-6 forcing (`(time,lat,lon)` `precip`/`pet`/`temp` on its
  own grid — the variable **names** are the contract; a run-config maps them, as
  wflow does via `[input.forcing]` RHS values) and a static description equivalent
  to HM-1 (the OUR-referenced name set on a co-registered grid), driven by a run
  config equivalent to HM-4's rewrite fields — **including `time.timestepsecs`**
  and a calendar (wflow's rewrite sets `standard` on the wf3 forcing/TOML to match
  the weathergenr `noleap` origin, distinct from the wf1 `proleptic_gregorian`
  pin; arch-4).
- **Produce** HM-5 per-run discharge CSVs with the **`<header>_<mapid>` column
  identity** the reduction (rule 3.11) keys on — the single degree of freedom that
  flows HM-4 `[output.csv].column` → HM-5 → HM-7, and the `Q_` header prefix the
  reduction hard-codes (`export_wflow_results.py:61`). A swap that renames the
  gauge column silently breaks HM-7 — exactly the failure the relational
  `validate_hm_gauge_column_identity` (§5.5) exists to catch.
- **Repo files it replaces:** rule 1.03 build (`hydromt build`), rule 1.08/3.09
  forcing prep, rules 1.09/3.10 `run_wflow` `shell:` bodies
  (`julia … Wflow.run()`), and `downscale_climate_forcing.py`'s TOML-rewrite (HM-4
  fields) if the run-config format changes. **Files it must NOT change:** rule
  3.11 `export_wflow_results` (the reduction) — provided HM-5's column identity is
  honored.
- **Contracts it must satisfy:** HM-1 (static reliance), HM-2 (forcing in), HM-4
  (run-config rewrite fields), HM-5 (output column identity), HM-7 (reduction
  input). HM-6a/HM-6b warm state is *not* a substitution constraint (unconsumed
  sink). Validators HM-1,2,4,5,7 plus the relational
  `validate_hm_gauge_column_identity` (§5.5 — the single-degree-of-freedom
  check across HM-4→HM-5→HM-7) are the acceptance check.

## 6. Alternatives considered

### 6.1 Contract-doc granularity — combined / per-seam / per-artifact

- **One combined `interchange-contracts.md`.** Rejected: the two seams are
  *independent* substitution stories (a generator swap and a model swap touch
  disjoint rule sets); one file forces a reader interested in one seam to wade
  through the other, and the bounded-substitution walkthroughs don't compose.
- **One file per artifact (13 files).** Rejected: over-decomposed; a swapper reads
  a seam end-to-end, and cross-artifact invariants (the HM-4→HM-5→HM-7 column
  identity; WG-4↔WG-5 catalog coupling) live *between* artifacts, so per-file
  fragments hide exactly the couplings that matter.
- **One file per seam (SELECTED).** Matches the `dev/reference/workflows/*.md` precedent, keeps
  each substitution story whole, and hosts the cross-artifact invariants in one
  place. Two docs.

### 6.2 Validator placement — package module vs test-local helper vs schema library

- **Third-party schema library** (e.g. `xarray-schema`, `pandera`, `frictionless`).
  Rejected: a new dependency (requires prior approval; `new-dependency-requires-approval`
  memory), and hand-rolled divergence-report validators suffice at this scale
  (15 validators over 13 pinned artifacts) on plain xarray/geopandas.
- **`tests/`-local helper functions.** Considered: simplest, zero package
  footprint. Rejected as primary: intake decision 2 wants validators "reusable as
  future in-pipeline guards" — a `tests/`-only helper is not liftable without a
  later move into the package. Sacrifices C5 for a marginal footprint win.
- **Unwired package module `blueearth_cst/shared/interchange_contracts.py`
  (SELECTED).** Liftable (C5), pure-function, imported by the test file and by no
  rule (C2 — unreferenced by the DAG, so zero behavior change), stdlib+xarray only
  (C3). The one judgment call: it adds a module to the package that nothing runs
  yet — justified because "unreferenced module" ≠ "pipeline behavior change" and
  the alternative forfeits liftability.

### 6.3 temp()-artifact coverage — skip+capture vs re-derivation proxy vs fixture-persist

- **Re-derivation proxy** (assert the temp NC's shape ≈ a persisted sibling's
  shape). Rejected: it checks a *proxy*, not the real artifact; it would let the
  suite report "green" for a contract it never actually exercised — confirmation
  bias against C1/C6.
- **Persist the temp NCs into the fixture** (drop `temp()` for the fixture, or
  commit captured NCs). Rejected here: it is a **behavior change** (touches
  Snakefile `temp()` wrapping or bloats the tracked/dev fixture) — out of the
  zero-behavior-change milestone; and `temp()` on large `RLZ_NUM×ST_NUM` runs is a
  disk-management contract of its own (AGENTS.md).
- **Skip-until-captured + documented `--notemp` procedure (SELECTED).** Validator
  present and correct, `pytest.skip` while the file is absent, un-skipped by an
  implementer-run capture; honest coverage statement (§5.5). Satisfies C6 (states
  plainly what is/isn't continuously verified) without a behavior change.

### 6.4 In-pipeline enforcement now vs docs+validators-as-tests

Fixed by intake decision 2 (docs + validators-as-tests; no in-pipeline
enforcement). Recorded here for completeness: wiring a validator into a rule body
would touch Snakefile rule bodies (behavior-adjacent) and is deferred; the
validators are written liftable so that future milestone is a small lift, not a
rewrite (§5.5).

### 6.5 Validator error idiom — report list vs ValueError vs AssertionError (v2)

Added in v2 (repo-2). Analogs in-repo: `compare_project_consistency` (pure
`-> list[str]` divergence report, empty ⇒ pass — the drift-guard) and
`validate_experiment_name` (`-> str`, raises `ValueError`).

- **Bare `assert` / `AssertionError` (v1's implied idiom).** Rejected: stripped
  under `python -O` / `PYTHONOPTIMIZE`, so a future optimized in-pipeline guard
  (the C5 lift target) would silently no-op instead of blocking a bad artifact —
  fails open on the exact path the design is built for. Banned in validator bodies.
- **`-> None` raising `ValueError` on first violation** (`validate_experiment_name`
  analog). Considered: a real house idiom and `-O`-safe. Rejected as primary: stops
  at the first bad fact, so a swapper debugging a candidate artifact sees one
  violation at a time; and a raising guard is less composable with a test than a
  returned list.
- **Pure `-> list[str]` divergence report (SELECTED,
  `compare_project_consistency` analog).** `-O`-safe, surfaces every violation at
  once, and the same function serves a test (`assert result == []`) and a future
  guard (raise/log when non-empty) with no move (C5). "Asserted-if-present" units
  (arch-2) fall out naturally: append a message only when the attr is present and
  wrong.

### 6.6 Fixture-independent coverage — synthetic-all vs committed fixture vs skipif-only (v3)

Added in v3 (ext1-2). The v2 fix (fixture-absent `skipif` on all fixture tests,
repo-1) made a fixtureless checkout green — but green **by executing nothing**,
indistinguishable from a real pass. Three repairs weighed:

- **Skipif-only (the v2 status quo).** Rejected: a regression in validator
  logic or in a contract expectation lands undetected exactly where per-commit
  checks are most likely to run (fresh clone, CI); the milestone's
  "machine-checked" claim would quietly hold only on machines that happen to
  carry the untracked fixture.
- **Commit a small contract fixture.** Rejected: the interchange artifacts are
  **generated run outputs** (staticmaps.nc and the netCDFs are multi-MB
  binaries; the repo's hard constraint forbids committing run outputs), and a
  committed copy acquires independent provenance that rots against the
  regenerated fixture — a drift-prone second source of truth for the very
  contracts being pinned. A *minimal hand-authored* fixture is just the
  synthetic-object layer with extra git weight and none of its in-memory
  cheapness.
- **Synthetic pass/fail for every validator + visibly-reported integration
  skips (SELECTED).** Validators are pure functions over small parsed objects,
  so a conforming + broken object pair per validator is cheap (30 small tests,
  no file I/O), executes the full assertion logic on every checkout, and adds
  no artifact to git. The real-fixture layer stays the integration check under
  the repo's `skipif` convention, with absence surfaced as a named skip reason
  plus a documented `pytest -rs` acceptance step (§5.5). Logic coverage
  everywhere; artifact coverage wherever the fixture exists; nothing silent.

## 7. Consequences and risks

**Positive.**

- Both substitution seams become **explicit, documented, and machine-checked**;
  "swappable" is checkable for the persisted majority of artifacts, not a slogan.
- A future generator/model swap (P3-2c, R6 model-flexibility) starts from a pinned
  contract + an acceptance validator set, not from Snakefile archaeology.
- The `interchange_contracts.py` validators are **liftable into in-pipeline
  guards** with no move — a ready foundation if enforcement is later wanted.
- Two cross-artifact invariants are surfaced, pinned, **and first-class
  checked by dedicated relational validators** (v3, ext1-1): the
  HM-4→HM-5→HM-7 gauge-column identity (`validate_hm_gauge_column_identity`)
  and the WG-5 catalog↔grid coupling (`validate_wg5_catalog_grid`) — both in
  the continuously-verified class, since all their inputs persist.
- Every validator's logic executes on **every checkout** via the Layer-1
  synthetic pass/fail tests (v3, ext1-2) — a fixtureless green can no longer
  mean nothing-was-checked, and fixture absence is a named, reported skip
  condition rather than silence.

**Negative / new obligations.**

- **Temp()-content contracts are not continuously green on disk** (WG-4, WG-6,
  HM-6b): 3 of the 15 validators skip the *on-disk* check on the default fixture,
  un-skipped only by an implementer `--notemp` capture. Their **logic is proven**
  every suite by the Layer-1 synthetic pass/fail tests (risk-2, generalized in
  v3), so this is a coverage gap on the *real artifact*, not on the validator's
  correctness. Counts per the single authoritative counting axis in §5.5
  (stated once there, referenced here — not re-quoted). A reader must not read
  "15 validators" as "15 continuously green on disk."
- **A new package module runs in no pipeline** — a small standing surface
  (`interchange_contracts.py`) that exists only for tests until a future guard
  lifts it. Recorded so it is not mistaken for dead code.
- **Contracts are era5-grounded.** chirps-branch facts (WG-1 chirps precip-only,
  the orography sidecar) are documented from code but not fixture-verified — a
  chirps regression would not trip these validators until a chirps fixture exists.

**Neutral.**

- The contract docs record a **snapshot** of current shapes; if a later,
  behavior-changing milestone alters an artifact, the doc + validator must be
  updated in that milestone (the contract is a living record, like
  `dev/reference/workflows/*.md`).

**Risks.**

- **R1 — a "pinned" fact is actually incidental.** Pinning something no consumer
  relies on (e.g. a provenance attr) would over-constrain a swap. Mitigation: the
  per-artifact "pinned vs unpinned" column forces each inclusion to name the
  consumer that relies on it; §5.1's tier rule is the test. **v2 exercised this
  test on HM-2 units** (arch-2): units are populated but no consumer reads them
  (wflow is name-keyed), so they were demoted from pinned to asserted-if-present —
  a worked example that R1's discipline is applied, not just declared.
- **R1b — a redundant "pin" pads coverage without adding a check.** An existence
  pin already guaranteed elsewhere (HM-6a's name+location, covered by HM-4) inflates
  the validator count without verifying an independent contract. Mitigation (v2,
  risk-1): HM-6a carries no validator; its only contract surface (name/location) is
  pinned by HM-4's `[state].path_output` declaration.
- **R2 — the era5-only fixture masks a branch-specific contract error.** A chirps
  or eobs code path could diverge from the documented contract undetected.
  Mitigation: §5.5 marks chirps facts `not fixture-verified` explicitly; the
  validator index flags the gap rather than implying coverage.
- **R3 — silent temp()-capture rot.** If the `--notemp` procedure is never run,
  the three skip-validators' *on-disk* check stays unexercised and could drift from
  the real artifact. Mitigation: documented procedure in the validator index + an
  honest coverage statement; **the Layer-1 synthetic pass/fail tests (risk-2 in
  v2, extended to all 15 validators in v3) keep each validator's assertion logic
  exercised every suite regardless of capture**, so rot is bounded to the
  artifact-vs-logic gap, not both; a future in-pipeline-guard milestone
  exercises them live on disk.
- **R4 — CST-scope overreach.** A validator that asserts an upstream-owned field
  (a wflow physics param, a full staticmaps schema) would breach C3. Mitigation:
  validators assert only the §5.1-tier-1/2 surface; the "deliberately unpinned"
  column is the audit trail.

## 8. Migration + commit plan

**Branch.** One task branch off the milestone line per the repo branch model
(`branch-model` memory). Commit prefix `p32b:`. **Zero pipeline edits**, so every
per-commit gate is cheap: `pixi run pytest` (full suite) + three `--dry-run`s
(all three Snakefiles) clean. No re-record, no semantic-tree-diff needed (no output
changes).

**Commit sequence (small, reviewable):**

1. `p32b: weather-generator seam contract doc` — add
   `dev/reference/contracts/weather-generator-seam.md` (WG-1..WG-6 inventory + bounded-
   substitution walkthrough + validator index). Docs-only. Gate: three dry-runs
   clean (no code touched), `pytest` unchanged-green.
2. `p32b: hydrological-model seam contract doc` — add
   `dev/reference/contracts/hydrological-model-seam.md` (HM-1..HM-5, **HM-6a/HM-6b split**,
   HM-7 + walkthrough + index), including the grounded warm-state finding (HM-6a/b
   unconsumed sink; HM-6a existence via HM-4, no validator), the HM-6a path
   derivation (`dir_output` + `[state].path_output`), the HM-4 rewrite-field set
   (incl. `time.timestepsecs` + the wf3 `standard`-calendar note), and the
   HM-4→HM-5→HM-7 column-identity invariant. Docs-only. Same gate.
3. `p32b: interchange-contract validators (persisted + relational) + synthetic layer` —
   add `blueearth_cst/shared/interchange_contracts.py` with the 10
   persisted-artifact validators **plus the 2 relational validators**
   (`validate_hm_gauge_column_identity`, `validate_wg5_catalog_grid`; ext1-1) as
   pure `-> list[str]` divergence reports over parsed objects (no `assert`/
   `AssertionError` in bodies; §6.5) + `tests/test_interchange_contracts.py`
   with **both layers** (§5.5): a synthetic pass/fail pair per validator (24
   fixture-independent tests at this commit) and the 12 real-fixture
   integration cases, each integration case carrying the fixture-absent
   `skipif` with the `_FIXTURE_ABSENT` reason constant (repo-1/ext1-2). Gate:
   **`pixi run pytest -rs tests/test_interchange_contracts.py` green** —
   integration cases pass with the fixture, skip *visibly* under
   `_FIXTURE_ABSENT` without it; the synthetic tests pass either way; three
   dry-runs clean (the new module is imported by no Snakefile — confirm the DAG
   is unchanged); full suite green (incl. on a fixtureless checkout).
4. `p32b: temp()-artifact validators (skip-on-disk + synthetic-proven) + capture procedure` —
   add the WG-4/WG-6/HM-6b validators (pure `-> list[str]`) with **(a)** their
   real-artifact test cases carrying both the fixture-absent `skipif` and the
   temp-absent skip, and **(b)** their Layer-1 synthetic pass/fail pairs
   (risk-2; completing 30 synthetic tests total); and document the `--notemp`
   capture procedure in both seam docs' validator indexes. Gate: `pytest -rs`
   green (the three real-artifact cases **skip** with the documented reason;
   all 30 synthetic cases **pass**); three dry-runs clean.
5. `p32b: seam-contract index + roadmap close-out note` — a short
   `dev/reference/contracts/README.md` (or roadmap § update) linking the two seam docs and
   stating coverage by **reference to the §5.5 counting axis** (15 validators;
   fixture present → 12 integration-green + 3 skip-until-captured on disk;
   fixtureless → 15 named skips + 30 synthetic tests still executed, an
   explicitly stated unmet integration gate; chirps not-fixture-verified).
   Gate: dry-runs clean; `pytest` green.

(Commits 1–2 could merge into one docs commit and 3–4 into one validator commit;
kept split for reviewability. No commit edits a Snakefile, a `script:` target, a
module runtime path, `blueearth_cst/weathergen/*.R`, or any fixture artifact.)

## 9. Validation plan

**Reproducibility & validation gates (who checks each):**

- **Per-commit gate** — three `snakemake --dry-run`s + `pixi run pytest
  tests/test_cli.py` prove **zero behavior change** (DAG unchanged; the new module
  is DAG-invisible). *cst-architect verifies at each commit; delegated implementer
  runs.*
- **Validator suite (two layers, §5.5)** — `pixi run pytest -rs
  tests/test_interchange_contracts.py`: with the fixture present, the **12
  integration checks** (10 persisted per-artifact + 2 relational) are **green**
  and the 3 temp real-artifact cases **skip** (not fail); **all 30 synthetic
  pass/fail cases pass regardless of fixture**. On a fixtureless checkout all
  15 integration cases skip **and the `-rs` report must show them under the
  named `_FIXTURE_ABSENT` reason** — the checker reads the skip report, so an
  all-skip integration layer is a visible, named unmet-integration condition,
  never an indistinguishable green (ext1-2). *python-engineer implements;
  cst-architect confirms the green/skip/synthetic split matches the §5.5
  counting axis.*
- **Relational fail-path check (ext1-1)** — the synthetic fail cases for the two
  relational validators each **break one member of a correlated set** (rename
  one Qstats gauge column against a conforming TOML+output_rlz pair; drop one
  catalog entry key from a conforming grid) and must return a non-empty report.
  *python-engineer implements; cst-architect reviews the broken-member choice.*
- **Full suite** — `pixi run pytest tests/` unchanged-green (no pre-existing test
  perturbed by the new module import), **including on a machine without
  `examples/test_local`** (integration layer skips visibly; the synthetic layer
  still executes every validator — repo-1/ext1-2). *python-engineer.*
- **Scope audit (C3)** — a review pass that every validator asserts only a
  §5.1-tier-1/2 surface (no wflow physics param, no full-schema assertion) and
  every "pinned" fact in the docs names a consumer. *cst-architect / model-validator.*
- **temp()-capture (optional, un-skip path)** — the documented `--notemp` capture
  run flips the WG-4/WG-6/HM-6b *on-disk* cases to green; **not run in this
  milestone** (would modify the fixture). *model-builder, when a future run wants
  full on-disk coverage.*

**Milestone gate = user review** of the two seam contract docs + the green/skip
validator suite (intake decision 4). Acceptance criteria the deliverables must
meet (all counts per the §5.5 counting axis — the single authoritative
statement, referenced, never re-quoted):

- **Complete inventory** — every §5.2/§5.3 artifact has a doc row; each pinned
  artifact has a validator (HM-6a is a doc row whose existence is pinned via HM-4,
  by design carrying no validator — risk-1); **both cross-artifact invariants
  have dedicated relational validators** (`validate_hm_gauge_column_identity`,
  `validate_wg5_catalog_grid` — ext1-1); the excluded non-interchange artifacts
  (`sim_dates.csv`/`resampled_dates.csv`, `wf1_raw/extract_historical.nc`) are
  noted considered-and-excluded.
- **Green/skip on fixture, visibly reported** — with the fixture present, the 12
  integration checks pass and the 3 temp real-artifact cases skip with the
  documented reason (not error). On a fixtureless checkout all 15 integration
  cases skip **under the named `_FIXTURE_ABSENT` reason, verified by reading
  the `pytest -rs` report**, and the suite is green (repo-1/ext1-2).
- **Fixture-independent logic coverage** — every one of the 15 validators has a
  synthetic pass AND fail test that executes on every checkout (30 tests); no
  validator's logic is attested only by skipped tests (ext1-2, generalizing
  risk-2).
- **`-O`-safe validator idiom** — validators return `-> list[str]` divergence
  reports (empty ⇒ pass); no `assert`/`AssertionError` in validator bodies (repo-2,
  §6.5).
- **Zero behavior change** — three dry-runs + full suite green at every commit
  (fixtured and fixtureless); nothing re-recorded; `git diff` touches only
  `dev/reference/contracts/**`, `blueearth_cst/shared/interchange_contracts.py`,
  `tests/test_interchange_contracts.py`.
- **Bounded-substitution walkthrough** — each seam doc names the exact repo files
  a swap touches and the contracts it must satisfy (§5.6).
- **Scope-clean (C3)** — no validator or doc row asserts an upstream-owned
  internal; each pinned fact traces to an OUR-side consumer.

## 10. Open questions

- **OQ-1 — merge the docs/validator commits?** §8 splits into 5 commits for
  reviewability; a reviewer may prefer 2 (one docs, one validators). Taste call.
- **OQ-2 — `dev/reference/contracts/` vs `dev/reference/workflows/` placement.** The seam docs are
  contracts *across* workflows, not per-workflow, so a new `dev/reference/contracts/` subdir
  is proposed; a reviewer preferring to co-locate with `dev/reference/workflows/*.md` may
  fold them there. Recommend the new subdir (different axis: seam, not workflow).
- **OQ-3 — chirps fixture for branch coverage (future).** The WG-1 chirps facts
  and orography sidecar are documented-not-verified. A chirps fixture would flip
  them to continuously-verified — out of scope here (no chirps fixture exists;
  behavior-adjacent to add one). Recorded so the coverage gap is known.
- **OQ-4 — run the `--notemp` capture to flip the three temp validators' *on-disk*
  cases green?** Deferred to a future run (modifies the fixture; not a
  contracts-only edit). Their assertion logic is already proven every suite by the
  synthetic tests (§5.5, risk-2), so this only closes the artifact-vs-logic gap. The
  procedure is documented so it is a one-command lift when wanted.
- **OQ-5 — lift validators into in-pipeline guards (future).** Intake decision 2
  keeps enforcement out of this milestone; the module is written liftable. A later
  milestone may wire `validate_*` into the relevant rule bodies (behavior change —
  its own cycle).
- **OQ-6 — WG-3 config-surface depth.** The contract pins the weathergenr config
  *key set + types* the R side reads, derived read-only from `*.R`. Whether to pin
  value *ranges* (e.g. `warm.signif.level ∈ (0,1)`) or leave them to weathergenr
  is a depth call; recommend key-set-only (a replacement generator may define its
  own config surface entirely — WG-3 is the *current* generator's, not a universal
  contract).

## 11. Revision log

- **v1 (2026-07-24)** — initial draft for the G1 framing gate. Grounded against:
  the two authoritative intakes; the p32a design (accepted precedent, structure +
  rigor); all three Snakefiles (post-p32a: wf1 rule 1.10 `extract_climate_grid_wf1`
  + the `climate_analysis/` modules already landed); the fixture tree
  (`examples/test_local`, era5) inspected with xarray for dims/coords/vars/units/
  attrs on `extract_historical.nc` (keyed + wf1_raw), `inmaps_historical.nc`,
  `staticmaps.nc`; the base vs per-cst `wflow_sbm.toml` diff (the 8-field
  rewrite surface + the `cold_start__flag=true` warm-state finding); the
  weathergen R scripts (read-only); the rule-3.08 catalog YAML; the stress-test /
  output / Qstats / basin CSVs; and `dev/reference/workflows/climate_experiment.md` (contract-doc
  precedent). Key grounded corrections to the intake's hints: (1) **warm-state is
  an unconsumed named sink, not a chaining invariant** (cold_start stays true; no
  `instates` input on rule 3.10) — HM-6; (2) **`output.csv` column identity is
  config-driven** (`time,Q_130000086`, one col per `[output.csv].column` entry),
  not a literal `*_basavg`/multi-gauge list — HM-5, flowing HM-4→HM-5→HM-7 as one
  degree of freedom; (3) **extraction temp is Kelvin**, model-grid forcing temp is
  °C — the conversion is HM-2's contract, correcting p32a's °C assumption. Load-
  bearing decisions: contract docs under `dev/reference/contracts/` one-file-per-seam;
  validators in an unwired liftable package module
  (`blueearth_cst/shared/interchange_contracts.py`); temp()-content contracts
  handled by skip-until-captured (all temp() NCs absent from the fixture) with an
  honest coverage statement (10 continuously green, 3 skip-until-captured, chirps
  not-fixture-verified). Open questions OQ-1..OQ-6 flagged.
- **v2 (2026-07-24)** — revised after the internal panel (risk / architecture /
  repo-fit; 0 blocking / 7 major / 9 minor). All 16 findings dispositioned in
  the run ledger (final ledger in the review record,
  `interchange-contracts-design-review-record.md`); **no finding rejected**
  (no user-ratification trigger). Material changes, by index group:
  - **A (risk-1, arch-1, arch-3):** HM-6 **split into HM-6a (wf1, persisted) /
    HM-6b (wf3, temp)**, mirroring HM-2/WG-6. HM-6a wf1 path **corrected** to
    `run_default/outstate/outstates.nc` with its derivation (`dir_output` +
    `[state].path_output`) recorded. **HM-6a carries no validator** — its
    name+location surface is already pinned by HM-4, so a standalone existence check
    would pad the count (risk-1 redundancy point, decided pinned-via-HM-4 not
    duplicated). **One counting axis** written once in §5.5 and reused verbatim in
    §7/§9/§11: *13 validator functions across 14 doc rows; fixture present → 10
    green + 3 skip-until-captured on disk; fixtureless → all 13 skip, suite green; 3
    temp validators synthetic-proven.*
  - **B (arch-2, repo-3, risk-4):** HM-2 units contract **rebuilt on observed
    attrs**. Fixture-verified layout recorded verbatim: `precip` carries **both**
    `units='mm d**-1'` and `unit='mm'`; `pet` `unit='mm'`; `temp` `unit='degree C.'`
    (all wflow-native values under the `unit` **singular** key; `units` plural
    survives on precip only). **Units DEMOTED from pinned to asserted-if-present**
    (wflow is name-keyed via `[input.forcing]`; no consumer reads them — R1). Two
    reviewer sub-claims **corrected from primary source**: WG-1 uses `units`
    (plural), not `unit` (contra repo-3b — v1 prose was right); HM-2 temp **does**
    carry `unit='degree C.'`, so the K→°C divergence note **is** fixture-grounded
    (contra arch-2 parenthetical). HM-2 precip value corrected to `mm` (under
    `unit`).
  - **C (repo-1):** **all 13** validator tests get the repo's `skipif(not
    os.path.exists(<fixture>))` guard (not just the 3 temp cases); coverage restated
    as green-when-fixture-present, suite green on fixtureless checkouts.
  - **D (repo-2):** validator idiom fixed to pure **`-> list[str]` divergence
    report** (empty ⇒ pass; the `compare_project_consistency` drift-guard analog);
    `assert`/`AssertionError` **banned in validator bodies** for `-O` safety (§6.5,
    new subsection).
  - **E (risk-2):** **synthetic in-memory pass+fail unit tests** added per temp
    validator so their assertion logic is executed every suite (fixture-independent,
    no fixture edit, no behavior change); §9 acceptance updated to skip-on-fixture
    AND synthetic-proven.
  - **F (arch-4..7, risk-3, risk-5, repo-4):** HM-4 gains `time.timestepsecs` +
    the wf3 `standard`-calendar note; WG-5 absolute-uri noted deliberately-unpinned;
    HM-1 unpinned count corrected to **~39** (44 total staticmaps vars) and the
    `[input.forcing]` mapping direction corrected (CSDMS-name keys → var-name
    values); considered-and-excluded notes added for `wf1_raw/extract_historical.nc`
    and `sim_dates.csv`/`resampled_dates.csv`; the WG-5 "indirect coverage of
    WG-4/WG-6" overstatement dropped (risk-3, bookkeeping-only); repo-4 placement
    audited clean (ledger row only, no doc change).
- **v3 (2026-07-24)** — revised after external review round 1 (verdict revise;
  2 majors, both accepted; ledger Round = external-r1).
  - **ext1-1 — relational validators.** The per-artifact
    `validate_<id>(obj) -> list[str]` interface could not check the design's
    own two cross-artifact invariants (every individual validator green while a
    renamed/omitted gauge column silently corrupts the response surface). Added
    two dedicated **relational validators** (same idiom, explicit multi-object
    signatures; §5.5): `validate_hm_gauge_column_identity(toml_cfg,
    output_rlz_df, qstats_df)` for the HM-4→HM-5→HM-7 chain — newly grounded on
    the reduction's **hard-coded `Q_` prefix filter and first-file column
    indexing** (`export_wflow_results.py:60-61,123,136`; the silent-corruption
    path made concrete), with a C3 scope boundary (the outlets-map id
    derivation stays wflow-owned: documented, not asserted) — and
    `validate_wg5_catalog_grid(catalog_cfg, rlz_num, st_num)` for the
    catalog-entry grid (cst_0 included, Snakefile lines 318–319; intended grid
    derived from the fixture's P3-1 config snapshot via `stress_test_grid`,
    `snake_utils.py:336`). All relational inputs persist on the fixture (12
    TOMLs, 12 output CSVs, Qstats, the 14-entry catalog), so both join the
    continuously-verified class. Folded into G2, the WG-5/HM-5 rows, §5.5 (new
    relational block + wiring), §5.6 acceptance sets, §7, §8 commit 3, §9.
  - **ext1-2 — all-skip green (re-raises repo-1/risk-2); prior fix defective.**
    The v2 repo-1 fix (fixture-absent `skipif` on all 13 fixture tests) made a
    fixtureless checkout — fresh clone, CI, exactly where per-commit checks run
    — report **green having executed zero validator logic**: green
    indistinguishable from nothing-checked. Repair combines two of the
    reviewer's directions: **(a)** the risk-2 synthetic pass/fail mechanism
    **generalized from the 3 temp validators to ALL 15 validators** (30
    fixture-independent in-memory tests; validators sharpened to take parsed
    objects only, the test layer owns I/O); **(b)** the real-fixture layer
    retained as the integration check under the repo `skipif` convention, with
    a module-level `_FIXTURE_ABSENT` reason constant and a documented
    `pytest -rs` acceptance step, so fixture absence is a **named, reported
    unmet integration gate**, not silence. **(c)** A committed contract
    fixture was **rejected** (generated run outputs; git size + provenance
    drift — new §6.6 records all three options). Counting axis restated
    **once** in §5.5 (15 validators; fixture present → 12 integration-green +
    3 skip-until-captured; fixtureless → 15 named skips + 30 synthetic tests
    still executed) and **referenced — never re-quoted — from §7/§8/§9** (the
    v1→v2 lesson applied strictly).
