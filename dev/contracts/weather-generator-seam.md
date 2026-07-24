# Contract: weather-generator seam (WG-1 .. WG-6)

> **Genre:** dev-facing interchange contract. **Audience:** a future *swapper* —
> someone replacing `weathergenr` with an alternative weather generator, or the
> R6 model-flexibility work — read end-to-end. Not an end-user doc (hence `dev/`,
> not `docs/`; precedent `dev/workflows/climate_experiment.md`).
> **Source of record:** `dev/p32b/interchange-contracts-design.md` (ACCEPTED
> 2026-07-24, §5.2 / §5.4 / §5.6 / §5.5). Every load-bearing fact below cites a
> Snakefile line, a script line, or an observed fixture artifact; do not add a
> contract fact that is not so grounded.

## Scope and method

The **weather-generator seam** is the point in `Snakefile_climate_experiment`
(wf3) where the stochastic weather generator could be swapped for an alternative
without re-architecting the rest of the pipeline. `weathergenr` (R) is the
current occupant, but **this contract is generator-agnostic**: it pins what wf3
hands *in* to the generator and expects *out* of it, not weathergenr's internals.

**Grounded in** the fixture tree `examples/test_local` (era5 branch,
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

- **path pattern:** `climate_historical/<key>/extract_historical.nc`, where
  `<key> = <clim_source>_<startYYYYMMDD>_<endYYYYMMDD>` (P3-1 keyed store).
- **producer:** rule 3.02 `extract_climate_grid`
  (`blueearth_cst/climate_analysis/extract_historical_climate.py`).
- **consumer:** rule 3.06 `generate_weather_realization` (weathergenr
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

- **path pattern:** `<exp>/stress_test/cst_<m>.csv` (`m ≥ 1`).
- **producer:** rule 3.03 `climate_stress_parameters`
  (`blueearth_cst/experiment/prepare_cst_parameters.py`).
- **consumer:** rule 3.07 `generate_climate_stress_test` (weathergenr
  `impose_climate_change.R`), passed in as `st_csv`.
- **shape:** a CSV with **header exactly** `month,temp_mean,precip_mean,precip_variance`
  and **12 rows**, `month ∈ 1..12`.
- **semantics:** `temp_mean` additive (°C); `precip_mean` / `precip_variance`
  multiplicative factors (fixture example row values `0.0, 0.7, 1.0`).
- **naming pattern:** one file per perturbation `m = 1..ST_NUM`; `cst_0` is
  **reserved** (no file — the unperturbed baseline, naming.md §4).
- **temp() lifecycle:** not `temp()`.
- **pinned surface:** the exact header, the 12-row `month` domain, the additive-
  vs-multiplicative column semantics, the one-file-per-`m` naming with `cst_0`
  reserved.
- **deliberately unpinned:** —
- **validator:** `validate_wg2`.

## WG-3 — weathergenr config surface

- **path pattern:** `<exp>/weathergen_config.yml` and
  `<exp>/realization_<n>/weathergen_config_rlz_<n>_cst_<m>.yml`.
- **producer:** rules 3.04 / 3.05 `prepare_weagen_config[_st]`
  (`blueearth_cst/experiment/prepare_weagen_config.py`).
- **consumer:** rules 3.06 / 3.07 (R side).
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

- **path pattern:** `<exp>/realization_<n>/rlz_<n>_cst_0.nc` (baseline) and
  `<exp>/realization_<n>/rlz_<n>_cst_<m>.nc` (`m ≥ 1`, perturbed).
- **producer:** rule 3.06 (cst_0) / rule 3.07 (cst_m).
- **consumer:** rule 3.08 `climate_data_catalog` + rule 3.09
  `downscale_climate_realization`.
- **shape:** the **generator OUTPUT contract** — a raster netCDF the hydromt
  catalog reads: `(time, lat, lon)` daily grid with **at least `precip`, `temp`**
  (+ `pet` if present) on an EPSG:4326 grid, `crs=4326` / `category=meteo`
  metadata (so `raster_xarray` + `harmonise_dims` load it — WG-5).
- **naming pattern:** `rlz_<n>_cst_<m>.nc` — a **DAG-globbed pattern**
  (rule 3.08 `expand`, `Snakefile_climate_experiment:318-319`; rule 3.09
  wildcards).
- **temp() lifecycle:** **`temp()`** (both cst_0 and cst_m). Deleted after
  consumers finish — **absent on the completed fixture**.
- **pinned surface:** the `(time, lat, lon)` raster shape, the minimal
  `{precip, temp}` variable set, the EPSG:4326 grid + `crs`/`category` metadata,
  the DAG-globbed naming pattern.
- **deliberately unpinned:** exact variable superset, internal attrs.
- **validator:** `validate_wg4` — **skip-until-captured on disk** (temp()
  content absent by default); logic proven every suite by a synthetic pass/fail
  pair. See the `--notemp` capture procedure below.

## WG-5 — hydromt climate data catalog (side channel)

- **path pattern:** `<exp>/data_catalog_climate_experiment.yml` (rule-3.08 side
  channel).
- **producer:** rule 3.08 `climate_data_catalog`
  (`blueearth_cst/climate_analysis/prepare_climate_data_catalog.py`).
- **consumer:** rule 3.09 `downscale_climate_realization` (as the `-d` catalog).
- **shape (pinned-as-reliance — hydromt data-catalog schema, OUR emitted
  subset):** one entry per `rlz_<n>_cst_<m>` (**including `cst_0`**), each
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
  `C:\Users\...\rlz_1_cst_1.nc`) emitted by `prepare_climate_data_catalog.py`.
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

- **path pattern:** `<exp>/realization_<n>/inmaps_rlz_<n>_cst_<m>.nc`.
- **producer:** rule 3.09 `downscale_climate_realization`
  (`blueearth_cst/experiment/downscale_climate_forcing.py`).
- **consumer:** rule 3.10 `run_wflow`.
- **shape:** **Wflow forcing on the MODEL grid** — the wf3 twin of
  `inmaps_historical.nc`; the same contract as HM-2 (see the hydrological-model
  seam doc, HM-2): `(time, lat, lon)` `float32` `precip` / `pet` / `temp` on the
  staticmaps grid, `spatial_ref` EPSG:4326 + `GeoTransform`, daily. This is the
  wflow-seam forcing input; **pinned once in HM-2, cross-referenced here.**
- **naming pattern:** `inmaps_rlz_<n>_cst_<m>.nc`.
- **temp() lifecycle:** **`temp()`** — deleted after rule 3.10 finishes,
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

- `experiments/<exp>/{sim_dates.csv, resampled_dates.csv}` —
  weathergenr-internal run diagnostics. Verified: neither name appears as a
  produced or consumed path in any Snakefile, Python module, or R script.
- `climate_historical/wf1_raw/extract_historical.nc` (rule 1.10
  `extract_climate_grid_wf1`) — shares WG-1's extraction schema but feeds the
  wf1 model-parity **plots**, not either substitution seam.

The completeness audit (both rule graphs walked) otherwise **confirms**
WG-1..WG-6 cover every interchange handoff at this seam; pipeline-internal
intermediates (build configs, guard/sequencing sentinels, log/benchmark gathers)
are correctly out.

---

## Bounded-substitution walkthrough — replacing weathergenr

A drop-in generator (design §5.6) must:

- **Consume** WG-1 (`extract_historical.nc`, the 7-var K grid) and WG-2 (the
  `cst_<m>.csv` perturbation grid) — or provide its own reader for them.
- **Produce** WG-4 netCDFs at the DAG-globbed paths
  `realization_<n>/rlz_<n>_cst_<m>.nc` (incl. `cst_0`), each a `(time, lat, lon)`
  EPSG:4326 raster with ≥ `precip`, `temp` and `crs=4326` / `category=meteo`, so
  the hydromt catalog (WG-5) loads it via `raster_xarray` + `harmonise_dims`.
- **Repo files it replaces:** rules 3.04–3.07 `shell:` / `script:` targets in
  `Snakefile_climate_experiment` (the two `Rscript --vanilla` bodies pointing at
  `weathergen/*.R`, plus the two config-prep scripts if the WG-3 config surface
  changes).
- **Files it must NOT change (the pinned boundaries):** rule 3.02 (WG-1
  producer), rule 3.08 (WG-5 catalog), rule 3.09 (WG-6 downscale).
- **Contracts it must satisfy:** WG-1 / WG-2 (in), WG-4 shape + naming (out),
  and — if it emits its own catalog — WG-5 **including the catalog↔grid
  invariant** (an entry per realization × cst incl. `cst_0`). Acceptance check:
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
| `validate_wg1` | WG-1 | `climate_historical/<key>/extract_historical.nc` | **yes** (persists); chirps facts **not fixture-verified (no chirps fixture)** |
| `validate_wg2` | WG-2 | `<exp>/stress_test/cst_<m>.csv` | **yes** (persists) |
| `validate_wg3` | WG-3 | `<exp>/weathergen_config.yml`, `<exp>/realization_<n>/weathergen_config_rlz_<n>_cst_<m>.yml` | **yes** (persists) |
| `validate_wg4` | WG-4 | `<exp>/realization_<n>/rlz_<n>_cst_<m>.nc` | **no** — `temp()` content absent; skip-until-captured on disk, synthetic-proven every suite |
| `validate_wg5` | WG-5 | `<exp>/data_catalog_climate_experiment.yml` | **yes** (catalog persists) |
| `validate_wg5_catalog_grid` (relational) | WG-5 entry-key grid vs intended `rlz × cst` (incl. `cst_0`) | `<exp>/data_catalog_climate_experiment.yml` + the run's config snapshot | **yes** (all inputs persist) |
| `validate_wg6` | WG-6 | `<exp>/realization_<n>/inmaps_rlz_<n>_cst_<m>.nc` | **no** — `temp()` content absent; skip-until-captured on disk, synthetic-proven every suite |

`validate_wg5_catalog_grid(catalog_cfg, rlz_num, st_num) -> list[str]` checks the
WG-5 entry-key set against the **intended** grid: expected keys exactly
`{rlz_<n>_cst_<m> : n ∈ 1..rlz_num, m ∈ 0..st_num}` (**cst_0 included** — rule
3.08 consumes both the cst_0 list and the perturbed `expand` grid,
`Snakefile_climate_experiment:318-319`). Missing and unexpected keys are each
reported. The intended grid is derived from the run's *recorded* P3-1 config
snapshot (`<exp>/config/snake_config_climate_experiment.yml`) via the same
`stress_test_grid` helper the Snakefile uses (`shared/snake_utils.py:336`), so
the check is self-consistent with the tree even if the tracked test config later
drifts. A dropped or extra catalog entry is invisible to per-artifact
`validate_wg5` (each remaining entry is well-formed) but breaks the
realization × cst fan-out rule 3.09 depends on.

### `--notemp` capture procedure (temp() on-disk validators)

The `temp()`-content validators `validate_wg4` (WG-4) and `validate_wg6` (WG-6)
have **no on-disk integration check on the default fixture**: both artifacts are
wrapped in Snakemake `temp()` and deleted after their consumers finish, so no
`rlz_<n>_cst_<m>.nc` / `inmaps_rlz_<n>_cst_<m>.nc` survive a completed run. Their
Layer-2 integration cases (`test_wg4_integration`, `test_wg6_integration`) carry
**both** the `_FIXTURE_ABSENT` skipif and a runtime
`pytest.skip("temp() artifact absent; capture via --notemp")` guarding on the
NC's presence. Their logic is proven on **every** checkout by their Layer-1
synthetic pass/fail pairs regardless.

**This milestone does NOT run the capture** — passing `--notemp` and letting the
artifacts persist would modify the untracked `examples/test_local` fixture, which
is out of a contracts-only milestone. The procedure below is the one-command lift
a **future run** performs when full on-disk coverage is wanted (design OQ-4).

**Capture sketch** (run from the repo root inside `pixi shell`, after the wf1
model exists — wf3 needs `hydrology_model/` artifacts):

```bash
snakemake all -c 3 -s Snakefile_climate_experiment \
  --configfile config/workflows/snake_config_model_test.yml --notemp
```

`--notemp` tells Snakemake **not** to delete `temp()`-flagged outputs after their
consuming jobs complete, so the run leaves the intermediate netCDFs on disk.

**Paths that then appear** under `examples/test_local` (the paths the
skip-guards test for):

| validator | artifact captured | fixture path (`<exp>` = `experiments/experiment`) |
|---|---|---|
| `validate_wg4` | WG-4 generator output NC | `<exp>/realization_<n>/rlz_<n>_cst_<m>.nc` |
| `validate_wg6` | WG-6 downscaled forcing NC | `<exp>/realization_<n>/inmaps_rlz_<n>_cst_<m>.nc` |

(HM-6b's `outstates_rlz_<n>_cst_<m>.nc` is captured by the same run — documented
in the hydrological-model seam doc.)

**Which cases un-skip:** with these artifacts present, `test_wg4_integration` and
`test_wg6_integration` here (plus `test_hm6b_integration` in the other seam doc)
stop hitting their `pytest.skip` and run their on-disk assertion — the **three**
temp validators' *on-disk* integration checks flip from skip-until-captured to
green. No test code or validator changes; the guards resolve to the real-artifact
path automatically once the files exist. Re-running **without** `--notemp` (or a
`snakemake --delete-temp-output`) restores the default temp-deleted fixture state.
