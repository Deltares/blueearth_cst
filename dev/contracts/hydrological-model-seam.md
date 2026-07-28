# Contract: hydrological-model seam (HM-1 .. HM-7)

> **Genre:** dev-facing interchange contract. **Audience:** a future *swapper* —
> someone replacing Wflow-SBM with an alternative hydrological engine, or the R6
> model-flexibility work — read end-to-end. Not an end-user doc (hence `dev/`,
> not `docs/`; precedent `dev/workflows/climate_experiment.md`).
> **Source of record:** `dev/p32b/interchange-contracts-design.md` (ACCEPTED
> 2026-07-24, §5.3 / §5.4 / §5.6 / §5.5). Every load-bearing fact below cites a
> Snakefile line, a script line, or an observed fixture artifact; do not add a
> contract fact that is not so grounded.

## Scope and method

The **hydrological-model seam** spans `Snakefile_model_creation` (wf1 build) and
`Snakefile_climate_experiment` (wf3 run) — the point where the hydrological
engine could be swapped without re-architecting the pipeline. Wflow-SBM (built by
hydromt) is the current occupant, but **this contract is model-agnostic**: it
pins what the pipeline hands *in* (forcing + static grid + run config) and
expects *out* (discharge CSVs → response surface), not Wflow's physics.

**Grounded in** the fixture tree `examples/test_local` (era5 branch,
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

- **path pattern:** `hydrology_model/staticmaps.nc`.
- **producer:** rule 1.03 `create_model` (hydromt build).
- **consumers:** wf1 rules 1.04 / 1.05 / 1.10 / 1.11; wf3 rule 3.09
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

- **path pattern:** `hydrology_model/forcing/inmaps_historical.nc` (wf1
  forcing); wf3 twin `<exp>/realization_<n>/inmaps_rlz_<n>_cst_<m>.nc` (= WG-6 on
  the weather-generator seam).
- **producer → consumer:** rule 1.08 `add_forcing` (hydromt update) → rule 1.09
  `run_wflow`; wf3 rule 3.09 → rule 3.10.
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

- **path pattern:** `hydrology_model/staticgeoms/*` (`region.geojson`,
  `basins.geojson`, `outlets.geojson`, `rivers.geojson`, `outlet_index.csv`, …).
- **producer:** rule 1.03 side-effect + rules 1.05 / 1.06.
- **consumers:** wf1 plot rules; wf3 rule 3.02 (`region.geojson` via
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

- **path pattern:** `hydrology_model/wflow_sbm.toml` (base) and per-cst
  `<exp>/model_runs/wflow_sbm_rlz_<n>_cst_<m>.toml`.
- **producer:** tracked template / rule 3.09 rewrite.
- **consumer:** rule 1.09 / rule 3.10 `run_wflow` (`Wflow.run()`).
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

- **path pattern:** wf1 `hydrology_model/run_default/output.csv`; wf3
  `<exp>/model_runs/output_rlz_<n>_cst_<m>.csv`.
- **producer:** rule 1.09 / rule 3.10 `run_wflow`.
- **consumers:** wf1 rule 1.11 plots; wf3 rule 3.11 `export_wflow_results`.
- **pinned surface — column identity is config-driven, NOT a literal list:** a
  `time` index (ISO-8601, daily) + **one column per `[output.csv].column`
  entry**, named `<header>_<mapid>`. Fixture: `time,Q_130000086` (one gauge).
- **the single degree of freedom:** the gauge-column set flows **TOML
  `[output.csv].column` → `output_rlz` → Qstats** as one degree of freedom — the
  key bounded-substitution invariant, checked end-to-end by
  `validate_hm_gauge_column_identity` (below).
- **consumer-prefix reliance (grounded, `export_wflow_results.py:61-62`):** rule
  3.11 selects gauge columns by a **hard-coded `Q_` prefix**
  (`Q_vars = [x for x in sim.columns if x.startswith("Q_")]`, line 61) and
  basin-average columns by a `basavg` substring (line 62) — so the TOML `header`
  values are load-bearing **beyond mere identity**.
- **temp() lifecycle:** not `temp()` — persists (both wf1 `output.csv` and wf3
  `output_rlz_*` CSVs).
- **deliberately unpinned:** numeric discharge values (not a contract; they
  change per run).
- **validator:** `validate_hm5` (per-artifact column-identity); cross-file
  identity by the relational `validate_hm_gauge_column_identity`.

## HM-6a — wf1 warm state (persisted, no validator)

- **path pattern:** `hydrology_model/run_default/outstate/outstates.nc`.
- **producer → consumer:** rule 1.09 `run_wflow` → **(nothing in-repo)**.
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

- **path pattern:** `<exp>/model_runs/outstates_rlz_<n>_cst_<m>.nc`.
- **producer → consumer:** rule 3.10 `run_wflow` → **(nothing in-repo)**.
- **THIN — "named output sink, unconsumed" (corrects the intake's chaining
  hint).** Verified: the per-cst TOML keeps `cold_start__flag = true` and
  declares **no `instates` input** on rule 3.10; wf3 fans out in parallel over
  `(rlz, cst)` with no cross-cst edge — **no warm-state chaining invariant** our
  DAG relies on (design §5.3 warm-state finding).
- **contract surface:** the file is a declared wflow state **output** whose name
  (`outstates_<climate_name>.nc`) and flat location (wf3 `dir_output="."`, HM-4)
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

## HM-7 — response-surface reduction (Qstats / basin)

- **path pattern:** `<exp>/model_results/Qstats.csv`, `<exp>/model_results/basin.csv`.
- **producer:** rule 3.11 `export_wflow_results`.
- **consumer:** CST-API / GUI (terminal in-repo).
- **pinned surface:** `Qstats.csv` header `statistic,tavg,prcp,<gauge-cols>`
  where `<gauge-cols>` = HM-5's `<header>_<mapid>` set (fixture `Q_130000086`),
  ordered per `export_wflow_results.py:66-67`; rows keyed by `statistic` × the
  `(tavg, prcp)` perturbation grid. `basin.csv` header `tavg,prcp` (the
  perturbation-axis index). These are the **response-surface hand-off** to the
  platform.
- **temp() lifecycle:** not `temp()` (`rule all`, manifested).
- **deliberately unpinned:** the `RT_*.csv` response tables (non-manifest side
  products).
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
  gauge-less Qstats**;
- a **mismatch in a later csv** `KeyError`s deep in the reduction.

Exactly the failure no per-artifact validator can see.

**The validator checks:**

1. every non-`time` `output_rlz_df` column traces to a declared
   `[output.csv].column` entry in `toml_cfg` (map-typed entries →
   `<header>_<id>` pattern; entries without `map` → exact `header`), and every
   declared entry is represented;
2. the map-typed gauge columns carry the `Q_` prefix rule 3.11 hard-codes;
3. `qstats_df`'s gauge columns (header minus `statistic,tavg,prcp`, ordered per
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
`Qstats.csv` — rlz {1,2} × cst {0..6}), so this relational check is in the
**continuously-verified** class. The test parametrizes over all 12 fixture
`(toml, output_rlz)` pairs against the one `Qstats.csv`.

---

## Considered and corrected — the warm-state finding (design §5.3, C4)

The intake's seam inventory hinted at "per-cst run chaining" via
`instates`/`outstates`. The fixture + TOML diff (`wflow_sbm_rlz_1_cst_1.toml` vs
base `wflow_sbm.toml`) shows `cold_start__flag = true` in **both**, rule 3.10
declares only a `forcing_path` + `toml_path` input (no `instates`), and the
`path_input` the rewrite sets (`downscale_climate_forcing.py:79`) points at a
**non-existent** `hydrology_model/instate/instates.nc` that cold-start never
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
  identity** the reduction (rule 3.11) keys on — the single degree of freedom
  that flows HM-4 `[output.csv].column` → HM-5 → HM-7, and the `Q_` header prefix
  the reduction hard-codes (`export_wflow_results.py:61`). A swap that renames the
  gauge column silently breaks HM-7 — exactly the failure the relational
  `validate_hm_gauge_column_identity` exists to catch.
- **Repo files it replaces:** rule 1.03 build (`hydromt build`), rule 1.08 / 3.09
  forcing prep, rules 1.09 / 3.10 `run_wflow` `shell:` bodies
  (`julia … Wflow.run()`), and `downscale_climate_forcing.py`'s TOML-rewrite
  (HM-4 fields) if the run-config format changes.
- **Files it must NOT change:** rule 3.11 `export_wflow_results` (the reduction)
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
| `validate_hm1` | HM-1 | `hydrology_model/staticmaps.nc` | **yes** (persists) |
| `validate_hm2` | HM-2 (+ WG-6 twin) | `hydrology_model/forcing/inmaps_historical.nc`; wf3 twin `inmaps_rlz_<n>_cst_<m>.nc` | **yes** for wf1 `inmaps_historical.nc`; wf3 twin (WG-6) `temp()` → skip-until-captured |
| `validate_hm3` | HM-3 | `hydrology_model/staticgeoms/{region.geojson, outlets.geojson, outlet_index.csv}` | **yes** (persists) |
| `validate_hm4` | HM-4 | `hydrology_model/wflow_sbm.toml`; `<exp>/model_runs/wflow_sbm_rlz_<n>_cst_<m>.toml` | **yes** (both base + per-cst TOMLs persist) |
| `validate_hm5` | HM-5 | wf1 `run_default/output.csv`; wf3 `<exp>/model_runs/output_rlz_<n>_cst_<m>.csv` | **yes** (both persist; wf3 `output_rlz_*` NOT `temp()`) |
| `validate_hm_gauge_column_identity` (relational) | HM-4 → HM-5 → HM-7 gauge-column identity | per-cst TOMLs + `output_rlz_*` CSVs + `Qstats.csv` | **yes** (all inputs persist) |
| *(HM-6a)* | HM-6a | `hydrology_model/run_default/outstate/outstates.nc` | **no validator** — existence pinned transitively via HM-4's `[state].path_output` |
| `validate_hm6b` | HM-6b | `<exp>/model_runs/outstates_rlz_<n>_cst_<m>.nc` | **no** — `temp()` content absent; skip-until-captured on disk, synthetic-proven every suite |
| `validate_hm7` | HM-7 | `<exp>/model_results/{Qstats.csv, basin.csv}` | **yes** (persists; `rule all`, manifested) |

### `--notemp` capture procedure (temp() on-disk validators)

The `temp()`-content validator `validate_hm6b` (HM-6b, the wf3 warm state) has
**no on-disk integration check on the default fixture**: `outstates_rlz_<n>_cst_<m>.nc`
is wrapped in Snakemake `temp()` and deleted after rule 3.10 finishes, so it does
not survive a completed run. Its Layer-2 integration case
(`test_hm6b_integration`) carries **both** the `_FIXTURE_ABSENT` skipif and a
runtime `pytest.skip("temp() artifact absent; capture via --notemp")` guarding on
the NC's presence. Its logic is proven on **every** checkout by its Layer-1
synthetic pass/fail pair regardless.

**This milestone does NOT run the capture** — passing `--notemp` and letting the
artifact persist would modify the untracked `examples/test_local` fixture, out of
a contracts-only milestone. The procedure below is the one-command lift a
**future run** performs when full on-disk coverage is wanted (design OQ-4).

**Capture sketch** (run from the repo root inside `pixi shell`, after the wf1
model exists — wf3 needs `hydrology_model/` artifacts):

```bash
snakemake all -c 3 -s Snakefile_climate_experiment \
  --configfile config/workflows/snake_config_model_test.yml --notemp
```

`--notemp` tells Snakemake **not** to delete `temp()`-flagged outputs after their
consuming jobs complete, so the run leaves the intermediate netCDFs on disk.

**Paths that then appear** under `examples/test_local` (the path the skip-guard
tests for):

| validator | artifact captured | fixture path (`<exp>` = `experiments/experiment`) |
|---|---|---|
| `validate_hm6b` | HM-6b wf3 warm state NC | `<exp>/model_runs/outstates_rlz_<n>_cst_<m>.nc` |

(The same run also captures WG-4 `rlz_<n>_cst_<m>.nc` and WG-6
`inmaps_rlz_<n>_cst_<m>.nc` — documented in the weather-generator seam doc.)

**Which cases un-skip:** with these artifacts present, `test_hm6b_integration`
here (plus `test_wg4_integration` and `test_wg6_integration` on the other seam)
stop hitting their `pytest.skip` and run their on-disk assertion — the **three**
temp validators' *on-disk* integration checks flip from skip-until-captured to
green. No test code or validator changes; the guards resolve to the real-artifact
path automatically once the files exist. Re-running **without** `--notemp` (or a
`snakemake --delete-temp-output`) restores the default temp-deleted fixture state.
