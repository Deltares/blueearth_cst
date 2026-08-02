# Equivalence gate log — ADR 0001 constant-parameter restoration (t260719a)

Step 3c of the ADR validation protocol. **Fail-closed, two-sided.** Each of the
13 mappable constants is RESTORED only if authoritative evidence for **both** the
0.x side (the short-name contract the value was authored under, git `6ebc46f`) and
the 1.x side (the Wflow.jl 1.x CSDMS-named parameter it will run under) is recorded
**and** a per-parameter comparison establishes that the number denotes the same
physical quantity — same units, sign convention, scale, and timestep basis — on
both sides. A log entry citing only the 1.x side is **incomplete and fails the
parameter closed** (ADR §Decision).

**Verdict summary: all 13 mappable constants PASS. Restored count = 13** (+ `KsatHorFrac`
retained; `InfiltCapSoil` dropped, `wflow_v1: None`). `MaxLeakage`'s 1.x default was
resolved to `0.0` → it is the 9th proven default-equal pin (not a decision addendum).
No parameter fails closed. Gate run 2026-07-21.

## Authoritative sources (both sides)

- **0.x side — Wflow.jl v0.8** (last 0.x line before the 1.0 CSDMS rename; the
  representative 0.x contract for the git `6ebc46f` template, whose short names
  match `naming.py`'s `wflow_v0` keys — `vertical.cfmax`, `vertical.g_tt`, …):
  - Vertical parameter table (units/sign/scale/default):
    `https://deltares.github.io/Wflow.jl/v0.8/model_docs/params_vertical/`
  - Glacier melt equation (resolves the `g_tt` role):
    `https://deltares.github.io/Wflow.jl/v0.8/model_docs/shared_concepts/`
- **1.x side — Wflow.jl stable (1.x)** SBM parameter reference (standard name,
  description, unit, default):
  `https://deltares.github.io/Wflow.jl/stable/model_docs/parameters_landhydrology_sbm.html`
- **Mapping of record — vendored `hydromt_wflow/naming.py`** (`.pixi/envs/default/
  Lib/site-packages/hydromt_wflow/naming.py`, lines 99–254): `wflow_v0` short name →
  `wflow_v1` CSDMS name for every entry, incl. the `["vertical.g_ttm",
  "vertical.g_tt"] → glacier_ice__melting_temperature_threshold` collapse and
  `InfiltCapSoil → None`.
- **Input flux timestep basis (both versions)** — bundled
  `docs/wflow-user-guide/02-required-files.md`: "The time unit of input flux
  parameters should be day (the model base time step size), these input parameters
  are converted to the user-defined model time step size during initialization."
  So `mm Δt⁻¹` fluxes are authored/read on a **per-day** basis on both sides.

## Key finding (why all 13 pass, and it is not manufactured)

Wflow.jl 1.x is the **same model with CSDMS-renamed parameters**, not a physics
change: the 1.x parameter reference is near-verbatim the v0.8 table (identical
descriptions, units, and — with two noted exceptions — identical engine defaults).
The gate exists to catch a rename that hides a semantic change; the one plausible
such case here (`G_TT`, whose one-line 0.x label reads "snowfall above glacier" but
whose 1.x name means "glacier **melt** threshold") was **actively investigated** and
resolved as preserved by the 0.x melt equation itself (below), not waved through.

## Two-sided evidence table

Units/sign/scale/timestep are **identical on both sides for all 13**. `0.x def` and
`1.x def` columns give the engine default (context only — the gate certifies
units/semantics, not value equality; restore values are the ADR reference choices).

| # | 0.x short name → 1.x CSDMS name | 0.x def (v0.8) | 1.x def (stable) | units (both) | restore val | verdict |
|---|---|---|---|---|---|---|
| 1 | `g_cfmax` → `glacier_ice__degree_day_coefficient` | 3.0 | 3.0 | mm °C⁻¹ day⁻¹ | 5.3 | **PASS** |
| 2 | `g_sifrac` → `glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction` | 0.001 | 0.001 | day⁻¹ | 0.002 | **PASS** |
| 3 | `g_ttm`/`g_tt` → `glacier_ice__melting_temperature_threshold` | 0.0 | 0.0 | °C | 1.3 | **PASS** |
| 4 | `infiltcappath` → `compacted_soil_surface_water__infiltration_capacity` | 10.0 | 10 | mm day⁻¹ | 5 | **PASS** |
| 5 | `maxleakage` → `soil_water_saturated_zone_bottom__max_leakage_volume_flux` | 0.0 | **0.0** | mm day⁻¹ | 0 | **PASS** |
| 6 | `cfmax` → `snowpack__degree_day_coefficient` | 3.75653 | 3.75 | mm °C⁻¹ day⁻¹ | 3.75653 | **PASS** |
| 7 | `tt` → `atmosphere_air__snowfall_temperature_threshold` | 0.0 | 0.0 | °C | 0 | **PASS** |
| 8 | `tti` → `atmosphere_air__snowfall_temperature_interval` | 1.0 | 1.0 | °C | 2 | **PASS** |
| 9 | `ttm` → `snowpack__melting_temperature_threshold` | 0.0 | 0.0 | °C | 0 | **PASS** |
| 10 | `water_holding_capacity` → `snowpack__liquid_water_holding_capacity` | 0.1 | 0.1 | — | 0.1 | **PASS** |
| 11 | `cf_soil` → `soil_surface_water__infiltration_reduction_parameter` | 0.038 | 0.038 | — | 0.038 | **PASS** |
| 12 | `e_r` → `vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio` | 0.1 | 0.1 | — | 0.11 | **PASS** |
| 13 | `rootdistpar` → `soil_wet_root__sigmoid_function_shape_parameter` | −500.0 | −500.0 | — | −500 | **PASS** |
| — | `infiltcapsoil` → **None (deprecated)** | 100.0 | — | mm day⁻¹ | — | **DROP** |

## Per-parameter findings (risky params first)

### 3. `G_TT` (1.3) → `glacier_ice__melting_temperature_threshold` — g_ttm/g_tt collapse [PASS]
- **0.x:** v0.8 table labels `G_TT` "threshold temperature for snowfall above
  glacier" (°C, default 0.0) — but the v0.8 glacier **melt equation** is
  `Q_m = g_cfmax·(T_a − g_tt)` for `T_a > g_tt`, i.e. `g_tt` **is** the melt-onset
  threshold in the degree-day glacier melt term (dual role; the doc explicitly
  states "`g_tt` can be taken as equal to the snow `tt`"). The git `6ebc46f`
  template sets **only `g_tt`** (there is no `G_TTM` entry in the block), so the
  many-to-one collapse `["vertical.g_ttm", "vertical.g_tt"] → glacier_ice__melting_
  temperature_threshold` in `naming.py` is benign **regardless** of whether a
  distinct `g_ttm` was ever populated: the single restored value 1.3 maps onto the
  single 1.x melt-threshold name with no ambiguity.
- **1.x:** `glacier_ice__melting_temperature_threshold` = "Threshold temperature
  for glacier melt" (°C, default 0.0).
- **Comparison:** same physical role (glacier melt-onset threshold in the
  degree-day melt equation), same unit (°C), same sign convention, no scale/timestep
  issue (temperature). `naming.py` collapsing both `vertical.g_ttm` and
  `vertical.g_tt` onto this one name is consistent with the 0.x model having a
  single melt threshold. **The apparent "snowfall vs melt" mismatch is a
  partial-label artifact, not a semantic drift.** **PASS.** (1.3 is a reference
  choice vs default 0.0.)

### 5. `MaxLeakage` (0) → `soil_water_saturated_zone_bottom__max_leakage_volume_flux` — 1.x default resolved [PASS]
- **0.x:** "maximum leakage from saturated zone" (mm Δt⁻¹, default 0.0, per-day input).
- **1.x:** "Maximum leakage from saturated zone" (mm Δt⁻¹, **default 0.0 mm day⁻¹**
  — resolved from the stable SBM parameter reference; the value the ADR footnote 2
  flagged as unverified in the bundled tables).
- **Comparison:** identical units/semantics/timestep basis; **1.x default = 0.0 =
  restore value** → per ADR footnote 2 this is the **9th proven default-equal pin**
  (drift-protection class), **not** a fifth reference choice and **no decision
  addendum required**. **PASS.**

### 1. `G_Cfmax` (5.3) → `glacier_ice__degree_day_coefficient` [PASS]
- 0.x "Degree-day factor for glacier" / 1.x "Degree-day factor for melt from
  glacier"; both mm °C⁻¹ Δt⁻¹, per-day, default 3.0. Same quantity. **PASS.**
  (5.3 is a reference choice vs default 3.0; v0.8 notes g_cfmax ≈ snow cfmax × 1–2.)

### 2. `G_SIfrac` (0.002) → `glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction` [PASS]
- Identical description both sides ("fraction of the snowpack on top of the glacier
  converted into ice"), Δt⁻¹ per-day, default 0.001. Same quantity. **PASS.**
  (0.002 is a reference choice; v0.8 notes typical range 0.001–0.006.)

### 4. `InfiltCapPath` (5) → `compacted_soil_surface_water__infiltration_capacity` [PASS]
- 0.x "infiltration capacity of the compacted areas" / 1.x "Infiltration capacity of
  compacted areas"; both mm Δt⁻¹ per-day, default 10.0. Same quantity. **PASS.**
  (5 is a reference choice vs default 10.)

### 6–13. Snow / soil / vegetation set [all PASS]
- `Cfmax`, `TT`, `TTI`, `TTM`, `WHC`, `cf_soil`, `EoverR`, `rootdistpar`: 0.x and 1.x
  descriptions match (near-verbatim), units identical (°C, mm °C⁻¹ day⁻¹, or
  dimensionless), sign preserved (incl. `rootdistpar` negative), and engine defaults
  identical **except** the two hydromt-template reference choices `TTI` (restore 2 vs
  engine default 1.0) and `EoverR` (restore 0.11 vs default 0.1) — value choices, not
  unit/semantic drift. All units/semantics preserved. **PASS.**

## Notes for downstream steps (config_restored, landing script, ADR reconciliation)

- **Restored set for `config_restored` = all 13 CSDMS names + `KsatHorFrac`**
  (`setup_constant_pars` block). No parameter is held out fail-closed. `InfiltCapSoil`
  stays dropped (`wflow_v1: None`).
- **Value provenance is unchanged** from the ADR table — the gate certifies
  units/semantics, not values; restore values remain the git `6ebc46f` template.
- **ADR reconciliation:** ADR footnote 2 offered three `MaxLeakage` branches; the
  gate took the **default-equal (9th pin)** branch — worth a one-line note in the ADR
  Revisions/Consequences so the "resolved (0–13), not promised" count is stated as
  **13**. The one nuance surfaced beyond the ADR: `TTI` and `EoverR` restore values
  differ from the **engine** default (they equal the **hydromt doc-template** value),
  so the "8 template-equal" set is template-equal, not engine-default-equal — the
  drift-protection framing still holds but the two are, strictly, low-stakes reference
  choices vs the engine default. Does not affect any gate verdict.
