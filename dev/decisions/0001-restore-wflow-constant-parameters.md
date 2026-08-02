Status: accepted
Date: 2026-07-21
Deciders: Ümit Taner
Consulted: M2b handoff (dev/milestones/phase-1/m02b/handoff.md §3); advisor review 2026-07-21;
           design-review-loop 2026-07-21 (internal risk/architecture/repo-fit panel;
           external GPT rounds 1 & 2 via codex; user arbitration at the round cap)
Supersedes: none
Revisions:
  - 2026-07-21: initial draft (task t260719a) [= design-v1].
  - 2026-07-21: revision r1 after full internal panel (18 findings; 2 blocking).
    Landing location corrected to the TOML scalar; validation protocol,
    justification, and manifest scope substantially rewritten. Core restore
    posture unchanged.
  - 2026-07-21: revision r2 after external review round 1 (5 findings; 1
    blocking). Restoration made conditional on a fail-closed per-parameter
    equivalence gate; the discharge comparator switched to a time-index-aligned
    numeric CSV comparator shared by reproducibility, materiality, and the
    durable baseline check; all discharge conclusions restricted to the
    aggregate restored set; the wf3 re-run branch operationalized; `MaxLeakage`
    reclassified as unverified (accounting corrected to 8 + 4 + 1). Core
    restore posture unchanged.
  - 2026-07-21: r3 arbitration revision after external review round 2 (3
    findings; 2 blocking), confined to ext2-1/ext2-2/ext2-3. ext2-1: equivalence
    gate now requires authoritative evidence for BOTH the 0.x and 1.x sides of
    every mapping, fails closed when either side is unavailable, and the restored
    count is stated as gate-resolved (0–13), not promised. ext2-2: steps 2/4 now
    pin two separately-materialized build configs (unchanged baseline snapshot
    vs restored candidate) so the comparison is never the edited file against
    itself. ext2-3: step 7's wf3 coherence review is now a defined, enumerated
    test over `Qstats.csv`/`basin.csv`. All other sections carried forward
    verbatim from r2.
  - 2026-07-21: **accepted at G2** (= design-v4) after the design-review-loop
    converged under arbitration. 26 findings total (2 internal blocking, 3
    external blocking across 2 rounds), all resolved. Consolidated review record:
    `dev/decisions/0001-restore-wflow-constant-parameters/review-record.md`. Implementation (t260719a)
    is a separate, build-heavy task.
  - 2026-07-21: **implementation — equivalence gate resolved (build-independent
    slice).** The step-3c gate ran against authoritative two-sided sources
    (Wflow.jl v0.8 `params_vertical` + glacier melt equation for the 0.x side;
    Wflow.jl stable `parameters_landhydrology_sbm` for the 1.x side; `naming.py`
    for the mapping). **All 13 mappable constants PASS** — units/sign/scale/
    timestep basis are preserved on both sides (Wflow.jl 1.x is the same model
    with CSDMS-renamed parameters). The two flagged priority cases resolved as
    preserved, not fail-closed: (a) the `g_ttm`/`g_tt`→`glacier_ice__melting_
    temperature_threshold` collapse is sound — the v0.8 glacier melt equation
    `Q_m = g_cfmax·(T_a − g_tt)` shows `g_tt` IS the melt-onset threshold (no
    separate `G_TTM` exists); (b) `MaxLeakage`'s 1.x default resolved to **0.0**,
    so it is the **9th proven default-equal pin** (drift-protection class), not a
    decision addendum. **Restored count = 13** (was "resolved (0–13), not
    promised"); + `KsatHorFrac` retained, `InfiltCapSoil` dropped. Full gate log:
    `dev/decisions/0001-restore-wflow-constant-parameters/equivalence-gate-log.md`. Build-independent artifacts
    landed: `config_baseline.yml`/`config_restored.yml`, the step-6 discharge
    comparator + `record --workflow` merge semantics in `check_baseline.py` (v2
    manifest; tested), and the step-3a/3b landing/precedence script
    `dev/scripts/verify_constant_pars.py`. Steps 2/4/4b/5/7 (two clean builds,
    reproducibility, materiality, re-record) remain build-heavy and pending.
  - 2026-07-21: **implementation COMPLETE — build-heavy execution (steps
    2/4/4b/5/7).** Three clean wf1 builds on the test basin (restored + baseline
    ×2), Julia 1.11.7 + Wflow. **Landing (3a/3b):** 14/14 constants land as
    `input.static.<name>.value` scalars, no shadowing (`snow__flag=true`,
    `glacier__flag=false`). **Reproducibility (4b):** two clean baseline builds are
    **bit-identical** on discharge (`max|ΔQ|/mean(Q_ref) = 0`) — attribution
    airtight. **Materiality (5): IMMATERIAL** — restored vs baseline **0/7670**
    timesteps exceed tolerance (`max|ΔQ|/mean = 1.66e-4`, peak local relative
    **0.71% < 1% RTOL**); nonzero-but-sub-tolerance, the expected signature of the
    active reference-choice params (InfiltCapPath 5/10, EoverR 0.11/0.1, TTI 2/1)
    with the glacier set inert. **Re-record (step 7, immaterial branch):** promoted
    `config_restored` → `config/wflow_build_model.yml` (surgical block swap), full
    `--forceall` rebuild of `examples/test_local`, `record --workflow
    model_creation` — 5 wf1 targets incl. the new discharge target + sidecar
    reference series (`dev/baseline/discharge_ref/`); wf2/wf3 manifest rows
    preserved byte-identical (verified per-key). Full evidence:
    `dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md`.

# ADR 0001 — Reconcile the dropped Wflow constant parameters via evidence-gated CSDMS restoration

### Context

The upstream Deltares model-build template set 15 spatially-uniform constant
parameters through hydromt_wflow's `setup_constant_pars` (snow, glacier, soil,
and vegetation physics). The M2b library upgrade (hydromt_wflow 0.x → 1.x)
broke this: 1.x rejects the old short names (`Cfmax`, `TT`, …) and requires
CSDMS Standard Names. Under M2b's declared "intentional drift, re-baseline
aggressively" policy, the block was reduced to the single parameter Wflow.jl
1.x *errors* without — `KsatHorFrac` (=100, now under its CSDMS name) — and the
other 14 were dropped, so the model now runs on **Wflow.jl 1.x's internal
defaults** for all of them. That is the current baseline, preserved by
construction through R3/R4/R5.

**Where these constants actually land (verified on the built model).**
`setup_constant_pars` does **not** write a map into `staticmaps.nc`. For each
recognized name it calls `self.config.set("input.static.{name}.value", value)`
— a **scalar TOML entry** in `wflow_sbm.toml`
(`hydromt_wflow/wflow_base.py:1461`). This is confirmed on the already-built
model: the one retained constant is present in
`examples/test_local/hydrology_model/wflow_sbm.toml` as
`[input.static.subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio] value = 100`
(lines 82–83), and it is **absent** from the built `staticmaps.nc`'s 50
variables. The plugin docstring ("Adds model layer … constant parameter map")
contradicts the code and misled the v1 draft. Every landing assertion,
attribution, fingerprint, and manifest target in this ADR therefore targets the
**TOML scalar**, not staticmaps.nc. (This resolves the v1 "staticmaps.nc and/or
the TOML" hedge.)

Two forces make finishing this non-trivial and worth a record.

**First, it is a scientific choice, not a mechanical rename** — but a narrower
choice than v1 claimed. The values are, for the majority of the set, the tool's
own documented default template, not Deltares-tuned calibration. The bundled
hydromt_wflow 1.x user guide (`docs/hydromt-wflow/user-guide.md:300–314`) ships
a canonical `setup_constant_pars` example whose values match this ADR's set
**exactly** for eight non-glacier snow/soil/veg parameters (`Cfmax` 3.75653,
`TT` 0, `TTI` 2, `TTM` 0, `WHC` 0.1, `cf_soil` 0.038, `EoverR` 0.11,
`rootdistpar` -500). They **differ** on exactly four parameters — the glacier
set plus `InfiltCapPath`:

| param | this ADR | 1.x doc template |
|---|---|---|
| `InfiltCapPath` (`compacted_soil_surface_water__infiltration_capacity`) | 5 | 10 |
| `G_Cfmax` (`glacier_ice__degree_day_coefficient`) | 5.3 | 3 |
| `G_SIfrac` (`glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction`) | 0.002 | 0.001 |
| `G_TT` (`glacier_ice__melting_temperature_threshold`) | 1.3 | 0 |

The thirteenth mappable parameter, **`MaxLeakage` (0), is in neither class**:
it does not appear in the doc-template example at all, and the bundled Wflow
1.x docs carry only its name mapping
(`docs/wflow-user-guide/02-required-files.md:73` —
`soil_water_saturated_zone_bottom__max_leakage_volume_flux = "MaxLeakage"`),
with no default or units. Its default-equality under Wflow.jl 1.x is therefore
**unverified** in this repo and it is handled fail-closed by the equivalence
gate (§ Decision), not assumed equal.

So the 13 mappable parameters split **8 + 4 + 1**: the restore-over-adopt
distinction collapses to **drift-protection** for the 8 template-equal params
(pin the documented default explicitly under a stable CSDMS name so it cannot
silently follow Wflow.jl default drift), a **genuine reference choice** for the
4 differing params, whose values come from git `6ebc46f` (the upstream template
block, not the hydromt doc default), and a **gate-conditional case** for
`MaxLeakage`. The justification is framed accordingly below; we do **not**
claim the whole set is "Deltares-tuned."

**Second, the failure modes are silent — but not the mode v1 named.** The v1
claim ("hydromt may accept a CSDMS name and never write it, a no-op that looks
like success") is **false**: `setup_constant_pars` **raises** `ValueError` on an
unrecognized name (`wflow_base.py:1449`) — a loud failure — and for a recognized
name deterministically writes `input.static.<name>.value` to the TOML. The real
silent modes are two:

- **Precedence / inactivity**: a constant `input.static.X.value` that Wflow.jl
  ignores at run time because a per-cell staticmaps layer exists for the same
  variable `X`, or because the owning module is inactive (`glacier__flag =
  false` in the built TOML, line 19). A value present in the TOML but not read
  by the engine passes a naive landing check yet does nothing. Guarded by the
  precedence check (step 3b) and logged as an open question there.
- **Semantic drift under the rename**: a name mapping is not evidence of
  parameterization equivalence. A 0.x numeric value could be written under a
  1.x name whose units, sign convention, or meaning differ, producing silently
  wrong hydrology on process-active basins while every landing check passes.
  Guarded by the fail-closed equivalence gate (§ Decision; step 3c) — a
  parameter is not restored until this is ruled out.

A third subtlety governs the discharge evidence: **several constants only bite
when their process is active on the basin.** The built test basin has
`snow__flag = true` and `glacier__flag = false` (`wflow_sbm.toml:15–19`). So the
snow module is **enabled** (the snow params `Cfmax`, `TT`, `TTI`, `TTM`, `WHC`
*can* affect discharge here), whereas the glacier module is **off** so its params
(`G_Cfmax`, `G_SIfrac`, `G_TT`) are **inert** on this basin regardless of value.
Note the flag proves snow is *enabled*, not that snow materially bites this
basin's discharge over the 2000–2020 forcing window — that informativeness is
asserted from the flag, not measured, so a null snow diff must be read as
non-informative (see step 5), not as proof the snow restorations are a no-op. A
null discharge move for the glacier subset on the reference basin therefore
proves nothing about the snow/glacier basins CST is meant to run globally. The
decision weighs cross-basin correctness, not only test-basin sensitivity, and
the validation protocol partitions the parameters so the measurement is not
over-read (§ Validation).

All 15 mappings are resolved against `hydromt_wflow/naming.py`. Restorability is
settled: 1 retained (`KsatHorFrac`), 1 deprecated (`InfiltCapSoil`, `wflow_v1:
None` → forced drop), 13 mappable.

### Decision

We will **restore the 13 mappable constants under their CSDMS Standard Names —
conditional, per parameter, on a fail-closed equivalence gate** — keep
`KsatHorFrac`, and **drop the deprecated `InfiltCapSoil`**. Restoration is the
default posture; **adopt-Wflow.jl-defaults is the null we reject**; measurement
**blesses and sizes** the move, it does not decide it. But the posture is not
unconditional: **no value ships until the gate establishes that it means the
same physical thing under its 1.x name.**

**Equivalence gate (fail-closed, two-sided).** Restoration of each parameter is
conditional on this gate, executed before the restored build (protocol
step 3c). Establishing that "units, sign convention, scale, and semantics are
preserved" is a **comparison**, and a comparison has two sides: the 0.x contract
the value was authored against and the 1.x contract it will run under. The gate
requires authoritative evidence for **both**:

- **Authoritative sources (both required):**
  - the **0.x side** — the applicable hydromt_wflow 0.x / Wflow (legacy)
    parameter documentation or code contract that defines what the short-name
    value (`Cfmax`, `TT`, …) meant: its units, sign convention, scale, and
    timestep basis at the time the git `6ebc46f` template authored it. Neither
    the template commit (which gives only the numbers) nor `naming.py` (which
    gives only name-to-name mappings) establishes this side; it must be sourced
    from the 0.x parameter reference/code directly.
  - the **1.x side** — the Wflow.jl 1.x parameter documentation (the engine's
    parameter reference) that defines the CSDMS-named parameter the value will
    be written under. The bundled `docs/wflow-user-guide/` mirrors the name
    tables but carries no defaults or units for these constants, so the gate
    consults the upstream 1.x parameter docs directly.
- **Required evidence, per parameter:** both the 0.x definition and the 1.x
  definition are recorded, and the **per-parameter comparison** is recorded — a
  finding that the number denotes the same physical quantity, on the same scale,
  with the same sign convention and timestep basis, on **both** sides. The two
  definitions and the comparison are written to the gate log (step 3c output) so
  a future reader need not re-litigate it. A gate log that cites only the 1.x
  side is **incomplete and fails the parameter closed** (it silently assumes the
  0.x side that ext1-1's silent-wrong-hydrology mode requires the gate to prove).
- **Fail-closed outcome (either side missing):** a parameter is **not restored**
  whenever **either** side's authoritative definition is missing, ambiguous, or
  negative, or the recorded comparison does not establish preservation. It is
  removed from the build set, logged, and returned to **design-level
  reconsideration** (a decision addendum to this ADR before any later
  restoration). A possibly-invalid value is never shipped on the strength of the
  approved posture alone, and never on one-sided (1.x-only) evidence.
- **Highest-risk cases, gated first:** the 4 differing params
  (`InfiltCapPath`, `G_Cfmax`, `G_SIfrac`, `G_TT`); the `g_ttm`/`g_tt`
  many-to-one collapse onto a single v1 name (footnote 1); and `MaxLeakage`,
  whose 1.x default is undocumented in the bundled tables (footnote 2). For
  `MaxLeakage` the gate must additionally resolve the 1.x default in order to
  classify it: default **equal to 0** → it becomes the ninth proven
  default-equal pin (drift-protection class); default **different** → restoring
  0 would be a *new* reference choice requiring a decision addendum, not silent
  inclusion; **unresolvable** → not restored.

The justification is split by what the value actually is:

- For the **8 template-equal parameters** the benefit is **drift-protection and
  self-documentation**: pinning the tool's own documented default explicitly
  under a stable CSDMS name so wf1 hydrology cannot silently follow a future
  Wflow.jl default change, and so the config states its physics rather than
  inheriting it invisibly. This is *not* a reference-continuity or
  Deltares-tuning claim.
- For the **4 differing parameters** (`InfiltCapPath`, `G_Cfmax`, `G_SIfrac`,
  `G_TT`) the benefit is a genuine **reference choice**: restoring the upstream
  template value (git `6ebc46f`) over both the Wflow.jl 1.x engine default and
  the hydromt doc default. This is where the real scientific decision is made
  and where a future reader should look first. The gate verifies the
  units/semantics so the choice is at least well-formed.
- **`MaxLeakage` (1 parameter) is gate-conditional**: its class (ninth
  default-equal pin, fifth reference choice needing an addendum, or not
  restored) is decided by the gate, as above. Until the gate resolves it, no
  default-equality or no-op claim is made for it.

The restoration is **evidence-gated in its blessing, not its posture**. The
*effect* of the restored set is measured, never asserted, and the measurement
is read only where it is informative (see the partition in § Validation) and
**only at the aggregate level** — all restored parameters change
simultaneously, so the discharge diff attributes nothing to any individual
parameter (step 5). Implementation builds the model twice into clean project
dirs from **two separately-pinned build configurations** — a restored-candidate
config vs an unchanged pre-restoration baseline config (never the edited config
against itself; see steps 2 and 4) — and:

1. asserts every restored value lands as `input.static.<csdms_name>.value` in
   `wflow_sbm.toml` (the contract the plugin actually writes), and that no
   competing staticmaps layer shadows it;
2. diffs discharge (`run_default/output.csv`, the `river_water__volume_flow_rate`
   series at `outlets`) restored-vs-reference with the step-6 comparator to
   size the **net** move of the restored set, read only for the process-active
   subset; and
3. records the result as a one-time, understood, **documented baseline move**.

Default-equality claims rest solely on documentation — the template match and
the equivalence gate — never on a null net discharge diff, which cannot
distinguish equal defaults from inactive processes, weak sensitivity, or
offsetting effects (step 5).

**Parameter reconciliation** (0.x values from git `6ebc46f`; CSDMS names from
`hydromt_wflow/naming.py`). The `activity class` column is the discharge-evidence
partition for the built test basin (`snow__flag=true`, `glacier__flag=false`);
`vs 1.x-template` flags the 4 params that differ from the hydromt doc default
and the 1 param the template does not cover. Every RESTORE row is conditional
on the equivalence gate.

| 0.x name | value | CSDMS Standard Name (`wflow_v1`) | posture | activity class (test basin) | vs 1.x-template |
|---|---|---|---|---|---|
| KsatHorFrac | 100 | subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio | RETAINED | process-active | equal |
| Cfmax | 3.75653 | snowpack__degree_day_coefficient | RESTORE | snow (active here) | equal |
| TT | 0 | atmosphere_air__snowfall_temperature_threshold | RESTORE | snow (active here) | equal |
| TTI | 2 | atmosphere_air__snowfall_temperature_interval | RESTORE | snow (active here) | equal |
| TTM | 0 | snowpack__melting_temperature_threshold | RESTORE | snow (active here) | equal |
| WHC | 0.1 | snowpack__liquid_water_holding_capacity | RESTORE | snow (active here) | equal |
| G_Cfmax | 5.3 | glacier_ice__degree_day_coefficient | RESTORE | glacier (INERT here) | **differs (3)** |
| G_SIfrac | 0.002 | glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction | RESTORE | glacier (INERT here) | **differs (0.001)** |
| G_TT | 1.3 | glacier_ice__melting_temperature_threshold¹ | RESTORE | glacier (INERT here) | **differs (0)** |
| cf_soil | 0.038 | soil_surface_water__infiltration_reduction_parameter | RESTORE | process-active | equal |
| EoverR | 0.11 | vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio | RESTORE | process-active | equal |
| InfiltCapPath | 5 | compacted_soil_surface_water__infiltration_capacity | RESTORE | process-active | **differs (10)** |
| MaxLeakage | 0 | soil_water_saturated_zone_bottom__max_leakage_volume_flux | RESTORE (gate-conditional)² | process-active | **absent — UNVERIFIED** |
| rootdistpar | -500 | soil_wet_root__sigmoid_function_shape_parameter | RESTORE | process-active | equal |
| InfiltCapSoil | 600 | None (deprecated in Wflow.jl 1.x) | DROP | — | — |

¹ `naming.py` collapses old `g_ttm` and `g_tt` onto one v1 name
(`glacier_ice__melting_temperature_threshold`); the template set only `G_TT`, so
restore `1.3`. This collapse is one of the two-old-names-onto-one-v1-name cases
and a priority case for the equivalence gate (step 3c).
² `MaxLeakage: 0` disables the saturated-zone bottom leakage flux. Its Wflow.jl
1.x default is **not verified in this repo**: the value is absent from the
hydromt doc template (`user-guide.md:300–314`), and the bundled Wflow docs map
only the name — `docs/wflow-user-guide/02-required-files.md:73` — with no
default or units. It is handled fail-closed by the equivalence gate: restored
only if **both** the 0.x contract and the 1.x parameter docs confirm its
units/semantics AND the 1.x default is resolved (equal → ninth default-equal
pin; different → decision addendum required as a fifth reference choice;
unresolved → not restored). No net-zero-on-discharge expectation is asserted — a
net measurement could not support one anyway (step 5).

### Validation protocol (implementation)

Partition first. The parameters split three ways for what evidence certifies
them:

- **process-active on the test basin** (`cf_soil`, `EoverR`, `InfiltCapPath`,
  `MaxLeakage` (gate-conditional), `rootdistpar`): the discharge diff is a
  **real test** of whether the restored set's net effect on this basin differs
  from the current-default configuration.
- **snow set, module enabled but bite unmeasured** (`Cfmax`, `TT`, `TTI`, `TTM`,
  `WHC`): `snow__flag=true`, so the discharge diff *can* be a real test — but only
  if snow actually contributes to discharge in the 2000–2020 forcing. A **null
  snow diff is non-informative**: it could mean the restored values equal the
  engine default, *or* that snow is negligible in this forcing window. Do **not**
  read a null snow diff as "tested → validated no-op"; it falls back to the
  landing assertion plus cross-basin reference-continuity, exactly like the
  glacier set. A **non-null** snow diff is a genuine test-basin move,
  attributable to the restored set as a whole.
- **glacier (inert here**, `glacier__flag=false`: `G_Cfmax`, `G_SIfrac`,
  `G_TT`): the discharge diff is **~0 by construction** regardless of value.
  These are certified by the landing assertion (value present in the TOML, no
  shadowing layer), the equivalence gate (they are 3 of the 4 params that
  differ from the doc template), and **cross-basin reference-continuity**. The
  measurement is **not** presented as their evidence.

**Aggregate-only reading (applies to every step below).** All restored
parameters change simultaneously between the two builds, so the discharge diff
identifies no individual parameter's contribution, cannot prove that any value
equals the engine default, and cannot detect cancellation. A net-null can arise
from inactive processes, equal defaults, weak sensitivity, **or offsetting
nonzero effects** — the protocol never converts a net-null into a per-parameter
claim. Per-parameter attribution would require one-at-a-time (or grouped)
controlled runs, which are **out of scope** for this task; if a per-parameter
question ever becomes consequential, that is a new, separately-scoped
experiment.

Steps:

1. **Resolve mappings** — done (table above); restorability settled.
   1b. **Pin the landing location (empirical, early).** Before writing any
   assertion, inspect the built `wflow_sbm.toml` for one representative restored
   parameter and confirm it lands as `input.static.<name>.value` (already
   verified for `KsatHorFrac`; re-confirm for one restored snow param after the
   first build). This pins the manifest/assertion target to the TOML and
   forecloses the staticmaps path.
2. **Restored build (from a separately-pinned restored-candidate config).** Run
   the step-3c equivalence gate **first** (it is a documentation analysis and
   needs no build). Before either build in this protocol, **pin two distinct,
   separately-materialized build configurations** so the later comparison can
   never degenerate into an identity comparison:
   - **`config_baseline`** — the *unchanged* pre-restoration config: the current
     in-repo `config/wflow_build_model.yml`, snapshotted (copied to a distinct
     path, e.g. `dev/decisions/0001-restore-wflow-constant-parameters/config_baseline.yml`) and pinned to the
     current git revision, recorded by that revision hash. This is the actual
     M2b baseline; it is **not edited** by this task.
   - **`config_restored`** — a *distinct* candidate config file (e.g.
     `dev/decisions/0001-restore-wflow-constant-parameters/config_restored.yml`), derived from the baseline
     snapshot, that adds the **gate-passing subset** of the 13 CSDMS entries +
     `KsatHorFrac` to the `setup_constant_pars` block. Any gate-failed or
     unverified parameter stays out of the block and follows the fail-closed
     branch (§ Decision).

   Build wf1 from `config_restored` into a *clean* project dir (the restored
   output dir). Record which config produced this model (config path + git rev)
   in the build provenance. In the same edit that materializes `config_restored`
   (never in the shared in-repo file that `config_baseline` snapshots),
   **rewrite the block's lines 29–33 comment** (which currently documents the
   *opposite* drop decision) to a one-line pointer to this ADR and the restore
   rationale, so the shipped config does not self-contradict and invite a
   re-drop. The eventual in-repo landing of `config_restored` (promoting it to
   `config/wflow_build_model.yml`) happens only after the protocol blesses the
   move; the comparison itself is always `config_restored` vs the pinned
   `config_baseline`, never an edited file against itself.
3. **Equivalence gate + landing assertion (form: one-time `dev/scripts/`
   verification script + recorded gate log).** `test_cli.py` is dry-run only
   and `test_model_creation.py` does not build, so the assertion has no home in
   the standing test suite. Implement 3a/3b as a one-time script under
   `dev/scripts/` run during implementation and recorded in the baseline-diffs
   note (not a permanent `--run-integration` test); 3c is a documentation
   analysis whose per-parameter log lands in the same note.
   - 3a. For each gate-passed restored parameter, assert
     `input.static.<csdms_name>.value` is present in `wflow_sbm.toml` and
     equals the reference value. (A recognized name is always written; an
     unrecognized name would already have raised at build time — so this
     asserts the value the plugin wrote, and catches a mapping typo that
     silently mapped to a different recognized name.)
   - 3b. **Precedence / inactivity check (guards the first silent mode).** For
     each restored parameter, confirm no competing per-cell staticmaps layer
     exists for the same CSDMS variable (which would shadow the scalar), and note
     whether the owning module is active (`glacier__flag=false` ⇒ the 3 glacier
     scalars are inert at run time on this basin). **Open question for
     implementation:** whether Wflow.jl 1.x actually reads an inactive-module
     `input.static.X.value`; verify against engine behavior, do not assume.
   - 3c. **Equivalence gate (fail-closed, two-sided precondition; executed
     BEFORE the step-2 restored build).** `naming.py` (`WFLOW_NAMES`) is a pure
     name-to-name map with no units/scale/sign field, so it cannot certify that
     the 0.x numeric value is dimensionally valid under the 1.x name.
     Per parameter, establish and record **both** sides of the comparison:
     - **0.x side:** cite the applicable 0.x hydromt_wflow / Wflow (legacy)
       parameter documentation or code contract; record the short-name value's
       units, sign convention, scale, and timestep basis as authored under
       git `6ebc46f`.
     - **1.x side:** cite the **Wflow.jl 1.x parameter documentation** (the
       authoritative 1.x source); record the CSDMS-named parameter's units, sign
       convention, scale, and semantics.
     - **Comparison:** record the per-parameter finding that the number denotes
       the same physical quantity on the same scale/sign/timestep basis under
       the 1.x name as it did under the 0.x name. A log entry citing only the
       1.x side is incomplete and **fails that parameter closed**.

     Priority cases: the 4 differing params (`InfiltCapPath`, `G_Cfmax`,
     `G_SIfrac`, `G_TT`), the `g_ttm`/`g_tt` collapse (footnote 1), and
     `MaxLeakage` (footnote 2 — the gate must additionally resolve its 1.x
     default to classify it). **Fail-closed (either side):** any parameter for
     which **either** the 0.x definition or the 1.x definition is missing,
     ambiguous, or negative — or whose recorded comparison does not establish
     preservation — is NOT restored. It is removed from the step-2 build set,
     logged, and returned to design-level reconsideration via a decision
     addendum. Because every parameter is gate-conditional and evidence for the
     other twelve has not been pre-established, the protocol **cannot promise a
     count**: the restored count is **resolved at the gate and recorded (0–13),
     not promised**.
4. **Reference build (freshness guard; from the pinned baseline config).** Build
   from the pinned **`config_baseline`** snapshot (step 2) — the unchanged
   pre-restoration config at the recorded git revision — into a separate *clean*
   project dir (the reference output dir). No `ancient()`/timestamp-blessed
   reuse of `examples/test_local`; both builds are from scratch so neither is
   stale. The restored-vs-reference comparison in steps 4b/5 is therefore
   `config_restored`-output vs `config_baseline`-output — **two distinct configs,
   never the edited file against itself**. Record which config (path + git rev)
   produced this model in the build provenance, alongside the step-2 record, so
   the two output dirs are unambiguously attributable to their source configs.
   4b. **Clean-build discharge reproducibility check (attribution guard).** Build
   the **`config_baseline`** snapshot a second time into another clean dir and
   confirm the two discharge series **pass the step-6 comparator** (same
   implementation, same tolerances). This establishes that two clean builds of
   the *same* config are reproducible on discharge, so the later
   restored-vs-reference diff is attributable to the configuration change (the
   restored set as a whole) and not to build/solver nondeterminism.
   (Counter-evidence that this is plausible: the aggregated `Qstats.csv` already
   hashes cleanly in-env; the risk is specifically that raw daily `output.csv`
   at full float64 is far more LSB-sensitive.)
5. **Discharge materiality (informative subset only; aggregate only).** Run
   Wflow on the restored and reference builds; compare the discharge series
   (`run_default/output.csv`, the `Q` column = `river_water__volume_flow_rate`
   at `outlets`) with the **step-6 comparator**.
   - **Metric + threshold (concrete, falsifiable):** `material :=` the step-6
     comparator **fails** on restored-vs-reference — equivalently:
     `max_t |Q_restored(t) − Q_reference(t)| > ATOL` with
     `ATOL = 1e-3 × mean(Q_reference)`, **or** any timestep with
     `Q_reference(t) ≥ ATOL` moving by more than `RTOL = 1%` relative.
     Anything below both at every timestep is recorded as a solver-noise
     **no-op**. This is the **same code path** as steps 4b and 7, so the
     classification, the attribution guard, and the durable regression gate
     cannot disagree.
   - **immaterial** → the restored set is a documented aggregate no-op **for
     this basin's active processes** — nothing more. It does not show that any
     individual parameter equals the engine default (see the aggregate-only
     note above: inactive processes, equal defaults, weak sensitivity, and
     offsetting effects are indistinguishable in a net-null). Proceed to
     step 7, immaterial branch.
   - **material** → a sanctioned, documented baseline move of the restored set.
     Proceed to step 7, material branch (wf3 re-run and multi-workflow
     re-record).
   - **Certification boundary.** This diff certifies the **net effect of the
     restored set on the test basin only**. It certifies zero of the 3 glacier
     parameters' hydrological effect (inert here); for the snow set a null
     result certifies nothing (module enabled but bite unmeasured); and it does
     not certify the restorations on any *other* basin. All of those rest on
     the landing assertion, the equivalence gate, and cross-basin
     reference-continuity, not on this discharge sensitivity.
6. **Discharge comparator in `check_baseline.py` (scoped code work, not a
   manifest edit) — time-index-aligned numeric CSV comparator (selected).**
   `check_baseline.py` cannot express the tolerance-gated discharge judgment as
   it stands: `FINGERPRINTERS = {png, nc, csv, yaml}` and the `csv` comparator
   is `diff_hashed` — byte-exact normalized SHA256 with **no tolerance**; only
   `diff_nc` carries `--tolerance`. Raw daily `output.csv` is full float64
   (~17 sig-digits), maximally LSB-sensitive, so an exact SHA fails on any
   solver/env drift and a match cannot be attributed to the config change.
   v2's option (b) — a summary/rounded fingerprint (min/max/mean/std) — is
   **withdrawn**: summaries cannot enforce the per-timestep contract of step 5
   (a timing shift or compensating positive and negative deviations preserves
   the summaries while exceeding per-timestep tolerance), so the durable check
   could pass a regression the ADR itself classifies as material. **Selected:
   option (a), specified as follows.**
   - **Inputs:** two `output.csv` files (candidate vs reference). Parse each
     with the timestamp column as the index; extract the `Q` discharge column.
   - **Structural checks (any hit ⇒ structural FAIL, never a numeric pass):**
     duplicate timestamps in either file; unequal time-index sets (report the
     symmetric difference); NaN or non-numeric values in either aligned series.
   - **Numeric tolerance, per aligned timestep `t`,** with
     `ATOL := 1e-3 × mean(Q_ref)` (mean over the full aligned reference series)
     and `RTOL := 0.01`: `fail(t) := |ΔQ(t)| > ATOL` **OR**
     (`Q_ref(t) ≥ ATOL` AND `|ΔQ(t)| > RTOL × Q_ref(t)`).
   - **Zero/near-zero reference handling:** where `Q_ref(t) < ATOL` the
     relative clause is skipped (division-safe) and the absolute clause alone
     governs — near-dry timesteps cannot manufacture materiality through an
     exploding relative change. Note the relative clause is deliberately the
     **low-flow tightener**: for `Q_ref(t)` between `ATOL` and
     `ATOL/RTOL = 0.1 × mean(Q_ref)` it is stricter than the absolute clause
     (1% moves at low flow matter — CST's indicators include 7-day low flow);
     above that it never binds.
   - **Verdict + report:** pass iff no structural mismatch and no failing
     timestep. Report the max normalized absolute diff
     (`max|ΔQ|/mean(Q_ref)`), the max relative diff over the
     `Q_ref ≥ ATOL` subset, the failing-timestep count, and the first and worst
     offending timestamps — recorded in the baseline-diffs note.
   - **One implementation, three call sites:** step 4b (reproducibility — pass
     required), step 5 (materiality — fail ⇒ material), step 7 (durable
     baseline check — fail ⇒ regression). A single code path is what makes the
     three judgments incapable of disagreeing.
   - **Durable-check storage:** a per-timestep comparator needs the full
     reference series, not a summary, so `record` stores a reduced copy (time
     index + `Q` column) of `run_default/output.csv` under `dev/baseline/`
     (~7.7k daily rows for 2000–2020 — trivial size), and the manifest row for
     the discharge target references it; `check` runs the comparator
     current-vs-stored. This replaces v2's summary-fingerprint storage.
   - Either way this is a **`check_baseline.py` + FINGERPRINTERS/TARGETS code
     change**, named as such in the implementation scope. **No `toml`
     fingerprinter is added:** the TOML scalar is verified by the step-3
     landing assertion, not by the manifest, so `check_baseline` gains only the
     discharge target.
7. **Manifest extension + re-record (minimum scope, branched on step-5
   materiality).**
   - Add **only the discharge target** (`run_default/output.csv` via the step-6
     comparator and stored reference series) to `dev/baseline/manifest.json`.
     This is the minimum that makes *this* move visible. The general
     staticmaps/TOML manifest expansion is the separable **cross-cutting
     baseline-manifest-integrity followup** (`dev/followups.md` § baseline
     manifest integrity; tracked in t260719a per the pre-R6 tracker) and is
     **deferred and cross-referenced**, not decided here.
   - **`record` is all-or-nothing.** `cmd_record` calls
     `compute_manifest(args.project_dir)` with no workflow filter and refuses on
     any missing target; `--workflow` exists only on `check`. A wf1-only clean
     re-record therefore either needs a full three-workflow pipeline run into the
     recorded dir **or** a new `--workflow` on `record`. **Scope decision:** add
     `--workflow` to the `record` subcommand so a wf1-slice re-record is possible
     without dragging the CMIP6 fetch and the Julia wf3 run into this task.
     **This is NOT symmetric with `check --workflow`:** `cmd_record` currently
     overwrites the whole manifest (`targets = compute_manifest(project_dir)`,
     lines 241/247–253), so naively filtering `compute_manifest` would **delete
     the recorded wf2/wf3 fingerprints**. A filtered `record` must therefore
     (a) filter targets to the selected workflow, (b) scope the
     incomplete-manifest refusal to the selected workflow only, and (c) **merge**
     into the existing manifest (update only the selected rows, preserve the
     rest) rather than overwrite. Specify these merge semantics in the
     implementation brief; a plain overwrite ships a silent manifest-clobber.
   - **Branch: material (step 5).** A material wf1 discharge move propagates
     into wf3's model-dependent targets, so the re-baseline must actually be
     multi-workflow, not claimed as such:
     1. Run wf3 (`Snakefile_climate_experiment`) on the **restored** build,
        reusing existing wf2 artifacts where the DAG requires them (wf2's CMIP6
        change factors do not depend on the Wflow build and are untouched).
     2. **Review the wf3 diffs on `Qstats.csv` and `basin.csv` for coherence
        with the wf1 move — defined tests, not a subjective read.** First
        establish the **model-dependent field set**: the columns/indicators in
        these two files whose values are a function of the Wflow build (discharge
        statistics and derived hydrological indicators) as opposed to fields
        determined by the perturbation grid, run configuration, or metadata. Any
        column outside the model-dependent set is **expected invariant**: it must
        pass the step-6 comparator (or an exact match where it is categorical /
        an identifier) between the pre- and post-restoration wf3 runs; **any
        movement in an invariant field is an incoherence** (a re-record stopper —
        see item 4). For the model-dependent fields, apply per-indicator
        expectations, and only where scientifically defensible:
        - **`Qstats.csv` flow-magnitude indicators** that are monotonic
          functionals of the discharge series (e.g. mean discharge, and the
          high-flow / max-discharge statistics): expect the **same sign** as the
          wf1 outlet-discharge move (a wf1 discharge increase should not produce a
          decrease in mean/high-flow indicators), and a magnitude **bounded by**
          the wf1 relative discharge change scaled to the indicator — quantify as
          the indicator's relative change lying within `[0, k · Δ_wf1_rel]` with
          the same sign, where `Δ_wf1_rel` is the step-5 max relative discharge
          move and `k` a documented slack factor (default `k = 3`) absorbing the
          nonlinearity of the statistic; a move that reverses sign or exceeds
          `k · Δ_wf1_rel` is flagged for causal inspection, not auto-blessed.
        - **`Qstats.csv` low-flow indicators (e.g. 7-day low flow)** and any
          quantile/timing indicator that is **not** a monotonic functional of
          mean discharge: **no directional or proportional-magnitude expectation
          is scientifically defensible** (a mean-discharge increase can lower,
          raise, or leave a low-flow statistic unchanged depending on where in
          the hydrograph the restored set moves water). For these, do **not**
          assert a sign or bound; instead require **documented causal
          inspection** — trace the indicator move to the wf1 hydrograph change
          (which timesteps/seasons moved, whether the low-flow window is
          affected) and record the explanation in the baseline-diffs note. An
          unexplained move here is an incoherence (item 4).
        - **`basin.csv` indicators:** classify each column at implementation into
          (i) model-dependent flow/water-balance indicators — apply the
          monotonic-functional rule above where the indicator is a monotonic
          functional of discharge/water balance, else the causal-inspection rule;
          and (ii) invariant fields (basin identifiers, areas, static attributes,
          grid/perturbation metadata) — expected invariant, checked as above.
        The concrete column→class assignment for both files is enumerated in the
        implementation brief and recorded in the baseline-diffs note so the
        arbiter is reproducible; a column whose class is genuinely ambiguous is
        treated as causal-inspection (never silently blessed).
     3. Record **both** slices — wf1 and the affected wf3 targets — via
        `record --workflow` under the merge semantics above (or one full-run
        record). wf2 rows are preserved unchanged.
     4. **Failure handling:** if the wf3 diffs are incoherent with the wf1 move
        — a model-dependent indicator reverses sign or exceeds its bound without
        a documented causal explanation, an invariant field moves at all, or a
        no-expectation indicator moves without documented causal inspection —
        **stop — do not re-record**. Treat it as an unexplained regression:
        check for stale intermediates, weathergenr seed/state, and env drift; if
        it remains unexplained, return to the decision level rather than blessing
        the baseline.
   - **Branch: immaterial (step 5).** Record the wf1 slice only
     (`record --workflow` for model_creation); wf2/wf3 rows are preserved by
     the merge semantics. **Residual and its handling:** the standing wf3 CSV
     fingerprints are byte-exact, so even a sub-tolerance wf1 move could
     perturb them. If a later wf3 check fails after an immaterial wf1 move,
     re-run wf3, compare the moved targets; if the movement is consistent with
     the recorded sub-tolerance wf1 diff and otherwise explained, re-record the
     wf3 slice with a note; if not, stop and investigate as above.
   - **Docstring/scope note.** The new discharge target is a build
     intermediate, not a `rule all` target of `Snakefile_model_creation` (whose
     `rule all` lists only the 3 PNGs + config snapshot + `outlet_index.csv`).
     Update the `check_baseline.py` module docstring / add a comment on the new
     row recording that wf1 model artifacts are deliberately fingerprinted beyond
     `rule all` for parameter-preservation coverage, so a maintainer does not
     prune it as stray.
   - Document the discharge move (comparator report + material/immaterial
     verdict + the executed branch + the gate log) in `dev/tasks/t260719a` or a
     `baseline_diffs.md`.

### Consequences

*Positive*
- wf1 hydrology pins its snow/glacier/soil/vegetation constants explicitly under
  stable CSDMS names — the 8 template-equal params are protected against Wflow.jl
  default drift, and the 4 differing params carry an explicit, documented
  reference choice, rather than the config silently inheriting engine defaults.
- The units/semantics risk of the 0.x → CSDMS rename is a **decision-level
  precondition**, not an implementation footnote: the fail-closed equivalence
  gate means no restored value can ship with unverified physics, and a failed
  gate produces a recorded design question instead of a silently wrong basin.
- The baseline gains **discharge** coverage in `check_baseline` via a
  time-index-aligned numeric comparator whose single implementation drives
  reproducibility, materiality, and the durable check — the three judgments
  cannot disagree by construction.
- The genuine silent failure mode — a TOML scalar shadowed by a staticmaps layer
  or ignored by an inactive module — is checked **once** by the landing +
  precedence script at implementation time, and the accepted-but-unwritten mode
  is shown to be impossible (the plugin raises). This is a one-time verification,
  **not** a durable regression contract (see the residual gap below).

*Negative*
- A one-time, intentional baseline move whose size is measured, not assumed —
  at the **aggregate restored-set level only**; this protocol deliberately buys
  no per-parameter attribution (one-at-a-time runs are out of scope).
  R1/R2/R6's "preserve the baseline" rule does not apply — this task is
  explicitly allowed to move wf1's slice with a documented diff.
- **Blast radius beyond wf1, now operationalized.** `climate_experiment` (wf3)
  reuses the built Wflow model, so **if the discharge move is material on the
  test basin**, the move is a **multi-workflow** baseline move: step 7's
  material branch runs wf3 on the restored model, reviews `Qstats.csv` /
  `basin.csv` against the defined per-indicator coherence tests (model-dependent
  field set, monotonic-functional sign/magnitude bounds, invariant fields, and
  documented causal inspection for no-expectation indicators), and re-records
  both slices — with a stop-and-investigate rule if downstream outputs move
  incoherently. (Immaterial → wf1-only record, with a specified recovery path if
  byte-exact wf3 fingerprints are later perturbed by the sub-tolerance move.)
- The restored count is **resolved at the gate and recorded (0–13), not
  promised**: every parameter is gate-conditional (both the 0.x and 1.x sides of
  its mapping must be authoritatively established), so `MaxLeakage` — and any
  other parameter for which either side's evidence is missing, ambiguous, or
  negative — may exit the restored set fail-closed and return as a decision
  addendum. The protocol cannot commit to a count in advance; the final count is
  whatever the gate certifies, and it is recorded.
- Required code/data changes: the time-aligned discharge comparator, `record
  --workflow` with merge semantics, a stored reference discharge series under
  `dev/baseline/`, plus the one-time landing/precedence script and the gate log
  — more than a manifest edit.
- Larger build-config surface (up to 13 explicit parameters) to maintain against
  future hydromt_wflow renames.
- **Residual durable-coverage gap (open item).** Because this task de-scopes the
  manifest to discharge only and makes the landing/precedence check a
  one-time script, the restored constant *values* in the TOML have **no
  durable regression check** after implementation: the glacier scalars (inert
  here), and the snow scalars if snow does not bite this forcing, could drift
  silently until the deferred baseline-manifest-integrity followup adds a
  TOML-scalar fingerprint. This de-scope is sanctioned, but the gap is named here
  and carried as an open item for that followup (`dev/followups.md` § baseline
  manifest integrity).

*Neutral*
- The discharge measurement certifies the **net effect of the restored set on
  the test basin only** — never a per-parameter effect or default-equality.
  For the 3 glacier params it certifies nothing (inert here); for the
  snow/soil/veg params it does not certify any *other* basin. The cross-basin
  correctness argument — not test-basin sensitivity — is what justifies the
  glacier and cross-basin snow restorations, recorded here so a future reader
  does not "re-drop" them after seeing a null or immaterial test-basin diff.
- `KsatHorFrac` is unchanged; this task does not revisit it.

### Alternatives considered

- **Defer / do nothing now.** On the facts this is the cheapest viable option:
  the only build-breaking parameter (`KsatHorFrac`) is already retained, the
  baseline runs green through R3/R4/R5, and 8/13 restored values equal the doc
  template default so the restore is near-null for most of the set. **Rejected**
  on explicit grounds: (1) the CSDMS mapping surface and the precedence/units
  silent-failure surface are cheapest to nail down **now**, while the M2b context
  is fresh — deferring pays a re-discovery cost when a snow/glacier basin is
  eventually assessed and the missing constants surface as a mystery; (2) the 4
  differing params are a real, unmade reference choice that a defer leaves
  silently resolved to whatever the engine default happens to be, on any basin,
  indefinitely; (3) the drift-protection benefit for the 8 template-equal params
  accrues only once they are pinned. Defer would be preferable if a snow/glacier
  basin assessment were not on the roadmap and engine defaults were stable and
  documented — neither holds.
- **Adopt Wflow.jl 1.x defaults wholesale (keep the M2b state).** Minimal config;
  rides engine improvements automatically. **Rejected:** it makes the fork's
  hydrology silently dependent on Wflow.jl default drift and, for the 4 differing
  params, diverges from the chosen reference precisely on glacier basins the
  current test basin cannot surface — a latent cross-basin defect. Preferable only
  if CST abandoned reference continuity in favor of "always latest engine
  defaults."
- **Restore only where the 1.x default differs (omit params equal to the
  default).** Smallest correct config. **Rejected:** identifying "where the
  default differs" would itself require a complete per-parameter default census
  against the 1.x engine docs (the equivalence gate verifies units/semantics,
  and resolves the default only where classification demands it, e.g.
  `MaxLeakage`) or one-at-a-time runs — the net discharge diff of the whole set
  cannot supply it (a net-null does not prove any parameter equals its
  default). Beyond the evidence cost, the standing reasons hold: (1)
  drift-protection — an omitted param that equals today's default silently
  follows a future default change; and (2) self-documentation — the config
  should state its physics rather than leave parameters implicit. Preferable if
  config minimalism strictly outranked drift-protection and self-documentation.
- **Drop any parameter immaterial on the test basin.** The literal "measure then
  decide per basin" reading. **Rejected:** test-basin immateriality is a
  property of the **aggregate set on this basin** — an artifact of inactive
  glacier processes (and, on a different basin, could be inactive snow), or even
  of offsetting effects — so it identifies no individual droppable parameter and
  would wrongly drop cross-basin-relevant constants. Measurement sets the
  *size* of the move and blesses it; it does not decide whether to restore.

### Related

- Task `t260719a` (`dev/TODO.md`); campaign tracker
  `dev/tasks/2026-07-21_pre-r6-followups.md`.
- M2b handoff decision #3 (`dev/milestones/phase-1/m02b/handoff.md`) — original drop + the
  first CSDMS mappings; `dev/milestones/phase-1/m02b/baseline_diffs.md`.
- Baseline-manifest scope/staleness context and the **deferred general
  staticmaps/TOML manifest expansion**: `dev/followups.md`
  (§ cross-cutting — baseline manifest integrity) and `dev/milestones/r01/baseline_diffs.md`.
- Roadmap cross-cutting rule "every milestone preserves the M1 baseline unless
  intentionally changing behavior" (`dev/roadmap.md`) — this task is a sanctioned
  wf1-slice (and, if material, wf3-slice) move.
- Source of truth: `hydromt_wflow/wflow_base.py:1429–1461` (`setup_constant_pars`
  writes `input.static.<name>.value`); `config/wflow_build_model.yml`
  (`setup_constant_pars` block); `hydromt_wflow/naming.py` (CSDMS lookup);
  `docs/hydromt-wflow/user-guide.md:300–314` (1.x default template);
  `docs/wflow-user-guide/02-required-files.md:73` (bundled 1.x name table —
  maps `MaxLeakage`'s name only, no default/units; why the gate consults the
  upstream Wflow.jl 1.x parameter docs);
  the applicable hydromt_wflow 0.x / Wflow (legacy) parameter reference (the
  0.x side of the equivalence gate — units/sign/scale of each short-name value
  as authored under git `6ebc46f`); `dev/scripts/check_baseline.py`
  (FINGERPRINTERS / TARGETS / diff_nc vs diff_hashed / record vs check);
  `examples/test_local/hydrology_model/wflow_sbm.toml` (built landing evidence).
