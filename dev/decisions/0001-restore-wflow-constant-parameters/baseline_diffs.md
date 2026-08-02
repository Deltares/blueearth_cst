# Baseline diffs — ADR 0001 constant-parameter restoration (t260719a)

Build-heavy execution record (ADR 0001 §"Validation protocol", steps 2/3a/3b/4/4b/5).
Measurement phase run 2026-07-21 on branch `fix/pre-r6-followups`; the equivalence
gate (step 3c) and the build-independent artifacts were committed at `1d011c0`.

## Build provenance (three clean builds)

All three built with `Snakefile_model_creation` targeting
`{project_dir}/hydrology_model/run_default/output.csv` (7670 daily timesteps,
2000-01-02 … 2020-12-31, outlet `Q_130000086`), same snake config
(`config/snake_config_model_test.yml` region/resolution/forcing), differing only in
`project_dir` + `model_build_config`. Data catalog `config/deltares_data.yml`
(root `C:\data\wflow_global\hydromt`); Julia 1.11.7 + Wflow via `--project=.`.

| build | project_dir (gitignored) | model_build_config | role |
|---|---|---|---|
| restored   | `examples/const_pars_restored`   | `dev/decisions/0001-restore-wflow-constant-parameters/config_restored.yml` (git `1d011c0`) | candidate (step 2) |
| baseline_a | `examples/const_pars_baseline_a` | `dev/decisions/0001-restore-wflow-constant-parameters/config_baseline.yml` (snapshot of `config/wflow_build_model.yml` @ `48bc1d3`) | reference (step 4) |
| baseline_b | `examples/const_pars_baseline_b` | `dev/decisions/0001-restore-wflow-constant-parameters/config_baseline.yml` (same) | reproducibility (step 4b) |

Two separately-materialized configs (never the edited file against itself, per ADR
ext2-2). All three build rc=0.

## Step 3a/3b — landing + precedence (restored build) — PASS

`verify_constant_pars.py --model-dir examples/const_pars_restored/hydrology_model
--config dev/decisions/0001-restore-wflow-constant-parameters/config_restored.yml` → exit 0.

**14/14 constants landed as `input.static.<name>.value` scalars, no shadowing.**
`snow__flag = True`, `glacier__flag = False`. Every value equals its
`config_restored` reference (KsatHorFrac 100; Cfmax 3.75653; TT 0; TTI 2; TTM 0;
WHC 0.1; cf_soil 0.038; EoverR 0.11; InfiltCapPath 5; MaxLeakage 0; rootdistpar
−500; G_Cfmax 5.3; G_SIfrac 0.002; G_TT 1.3). The 3 glacier scalars landed
correctly but are **inert on this basin** (`glacier__flag=false`) — certified by
the landing assertion, not discharge (ADR §Validation). No `staticmaps.nc`
variable collides with any CSDMS name.

## Step 4b — clean-build discharge reproducibility (attribution guard) — PASS

`check_baseline.py compare --ref baseline_a --cur baseline_b` → exit 0.

**0/7670 timesteps exceed tolerance; `max |dQ|/mean(Q_ref) = 0` (bit-identical).**
Two clean builds of the *same* config produce byte-identical discharge, so the
restored-vs-reference diff below is fully attributable to the configuration change
(the restored set as a whole), not to build/solver nondeterminism. (Confirms the
ADR step-4b concern — that raw daily `output.csv` at float64 might be LSB-sensitive
— does not bite here.)

## Step 5 — discharge materiality (restored vs reference) — IMMATERIAL (nonzero)

`check_baseline.py compare --ref baseline_a --cur restored` → exit 0.

| metric | value |
|---|---|
| timesteps exceeding tolerance | **0 / 7670** |
| ATOL (`1e-3 × mean(Q_ref)`) | 0.01095 |
| RTOL | 1% |
| `max |dQ|/mean(Q_ref)` | **1.655e-4** (0.017%) |
| `max relative (Q_ref ≥ ATOL)` | **7.091e-3** (0.71%, under RTOL) |

**Verdict: IMMATERIAL** — the restored set's net effect on this basin's discharge
is below both tolerance clauses at every timestep → ADR step-7 **immaterial
branch**. The move is **nonzero** (not a bit-exact no-op): peak local relative
0.71% sits just under the 1% RTOL. This is the expected, coherent signature — the
active reference-choice params (InfiltCapPath 5 vs engine 10, EoverR 0.11 vs 0.1,
TTI 2 vs 1) are read by the engine and perturb discharge slightly; the 3 glacier
params contribute nothing (inert). The nonzero-but-sub-tolerance move is
corroborating evidence the restored scalars are actually consumed, not shadowed.

**Certification boundary (ADR step 5):** this certifies the net effect of the
restored set on the **test basin only**. It certifies zero glacier effect (inert
here) and, for snow, a null-ish result certifies nothing about snow-dominated
basins; those rest on the landing assertion + equivalence gate + cross-basin
reference-continuity, not on this sensitivity.

## Step 7 — re-record plan (immaterial branch) — PENDING user go-ahead

The manifest keys bake in `project_dir` (`resolve()` in `check_baseline.py`), and
sidecar reference series are keyed `sha1(resolved_path)`, so re-baselining to the
restored model requires `examples/test_local` itself to BE the restored model at
record time. Immaterial-branch plan (the one irreversible, tracked-file step):

1. **Promote** `config_restored.yml` → `config/wflow_build_model.yml` (14-param
   block + the ADR-pointer comment; supersedes the old drop-comment).
2. **Rebuild** `examples/test_local` wf1 from the promoted config.
3. **Re-run** `verify_constant_pars.py` on `test_local` (final canonical landing check).
4. **`record --workflow model_creation`** from `examples/test_local` — merges the
   wf1 slice (3 PNGs + snake-config yaml + the new discharge target/reference
   series), preserving wf2/wf3 rows.

**wf3 residual (ADR immaterial branch):** the standing wf3 CSV fingerprints
(`Qstats.csv`, `basin.csv`) are byte-exact; the sub-tolerance wf1 move could
perturb them if wf3 is later re-run. Not re-recorded now (immaterial → wf1-only).
If a later wf3 check fails, follow the ADR recovery path: re-run wf3, confirm the
movement is consistent with this recorded 1.66e-4 wf1 diff, re-record the wf3 slice
with a note; else stop and investigate.

**Residual durable-coverage gap (ADR Consequences):** the restored TOML scalars
have no durable regression check after this (landing is a one-time script; the
manifest covers discharge only). Carried as the deferred baseline-manifest-integrity
followup.
