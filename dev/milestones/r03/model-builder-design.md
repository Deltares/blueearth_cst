# R03 — Workflow 1: model builder — design

**Date.** 2026-07-19.

## Goal

Clean up `Snakefile_model_creation` and the scripts it calls —
orchestration *and* analytical code — and establish the cross-cutting
Snakefile patterns (shared `get_config` helper, per-rule `log:` /
`benchmark:`) that R4 and R5 inherit. First milestone to *apply* the R1
config contract (via the deferred contract doc) and the R2 naming
conventions to code.

**R3 is a behavior-preserving refactor.** It makes **zero manifest
edits** and needs no `check_baseline record`: the gate stays clean 14/14
after every commit. The one physics change previously scoped here —
restoring the dropped CSDMS constant parameters — is deliberately **split
into a dedicated scientific-review task** (`t260719a`; see §5), because
restoring physics parameters is a scientific decision that must not be
made mechanically inside a refactor. That decision, and its baseline
move, live in the dedicated task.

One honest caveat, stated once here and relied on throughout: a clean
14/14 does *not* by itself prove workflow 1's *physical* artifacts are
unchanged, because the manifest fingerprints only 3 size-only PNGs and
one snake-config snapshot YAML for workflow 1 — it does **not**
fingerprint `staticmaps.nc`, `wflow_sbm.toml`, or `run_default/output.csv`
(verified in `dev/scripts/check_baseline.py` `TARGETS`). R3's guarantee
for those artifacts rests on *not editing the build config or any physics
path* — the gate is necessary, not sufficient. Adding the missing
staticmaps/TOML fingerprints is chartered to `t260719a`, the task that
actually changes physics (§5).

## Why now

R1 sealed the sectioned config; R2 sealed the naming contract. Both were
deliberately code-free, so workflow 1 still carries every deferred item
that accumulated since M2b: the "temporary hydromt fix" rule, the
outlet-naming shim in `plot_results.py`, deprecated `_fid`/`_file`
labels, zero `log:` / `benchmark:` directives, and a duplicated
`get_config` in all three Snakefiles. (The 14 dropped `setup_constant_pars`
values are also outstanding but are now `t260719a`, not R3 — §5.) R4 and
R5 are specified as "inherit the patterns established in R3" — none of
those patterns exist yet. R3 mints them once, on the smallest workflow,
before they are copied twice.

## Approach

Same contract-first discipline as R1/R2, applied to code:

1. **Contract before code.** `dev/reference/workflows/model_creation.md` (deferred
   from R1 by the 2026-07-17 amendment) is R3's first commit, written
   against the *current* behavior. Code commits then change behavior
   against a recorded contract, not from memory.
2. **Behavior-preserving throughout; `check_baseline check` stays clean,
   with zero manifest edits.** Every commit — helper extraction, log
   directives, label renames, the structured sentinel, outlet contract,
   the outlet-index `rule all` output, and the gauges-hardening error
   check — leaves the existing 14 manifest targets unchanged and requires
   no `record` run. One bounded caveat, labeled where it appears: the
   gauges hardening (§7.1) turns a silent drop into a loud error —
   output-neutral on a valid config, an error-behavior change only on an
   invalid one. The outlet-index CSV is a new `rule all` output but is
   *not* added to the manifest in R3 (§4) — its fingerprint defers to
   `t260719a` — precisely so no commit needs a re-record.
3. **Grandfathered vs applied.** Per `dev/reference/naming.md`, R3 is
   the owning milestone for workflow-1 identifiers: deprecated suffixes
   *inside this workflow's rules and scripts* are renamed (§6); anything
   outside workflow 1, and every §7 contract surface (output filenames,
   config keys, catalog names), stays grandfathered. No output filename
   changes in R3 — so no migration note is needed (§4 decision).
4. **Milestone boundaries beat followup tags.** Where `dev/followups.md`
   assigns R3 work that lives in another Snakefile, this design re-tags
   it rather than absorbing it (§2, §9). The iron rule wins.

## What changes

### 1. Contract doc `dev/reference/workflows/model_creation.md` (opening act)

Format per `dev/milestones/r01/modularity-contracts-design.md` §4, target < 100
lines. Content, grounded in the current Snakefile:

- **Owned config keys** — `workflows.model_creation.{wflow_outvars,
  model_build_config, waterbodies_config, output_locations,
  observations_timeseries}`.
- **Reads from shared** — `shared.basin.region`, `shared.basin.resolution`,
  `shared.historical_window.{starttime,endtime}`, `shared.clim_historical`.
- **Reads from project** — `project.{project_dir, static_dir, data_sources}`.
- **Input contract** — catalog sources required by
  `config/wflow_build_model.yml` and `wflow_update_waterbodies.yml`
  (merit_hydro, era5, reservoir/lake/glacier sources, …; enumerated
  during drafting from the build configs, not invented here).
- **Output contract**, split by role (the minor-finding distinction —
  these are not all `rule all` targets):
  - *Direct `rule all` targets* (what R3's `rule all` names statically):
    `plots/wflow_model_performance/{hydro_wflow_1,basin_area,precip}.png`
    and `config/snake_config_model_creation.yml`.
  - *Downstream-contract artifacts* (produced by intermediate rules,
    consumed by workflows 2/3, not in this `rule all`):
    `hydrology_model/staticmaps.nc`, `staticgeoms/region.geojson`,
    `staticgeoms/outlets.geojson`, `wflow_sbm.toml`,
    `climate_historical/wflow_data/inmaps_historical.nc`,
    `run_default/output.csv`.
  - *Side-effect artifacts* (bookkeeping, no downstream reader):
    `staticgeoms/reservoirs_lakes_glaciers.txt` sentinel; plus the new
    outlet-index mapping artifact (§4) and per-rule logs/benchmarks (§6),
    whose tracked-vs-ephemeral status is decided there.
  Includes the outlet-naming convention decided in §4.
- **Downstream consumers** — workflow 2 reads
  `staticgeoms/region.geojson` (as `ancient(...)` input to
  `monthly_stats_hist`/`_fut`); workflow 3 reads the built model, its
  TOML, and the forcing layout. This section is why the dedicated
  constant-restoration task (`t260719a`) can predict its
  workflow-1→workflow-3 baseline propagation rather than discover it.

### 2. Test ratchets: flip one, defer one (tension: cross-Snakefile)

`tests/test_cli.py` now carries two *known-failure ratchets* (not
xfails): each asserts non-zero exit **and** the specific DAG exception.
`dev/followups.md` assigns fixing both to R3; the roadmap forbids
touching R4/R5 Snakefiles. Resolution — split by where the fix lives,
not by where the symptom shows:

- **Workflow-2 ratchet (`MissingInputException` on `region.geojson`) —
  fixed in R3, from the test side.** The exception is *correct
  behavior*: workflow 2 declares a cross-workflow input that workflow 1
  produces, and the workflows are documented as running in order. The
  defect is in the test fixture, which dry-runs workflow 2 against an
  empty project dir. R3 pre-stages a minimal `region.geojson` in the
  dry-run fixture and flips the ratchet back to
  `assert result.returncode == 0`. The staged file MUST be:
  - created under a **test-owned temporary `project_dir`** (a pytest
    `tmp_path`-based fixture), never under the tracked
    `examples/test_local` baseline directory;
  - **syntactically valid minimal GeoJSON** (a `FeatureCollection` the
    dry-run's parse-time consumers accept), not a zero-byte placeholder;
  - **torn down with the fixture** (no residue between runs);
  - accompanied by an **assertion that `Snakefile_climate_projections`
    still declares the exact workflow-1 output path**
    (`{basin_dir}/staticgeoms/region.geojson`) as an input — so the
    fixture can never silently diverge from the real cross-workflow
    contract it stands in for.

  This proves **DAG construction, not semantic usability**: a user who
  runs workflow 2 first still hits a raw `MissingInputException`. That is
  correct-but-terse; a friendlier production error stays R4 territory.
  `tests/` is shared infrastructure and `Snakefile_climate_projections`
  is untouched — no boundary violation. *Rejected alternative:* making
  `region.geojson` optional in workflow 2 — that is an R4 content change
  *and* would weaken a correct input contract.
- **Workflow-3 ratchet (`CyclicGraphException` at
  `generate_climate_stress_test`) — deferred to R5, ratchet retained.**
  The only fix is a `wildcard_constraints`/`ruleorder` edit inside
  `Snakefile_climate_experiment` — R5 territory, and entangled with the
  `st_num2 → st_num` fold that `dev/reference/naming.md` §4 already
  assigns to R5. Fixing the constraint in R3 and restructuring the same
  wildcards in R5 means two passes over one surface. R3 re-tags this
  followup from R3 to R5 in `dev/followups.md`. *Rejected alternative:*
  treating the one-line constraint as blessed cross-cutting hygiene like
  the helper move — rejected because unlike §3 it changes DAG semantics
  (which paths a rule can match), which is exactly what "territory"
  protects.

Side benefit: the retained ratchet asserts a *specific* exception class,
so it doubles as a guard on §3 — if the helper move breaks config
reading in workflow 3, the error class changes and the ratchet goes red.

### 3. `src/snake_utils.py` — shared `get_config` (cross-cutting)

`src/snake_utils.py` does not exist yet (verified). R3 creates it and
moves the verbatim `get_config(config, arg, default=None, optional=True)`
body (currently duplicated at the top of all three Snakefiles) into it;
all three Snakefiles import it.

**Import mechanism — decided up front, not at implementation.** A plain
`from src.snake_utils import get_config` relies on the repo root being on
`sys.path`, which holds only when Python is invoked from the repo root.
Enumerating every documented invocation path:

| Invocation path | Working dir at parse | Repo root importable? |
| --------------- | -------------------- | --------------------- |
| Root-level `snakemake -s Snakefile_* --configfile …` (the `AGENTS.md` / README commands) | repo root | yes |
| `run_snake_test.cmd` (Windows wrapper) | repo root | yes |
| `run_snake_docker.sh` (Linux/Docker wrapper) | repo root (WORKDIR) | yes |
| `pytest tests/test_cli.py` → `os.chdir(SNAKEDIR)` then `snakemake` | repo root | yes |

All four already run from the repo root, so a bare import would work
today. But "works because every caller happens to `cd` to root" is
fragile — a future caller that does not is a silent breakage. So R3
specifies a **deterministic bootstrap keyed off the Snakefile's own
location**, independent of the working directory: at the top of each
Snakefile, prepend the Snakefile's directory to `sys.path` before
importing (`workflow.basedir` is the idiomatic Snakemake handle for the
Snakefile directory; an explicit `Path(workflow.snakefile).parent`
fallback if `basedir` semantics differ across the pinned Snakemake
version — confirmed against the pinned version during commit 2). This is
a one-time cost in R3 and is exactly what R4/R5 inherit.

**Boundary reconciliation.** This touches `Snakefile_climate_projections`
and `Snakefile_climate_experiment`, but the roadmap explicitly blesses it
("behavior of R4/R5's Snakefiles unchanged; only the helper sourcing
moves"). Precisely: the R4/R5 diffs are (a) delete the local 8-line
`get_config` def, (b) add the two-line `sys.path` bootstrap + import. The
bootstrap line is part of the blessed "helper sourcing moves" change — it
is the mechanism that makes the shared helper importable, not new
behavior; the roadmap's authorization covers it. No rule, param, config
read, or path changes. Safety: one isolated commit;
dry-run of workflow 1 must pass and both ratchets must still report their
exact known exceptions (§2) — any deviation means the move was not
behavior-preserving.

**Exact-equivalence tests land in the same commit, before the three
local defs are deleted** (§8): missing-required (raises `ValueError`),
missing-optional (returns the default), explicit default returned,
`None` stored as a value (returned as `None`, not treated as missing),
and falsey values (`0`, `""`, `False`, `[]` returned as-is, not
defaulted). This pins the semantics the old inline copies had so the
move is provably identity-preserving, not just green on the smoke test.
New identifiers in this module follow `dev/reference/naming.md` (it is
the guide's first consumer).

### 4. Outlet station naming — decision (M2b carryover)

Context: hydromt_wflow 1.x `setup_outlets` labels outlets with
subcatchment IDs (`130000086`, …) instead of 0.x's contiguous `1..N`, and
the output CSV column renamed `Q_gauges → Q_outlets`.
`src/plot_results.py` currently rebuilds `station_name` as
`wflow_{1..N}` to keep `hydro_wflow_1.png` stable. R2 explicitly punted
the decision here.

**Decision: keep the positional `1..N` convention; promote it from shim
to documented contract.**

- **Justification is proportionality + contract stability, not
  impossibility.** `rule all` and the baseline manifest name
  `hydro_wflow_1.png` *statically*, while outlet IDs are basin-derived.
  Data-derived ID filenames *could* be produced (Snakemake checkpoints,
  directory outputs, or a stable index artifact) — the review is right
  that "cannot" overstates it — but every such mechanism adds
  orchestration cost and turns the manifest into a basin-specific path
  inventory, a price out of proportion to R3's refactor scope. For a
  rapid multi-basin assessment tool, `hydro_wflow_1` meaning "first
  outlet" in every basin is a feature. The rejected alternative loses on
  cost and contract churn, not feasibility.
- **Unconditional machine-readable mapping (replaces the earlier
  `performance_metrics.csv` promise).** That promise was wrong:
  `src/plot_results.py` writes `performance_metrics.csv` from
  `df_perf_all`, which is populated only when observations overlap the
  run; the tracked seed config has `observations_timeseries: None`, so on
  the canonical run that file is empty and cannot carry the mapping.
  Instead, R3 emits a **dedicated outlet-index CSV**
  (`hydrology_model/staticgeoms/outlet_index.csv`, columns
  `station_name` = `wflow_{1..N}`, `subcatchment_id`, `x`, `y`) derived
  directly from `outlets.geojson` — populated on **every** run,
  observations or not. Surfacing the real ID in the plot title/legend is
  retained as a human aid but is *not* the machine-traceability path.
- **`rule all` yes; manifest — deferred to `t260719a`, not R3.** The
  artifact is a `rule all` output: reproducible, public, Snakemake-tracked,
  and populated on every run — which fully satisfies the review's "one
  unconditional machine-readable mapping." Whether it enters the
  `check_baseline` manifest is a separate question, and the honest answer
  is **not in R3.** Adding a manifest target requires *generating* the
  file (a workflow-1 run — it is new in R3, so not present from the
  2026-07-18 rebuild) and running `check_baseline record`; that is a
  re-record step, with the same staleness exposure R3 pushes to
  `t260719a`, and until it is recorded `check` fails on the missing
  target. Both collide with R3's clean-throughout posture. So R3 makes
  **zero manifest edits**; the outlet-index fingerprint is folded into
  `t260719a`, which is already chartered to add the `staticmaps.nc` /
  `wflow_sbm.toml` fingerprints (§5). The review's "if it becomes a
  public output it *should* enter the manifest" is honored there, in the
  task that already touches the manifest — not by forcing a record run
  into a behavior-preserving refactor.
- `Q_outlets` (CSV column) is upstream hydromt_wflow vocabulary —
  tier-1-adjacent; accepted as-is.

Consequences: no *existing* output filename changes → no migration note.
The convention is written into the contract doc (§1) and the
`plot_results.py` comment updated from "temporary rebuild" to "documented
convention". *Rejected alternative:* real subcatchment IDs in filenames —
rejected on the proportionality argument above; revisit only if a future
consumer needs ID-addressable plot files (it would read
`outlet_index.csv` or `outlets.geojson`).

### 5. CSDMS constant-parameter restoration — deferred to a dedicated task

**Not in R3.** Restoring the dropped `setup_constant_pars` values is a
*scientific* decision — it changes basin physics — and must not be made
mechanically while resolving CSDMS names inside a refactor. R3 therefore
**does not touch `config/wflow_build_model.yml` and does not move the
baseline.** The work has a standalone home: task `t260719a` (`dev/TODO.md`,
area `constant-params`; scope in `dev/followups.md` → M2b section).
Rationale for splitting it out rather than deferring to R4/R5: no later
milestone owns the build config, so without an explicit task the item
would have no home at all.

**Authoritative inventory** (the `dev/milestones/phase-1/m02b/handoff.md` prose
miscounts — it says "14 constant pars / other 13," but its own explicit
parenthesized list carries **15 names**; the list controls):

- **15 original** parameters.
- **1 retained** — `KsatHorFrac` (the build errors without it).
- **14 dropped**, comprising:
  - **8 known CSDMS mappings** — `Cfmax`, `WHC`, `TT`, `TTI`, `TTM`,
    `G_Cfmax`, `MaxLeakage`, `InfiltCapPath` (handoff decision #3).
  - **1 deprecated** — `InfiltCapSoil` (`wflow_v1: None` in
    `hydromt_wflow.naming`) → stays dropped.
  - **5 unresolved** — `cf_soil`, `EoverR`, `rootdistpar`, `G_SIfrac`,
    `G_TT` — CSDMS mapping or deprecation status not yet confirmed.

**What the dedicated task must carry** (the mitigations the review
raised, which belong to the task that actually moves the baseline):

1. A committed, machine-readable **parameter-reconciliation table** —
   per parameter: source (short) name, CSDMS name, old value, Wflow 1.x
   effective default, units, semantics, storage location, observed built
   value, and a **restore / adopt-new-default / drop-deprecated**
   classification. Any parameter whose units or semantics changed between
   0.x and 1.x is escalated for explicit review, not restored on autopilot.
2. A **direct assertion against `staticmaps.nc`/`wflow_sbm.toml`** that
   each restored value actually lands — a name accepted but silently
   no-op'd is the failure mode; presence-and-value, not presence-only.
3. A **data-level workflow-1 discharge comparison** (normalized
   discharge-series statistics / flow-duration quantiles from
   `run_default/output.csv`), *not* PNG size — the manifest does not
   fingerprint discharge, and size-only ±10% PNG deltas can miss a real
   curve change and flag harmless rendering churn.
4. A **clean, dedicated project directory** for the re-record run, with a
   per-target **freshness check** (existence-based Snakemake timestamps
   plus the in-place `staticmaps` mutation under `ancient()` inputs can
   otherwise bless stale artifacts into the new baseline). The task
   should also **add `staticmaps.nc` and `wflow_sbm.toml` fingerprints to
   the manifest**, since workflow 1's current slice is only 3 size-only
   PNGs + a snake-config snapshot and cannot by itself evidence a physics
   change.
5. Owned/propagated/unchanged baseline classification: owned = workflow-1
   targets; propagated = workflow-3 `Qstats.csv`/`basin.csv` (same
   staticmaps, no workflow-3 code edit); unchanged = workflow-2's targets
   (consume only `region.geojson`), a free independence cross-check.

### 6. Snakefile hygiene: `log:`/`benchmark:`, labels, `ancient()`, ruleorder

- **`log:` + `benchmark:` on every non-trivial rule.** *"Non-trivial"* is
  defined once, for R4/R5 to inherit: any rule that runs a `shell:` or
  `script:` doing real work — i.e. every rule except a pure
  file-copy/aggregation like `copy_config`. In workflow 1 that is:
  `prepare_build_config`, `create_model`, `add_reservoirs_lakes_glaciers`,
  `add_gauges_and_outputs`, `setup_runtime`, `add_forcing`, `run_wflow`,
  `plot_results`, `plot_map`, `plot_forcing`.
  - **Path convention (inherited verbatim by R4/R5):** `log:` →
    `{project_dir}/logs/{rule}.log` (per-rule; for wildcard rules,
    `{project_dir}/logs/{rule}/{wildcards…}.log` so concurrent jobs never
    collide); `benchmark:` → `{project_dir}/benchmarks/{rule}.tsv`
    (Snakemake's native TSV benchmark format).
  - **Tracked status:** logs and benchmarks are **ephemeral run
    artifacts**, not manifest targets and not committed —
    `{project_dir}/` is already gitignored and never fingerprinted. This
    keeps them off the baseline entirely, consistent with R3 being
    behavior-preserving.
  - **Shell rules** redirect directly: `> {log} 2>&1`.
  - **`script:` rules** are *not* auto-redirected by Snakemake, so they
    use **one tested context manager** — a second member of
    `snake_utils` (e.g. `tee_to_log(snakemake.log[0])`) that MUST:
    (i) redirect `stdout`/`stderr` to the log path; (ii) restore both
    streams in a `finally` so the redirection cannot leak across the
    process or subsequent imports; (iii) **not swallow exceptions** —
    re-raise so the traceback still reaches Snakemake and the rule fails
    loudly (an empty log on a swallowed error would let Snakemake treat a
    broken rule as complete); (iv) `mkdir -p` the log's parent; (v) use
    the unique per-rule/per-wildcard path from the convention above.
    Unit-tested (§8) for both normal completion (streams restored, log
    written) and exception propagation (exception re-raised, streams
    still restored).
  - This is the pattern R4/R5 copy, and it unblocks the deferred
    exhaustive M1 warnings triage: R3 sweeps the captured *workflow-1*
    logs and fixes **only output-neutral** warnings it owns — deprecation
    /API swaps like the pandas frequency renames (`"A"`→`"YE"`), which
    change no result. Any warning whose fix would alter an output value
    defers (it would move the baseline, violating R3's behavior-preserving
    posture). The triage closes fully only in R5 when all three Snakefiles
    have logs (the followup's "across all three Snakefiles" is reached
    cumulatively — the followup entry is annotated accordingly).
- **Deprecated label renames (naming applied).** Within
  `Snakefile_model_creation` and the scripts it calls, `gauges_fid`,
  `forcing_fid`, `csv_file`, `toml_fid`, `gauges_output_fid` →
  `_path`-style names per `dev/reference/naming.md` §3/§5. These are
  internal Snakemake labels and script params (not §7 contract
  surfaces); the paired script reads (`sm.input.*`/`sm.params.*`) are
  updated in the same commit. Exception: `gauges_fn` as a *hydromt_wflow
  API kwarg* at the `setup_gauges(gauges_fn=...)` call site is upstream
  vocabulary (tier 1) and stays. Labels outside workflow 1 stay
  grandfathered.
- **`ancient()` guards commented, kept.** `add_reservoirs_lakes_glaciers`
  and `add_gauges_and_outputs` mutate their own input (`staticmaps.nc`)
  in place; `ancient()` prevents the resulting re-trigger loop. The
  clean fix (immutable staged outputs per step) is a restructure —
  R6-scale. R3 documents the pattern in-place per the roadmap's
  "tightened (preferred) or commented in-place with the reason" rule.
- **Ruleorder:** `Snakefile_model_creation` contains no `ruleorder:`
  (verified); nothing to tighten here. The load-bearing one is in
  workflow 2 (R4).

### 7. Script cleanups: the two named reviews

- **`src/setup_reservoirs_lakes_glaciers.py` ("temporary hydromt fix").**
  Current behavior: a separate rule (split out of `create_model`, per its
  own comment "can be moved back when hydromt is updated") opens the
  built model `r+`, iterates methods from
  `config/wflow_update_waterbodies.yml`, catches
  `NoDataException`/`FileNotFoundError` per method (basins legitimately
  lack reservoirs/lakes/glaciers), and writes a sentinel
  `reservoirs_lakes_glaciers.txt` for Snakemake tracking. Cleanup
  target, two-branch with an executable gate:
  - **(a) attempted first — a fixed 2-hour timeboxed experiment:** move
    the `wflow_update_waterbodies.yml` methods into `create_model`'s
    build config and run `hydromt build wflow_sbm` on the tracked seed
    basin. **Acceptance test:** the build completes, `staticmaps.nc`
    contains the same reservoir/lake/glacier variables the separate rule
    produced today, **and** `check_baseline check` stays clean (no output
    drift). If all three hold within the 2 h, fold the rule away and drop
    the sentinel.
  - **(b) default if (a) fails or exceeds the timebox:** keep the rule,
    finish the encapsulation the roadmap asks for — a comment naming the
    upstream hydromt issue and an explicit removal trigger — and replace
    the free-text sentinel with a **structured** one (a small
    machine-readable table: method → `ok`/`skipped`/`failed` + reason) so
    the §6 log sweep can parse it. Changing the sentinel format is an
    observable interface change: it is recorded in the contract doc (§1,
    side-effect artifacts) and covered by the `setup_reservoirs_lakes_glaciers`
    unit test (§8). The sentinel is not a manifest target, so this does
    not touch the baseline.
- **`src/setup_gauges_and_outputs.py` (correctness + units).** Current
  behavior: maps user-facing output names to CSDMS via the module-level
  `WFLOW_VARS` dict, runs `setup_outlets` (Q at outlets),
  `setup_gauges` when `output_locations` is set (Q, P at gauges), and
  `setup_config_output_timeseries` for `_basavg` extras. Review targets:
  1. **Silent drops (error-behavior hardening).** `extras = [v for v in
     outputs if v != "river discharge" and v in WFLOW_VARS]` silently
     ignores any `wflow_outvars` entry not in `WFLOW_VARS` — a config
     typo produces no output and no error. Fix: raise on unknown names.
     Classify this honestly: it is an **error-behavior change**, not
     behavior-preservation in the general sense — but it is
     **output-neutral on a *valid* config** (every name already resolves),
     so the baseline stays clean. It only changes what happens on an
     *invalid* config (loud error instead of a silent gap). Covered by a
     unit test asserting the raise.
  2. **Units.** Verify header/param pairings (`Q`, `P`, `*_basavg`)
     against the Wflow 1.x CSDMS units of each mapped variable; record
     the label → CSDMS → unit table in the contract doc. Documentation +
     assertion only; no output change.
  3. **`WFLOW_VARS` completeness vs `plot_results.py` — documented, not
     changed in R3.** The plot script skips climate panels unless
     `P/T/EP_subcatchment` exist (its comment defers configuring them to
     R3). Changing the default `wflow_outvars` to make the standard plot
     suite complete **would move the baseline** (new plot outputs, and if
     the tracked seed config changes, its copied
     `snake_config_model_creation.yml` snapshot too). To keep R3
     behavior-preserving, R3 does **not** change `wflow_outvars` or the
     tracked config. It instead **documents** in the contract doc: (i) the
     recommended complete output set; (ii) that the canonical
     `config/snake_config_model_test.yml` and the pytest fixture
     `tests/snake_config_model_test.yml` currently carry *different*
     output sets (a discrepancy the review flagged) — recorded as a
     known state, not fixed here. Actually enabling the complete suite is
     a followup (it shares the baseline-move property with `t260719a`).

### 8. Unit tests

Per the M02c discipline (monkeypatch over `sys.modules.setdefault` for
shared heavy deps; see `dev/followups.md` R3+ notes):

- `snake_utils.get_config` — **exact equivalence** (§3): missing-required
  raises `ValueError`, missing-optional returns the default, explicit
  default returned, `None`-valued key returned as `None`, falsey values
  (`0`/`""`/`False`/`[]`) returned as-is. Lands *with* commit 2, before
  the inline defs are deleted.
- `snake_utils.tee_to_log` — the log context manager (§6): normal
  completion (streams restored in `finally`, log file written, parent
  created) and exception path (exception re-raised, streams still
  restored).
- `setup_gauges_and_outputs` — mapping correctness, **raise on unknown
  `wflow_outvars`** (§7.1), extras selection.
- `setup_reservoirs_lakes_glaciers` — method dispatch, per-method no-data
  capture, **structured sentinel** content (§7b).
- `prepare_build_config` — resolution/region merge.

Mocked `WflowSbmModel`; no full builds in unit tests —
`tests/test_model_creation.py` remains the heavy integration gate.

### 9. Disambiguation: `extract_climate_grid` is R5, not R3

`dev/followups.md` lists the truncation warning under R3 and the
hardcoded-date-range fix under R5. The rule `extract_climate_grid` and
`src/extract_historical_climate.py` are invoked from
`Snakefile_climate_experiment` — workflow-3 territory. Both items (and
the related config-staleness fix: declaring the relevant config keys as
rule inputs) defer to R5; R3 re-tags the followup entry. What R3 *does*
absorb is the general pattern the followup names: an audit of workflow-1
rules for config keys that are read but never wired into behavior (none
are currently known in workflow 1; the audit confirms).

## Verification

- **Per-commit:** `snakemake all -c 1 -s Snakefile_model_creation
  --configfile config/snake_config_model_test.yml --dry-run` after every
  Snakefile-touching commit; `pytest tests/test_cli.py` after every
  commit touching a Snakefile or script signature (per `AGENTS.md`).
- **Ratchet discipline:** after the fixture-repair commit, `test_cli.py`
  asserts success for workflows 1 and 2 and the retained
  `CyclicGraphException` ratchet for workflow 3.
- **Clean-baseline gate — held throughout, no `record`.** R3 is
  behavior-preserving and adds no manifest target, so `check_baseline
  check` reports **14/14 clean after every commit** and no commit runs
  `record`. Discriminating self-check: walk the commit list (below) and
  confirm none requires a `record` or a full workflow run to stay green —
  if one did, R3 would not be clean-throughout. (The gate remains
  *necessary, not sufficient* for physics — see the Goal caveat; that is
  why `t260719a` adds the staticmaps/TOML fingerprints, not R3.)
- **Freshness on any re-run:** if a commit's verification re-runs
  workflow 1 (e.g. the waterbodies fold-in test, §7a — the one place R3
  builds), it runs into a **clean, dedicated project directory** — never
  a stale `examples/test_local` where existence-based Snakemake
  timestamps and the in-place `staticmaps` mutation under `ancient()`
  could leave old artifacts in place. This is a correctness check, not a
  baseline record.
- **Suite:** full `pixi run pytest tests/` green with the new unit tests
  (get_config equivalence, the log context manager, gauges unknown-name
  raise, structured sentinel, prepare_build_config merge); final counts
  recorded in the roadmap seal.

## Out of scope

| Item                                                     | Why deferred                                             | Where |
| -------------------------------------------------------- | -------------------------------------------------------- | ----- |
| **CSDMS constant-parameter restoration** (`config/wflow_build_model.yml`) | **Scientific decision + baseline move; must not be mechanical in a refactor (§5)** | **`t260719a`** |
| Enabling the complete `wflow_outvars` plot suite (changing the tracked config) | Moves the baseline; §7.3 documents, does not change | followup |
| `Snakefile_climate_projections` content (incl. its `ruleorder:`) | R4 territory; only the §3 import lands            | R4    |
| `Snakefile_climate_experiment` content (incl. the `CyclicGraphException` fix, `st_num2` fold) | R5 territory; ratchet retained | R5 |
| `extract_climate_grid` truncation warning + config-staleness | Rule lives in workflow 3 (§9)                        | R5    |
| CMIP6 attrs lost on `monthly_change_scalar_merge`        | Workflow-2 script chain                                   | R4    |
| Exhaustive M1 warnings triage closure                    | Needs logs from all three Snakefiles; R3 sweeps workflow 1 only | R5 |
| Staticmaps in-place-mutation restructure (drop `ancient()`) | Structural; documented in-place instead                | R6    |
| Repo-wide directory restructuring; operational `enabled:` | Roadmap                                                  | R6    |
| Naming linter                                            | Only if drift appears                                     | R3+   |

## Risks and open questions

- **Riskiest remaining item: the `script:`-rule log context manager (§6).**
  Replacing `stdout`/`stderr` inside a Snakemake script process is the one
  place R3 can subtly break rule completion — a leaked stream, a swallowed
  traceback, or an empty log Snakemake reads as a finished product. The
  mitigation is baked into the API contract (restore in `finally`,
  re-raise exceptions, unique per-rule paths) and its two-case unit test
  (normal + exception). Treat this as the commit to review hardest.
- **PNG gate weakness (advisory only in R3).** Size-only ±10% fingerprints
  may not register real physics changes and can move on harmless rendering
  churn. R3 changes no physics, so this is not load-bearing here — but it
  is why `t260719a` must carry *data-level* discharge evidence rather than
  leaning on PNG deltas. Noted so the constraint transfers with the task.
- **Fixture pre-staging proves DAG construction, not usability.** Flipping
  the workflow-2 ratchet via a staged `region.geojson` means a user who
  runs workflow 2 first still gets a raw `MissingInputException`. That is
  correct-but-terse; the contract doc records the ordering requirement,
  the fixture asserts it still stands in for the real input path (§2), and
  friendlier onboarding is R4's call.
- **Scientific-position question travels with `t260719a`, not R3.** Whether
  the pre-M2b constant set is the right scientific baseline (vs. Wflow 1.x
  defaults embodying newer upstream science) is a real open question, but
  R3 no longer decides it. It is flagged in the dedicated task's scope for
  its scientific review. Recorded here only so the split does not lose it.

## Tag

`r03-model-builder`. Branch `milestone/r03-model-builder` off `main`
(which carries the sealed R2 state).

## Estimated commit decomposition (~11 commits)

With constant restoration removed, R3 is behavior-preserving and there is
**no baseline-record commit and no manifest edit**. `check_baseline check`
stays 14/14 after every commit; the outlet-index CSV ships as a `rule all`
output only, its manifest fingerprint deferred to `t260719a` (§4).

1. `r03: add dev/reference/workflows/model_creation.md contract doc` (current
   behavior; includes the output-role split and the wflow_outvars
   config-set discrepancy note)
2. `r03: collapse get_config into src/snake_utils.py with basedir bootstrap`
   *(touches all three Snakefiles; ships the get_config equivalence tests
   in the same commit; guarded by dry-run + both ratchets)*
3. `r03: add tee_to_log context manager and its unit test`
4. `r03: add log and benchmark directives to Snakefile_model_creation`
5. `r03: pre-stage region.geojson in dry-run fixture, flip projections ratchet`
6. `r03: rename deprecated path labels in workflow-1 rules and scripts`
7. `r03: harden setup_gauges_and_outputs (raise on unknown wflow_outvars, units audit)`
   *(error-behavior change; output-neutral on a valid config)*
8. `r03: emit outlet_index.csv as a rule_all output (documented outlet convention)`
   *(the unconditional position→subcatchment-ID mapping, §4; `rule all`
   output only, no manifest edit)*
9. `r03: encapsulate waterbodies fix with issue ref and structured sentinel`
   *(the §7a fold-in experiment is attempted here first; if it passes, the
   rule folds away instead of gaining a structured sentinel)*
10. `r03: add unit tests for workflow-1 helpers`
    *(gauges, reservoirs sentinel, prepare_build_config)*
11. `r03: seal milestone — dev/roadmap.md R3 section + dev/reference/branches-and-tags.md`
    (+ tag `r03-model-builder`)

R3 writes no `dev/milestones/r03/baseline_diffs.md` — it changes no baseline target,
so there is no diff to record (contrast R1, which re-recorded config-
snapshot hashes). Commit 11 names its exact files rather than "durable
refs"; if the roadmap seal and the branches-and-tags update turn out to be
independent, split them per the one-logical-change rule. Riskiest commit:
**3** (stream-redirection correctness in the log context manager); most
cross-cutting: **2** (all three Snakefiles). Discriminating check
(Verification): no commit above requires `check_baseline record` or a
baseline-recording workflow run.

## References

- `dev/roadmap.md` — R3 section, cross-cutting principles, commit strategy.
- `dev/milestones/r01/modularity-contracts-design.md` §4 — contract-doc format.
- `dev/reference/naming.md` — the applied naming contract (§3, §5, §6, §7).
- `dev/followups.md` — R3 / R3+ / cross-cutting items absorbed or re-tagged here.
- `dev/milestones/phase-1/m02b/handoff.md` — decisions #3 (CSDMS remap — note its
  prose miscount; the 15-name list controls), #4 (`mod.close()`), #5
  (outlet renames); historical record.
- `dev/milestones/r01/baseline_diffs.md` — why the manifest was rebuilt; provenance
  of the current 14/14 gate.
- `tests/test_cli.py` — ratchet semantics and flip discipline.
- `dev/TODO.md` `t260719a` + `dev/followups.md` M2b section — the
  constant-restoration task split out of R3 (§5).
- `dev/milestones/r03/model-builder-design-review-gpt-20260719.md` — the independent
  review this revision addresses.
