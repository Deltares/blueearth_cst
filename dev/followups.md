# Followups

Issues surfaced during pre-M1 cleanup that belong to later milestones.
Per the roadmap's "no milestone touches the next milestone's territory" rule,
captured here and resolved in passing when the relevant milestone starts.

Convention: one bullet per item. Keep the diagnosis date and reproducible
context so future-you can confirm the issue still applies before fixing.

---

## Post-P3-3 (surfaced 2026-07-25 during the P3-3 batching milestone)

- **Make the wf3 batch-size default genuinely disk-aware.** Design §6.1 names
  three ceilings on `B` and calls the **disk ceiling the BINDING constraint** on
  large `RLZ_NUM×ST_NUM` runs, capped so `p × B × (forcing_size + state_size)`
  stays inside a stated headroom. The landed default implements only the
  *parallelism* ceiling (`ceil(K / -c N)`), which scales `B` **up** with sweep
  size and therefore grows peak temp disk as the sweep grows — backwards from
  what §6.1 asks. Commit `3392587` bounds it with an overridable
  `batch_size_max` (default 8); that caps the blast radius but is a constant, not
  a disk computation. A real cap needs (a) a stated disk-headroom config key and
  (b) a per-run forcing+state size estimate, and (b) is the hard part: at parse
  time the forcing NCs are `temp()` and do not exist yet, so the estimate has to
  come from the wflow grid dimensions × run length × variable count, or from a
  measured prior run recorded in config. Verified 2026-07-25: fixture (K=12,
  `-c 3`) is unaffected — `min(ceil(12/3), 8) = 4`, so every P3-3 measurement
  stands; the clamp only binds from K > 24 at `-c 3`. Confirm the hazard still
  applies before fixing (it is scale-dependent and invisible on the seed
  fixture, whose peak footprint is 120 MB).
- **Consider recovering per-cst persistence isolation under batching.** C5 is
  DEGRADED by design (blast radius `B`): one failing cst causes Snakemake to
  delete the `B−1` completed sibling CSVs and re-run the whole batch, and rule
  3.11 is blocked sweep-wide until it succeeds. Measured exactly as documented
  (`dev/p33/batching-results.md` GN-4). §6.1 names the mechanism worth probing:
  the `--keep-incomplete` ↔ `--keep-going` interaction (does `--keep-incomplete`
  preserve successfully-written sibling CSVs across a failed batch job, and does
  the sweep then re-run only the failed cst?), with **accept-the-degradation as
  the explicit fallback** if the probe fails. Only worth doing if the blast
  radius actually bites in practice.

---

## Post-R6 (surfaced 2026-07-23 during the R6 milestone validation)

- **Make the projections summary CSV column order deterministic.**
  `blueearth_cst/projections/get_change_climate_proj.py` `intersection()`
  returns `list(set(lst1) & set(lst2))` — hash-order dependent with unpinned
  `PYTHONHASHSEED`, so `annual_change_scalar_stats_summary{,_mean}.csv` flip
  `precip`/`temp` column order run-to-run (demonstrated via a 15× hash-seed
  experiment; values identical by label, sibling `.nc` unaffected; consumers
  read by name, so harmless today). Fix: `sorted(set(lst1) & set(lst2))`.
  One-line, but touches a computational-path file — needs a baseline-gated
  commit, hence deferred out of R6.
- ~~**`semantic_tree_diff.py` exclusion refinement.**~~ **CLOSED 2026-07-25 —
  already fixed, no action taken.** The cited defect (stray `.log`/`.txt` under
  `hydrology_model/` reaching the hash comparator as benign FAILs) was resolved
  by P3-1 commit `576b6a6` ("exclude run-log files (5b)"), which added a
  file-level rule to `_is_excluded`: `rel.suffix == ".log" or rel.name ==
  "log.txt"`. Verified 2026-07-25 by calling `_is_excluded` directly on the
  three exact paths — `hydrology_model/hydromt.log`,
  `hydrology_model/run_default/log.txt`,
  `experiments/experiment/model_runs/log.txt` — all three EXCLUDED. The only
  other non-standard-extension file in the tree,
  `hydrology_model/staticgeoms/reservoirs_lakes_glaciers.txt`, is correctly
  hash-compared: it is content-bearing and deterministic, and passed as one of
  the 102 CLEAN files in the P3-3 value-identity gate. **The residual
  suggestion (a generic extension-level volatile class / per-tree exclude
  globs) is deliberately NOT implemented:** this is a *gate* tool, and widening
  its exclusions without a demonstrated false FAIL trades real detection for
  nothing. Reopen only with a concrete benign-FAIL case.
- **Dead-fixture audit: `tests/wflow_build_model.yml`.** No config value points
  at it (design §2, confirmed in R6); its own `config_fn:` line was the only
  reference to the now-moved `config/wflow_sbm.toml`. Confirm dead and remove,
  or wire it up intentionally.
- **`scripts/run_snake_test.cmd` modernization.** Uses `conda activate`, needs
  graphviz `dot`, ends in `pause` — hostile to non-interactive/pixi use; both
  R6 e2e runs substituted direct `snakemake`/wrapper invocations. Either port
  it to `pixi run` + no `pause`, or fold its role into
  `scripts/run_workflows.py` docs and retire it.

---

## Cross-cutting — baseline manifest integrity

- **[RESOLVED 2026-07-18] Baseline rebuilt from a tracked seed config.**
  `dev/baseline/manifest.json` was re-recorded from the now-tracked
  `config/snake_config_model_test.yml` (project_dir `examples/test_local`,
  3 models, single `far` horizon, current libraries), after a fresh run of
  all three workflows. `record` → `check` round-trips clean (14/14). The
  untracked `snake_config_model_test_local.yml` that seeded the stale M2b
  baseline is retired, so the divergence cannot recur. The model-independent
  workflow-1 PNG drift noted below was not separately investigated — it is
  moot now that the whole baseline is re-recorded from a known, tracked
  config; revisit only if a future `check` shows unexplained PNG drift.
  Original diagnosis retained below for provenance.

- **Rebuild `dev/baseline/manifest.json` against current libraries with a
  tracked seed config.** *Surfaced 2026-07-18 during R01 Task 5.* The M2b
  manifest (last recorded 2026-05-08, commit `159e197`) was recorded from
  an **untracked, 3-model local config**, while the tracked canonical
  (`config/snake_config_model_test.yml`) has used an **8-model** list since
  before `pre-r01`. The two have been silently divergent since M2b — the
  workflow smoke tests never compare against the manifest, so nobody caught
  it. R01 Task 5 was the first `check_baseline check` against the canonical
  model set and exposed the mismatch (see `dev/r01/baseline_diffs.md`).

  A fresh 8-model canonical run also shows **model-independent** drift on
  workflow-1 plots (`basin_area/hydro_wflow_1/precip.png`, 19–32% size) —
  workflow 1 never reads the climate model list, so the M2b local config
  must have differed from the canonical in ways beyond model count, and
  **cannot be reconstructed**.

  *Fix:* choose a deliberate model set, commit a **tracked** seed config
  (or parameterize `check_baseline` with one) so the baseline is
  reproducible, run all three workflows on current libraries, and record a
  fresh manifest. Investigate whether the workflow-1 PNG drift is
  rendering-only (matplotlib) or real content before blessing it. Until
  then the M2b manifest remains the contract of record, with R01 sealed on
  invariance-by-construction rather than a re-record.

---

## Cross-cutting — workflow ergonomics

- **[PARKED 2026-07-19] Per-rule progress messages.** Add `message:`
  directives to the long-running rules so each announces itself in plain
  language when it starts (e.g. "Building Wflow model from global data…"),
  layered on top of Snakemake's built-in `N of M steps (X%) done` counter and
  the per-rule timestamps. Snakemake cannot show progress *inside* an external
  step (hydromt build, Julia) — only start/end — but the tool's own streamed
  output (now visible via `tee`) covers the in-between. Cross-cutting: apply
  across all three `Snakefile_*` as a consistent pattern; R4/R5 would inherit
  it. Per-rule wall-clock is already captured by the `benchmark:` TSVs added in
  R3. Deferred by choice, not a blocker — pick up when convenient (a natural
  fit alongside R4/R5 Snakefile work or R6 polish).

- **[RESOLVED 2026-07-21, commit `d13ba37` (t260721a, `fix/pre-r6-followups`).]**
  wf1's three shell rules now route through `src/run_logged.py` (a CLI over
  `snake_utils.run_and_tee`), a portable Python tee wrapper that keeps live
  console output, writes the log, and exits with the child's own return code.
  Verified end-to-end: a deliberately-failing child propagates its non-zero code
  (the old `| tee` masked it to 0 under cmd.exe). Original diagnosis retained
  below for provenance.

- **[Latent robustness, not a blocker] wf1's `| tee {log}` shell rules mask the
  exit code on failure.** *Surfaced 2026-07-20 during R5 (design §2 ruling).*
  `Snakefile_model_creation`'s three shell rules (lines 89, 167, 182 — `hydromt
  build`, `hydromt update`, Julia `Wflow.run()`) use `... 2>&1 | tee {log}`.
  A bare `cmd | tee` pipeline returns `tee`'s exit status, not `cmd`'s, unless
  bash `pipefail` is active. On **Windows/cmd.exe** Snakemake injects **no**
  `set -euo pipefail` prefix (that prefix is bash-only — verified against
  Snakemake 9.6.2 `shell.py`), so a genuine `cmd` failure is reported as
  success. Verified empirically 2026-07-20 in a scratch Snakefile: a
  deliberately-failing command under `| tee` → Snakemake reports success;
  under `> {log} 2>&1` → Snakemake fails ("command exited with non-zero exit
  code"). On POSIX/bash the `pipefail` prefix protects, and on **success** the
  wf1 `| tee` rules run correctly (R3 sealed via a full `--forceall` wf1 rebuild
  that wrote all three tee logs and passed 14/14 on this machine) — so this is a
  **latent** failure-masking gap that only bites if a wf1 rule actually fails
  mid-run, **not** a gate blocker.

  *Fix:* migrate wf1's three shell rules to the exit-preserving `> {log} 2>&1`
  form R5 adopted for workflow 3's shell rules, **or** adopt a portable Python
  tee wrapper repo-wide if live console streaming must return (the tee form was
  deliberately chosen in commit `4a67d79` to restore live output). Owner:
  `cst-architect`. Activation: **next time wf1 shell-rule robustness is worked
  on.** wf3's own new shell rules already use `> {log} 2>&1` (R5 commit 8), so
  R5 introduces no new instance of the masking.

---

## R3 — Workflow 1: model builder

- **Resolve test_cli xfails.** Two of the three parametrizations in
  `tests/test_cli.py` are marked `xfail` since M2:
  - `Snakefile_climate_projections`: dry-run trips
    `MissingInputException` because the workflow expects
    `staticgeoms/region.geojson` (produced by Snakefile_model_creation)
    even when only dry-running. Either change the test fixture to
    pre-stage that file, or refactor `Snakefile_climate_projections` so
    `--dry-run` doesn't require it.
  - `Snakefile_climate_experiment`: dry-run trips
    `CyclicGraphException` at `rule generate_climate_stress_test`. The
    rule's wildcard pattern `rlz_{rlz_num}_cst_{st_num}.nc` overlaps
    with `generate_weather_realization`'s output `rlz_{rlz_num}_cst_0.nc`.
    Production configs (`config/snake_config_model_test_local.yml`) work
    fine because Snakemake disambiguates from concrete paths in
    `expand(...)`, but the `--dry-run` resolver flags the cycle on the
    test config. Add a wildcard constraint (`{st_num,[1-9][0-9]*}` or
    similar) or a `ruleorder:` directive.

  These are pre-M2 failures masked by the fact that M1 closure didn't
  actually run pytest.

  *Split 2026-07-19 (`dev/r03/model-builder-design.md` §2), by where the fix
  lives:* the `MissingInputException` is a workflow-2 **test-fixture** defect
  (dry-run against an empty project dir) — **fixed in R3** by pre-staging a
  minimal valid `region.geojson` and flipping that ratchet. The
  `CyclicGraphException` fix is a `wildcard_constraints`/`ruleorder` edit
  **inside `Snakefile_climate_experiment`** (R5 territory, entangled with the
  `st_num2 → st_num` fold that `dev/conventions/naming.md` §4 already assigns
  to R5) — **deferred to R5**; the ratchet is retained until then.

- **[RESOLVED 2026-07-21, t260716a′ (`fix/pre-r6-followups`).] Redo M1 warnings
  triage exhaustively.** Swept 82 captured `.log` files across all three workflows
  (per-rule `log:` directives now present via R3/R4/R5). **Bucket 3 (our-code):
  empty** — no warnings framed in `src/`, the Snakefiles, `dev/scripts/`, or the R
  layer. **Bucket 2:** one item, intended hydromt behavior (the `0.00833` vs native
  `0.008333333333325754` resolution snap) — won't-fix (a config match is fragile +
  would drift the tracked snake-config fingerprint for zero model change).
  **Bucket 1:** hydromt CRS/forcing/model-dir warnings + a new-but-captured 62×
  `Error in sys.excepthook` shutdown cascade from `hydromt build -vv` (post-success;
  upstream subprocess, not our tee wrapper — absent from the Julia/hydromt-update
  logs that use the same wrapper). No code changes. Full re-triage recorded in
  `dev/phase-1/m01/warnings.md` § "Exhaustive re-triage — 2026-07-21".

- **`extract_climate_grid` silently truncates the historical range.**
  *[Truncation WARNING resolved 2026-07-21, commit `ce56bc3` (t260716a,
  `fix/pre-r6-followups`): `prep_historical_climate` now emits an advisory
  warning when the extracted span falls short of the requested window. The
  config-staleness half below remains OPEN.]*
  When the snake config's `historical:` window asks for years that the
  staged source doesn't cover, the rule produces a shorter `extract_historical.nc`
  without any warning. Downstream rules then fail in cryptic ways far from
  the actual cause.

  *Observed 2026-05-07:* config asked `historical: 1980, 2010` (31 years),
  the staged era5 only had data from 2000-01-01 onward, so the extracted
  netCDF held 2000–2010 = **11 years**. That fell below weathergenr's
  16-year wavelet minimum and crashed `generate_weather_realization`
  with `'series' must have at least 16 observations`.

  *Fix:* in `extract_climate_grid` (or its underlying script), log a warning
  when the extracted time span is shorter than the requested span. Optionally,
  fail the rule if the shortfall is large enough to break a downstream step
  (e.g. < 16 historical years when weathergenr is in the pipeline).

  *Related Snakemake-staleness issue:* **[RESOLVED 2026-07-21, t260716a′ — by R5's
  wiring + verification, no new code.]** The 2026-05-07 repro (changing `historical:`
  didn't re-extract) predates R5, when the dates were hardcoded and `historical:`
  was **never read** by `extract_climate_grid` — so of course the edit had no effect.
  R5 wired `shared.historical_window` into the rule as `params`
  (`starttime`/`endtime`, `Snakefile_climate_experiment:78-82`), and Snakemake 9.6.2
  applies its default `params` rerun-trigger (no `--rerun-triggers` override in the
  repo). **Verified empirically 2026-07-21:** a dry-run against the built
  `examples/test_local` with `historical_window.endtime` changed 2020→2019 schedules
  `extract_climate_grid` with reason *"Params have changed since last execution:
  before '2020-12-31…' now '2019-12-31…'"*. So config edits to the historical window
  now propagate automatically; `--forcerun` is no longer needed. Declaring the whole
  config as an input (the original suggestion) is unnecessary and coarser (would
  re-run on any unrelated config edit). The broader "audit every rule whose behavior
  depends on an unread config key" remains a general note (see R5 section below), not
  part of this item.

  *Workaround applied 2026-05-07:* `historical: 2000, 2020` in the local test
  config + `--forcerun extract_climate_grid` for the immediate run. Treats
  the symptom; doesn't fix either of the two underlying issues.

---

## R5 — Workflow 3: climate experiment

- **`extract_climate_grid` ignores the `historical:` config and hardcodes
  the date range.** Pre-R5 unblocking edit on 2026-05-07 changed the
  hardcoded `starttime="2000-01-01"` / `endtime="2020-12-31"` in
  `src/extract_historical_climate.py`. The snake config's `historical:`
  key is read by `Snakefile_climate_projections` (workflow 2) but never
  by `Snakefile_climate_experiment` (workflow 3) — the rule
  `extract_climate_grid` only receives `data_sources` and `clim_source`
  as params; `historical:` is silently ignored.

  *Proper fix:* in `Snakefile_climate_experiment`, parse `historical:`
  from the config and pass `starttime` / `endtime` as rule params; in
  `src/extract_historical_climate.py`, replace the hardcoded date
  strings with `sm.params.starttime` / `sm.params.endtime`. While at it,
  drop the misleading function defaults at lines 20–21 (currently still
  say `1980` / `2010`).

  *Also touches the R3 followup:* this is the same shape as the
  config-key-not-wired pattern — fixing it should be paired with a
  general audit of every rule whose behavior depends on a config key
  that isn't actually read.

- **`weathergenr::write_netcdf` does not propagate `spatial_ref` attributes
  from `template_path` to the output.** Confirmed 2026-05-07: the historical
  template (`extract_historical.nc`) has `x_dim='longitude'` and
  `y_dim='latitude'` on its `spatial_ref` variable, but the realization
  files written by `write_netcdf` (`rlz_*_cst_0.nc`) have an *empty*
  attribute list on their `spatial_ref` variable. Downstream
  (`impose_climate_change.R`) then crashes when it uses the realization
  as its own template, because `write_netcdf`'s `x_dim` lookup returns
  `0` (numeric, from `ncatt_get` on a missing attr) — which slips past
  the existence check and causes
  `Error in nc_in$dim[[x_dim_name]] : attempt to select less than one element`.

  *Workaround applied 2026-05-07:* in `src/weathergen/generate_weather.R`,
  after each `write_netcdf` call, manually copy `spatial_ref` attributes
  from the historical input file to the just-written realization file
  via `ncdf4::ncatt_get` / `ncatt_put`. Marked clearly so it can be
  removed when weathergenr is fixed.

  *Proper fix:* in `tanerumit/weathergenr` `R/io_netcdf.R`, the
  attribute-copy loop in `write_netcdf` looks correct on the surface
  (`ncatt_get(nc_in, spatial_ref)` → `ncatt_put`) but evidently isn't
  executing or isn't writing through. Investigate why the loop produces
  zero attributes on the output. Separately, the missing-attribute check
  should also assert `hasatt = TRUE` on the `ncatt_get` result, not just
  test the value for NA / NULL — the current check accepts the numeric
  `0` returned for a missing attribute and crashes one line later.

- **weathergenr's wavelet minimum surfaces as a cryptic error.**
  `wavelet_cwt.R` enforces `length(series) >= 16` on the *annual* aggregate
  (i.e. ≥ 16 historical years), but the user-facing error is just
  `'series' must have at least 16 observations` — no mention of years,
  wavelet, or how to remedy.

  *Fix:* improve the error in `tanerumit/weathergenr` (upstream of this repo).
  Suggested message: *"historical period (N years) is below weathergenr's
  wavelet minimum of 16 years; extend the historical range or reduce the
  wavelet decomposition depth."*

  *Note:* this fix lives in the weathergenr package, not this repo. Mention
  in R5 deliverables if R5 is also touching the R layer; otherwise track as
  a separate weathergenr issue.

---

## R3+ — Surfaced during M02c (test coverage)

Lessons learned writing the M02c unit tests. Not bugs — testing-discipline
notes for R3-R5 to inherit when they add their own test files.

- **Test pollution between `sys.modules.setdefault` files.** pytest collects
  test files in alphabetical order. The first file to call
  `sys.modules.setdefault("hydromt", <stub>)` (or any heavy dep) wins, and
  later files using `setdefault` for the same key get a silent no-op —
  their import of the source module then binds to the *previous* test
  file's stub. Symptom: tests pass when run in isolation, fail in the full
  suite with `KeyError` on fixture-set catalog data.

  *Pattern:* don't rely on `setdefault` alone for shared keys. Use
  `monkeypatch.setattr(<source_module>.<dep>, "<attr>", <fake>)` inside
  fixtures so each test gets a clean override regardless of collection
  order. See `tests/test_prepare_climate_data_catalog.py` for the
  reference implementation; commit `f65244e` for the diagnosis.

- **dask cannot be stubbed at module level.** pandas does a lazy
  `import dask` and accesses `dask.__spec__` during type compatibility
  checks. A `types.SimpleNamespace` stub for dask there raises
  `ValueError: dask.__spec__ is not set` during collection of *any* test
  file that imports pandas. dask is in the env via pixi; let it import
  normally. If the cost matters, mock the specific dask object at call
  time within the test, not at module level.

---

## R3+ — Surfaced during M2b (library upgrades)

Items surfaced during the hydromt 0.x → 1.x / hydromt_wflow 0.x → 1.x /
Wflow.jl 0.7 → 1.0.2 / pandas 3.x / Python 3.12 jump. See `dev/phase-1/m02b/`
for the full M2b record.

- **`hydromt 1.x` `to_dict` / `to_yml` silently strips `driver.options.preprocess`.**
  Round-tripping a catalog dict through `DataCatalog().from_dict(...).to_yml(path)`
  loses the preprocess hook even though `from_dict` preserves it on read.
  *Workaround applied:* `src/prepare_climate_data_catalog.py` bypasses
  `to_yml` and uses `yaml.safe_dump` directly.
  *Proper fix:* file upstream against `hydromt`. Reproducer is the
  three-line snippet in `dev/phase-1/m02b/handoff.md` decision section.

- **conda-forge does not ship `julia` for win-64 at all.** linux-64 / osx-64
  have 1.10.x and 1.12.x but skip 1.11.x; win-64 has nothing. This blocks the
  "single env via pixi" goal on Windows for the Julia layer. Today juliaup
  manages Julia 1.11.7 outside pixi.
  *Possible fixes:* (a) wait for conda-forge to ship win-64 Julia; (b) wrap
  juliaup in a pixi `[tasks]` step that calls `juliaup install 1.11.7` at
  env-setup time; (c) move to a different distribution channel.

- **[RESOLVED 2026-07-17] weathergenr crashed loading on Windows —
  root cause was conda-forge's r45 `r-waveslim` build, not `ncdf4`.**
  `library(weathergenr)` (and the install's lazy-load step) died with
  `Mingw-w64 runtime failure: 32 bit pseudo relocation ... out of range`.
  Isolated by loading each Import in turn: the first 15 loaded fine and
  only `waveslim` overflowed — its Fortran DLL carries a 32-bit
  pseudo-relocation to libgfortran that lands ~2.7 GB away (past the
  signed 2 GB range), so load order can't dodge it. The earlier "likely
  ncdf4" and "user lib on Windows" notes were both wrong: under pixi
  `Rscript --vanilla` the only libPath is the conda site-lib, and ncdf4
  is fine. The bug is specific to the **r45** (R 4.5) waveslim build; the
  **r44** build loads and runs `modwt` cleanly.
  *Fix applied:* pin `r-base = "4.4.*"` in `pixi.toml` so the solver picks
  the working r44 waveslim (and r44 builds of the other Fortran deps).
  Also switched `install_weathergenr.R` from `pak` (conda-forge `r-pak` is
  separately broken on win-64 — "Wrong OS or architecture") to
  `remotes::install_github(dependencies=FALSE, upgrade="never")`, which
  touches nothing but weathergenr itself. Verified: `pixi run install-rdeps`
  installs and `library(weathergenr)` loads.
  *Revisit when:* conda-forge ships a fixed r45 `r-waveslim` (or R 4.6)
  Fortran build — then the `r-base` pin can move forward again.

- **`setup_constant_pars` short names → CSDMS Standard Names.**
  *Re-tagged 2026-07-19: standalone scientific-review task `t260719a`, split
  out of R3.* hydromt_wflow 1.x's `setup_constant_pars` rejects the short
  parameter names from 0.x and requires CSDMS Standard Names instead. M2b
  dropped **14 of the 15** originally-set constants under the "intentional
  drift, re-baseline aggressively" policy and kept only `KsatHorFrac` (which
  the build errors without).

  **Authoritative inventory (the handoff prose miscounts — its explicit
  parenthesized list of 15 names controls, not its "14 constant pars / other
  13" prose):** 15 original / 1 retained (`KsatHorFrac`) / **14 dropped**,
  where the 14 dropped = **8 known CSDMS mappings** (`Cfmax`, `WHC`, `TT`,
  `TTI`, `TTM`, `G_Cfmax`, `MaxLeakage`, `InfiltCapPath`) + **`InfiltCapSoil`**
  (deprecated, `wflow_v1: None` in `hydromt_wflow.naming` → stays dropped) +
  **5 unresolved** (`cf_soil`, `EoverR`, `rootdistpar`, `G_SIfrac`, `G_TT`)
  whose CSDMS mapping or deprecation status is not yet confirmed.

  Restoring physics parameters is a scientific decision, not a mechanical
  rename, so it is **out of R3** (which is a behavior-preserving refactor,
  per `dev/r03/model-builder-design.md`). The dedicated task owns
  `config/wflow_build_model.yml` and the resulting baseline move. Its scope
  must carry: a **parameter-reconciliation table** (per param: old value,
  Wflow 1.x effective default, units, semantics, storage location, observed
  built value, and a restore / adopt-new-default / drop-deprecated
  classification); a **direct staticmaps.nc/TOML assertion** that each
  restored value actually lands (a name accepted but silently no-op'd is the
  failure mode); a **data-level workflow-1 discharge comparison** (not PNG
  size — the manifest does not fingerprint discharge); and a **clean
  dedicated project-dir re-record** with a freshness check on every recorded
  target (existence-based Snakemake timestamps + `ancient()` inputs can bless
  stale artifacts). The task should also ADD staticmaps/TOML fingerprints to
  the manifest, since workflow 1's slice is currently only 3 size-only PNGs +
  a snake-config snapshot. CSDMS lookup tables in `hydromt_wflow.naming` and
  `hydromt_wflow.version_upgrade`. Concrete 8-mapping remap in
  `dev/phase-1/m02b/handoff.md` decision #3.

- **[RESOLVED 2026-07-21, t260720e — does-not-reproduce, no fix.] CMIP6 `precip` /
  `temp` `.attrs` lost on `monthly_change_scalar_merge`.** Under the current pinned
  env (hydromt 1.3.1) the merged `annual_change_scalar_stats_summary.nc` carries the
  **full CF set** (`cell_measures`, `cell_methods`, `comment`, `long_name`,
  `original_name`, `standard_name`, `units`) on both `precip` and `temp` — verified
  on the real-CMIP6-read output in `examples/test_local` AND in the recorded manifest
  fingerprint (`check --workflow climate_projections` passes on the `.nc`). R4's
  `probe_attrs_chain.py` proved no wf2 code drops attrs, and the values are
  CMIP6-native, so the hydromt read preserves them. The M2b `{}` diagnosis no longer
  reproduces; original root cause not re-litigated (moot). Absorbed the old t260716c
  "CMIP6 attr loss on merge" item. Full disposition: `dev/r04/chain-audit.md`
  § D-ATTRS.

- **Outlet station naming convention decision.** hydromt_wflow 1.x's
  `setup_outlets` uses subcatchment IDs (e.g. `130000086`, `1`, `2`, …) for
  outlet stations rather than the contiguous `1..N` of 0.x. The CSV column
  also renamed `Q_gauges` → `Q_outlets`. M2b's `src/plot_results.py`
  rebuilds `station_name` as `1..N` to keep `hydro_wflow_1.png` visually
  stable; R3 should pick a consistent project-wide convention (real
  subcatchment IDs vs `1..N` rebuild) and document it.

- **Retire the "CMIP6 GCS throughput regression" follow-up.** The original
  M2b mid-flight estimate was ~6 h for the full 3-model × 2-scenario fetch;
  the as-shipped run completed in 24 min after the eager `.load()` patch
  in `src/get_stats_climate_proj.py`. The followup line item was based on
  the slow path and no longer applies. (No file currently lists it
  separately — leaving this here as a reminder if it resurfaces.)

## R6 — Functional modularization (capability boundaries)

- **Climate analysis/visualization as a model-independent subworkflow.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation 4 follow-up).*
  We should be able to analyze and visualize climate data — gridded meteo
  diagnostics, forcing climatology, projection change factors — **without**
  building a hydrology model. Today the WF1 climate QA plots
  (`src/plot_results.py` §4) are coupled to the built wflow model
  (`mod.forcing.data`, `staticmaps["subcatchment"]`), and the forcing itself
  (`inmaps_historical.nc`) is a *product* of the model build. Yet the natural
  minimal dependency for climate analysis is a region/AOI geometry + data
  catalog — which WF3's `extract_climate_grid` (rule 3.02: `region.geojson` +
  clim source → `extract_historical.nc`) and WF2's `monthly_stats_*` already
  demonstrate (both depend only on `region.geojson`, not the full model).
  Direction: factor a shared **climate-analysis subworkflow/component** whose
  inputs are (region/AOI, gridded climate dataset) and whose outputs are
  climate diagnostics/plots, consumed by WF1 QA, WF2, and WF3 alike; degrade
  gracefully (region-only → basin-level; + subcatchment map → per-subcatchment).
  This is *functional* decomposition (capability boundaries), a **new axis**
  beyond the R6 roadmap's current layout/`enabled:` pain points (roadmap §R6) —
  add it to the R6 lock list when R6 scoping begins.
  **Tension to resolve:** ADR 0002
  (`dev/decisions/0002-revive-subcatchment-climate-plots.md`) currently sources
  the climate plots from `mod.forcing.data` (re-couples to the build); a modular
  design would source raw gridded climate (catalog + region) instead. Keep this
  in mind when ADR 0002 is implemented — it may argue for sourcing from
  `extract_climate_grid`-style extraction rather than the model forcing. To be
  discussed at R6 scoping; not to be designed or implemented yet.

- **Reconsider the WF1 rule arrangement — bundle/split + rename.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation re: WF1's 11
  rules).* NOT covered by R6's current lock list (which is repo/directory
  layout + `enabled:`); this is rule-level composition *within* a workflow — a
  new R6 axis. WF1 today has 12 rules (1.01–1.12; see `Snakefile_model_creation`
  and naming.md §9): copy_config, prepare_build_config, create_model,
  add_reservoirs_lakes_glaciers, add_gauges_and_outputs, write_outlet_index,
  setup_runtime, add_forcing, run_wflow, plot_results, plot_map, plot_forcing.
  Candidates to weigh:
  - **Plotting is three separate rules** (plot_results 1.10, plot_map 1.11,
    plot_forcing 1.12), each a `script:` emitting PNGs and now sharing
    `save_figure`. Consider consolidating into fewer rules (or one parameterized
    "plots" rule / a plotting sub-component) and a shared plotting module.
  - **Model-update chain is finely split** (create_model → add_reservoirs… →
    add_gauges… → write_outlet_index → setup_runtime → add_forcing). Some splits
    are historical: `add_reservoirs_lakes_glaciers`'s own comment says it "can be
    moved back to create_model when hydromt is updated" — a standing re-merge
    candidate.
  - **Verb standardization**: rules mix `create_`/`add_`/`setup_`/`prepare_`/
    `write_`/`plot_`/`run_`; `prepare_build_config` vs `setup_runtime` vs
    `create_model` overlap semantically. Align on a small verb vocabulary
    (naming.md §2 already prescribes `verb_noun`).
  **Key tradeoff — do not bundle blindly:** separate rules give Snakemake
  parallelism and *targeted* re-runs (edit forcing → only `plot_forcing` reruns);
  bundling coarsens the DAG and re-runs more on any change. Weigh granularity vs.
  readability per rule. Interactions: any reorg renumbers the `W.NN` scheme
  (naming.md §9 documents this as a mechanical cost), touches CLI target names
  (a naming.md §7 contract-surface rename → migration note), and overlaps the
  climate-subworkflow item above (plotting may move out of WF1 entirely). Same
  lens applies to WF3 (also 11 rules). To be discussed at R6 scoping; not to be
  designed or implemented yet.
