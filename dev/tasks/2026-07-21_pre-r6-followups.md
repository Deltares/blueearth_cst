> **PRE-BOARD RECORD — kept, not a board item (bannered 2026-08-07).**
> `dev/tasks/` became the todo-board's OPEN set on 2026-08-07; before that it
> held one record per closed task, which is what this is. It is not a
> `todo-item`, carries no `type:` frontmatter, and the board's loader ignores
> it by design. Closures from this campaign are NOT backfilled into
> `dev/LOG.md` — the pre-board ledger is `dev/followups-archive.md`.

# Pre-R6 followups campaign

Clearing the deferred backlog before the R6 structural refactor, so R6 does not
carry latent defects. Branch `fix/pre-r6-followups` off `main`.

**Scope decisions (2026-07-21):**
- t260719a (CSDMS constant-params restoration) — **included** as the final wave
  (settle the baseline before R6 freezes structure).
- Upstream weathergenr fixes (spatial_ref, wavelet message) — **out**, tracked as
  separate `tanerumit/weathergenr` issues; in-repo workarounds stay.
- wf1 `| tee` exit-masking — resolve with a **portable Python tee wrapper**
  (keeps live console output + fixes exit-safety).

Cut by cost: does the item need a workflow run + baseline re-record, or just
`pytest` + an xfail flip?

## Wave A — no-run batch (DONE 2026-07-21)

All tripwire-wired, latent on the current seed, verified by `pytest` alone.
Suite before: 119 passed / 3 skipped / 7 xfailed. After: **123 / 3 / 1**
(the lone remaining xfail is the upstream hydromt `to_yml` workaround, M2b).

| Task | Fix | Commit |
|---|---|---|
| t260720a | `prepare_cst_parameters` read `variance["min"]` into the max endpoint → `["max"]` | `d2de843` |
| t260720c | D-CAL: cftime-safe slicing in `get_change_annual_clim_proj` (convert object-dtype index up front) | `c57eda0` |
| t260720d | D-VAR/D-MEM: fail loud (ValueError) on asymmetric hist/clim vars & members; corrected stale split-IDs in tests | `735cc20` |
| t260716a (part) | `extract_climate_grid` truncation warning in `prep_historical_climate` | `ce56bc3` |

## Wave B — run + re-record (DONE 2026-07-21)

- **t260720e (D-ATTRS) — RESOLVED 2026-07-21: does-not-reproduce, no fix, no
  re-record.** The current-pins summary `.nc` (product of the real CMIP6 reads in
  the R01 rebuild + R4 re-run) carries the full 7-attr CF set on `precip`/`temp`,
  and the recorded manifest fingerprint carries the same 7 — `check --workflow
  climate_projections` passes on the `.nc`. With R4's probe having shown no wf2 code
  drops attrs and the values being CMIP6-native, hydromt 1.3.1's read preserves them;
  the M2b `{}` diagnosis no longer reproduces (root cause moot). No baseline move.
  Disposition in `dev/milestones/r04/chain-audit.md` § D-ATTRS + `dev/tasks/` § M2b.
  (Aside: the same check flags the 2 wf2 summary **CSVs** as byte-diffs — the
  **pre-existing R4 serialization non-determinism**, `dev/milestones/r04/baseline_diffs.md`,
  not D-ATTRS and not a regression.)

## Wave C — decision-gated code / sweeps

- **wf1 `| tee` exit-masking (t260721a) — DONE 2026-07-21, commit `d13ba37`.**
  `Snakefile_model_creation`'s 3 shell rules route through `src/run_logged.py`
  (a CLI over `snake_utils.run_and_tee`): live console output + log + child exit
  code. Verified end-to-end (failing child propagates its code; old `| tee`
  masked to 0). Tests in `tests/test_run_logged.py`; DAG dry-runs green.
- **M1 warnings triage sweep (t260716a′) — DONE 2026-07-21.** Swept 82 captured
  per-rule logs across all three workflows (no fresh runs needed — logs already on
  disk from the t260719a wf1 rebuild + the R5 seal). **Bucket 3 (our-code): empty;
  Bucket 2:** one intended hydromt resolution-snap (won't-fix); **Bucket 1:** hydromt
  warnings + a captured 62× `sys.excepthook` shutdown cascade from `hydromt build
  -vv` (upstream, post-success, not our tee wrapper). No code changes. Recorded in
  `dev/milestones/phase-1/m01/warnings.md` § "Exhaustive re-triage".
- **`extract_climate_grid` config-staleness (t260716a′) — RESOLVED 2026-07-21, no
  new code.** The 2026-05-07 repro predated R5 (dates hardcoded, `historical:`
  unread). R5 wired `historical_window` as `params`; Snakemake 9.6.2's default
  `params` rerun-trigger now re-runs the rule on a window edit — **verified** by a
  dry-run (endtime 2020→2019 → "Params have changed" schedules the rule). Config-
  as-input is unnecessary/coarser. Detail in `dev/tasks/` § R3.

## Wave D — big / scientific (design ACCEPTED, implementation PENDING)

- **t260719a — design ACCEPTED 2026-07-21 (G2) as
  [ADR 0001](../decisions/0001-restore-wflow-constant-parameters.md)** via a full
  `design-review-loop` (internal risk/architecture/repo-fit panel + 2 external
  GPT rounds via codex + user arbitration at the cap; 26 findings all resolved;
  consolidated record `dev/decisions/0001-restore-wflow-constant-parameters/review-record.md`). The
  review caught two class-of-error bugs before any build compute: the protocol
  targeted `staticmaps.nc` but constants land as TOML scalars, and a
  build-sequence identity-comparison trap. **Implementation is the remaining,
  build-heavy work** (route to `model-builder`). All 15
  CSDMS mappings resolved from `hydromt_wflow/naming.py` (the M2b "5 unresolved"
  all map to real names; none is `wflow_v1: None`): 1 retained (`KsatHorFrac`),
  1 forced-drop (`InfiltCapSoil`, deprecated), 13 mappable → RESTORE. Decision:
  evidence-gated restoration (adopt-default is the null; restore posture
  justified by cross-basin correctness; the discharge diff blesses the move,
  it does not decide per-param drop). Implementation protocol (restored-vs-clean
  builds, staticmaps/TOML landing asserts, discharge materiality, manifest
  extension + re-record) is §"Validation protocol" of the ADR.
  - **Implementation — build-independent slice DONE 2026-07-21.** Artifacts under
    `dev/decisions/0001-restore-wflow-constant-parameters/` + `dev/scripts/`:
    - **Equivalence gate (step 3c) — all 13 PASS, restored count = 13** (gate log
      `dev/decisions/0001-restore-wflow-constant-parameters/equivalence-gate-log.md`). Two-sided authoritative
      evidence (Wflow.jl v0.8 params + glacier melt eqn / stable 1.x param ref /
      `naming.py`). `g_ttm`/`g_tt` collapse sound (`g_tt` is the melt threshold);
      `MaxLeakage` 1.x default = 0 → 9th default-equal pin. Nothing fails closed.
    - `config_baseline.yml` (pinned M2b snapshot) + `config_restored.yml` (13 CSDMS
      entries + KsatHorFrac; drop-comment rewritten to point at ADR 0001).
    - `check_baseline.py`: step-6 time-index-aligned discharge comparator (`compare`
      subcommand + `discharge` target, v2 manifest, sidecar ref series) and step-7
      `record --workflow` merge semantics. Tests: `tests/test_check_baseline_scope.py`
      (rewritten, 15 targets) + new `tests/test_check_baseline_discharge.py`; 19 pass.
    - `dev/scripts/verify_constant_pars.py` (step 3a/3b landing/precedence);
      smoke-tested against the baseline model (KsatHorFrac OK, other 13 ABSENT).
  - **Build-heavy execution — DONE 2026-07-21** (ran locally; Julia 1.11.7 + the
    Deltares catalog were reachable, so no model-builder handoff needed). Three clean
    wf1 builds (restored + baseline ×2). **Landing (3a/3b):** 14/14 scalars land, no
    shadowing. **Reproducibility (4b):** baseline builds bit-identical on discharge
    (`max|ΔQ|/mean = 0`). **Materiality (5): IMMATERIAL** — 0/7670 timesteps exceed
    tolerance (`max|ΔQ|/mean = 1.66e-4`, peak local 0.71% < 1% RTOL); nonzero but
    sub-tolerance. **Re-record (7, immaterial branch):** promoted `config_restored`
    → `config/wflow_build_model.yml`, `--forceall` rebuild of `examples/test_local`,
    `record --workflow model_creation` (wf1 slice incl. new discharge target +
    sidecar; wf2/wf3 preserved byte-identical). Evidence:
    `dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md`. **t260719a COMPLETE.**
- **Baseline manifest rebuild** — entangled with t260719a (both touch the
  manifest). The M2b manifest is still the contract of record (invariance-by-
  construction). ADR 0001 step 7 folds the clean re-record + fingerprint
  extension into t260719a.

## Board reconciliation (verified 2026-07-21)

- t260716a `test_cli` xfails — DONE (MissingInput in R3, CyclicGraph in R5).
- t260716b historical wiring — DONE in R5 (`shared.historical_window`).
- t260716c outlet naming — DONE in R3 (`outlet_index.csv`).
- t260716c "CMIP6 attr loss on merge" — duplicate of **t260720e** (D-ATTRS); merge.


## Closure record, carried from `dev/TODO.md` (2026-08-07)

`dev/TODO.md` became **generated output** when the board was adopted, and
`todoboard render` rewrites it whole. These paragraphs were hand-written
there and are the only record of eleven closed task IDs with their commits,
so they are moved here rather than lost to the first render.

Left deliberately without a row: R7-8 (wflow `log.txt` placement â€” gate-invisible
cosmetic whose verification needs a full wf3 run), R7-7 (working as intended),
R7-21 candidates (b)/(c) (conditional on misattribution recurring), and the
R7-15/17/18 parking rulings.

**Done this campaign (2026-07-21, `fix/pre-r6-followups`):** t260720a
(`variance.max` endpoint, `d2de843`), t260720c (D-CAL cftime, `c57eda0`),
t260720d (D-VAR/D-MEM fail-loud, `735cc20`), t260716a truncation warning
(`ce56bc3`), t260721a (wf1 tee wrapper, `d13ba37`), **t260719a** (CSDMS
constant-params restoration via [ADR 0001](decisions/0001-restore-wflow-constant-parameters.md);
gate all-13-PASS, discharge IMMATERIAL, wf1 baseline re-recorded; evidence
`dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md`), **t260716aâ€²** (M1 warnings
re-triage â€” bucket 2/3 empty of defects; `extract_climate_grid` config-staleness
resolved by R5 params-wiring + verified; docs-only), **t260720e** (D-ATTRS â€”
confirmed does-not-reproduce under current pins: summary `.nc` + recorded manifest
both carry full CF attrs; no fix, no re-record; docs-only).

**Closed as already-done (verified 2026-07-21):** t260716a `test_cli` xfails
(R3+R5), t260716b `historical:` wiring (R5), t260716c outlet naming (R3).
Upstream weathergenr (t260716b tail) is a separate `tanerumit/weathergenr`
concern, out of this repo's board.

> **Detail lives in `followups.md`**, kept as the milestone-scoped backlog store
> (it carries reproducible context and is referenced by live tests). Promote an
> individual item to an `active` row here when its milestone starts.
