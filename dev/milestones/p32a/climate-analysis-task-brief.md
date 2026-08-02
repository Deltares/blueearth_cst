# Task Brief — P3-2a Model-independent climate analysis (implementation)

> **Handoff from the ACCEPTED design.** The authoritative, load-bearing spec is
> `dev/milestones/p32a/climate-analysis-design.md` (ACCEPTED 2026-07-24, v4 post-arbitration).
> Read it in full before touching code — this brief bounds and sequences the
> work; the design owns every path, callable, mechanism, tolerance, and gate
> definition. Where the two differ, the design wins. Audit trail:
> `dev/milestones/p32a/climate-analysis-design-review-record.md`; scoping anchors:
> `dev/milestones/p32a/climate-analysis-intake.md` (its four confirmed decisions are fixed).

### Context

- **Canonical ruleset:** `AGENTS.md` (repo root). Honor its Hard Constraints —
  CST automation scope: hydromt/hydromt_wflow/wflow consumed verbatim, no
  vendored-package edits; no new dependencies.
- **One sanctioned value change:** the wf1 subcatchment climate plots
  (`clim_*` PNGs) re-sourced from raw gridded climate under model parity
  (design §5.2). Everything else is value-identical: wf2/wf3 semantic-diff
  clean, manifested wf1 slice (`check_baseline` TARGETS) unchanged, P3-1
  keyed-store + `.guard_ok` contract preserved byte-exactly (C2/C3).
- **Key mechanisms are review-pinned — implement them exactly:** wf1 extraction
  input is `ancient(staticmaps.nc)` with a `bbox=` kwarg added to
  `prep_historical_climate` (arch-1; wf3 call unchanged); a dedicated wrapper
  `model/extract_climate_wf1.py` is the rule's `script:` — the moved module's
  wf3 `__main__` stays verbatim (ext1-4); aggregation is Option B
  reproject-first through `model/climate_parity.py` using the named
  `hydromt.model.processes.meteo.{precip,temp,pet}` calls (ext1-1/ext1-2);
  chirps orography sidecar is a declared `oro_nc` rule output + declared
  rule-1.10 input, never dirname discovery (ext2-1); eobs is excluded by a
  parse-time config error + wrapper assert (ext2-3); chirps plot acceptance is
  blocked pending the defer-and-pin tolerance procedure (ext2-2 — the fixture
  is era5, so this milestone records the procedure, not chirps numbers).
- **Milestone mechanics (per `branch-model`):** task branch `task/p32a-climate-analysis`
  off `main` tip; commit prefix `p32a:` (already registered); merge to `main`;
  milestone branch/tag at close per the standing model. Capture the pre-P3-2a
  `main` SHA before starting.

### Goal

Land the accepted P3-2a design: the `blueearth_cst/climate_analysis/`
subpackage (three lifted modules, shim-window migration), the wf1-owned
raw-climate extraction rule + model-parity module, the re-sourced wf1
subcatchment climate plots, the mechanical wf3 rewire, and the
ladder-characterized diff + baseline record — with shims deleted before close.

### Non-goals

- No 4th Snakefile / entry point; no `workflows:`-section or wrapper changes.
- No P3-2b interchange contracts; no new plot types or analysis products.
- No change to `climate_forcing_by_subcatchment`'s signature/algorithm, to
  `plot_clim`, or to the shared extraction's wf3 behavior (beyond the optional
  `bbox=` kwarg).
- No manifest extension (`clim_*` plots stay unmanifested — knowing divergence
  from intake decision 4, recorded per repo-2); no eobs wf1 raw-path support.
- No area weighting, no polygon-zonal aggregation (deferred OQ-5/OQ-8).

### Allowed scope

**Permitted:** `blueearth_cst/climate_analysis/**` (new);
`blueearth_cst/model/{extract_climate_wf1,climate_parity}.py` (new);
`blueearth_cst/model/plot_results.py` (§4 + imports only);
`blueearth_cst/experiment/extract_historical_climate.py`,
`blueearth_cst/model/climate_forcing.py`,
`blueearth_cst/projections/prepare_climate_data_catalog.py` (shim → delete);
`Snakefile_model_creation` (new rule, rule-1.10 rewiring, eobs guard, header
renumber); `Snakefile_climate_experiment` (rule 3.02 + 3.08 `script:` strings
only); `tests/**`; `dev/milestones/p32a/**`; `dev/roadmap.md` (status only).

**Approval-gated:** any edit to `Snakefile_climate_projections`; any rule-3.02
change beyond the `script:` string; any change to P3-1 store/guard paths —
the design says zero; if unavoidable, PAUSE and raise it.

**Forbidden:** vendored upstream packages; `pixi.lock` / `Manifest.toml`;
`blueearth_cst/weathergen/*.R`; `dev/baseline/manifest.json` (no re-record —
the change is unmanifested); `examples/test_local` by hand.

### Required changes (checklist)

The design §8 commit plan, verbatim — one `p32a:` commit each; every commit
`--dry-run`-clean on all three Snakefiles + `test_cli.py`-green:

1. Create `climate_analysis/` subpackage (real code) + re-export shims at old
   paths; grep-derived importer inventory; repoint the three moved-module
   tests to the new import paths (pure repoint — assertions verbatim, repo-1/
   repo-4). Gate: full `pytest tests/`, three `--dry-run`s.
2. Rewire wf3: rule 3.02 + rule 3.08 `script:` paths + live Python imports.
   Rule-3.02 diff shows ONLY the `script:` string changed (C3 pin). Gate:
   `test_cli.py`, wf3 `--dry-run`, semantic-tree-diff clean on a wf3 slice.
3. New `extract_climate_grid_wf1` rule (`input: basin_nc = ancient(staticmaps.nc)`,
   branch-resolved `oro_nc` output on chirps, wf1-owned
   `climate_historical/wf1_raw/` outputs) + wrapper script + `bbox=` kwarg +
   parse-time eobs guard + `climate_parity.py` + unit tests (bbox tolerance
   ext1-5; exactly-once era5 analytic + chirps sidecar-chaining ext1-1).
   Gate: wf1 `--dry-run` on the default tracked config, `test_cli.py`, new
   unit tests green, rule-3.02 output untouched.
4. Re-source `plot_results.py` §4: parity grid → unchanged groupby; drop
   `mod.forcing.data` + the `forcing_path` input; add raw-climate input,
   branch-resolved `oro_nc` input, `data_sources`/`clim_source` params; keep
   the `script = ...` input label (repo-5). Gate: wf1 `--dry-run`,
   `test_cli.py`, full `pytest tests/`, wf1 e2e producing the new `clim_*`
   plots.
5. `dev/milestones/p32a/compare_climate_ladder.py` + run it on the fixture: side-by-side
   old/new plots, persisted ladder states, A0/A1/A2 tables, component deltas
   (`A2−A1`, exact precip null-check, `A2−A0`, grid-level G vs
   `inmaps_historical.nc`), `wf1_raw` vs keyed-store `allclose`; record all in
   the `dev/milestones/p32a/` baseline note; `check_baseline check` no-op confirm.
   Gate: every residual assigned to a named edge; G within §5.2 tolerances
   (temp ≤ 0.05 °C, precip/pet ≤ 0.05 mm d⁻¹); manifested slice unchanged.
6. Delete shims + migration note (old→new module map). Gate: full
   `pytest tests/`, three `--dry-run`s, execution-level check of both wf3
   `script:` sites + the wf1 rule via the `run_workflows.py` e2e (risk-6).

### Validation

Design §9 verbatim; a commit is not done until its named gates pass. Ladder:
(1) narrow — `pytest tests/test_cli.py` + the commit's unit tests every
commit; (2) new behavioral tests — bbox tolerance, exactly-once (era5 +
chirps synthetic), eobs parse-time rejection; (3) execution smokes —
commit-4 wf1 e2e, commit-5 ladder run, commit-6
`pixi run python scripts/run_workflows.py --config config/workflows/snake_config_model_test.yml`;
(4) full gate — `pytest tests/` green + clean `--dry-run` × 3; (5) baseline —
`dev/scripts/semantic_tree_diff.py` pre/post wf2+wf3 clean modulo nothing,
`check_baseline check` manifested-slice unchanged, C1 grep
(`WflowSbmModel|mod\.forcing|mod\.staticmaps` absent from `climate_analysis/`).

### Acceptance criteria

- All 6 commits landed; C1 grep clean; rule-3.02 diff = `script:` string only;
  `.guard_ok`/keyed-store paths byte-identical; wf2/wf3 semantic-diff clean;
  manifested wf1 slice unchanged; exactly-once + bbox unit tests green;
  fixture `wf1_raw` ≈ keyed-store (`allclose`); on era5 fixture `A2−A0` and
  G ≈ 0 within tolerances with precip null-check exactly zero; eobs config
  fails wf1 dry-run at parse time with the named error; shims gone; ladder +
  baseline note + migration map landed.
- **User sign-off at the milestone gate** on the side-by-side plots + ladder
  record before merge/tag.
- **Rollback trigger:** any wf2/wf3 value diff, any manifested-slice drift,
  any G/ladder residual not assignable to a named edge, or any rule-3.02
  change beyond the script string → stop, do not merge, surface the evidence.

### Output requirements

- The commits on `task/p32a-climate-analysis`, merged to `main` after the
  milestone gate; milestone branch/tag per `branch-model`.
- A **Results delta**: the ladder-decomposed characterization of the one
  sanctioned change (per variable × edge, max-abs/RMSE), confirmation that
  nothing else moved, and the no-op manifest confirmation.

### Task constraints

- Design §8 sequencing is binding (shims must exist before rewires; execution
  check before shim deletion).
- Preserve `workflow.configfiles[0]` forwarding, `get_config`, `tee_to_log`,
  and per-rule `log:`/`benchmark:` house patterns; naming per
  `dev/conventions/naming.md` (OQ-1: rename file to `subcatchment_climate.py`,
  keep the function name).
- `--dry-run` never validates a `script:` body — the commit-6 execution check
  is mandatory, not optional.

**Human gates** (otherwise drive commit-to-commit autonomously per the
standing preference):

- **Gate 1 — milestone gate, before merge/tag, PAUSE:** present the
  side-by-side old/new `clim_*` plots, the ladder record (every residual on a
  named edge), the semantic-diff + `check_baseline` results, and the knowing
  manifest divergence (repo-2) for user sign-off.
- Raise immediately if `Snakefile_climate_projections` needs edits, rule 3.02
  needs more than the script string, or any P3-1 store/guard path moves.
